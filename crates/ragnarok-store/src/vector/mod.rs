//! In-process vector engine backed by `qdrant-edge`'s embedded [`EdgeShard`]
//! with the specified collection layout:
//!
//! * `dense`   — 1024-dim cosine, HNSW index, int8 scalar quantization pinned
//!   in RAM, f32 originals on memmap
//! * `colbert` — 128-dim multi-vector, MaxSim comparator, no HNSW (m=0),
//!   cold/on-disk (only touched for rescoring top candidates)
//! * `lexical` — sparse index, on-disk placement
//!
//! The shard keeps its own WAL, so upserts are durable incrementally and a
//! daemon restart reopens the segment set in place instead of rebuilding an
//! index from a monolithic snapshot — the property that matters at
//! million-point scale.

use qdrant_edge::{
    Condition, Distance, EdgeConfig, EdgeShard, EdgeSparseVectorParams, EdgeVectorParams,
    FieldCondition, Filter, HnswIndexConfig, Match, Memory, MultiVectorComparator,
    MultiVectorConfig, NamedQuery, NamedVectors, OperationError, PointId, PointInsertOperations,
    PointOperations, PointStruct, QuantizationConfig, QueryEnum, QueryRequest, RetrieveRequest,
    ScalarQuantizationConfig, ScalarType, ScoringQuery, UpdateOperation, Vector, VectorInternal,
    VectorStructInternal, WithPayloadInterface, WithVector,
};
use ragnarok_core::scope::Permitted;
use ragnarok_core::{ChunkId, EmbeddingTriple, Error, Result, SparseVector};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Named-vector identifiers inside the shard (mirroring the spec's
/// collection names).
const VEC_DENSE: &str = "dense";
const VEC_COLBERT: &str = "colbert";
const VEC_LEXICAL: &str = "lexical";

/// One retrieval result: point id + similarity score (higher = better).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredPoint {
    pub id: ChunkId,
    pub score: f32,
}

/// ColBERT late-interaction score: mean over query rows of the best matching
/// document token (`Σ_i max_j q_i·d_j / |Q|`). All rows must be
/// L2-normalized for the result to stay in `[0, 1]`.
pub fn maxsim(query: &[Vec<f32>], doc: &[Vec<f32>]) -> f32 {
    if query.is_empty() || doc.is_empty() {
        return 0.0;
    }
    let mut total = 0.0f32;
    for q in query {
        let mut best = 0.0f32;
        for d in doc {
            let dot: f32 = q.iter().zip(d.iter()).map(|(a, b)| a * b).sum();
            if dot > best {
                best = dot;
            }
        }
        total += best;
    }
    total / query.len() as f32
}

/// Reciprocal Rank Fusion over multiple ranked candidate lists:
/// `score(id) = Σ_lists 1 / (k + rank)` with ranks starting at 1.
/// Output is sorted by fused score, descending.
pub fn reciprocal_rank_fusion(lists: &[&[ScoredPoint]], k: f32) -> Vec<ScoredPoint> {
    let mut fused: std::collections::HashMap<ChunkId, f32> = std::collections::HashMap::new();
    for list in lists {
        for (rank, p) in list.iter().enumerate() {
            *fused.entry(p.id).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
        }
    }
    let mut out: Vec<ScoredPoint> = fused
        .into_iter()
        .map(|(id, score)| ScoredPoint { id, score })
        .collect();
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out
}

fn op_err(e: OperationError) -> Error {
    Error::VectorStore(format!("qdrant-edge: {e}"))
}

/// Shard layout: named vectors per spec.
#[allow(deprecated)] // `always_ram` field is required by the struct literal
fn edge_config(dense_dim: usize) -> EdgeConfig {
    let mut vectors = HashMap::new();
    vectors.insert(
        VEC_DENSE.to_string(),
        EdgeVectorParams {
            size: dense_dim,
            distance: Distance::Cosine,
            on_disk: None,
            // f32 originals evictable via page cache; the int8 copy below is
            // what the HNSW traversal actually scores against.
            memory: Some(Memory::Cached),
            multivector_config: None,
            datatype: None,
            quantization_config: Some(QuantizationConfig::from(ScalarQuantizationConfig {
                r#type: ScalarType::Int8,
                quantile: Some(0.99),
                always_ram: None,
                memory: Some(Memory::Pinned),
            })),
            hnsw_config: Some(HnswIndexConfig {
                m: 16,
                ef_construct: 128,
                full_scan_threshold: 10_000,
                max_indexing_threads: 0,
                on_disk: None,
                memory: Some(Memory::Pinned),
                payload_m: None,
                inline_storage: None,
            }),
        },
    );
    vectors.insert(
        VEC_COLBERT.to_string(),
        EdgeVectorParams {
            size: 128,
            distance: Distance::Cosine,
            on_disk: None,
            // MaxSim scoring needs the original rows; they are only touched
            // during rescoring, so keep them purely memmap-backed.
            memory: Some(Memory::Cold),
            multivector_config: Some(MultiVectorConfig {
                comparator: MultiVectorComparator::MaxSim,
            }),
            datatype: None,
            quantization_config: None,
            // m=0: no HNSW graph for the rescore layer (per spec).
            hnsw_config: Some(HnswIndexConfig {
                m: 0,
                ef_construct: 128,
                full_scan_threshold: 10_000,
                max_indexing_threads: 0,
                on_disk: None,
                memory: Some(Memory::Cold),
                payload_m: None,
                inline_storage: None,
            }),
        },
    );
    let mut sparse_vectors = HashMap::new();
    sparse_vectors.insert(
        VEC_LEXICAL.to_string(),
        EdgeSparseVectorParams {
            full_scan_threshold: None,
            on_disk: None,
            memory: Some(Memory::Cold),
            modifier: None,
            datatype: None,
        },
    );
    EdgeConfig {
        vectors,
        sparse_vectors,
        ..Default::default()
    }
}

/// Combined engine holding the three named representations, memmap-backed
/// by a qdrant edge shard on disk.
pub struct VectorEngine {
    shard: EdgeShard,
    dir: PathBuf,
}

impl VectorEngine {
    /// Open (or create) the edge shard at `dir`.
    pub fn open(dir: &Path, dense_dim: usize) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let config = edge_config(dense_dim);
        let persisted = dir.join("edge_config.json").exists();
        let shard = if persisted {
            EdgeShard::load(dir, Some(config)).map_err(op_err)?
        } else {
            EdgeShard::new(dir, config).map_err(op_err)?
        };
        Ok(Self {
            shard,
            dir: dir.to_path_buf(),
        })
    }

    /// Same as [`open`]; kept for call-site clarity.
    pub fn create(dir: &Path, dense_dim: usize) -> Result<Self> {
        Self::open(dir, dense_dim)
    }

    /// Insert or replace all three representations of one chunk. `doc_id`
    /// goes into the point payload so scoped searches can push a
    /// `match_any` filter into the traversal.
    pub fn upsert(&self, id: ChunkId, doc_id: &str, triple: &EmbeddingTriple) -> Result<()> {
        let (idx, val): (Vec<u32>, Vec<f32>) = triple.sparse.iter().map(|(&k, &v)| (k, v)).unzip();
        let vectors = NamedVectors(
            [
                (
                    VEC_DENSE.to_string(),
                    Vector::new_dense(triple.dense.clone()),
                ),
                (
                    VEC_COLBERT.to_string(),
                    Vector::new_multi(triple.colbert.clone()).map_err(op_err)?,
                ),
                (
                    VEC_LEXICAL.to_string(),
                    Vector::new_sparse(idx, val).map_err(op_err)?,
                ),
            ]
            .into_iter()
            .collect(),
        );
        let point = PointStruct::new(
            PointId::Uuid(id.into_uuid()),
            vectors,
            serde_json::json!({ "doc_id": doc_id }),
        );
        self.shard
            .update(UpdateOperation::PointOperation(
                PointOperations::UpsertPoints(PointInsertOperations::PointsList(vec![point.0])),
            ))
            .map_err(op_err)
    }

    /// Delete all representations of one chunk.
    pub fn remove(&self, id: ChunkId) -> Result<()> {
        self.shard
            .update(UpdateOperation::PointOperation(
                PointOperations::DeletePoints {
                    ids: vec![PointId::Uuid(id.into_uuid())],
                },
            ))
            .map_err(op_err)
    }

    /// Top-k nearest neighbours on the dense (quantized HNSW) vector,
    /// optionally restricted by a payload filter (scope enforcement).
    pub fn dense_search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<Filter>,
    ) -> Result<Vec<ScoredPoint>> {
        self.nearest(VEC_DENSE, VectorInternal::Dense(query.to_vec()), k, filter)
    }

    /// Top-k matches on the sparse lexical index.
    pub fn sparse_search(
        &self,
        query: &SparseVector,
        k: usize,
        filter: Option<Filter>,
    ) -> Result<Vec<ScoredPoint>> {
        let (indices, values): (Vec<u32>, Vec<f32>) = query.iter().map(|(&k, &v)| (k, v)).unzip();
        let sparse = qdrant_edge::SparseVector::new(indices, values)
            .map_err(|e| Error::VectorStore(format!("bad sparse query: {e}")))?;
        self.nearest(VEC_LEXICAL, VectorInternal::Sparse(sparse), k, filter)
    }

    /// Translate a resolved [`Permitted`] into a shard payload filter:
    /// whitelist → `doc_id ∈ allow` must-match, blacklist →
    /// `doc_id ∉ deny` must-not. Both clauses live in one `Filter`, so the
    /// traversal never visits out-of-scope candidates.
    pub fn scope_filter(perm: &Permitted) -> Option<Filter> {
        let mut filter = Filter {
            must: None,
            must_not: None,
            should: None,
            min_should: None,
        };
        if let Some(allow) = &perm.allow {
            filter.must = Some(vec![Condition::Field(FieldCondition::new_match(
                "doc_id"
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("static key is valid")),
                Match::from(allow.iter().cloned().collect::<Vec<String>>()),
            ))]);
        }
        if !perm.deny.is_empty() {
            filter.must_not = Some(vec![Condition::Field(FieldCondition::new_match(
                "doc_id"
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("static key is valid")),
                Match::from(perm.deny.iter().cloned().collect::<Vec<String>>()),
            ))]);
        }
        (filter.must.is_some() || filter.must_not.is_some()).then_some(filter)
    }

    fn nearest(
        &self,
        name: &str,
        vector: VectorInternal,
        k: usize,
        filter: Option<Filter>,
    ) -> Result<Vec<ScoredPoint>> {
        let mut req = QueryRequest::new(k);
        req.query = Some(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery::new(
            vector, name,
        ))));
        req.filter = filter;
        let scored = self.shard.query(req).map_err(op_err)?;
        Ok(scored
            .into_iter()
            .filter_map(|p| match p.id {
                PointId::Uuid(u) => Some(ScoredPoint {
                    id: ChunkId::from_uuid(u),
                    score: p.score,
                }),
                PointId::NumId(_) => None,
            })
            .collect())
    }

    /// MaxSim score of one candidate against the query matrix.
    pub fn colbert_score(&self, id: ChunkId, query_rows: &[Vec<f32>]) -> Result<f32> {
        Ok(self
            .colbert_scores(&[id], query_rows)?
            .into_iter()
            .next()
            .map(|(_, s)| s)
            .unwrap_or(0.0))
    }

    /// Batch MaxSim scoring: fetches each candidate's stored token matrix
    /// from the (memmap) colbert vector and applies [`maxsim`] locally, in a
    /// single `retrieve` round-trip.
    pub fn colbert_scores(
        &self,
        ids: &[ChunkId],
        query_rows: &[Vec<f32>],
    ) -> Result<Vec<(ChunkId, f32)>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut req =
            RetrieveRequest::new(ids.iter().map(|c| PointId::Uuid(c.into_uuid())).collect());
        req.with_vector = Some(WithVector::Selector(vec![VEC_COLBERT.to_string()]));
        req.with_payload = Some(WithPayloadInterface::Bool(false));
        let records = self.shard.retrieve(req).map_err(op_err)?;
        let mut out = Vec::with_capacity(records.len());
        for rec in records {
            let Some(VectorStructInternal::Named(map)) = rec.vector else {
                continue;
            };
            let Some(VectorInternal::MultiDense(md)) = map.get(VEC_COLBERT) else {
                continue;
            };
            let PointId::Uuid(u) = rec.id else { continue };
            let doc: Vec<Vec<f32>> = md.multi_vectors().map(|row| row.to_vec()).collect();
            out.push((ChunkId::from_uuid(u), maxsim(query_rows, &doc)));
        }
        Ok(out)
    }

    pub fn len(&self) -> usize {
        self.shard.info().map(|i| i.points_count).unwrap_or(0)
    }

    /// Number of indexed (HNSW) vectors — diagnostics only.
    pub fn indexed_len(&self) -> usize {
        self.shard
            .info()
            .map(|i| i.indexed_vectors_count)
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Flush the WAL and all segments — durable immediately, no separate
    /// snapshot file required.
    pub fn snapshot(&self) -> Result<()> {
        self.shard.flush().map_err(op_err)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triple(seed: f32, tokens: &[&str]) -> EmbeddingTriple {
        let mut dense = vec![0.0f32; 1024];
        for (i, d) in dense.iter_mut().enumerate() {
            *d = ((i as f32 * 0.37) + seed).sin();
        }
        let n: f32 = dense.iter().map(|x| x * x).sum::<f32>().sqrt();
        dense.iter_mut().for_each(|x| *x /= n);
        let sparse: SparseVector = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.len() as u32 * 97 + i as u32, 1.0))
            .collect();
        let mut row = vec![0.0f32; 128];
        for (j, v) in row.iter_mut().enumerate() {
            *v = ((j as f32 * 0.13) + seed).sin();
        }
        let n: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        row.iter_mut().for_each(|x| *x /= n);
        EmbeddingTriple {
            dense,
            sparse,
            colbert: vec![row],
        }
    }

    #[test]
    fn rrf_fuses_lists() {
        let a = ScoredPoint {
            id: ChunkId::new(),
            score: 0.9,
        };
        let b = ScoredPoint {
            id: ChunkId::new(),
            score: 0.5,
        };
        let fused = reciprocal_rank_fusion(&[&[a, b], &[b, a]], 60.0);
        assert!(fused.iter().all(|p| p.score > 0.0));
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn maxsim_scores_best_token_matches() {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let doc = vec![vec![0.9, 0.1], vec![0.0, 0.99]];
        let s = maxsim(&q, &doc);
        assert!(s > 0.9 && s <= 1.0);
        assert_eq!(maxsim(&[], &doc), 0.0);
        assert_eq!(maxsim(&q, &[]), 0.0);
    }

    #[test]
    fn engine_roundtrip_dense_sparse_colbert() {
        let dir = tempfile::tempdir().unwrap();
        let e = VectorEngine::open(dir.path(), 1024).unwrap();
        let a = ChunkId::new();
        let b = ChunkId::new();
        e.upsert(a, "doc-a", &triple(0.0, &["alpha", "beta"]))
            .unwrap();
        e.upsert(b, "doc-b", &triple(9.0, &["gamma"])).unwrap();
        assert_eq!(e.len(), 2);

        let q = triple(0.0, &["alpha"]);
        let dense_hits = e.dense_search(&q.dense, 5, None).unwrap();
        assert_eq!(dense_hits.first().unwrap().id, a);

        let mut sq = SparseVector::new();
        sq.insert("alpha".len() as u32 * 97, 1.0);
        let sparse_hits = e.sparse_search(&sq, 5, None).unwrap();
        assert!(sparse_hits.iter().any(|p| p.id == a));

        let s = e.colbert_score(a, &q.colbert).unwrap();
        assert!(s > 0.8);
        let batch = e.colbert_scores(&[a, b], &q.colbert).unwrap();
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn remove_and_persist() {
        let dir = tempfile::tempdir().unwrap();
        let a = ChunkId::new();
        {
            let e = VectorEngine::open(dir.path(), 1024).unwrap();
            e.upsert(a, "doc-a", &triple(1.0, &["x"])).unwrap();
            e.upsert(ChunkId::new(), "doc-b", &triple(2.0, &["y"]))
                .unwrap();
            e.snapshot().unwrap();
        }
        // reopen: memmap state survives without any snapshot rebuild
        let e = VectorEngine::open(dir.path(), 1024).unwrap();
        assert_eq!(e.len(), 2);
        e.remove(a).unwrap();
        assert_eq!(e.len(), 1);
        let q = triple(1.0, &["x"]);
        assert!(e
            .dense_search(&q.dense, 5, None)
            .unwrap()
            .iter()
            .all(|p| p.id != a));
    }

    #[test]
    fn scope_filter_restricts_candidates() {
        use ragnarok_core::scope::Permitted;
        use std::collections::HashSet;

        let dir = tempfile::tempdir().unwrap();
        let e = VectorEngine::open(dir.path(), 1024).unwrap();
        let a = ChunkId::new();
        let b = ChunkId::new();
        e.upsert(a, "doc-a", &triple(0.0, &["alpha"])).unwrap();
        e.upsert(b, "doc-b", &triple(0.05, &["alpha"])).unwrap();

        let q = triple(0.0, &["alpha"]);
        // whitelist: only doc-b visible
        let perm = Permitted {
            allow: Some(HashSet::from(["doc-b".to_string()])),
            deny: HashSet::new(),
        };
        let hits = e
            .dense_search(&q.dense, 5, VectorEngine::scope_filter(&perm))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, b);

        // blacklist: doc-a excluded → only doc-b remains
        let perm = Permitted {
            allow: None,
            deny: HashSet::from(["doc-a".to_string()]),
        };
        let hits = e
            .dense_search(&q.dense, 5, VectorEngine::scope_filter(&perm))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, b);
        assert!(VectorEngine::scope_filter(&Permitted::all()).is_none());
    }
}
