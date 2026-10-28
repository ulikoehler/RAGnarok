//! In-process vector engine with the Qdrant-compatible collection layout:
//! dense HNSW + int8 scalar quantization, sparse inverted index, and a
//! multi-vector store scored by MaxSim late interaction.

mod colbert;
mod dense;
mod quant;
mod sparse;

pub use colbert::ColbertStore;
pub use dense::DenseIndex;
pub use quant::{QuantizedVec, ScalarQuantizer};
pub use sparse::SparseIndex;

use dashmap::DashMap;
use ragnarok_core::{ChunkId, EmbeddingTriple, Error, Result, SparseVector};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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

/// Persisted state of the whole engine (quantized dense vectors, sparse
/// postings, colbert matrices). The HNSW graph is rebuilt on load.
#[derive(Serialize, Deserialize)]
struct EngineSnapshot {
    dense: Vec<(ChunkId, QuantizedVec)>,
    sparse_docs: Vec<(ChunkId, SparseVector)>,
    colbert: Vec<(ChunkId, Vec<Vec<f32>>)>,
}

/// Combined engine holding the three named representations.
pub struct VectorEngine {
    pub dense: DenseIndex,
    pub sparse: SparseIndex,
    pub colbert: ColbertStore,
    /// Points pending deletion in dense (filtered at query time).
    deleted: DashMap<ChunkId, ()>,
    dir: PathBuf,
}

impl VectorEngine {
    pub fn create(dir: &Path, dense_dim: usize) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        Ok(VectorEngine {
            dense: DenseIndex::new(dense_dim),
            sparse: SparseIndex::new(),
            colbert: ColbertStore::new(),
            deleted: DashMap::new(),
            dir: dir.to_path_buf(),
        })
    }

    /// Load from `dir` if a snapshot exists, else create empty.
    pub fn open(dir: &Path, dense_dim: usize) -> Result<Self> {
        let engine = Self::create(dir, dense_dim)?;
        let snap = dir.join("snapshot.bin");
        if snap.exists() {
            let bytes = std::fs::read(&snap)?;
            let snap: EngineSnapshot = bincode::deserialize(&bytes)
                .map_err(|e| Error::VectorStore(format!("snapshot decode: {e}")))?;
            for (id, qv) in snap.dense {
                engine.dense.insert_quantized(id, qv)?;
            }
            for (id, sv) in snap.sparse_docs {
                engine.sparse.upsert(id, &sv);
            }
            for (id, rows) in snap.colbert {
                engine.colbert.upsert(id, rows);
            }
            engine.sparse.rebuild_postings();
        }
        Ok(engine)
    }

    pub fn upsert(&self, id: ChunkId, triple: &EmbeddingTriple) -> Result<()> {
        self.deleted.remove(&id);
        self.dense.upsert(id, &triple.dense)?;
        self.sparse.upsert(id, &triple.sparse);
        self.colbert.upsert(id, triple.colbert.clone());
        Ok(())
    }

    pub fn remove(&self, id: ChunkId) {
        self.deleted.insert(id, ());
        self.dense.remove(id);
        self.sparse.remove(id);
        self.colbert.remove(id);
    }

    pub fn dense_search(&self, query: &[f32], k: usize) -> Result<Vec<ScoredPoint>> {
        let hits = self.dense.search(query, k.saturating_mul(2).max(k + 8))?;
        Ok(hits
            .into_iter()
            .filter(|p| !self.deleted.contains_key(&p.id))
            .take(k)
            .collect())
    }

    pub fn sparse_search(&self, query: &SparseVector, k: usize) -> Vec<ScoredPoint> {
        self.sparse
            .search(query, k)
            .into_iter()
            .filter(|p| !self.deleted.contains_key(&p.id))
            .collect()
    }

    /// MaxSim score of one candidate against the query matrix.
    pub fn colbert_score(&self, id: ChunkId, query_rows: &[Vec<f32>]) -> f32 {
        self.colbert
            .get(id)
            .map(|doc| maxsim(query_rows, &doc))
            .unwrap_or(0.0)
    }

    pub fn len(&self) -> usize {
        self.colbert.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Persist quantized vectors + sparse + colbert stores.
    pub fn snapshot(&self) -> Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let snap = EngineSnapshot {
            dense: self.dense.quantized_entries(),
            sparse_docs: self.sparse.doc_entries(),
            colbert: self.colbert.entries(),
        };
        let bytes = bincode::serialize(&snap)
            .map_err(|e| Error::VectorStore(format!("snapshot encode: {e}")))?;
        let tmp = self.dir.join("snapshot.bin.tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(tmp, self.dir.join("snapshot.bin"))?;
        Ok(())
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
        row[0] = 1.0;
        EmbeddingTriple {
            dense,
            sparse,
            colbert: vec![row],
        }
    }

    #[test]
    fn maxsim_semantics() {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let d = vec![vec![1.0, 0.0], vec![0.9, 0.1]];
        // best for q0 = 1.0, best for q1 = 0.1 -> mean 0.55
        assert!((maxsim(&q, &d) - 0.55).abs() < 1e-6);
        assert_eq!(maxsim(&q, &[]), 0.0);
        assert_eq!(maxsim(&[], &d), 0.0);
    }

    #[test]
    fn rrf_orders_by_rank() {
        let a = vec![
            ScoredPoint {
                id: ChunkId::new(),
                score: 0.9,
            },
            ScoredPoint {
                id: ChunkId::new(),
                score: 0.8,
            },
        ];
        let b = vec![
            ScoredPoint {
                id: a[1].id,
                score: 0.99,
            },
            ScoredPoint {
                id: a[0].id,
                score: 0.1,
            },
        ];
        let fused = reciprocal_rank_fusion(&[&a, &b], 60.0);
        // both have rank1+rank2 -> equal fused score
        assert!((fused[0].score - fused[1].score).abs() < 1e-9);
        // rrf value for rank 1 twice: 2/(60+1) etc.
        let expected = 1.0 / 61.0 + 1.0 / 62.0;
        assert!((fused[0].score - expected).abs() < 1e-9);
    }

    #[test]
    fn engine_upsert_search_remove() {
        let dir = tempfile::tempdir().unwrap();
        let eng = VectorEngine::open(dir.path(), 1024).unwrap();
        let id = ChunkId::new();
        let t = triple(0.1, &["hello", "world"]);
        eng.upsert(id, &t).unwrap();
        assert_eq!(eng.len(), 1);

        let hits = eng.dense_search(&t.dense, 5).unwrap();
        assert_eq!(hits[0].id, id);
        let shits = eng.sparse_search(&t.sparse, 5);
        assert_eq!(shits[0].id, id);
        assert!(eng.colbert_score(id, &t.colbert) > 0.0);

        eng.remove(id);
        assert!(eng.dense_search(&t.dense, 5).unwrap().is_empty());
        assert!(eng.sparse_search(&t.sparse, 5).is_empty());
        assert_eq!(eng.len(), 0);
    }

    #[test]
    fn engine_snapshot_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let id = ChunkId::new();
        let t = triple(0.5, &["persist"]);
        {
            let eng = VectorEngine::open(dir.path(), 1024).unwrap();
            eng.upsert(id, &t).unwrap();
            eng.snapshot().unwrap();
        }
        let eng = VectorEngine::open(dir.path(), 1024).unwrap();
        assert_eq!(eng.len(), 1);
        assert_eq!(eng.dense_search(&t.dense, 5).unwrap()[0].id, id);
        assert_eq!(eng.sparse_search(&t.sparse, 5)[0].id, id);
        assert!(eng.colbert_score(id, &t.colbert) > 0.0);
    }
}
