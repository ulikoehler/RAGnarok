use crate::metadata::MetadataDb;
use crate::vector::{reciprocal_rank_fusion, ScoredPoint, VectorEngine};
use ragnarok_core::scope::Permitted;
use ragnarok_core::{
    Chunk, DocumentId, DocumentRecord, EmbeddingTriple, Result, SearchHit, SearchRequest,
};
use ragnarok_inference::Embedder;
use serde::Serialize;
use std::path::Path;
use std::sync::Arc;

/// Aggregate index statistics for telemetry.
#[derive(Debug, Clone, Serialize)]
pub struct EngineStats {
    pub documents: u64,
    pub chunks: u64,
    pub dense_points: usize,
    pub sparse_docs: usize,
    pub colbert_docs: usize,
}

/// The hybrid retrieval engine: embeds the query, retrieves candidates from
/// the dense and lexical collections, fuses via RRF, rescores with ColBERT
/// MaxSim and enriches hits with stored offsets.
///
/// ```text
/// Query ──> dense + sparse + colbert embeddings
///      ──> Qdrant-style hybrid retrieval (top-K per representation)
///      ──> Reciprocal Rank Fusion
///      ──> ColBERT MaxSim rescoring
///      ──> ranked SearchHits
/// ```
pub struct SearchEngine {
    pub vectors: VectorEngine,
    pub meta: Arc<MetadataDb>,
    embedder: Arc<dyn Embedder>,
    /// Cross-encoder rerank stage — absent on hashing builds unless one
    /// is attached explicitly (`with_reranker`).
    reranker: Option<Arc<dyn ragnarok_inference::Reranker>>,
}

/// Fraction of fused candidates forwarded to MaxSim rescoring
/// (>= limit * this factor).
const RESCORE_FACTOR: usize = 4;
const RESCORE_MIN: usize = 32;

impl SearchEngine {
    pub fn new(vectors: VectorEngine, meta: Arc<MetadataDb>, embedder: Arc<dyn Embedder>) -> Self {
        SearchEngine {
            vectors,
            meta,
            embedder,
            reranker: None,
        }
    }

    /// Embedding backend name — stamped on each indexed document as
    /// `metadata["embedder"]`; the global reindexer's `only_stale` mode
    /// compares it against the live backend.
    pub fn embedder_name(&self) -> &'static str {
        self.embedder.name()
    }

    /// Attach a cross-encoder reranker (post-RRF stage). Call at
    /// bootstrap while the engine is uniquely owned (`Arc::get_mut`).
    pub fn set_reranker(&mut self, r: Arc<dyn ragnarok_inference::Reranker>) {
        self.reranker = Some(r);
    }

    /// Whether the rerank stage is active for a request.
    fn rerank_top(&self, req: &SearchRequest) -> usize {
        if self.reranker.is_some() {
            req.rerank_top
        } else {
            0
        }
    }

    /// Open an engine persisted under `data_dir` (`ragnarok.db` + `vectorstore/`).
    pub fn open(
        data_dir: &Path,
        embedder: Arc<dyn Embedder>,
        dense_dim: usize,
    ) -> Result<Arc<Self>> {
        std::fs::create_dir_all(data_dir)?;
        let meta = Arc::new(MetadataDb::open(&data_dir.join("ragnarok.db"))?);
        let vectors = VectorEngine::open(&data_dir.join("vectorstore"), dense_dim)?;
        Ok(Arc::new(Self::new(vectors, meta, embedder)))
    }

    /// Store a fully-embedded document: metadata + chunk offsets + vectors.
    /// Re-indexing replaces both chunks and points (deterministic ids).
    pub async fn index_document(
        &self,
        record: &DocumentRecord,
        chunks: &[Chunk],
        triples: &[EmbeddingTriple],
    ) -> Result<()> {
        if chunks.len() != triples.len() {
            return Err(ragnarok_core::Error::Internal(format!(
                "chunks/triples length mismatch: {} vs {}",
                chunks.len(),
                triples.len()
            )));
        }
        let meta = self.meta.clone();
        let mut record = record.clone();
        record
            .metadata
            .insert("embedder".into(), self.embedder.name().to_owned());
        let chunks = chunks.to_vec();
        let triples = triples.to_vec();
        let vectors = &self.vectors;
        let doc_id = record.id.to_string();
        // chunk ids are `chunk_id(doc, idx)` — re-indexing a document
        // that now yields FEWER chunks would leave stale vector points
        // behind (they'd resurface as unresolvable hits). Remove the
        // points of every previously stored chunk not in the new set.
        let old_ids: std::collections::HashSet<_> = meta
            .chunks_for_document(record.id)?
            .into_iter()
            .map(|c| c.id)
            .collect();
        let new_ids: std::collections::HashSet<_> = chunks.iter().map(|c| c.id).collect();
        for stale in old_ids.difference(&new_ids) {
            vectors.remove(*stale)?;
        }
        // vectors first (cheap), then the sqlite txn
        for (c, t) in chunks.iter().zip(triples.iter()) {
            vectors.upsert(c.id, &doc_id, t)?;
        }
        tokio::task::spawn_blocking(move || -> Result<()> {
            meta.upsert_document(&record)?;
            meta.replace_chunks(record.id, &chunks)
        })
        .await
        .map_err(|e| ragnarok_core::Error::Internal(format!("join: {e}")))??;
        Ok(())
    }

    /// Remove a document and all of its vectors.
    pub fn delete_document(&self, id: DocumentId) -> Result<bool> {
        if self.meta.get_document(id)?.is_none() {
            return Ok(false);
        }
        let chunk_ids = self.meta.delete_document(id)?;
        for cid in chunk_ids {
            self.vectors.remove(cid)?;
        }
        Ok(true)
    }

    /// Run the full hybrid pipeline without scope restrictions.
    pub async fn search(&self, req: &SearchRequest) -> Result<Vec<SearchHit>> {
        self.search_scoped(req, &Permitted::all()).await
    }

    /// Hybrid pipeline under a resolved document scope. The whitelist/
    /// blacklist compiles to a shard payload filter pushed into the dense
    /// and sparse traversals; enrichment double-checks each hit so a scope
    /// can never leak a document id.
    pub async fn search_scoped(
        &self,
        req: &SearchRequest,
        perm: &Permitted,
    ) -> Result<Vec<SearchHit>> {
        let q = self.embedder.embed(&req.query).await?;

        // stage 1: per-representation candidate retrieval, scope pushed
        // into the shard as a payload filter on doc_id
        let filter = VectorEngine::scope_filter(perm);
        let dense_hits =
            self.vectors
                .dense_search(&q.dense, req.dense_candidates, filter.clone())?;
        let sparse_hits = self
            .vectors
            .sparse_search(&q.sparse, req.sparse_candidates, filter)?;

        // stage 2: reciprocal rank fusion
        let fused = reciprocal_rank_fusion(&[&dense_hits, &sparse_hits], req.rrf_k);
        let rescore_n = (req.limit * RESCORE_FACTOR).max(RESCORE_MIN);
        let fused_max = fused.first().map(|p| p.score).unwrap_or(1.0);

        // stage 3: ColBERT MaxSim rescoring of the top fused candidates,
        // batched into a single retrieve from the memmap-backed multivectors
        let cand_ids: Vec<ragnarok_core::ChunkId> =
            fused.iter().take(rescore_n).map(|p| p.id).collect();
        let colbert_map: std::collections::HashMap<_, _> = self
            .vectors
            .colbert_scores(&cand_ids, &q.colbert)?
            .into_iter()
            .collect();
        let mut rescored: Vec<(ScoredPoint, f32)> = fused
            .iter()
            .take(rescore_n)
            .map(|p| (*p, colbert_map.get(&p.id).copied().unwrap_or(0.0)))
            .collect();

        // final score: normalized RRF + weighted MaxSim
        for (p, c) in rescored.iter_mut() {
            let rrf_norm = if fused_max > 0.0 {
                p.score / fused_max
            } else {
                0.0
            };
            p.score = rrf_norm * (1.0 - req.colbert_weight) + *c * req.colbert_weight;
        }
        rescored.sort_by(|a, b| {
            b.0.score
                .total_cmp(&a.0.score)
                .then(a.0.id.0.cmp(&b.0.id.0))
        });

        // enrich with stored chunk + document metadata — re-check the
        // scope here as defense-in-depth (the shard filter already did
        // it). Iterate ALL rescored candidates: a cap of `limit*2` could
        // starve the result set when the post-filter drops several
        // candidates (deny scopes, deleted-but-unindexed docs). With a
        // reranker attached we need up to `rerank_top` texts, not just
        // `limit`, so the pool extends that far.
        let want = req.limit.max(self.rerank_top(req));
        let mut pool: Vec<(f32, Chunk, DocumentRecord)> = Vec::with_capacity(want.min(64));
        for (p, _) in rescored.iter() {
            if pool.len() >= want {
                break;
            }
            if let Some((chunk, doc)) = self.meta.get_chunk_with_doc(p.id)? {
                if !perm.allows(&doc.id.to_string()) {
                    continue;
                }
                pool.push((p.score, chunk, doc));
            }
        }

        // stage 4: cross-encoder rerank of the first `rerank_top`
        // candidates — pairwise scores replace the fused score in the
        // reranked window; a failed pass keeps the hybrid order.
        if let Some(rk) = &self.reranker {
            let top = self.rerank_top(req).min(pool.len());
            if top > 1 {
                let texts: Vec<String> =
                    pool[..top].iter().map(|(_, c, _)| c.text.clone()).collect();
                match rk.score(&req.query, &texts).await {
                    Ok(scores) if scores.len() == top => {
                        let mut order: Vec<usize> = (0..top).collect();
                        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
                        let mut ranked: Vec<(f32, Chunk, DocumentRecord)> = Vec::with_capacity(top);
                        for i in order {
                            let (_, c, d) = pool[i].clone();
                            ranked.push((scores[i], c, d));
                        }
                        pool.splice(0..top, ranked);
                    }
                    Ok(scores) => {
                        tracing::warn!(
                            expected = top,
                            got = scores.len(),
                            "reranker returned wrong score count — keeping hybrid order"
                        );
                    }
                    Err(e) => {
                        tracing::warn!("reranker failed ({e}) — keeping hybrid order");
                    }
                }
            }
        }

        let hits = pool
            .into_iter()
            .take(req.limit)
            .enumerate()
            .map(|(rank, (score, chunk, doc))| SearchHit {
                chunk_id: chunk.id,
                document_id: doc.id,
                path: doc.path,
                doc_type: doc.doc_type,
                page: chunk.page,
                bbox: chunk.bbox,
                line_range: chunk.line_range,
                text: chunk.text,
                score,
                rank,
            })
            .collect();
        Ok(hits)
    }

    /// Chunk ids whose scores matter for a query (used by tests/diagnostics).
    pub async fn raw_candidates(&self, req: &SearchRequest) -> Result<Vec<ScoredPoint>> {
        let q = self.embedder.embed(&req.query).await?;
        let dense = self
            .vectors
            .dense_search(&q.dense, req.dense_candidates, None)?;
        let sparse = self
            .vectors
            .sparse_search(&q.sparse, req.sparse_candidates, None)?;
        Ok(reciprocal_rank_fusion(&[&dense, &sparse], req.rrf_k))
    }

    pub fn stats(&self) -> Result<EngineStats> {
        // each point carries all three named representations
        let points = self.vectors.len();
        Ok(EngineStats {
            documents: self.meta.doc_count()?,
            chunks: self.meta.chunk_count()?,
            dense_points: points,
            sparse_docs: points,
            colbert_docs: points,
        })
    }

    /// Persist vector snapshot (metadata lives in SQLite already).
    pub fn snapshot(&self) -> Result<()> {
        self.vectors.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ragnarok_core::{document_id, DocType, SourceId};
    use ragnarok_inference::HashingEmbedder;

    async fn engine() -> (tempfile::TempDir, Arc<SearchEngine>) {
        let dir = tempfile::tempdir().unwrap();
        let e = SearchEngine::open(dir.path(), Arc::new(HashingEmbedder::default()), 1024).unwrap();
        (dir, e)
    }

    fn chunk(doc: DocumentId, i: u32, text: &str) -> Chunk {
        Chunk {
            id: ragnarok_core::chunk_id(doc, i),
            document_id: doc,
            chunk_index: i,
            page: 0,
            bbox: None,
            line_range: None,
            text: text.into(),
            token_count: text.split_whitespace().count(),
        }
    }

    async fn index(e: &SearchEngine, path: &str, text: &str) -> DocumentId {
        let src = SourceId::from("s1");
        let doc = document_id(&src, path);
        let rec = DocumentRecord {
            id: doc,
            source_id: src.clone(),
            path: path.into(),
            hash: "h".into(),
            doc_type: DocType::Text,
            size: text.len() as u64,
            modified: None,
            indexed_at: chrono::Utc::now(),
            chunk_count: 1,
            metadata: Default::default(),
        };
        let chunks = vec![chunk(doc, 0, text)];
        let triples = e.embedder.embed_batch(&[text.to_string()]).await.unwrap();
        e.index_document(&rec, &chunks, &triples).await.unwrap();
        doc
    }

    #[tokio::test]
    async fn hybrid_search_ranks_relevant_first() {
        let (_t, e) = engine().await;
        index(&e, "/cats.txt", "cats are fluffy pets that purr").await;
        index(&e, "/dogs.txt", "dogs are loyal pets that bark").await;
        index(&e, "/physics.txt", "quantum electrodynamics lecture").await;

        let hits = e.search(&SearchRequest::new("cats purr")).await.unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].path, "/cats.txt");
        assert_eq!(hits[0].rank, 0);
        assert!(hits[0].score >= hits.last().unwrap().score);
    }

    #[tokio::test]
    async fn delete_document_removes_hits() {
        let (_t, e) = engine().await;
        let doc = index(&e, "/a.txt", "unique pangolin sentence").await;
        let hits = e.search(&SearchRequest::new("pangolin")).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert!(e.delete_document(doc).unwrap());
        let hits = e.search(&SearchRequest::new("pangolin")).await.unwrap();
        assert!(hits.is_empty());
        assert!(!e.delete_document(doc).unwrap());
    }

    #[tokio::test]
    async fn stats_and_snapshot() {
        let (t, e) = engine().await;
        index(&e, "/a.txt", "alpha beta gamma").await;
        let s = e.stats().unwrap();
        assert_eq!(s.documents, 1);
        assert_eq!(s.chunks, 1);
        assert_eq!(s.colbert_docs, 1);
        e.snapshot().unwrap();
        // reopen from disk — the WAL is single-writer, so the first engine
        // must release its lock before a second one can open the same dir
        drop(e);
        let e2 = SearchEngine::open(t.path(), Arc::new(HashingEmbedder::default()), 1024).unwrap();
        let hits = e2.search(&SearchRequest::new("alpha")).await.unwrap();
        assert_eq!(hits[0].path, "/a.txt");
    }

    #[tokio::test]
    async fn index_rejects_mismatched_lengths() {
        let (_t, e) = engine().await;
        let doc = document_id(&SourceId::from("s"), "/x");
        let rec = DocumentRecord {
            id: doc,
            source_id: "s".into(),
            path: "/x".into(),
            hash: "h".into(),
            doc_type: DocType::Text,
            size: 1,
            modified: None,
            indexed_at: chrono::Utc::now(),
            chunk_count: 1,
            metadata: Default::default(),
        };
        let r = e
            .index_document(&rec, &[chunk(doc, 0, "a")], &[]) // 1 chunk, 0 triples
            .await;
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn raw_candidates_returns_fused_ids() {
        let (_t, e) = engine().await;
        let id = index(&e, "/c.txt", "candidate retrieval path").await;
        let cands = e
            .raw_candidates(&SearchRequest::new("retrieval"))
            .await
            .unwrap();
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].id, ragnarok_core::chunk_id(id, 0));
    }

    #[tokio::test]
    async fn reindex_replaces_old_chunks() {
        let (_t, e) = engine().await;
        index(&e, "/r.txt", "version one text").await;
        index(&e, "/r.txt", "version two text").await; // same path -> same doc id
        let s = e.stats().unwrap();
        assert_eq!(s.documents, 1);
        assert_eq!(s.chunks, 1); // replaced, not duplicated
    }

    #[tokio::test]
    async fn limit_is_respected() {
        let (_t, e) = engine().await;
        for i in 0..8 {
            index(&e, &format!("/{i}.txt"), &format!("shared token doc {i}")).await;
        }
        let mut req = SearchRequest::new("shared token");
        req.limit = 3;
        assert_eq!(e.search(&req).await.unwrap().len(), 3);
    }

    /// Content-keyed reranker — scores by whether the chunk text contains
    /// the marker word. Deterministic regardless of the hybrid order the
    /// hashing embedder happens to produce.
    struct MarkerReranker(&'static str);
    #[async_trait::async_trait]
    impl ragnarok_inference::Reranker for MarkerReranker {
        fn name(&self) -> &'static str {
            "marker"
        }
        async fn score(&self, _q: &str, docs: &[String]) -> Result<Vec<f32>> {
            Ok(docs
                .iter()
                .map(|d| if d.contains(self.0) { 1.0 } else { 0.0 })
                .collect())
        }
    }

    /// Broken reranker — engine must fall back to hybrid order.
    struct FailingReranker;
    #[async_trait::async_trait]
    impl ragnarok_inference::Reranker for FailingReranker {
        fn name(&self) -> &'static str {
            "failing"
        }
        async fn score(&self, _q: &str, _docs: &[String]) -> Result<Vec<f32>> {
            Err(ragnarok_core::Error::Inference("boom".into()))
        }
    }

    #[tokio::test]
    async fn reranker_reorders_top_candidates() {
        let (_t, e) = engine().await;
        index(&e, "/a.txt", "shared token alpha").await;
        index(&e, "/b.txt", "shared token pick-me beta").await;
        index(&e, "/c.txt", "shared token gamma").await;

        let mut e = e;
        Arc::get_mut(&mut e)
            .unwrap()
            .set_reranker(Arc::new(MarkerReranker("pick-me")));

        let mut req = SearchRequest::new("shared token");
        req.limit = 3;
        let hits = e.search(&req).await.unwrap();
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].path, "/b.txt");
        assert_eq!(hits[0].score, 1.0); // rerank score replaces fused score
        assert_eq!(hits[1].score, 0.0);
        assert_eq!(
            hits.iter().map(|h| h.rank).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[tokio::test]
    async fn rerank_top_zero_disables_stage() {
        let (_t, e) = engine().await;
        index(&e, "/a.txt", "shared token alpha").await;
        index(&e, "/b.txt", "shared token pick-me beta").await;
        let mut base_req = SearchRequest::new("shared token");
        base_req.rerank_top = 0;
        let baseline = e.search(&base_req).await.unwrap();

        let mut e = e;
        Arc::get_mut(&mut e)
            .unwrap()
            .set_reranker(Arc::new(MarkerReranker("pick-me")));

        let mut req = SearchRequest::new("shared token");
        req.rerank_top = 0;
        let hits = e.search(&req).await.unwrap();
        // rerank disabled → identical hybrid order and scores
        assert_eq!(
            hits.iter().map(|h| (&h.path, h.score)).collect::<Vec<_>>(),
            baseline
                .iter()
                .map(|h| (&h.path, h.score))
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn failing_reranker_keeps_hybrid_order() {
        let (_t, e) = engine().await;
        index(&e, "/a.txt", "shared token alpha").await;
        index(&e, "/b.txt", "shared token beta").await;
        let baseline = e.search(&SearchRequest::new("shared token")).await.unwrap();

        let mut e = e;
        Arc::get_mut(&mut e)
            .unwrap()
            .set_reranker(Arc::new(FailingReranker));
        let hits = e.search(&SearchRequest::new("shared token")).await.unwrap();
        let base_paths: Vec<_> = baseline.iter().map(|h| &h.path).collect();
        let hit_paths: Vec<_> = hits.iter().map(|h| &h.path).collect();
        assert_eq!(hit_paths, base_paths); // hybrid order preserved
    }

    #[tokio::test]
    async fn lexical_reranker_promotes_best_coverage() {
        let (_t, e) = engine().await;
        // hybrid (hashing) ranks all "purr"-docs similar; the lexical
        // reranker scores query-token coverage
        index(&e, "/partial.txt", "cats sleep all day").await;
        index(&e, "/full.txt", "cats purr loudly").await;
        let mut e = e;
        Arc::get_mut(&mut e)
            .unwrap()
            .set_reranker(Arc::new(ragnarok_inference::LexicalReranker));

        let mut req = SearchRequest::new("cats purr");
        req.limit = 2;
        let hits = e.search(&req).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].path, "/full.txt");
        assert!(hits[0].score >= hits[1].score);
    }
}
