use super::ScoredPoint;
use dashmap::DashMap;
use parking_lot::RwLock;
use ragnarok_core::{ChunkId, SparseVector};
use std::collections::HashMap;

/// Sparse lexical index (Qdrant `lexical` collection semantics): an inverted
/// index over token-id → weighted postings, scored by cosine-ish overlap.
/// Postings are rebuilt from the per-document weight maps held in memory;
/// the engine persists those maps in its on-disk snapshot.
pub struct SparseIndex {
    /// doc -> token weights (source of truth, persisted).
    docs: DashMap<ChunkId, SparseVector>,
    /// token -> [(doc, weight)] postings, derived from `docs`.
    postings: RwLock<HashMap<u32, Vec<(ChunkId, f32)>>>,
}

impl SparseIndex {
    pub fn new() -> Self {
        SparseIndex {
            docs: DashMap::new(),
            postings: RwLock::new(HashMap::new()),
        }
    }

    pub fn upsert(&self, id: ChunkId, sparse: &SparseVector) {
        // remove stale postings first
        if let Some(old) = self.docs.get(&id) {
            let old_keys: Vec<u32> = old.keys().copied().collect();
            drop(old);
            let mut p = self.postings.write();
            for tok in old_keys {
                if let Some(list) = p.get_mut(&tok) {
                    list.retain(|(d, _)| *d != id);
                    if list.is_empty() {
                        p.remove(&tok);
                    }
                }
            }
        }
        let mut p = self.postings.write();
        for (tok, w) in sparse {
            p.entry(*tok).or_default().push((id, *w));
        }
        drop(p);
        self.docs.insert(id, sparse.clone());
    }

    /// Rebuild postings from `docs` (after bulk snapshot load).
    pub fn rebuild_postings(&self) {
        let mut p = HashMap::new();
        for e in self.docs.iter() {
            for (tok, w) in e.value() {
                p.entry(*tok).or_insert_with(Vec::new).push((*e.key(), *w));
            }
        }
        *self.postings.write() = p;
    }

    pub fn remove(&self, id: ChunkId) {
        if let Some((_, old)) = self.docs.remove(&id) {
            let mut p = self.postings.write();
            for tok in old.keys() {
                if let Some(list) = p.get_mut(tok) {
                    list.retain(|(d, _)| *d != id);
                    if list.is_empty() {
                        p.remove(tok);
                    }
                }
            }
        }
    }

    /// Score docs sharing tokens with the query: `Σ wq * wd` normalized by
    /// query norm (weights are already in `(0,1]`).
    pub fn search(&self, query: &SparseVector, k: usize) -> Vec<ScoredPoint> {
        if query.is_empty() || k == 0 {
            return Vec::new();
        }
        let qnorm: f32 = query.values().map(|w| w * w).sum::<f32>().sqrt().max(1e-9);
        let mut scores: HashMap<ChunkId, f32> = HashMap::new();
        let postings = self.postings.read();
        for (tok, qw) in query {
            if let Some(list) = postings.get(tok) {
                for (id, dw) in list {
                    *scores.entry(*id).or_insert(0.0) += qw * dw;
                }
            }
        }
        drop(postings);
        let mut out: Vec<ScoredPoint> = scores
            .into_iter()
            .map(|(id, s)| ScoredPoint {
                id,
                score: s / qnorm,
            })
            .collect();
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        out.truncate(k);
        out
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn doc_entries(&self) -> Vec<(ChunkId, SparseVector)> {
        self.docs
            .iter()
            .map(|e| (*e.key(), e.value().clone()))
            .collect()
    }
}

impl Default for SparseIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sv(pairs: &[(u32, f32)]) -> SparseVector {
        pairs.iter().copied().collect()
    }

    #[test]
    fn exact_token_match_wins() {
        let idx = SparseIndex::new();
        let a = ChunkId::new();
        let b = ChunkId::new();
        idx.upsert(a, &sv(&[(1, 1.0), (2, 0.5)]));
        idx.upsert(b, &sv(&[(3, 1.0)]));
        let hits = idx.search(&sv(&[(1, 1.0)]), 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, a);
        assert!(hits[0].score > 0.0);
    }

    #[test]
    fn overlap_scoring() {
        let idx = SparseIndex::new();
        let a = ChunkId::new();
        let b = ChunkId::new();
        idx.upsert(a, &sv(&[(1, 1.0), (2, 1.0)]));
        idx.upsert(b, &sv(&[(1, 0.5)]));
        let hits = idx.search(&sv(&[(1, 1.0), (2, 1.0)]), 10);
        assert_eq!(hits[0].id, a);
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn update_and_remove() {
        let idx = SparseIndex::new();
        let a = ChunkId::new();
        idx.upsert(a, &sv(&[(1, 1.0)]));
        idx.upsert(a, &sv(&[(2, 1.0)])); // replace
        assert!(idx.search(&sv(&[(1, 1.0)]), 10).is_empty());
        assert_eq!(idx.search(&sv(&[(2, 1.0)]), 10)[0].id, a);
        idx.remove(a);
        assert!(idx.search(&sv(&[(2, 1.0)]), 10).is_empty());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn rebuild_from_docs() {
        let idx = SparseIndex::new();
        let a = ChunkId::new();
        idx.docs.insert(a, sv(&[(7, 1.0)]));
        idx.rebuild_postings();
        assert_eq!(idx.search(&sv(&[(7, 1.0)]), 5)[0].id, a);
    }

    #[test]
    fn empty_query() {
        let idx = SparseIndex::new();
        assert!(idx.search(&SparseVector::new(), 5).is_empty());
    }
}
