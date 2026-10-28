use super::quant::{QuantizedVec, ScalarQuantizer};
use super::ScoredPoint;
use dashmap::DashMap;
use hnsw_rs::prelude::{DistCosine, Hnsw};
use parking_lot::RwLock;
use ragnarok_core::{ChunkId, Error, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Dense ANN index: HNSW over cosine distance with int8 scalar-quantized
/// vectors resident in RAM (Qdrant `dense` collection semantics).
pub struct DenseIndex {
    hnsw: RwLock<Hnsw<'static, f32, DistCosine>>,
    quant: ScalarQuantizer,
    /// int8 vectors — the only copies kept in RAM.
    vectors: DashMap<ChunkId, QuantizedVec>,
    /// internal hnsw id <-> chunk id
    to_chunk: RwLock<HashMap<usize, ChunkId>>,
    to_internal: DashMap<ChunkId, usize>,
    next_internal: AtomicUsize,
    dim: usize,
}

const EF_SEARCH: usize = 128;

impl DenseIndex {
    pub fn new(dim: usize) -> Self {
        DenseIndex {
            hnsw: RwLock::new(Hnsw::new(
                32,        // max_nb_connection (M)
                1_000_000, // capacity hint
                16,        // max_layer
                200,       // ef_construction
                DistCosine,
            )),
            quant: ScalarQuantizer::new(dim),
            vectors: DashMap::new(),
            to_chunk: RwLock::new(HashMap::new()),
            to_internal: DashMap::new(),
            next_internal: AtomicUsize::new(1), // 0 reserved by hnsw entry point
            dim,
        }
    }

    /// Insert or replace a dense vector.
    pub fn upsert(&self, id: ChunkId, v: &[f32]) -> Result<()> {
        let qv = self.quant.quantize(v)?;
        self.insert_quantized(id, qv)
    }

    /// Insert an already-quantized vector (used on snapshot load).
    pub fn insert_quantized(&self, id: ChunkId, qv: QuantizedVec) -> Result<()> {
        let deq = self.quant.dequantize(&qv);
        let internal = *self.to_internal.entry(id).or_insert_with(|| {
            let i = self.next_internal.fetch_add(1, Ordering::Relaxed);
            self.to_chunk.write().insert(i, id);
            i
        });
        self.hnsw.read().insert((&deq, internal));
        self.vectors.insert(id, qv);
        Ok(())
    }

    /// k-NN cosine search; rescored against the dequantized vectors.
    /// Returns at most `k` hits sorted by score desc. Deleted points are
    /// filtered by [`VectorEngine`], not here.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<ScoredPoint>> {
        if query.len() != self.dim {
            return Err(Error::VectorStore(format!(
                "query dim {} != index dim {}",
                query.len(),
                self.dim
            )));
        }
        if self.vectors.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let neighbours = self
            .hnsw
            .read()
            .search(query, k.min(self.vectors.len()), EF_SEARCH);
        let to_chunk = self.to_chunk.read();
        let mut out = Vec::with_capacity(neighbours.len());
        for n in neighbours {
            let Some(&id) = to_chunk.get(&n.d_id) else {
                continue;
            };
            // rescore on the dequantized int8 vector
            let score = self
                .vectors
                .get(&id)
                .map(|qv| self.quant.cosine(query, &qv))
                .unwrap_or(0.0);
            out.push(ScoredPoint { id, score });
        }
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        out.truncate(k);
        Ok(out)
    }

    pub fn remove(&self, id: ChunkId) {
        self.vectors.remove(&id);
        // HNSW graph entries are tombstoned implicitly: callers filter
        // results against `vectors`. A rebuild on snapshot-load compacts.
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn quantized_entries(&self) -> Vec<(ChunkId, QuantizedVec)> {
        self.vectors
            .iter()
            .map(|e| (*e.key(), e.value().clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(dim: usize, seed: f32) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.618 + seed).sin()).collect();
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= n);
        v
    }

    #[test]
    fn self_is_top_hit() {
        let idx = DenseIndex::new(64);
        let id = ChunkId::new();
        let v = unit_vec(64, 1.0);
        idx.upsert(id, &v).unwrap();
        for i in 0..10 {
            idx.upsert(ChunkId::new(), &unit_vec(64, 100.0 + i as f32))
                .unwrap();
        }
        let hits = idx.search(&v, 3).unwrap();
        assert_eq!(hits[0].id, id);
        assert!(hits[0].score > 0.99);
        assert!(hits.len() <= 3);
    }

    #[test]
    fn nearest_ordering() {
        let idx = DenseIndex::new(32);
        let near = ChunkId::new();
        let far = ChunkId::new();
        idx.upsert(near, &unit_vec(32, 1.00)).unwrap();
        idx.upsert(far, &unit_vec(32, 1.30)).unwrap();
        let hits = idx.search(&unit_vec(32, 1.01), 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert!(hits[0].score >= hits[1].score);
    }

    #[test]
    fn dim_validation_and_empty() {
        let idx = DenseIndex::new(8);
        assert!(idx.search(&[0.0; 4], 1).is_err());
        assert!(idx.search(&[0.0; 8], 1).unwrap().is_empty());
        assert!(idx.upsert(ChunkId::new(), &[0.0; 4]).is_err());
    }
}
