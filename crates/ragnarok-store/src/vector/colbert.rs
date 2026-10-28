use dashmap::DashMap;
use ragnarok_core::ChunkId;

/// Multi-vector store (Qdrant `colbert` collection semantics): no HNSW
/// (m = 0) — used strictly as a rescoring layer over fused candidates.
pub struct ColbertStore {
    docs: DashMap<ChunkId, Vec<Vec<f32>>>,
}

impl ColbertStore {
    pub fn new() -> Self {
        ColbertStore {
            docs: DashMap::new(),
        }
    }

    pub fn upsert(&self, id: ChunkId, rows: Vec<Vec<f32>>) {
        self.docs.insert(id, rows);
    }

    pub fn get(&self, id: ChunkId) -> Option<Vec<Vec<f32>>> {
        self.docs.get(&id).map(|e| e.clone())
    }

    pub fn remove(&self, id: ChunkId) {
        self.docs.remove(&id);
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    pub fn entries(&self) -> Vec<(ChunkId, Vec<Vec<f32>>)> {
        self.docs
            .iter()
            .map(|e| (*e.key(), e.value().clone()))
            .collect()
    }
}

impl Default for ColbertStore {
    fn default() -> Self {
        Self::new()
    }
}
