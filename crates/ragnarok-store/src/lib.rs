//! ragnarok-store — embedded storage layer.
//!
//! [`VectorEngine`] wraps a `qdrant-edge` [`EdgeShard`](qdrant_edge::EdgeShard)
//! — Qdrant's in-process, memmap/WAL-backed storage engine — with the
//! collection layout from the spec:
//!
//! * `dense`   — 1024-dim cosine, HNSW index, int8 scalar quantization in RAM
//! * `colbert` — 128-dim multi-vector, MaxSim comparator, no HNSW (m=0)
//! * `lexical` — sparse index, persisted on disk
//!
//! [`MetadataDb`] (SQLite) tracks files, chunks, offsets and IMAP sync state.
//! [`SearchEngine`] fuses the three representations via Reciprocal Rank
//! Fusion and rescores with ColBERT MaxSim.

pub mod engine;
pub mod metadata;
pub mod scope;
pub mod vector;

pub use engine::{EngineStats, SearchEngine};
pub use metadata::MetadataDb;
pub use scope::{resolve_permitted, user_permitted};
pub use vector::{maxsim, reciprocal_rank_fusion, ScoredPoint, VectorEngine};
