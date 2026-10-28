//! ragnarok-store — embedded storage layer.
//!
//! Mirrors the collection layout specified for an embedded Qdrant instance
//! (the `qdrant-edge` crate drags in ~120 dependencies incl. RocksDB/io-uring,
//! so the storage engine is implemented in-process behind [`VectorEngine`]
//! with identical semantics):
//!
//! * `dense`   — 1024-dim cosine, HNSW index, int8 scalar quantization in RAM
//! * `colbert` — 128-dim multi-vector, MaxSim comparator, no HNSW (m=0)
//! * `lexical` — sparse inverted index, persisted on disk
//!
//! [`MetadataDb`] (SQLite) tracks files, chunks, offsets and IMAP sync state.
//! [`SearchEngine`] fuses the three representations via Reciprocal Rank
//! Fusion and rescores with ColBERT MaxSim.

pub mod engine;
pub mod metadata;
pub mod vector;

pub use engine::{EngineStats, SearchEngine};
pub use metadata::MetadataDb;
pub use vector::{maxsim, reciprocal_rank_fusion, ScalarQuantizer, ScoredPoint};
