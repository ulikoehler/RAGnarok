//! ragnarok-inference — BGE-M3-style triple embedding pipeline.
//!
//! Every text chunk is mapped to three representations:
//! * **dense** — 1024-dim L2-normalized semantic vector
//! * **sparse** — lexical token-id → weight mapping
//! * **colbert** — N×128 token-level matrix for MaxSim late interaction
//!
//! [`HashingEmbedder`] is a deterministic, dependency-free implementation of
//! the [`Embedder`] contract used for development and tests. [`OnnxBgeM3`]
//! (feature `onnx`) runs the real model via ONNX Runtime.

mod embedder;
mod hashing;
mod pool;
pub mod provision;
mod tokenizer;

#[cfg(feature = "onnx")]
pub mod onnx;

pub use embedder::Embedder;
pub use hashing::HashingEmbedder;
pub use pool::InferencePool;
pub use provision::{ModelState, ModelStatus, Provisioner, SharedModelStatus};
pub use tokenizer::{token_id, Token, Tokenizer, SPARSE_VOCAB};

/// Dense embedding dimensionality (BGE-M3).
pub const DENSE_DIM: usize = 1024;
/// ColBERT token-vector dimensionality.
pub const COLBERT_DIM: usize = 128;

/// Whether this binary was compiled with the `onnx` feature — surfaced by
/// `/inference` so the UI can explain why backend `onnx` can't be started.
pub const ONNX_COMPILED: bool = cfg!(feature = "onnx");
