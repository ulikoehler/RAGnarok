use async_trait::async_trait;
use ragnarok_core::{EmbeddingTriple, Result};

/// Produces the BGE-M3-style triple representation for a text.
///
/// Contract for all implementations:
/// * `dense` is exactly [`crate::DENSE_DIM`] elements and L2-normalized.
/// * `sparse` keys are `< crate::SPARSE_VOCAB`, weights in `(0, 1]`.
/// * every `colbert` row has [`crate::COLBERT_DIM`] elements, L2-normalized.
/// * embedding the same text twice yields identical output (deterministic),
///   so re-indexed documents stay comparable.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Human-readable backend name for telemetry/logging.
    fn name(&self) -> &'static str;

    /// Embed a single text.
    async fn embed(&self, text: &str) -> Result<EmbeddingTriple>;

    /// Embed a batch. Default implementation is sequential; backends with
    /// real batching (ONNX) should override.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingTriple>> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.embed(t).await?);
        }
        Ok(out)
    }
}
