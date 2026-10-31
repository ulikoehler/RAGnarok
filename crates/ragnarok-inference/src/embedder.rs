use async_trait::async_trait;
use ragnarok_core::{EmbeddingTriple, Result};

/// Produces the BGE-M3-style triple representation for a text.
///
/// Contract for all implementations:
/// * `dense` is exactly [`Embedder::dense_dim`] elements and L2-normalized.
/// * `sparse` keys are `< crate::SPARSE_VOCAB`, weights in `(0, 1]`.
/// * every `colbert` row has [`crate::COLBERT_DIM`] elements, L2-normalized.
/// * embedding the same text twice yields identical output (deterministic),
///   so re-indexed documents stay comparable.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Human-readable backend name for telemetry/logging.
    fn name(&self) -> &'static str;

    /// Dense vector dimensionality — the vector store is created with
    /// this size. Defaults to BGE-M3's 1024; other registry models
    /// report their own (e5-small → 384, e5-base → 768, …).
    fn dense_dim(&self) -> usize {
        crate::DENSE_DIM
    }

    /// Embed a single document/chunk text.
    async fn embed(&self, text: &str) -> Result<EmbeddingTriple>;

    /// Embed a search query. Identical to [`Embedder::embed`] except for
    /// models trained with role prefixes (E5 `query:`/`passage:`, BGE-en
    /// instruction) — those prepend the query prefix here and the
    /// document prefix in `embed`/`embed_batch`.
    async fn embed_query(&self, text: &str) -> Result<EmbeddingTriple> {
        self.embed(text).await
    }

    /// Embed a batch. Default implementation is sequential; backends with
    /// real batching (ONNX) should override.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingTriple>> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.embed(t).await?);
        }
        Ok(out)
    }

    /// When a GPU execution provider was abandoned mid-run, the reason —
    /// surfaced via `/inference` and persisted so later restarts skip the
    /// dead device instead of crashing into it again.
    fn gpu_dead(&self) -> Option<String> {
        None
    }

    /// Execution provider actually in use — e.g.
    /// `WebGpuExecutionProvider` or `cpu` after a GPU fallback.
    fn provider_label(&self) -> Option<String> {
        None
    }
}
