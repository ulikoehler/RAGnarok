//! Cross-encoder reranking — scores (query, document) *pairs* after the
//! retrieval fusion stage and re-orders the top candidates.
//!
//! Unlike the bi-encoder `Embedder` (which maps query and document
//! independently and compares vectors), a cross-encoder feeds both texts
//! through one forward pass, so attention sees query↔document token
//! interactions. That is strictly more accurate but O(N) forward passes
//! per query — hence the stage is capped to the top `rerank_top` fused
//! candidates (default 20).
//!
//! [`OnnxReranker`] (feature `onnx`) runs `bge-reranker-base` via ONNX
//! Runtime; [`LexicalReranker`] is a deterministic token-overlap scorer
//! for hashing builds and tests.

use async_trait::async_trait;
#[cfg(feature = "onnx")]
use ragnarok_core::Error;
use ragnarok_core::Result;

/// Pairwise relevance scorer — one score per input document, same order.
/// Async because ONNX forward passes run on the blocking pool.
#[async_trait]
pub trait Reranker: Send + Sync {
    fn name(&self) -> &'static str;
    /// Score every `docs[i]` against `query`. Higher is more relevant;
    /// implementations return scores in `[0,1]` (ONNX: sigmoid on logits).
    async fn score(&self, query: &str, docs: &[String]) -> Result<Vec<f32>>;
}

/// Deterministic lexical scorer — normalized coverage of unique query
/// tokens in the document. No model, no dependencies; used in tests and
/// as a light rerank stage on hashing-only builds.
pub struct LexicalReranker;

fn vocab(text: &str) -> std::collections::HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 1)
        .map(|t| t.to_lowercase())
        .collect()
}

#[async_trait]
impl Reranker for LexicalReranker {
    fn name(&self) -> &'static str {
        "lexical-reranker"
    }

    async fn score(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        let qv = vocab(query);
        if qv.is_empty() {
            return Ok(vec![0.0; docs.len()]);
        }
        Ok(docs
            .iter()
            .map(|d| {
                let dv = vocab(d);
                qv.intersection(&dv).count() as f32 / qv.len() as f32
            })
            .collect())
    }
}

/// `bge-reranker-base` (XLM-RoBERTa cross-encoder) via ONNX Runtime.
/// Expects `model_int8.onnx`/`model.onnx` + `tokenizer.json` — layout of
/// the `Xenova/bge-reranker-base` export the provisioner downloads.
#[cfg(feature = "onnx")]
pub struct OnnxReranker {
    session: parking_lot::Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
}

#[cfg(feature = "onnx")]
impl OnnxReranker {
    /// Load model + tokenizer from `dir`.
    pub fn load(dir: &std::path::Path) -> Result<std::sync::Arc<Self>> {
        use std::path::PathBuf;
        let onnx_path: PathBuf = [
            "model_int8.onnx",
            "model_quantized.onnx",
            "model.onnx",
            "reranker.onnx",
        ]
        .iter()
        .map(|f| dir.join(f))
        .find(|p| p.exists())
        .ok_or_else(|| Error::Inference(format!("no reranker .onnx found in {}", dir.display())))?;
        let tok_path = dir.join("tokenizer.json");
        if !tok_path.exists() {
            return Err(Error::Inference(format!(
                "tokenizer.json missing in {}",
                dir.display()
            )));
        }
        let env = ort::init()
            .build()
            .map_err(|e| Error::Inference(format!("ort env: {e}")))?;
        let session = ort::session::Session::builder(&env)
            .map_err(|e| Error::Inference(format!("ort init: {e}")))?
            .with_intra_threads(num_cpus())
            .map_err(|e| Error::Inference(format!("ort threads: {e}")))?
            .commit_from_file(&onnx_path)
            .map_err(|e| Error::Inference(format!("ort load {}: {e}", onnx_path.display())))?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| Error::Inference(format!("tokenizer load: {e}")))?;
        Ok(std::sync::Arc::new(OnnxReranker {
            session: parking_lot::Mutex::new(session),
            tokenizer,
        }))
    }

    /// Blocking batch forward — one (query, doc) pair per row.
    fn forward(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        let mut encs = Vec::with_capacity(docs.len());
        for d in docs {
            encs.push(
                self.tokenizer
                    .encode((query, d.as_str()), true)
                    .map_err(|e| Error::Inference(format!("tokenize: {e}")))?,
            );
        }
        let batch = encs.len();
        let seq = encs.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
        if batch == 0 {
            return Ok(vec![]);
        }
        if seq == 0 {
            return Err(Error::Inference("empty token sequence".into()));
        }
        let mut ids = vec![0i64; batch * seq];
        let mut mask = vec![0i64; batch * seq];
        let mut type_ids = vec![0i64; batch * seq];
        for (i, enc) in encs.iter().enumerate() {
            for (j, &t) in enc.get_ids().iter().enumerate() {
                ids[i * seq + j] = t as i64;
            }
            for (j, &m) in enc.get_attention_mask().iter().enumerate() {
                mask[i * seq + j] = m as i64;
            }
            for (j, &t) in enc.get_type_ids().iter().enumerate() {
                type_ids[i * seq + j] = t as i64;
            }
        }
        let shape = vec![batch, seq];
        let mk = |data: Vec<i64>| {
            ort::value::Tensor::from_array((shape.clone(), data))
                .map_err(|e| Error::Inference(e.to_string()))
        };
        let ids = mk(ids)?;
        let mask = mk(mask)?;
        let type_ids = mk(type_ids)?;

        let mut session = self.session.lock();
        // feed exactly the inputs the graph declares — XLM-R exports vary
        // on whether `token_type_ids` is present
        let wants_types = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");
        let outputs = if wants_types {
            session
                .run(ort::inputs![ids, mask, type_ids])
                .map_err(|e| Error::Inference(format!("ort run: {e}")))?
        } else {
            session
                .run(ort::inputs![ids, mask])
                .map_err(|e| Error::Inference(format!("ort run: {e}")))?
        };
        let logits = outputs
            .get("logits")
            .ok_or_else(|| Error::Inference("reranker: no logits output".into()))?;
        let (_, data) = logits
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Inference(format!("logits extract: {e}")))?;
        if data.len() < batch {
            return Err(Error::Inference(format!(
                "logits too short: {} for batch {batch}",
                data.len()
            )));
        }
        Ok(data[..batch]
            .iter()
            .map(|&l| 1.0 / (1.0 + (-l).exp()))
            .collect())
    }
}

#[cfg(feature = "onnx")]
#[async_trait]
impl Reranker for OnnxReranker {
    fn name(&self) -> &'static str {
        "bge-reranker-onnx"
    }

    async fn score(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        let q = query.to_string();
        let d = docs.to_vec();
        let this = unsafe { &*(self as *const Self) };
        tokio::task::spawn_blocking(move || this.forward(&q, &d))
            .await
            .map_err(|e| Error::Inference(format!("blocking join: {e}")))?
    }
}

#[cfg(feature = "onnx")]
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn lexical_scores_coverage() {
        let r = LexicalReranker;
        let docs = vec![
            "cats purr loudly".to_string(),
            "dogs bark at cats".to_string(),
            "quantum physics".to_string(),
        ];
        let s = r.score("cats purr", &docs).await.unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s[0], 1.0); // both query tokens covered
        assert_eq!(s[1], 0.5); // only "cats"
        assert_eq!(s[2], 0.0);
    }

    #[tokio::test]
    async fn lexical_empty_query_yields_zeros() {
        let r = LexicalReranker;
        let s = r.score("", &["doc".into()]).await.unwrap();
        assert_eq!(s, vec![0.0]);
    }
}
