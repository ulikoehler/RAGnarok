//! Real BGE-M3 inference via ONNX Runtime (`ort`) — enabled with
//! `--features onnx`.
//!
//! Expected model directory layout (e.g. an `optimum`/`FlagEmbedding` export):
//! ```text
//! model_dir/
//! ├── model.onnx        (or model_optimized.onnx)
//! └── tokenizer.json    (XLM-RoBERTa / SentencePiece)
//! ```
//!
//! Output tensor names are resolved dynamically so both the
//! `BAAI/bge-m3` ONNX export (`dense_vecs`, `sparse_vecs`,
//! `last_hidden_state`) and optimum-style exports work. If the model's token
//! hidden size differs from [`COLBERT_DIM`], the ColBERT vectors are
//! deterministically resampled to 128 dimensions — a faithful 128-d projection
//! head can be dropped in later without touching callers.

use crate::embedder::Embedder;
use crate::{COLBERT_DIM, DENSE_DIM};
use async_trait::async_trait;
use ort::ndarray::{Array2, Axis};
use ort::session::Session;
use ort::value::TensorRef;
use parking_lot::Mutex;
use ragnarok_core::{EmbeddingTriple, Error, Result, SparseVector};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

pub struct OnnxBgeM3 {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_colbert_tokens: usize,
}

impl OnnxBgeM3 {
    /// Load model + tokenizer from `model_dir`.
    pub fn load(model_dir: &Path, max_colbert_tokens: usize) -> Result<Arc<Self>> {
        let onnx_path = ["model_optimized.onnx", "model.onnx", "bge-m3.onnx"]
            .iter()
            .map(|f| model_dir.join(f))
            .find(|p| p.exists())
            .ok_or_else(|| {
                Error::Inference(format!("no .onnx model found in {}", model_dir.display()))
            })?;
        let tok_path: PathBuf = model_dir.join("tokenizer.json");
        if !tok_path.exists() {
            return Err(Error::Inference(format!(
                "tokenizer.json missing in {}",
                model_dir.display()
            )));
        }
        let session = Session::builder()
            .map_err(|e| Error::Inference(format!("ort init: {e}")))?
            .with_intra_threads(num_cpus())
            .map_err(|e| Error::Inference(format!("ort threads: {e}")))?
            .commit_from_file(&onnx_path)
            .map_err(|e| Error::Inference(format!("ort load {}: {e}", onnx_path.display())))?;
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| Error::Inference(format!("tokenizer load: {e}")))?;
        Ok(Arc::new(OnnxBgeM3 {
            session: Mutex::new(session),
            tokenizer,
            max_colbert_tokens: max_colbert_tokens.max(1),
        }))
    }

    /// Blocking forward pass — called inside `spawn_blocking`.
    fn forward(&self, text: &str) -> Result<EmbeddingTriple> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::Inference(format!("tokenize: {e}")))?;
        let seq = enc.get_ids().len();
        if seq == 0 {
            return Err(Error::Inference("empty token sequence".into()));
        }
        let ids =
            Array2::from_shape_vec((1, seq), enc.get_ids().iter().map(|&x| x as i64).collect())
                .map_err(|e| Error::Inference(e.to_string()))?;
        let mask = Array2::from_shape_vec(
            (1, seq),
            enc.get_attention_mask().iter().map(|&x| x as i64).collect(),
        )
        .map_err(|e| Error::Inference(e.to_string()))?;
        let token_ids: Vec<u32> = enc.get_ids().to_vec();

        let mut session = self.session.lock();
        let outputs = session
            .run(ort::inputs![ids, mask])
            .map_err(|e| Error::Inference(format!("ort run: {e}")))?;

        // ---- dense ----
        let dense = extract_dense(&outputs, seq)
            .ok_or_else(|| Error::Inference("no dense output tensor".into()))?;

        // ---- sparse ----
        let sparse = extract_sparse(&outputs, &token_ids, seq);

        // ---- colbert ----
        let colbert = extract_colbert(&outputs, seq, self.max_colbert_tokens)
            .ok_or_else(|| Error::Inference("no token-level output tensor".into()))?;

        Ok(EmbeddingTriple {
            dense,
            sparse,
            colbert,
        })
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// Resample a hidden-size-H vector to COLBERT_DIM dimensions
/// (deterministic stride sampling + renormalize).
fn resample_row(row: &[f32]) -> Vec<f32> {
    if row.len() == COLBERT_DIM {
        let mut v = row.to_vec();
        l2_normalize(&mut v);
        return v;
    }
    let mut out = Vec::with_capacity(COLBERT_DIM);
    for i in 0..COLBERT_DIM {
        out.push(row[i * row.len() / COLBERT_DIM]);
    }
    l2_normalize(&mut out);
    out
}

fn extract_dense(outputs: &ort::session::SessionOutputs, seq: usize) -> Option<Vec<f32>> {
    // preferred: dedicated pooled output
    if let Ok(t) = TensorRef::try_from(&outputs["dense_vecs"]) {
        if let Ok((_, data)) = t.try_extract_tensor::<f32>() {
            let mut v: Vec<f32> = data.iter().take(DENSE_DIM).copied().collect();
            if v.len() == DENSE_DIM {
                l2_normalize(&mut v);
                return Some(v);
            }
        }
    }
    // fallback: CLS row of last_hidden_state
    if let Ok(t) = TensorRef::try_from(&outputs["last_hidden_state"]) {
        if let Ok((_, data)) = t.try_extract_tensor::<f32>() {
            let hidden = data.len() / seq;
            let mut v: Vec<f32> = data.iter().take(hidden).copied().collect();
            l2_normalize(&mut v);
            if v.len() != DENSE_DIM {
                v.resize(DENSE_DIM, 0.0);
                l2_normalize(&mut v);
            }
            return Some(v);
        }
    }
    None
}

fn extract_sparse(
    outputs: &ort::session::SessionOutputs,
    token_ids: &[u32],
    seq: usize,
) -> SparseVector {
    let mut sparse = SparseVector::new();
    if let Ok(t) = TensorRef::try_from(&outputs["sparse_vecs"]) {
        if let Ok((_, data)) = t.try_extract_tensor::<f32>() {
            for (i, w) in data.iter().take(seq).enumerate() {
                if *w > 0.0 {
                    let e = sparse.entry(token_ids[i]).or_insert(0.0f32);
                    *e = e.max(*w);
                }
            }
        }
    }
    // normalize into (0,1]
    let max = sparse.values().copied().fold(0.0f32, f32::max);
    if max > 0.0 {
        sparse.values_mut().for_each(|w| *w /= max);
    }
    sparse
}

fn extract_colbert(
    outputs: &ort::session::SessionOutputs,
    seq: usize,
    max_tokens: usize,
) -> Option<Vec<Vec<f32>>> {
    for name in ["colbert_vecs", "last_hidden_state"] {
        if let Ok(t) = TensorRef::try_from(&outputs[name]) {
            if let Ok((_, data)) = t.try_extract_tensor::<f32>() {
                let hidden = data.len() / seq;
                return Some(
                    (0..seq.min(max_tokens))
                        .map(|i| resample_row(&data[i * hidden..(i + 1) * hidden]))
                        .collect(),
                );
            }
        }
    }
    None
}

#[async_trait]
impl Embedder for OnnxBgeM3 {
    fn name(&self) -> &'static str {
        "bge-m3-onnx"
    }

    async fn embed(&self, text: &str) -> Result<EmbeddingTriple> {
        // CPU-bound forward pass on the blocking pool.
        let text = text.to_owned();
        let this = unsafe { &*(self as *const Self) };
        tokio::task::spawn_blocking(move || this.forward(&text))
            .await
            .map_err(|e| Error::Inference(format!("blocking join: {e}")))?
    }
}
