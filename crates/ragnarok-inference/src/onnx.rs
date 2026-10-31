//! Real embedding inference via ONNX Runtime (`ort`) — `--features onnx`.
//!
//! Driven by a [`ModelSpec`] from the model registry, so any single-file
//! or external-data encoder export works: BGE-M3 (CLS pooling, no
//! prefixes), multilingual-e5 (mean pooling + `query:`/`passage:`),
//! MiniLM/BGE-en variants, …
//!
//! Input tensors are bound by *declared* input name — `input_ids` and
//! `attention_mask` get real data, anything else the graph asks for
//! (e.g. `token_type_ids`) gets a zero tensor of the same [B,S] shape.
//! Output tensor names are resolved dynamically: pre-pooled
//! `dense_vecs`/`sentence_embedding` [B,H] are used directly, otherwise
//! the token-level `last_hidden_state`/`token_embeddings` [B,S,H] is
//! pooled per the spec. `sparse` stays empty for exports without a
//! sparse head. ColBERT vectors are resampled to 128 dims when the
//! model's hidden size differs.

use crate::embedder::Embedder;
use crate::models::{ModelSpec, Pooling};
use crate::COLBERT_DIM;
use async_trait::async_trait;
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;
use ragnarok_core::{EmbeddingTriple, Error, Result, SparseVector};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::{Tokenizer, TruncationParams};

/// Floor for the adaptive sequence cap — below this the doc fails hard.
const MIN_SEQ_LEN: usize = 256;

/// GPU session health. A WebGPU/Vulkan device loss is sticky at the
/// driver level — after one same-provider rebuild (covers a transient
/// OOM on a real GPU) we give up and continue on a plain CPU session
/// rather than let every following embed fail.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GpuState {
    Healthy,
    RetriedOnce,
    Dead,
}

pub struct OnnxEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    spec: &'static ModelSpec,
    max_colbert_tokens: usize,
    onnx_path: PathBuf,
    /// Configured provider string (may be `auto`).
    provider: String,
    /// Pool worker count — rebuilds must split CPU threads the same way.
    workers: usize,
    /// Input names the graph declares, in order.
    input_names: Vec<String>,
    gpu_state: Mutex<GpuState>,
    /// Why the GPU was abandoned — surfaced via [`Embedder::gpu_dead`]
    /// so the UI/CLI can show "running on CPU because …".
    dead_reason: Mutex<Option<String>>,
    /// EP of the live session — `WebGpuExecutionProvider`, `cpu`, …
    active_provider: Mutex<String>,
    /// Effective sequence cap — starts at `spec.max_seq` and is halved
    /// after OOM-style failures so inputs shrink to whatever the
    /// hardware tolerates.
    seq_cap: Mutex<usize>,
}

/// Backwards-compatible alias kept for downstream references.
pub type OnnxBgeM3 = OnnxEmbedder;

impl OnnxEmbedder {
    /// Load model + tokenizer from `model_dir` per `spec`. `provider`
    /// selects the ORT execution provider (`auto`/`cpu`/`vulkan`/…);
    /// `workers` is the pool worker count so intra-op threads are split
    /// instead of oversubscribed (workers × all-cores on one session
    /// mutex just thrash).
    pub fn load(
        model_dir: &Path,
        spec: &'static ModelSpec,
        max_colbert_tokens: usize,
        provider: &str,
        workers: usize,
    ) -> Result<Arc<Self>> {
        // quantized variants first when present — int8/uint8 exports are
        // ~4x smaller and faster on CPU and can even fit an iGPU the fp32
        // model OOMs on; the fp32 file stays the fallback
        let onnx_path = [
            "model_quantized.onnx",
            "model_int8.onnx",
            "model_uint8.onnx",
            "model_optimized.onnx",
            "model.onnx",
            "bge-m3.onnx",
        ]
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
        let (session, ep_label) = build_session(&onnx_path, provider, workers)?;
        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        let out_names: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
        tracing::info!(
            model = %onnx_path.display(),
            provider = %ep_label,
            inputs = %input_names.join(","),
            outputs = %out_names.join(","),
            "onnx embedder session ready"
        );
        let mut tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| Error::Inference(format!("tokenizer load: {e}")))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: spec.max_seq,
                ..Default::default()
            }))
            .map_err(|e| Error::Inference(format!("tokenizer truncation: {e}")))?;
        Ok(Arc::new(OnnxEmbedder {
            session: Mutex::new(session),
            tokenizer,
            spec,
            max_colbert_tokens: max_colbert_tokens.max(1),
            onnx_path,
            provider: provider.to_string(),
            workers: workers.max(1),
            input_names,
            gpu_state: Mutex::new(GpuState::Healthy),
            dead_reason: Mutex::new(None),
            active_provider: Mutex::new(ep_label),
            seq_cap: Mutex::new(spec.max_seq),
        }))
    }

    /// Reason the GPU was abandoned, if it was — for /inference + UI.
    pub fn gpu_dead(&self) -> Option<String> {
        self.dead_reason.lock().clone()
    }

    /// Record a GPU death discovered elsewhere — the persisted marker
    /// from a previous boot. Keeps /inference honest about *why* this
    /// instance runs on CPU, and shortcuts forward() into the CPU path.
    pub fn note_gpu_dead(&self, reason: String) {
        *self.dead_reason.lock() = Some(reason);
        *self.gpu_state.lock() = GpuState::Dead;
    }

    /// EP of the live session — for /inference + UI.
    pub fn provider_label(&self) -> String {
        self.active_provider.lock().clone()
    }

    /// One batched forward over `items` (id/mask pairs), each truncated
    /// to `cap` tokens, padded to the batch max. All declared inputs are
    /// fed; unknown names get a zero [B,S] i64 tensor.
    fn run_batch(
        &self,
        items: &[(Vec<i64>, Vec<i64>)],
        cap: usize,
    ) -> Result<Vec<EmbeddingTriple>> {
        let b = items.len();
        let lens: Vec<usize> = items.iter().map(|(ids, _)| ids.len().min(cap)).collect();
        let s = lens.iter().copied().max().unwrap_or(0);
        if s == 0 {
            return Err(Error::Inference("empty token sequence".into()));
        }
        let mut ids = vec![0i64; b * s];
        let mut mask = vec![0i64; b * s];
        for (i, ((ti, tm), n)) in items.iter().zip(lens.iter()).enumerate() {
            ids[i * s..i * s + n].copy_from_slice(&ti[..*n]);
            mask[i * s..i * s + n].copy_from_slice(&tm[..*n]);
        }
        let shape = vec![b, s];
        let mut inputs: Vec<(String, Tensor<i64>)> = Vec::with_capacity(self.input_names.len());
        for name in &self.input_names {
            let data = match name.as_str() {
                "input_ids" => ids.clone(),
                "attention_mask" => mask.clone(),
                // token_type_ids and anything else declared: zero tensor —
                // single-segment inputs are all segment 0 anyway
                _ => vec![0i64; b * s],
            };
            let t = Tensor::from_array((shape.clone(), data))
                .map_err(|e| Error::Inference(e.to_string()))?;
            inputs.push((name.clone(), t));
        }

        let mut session = self.session.lock();
        let outputs = session
            .run(inputs)
            .map_err(|e| Error::Inference(format!("ort run: {e}")))?;

        (0..b)
            .map(|i| {
                let n = lens[i];
                let ids_row = &ids[i * s..i * s + n];
                let mask_row = &mask[i * s..i * s + s];
                let dense = extract_dense(&outputs, i, b, s, mask_row, self.spec)
                    .ok_or_else(|| Error::Inference("no dense output tensor".into()))?;
                let sparse = extract_sparse(&outputs, i, b, s, ids_row);
                let colbert = extract_colbert(&outputs, i, b, s, n, self.max_colbert_tokens)
                    .ok_or_else(|| Error::Inference("no token-level output tensor".into()))?;
                Ok(EmbeddingTriple {
                    dense,
                    sparse,
                    colbert,
                })
            })
            .collect()
    }

    /// Recreate the ORT session on `provider` — GPU device-loss leaves
    /// the old one permanently broken.
    fn rebuild_session(&self, provider: &str) -> Result<()> {
        let (session, ep_label) = build_session(&self.onnx_path, provider, self.workers)?;
        *self.session.lock() = session;
        *self.active_provider.lock() = ep_label;
        Ok(())
    }

    /// Blocking batched forward — called inside `spawn_blocking`.
    /// `query` selects the spec's query vs document prefix.
    fn forward_batch(&self, texts: &[String], query: bool) -> Result<Vec<EmbeddingTriple>> {
        let prefix = if query {
            self.spec.query_prefix
        } else {
            self.spec.doc_prefix
        };
        let mut items = Vec::with_capacity(texts.len());
        for t in texts {
            let enc = self
                .tokenizer
                .encode(format!("{prefix}{t}"), true)
                .map_err(|e| Error::Inference(format!("tokenize: {e}")))?;
            let ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
            let mask: Vec<i64> = enc.get_attention_mask().iter().map(|&x| x as i64).collect();
            if ids.is_empty() {
                return Err(Error::Inference("empty token sequence".into()));
            }
            items.push((ids, mask));
        }

        let mut cap = *self.seq_cap.lock();
        loop {
            match self.run_batch(&items, cap) {
                Ok(t) => return Ok(t),
                Err(e) if is_device_loss(&e) => {
                    let on_cpu = *self.gpu_state.lock() == GpuState::Dead
                        || self.provider.eq_ignore_ascii_case("cpu");
                    if on_cpu {
                        // CPU can't lose a device — an OOM here just means
                        // the input is too big for RAM: shrink and retry.
                        let new_cap = cap / 2;
                        if new_cap < MIN_SEQ_LEN {
                            return Err(e);
                        }
                        tracing::warn!(cap = new_cap, error = %e,
                            "onnx input too large for CPU — shrinking seq cap");
                        *self.seq_cap.lock() = new_cap;
                        cap = new_cap;
                        continue;
                    }
                    let (provider, new_cap) = {
                        let mut st = self.gpu_state.lock();
                        match *st {
                            GpuState::Healthy => {
                                // first loss — maybe a transient OOM:
                                // rebuild on the same provider, halve cap
                                *st = GpuState::RetriedOnce;
                                (self.provider.clone(), (cap / 2).max(MIN_SEQ_LEN))
                            }
                            // second loss — the device is dead for good:
                            // CPU session for the rest of the process,
                            // cap keeps its shrunken value
                            _ => {
                                *st = GpuState::Dead;
                                *self.dead_reason.lock() = Some(truncate_reason(&e.to_string()));
                                ("cpu".to_string(), cap)
                            }
                        }
                    };
                    tracing::warn!(
                        provider = %provider,
                        cap = new_cap,
                        error = %e,
                        "onnx device lost — rebuilding session"
                    );
                    if let Err(re) = self.rebuild_session(&provider) {
                        tracing::warn!("onnx session rebuild failed: {re}");
                        return Err(e);
                    }
                    *self.seq_cap.lock() = new_cap;
                    cap = new_cap;
                }
                Err(e) => return Err(e),
            }
        }
    }
}

/// Keep the persisted/UI death reason compact — ORT errors carry long
/// internal paths and source locations.
fn truncate_reason(msg: &str) -> String {
    let msg = msg.replace('\n', " ");
    let end = msg
        .char_indices()
        .nth(240)
        .map(|(i, _)| i)
        .unwrap_or(msg.len());
    msg[..end].to_string()
}

/// Build a session on the shared process env with the resolved provider
/// chain — used for the initial load and for rebuilds after device loss.
/// `workers` splits intra-op threads: the pool already parallelizes
/// across texts, so workers × all-cores just oversubscribes.
/// Returns the session plus the provider label it actually got.
fn build_session(onnx_path: &Path, provider: &str, workers: usize) -> Result<(Session, String)> {
    let eps = crate::ep::execution_providers(provider)?;
    let env = crate::ep::environment()?;
    let ep_names: Vec<&str> = eps.iter().map(|e| e.name()).collect();
    let label = if ep_names.is_empty() {
        "cpu".to_string()
    } else {
        ep_names.join(",")
    };
    let intra = (num_cpus() / workers.max(1)).max(1);
    let session = Session::builder(&env)
        .map_err(|e| Error::Inference(format!("ort init: {e}")))?
        .with_execution_providers(&eps)
        .map_err(|e| Error::Inference(format!("ort providers: {e}")))?
        .with_intra_threads(intra)
        .map_err(|e| Error::Inference(format!("ort threads: {e}")))?
        .commit_from_file(onnx_path)
        .map_err(|e| Error::Inference(format!("ort load {}: {e}", onnx_path.display())))?;
    tracing::info!(providers = %label, intra_threads = intra, "onnx session initialized");
    Ok((session, label))
}

/// GPU failures that leave the session unusable — device loss and driver
/// OOM in all its spellings (`VK_ERROR_OUT_OF_DEVICE_MEMORY`,
/// `out of memory`, ORT's `OOM when allocating`). Shape/type errors
/// must NOT match (they'd rebuild in a loop).
fn is_device_loss(e: &Error) -> bool {
    let msg = e.to_string().to_lowercase();
    msg.contains("lost")
        || msg.contains("out of memory")
        || (msg.contains("out_of") && msg.contains("memory"))
        || msg
            .split(|c: char| !c.is_alphanumeric())
            .any(|w| w == "oom")
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

/// Try to pull a named output tensor: (dims, row-major f32 data).
fn out_f32(outputs: &ort::session::SessionOutputs, name: &str) -> Option<(Vec<i64>, Vec<f32>)> {
    let (shape, data) = outputs.get(name)?.try_extract_tensor::<f32>().ok()?;
    Some((shape.iter().copied().collect(), data.to_vec()))
}

/// Dense vector for batch row `i` of a [B,S] run.
/// Pre-pooled outputs (`dense_vecs`/`sentence_embedding` [B,H]) win;
/// otherwise the token-level [B,S,H] tensor is pooled per the spec
/// (CLS row, or attention-mask-weighted mean).
fn extract_dense(
    outputs: &ort::session::SessionOutputs,
    i: usize,
    b: usize,
    s: usize,
    mask_row: &[i64],
    spec: &ModelSpec,
) -> Option<Vec<f32>> {
    for name in ["dense_vecs", "sentence_embedding"] {
        if let Some((_, data)) = out_f32(outputs, name) {
            let h = data.len() / b;
            if h > 0 && data.len() >= (i + 1) * h {
                let mut v: Vec<f32> = data[i * h..(i + 1) * h].to_vec();
                v.resize(spec.dense_dim, 0.0);
                l2_normalize(&mut v);
                return Some(v);
            }
        }
    }
    for name in ["last_hidden_state", "token_embeddings"] {
        if let Some((_, data)) = out_f32(outputs, name) {
            let h = data.len() / (b * s);
            if h == 0 {
                continue;
            }
            let base = i * s * h;
            let mut v = match spec.pooling {
                Pooling::Cls => data[base..base + h].to_vec(),
                Pooling::Mean => {
                    let mut acc = vec![0f32; h];
                    let mut wsum = 0f32;
                    for (t, &m) in mask_row.iter().enumerate() {
                        if m > 0 {
                            wsum += 1.0;
                            for d in 0..h {
                                acc[d] += data[base + t * h + d];
                            }
                        }
                    }
                    if wsum > 0.0 {
                        acc.iter_mut().for_each(|x| *x /= wsum);
                    }
                    acc
                }
            };
            v.resize(spec.dense_dim, 0.0);
            l2_normalize(&mut v);
            return Some(v);
        }
    }
    None
}

/// Sparse lexical weights for batch row `i` — only exports with a
/// `sparse_vecs` [B,S] head produce any (none of the registered ones do).
fn extract_sparse(
    outputs: &ort::session::SessionOutputs,
    i: usize,
    b: usize,
    s: usize,
    ids_row: &[i64],
) -> SparseVector {
    let mut sparse = SparseVector::new();
    if let Some((_, data)) = out_f32(outputs, "sparse_vecs") {
        let w = data.len() / (b * s);
        if w > 0 {
            for (t, &tok) in ids_row.iter().enumerate() {
                let v = data[i * s * w + t * w];
                if v > 0.0 {
                    let e = sparse.entry(tok.max(0) as u32).or_insert(0.0f32);
                    *e = e.max(v);
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

/// ColBERT token vectors for batch row `i`: the first `n` real tokens of
/// the token-level output, each resampled to 128 dims.
fn extract_colbert(
    outputs: &ort::session::SessionOutputs,
    i: usize,
    b: usize,
    s: usize,
    n: usize,
    max_tokens: usize,
) -> Option<Vec<Vec<f32>>> {
    for name in ["colbert_vecs", "last_hidden_state", "token_embeddings"] {
        if let Some((_, data)) = out_f32(outputs, name) {
            let h = data.len() / (b * s);
            if h == 0 {
                continue;
            }
            let base = i * s * h;
            return Some(
                (0..n.min(max_tokens))
                    .map(|t| resample_row(&data[base + t * h..base + (t + 1) * h]))
                    .collect(),
            );
        }
    }
    None
}

#[async_trait]
impl Embedder for OnnxEmbedder {
    fn name(&self) -> &'static str {
        // the spec id, not a fixed label — docs are stamped with it and
        // the reindexer's only_stale mode compares, so a model switch
        // must read as a different embedder
        self.spec.id
    }

    fn dense_dim(&self) -> usize {
        self.spec.dense_dim
    }

    fn gpu_dead(&self) -> Option<String> {
        self.gpu_dead()
    }

    fn provider_label(&self) -> Option<String> {
        Some(self.provider_label())
    }

    async fn embed(&self, text: &str) -> Result<EmbeddingTriple> {
        self.embed_batch(&[text.to_owned()])
            .await
            .map(|mut v| v.swap_remove(0))
    }

    async fn embed_query(&self, text: &str) -> Result<EmbeddingTriple> {
        let text = text.to_owned();
        let this = unsafe { &*(self as *const Self) };
        tokio::task::spawn_blocking(move || {
            this.forward_batch(std::slice::from_ref(&text), true)
                .map(|mut v| v.swap_remove(0))
        })
        .await
        .map_err(|e| Error::Inference(format!("blocking join: {e}")))?
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingTriple>> {
        let texts = texts.to_vec();
        let this = unsafe { &*(self as *const Self) };
        tokio::task::spawn_blocking(move || this.forward_batch(&texts, false))
            .await
            .map_err(|e| Error::Inference(format!("blocking join: {e}")))?
    }
}
