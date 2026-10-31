//! Registry of ONNX embedding models RAGnarök can provision and run.
//!
//! Every entry describes one self-contained export: where the files live
//! (HuggingFace repo base + manifest), what the graph expects/produces
//! (pooling style, dense dimension, max sequence) and which text prefixes
//! the model was trained with — E5/Nomic-style models silently degrade
//! when their `query:`/`passage:` prefixes are missing, so they are part
//! of the spec, applied by [`crate::onnx::OnnxEmbedder`] at embed time.
//!
//! All non-BAAI entries use the `Xenova/*` mirrors, which ship the same
//! layout: `onnx/model_quantized.onnx` (int8, single-file, CPU-friendly)
//! plus `tokenizer.json` at the repo root. Manifest sizes/hashes are
//! unpinned for these (`size: 0` = stream-and-trust) — the BAAI bge-m3
//! fp32 manifest stays fully pinned.

use crate::provision::ModelFile;

/// How the pooled dense vector is derived from token-level outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// Row 0 of the token outputs (BGE family).
    Cls,
    /// Attention-mask-weighted mean over token outputs (E5/MiniLM).
    Mean,
}

/// A downloadable + runnable embedding model.
pub struct ModelSpec {
    /// Config-facing id — `inference.model = "…"`, also the default
    /// subdirectory under `data_dir/models/`.
    pub id: &'static str,
    /// HuggingFace repo base (`{base}/{remote}` per manifest file).
    pub repo: &'static str,
    /// Short capability note for the UI model picker.
    pub description: &'static str,
    /// Language coverage — `multilingual` or `en`.
    pub languages: &'static str,
    /// Dense vector dimensionality produced by the graph.
    pub dense_dim: usize,
    /// Position-embedding budget — the tokenizer truncates here and the
    /// runtime sequence cap starts here.
    pub max_seq: usize,
    pub pooling: Pooling,
    /// Prepended to queries before tokenizing (`""` = none).
    pub query_prefix: &'static str,
    /// Prepended to documents before tokenizing (`""` = none).
    pub doc_prefix: &'static str,
    /// Files to provision into `data_dir/models/{id}/`.
    pub files: &'static [ModelFile],
}

const fn mf(remote: &'static str, dest: &'static str) -> ModelFile {
    ModelFile {
        remote,
        dest,
        size: 0,
        sha256: None,
        git_sha1: None,
        required: true,
    }
}

/// Standard Xenova-mirror manifest: int8 single-file graph + tokenizer.
const XENOVA_INT8: &[ModelFile] = &[
    mf("onnx/model_quantized.onnx", "model_quantized.onnx"),
    mf("tokenizer.json", "tokenizer.json"),
];

/// BGE-english query instruction (documents go in bare).
const BGE_EN_QUERY: &str = "Represent this sentence for searching relevant passages: ";

static SPECS: &[ModelSpec] = &[
    ModelSpec {
        id: "bge-m3",
        repo: "https://huggingface.co/BAAI/bge-m3/resolve/main",
        description: "BGE-M3 fp32 — highest quality, 568M params (2.3 GB download; needs a real GPU or lots of patience)",
        languages: "multilingual",
        dense_dim: 1024,
        max_seq: 8192,
        pooling: Pooling::Cls,
        query_prefix: "",
        doc_prefix: "",
        files: crate::provision::MODEL_FILES,
    },
    ModelSpec {
        id: "bge-m3-int8",
        repo: "https://huggingface.co/Xenova/bge-m3/resolve/main",
        description: "BGE-M3 int8 — same model quantized, ~4× smaller/faster (570 MB)",
        languages: "multilingual",
        dense_dim: 1024,
        max_seq: 8192,
        pooling: Pooling::Cls,
        query_prefix: "",
        doc_prefix: "",
        files: XENOVA_INT8,
    },
    ModelSpec {
        id: "multilingual-e5-small",
        repo: "https://huggingface.co/Xenova/multilingual-e5-small/resolve/main",
        description: "multilingual-e5-small int8 — fast multilingual embedder (118 MB), good for mail-scale corpora",
        languages: "multilingual",
        dense_dim: 384,
        max_seq: 512,
        pooling: Pooling::Mean,
        query_prefix: "query: ",
        doc_prefix: "passage: ",
        files: XENOVA_INT8,
    },
    ModelSpec {
        id: "multilingual-e5-base",
        repo: "https://huggingface.co/Xenova/multilingual-e5-base/resolve/main",
        description: "multilingual-e5-base int8 — stronger multilingual quality (280 MB)",
        languages: "multilingual",
        dense_dim: 768,
        max_seq: 512,
        pooling: Pooling::Mean,
        query_prefix: "query: ",
        doc_prefix: "passage: ",
        files: XENOVA_INT8,
    },
    ModelSpec {
        id: "multilingual-e5-large",
        repo: "https://huggingface.co/Xenova/multilingual-e5-large/resolve/main",
        description: "multilingual-e5-large int8 — near-BGE-M3 multilingual quality (560 MB)",
        languages: "multilingual",
        dense_dim: 1024,
        max_seq: 512,
        pooling: Pooling::Mean,
        query_prefix: "query: ",
        doc_prefix: "passage: ",
        files: XENOVA_INT8,
    },
    ModelSpec {
        id: "paraphrase-multilingual-minilm",
        repo: "https://huggingface.co/Xenova/paraphrase-multilingual-MiniLM-L12-v2/resolve/main",
        description: "paraphrase-multilingual-MiniLM int8 — compact 12-layer multilingual model (118 MB)",
        languages: "multilingual",
        dense_dim: 384,
        max_seq: 512,
        pooling: Pooling::Mean,
        query_prefix: "",
        doc_prefix: "",
        files: XENOVA_INT8,
    },
    ModelSpec {
        id: "bge-small-en",
        repo: "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main",
        description: "bge-small-en-v1.5 int8 — tiny English-only embedder (34 MB), very fast",
        languages: "en",
        dense_dim: 384,
        max_seq: 512,
        pooling: Pooling::Cls,
        query_prefix: BGE_EN_QUERY,
        doc_prefix: "",
        files: XENOVA_INT8,
    },
    ModelSpec {
        id: "bge-base-en",
        repo: "https://huggingface.co/Xenova/bge-base-en-v1.5/resolve/main",
        description: "bge-base-en-v1.5 int8 — solid English-only quality (109 MB)",
        languages: "en",
        dense_dim: 768,
        max_seq: 512,
        pooling: Pooling::Cls,
        query_prefix: BGE_EN_QUERY,
        doc_prefix: "",
        files: XENOVA_INT8,
    },
    ModelSpec {
        id: "all-minilm-l6",
        repo: "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main",
        description: "all-MiniLM-L6-v2 int8 — smallest/fastest option, English-only (23 MB)",
        languages: "en",
        dense_dim: 384,
        max_seq: 512,
        pooling: Pooling::Mean,
        query_prefix: "",
        doc_prefix: "",
        files: XENOVA_INT8,
    },
];

/// Look up a model by its config id.
pub fn model_spec(id: &str) -> Option<&'static ModelSpec> {
    SPECS.iter().find(|s| s.id == id)
}

/// All registered models — powers the UI picker and error messages.
pub fn model_registry() -> &'static [ModelSpec] {
    SPECS
}

/// Default model when `inference.model` is unset.
pub const DEFAULT_MODEL_ID: &str = "bge-m3";
