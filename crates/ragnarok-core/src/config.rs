use crate::error::{Error, Result};
use crate::model::Source;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Root daemon configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RagnarokConfig {
    /// Base directory for all persisted state (SQLite, vector index, snapshots).
    pub data_dir: PathBuf,
    pub server: ServerConfig,
    pub resources: ResourceConfig,
    pub crawler: CrawlerConfig,
    pub inference: InferenceConfig,
    #[serde(default)]
    pub sources: Vec<Source>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    /// Bind address for the HTTP/WebSocket daemon.
    pub bind: String,
    /// Telemetry push interval in milliseconds.
    pub telemetry_interval_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResourceConfig {
    /// Fraction of total physical RAM strictly reserved for OS/user [0,1].
    pub reserve_fraction: f32,
    /// Upper bound on daemon usage as a fraction of total RAM [0,1].
    pub cap_fraction: f32,
    /// Absolute headroom: budget never exceeds `available - headroom`.
    pub headroom_bytes: u64,
    /// Usage fraction of budget where soft throttling engages.
    pub soft_fraction: f32,
    /// Usage fraction of budget where ingestion pauses entirely.
    pub hard_fraction: f32,
    /// Telemetry sampling interval.
    pub monitor_interval_ms: u64,
    /// Base number of concurrent ingestion permits.
    pub base_permits: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CrawlerConfig {
    /// Filesystem event debounce window.
    pub debounce_ms: u64,
    /// Target chunk size bounds in approximate tokens.
    pub chunk_min_tokens: usize,
    pub chunk_max_tokens: usize,
    /// Fractional overlap between consecutive chunks [0,1).
    pub chunk_overlap: f32,
    /// File size cap; larger files are skipped.
    pub max_file_bytes: u64,
    /// Tesseract OCR for scans and images.
    pub ocr: OcrConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OcrConfig {
    /// Master switch — run tesseract for `image/*` docs and PDFs whose
    /// text layer is thinner than `pdf_text_threshold`.
    pub enabled: bool,
    /// tesseract binary (PATH lookup by default).
    pub binary: String,
    /// pdftoppm binary for PDF page rasterization.
    pub pdftoppm: String,
    /// Language packs, e.g. `eng` or `deu+eng`.
    pub lang: String,
    /// Render DPI for pdftoppm rasterization.
    pub dpi: u32,
    /// Per-page / per-image OCR timeout.
    pub timeout_secs: u64,
    /// PDFs whose extractable text is shorter than this are treated as
    /// scans and OCRed page by page.
    pub pdf_text_threshold: usize,
    /// Max pages to OCR in a single scanned PDF.
    pub max_pages: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InferenceConfig {
    /// Which backend produces embeddings.
    pub backend: InferenceBackend,
    /// Directory containing the ONNX model + tokenizer (backend = onnx).
    /// Defaults to `data_dir/models/bge-m3`.
    pub model_dir: Option<PathBuf>,
    /// Base URL model files are provisioned from — the file layout of the
    /// `BAAI/bge-m3` `onnx/` export is expected (`{base}/onnx/model.onnx`
    /// etc.). Empty = pinned default (HuggingFace).
    pub model_url: Option<String>,
    /// Optional SHA-256 pin for `model.onnx` when mirroring from a custom
    /// `model_url`. The built-in default base pins every file already.
    pub model_sha256: Option<String>,
    /// Download missing model files automatically at startup (backend =
    /// onnx) and via `POST /inference/model/download`.
    #[serde(default = "default_true")]
    pub auto_provision: bool,
    /// Cross-encoder rerank stage after fusion (top `rerank_top`
    /// candidates per query). Needs the `onnx` feature and the
    /// `Xenova/bge-reranker-base` int8 model — provisioned under
    /// `reranker_dir` (default `data_dir/models/bge-reranker`).
    #[serde(default = "default_true")]
    pub rerank: bool,
    /// Reranker model directory override.
    pub reranker_dir: Option<PathBuf>,
    /// Mirror base URL for the reranker manifest.
    pub reranker_url: Option<String>,
    /// Optional SHA-256 pin for the reranker `model.onnx`.
    pub reranker_sha256: Option<String>,
    /// Number of concurrent inference workers.
    pub workers: usize,
    /// Maximum texts per internal batch.
    pub batch_size: usize,
    /// Cap on ColBERT token vectors per text.
    pub max_colbert_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceBackend {
    /// Deterministic local embedder (no model download; development/testing).
    Hashing,
    /// BGE-M3 via ONNX Runtime (requires model files in `model_dir`).
    Onnx,
}

impl Default for RagnarokConfig {
    fn default() -> Self {
        RagnarokConfig {
            data_dir: default_data_dir(),
            server: ServerConfig::default(),
            resources: ResourceConfig::default(),
            crawler: CrawlerConfig::default(),
            inference: InferenceConfig::default(),
            sources: Vec::new(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            bind: "127.0.0.1:7878".into(),
            telemetry_interval_ms: 250,
        }
    }
}

impl Default for ResourceConfig {
    fn default() -> Self {
        ResourceConfig {
            reserve_fraction: 0.30,
            cap_fraction: 0.70,
            headroom_bytes: 2 * 1024 * 1024 * 1024, // 2 GiB
            soft_fraction: 0.75,
            hard_fraction: 0.90,
            monitor_interval_ms: 1000,
            base_permits: 8,
        }
    }
}

impl Default for CrawlerConfig {
    fn default() -> Self {
        CrawlerConfig {
            debounce_ms: 500,
            chunk_min_tokens: 256,
            chunk_max_tokens: 512,
            chunk_overlap: 0.15,
            max_file_bytes: 512 * 1024 * 1024,
            ocr: OcrConfig::default(),
        }
    }
}

impl Default for OcrConfig {
    fn default() -> Self {
        OcrConfig {
            enabled: true,
            binary: "tesseract".into(),
            pdftoppm: "pdftoppm".into(),
            lang: "eng".into(),
            dpi: 300,
            timeout_secs: 60,
            pdf_text_threshold: 32,
            max_pages: 50,
        }
    }
}

/// Subset of crawler/resource settings that can be changed at runtime
/// via `PUT /api/v1/settings/runtime`; persisted as one JSON blob in
/// `runtime_settings`. Changes apply to *new* ingests — already indexed
/// chunks keep their original shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RuntimeConfig {
    pub chunk_min_tokens: usize,
    pub chunk_max_tokens: usize,
    pub chunk_overlap: f32,
    pub max_file_bytes: u64,
    /// tesseract on/off — flips the image parser + PDF scan fallback.
    pub ocr_enabled: bool,
    /// tesseract language packs.
    pub ocr_lang: String,
    /// Resource pressure thresholds (0..1, soft < hard).
    pub soft_fraction: f32,
    pub hard_fraction: f32,
    /// Base ingestion permits.
    pub base_permits: usize,
}

impl RuntimeConfig {
    /// Derive the runtime surface from the boot configuration.
    pub fn from_config(c: &CrawlerConfig, r: &ResourceConfig) -> Self {
        RuntimeConfig {
            chunk_min_tokens: c.chunk_min_tokens,
            chunk_max_tokens: c.chunk_max_tokens,
            chunk_overlap: c.chunk_overlap,
            max_file_bytes: c.max_file_bytes,
            ocr_enabled: c.ocr.enabled,
            ocr_lang: c.ocr.lang.clone(),
            soft_fraction: r.soft_fraction,
            hard_fraction: r.hard_fraction,
            base_permits: r.base_permits,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.chunk_min_tokens == 0
            || self.chunk_min_tokens > self.chunk_max_tokens
            || self.chunk_max_tokens > 8192
        {
            return Err(Error::InvalidInput(
                "chunk token bounds must satisfy 0 < min <= max <= 8192".into(),
            ));
        }
        if !(0.0..1.0).contains(&self.chunk_overlap) {
            return Err(Error::InvalidInput("chunk_overlap must be in [0,1)".into()));
        }
        if self.max_file_bytes == 0 {
            return Err(Error::InvalidInput("max_file_bytes must be > 0".into()));
        }
        if self.ocr_lang.trim().is_empty() {
            return Err(Error::InvalidInput("ocr_lang must not be empty".into()));
        }
        if !(0.05..=0.98).contains(&self.soft_fraction)
            || !(0.05..=0.99).contains(&self.hard_fraction)
            || self.soft_fraction >= self.hard_fraction
        {
            return Err(Error::InvalidInput(
                "pressure fractions must satisfy 0.05 <= soft < hard <= 0.99".into(),
            ));
        }
        if self.base_permits == 0 || self.base_permits > 256 {
            return Err(Error::InvalidInput(
                "base_permits must be in 1..=256".into(),
            ));
        }
        Ok(())
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self::from_config(&CrawlerConfig::default(), &ResourceConfig::default())
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        InferenceConfig {
            backend: InferenceBackend::Hashing,
            model_dir: None,
            model_url: None,
            model_sha256: None,
            auto_provision: true,
            rerank: true,
            reranker_dir: None,
            reranker_url: None,
            reranker_sha256: None,
            workers: 2,
            batch_size: 8,
            max_colbert_tokens: 64,
        }
    }
}

fn default_true() -> bool {
    true
}

/// `~/.local/share/ragnarok` on Linux, platform equivalent elsewhere.
pub fn default_data_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ragnarok")
}

/// `~/.config/ragnarok/config.toml`.
pub fn default_config_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ragnarok")
        .join("config.toml")
}

impl RagnarokConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Config(format!("cannot read {}: {e}", path.display())))?;
        toml::from_str(&text)
            .map_err(|e| Error::Config(format!("invalid config {}: {e}", path.display())))
    }

    /// Load config if it exists, otherwise return defaults.
    pub fn load_or_default(path: &Path) -> Result<Self> {
        if path.exists() {
            Self::load(path)
        } else {
            Ok(Self::default())
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = toml::to_string_pretty(self)
            .map_err(|e| Error::Config(format!("cannot serialize config: {e}")))?;
        std::fs::write(path, text)?;
        Ok(())
    }

    pub fn db_path(&self) -> PathBuf {
        self.data_dir.join("ragnarok.db")
    }

    pub fn vector_dir(&self) -> PathBuf {
        self.data_dir.join("vectorstore")
    }

    /// Validate numeric invariants; called once at startup.
    pub fn validate(&self) -> Result<()> {
        let r = &self.resources;
        if !(0.0..1.0).contains(&r.reserve_fraction) {
            return Err(Error::Config("reserve_fraction must be in [0,1)".into()));
        }
        if !(0.0..=1.0).contains(&r.cap_fraction) {
            return Err(Error::Config("cap_fraction must be in [0,1]".into()));
        }
        if r.soft_fraction >= r.hard_fraction {
            return Err(Error::Config(
                "soft_fraction must be < hard_fraction".into(),
            ));
        }
        let c = &self.crawler;
        if c.chunk_min_tokens == 0 || c.chunk_min_tokens >= c.chunk_max_tokens {
            return Err(Error::Config(
                "chunk_min_tokens must be > 0 and < chunk_max_tokens".into(),
            ));
        }
        if !(0.0..1.0).contains(&c.chunk_overlap) {
            return Err(Error::Config("chunk_overlap must be in [0,1)".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        RagnarokConfig::default().validate().unwrap();
    }

    #[test]
    fn config_toml_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut cfg = RagnarokConfig::default();
        cfg.server.bind = "0.0.0.0:9999".into();
        cfg.crawler.chunk_max_tokens = 300;
        cfg.save(&path).unwrap();
        let loaded = RagnarokConfig::load(&path).unwrap();
        assert_eq!(loaded.server.bind, "0.0.0.0:9999");
        assert_eq!(loaded.crawler.chunk_max_tokens, 300);
        assert_eq!(loaded.resources.reserve_fraction, 0.30);
        assert_eq!(loaded.resources.headroom_bytes, 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn partial_toml_uses_defaults() {
        let cfg: RagnarokConfig = toml::from_str(
            r#"
            [server]
            bind = "127.0.0.1:1234"
            "#,
        )
        .unwrap();
        assert_eq!(cfg.server.bind, "127.0.0.1:1234");
        assert_eq!(cfg.crawler.debounce_ms, 500);
        assert!(cfg.sources.is_empty());
    }

    #[test]
    fn ocr_config_defaults_and_overrides() {
        let cfg: RagnarokConfig = toml::from_str("").unwrap();
        assert!(cfg.crawler.ocr.enabled);
        assert_eq!(cfg.crawler.ocr.binary, "tesseract");
        assert_eq!(cfg.crawler.ocr.pdf_text_threshold, 32);
        let cfg: RagnarokConfig = toml::from_str(
            r#"
            [crawler.ocr]
            enabled = false
            lang = "deu+eng"
            dpi = 200
            "#,
        )
        .unwrap();
        assert!(!cfg.crawler.ocr.enabled);
        assert_eq!(cfg.crawler.ocr.lang, "deu+eng");
        assert_eq!(cfg.crawler.ocr.dpi, 200);
        // unknown keys inside [crawler.ocr] are rejected too
        assert!(toml::from_str::<RagnarokConfig>("[crawler.ocr]\nfrobnicate=1\n").is_err());
    }

    #[test]
    fn validate_rejects_bad_values() {
        let mut cfg = RagnarokConfig::default();
        cfg.resources.soft_fraction = 0.95;
        cfg.resources.hard_fraction = 0.90;
        assert!(cfg.validate().is_err());

        let mut cfg = RagnarokConfig::default();
        cfg.crawler.chunk_min_tokens = 512;
        cfg.crawler.chunk_max_tokens = 256;
        assert!(cfg.validate().is_err());

        let mut cfg = RagnarokConfig::default();
        cfg.crawler.chunk_overlap = 1.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn unknown_keys_are_rejected() {
        // misplaced data_dir under [storage] must not be silently ignored
        let err = toml::from_str::<RagnarokConfig>(
            r#"
            [storage]
            data_dir = "/tmp/x"
            "#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown field"), "{err}");

        let err = toml::from_str::<RagnarokConfig>(
            r#"
            [resources]
            mem_soft_ratio = 0.5
            "#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("mem_soft_ratio"), "{err}");
    }

    #[test]
    fn inference_provisioning_defaults() {
        // missing keys: auto_provision stays ON, urls/hashes stay None
        let cfg: RagnarokConfig = toml::from_str(
            r#"
            [inference]
            backend = "onnx"
            "#,
        )
        .unwrap();
        assert_eq!(cfg.inference.backend, InferenceBackend::Onnx);
        assert!(cfg.inference.auto_provision);
        assert!(cfg.inference.model_url.is_none());
        assert!(cfg.inference.model_sha256.is_none());

        // explicit override parses
        let cfg: RagnarokConfig = toml::from_str(
            r#"
            [inference]
            backend = "onnx"
            auto_provision = false
            model_url = "https://mirror.local/bge-m3"
            model_sha256 = "abc123"
            "#,
        )
        .unwrap();
        assert!(!cfg.inference.auto_provision);
        assert_eq!(
            cfg.inference.model_url.as_deref(),
            Some("https://mirror.local/bge-m3")
        );
    }

    #[test]
    fn derived_paths() {
        let cfg = RagnarokConfig::default();
        assert!(cfg.db_path().ends_with("ragnarok.db"));
        assert!(cfg.vector_dir().ends_with("vectorstore"));
        assert!(cfg.data_dir.ends_with("ragnarok"));
    }
}
