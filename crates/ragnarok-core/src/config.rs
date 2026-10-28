use crate::error::{Error, Result};
use crate::model::Source;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Root daemon configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
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
#[serde(default)]
pub struct ServerConfig {
    /// Bind address for the HTTP/WebSocket daemon.
    pub bind: String,
    /// Telemetry push interval in milliseconds.
    pub telemetry_interval_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
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
#[serde(default)]
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceConfig {
    /// Which backend produces embeddings.
    pub backend: InferenceBackend,
    /// Directory containing the ONNX model + tokenizer (backend = onnx).
    pub model_dir: Option<PathBuf>,
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
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        InferenceConfig {
            backend: InferenceBackend::Hashing,
            model_dir: None,
            workers: 2,
            batch_size: 8,
            max_colbert_tokens: 64,
        }
    }
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
        let text = std::fs::read_to_string(path).map_err(|e| {
            Error::Config(format!("cannot read {}: {e}", path.display()))
        })?;
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
            return Err(Error::Config("soft_fraction must be < hard_fraction".into()));
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
    fn derived_paths() {
        let cfg = RagnarokConfig::default();
        assert!(cfg.db_path().ends_with("ragnarok.db"));
        assert!(cfg.vector_dir().ends_with("vectorstore"));
        assert!(cfg.data_dir.ends_with("ragnarok"));
    }
}
