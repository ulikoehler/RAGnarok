//! ONNX model provisioning — downloads the pinned BGE-M3 export into
//! `data_dir/models/bge-m3/` on first use, verifies every byte (SHA-256 for
//! LFS objects, git blob SHA-1 for small files) and installs files
//! atomically via `.part` + rename. Feature-independent: the downloader
//! works in binaries built without `onnx` so the model can be staged
//! before switching the backend.

use ragnarok_core::{Error, Result, TelemetryEvent};
use sha2::Digest;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tokio::io::AsyncWriteExt;
use tokio::sync::broadcast;

/// Base URL the default manifest resolves against — the official
/// `BAAI/bge-m3` `onnx/` export (external-data layout).
pub const DEFAULT_MODEL_BASE: &str = "https://huggingface.co/BAAI/bge-m3/resolve/main";

/// One file of the model manifest. `remote` is joined onto the base URL;
/// `dest` is the filename inside the model directory.
#[derive(Debug, Clone)]
pub struct ModelFile {
    pub remote: &'static str,
    pub dest: &'static str,
    /// Expected content length — always verified.
    pub size: u64,
    /// Pinned SHA-256 (hex) — verified when present.
    pub sha256: Option<&'static str>,
    /// Pinned git-blob SHA-1 (hex) for small non-LFS files.
    pub git_sha1: Option<&'static str>,
    /// Required files abort the provisioning when they fail; optional
    /// extras (external-data companions a single-file export lacks) are
    /// skipped on 404.
    pub required: bool,
}

/// Files of the official `BAAI/bge-m3` onnx export. Hashes are pinned to
/// the upstream revision fetched at build-prep time — a different mirror
/// layout can be pointed at via `inference.model_url`.
pub const MODEL_FILES: &[ModelFile] = &[
    ModelFile {
        remote: "onnx/model.onnx",
        dest: "model.onnx",
        size: 724_923,
        sha256: Some("f84251230831afb359ab26d9fd37d5936d4d9bb5d1d5410e66442f630f24435b"),
        git_sha1: None,
        required: true,
    },
    ModelFile {
        remote: "onnx/model.onnx_data",
        dest: "model.onnx_data",
        size: 2_266_820_608,
        sha256: Some("1eebfb28493f67bba03ce0ef64bfdc7fc5a3bd9d7493f818bb1d78cd798416b4"),
        git_sha1: None,
        required: false,
    },
    ModelFile {
        remote: "onnx/Constant_7_attr__value",
        dest: "Constant_7_attr__value",
        size: 65_552,
        sha256: None,
        git_sha1: Some("3cdc05e5f550d8bd136f28efcea0ad5e6b4169c8"),
        required: false,
    },
    ModelFile {
        remote: "onnx/tokenizer.json",
        dest: "tokenizer.json",
        size: 17_082_821,
        sha256: Some("6710678b12670bc442b99edc952c4d996ae309a7020c1fa0096dd245c2faf790"),
        git_sha1: None,
        required: true,
    },
];

/// Provisioning state, shared with the server for `/inference` status.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelStatus {
    pub state: ModelState,
    /// Model directory this status refers to.
    pub dir: String,
    /// Bytes downloaded so far in the current run.
    pub bytes_done: u64,
    /// Total bytes to download (sum of remaining file sizes, if known).
    pub bytes_total: u64,
    /// File currently being fetched.
    pub current_file: Option<String>,
    /// Last error message when `state == failed`.
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelState {
    /// Model files absent, no download running.
    Missing,
    Downloading,
    Ready,
    Failed,
}

/// Shared handle — `Provisioner` writes, the server reads.
pub type SharedModelStatus = Arc<RwLock<ModelStatus>>;

/// Downloads and verifies the model manifest into `dir`.
pub struct Provisioner {
    dir: PathBuf,
    base_url: String,
    /// Optional pin overriding the manifest hash for `model.onnx`
    /// (custom `model_url` mirrors).
    model_sha256: Option<String>,
    files: Vec<ModelFile>,
    status: SharedModelStatus,
}

impl Provisioner {
    pub fn new(dir: PathBuf, base_url: Option<String>, model_sha256: Option<String>) -> Arc<Self> {
        Self::with_files(dir, base_url, model_sha256, MODEL_FILES.to_vec())
    }

    /// Provisioner over a custom manifest — the pinned default is
    /// `MODEL_FILES`; tests inject small fake files.
    pub fn with_files(
        dir: PathBuf,
        base_url: Option<String>,
        model_sha256: Option<String>,
        files: Vec<ModelFile>,
    ) -> Arc<Self> {
        let status = Arc::new(RwLock::new(ModelStatus {
            state: ModelState::Missing,
            dir: dir.display().to_string(),
            bytes_done: 0,
            bytes_total: 0,
            current_file: None,
            error: None,
        }));
        let p = Arc::new(Provisioner {
            dir,
            base_url: base_url
                .unwrap_or_else(|| DEFAULT_MODEL_BASE.into())
                .trim_end_matches('/')
                .to_string(),
            model_sha256,
            files,
            status,
        });
        p.refresh_state();
        p
    }

    pub fn status(&self) -> SharedModelStatus {
        self.status.clone()
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Recompute `Missing`/`Ready` from the filesystem (keeps a running
    /// download state untouched).
    pub fn refresh_state(&self) {
        let mut st = self.status.write().unwrap();
        if st.state == ModelState::Downloading {
            return;
        }
        st.state = if Self::ready_in(&self.dir) {
            ModelState::Ready
        } else {
            ModelState::Missing
        };
    }

    /// A model dir is usable when a graph file and the tokenizer exist —
    /// and, for the external-data export (tiny `model.onnx`), the data
    /// companion is present too.
    pub fn ready_in(dir: &Path) -> bool {
        let has_model = ["model_optimized.onnx", "model.onnx", "bge-m3.onnx"]
            .iter()
            .any(|f| dir.join(f).is_file());
        if !has_model || !dir.join("tokenizer.json").is_file() {
            return false;
        }
        let small_graph = dir
            .join("model.onnx")
            .metadata()
            .map(|m| m.len() < 50 * 1024 * 1024)
            .unwrap_or(false);
        if small_graph && !dir.join("model.onnx_data").is_file() {
            return false;
        }
        true
    }

    /// Files still missing from the manifest (skips files already on disk
    /// with the expected size — re-verify only when a hash is pinned).
    fn missing(&self) -> Vec<ModelFile> {
        self.files
            .iter()
            .filter(|f| !self.file_ok(&self.dir.join(f.dest), f))
            .cloned()
            .collect()
    }

    fn file_ok(&self, path: &Path, f: &ModelFile) -> bool {
        let Ok(meta) = path.metadata() else {
            return false;
        };
        // any pinned hash always re-verifies (cheap for small files;
        // skipped for >100MB files already present — a corrupt existing
        // file is rejected later by the ONNX loader anyway)
        if meta.len() != f.size {
            return false;
        }
        if (self.pin_for(f).is_some() || f.git_sha1.is_some()) && meta.len() <= 100 << 20 {
            return verify_file(path, f, self).unwrap_or(false);
        }
        true
    }

    /// Effective sha256 pin for a manifest file — a configured
    /// `model_sha256` overrides the manifest hash of `model.onnx`.
    fn pin_for(&self, f: &ModelFile) -> Option<&str> {
        if f.dest == "model.onnx" {
            self.model_sha256.as_deref().or(f.sha256)
        } else {
            f.sha256
        }
    }

    /// Download all missing manifest files. Emits `ModelDownload`
    /// telemetry as it goes; installs each file atomically.
    pub async fn download(
        &self,
        events: Option<&broadcast::Sender<TelemetryEvent>>,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<()> {
        let missing = self.missing();
        if missing.is_empty() {
            self.refresh_state();
            return Ok(());
        }
        {
            let mut st = self.status.write().unwrap();
            st.state = ModelState::Downloading;
            st.bytes_done = 0;
            st.bytes_total = missing.iter().map(|f| f.size).sum();
            st.error = None;
        }
        let emit = |ev: TelemetryEvent| {
            if let Some(tx) = events {
                let _ = tx.send(ev);
            }
        };

        std::fs::create_dir_all(&self.dir)?;
        let client = reqwest::Client::builder()
            .user_agent(concat!("ragnarok/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| Error::Inference(format!("http client: {e}")))?;

        for f in missing {
            if cancel.is_cancelled() {
                return Err(Error::Inference("model download cancelled".into()));
            }
            self.status.write().unwrap().current_file = Some(f.dest.into());
            emit(TelemetryEvent::ModelDownload {
                file: f.dest.into(),
                bytes_done: 0,
                bytes_total: Some(f.size),
                status: ragnarok_core::DownloadStatus::Started,
            });
            let url = format!("{}/{}", self.base_url, f.remote);
            if let Err(e) = self.fetch_one(&client, &url, &f, events, &cancel).await {
                if !f.required {
                    tracing::warn!(
                        "optional model file {} unavailable ({e}) — continuing",
                        f.dest
                    );
                    continue;
                }
                self.fail(format!("{}: {e}", f.dest));
                emit(TelemetryEvent::ModelDownload {
                    file: f.dest.into(),
                    bytes_done: 0,
                    bytes_total: Some(f.size),
                    status: ragnarok_core::DownloadStatus::Failed,
                });
                return Err(e);
            }
            emit(TelemetryEvent::ModelDownload {
                file: f.dest.into(),
                bytes_done: f.size,
                bytes_total: Some(f.size),
                status: ragnarok_core::DownloadStatus::Done,
            });
        }
        // final recompute — refresh_state() would skip while the status
        // still says `Downloading`, so settle it directly
        {
            let mut st = self.status.write().unwrap();
            st.state = if Self::ready_in(&self.dir) {
                ModelState::Ready
            } else {
                ModelState::Missing
            };
            st.current_file = None;
        }
        let st = self.status.read().unwrap().state;
        if st != ModelState::Ready {
            let msg: String = "model download finished but required files are missing".into();
            self.fail(msg.clone());
            return Err(Error::Inference(msg));
        }
        Ok(())
    }

    fn fail(&self, msg: String) {
        let mut st = self.status.write().unwrap();
        st.state = ModelState::Failed;
        st.error = Some(msg);
    }

    /// Stream one file to `dest.part`, verify, atomically rename.
    async fn fetch_one(
        &self,
        client: &reqwest::Client,
        url: &str,
        f: &ModelFile,
        events: Option<&broadcast::Sender<TelemetryEvent>>,
        cancel: &tokio_util::sync::CancellationToken,
    ) -> Result<()> {
        let tmp = self.dir.join(format!("{}.part", f.dest));
        let dest = self.dir.join(f.dest);
        let mut resp = client
            .get(url)
            .send()
            .await
            .map_err(|e| Error::Inference(format!("{url}: {e}")))?;
        if !resp.status().is_success() {
            return Err(Error::Inference(format!("{url}: HTTP {}", resp.status())));
        }

        let mut file = tokio::fs::File::create(&tmp).await?;
        let mut sha256 = sha2::Sha256::new();
        let mut done: u64 = 0;
        let mut last_emit = std::time::Instant::now() - std::time::Duration::from_secs(1);
        while let Some(chunk) = resp
            .chunk()
            .await
            .map_err(|e| Error::Inference(format!("{url} read: {e}")))?
        {
            if cancel.is_cancelled() {
                drop(file);
                let _ = tokio::fs::remove_file(&tmp).await;
                return Err(Error::Inference("model download cancelled".into()));
            }
            file.write_all(&chunk).await?;
            sha256.update(&chunk);
            done += chunk.len() as u64;
            {
                let mut st = self.status.write().unwrap();
                st.bytes_done += chunk.len() as u64;
            }
            if last_emit.elapsed() >= std::time::Duration::from_millis(500) {
                last_emit = std::time::Instant::now();
                if let Some(tx) = events {
                    let _ = tx.send(TelemetryEvent::ModelDownload {
                        file: f.dest.into(),
                        bytes_done: done,
                        bytes_total: Some(f.size),
                        status: ragnarok_core::DownloadStatus::Downloading,
                    });
                }
            }
        }
        file.flush().await?;
        drop(file);

        // integrity: exact size, then pinned hash
        if done != f.size {
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(Error::Inference(format!(
                "{}: expected {} bytes, got {done}",
                f.dest, f.size
            )));
        }
        if let Some(tx) = events {
            let _ = tx.send(TelemetryEvent::ModelDownload {
                file: f.dest.into(),
                bytes_done: done,
                bytes_total: Some(f.size),
                status: ragnarok_core::DownloadStatus::Verifying,
            });
        }
        let want256 = self.pin_for(f).map(|s| s.to_ascii_lowercase());
        let want1 = f.git_sha1.map(|s| s.to_ascii_lowercase());
        if let Some(want) = want256 {
            let got256 = hex::encode(sha256.finalize());
            if got256 != want {
                let _ = tokio::fs::remove_file(&tmp).await;
                return Err(Error::Inference(format!(
                    "{}: sha256 mismatch (got {got256}, want {want})",
                    f.dest
                )));
            }
        } else if let Some(want) = want1 {
            // git blob sha1 covers "blob {len}\0{content}" — recompute on
            // the verified-size temp file
            let expect = git_blob_sha1(&tmp)?;
            if expect != want {
                let _ = tokio::fs::remove_file(&tmp).await;
                return Err(Error::Inference(format!(
                    "{}: sha1 mismatch (got {expect}, want {want})",
                    f.dest
                )));
            }
        }
        tokio::fs::rename(&tmp, &dest).await?;
        Ok(())
    }
}

/// Re-verify a file on disk against its pinned hash(es).
fn verify_file(path: &Path, f: &ModelFile, p: &Provisioner) -> Result<bool> {
    if let Some(want) = p.pin_for(f) {
        let mut h = sha2::Sha256::new();
        std::io::copy(&mut std::fs::File::open(path)?, &mut h)?;
        Ok(hex::encode(h.finalize()).eq_ignore_ascii_case(want))
    } else if let Some(want) = f.git_sha1 {
        Ok(git_blob_sha1(path)?.eq_ignore_ascii_case(want))
    } else {
        Ok(true)
    }
}

/// Git blob object id: `sha1("blob {len}\0" + content)`.
fn git_blob_sha1(path: &Path) -> Result<String> {
    let data = std::fs::read(path)?;
    let mut h = sha1::Sha1::new();
    h.update(format!("blob {}\0", data.len()).as_bytes());
    h.update(&data);
    Ok(hex::encode(h.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use tokio_util::sync::CancellationToken;

    /// Tiny canned HTTP server: `path → (status, body)`.
    fn mock_server(
        files: Vec<(&'static str, u16, Vec<u8>)>,
    ) -> (String, std::thread::JoinHandle<()>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", l.local_addr().unwrap());
        let h = std::thread::spawn(move || {
            for stream in l.incoming() {
                let Ok(mut s) = stream else { continue };
                let mut buf = vec![0u8; 8192];
                let n = match s.read(&mut buf) {
                    Ok(n) => n,
                    Err(_) => continue,
                };
                let req = String::from_utf8_lossy(&buf[..n]);
                let path = req
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("/")
                    .trim_start_matches('/');
                let hit = files.iter().find(|(p, _, _)| *p == path);
                let (status, body): (u16, &[u8]) = match hit {
                    Some((_, code, b)) => (*code, b),
                    None => (404, b"nope"),
                };
                let reason = if status == 200 { "OK" } else { "Not Found" };
                let head = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = s.write_all(head.as_bytes());
                let _ = s.write_all(body);
            }
        });
        (base, h)
    }

    fn sha256_hex(data: &[u8]) -> String {
        let mut h = sha2::Sha256::new();
        h.update(data);
        hex::encode(h.finalize())
    }

    fn test_manifest(model_body: &[u8], tok_body: &[u8]) -> Vec<ModelFile> {
        vec![
            ModelFile {
                remote: "m/model.onnx",
                dest: "model.onnx",
                size: model_body.len() as u64,
                sha256: Some(Box::leak(sha256_hex(model_body).into_boxed_str())),
                git_sha1: None,
                required: true,
            },
            ModelFile {
                remote: "m/big.bin",
                dest: "big.bin",
                size: 10,
                sha256: None,
                git_sha1: None,
                required: false,
            },
            ModelFile {
                remote: "m/tokenizer.json",
                dest: "tokenizer.json",
                size: tok_body.len() as u64,
                sha256: Some(Box::leak(sha256_hex(tok_body).into_boxed_str())),
                git_sha1: None,
                required: true,
            },
        ]
    }

    #[tokio::test]
    async fn provisions_verifies_and_installs() {
        let dir = tempfile::tempdir().unwrap();
        let model = b"fake onnx graph payload".to_vec();
        let tok = br#"{"tokenizer":"fake"}"#.to_vec();
        // graph >50MB marker irrelevant here — we place model.onnx_data so
        // the ready-check's external-data heuristic passes
        let (base, _h) = mock_server(vec![
            ("m/model.onnx", 200, model.clone()),
            ("m/tokenizer.json", 200, tok.clone()),
            ("m/big.bin", 404, vec![]),
        ]);
        let files = test_manifest(&model, &tok);
        let p = Provisioner::with_files(dir.path().to_path_buf(), Some(base), None, files);
        // ready_in needs model.onnx_data only when model.onnx is small —
        // pre-place it to satisfy the external-data heuristic
        std::fs::write(dir.path().join("model.onnx_data"), b"weights").unwrap();

        let (tx, _rx) = tokio::sync::broadcast::channel(32);
        p.download(Some(&tx), CancellationToken::new())
            .await
            .unwrap();

        assert_eq!(p.status().read().unwrap().state, ModelState::Ready);
        assert_eq!(std::fs::read(dir.path().join("model.onnx")).unwrap(), model);
        assert_eq!(
            std::fs::read(dir.path().join("tokenizer.json")).unwrap(),
            tok
        );
        // optional 404 file skipped, no .part leftovers
        assert!(!dir.path().join("big.bin").exists());
        assert!(std::fs::read_dir(dir.path()).unwrap().all(|e| !e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".part")));
    }

    #[tokio::test]
    async fn hash_mismatch_aborts_without_install() {
        let dir = tempfile::tempdir().unwrap();
        let (base, _h) = mock_server(vec![
            ("m/model.onnx", 200, b"tampered".to_vec()),
            ("m/tokenizer.json", 200, b"{}".to_vec()),
        ]);
        let files = vec![ModelFile {
            remote: "m/model.onnx",
            dest: "model.onnx",
            size: 8,
            sha256: Some("0".repeat(64).leak()),
            git_sha1: None,
            required: true,
        }];
        let p = Provisioner::with_files(dir.path().to_path_buf(), Some(base), None, files);
        let err = p
            .download(None, CancellationToken::new())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("sha256 mismatch"), "{err}");
        assert!(!dir.path().join("model.onnx").exists());
        assert_eq!(p.status().read().unwrap().state, ModelState::Failed);
        assert!(p.status().read().unwrap().error.is_some());
    }

    #[test]
    fn ready_in_requires_graph_tokenizer_and_data() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!Provisioner::ready_in(dir.path()));
        std::fs::write(dir.path().join("model.onnx"), b"small graph").unwrap();
        assert!(!Provisioner::ready_in(dir.path())); // no tokenizer
        std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
        // small graph → external-data companion required
        assert!(!Provisioner::ready_in(dir.path()));
        std::fs::write(dir.path().join("model.onnx_data"), b"w").unwrap();
        assert!(Provisioner::ready_in(dir.path()));
    }

    #[test]
    fn missing_detects_partial_state() {
        let dir = tempfile::tempdir().unwrap();
        let files = test_manifest(b"model", b"tok");
        let p = Provisioner::with_files(dir.path().to_path_buf(), None, None, files);
        // nothing on disk → required + optional missing
        assert_eq!(p.missing().len(), 3);
        // plant required files with correct sizes+hashes
        std::fs::write(dir.path().join("model.onnx"), b"model").unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), b"tok").unwrap();
        assert_eq!(p.missing().len(), 1); // only the optional big.bin
        assert_eq!(p.missing()[0].dest, "big.bin");
    }

    #[test]
    fn git_blob_sha1_matches_git_object_id() {
        // git hash-object semantics: sha1("blob {len}\0" + content)
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("x");
        std::fs::write(&f, b"hello").unwrap();
        let mut h = sha1::Sha1::new();
        h.update(b"blob 5\0");
        h.update(b"hello");
        assert_eq!(git_blob_sha1(&f).unwrap(), hex::encode(h.finalize()));
    }
}
