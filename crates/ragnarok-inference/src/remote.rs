use crate::embedder::Embedder;
use crate::hashing::HashingEmbedder;
use async_trait::async_trait;
use ragnarok_core::{EmbeddingTriple, Error, Result};
use serde::Deserialize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

const MAX_ATTEMPTS: usize = 5;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Remote embedding over an OpenAI-compatible HTTP API.
///
/// `POST {base_url}/embeddings` with `{"model": …, "input": […],
/// "encoding_format": "float"}` — works against OpenAI, OpenRouter,
/// Ollama, LM Studio, vLLM and friends. **Document text leaves the
/// machine** unless the URL points at a local server.
///
/// Remote APIs only return dense vectors, so the [`Embedder`]
/// contract's other two channels come from a local
/// [`HashingEmbedder`]: `sparse` keeps lexical/RRF retrieval intact
/// (real term matching, not noise) and `colbert` keeps the multi-vector
/// schema populated — its hashed rows contribute near-constant MaxSim
/// scores across candidates, so the remote dense score dominates the
/// rescore without a special-cased search path.
///
/// `dense_dim` is learned from a warmup probe at construction (and
/// checked against `dimensions` when configured), so the vector store
/// opens with the right size and a remote model switch still trips the
/// `vector_dim` reindex guard.
pub struct RemoteEmbedder {
    client: reqwest::Client,
    /// `{base_url}/embeddings` — the request URL.
    url: String,
    model: String,
    api_key: Option<String>,
    /// `dimensions` request parameter (OpenAI v3 models only) — unset
    /// for providers that reject it.
    dimensions: Option<usize>,
    /// Prepended to query texts (e.g. Nomic `search_query:`).
    query_prefix: String,
    /// Prepended to document texts.
    document_prefix: String,
    /// Dense dimension learned from the first response — `dense_dim()`
    /// prefers it when set, so index-time embeds are validated against
    /// the real provider output.
    learned_dim: AtomicUsize,
    hashing: HashingEmbedder,
    name: &'static str,
    host_label: &'static str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedRow>,
}

#[derive(Deserialize)]
struct EmbedRow {
    index: usize,
    embedding: Vec<f32>,
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

impl RemoteEmbedder {
    /// Build the client and probe the endpoint once — validates the
    /// model name and credentials up-front and learns the dense
    /// dimension before the vector store opens.
    pub async fn connect(
        base_url: &str,
        model: &str,
        api_key: Option<String>,
        dimensions: Option<usize>,
        query_prefix: Option<&str>,
        document_prefix: Option<&str>,
        max_colbert_tokens: usize,
    ) -> Result<Self> {
        let model = model.trim();
        if model.is_empty() {
            return Err(Error::InvalidInput(
                "backend = remote needs `inference.remote_model`".into(),
            ));
        }
        let base = base_url.trim().trim_end_matches('/');
        if base.is_empty() {
            return Err(Error::InvalidInput(
                "backend = remote needs `inference.remote_url`".into(),
            ));
        }
        let host = reqwest::Url::parse(base)
            .map_err(|e| Error::InvalidInput(format!("invalid remote_url '{base}': {e}")))?;
        if !matches!(host.scheme(), "http" | "https") || host.host_str().is_none() {
            return Err(Error::InvalidInput(format!(
                "remote_url '{base}' is not an http(s) URL"
            )));
        }
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|e| Error::Inference(format!("remote client build failed: {e}")))?;
        let url = format!("{base}/embeddings");
        let api_key = api_key.filter(|k| !k.trim().is_empty());
        let this = RemoteEmbedder {
            client,
            url,
            model: model.to_owned(),
            api_key,
            dimensions,
            query_prefix: query_prefix.unwrap_or_default().to_owned(),
            document_prefix: document_prefix.unwrap_or_default().to_owned(),
            learned_dim: AtomicUsize::new(0),
            hashing: HashingEmbedder::new(max_colbert_tokens),
            name: Box::leak(format!("remote:{model}").into_boxed_str()),
            host_label: Box::leak(
                format!("remote:{}", host.host_str().unwrap_or("api")).into_boxed_str(),
            ),
        };
        // Warmup probe: bad model/key/URL fails here, not mid-ingest —
        // and the response dimension lands before `dense_dim()` is
        // needed for the store.
        let dim = this.request_dim(&["ragnarok warmup".to_string()]).await?;
        this.learned_dim.store(dim, Ordering::Relaxed);
        if let Some(cfg) = dimensions {
            if cfg != dim {
                return Err(Error::InvalidInput(format!(
                    "remote_dimensions={cfg} but '{model}' returned {dim}d — fix the config"
                )));
            }
        }
        tracing::info!(model = %model, url = %this.url, dim, "remote embedder connected");
        Ok(this)
    }

    /// One POST for `texts`, returning just the dense dimension — used
    /// by the warmup probe.
    async fn request_dim(&self, texts: &[String]) -> Result<usize> {
        let rows = self.request(texts).await?;
        Ok(rows[0].embedding.len())
    }

    /// POST `{url}` with retry/backoff on 429 + 5xx + transport errors.
    /// Returns the `data` rows in input order.
    async fn request(&self, texts: &[String]) -> Result<Vec<EmbedRow>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut body = serde_json::json!({
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        });
        if let Some(d) = self.dimensions {
            body["dimensions"] = serde_json::json!(d);
        }
        let mut delay = Duration::from_millis(500);
        let mut last_err = String::new();
        for attempt in 1..=MAX_ATTEMPTS {
            let mut req = self.client.post(&self.url).json(&body);
            if let Some(k) = &self.api_key {
                req = req.bearer_auth(k);
            }
            let resp = match req.send().await {
                Ok(r) => r,
                Err(e) => {
                    // Timeouts/connect failures are transient; an
                    // unparseable URL fails fast instead.
                    last_err = format!("remote request failed: {e}");
                    if e.is_connect() || e.is_timeout() || e.is_request() {
                        tracing::warn!(attempt, "{last_err} — retrying");
                        tokio::time::sleep(delay).await;
                        delay = (delay * 2).min(Duration::from_secs(8));
                        continue;
                    }
                    return Err(Error::Inference(last_err));
                }
            };
            let status = resp.status();
            if status.is_success() {
                let rows = resp
                    .json::<EmbedResponse>()
                    .await
                    .map_err(|e| Error::Inference(format!("malformed embeddings response: {e}")))?
                    .data;
                return self.sort_rows(rows, texts.len());
            }
            let retryable = status.as_u16() == 429 || status.is_server_error();
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .map(Duration::from_secs);
            let text = resp.text().await.unwrap_or_default();
            let detail: String = text.chars().take(300).collect();
            last_err = format!("remote HTTP {status}: {detail}");
            if !retryable || attempt == MAX_ATTEMPTS {
                return Err(Error::Inference(last_err));
            }
            let wait = retry_after.unwrap_or(delay);
            tracing::warn!(attempt, wait_s = wait.as_secs(), "{last_err} — retrying");
            tokio::time::sleep(wait).await;
            delay = (delay * 2).min(Duration::from_secs(8));
        }
        Err(Error::Inference(last_err))
    }

    /// OpenAI-compatible servers return `data[].index` — sort by it
    /// rather than trusting response order, and validate count + dims.
    fn sort_rows(&self, mut rows: Vec<EmbedRow>, expect: usize) -> Result<Vec<EmbedRow>> {
        if rows.len() != expect {
            return Err(Error::Inference(format!(
                "remote returned {} embeddings for {} inputs",
                rows.len(),
                expect
            )));
        }
        rows.sort_by_key(|r| r.index);
        let learned = self.learned_dim.load(Ordering::Relaxed);
        for (i, r) in rows.iter().enumerate() {
            if r.index != i {
                return Err(Error::Inference(format!(
                    "remote response indexes not 0..{expect} (got {i}→{})",
                    r.index
                )));
            }
            if r.embedding.is_empty() {
                return Err(Error::Inference(format!(
                    "remote returned an empty embedding at index {i}"
                )));
            }
            if learned > 0 && r.embedding.len() != learned {
                return Err(Error::Inference(format!(
                    "remote dim changed mid-run: {learned}d → {}d at index {i}",
                    r.embedding.len()
                )));
            }
            if i > 0 && r.embedding.len() != rows[0].embedding.len() {
                return Err(Error::Inference(
                    "remote returned mixed dimensions in one batch".into(),
                ));
            }
        }
        Ok(rows)
    }

    /// Remote dense + hashing side-channels → one triple per input.
    async fn embed_prefixed(&self, texts: &[String]) -> Result<Vec<EmbeddingTriple>> {
        let rows = self.request(texts).await?;
        let mut out = Vec::with_capacity(rows.len());
        for (row, raw) in rows.into_iter().zip(texts.iter()) {
            let mut dense = row.embedding;
            l2_normalize(&mut dense);
            let side = self.hashing.embed(raw).await?;
            out.push(EmbeddingTriple {
                dense,
                sparse: side.sparse,
                colbert: side.colbert,
            });
        }
        Ok(out)
    }
}

#[async_trait]
impl Embedder for RemoteEmbedder {
    fn name(&self) -> &'static str {
        self.name
    }

    fn dense_dim(&self) -> usize {
        let learned = self.learned_dim.load(Ordering::Relaxed);
        self.dimensions
            .or_else(|| (learned > 0).then_some(learned))
            .unwrap_or(0)
    }

    async fn embed(&self, text: &str) -> Result<EmbeddingTriple> {
        let mut out = self
            .embed_prefixed(&[format!("{}{text}", self.document_prefix)])
            .await?;
        Ok(out.remove(0))
    }

    async fn embed_query(&self, text: &str) -> Result<EmbeddingTriple> {
        let mut out = self
            .embed_prefixed(&[format!("{}{text}", self.query_prefix)])
            .await?;
        Ok(out.remove(0))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingTriple>> {
        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{t}", self.document_prefix))
            .collect();
        self.embed_prefixed(&prefixed).await
    }

    fn provider_label(&self) -> Option<String> {
        Some(self.host_label.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::AtomicU64;
    use std::sync::Arc;

    /// Tiny canned HTTP server (thread, blocking accepts): the
    /// responder gets the request JSON + Authorization header and
    /// returns `(status, body)`.
    fn mock_server(
        responder: impl Fn(&serde_json::Value, &str) -> (u16, String) + Send + 'static,
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for _ in 0..64 {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let (mut len, mut auth, mut line) = (0usize, String::new(), String::new());
                loop {
                    line.clear();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        break;
                    }
                    let l = line.trim().to_ascii_lowercase();
                    if l.is_empty() {
                        break;
                    }
                    if let Some(v) = l.strip_prefix("content-length:") {
                        len = v.trim().parse().unwrap_or(0);
                    }
                    if let Some(v) = l.strip_prefix("authorization:") {
                        auth = v.trim().to_string();
                    }
                }
                let mut body = vec![0u8; len];
                reader.read_exact(&mut body).ok();
                let req: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
                let (status, resp) = responder(&req, &auth);
                write!(
                    stream,
                    "HTTP/1.1 {status} x\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{resp}",
                    resp.len()
                )
                .ok();
            }
        });
        format!("http://127.0.0.1:{port}")
    }

    fn input_texts(req: &serde_json::Value) -> Vec<String> {
        serde_json::from_value(req["input"].clone()).unwrap_or_default()
    }

    fn rows_for(input: &[String], dim: usize, shuffled: bool) -> String {
        let mut data: Vec<serde_json::Value> = input
            .iter()
            .enumerate()
            .map(|(i, _)| serde_json::json!({"index": i, "embedding": vec![0.1_f32; dim]}))
            .collect();
        if shuffled {
            data.reverse();
        }
        serde_json::json!({"data": data}).to_string()
    }

    #[tokio::test]
    async fn learns_dim_and_embeds() {
        let base = mock_server(|req, _| (200, rows_for(&input_texts(req), 3, false)));
        let e = RemoteEmbedder::connect(&base, "m", None, None, None, None, 8)
            .await
            .unwrap();
        assert_eq!(e.dense_dim(), 3);
        assert_eq!(e.name(), "remote:m");
        let t = e.embed("hello").await.unwrap();
        assert_eq!(t.dense.len(), 3);
        let n: f32 = t.dense.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-5, "dense must be L2-normalized");
        assert!(!t.sparse.is_empty(), "lexical side-channel populated");
        assert!(!t.colbert.is_empty(), "colbert side-channel populated");
    }

    #[tokio::test]
    async fn sorts_out_of_order_batches() {
        let base = mock_server(|req, _| (200, rows_for(&input_texts(req), 2, true)));
        let e = RemoteEmbedder::connect(&base, "m", None, None, None, None, 8)
            .await
            .unwrap();
        let out = e
            .embed_batch(&["a".to_string(), "b".to_string()])
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
        // the lexical channel hashes the input — if the response rows
        // had stayed reversed, out[0] would carry "b"'s sparse terms
        let expect_a = e.hashing.embed("a").await.unwrap().sparse;
        assert_eq!(out[0].sparse, expect_a);
        assert_ne!(out[0].sparse, out[1].sparse);
    }

    #[tokio::test]
    async fn sends_document_and_query_prefixes() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let s2 = seen.clone();
        let base = mock_server(move |req, _| {
            s2.lock().unwrap().extend(input_texts(req));
            (200, rows_for(&input_texts(req), 2, false))
        });
        let e = RemoteEmbedder::connect(&base, "m", None, None, Some("q: "), Some("d: "), 8)
            .await
            .unwrap();
        e.embed("doc").await.unwrap();
        e.embed_query("question").await.unwrap();
        e.embed_batch(&["a".to_string(), "b".to_string()])
            .await
            .unwrap();
        let got = seen.lock().unwrap().clone();
        // index 0 is the warmup probe — the rest must carry prefixes
        assert_eq!(&got[1..], &["d: doc", "q: question", "d: a", "d: b"]);
    }

    #[tokio::test]
    async fn retries_on_429_and_sends_auth() {
        let calls = Arc::new(AtomicU64::new(0));
        let c2 = calls.clone();
        let base = mock_server(move |req, auth| {
            if auth != "bearer sk-test" {
                return (401, "{}".to_string());
            }
            if c2.fetch_add(1, Ordering::Relaxed) == 0 {
                return (429, "{}".to_string());
            }
            (200, rows_for(&input_texts(req), 2, false))
        });
        let e = RemoteEmbedder::connect(&base, "m", Some("sk-test".into()), None, None, None, 8)
            .await
            .unwrap();
        e.embed("x").await.unwrap();
        assert!(
            calls.load(Ordering::Relaxed) >= 2,
            "429 must have been retried"
        );
    }

    #[tokio::test]
    async fn rejects_configured_dim_mismatch() {
        let base = mock_server(|req, _| (200, rows_for(&input_texts(req), 4, false)));
        let err = RemoteEmbedder::connect(&base, "m", None, Some(8), None, None, 8)
            .await
            .err()
            .expect("dim mismatch must fail");
        assert!(err.to_string().contains("remote_dimensions=8"));
    }

    #[tokio::test]
    async fn requires_model_and_http_url() {
        let err = RemoteEmbedder::connect("http://x", "", None, None, None, None, 8)
            .await
            .err()
            .expect("empty model must fail");
        assert!(err.to_string().contains("remote_model"));
        let err = RemoteEmbedder::connect("ftp://x", "m", None, None, None, None, 8)
            .await
            .err()
            .expect("non-http scheme must fail");
        assert!(err.to_string().contains("http"));
    }
}
