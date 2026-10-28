//! ragnarok-server — Axum REST API, WebSocket telemetry broker and embedded
//! SPA asset server (spec §3.5 / §4).
//!
//! ```text
//! GET  /api/v1/system/stats
//! GET  /api/v1/sources
//! POST /api/v1/sources
//! DELETE /api/v1/sources/{id}
//! POST /api/v1/sources/{id}/rescan
//! POST /api/v1/search
//! GET  /api/v1/documents/{document_id}/stream
//! GET  /ws/v1/events            (telemetry, 250 ms stats + event bus)
//! GET  /*                       (embedded SPA)
//! ```

pub mod agent;
pub mod chat;
pub mod routes;
pub mod state;
pub mod ws;

pub use state::{AppState, SourceWithState};

use axum::routing::{delete, get, post};
use axum::Router;
use std::net::SocketAddr;
use std::sync::Arc;

/// Build the complete application router.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/v1/system/stats", get(routes::get_stats))
        .route(
            "/api/v1/sources",
            get(routes::list_sources).post(routes::add_source),
        )
        .route("/api/v1/sources/{id}", delete(routes::delete_source))
        .route("/api/v1/sources/{id}/rescan", post(routes::rescan_source))
        .route("/api/v1/search", post(routes::search))
        .route(
            "/api/v1/documents/{id}/stream",
            get(routes::stream_document),
        )
        .route("/ws/v1/events", get(ws::ws_events))
        .merge(chat::router(state.clone()))
        .route("/", get(routes::spa_root))
        .route("/{*path}", get(routes::spa))
        .with_state(state)
}

/// Bind + serve; returns when `cancel` fires.
pub async fn serve(
    state: Arc<AppState>,
    bind: &str,
    cancel: tokio_util::sync::CancellationToken,
) -> ragnarok_core::Result<()> {
    // Agent runs live in spawned tokio tasks — after a restart none of
    // them exist, so every `running` claim is dead. Release them all at
    // boot or threads stay bricked until the steal window expires.
    match state.engine.meta.chat_release_dead_claims() {
        Ok(n) if n > 0 => tracing::info!(released = n, "released dead chat run claims"),
        Err(e) => tracing::warn!(%e, "chat claim cleanup failed"),
        _ => {}
    }
    let app = router(state);
    let addr: SocketAddr = bind
        .parse()
        .map_err(|e| ragnarok_core::Error::Config(format!("bad bind address {bind}: {e}")))?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "ragnarok http/ws listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(async move { cancel.cancelled().await })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use ragnarok_core::{CrawlerConfig, ResourceConfig};
    use ragnarok_crawler::IngestPipeline;
    use ragnarok_inference::{HashingEmbedder, InferencePool};
    use ragnarok_resource::{BackpressureController, ResourceMonitor};
    use ragnarok_store::{MetadataDb, SearchEngine, VectorEngine};
    use std::path::Path;
    use tokio::sync::broadcast;
    use tokio_util::sync::CancellationToken;
    use tower::ServiceExt;

    async fn app(dir: &Path) -> (Arc<AppState>, Router, CancellationToken) {
        let meta = Arc::new(MetadataDb::open(&dir.join("m.db")).unwrap());
        let vectors = VectorEngine::open(&dir.join("v"), 1024).unwrap();
        let embedder = Arc::new(HashingEmbedder::default());
        let engine = Arc::new(SearchEngine::new(vectors, meta, embedder.clone()));
        let bp = BackpressureController::new(ResourceConfig::default());
        let pool = InferencePool::new(embedder, 2, 4, bp);
        let monitor = ResourceMonitor::sysinfo(ResourceConfig::default());
        let (tx, _rx) = broadcast::channel(64);
        let pipeline = IngestPipeline::new(
            engine.clone(),
            pool,
            monitor.controller(),
            &CrawlerConfig::default(),
            tx.clone(),
        );
        let cancel = CancellationToken::new();
        let st = AppState::new(
            engine,
            monitor,
            pipeline,
            tx,
            cancel.clone(),
            std::time::Duration::from_millis(50),
        );
        (st.clone(), router(st), cancel)
    }

    async fn json_get(app: &Router, uri: &str) -> (StatusCode, serde_json::Value) {
        let res = app
            .clone()
            .oneshot(Request::get(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let code = res.status();
        let body = to_bytes(res.into_body(), 1 << 20).await.unwrap();
        (code, serde_json::from_slice(&body).unwrap_or_default())
    }

    #[tokio::test]
    async fn stats_endpoint_returns_telemetry() {
        let dir = tempfile::tempdir().unwrap();
        let (_st, app, _c) = app(dir.path()).await;
        let (code, v) = json_get(&app, "/api/v1/system/stats").await;
        assert_eq!(code, StatusCode::OK);
        assert!(v["budget_bytes"].as_u64().unwrap() > 0);
        assert!(v.get("pressure").is_some());
        assert!(v.get("indexed_documents").is_some());
    }

    #[tokio::test]
    async fn sources_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let (_st, app, cancel) = app(dir.path()).await;

        // empty list
        let (code, v) = json_get(&app, "/api/v1/sources").await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v.as_array().unwrap().len(), 0);

        // register an FS source pointing at a temp dir with a file
        let docs = tempfile::tempdir().unwrap();
        std::fs::write(docs.path().join("hello.txt"), "hello world content").unwrap();
        let body = serde_json::json!({
            "name": "docs",
            "config": {"type": "file_system", "path": docs.path(), "watch": false}
        });
        let res = app
            .clone()
            .oneshot(
                Request::post("/api/v1/sources")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::CREATED);
        let body = to_bytes(res.into_body(), 1 << 16).await.unwrap();
        let id = serde_json::from_slice::<serde_json::Value>(&body).unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();

        // invalid fs path rejected
        let bad = serde_json::json!({
            "name": "x", "config": {"type": "file_system", "path": "/nonexistent-xyz"}
        });
        let res = app
            .clone()
            .oneshot(
                Request::post("/api/v1/sources")
                    .header("content-type", "application/json")
                    .body(Body::from(bad.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);

        // list contains it
        let (_c, v) = json_get(&app, "/api/v1/sources").await;
        assert_eq!(v.as_array().unwrap().len(), 1);
        assert_eq!(v[0]["name"], "docs");

        // give the background scan a moment, then search should hit
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let res = app
            .clone()
            .oneshot(
                Request::post("/api/v1/search")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"query":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let hits: serde_json::Value =
            serde_json::from_slice(&to_bytes(res.into_body(), 1 << 20).await.unwrap()).unwrap();
        assert!(!hits.as_array().unwrap().is_empty(), "expected a hit");

        // delete
        let res = app
            .clone()
            .oneshot(
                Request::delete(format!("/api/v1/sources/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NO_CONTENT);
        let res = app
            .clone()
            .oneshot(
                Request::delete(format!("/api/v1/sources/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
        cancel.cancel();
    }

    #[tokio::test]
    async fn document_stream_serves_file_and_blobs() {
        let dir = tempfile::tempdir().unwrap();
        let (st, app, cancel) = app(dir.path()).await;

        // index a real file via the pipeline
        let f = dir.path().join("note.txt");
        std::fs::write(&f, "stream me please").unwrap();
        let sid = ragnarok_core::SourceId::from("s");
        st.pipeline.ingest_path(&sid, &f).await.unwrap();
        let rec = st
            .engine
            .meta
            .get_document_by_path(&sid, &f.display().to_string())
            .unwrap()
            .unwrap();

        let res = app
            .clone()
            .oneshot(
                Request::get(format!("/api/v1/documents/{}/stream", rec.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = to_bytes(res.into_body(), 1 << 16).await.unwrap();
        assert_eq!(&body[..], b"stream me please");

        // virtual doc via blob path
        st.pipeline
            .ingest_virtual(
                &sid,
                "imap://INBOX/7.eml",
                b"From: a@x\r\nSubject: t\r\n\r\nmsg\r\n",
            )
            .await
            .unwrap();
        let vrec = st
            .engine
            .meta
            .get_document_by_path(&sid, "imap://INBOX/7.eml")
            .unwrap()
            .unwrap();
        let res = app
            .clone()
            .oneshot(
                Request::get(format!("/api/v1/documents/{}/stream", vrec.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers()["content-type"], "message/rfc822");
        let body = to_bytes(res.into_body(), 1 << 16).await.unwrap();
        assert!(body.starts_with(b"From:"));

        // unknown id -> 404
        let res = app
            .oneshot(
                Request::get(format!(
                    "/api/v1/documents/{}/stream",
                    ragnarok_core::DocumentId::new()
                ))
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
        cancel.cancel();
    }

    #[tokio::test]
    async fn spa_serves_embedded_index() {
        let dir = tempfile::tempdir().unwrap();
        let (_st, app, _c) = app(dir.path()).await;
        let res = app
            .clone()
            .oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers()["content-type"], "text/html");
        // client-side route -> fallback to index.html
        let res = app
            .oneshot(
                Request::get("/some/client/route")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn rescan_and_error_paths() {
        let dir = tempfile::tempdir().unwrap();
        let (st, app, cancel) = app(dir.path()).await;
        let docs = tempfile::tempdir().unwrap();
        std::fs::write(docs.path().join("f.txt"), "rescan me").unwrap();

        // rescan unknown id -> 404
        let res = app
            .clone()
            .oneshot(
                Request::post("/api/v1/sources/nope/rescan")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);

        // register fs source via API then rescan -> 202
        let body = serde_json::json!({
            "name": "d", "config": {"type": "file_system", "path": docs.path(), "watch": false}
        });
        let res = app
            .clone()
            .oneshot(
                Request::post("/api/v1/sources")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let id = serde_json::from_slice::<serde_json::Value>(
            &to_bytes(res.into_body(), 1 << 16).await.unwrap(),
        )
        .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();
        let res = app
            .clone()
            .oneshot(
                Request::post(format!("/api/v1/sources/{id}/rescan"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::ACCEPTED);

        // imap source can be registered but rescan is rejected (400)
        let imap = serde_json::json!({
            "name": "m",
            "config": {"type": "imap", "host": "127.0.0.1", "port": 1,
                       "username": "u", "password_env": "RAGNAROK_NO_SUCH_PW",
                       "mailbox": "INBOX"}
        });
        let res = app
            .clone()
            .oneshot(
                Request::post("/api/v1/sources")
                    .header("content-type", "application/json")
                    .body(Body::from(imap.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::CREATED);
        let mid = serde_json::from_slice::<serde_json::Value>(
            &to_bytes(res.into_body(), 1 << 16).await.unwrap(),
        )
        .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();
        let res = app
            .clone()
            .oneshot(
                Request::post(format!("/api/v1/sources/{mid}/rescan"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);

        // bad document id in stream -> 400
        let res = app
            .clone()
            .oneshot(
                Request::get("/api/v1/documents/not-a-uuid/stream")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);

        // missing file on disk -> 404
        let gone = dir.path().join("gone.txt");
        std::fs::write(&gone, "temp").unwrap();
        st.pipeline
            .ingest_path(&ragnarok_core::SourceId::from("s"), &gone)
            .await
            .unwrap();
        let rec = st
            .engine
            .meta
            .get_document_by_path(&"s".into(), &gone.display().to_string())
            .unwrap()
            .unwrap();
        std::fs::remove_file(&gone).unwrap();
        let res = app
            .clone()
            .oneshot(
                Request::get(format!("/api/v1/documents/{}/stream", rec.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
        cancel.cancel();
    }

    async fn post(app: &Router, uri: &str, body: serde_json::Value) -> (StatusCode, String) {
        let res = app
            .clone()
            .oneshot(
                Request::post(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let code = res.status();
        let bytes = to_bytes(res.into_body(), 1 << 20).await.unwrap();
        (code, String::from_utf8_lossy(&bytes).to_string())
    }

    // ------------------------------------------------------------------
    // scripted OpenAI-compatible mock LLM
    // ------------------------------------------------------------------

    /// One scripted response: a list of `(delay_ms, sse_chunk)` parts so
    /// tests can stall mid-stream (stop/steer windows).
    type Script = Vec<(u64, String)>;

    struct MockLlm {
        url: String,
        requests: std::sync::Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
    }

    fn sse_delta(delta: serde_json::Value) -> String {
        format!(
            "data: {}\n\n",
            serde_json::json!({"choices":[{"delta":delta}]})
        )
    }

    fn sse_end() -> String {
        "data: [DONE]\n\n".to_owned()
    }

    /// Streamed `content` then `[DONE]` — a plain answer (no task_done →
    /// parks as unfinished).
    fn script_text(text: &str) -> Script {
        vec![(
            0,
            format!(
                "{}{}",
                sse_delta(serde_json::json!({"content": text})),
                sse_end()
            ),
        )]
    }

    /// A single streamed tool-call delta then `[DONE]`; `args` is raw
    /// JSON, split mid-string to exercise accumulation across chunks.
    fn script_tool(name: &str, args: &str) -> Script {
        let (a, b) = args.split_at(args.len() / 2);
        vec![
            (
                0,
                sse_delta(serde_json::json!({"tool_calls":[{
                    "index":0,"id":"call_1","type":"function",
                    "function":{"name":name,"arguments":a}}]})),
            ),
            (
                0,
                sse_delta(serde_json::json!({"tool_calls":[{
                    "index":0,
                    "function":{"arguments":b}}]})),
            ),
            (0, sse_end()),
        ]
    }

    /// Final answer + `task_done` in the same completion — a finished run.
    fn script_finished(text: &str) -> Script {
        vec![(
            0,
            format!(
                "{}{}{}",
                sse_delta(serde_json::json!({"content": text})),
                sse_delta(serde_json::json!({"tool_calls":[{
                    "index":0,"id":"call_9","type":"function",
                    "function":{"name":"task_done","arguments":"{}"}}]})),
                sse_end()
            ),
        )]
    }

    fn find_sub(hay: &[u8], needle: &[u8]) -> Option<usize> {
        hay.windows(needle.len()).position(|w| w == needle)
    }

    /// Minimal HTTP/1.1 server: one connection per request, captures the
    /// JSON body, answers with the next scripted SSE body.
    async fn mock_llm(scripts: Vec<Script>) -> MockLlm {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!(
            "http://127.0.0.1:{}/v1",
            listener.local_addr().unwrap().port()
        );
        let scripts = std::sync::Arc::new(std::sync::Mutex::new(std::collections::VecDeque::from(
            scripts,
        )));
        let requests = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        tokio::spawn({
            let scripts = scripts.clone();
            let requests = requests.clone();
            async move {
                while let Ok((mut sock, _)) = listener.accept().await {
                    let scripts = scripts.clone();
                    let requests = requests.clone();
                    tokio::spawn(async move {
                        // read request head + body (Content-Length)
                        let mut buf = Vec::new();
                        let head_end = loop {
                            if let Some(p) = find_sub(&buf, b"\r\n\r\n") {
                                let head = String::from_utf8_lossy(&buf[..p]).to_lowercase();
                                let len: usize = head
                                    .lines()
                                    .find_map(|l| l.strip_prefix("content-length:"))
                                    .and_then(|v| v.trim().parse().ok())
                                    .unwrap_or(0);
                                if buf.len() >= p + 4 + len {
                                    break p;
                                }
                            }
                            let mut tmp = [0u8; 8192];
                            match sock.read(&mut tmp).await {
                                Ok(0) | Err(_) => return,
                                Ok(n) => buf.extend_from_slice(&tmp[..n]),
                            }
                        };
                        let head = String::from_utf8_lossy(&buf[..head_end]).to_lowercase();
                        let len: usize = head
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length:"))
                            .and_then(|v| v.trim().parse().ok())
                            .unwrap_or(0);
                        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(
                            &buf[head_end + 4..head_end + 4 + len],
                        ) {
                            requests.lock().unwrap().push(v);
                        }
                        let script = scripts.lock().unwrap().pop_front().unwrap_or_else(|| {
                            vec![(
                                0,
                                format!(
                                    "{}{}",
                                    sse_delta(serde_json::json!({"content": "(leer)"})),
                                    sse_end()
                                ),
                            )]
                        });
                        let total: usize = script.iter().map(|(_, c)| c.len()).sum();
                        let head = format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {total}\r\nconnection: close\r\n\r\n"
                        );
                        if sock.write_all(head.as_bytes()).await.is_err() {
                            return;
                        }
                        for (delay, chunk) in script {
                            if delay > 0 {
                                tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                            }
                            if sock.write_all(chunk.as_bytes()).await.is_err() {
                                return;
                            }
                        }
                    });
                }
            }
        });
        MockLlm { url, requests }
    }

    /// Point `chat_settings` at the mock provider.
    fn use_mock(st: &crate::state::AppState, mock: &MockLlm) {
        st.engine
            .meta
            .chat_put_settings(&ragnarok_store::metadata::ChatSettingsRow {
                base_url: mock.url.clone(),
                api_key: "test-key".into(),
                model: "mock-1".into(),
                max_steps: 12,
                context_tokens: 128_000,
                ..Default::default()
            })
            .unwrap();
    }

    fn parse_frames(sse: &str) -> Vec<serde_json::Value> {
        sse.split("\n\n")
            .filter_map(|f| {
                f.trim()
                    .strip_prefix("data: ")
                    .and_then(|d| serde_json::from_str(d).ok())
            })
            .collect()
    }

    async fn create_thread_id(app: &Router) -> String {
        let (code, body) = post(app, "/api/v1/chat/threads", serde_json::json!({})).await;
        assert_eq!(code, StatusCode::CREATED);
        serde_json::from_str::<serde_json::Value>(&body).unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string()
    }

    #[tokio::test]
    async fn chat_agent_tool_call_then_done() {
        let dir = tempfile::tempdir().unwrap();
        let (st, app, cancel) = app(dir.path()).await;
        let f = dir.path().join("velo.txt");
        std::fs::write(&f, "Velofahren stärkt Herz und Kreislauf.").unwrap();
        st.pipeline
            .ingest_path(&ragnarok_core::SourceId::from("s"), &f)
            .await
            .unwrap();
        let mock = mock_llm(vec![
            script_tool("search", r#"{"query":"velo","limit":3}"#),
            script_finished("Velofahren stärkt das Herz. [1]"),
        ])
        .await;
        use_mock(&st, &mock);
        let tid = create_thread_id(&app).await;

        let res = app
            .clone()
            .oneshot(
                Request::post(format!("/api/v1/chat/threads/{tid}/messages"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"content":"Was stärkt Velofahren?"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers()["content-type"], "text/event-stream");
        let body = to_bytes(res.into_body(), 1 << 20).await.unwrap();
        let frames = parse_frames(&String::from_utf8_lossy(&body));
        let kinds: Vec<&str> = frames.iter().filter_map(|f| f["type"].as_str()).collect();
        assert!(kinds.contains(&"tool_call"), "frames: {kinds:?}");
        assert!(kinds.contains(&"tool_result"), "frames: {kinds:?}");
        assert!(kinds.contains(&"citation"), "no citation frame");
        assert_eq!(kinds.last(), Some(&"done"), "frames: {kinds:?}");

        // two LLM round-trips; request 2 must carry the tool result message
        let reqs = mock.requests.lock().unwrap();
        assert_eq!(reqs.len(), 2, "requests: {reqs:?}");
        assert_eq!(reqs[0]["model"], "mock-1");
        assert!(reqs[0]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["function"]["name"] == "search"));
        let msgs2 = reqs[1]["messages"].as_array().unwrap();
        assert!(msgs2
            .iter()
            .any(|m| m["role"] == "tool" && m["content"].as_str().unwrap().contains("velo.txt")));
        drop(reqs);

        // persisted: user + assistant, tool_calls table populated
        let (_, v) = json_get(&app, &format!("/api/v1/chat/threads/{tid}/messages")).await;
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["model"], "mock-1");
        let calls = v["toolCalls"].as_array().unwrap();
        assert!(calls.iter().any(|c| c["name"] == "search"), "{calls:?}");

        // attach to the finished thread → replayed history ends in done
        let res = app
            .clone()
            .oneshot(
                Request::get(format!("/api/v1/chat/threads/{tid}/stream"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = to_bytes(res.into_body(), 1 << 16).await.unwrap();
        assert!(String::from_utf8_lossy(&body).contains("\"done\""));
        cancel.cancel();
    }

    #[tokio::test]
    async fn chat_unfinished_parks_and_continue_completes() {
        let dir = tempfile::tempdir().unwrap();
        let (st, app, cancel) = app(dir.path()).await;
        let mock = mock_llm(vec![
            script_text("Zwischenstand ohne Abschluss."),
            script_finished("Fertig."),
        ])
        .await;
        use_mock(&st, &mock);
        let tid = create_thread_id(&app).await;

        let res = app
            .clone()
            .oneshot(
                Request::post(format!("/api/v1/chat/threads/{tid}/messages"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"content":"recherchiere"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = to_bytes(res.into_body(), 1 << 20).await.unwrap();
        let frames = parse_frames(&String::from_utf8_lossy(&body));
        assert!(frames.iter().any(|f| f["type"] == "awaiting_continue"));
        let (_, v) = json_get(&app, &format!("/api/v1/chat/threads/{tid}")).await;
        assert_eq!(v["thread"]["awaiting"], true);
        assert_eq!(v["thread"]["running"], false);

        // a second message conflicts while the run is parked-free? — no:
        // parked means not running, so /continue resumes it.
        let res = app
            .clone()
            .oneshot(
                Request::post(format!("/api/v1/chat/threads/{tid}/continue"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = to_bytes(res.into_body(), 1 << 20).await.unwrap();
        let frames = parse_frames(&String::from_utf8_lossy(&body));
        assert_eq!(frames.last().and_then(|f| f["type"].as_str()), Some("done"));
        // the resumed run kept the prior context: request 2 sees the
        // parked assistant text + the completion nudge
        let reqs = mock.requests.lock().unwrap();
        assert_eq!(reqs.len(), 2);
        let msgs = reqs[1]["messages"].as_array().unwrap();
        assert!(msgs.iter().any(|m| m["content"]
            .as_str()
            .is_some_and(|c| c.contains("Zwischenstand"))));
        cancel.cancel();
    }

    #[tokio::test]
    async fn chat_stop_mid_stream_parks_then_continue() {
        let dir = tempfile::tempdir().unwrap();
        let (st, app, cancel) = app(dir.path()).await;
        let mock = mock_llm(vec![
            vec![
                (0, sse_delta(serde_json::json!({"content": "Teil eins. "}))),
                (700, sse_delta(serde_json::json!({"content": "Teil zwei."}))),
                (0, sse_end()),
            ],
            script_finished("Nach dem Stop vollendet."),
        ])
        .await;
        use_mock(&st, &mock);
        let tid = create_thread_id(&app).await;

        let res = app
            .clone()
            .oneshot(
                Request::post(format!("/api/v1/chat/threads/{tid}/messages"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"content":"erzähle"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        // stop while the mock stalls between chunks
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        let (code, v) = post(
            &app,
            &format!("/api/v1/chat/threads/{tid}/stop"),
            serde_json::json!({}),
        )
        .await;
        assert_eq!(code, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_str(&v).unwrap();
        assert_eq!(v["ok"], true);
        let body = to_bytes(res.into_body(), 1 << 20).await.unwrap();
        let frames = parse_frames(&String::from_utf8_lossy(&body));
        assert!(
            frames.iter().any(|f| f["type"] == "awaiting_continue"),
            "frames: {frames:?}"
        );
        let (_, v) = json_get(&app, &format!("/api/v1/chat/threads/{tid}")).await;
        assert_eq!(v["thread"]["awaiting"], true);

        // stop is idempotent for a non-running thread
        let (code, v) = post(
            &app,
            &format!("/api/v1/chat/threads/{tid}/stop"),
            serde_json::json!({}),
        )
        .await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&v).unwrap()["stopping"],
            false
        );

        // continue resumes the parked run
        let res = app
            .clone()
            .oneshot(
                Request::post(format!("/api/v1/chat/threads/{tid}/continue"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = to_bytes(res.into_body(), 1 << 20).await.unwrap();
        let frames = parse_frames(&String::from_utf8_lossy(&body));
        assert_eq!(frames.last().and_then(|f| f["type"].as_str()), Some("done"));
        cancel.cancel();
    }

    #[tokio::test]
    async fn chat_steer_injects_into_running() {
        let dir = tempfile::tempdir().unwrap();
        let (st, app, cancel) = app(dir.path()).await;
        let mock = mock_llm(vec![
            vec![(450, {
                let (a, b) = (r#"{"query":"x""#, r#"}"#);
                format!(
                    "{}{}{}",
                    sse_delta(serde_json::json!({"tool_calls":[{
                            "index":0,"id":"call_1","type":"function",
                            "function":{"name":"search","arguments":a}}]})),
                    sse_delta(serde_json::json!({"tool_calls":[{
                            "index":0,"function":{"arguments":b}}]})),
                    sse_end()
                )
            })],
            script_finished("Gelenkte Antwort."),
        ])
        .await;
        use_mock(&st, &mock);
        let tid = create_thread_id(&app).await;

        // steer on an idle thread → 409 (client falls back to /messages)
        let (code, _) = post(
            &app,
            &format!("/api/v1/chat/threads/{tid}/steer"),
            serde_json::json!({"content":"zu früh"}),
        )
        .await;
        assert_eq!(code, StatusCode::CONFLICT);

        let res = app
            .clone()
            .oneshot(
                Request::post(format!("/api/v1/chat/threads/{tid}/messages"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"content":"suche etwas"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        // steer while the mock delays the tool-call response
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let (code, body) = post(
            &app,
            &format!("/api/v1/chat/threads/{tid}/steer"),
            serde_json::json!({"content":"auch Termine prüfen"}),
        )
        .await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&body).unwrap()["queued"],
            true
        );
        let body = to_bytes(res.into_body(), 1 << 20).await.unwrap();
        let frames = parse_frames(&String::from_utf8_lossy(&body));
        assert!(frames.iter().any(|f| f["type"] == "steered"), "{frames:?}");
        assert_eq!(frames.last().and_then(|f| f["type"].as_str()), Some("done"));
        // the steered text reached the model in the next step's context
        let reqs = mock.requests.lock().unwrap();
        assert_eq!(reqs.len(), 2);
        assert!(reqs[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m["content"]
                .as_str()
                .is_some_and(|c| c.contains("auch Termine prüfen"))));
        drop(reqs);
        // and it persisted as a user message in the transcript
        let (_, v) = json_get(&app, &format!("/api/v1/chat/threads/{tid}/messages")).await;
        assert!(v["messages"].as_array().unwrap().iter().any(|m| {
            m["role"] == "user"
                && m["content"]
                    .as_str()
                    .unwrap()
                    .contains("auch Termine prüfen")
        }));
        cancel.cancel();
    }

    #[tokio::test]
    async fn search_rejects_empty_query() {
        let dir = tempfile::tempdir().unwrap();
        let (_st, app, _c) = app(dir.path()).await;
        let res = app
            .clone()
            .oneshot(
                Request::post("/api/v1/search")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"query":"   "}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }
}
