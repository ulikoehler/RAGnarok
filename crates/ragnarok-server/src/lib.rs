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

pub mod admin;
pub mod agent;
pub mod auth;
pub mod chat;
pub mod mcp;
pub mod metrics;
pub mod reindex;
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
        .route("/api/v1/sources/{id}/poll", post(routes::set_source_poll))
        .route(
            "/api/v1/settings/runtime",
            get(routes::get_runtime).put(routes::put_runtime),
        )
        .route("/api/mcp", post(mcp::handle))
        .route("/metrics", get(metrics::metrics))
        .route(
            "/api/v1/admin/reindex",
            post(reindex::start).get(reindex::status),
        )
        .route("/api/v1/admin/reindex/cancel", post(reindex::cancel_job))
        .route("/api/v1/audit", get(routes::audit_list))
        .route("/api/v1/auth/login", post(auth::login))
        .route("/api/v1/auth/logout", post(auth::logout))
        .route("/api/v1/auth/me", get(auth::me))
        .route(
            "/api/v1/apikeys",
            post(auth::create_api_key).get(auth::list_api_keys),
        )
        .route(
            "/api/v1/apikeys/{id}",
            axum::routing::delete(auth::delete_api_key),
        )
        .route(
            "/api/v1/admin/users",
            get(admin::list_users).post(admin::create_user),
        )
        .route(
            "/api/v1/admin/users/{id}",
            axum::routing::patch(admin::update_user).delete(admin::delete_user),
        )
        .route(
            "/api/v1/admin/users/{id}/grants",
            get(admin::get_grants).put(admin::set_grants),
        )
        .route(
            "/api/v1/filters",
            get(admin::list_filters).post(admin::create_filter),
        )
        .route(
            "/api/v1/filters/{id}",
            axum::routing::put(admin::update_filter).delete(admin::delete_filter),
        )
        .route("/api/v1/search", post(routes::search))
        .route("/api/v1/documents", get(routes::list_documents))
        .route(
            "/api/v1/documents/{id}",
            axum::routing::delete(routes::delete_document),
        )
        .route(
            "/api/v1/documents/{id}/chunks",
            get(routes::list_document_chunks),
        )
        .route(
            "/api/v1/documents/{id}/context",
            get(routes::document_context),
        )
        .route(
            "/api/v1/documents/{id}/reindex",
            post(routes::reindex_document),
        )
        .route(
            "/api/v1/documents/{id}/stream",
            get(routes::stream_document),
        )
        .route(
            "/api/v1/inference",
            get(routes::inference_status).put(routes::put_inference),
        )
        .route(
            "/api/v1/inference/model/download",
            post(routes::download_model),
        )
        .route(
            "/api/v1/inference/reranker/download",
            post(routes::download_reranker),
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
    auth::bootstrap_admin(&state.engine.meta)?;
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

    pub(crate) async fn app(dir: &Path) -> (Arc<AppState>, Router, CancellationToken) {
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
        auth::seed_test_session(&st);
        (st.clone(), router(st), cancel)
    }

    pub(crate) const TEST_BEARER: &str = "Bearer test-session-token";

    pub(crate) async fn json_get(app: &Router, uri: &str) -> (StatusCode, serde_json::Value) {
        let res = app
            .clone()
            .oneshot(
                Request::get(uri)
                    .header("authorization", TEST_BEARER)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let code = res.status();
        let body = to_bytes(res.into_body(), 1 << 20).await.unwrap();
        (code, serde_json::from_slice(&body).unwrap_or_default())
    }

    #[tokio::test]
    async fn runtime_settings_roundtrip_and_poll_override() {
        let dir = tempfile::tempdir().unwrap();
        let (_st, app, _c) = app(dir.path()).await;

        // GET reflects the boot defaults
        let (code, v) = json_get(&app, "/api/v1/settings/runtime").await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["chunk_max_tokens"], 512);
        assert_eq!(v["ocr_enabled"], true);

        // invalid values are rejected (min > max, soft >= hard)
        let mut bad = v.clone();
        bad["chunk_min_tokens"] = 900.into();
        bad["soft_fraction"] = 0.99.into();
        bad["hard_fraction"] = 0.5.into();
        let res = app
            .clone()
            .oneshot(
                Request::put("/api/v1/settings/runtime")
                    .header("content-type", "application/json")
                    .header("authorization", TEST_BEARER)
                    .body(Body::from(bad.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);

        // valid PUT applies live and persists
        let mut rt = v.clone();
        rt["chunk_max_tokens"] = 300.into();
        rt["ocr_enabled"] = false.into();
        rt["base_permits"] = 2.into();
        let res = app
            .clone()
            .oneshot(
                Request::put("/api/v1/settings/runtime")
                    .header("content-type", "application/json")
                    .header("authorization", TEST_BEARER)
                    .body(Body::from(rt.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let (code, v2) = json_get(&app, "/api/v1/settings/runtime").await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v2["chunk_max_tokens"], 300);
        assert_eq!(v2["ocr_enabled"], false);
        assert_eq!(v2["base_permits"], 2);
        // persisted blob survives — reload through meta
        let meta = ragnarok_store::MetadataDb::open(&dir.path().join("m.db")).unwrap();
        let blob = meta.get_setting("runtime").unwrap().unwrap();
        assert!(blob.contains("\"chunk_max_tokens\":300"));

        // poll override: register a feed source, set its interval
        let sid = "src_polltest";
        let mut src = ragnarok_core::Source::new(
            "polltest",
            ragnarok_core::SourceConfig::Feed {
                url: "http://127.0.0.1:1/x".into(),
                full_text: false,
                poll_interval_secs: 300,
            },
        );
        src.id = ragnarok_core::SourceId::from(sid);
        meta.upsert_source(&src).unwrap();
        let (code, _) = post_j(
            &app,
            &format!("/api/v1/sources/{sid}/poll"),
            serde_json::json!({"secs": 12}),
            TEST_BEARER,
        )
        .await;
        assert_eq!(code, StatusCode::BAD_REQUEST); // <30 rejected
        let (code, v) = post_j(
            &app,
            &format!("/api/v1/sources/{sid}/poll"),
            serde_json::json!({"secs": 45}),
            TEST_BEARER,
        )
        .await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["poll_interval_secs"], 45);
        // persisted into the stored config
        let stored = meta
            .list_sources()
            .unwrap()
            .into_iter()
            .find(|s| s.id.as_str() == sid)
            .unwrap();
        match stored.config {
            ragnarok_core::SourceConfig::Feed {
                poll_interval_secs, ..
            } => assert_eq!(poll_interval_secs, 45),
            _ => panic!("unexpected variant"),
        }
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
                    .header("authorization", TEST_BEARER)
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
                    .header("authorization", TEST_BEARER)
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
                    .header("authorization", TEST_BEARER)
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
                    .header("authorization", TEST_BEARER)
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
                    .header("authorization", TEST_BEARER)
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
                    .header("authorization", TEST_BEARER)
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
                    .header("authorization", TEST_BEARER)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        // eml is served as displayable text — message/rfc822 renders blank
        assert_eq!(res.headers()["content-type"], "text/plain; charset=utf-8");
        let body = to_bytes(res.into_body(), 1 << 16).await.unwrap();
        assert!(body.starts_with(b"From:"));

        // unknown id -> 404
        let res = app
            .oneshot(
                Request::get(format!(
                    "/api/v1/documents/{}/stream",
                    ragnarok_core::DocumentId::new()
                ))
                .header("authorization", TEST_BEARER)
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
            .oneshot(
                Request::get("/")
                    .header("authorization", TEST_BEARER)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers()["content-type"], "text/html");
        // client-side route -> fallback to index.html
        let res = app
            .oneshot(
                Request::get("/some/client/route")
                    .header("authorization", TEST_BEARER)
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
                    .header("authorization", TEST_BEARER)
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
                    .header("authorization", TEST_BEARER)
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
                    .header("authorization", TEST_BEARER)
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
                    .header("authorization", TEST_BEARER)
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
                    .header("authorization", TEST_BEARER)
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
                    .header("authorization", TEST_BEARER)
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
                    .header("authorization", TEST_BEARER)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
        cancel.cancel();
    }

    pub(crate) async fn post(
        app: &Router,
        uri: &str,
        body: serde_json::Value,
    ) -> (StatusCode, String) {
        post_auth(app, uri, body, TEST_BEARER).await
    }

    pub(crate) async fn post_auth(
        app: &Router,
        uri: &str,
        body: serde_json::Value,
        bearer: &str,
    ) -> (StatusCode, String) {
        let res = app
            .clone()
            .oneshot(
                Request::post(uri)
                    .header("content-type", "application/json")
                    .header("authorization", bearer)
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let code = res.status();
        let bytes = to_bytes(res.into_body(), 1 << 20).await.unwrap();
        (code, String::from_utf8_lossy(&bytes).to_string())
    }

    /// `post_auth` variant that parses the JSON body (non-JSON → `null`).
    pub(crate) async fn post_j(
        app: &Router,
        uri: &str,
        body: serde_json::Value,
        bearer: &str,
    ) -> (StatusCode, serde_json::Value) {
        let (code, text) = post_auth(app, uri, body, bearer).await;
        (code, serde_json::from_str(&text).unwrap_or_default())
    }

    pub(crate) async fn get_auth(
        app: &Router,
        uri: &str,
        bearer: &str,
    ) -> (StatusCode, serde_json::Value) {
        let res = app
            .clone()
            .oneshot(
                Request::get(uri)
                    .header("authorization", bearer)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let code = res.status();
        let body = to_bytes(res.into_body(), 1 << 20).await.unwrap();
        (code, serde_json::from_slice(&body).unwrap_or_default())
    }

    // ------------------------------------------------------------------
    // auth + scope enforcement
    // ------------------------------------------------------------------

    pub(crate) fn make_session(st: &AppState, uid: &str, role: &str) -> String {
        st.engine.meta.create_user(uid, uid, "x", role).unwrap();
        let token = format!("tok-{uid}");
        st.engine.meta.create_session(&token, uid, 3600).unwrap();
        format!("Bearer {token}")
    }

    pub(crate) async fn index_test_doc(st: &AppState, path: &str, text: &str) {
        st.pipeline
            .ingest_virtual(&ragnarok_core::SourceId::from("s1"), path, text.as_bytes())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn auth_flow_and_401_403() {
        let dir = tempfile::tempdir().unwrap();
        let (st, app, cancel) = app(dir.path()).await;

        // no token → 401
        let res = app
            .clone()
            .oneshot(
                Request::get("/api/v1/system/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

        // bad token → 401
        let (code, _) = get_auth(&app, "/api/v1/system/stats", "Bearer nope").await;
        assert_eq!(code, StatusCode::UNAUTHORIZED);

        // login with wrong password → 401 (seed user has hash "x")
        st.engine
            .meta
            .create_user(
                "u-login",
                "loginuser",
                &crate::auth::hash_password("pw123").unwrap(),
                "user",
            )
            .unwrap();
        let (code, _) = post_j(
            &app,
            "/api/v1/auth/login",
            serde_json::json!({"username": "loginuser", "password": "wrong"}),
            TEST_BEARER,
        )
        .await;
        assert_eq!(code, StatusCode::UNAUTHORIZED);

        // correct login → token
        let (code, v) = post_j(
            &app,
            "/api/v1/auth/login",
            serde_json::json!({"username": "loginuser", "password": "pw123"}),
            TEST_BEARER,
        )
        .await;
        assert_eq!(code, StatusCode::OK);
        let token = format!("Bearer {}", v["token"].as_str().unwrap());
        assert_eq!(v["user"]["role"], "user");

        // /me works; admin route is 403 for a plain user
        let (code, v) = get_auth(&app, "/api/v1/auth/me", &token).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["username"], "loginuser");
        let (code, _) = get_auth(&app, "/api/v1/admin/users", &token).await;
        assert_eq!(code, StatusCode::FORBIDDEN);

        // admin (seeded by `app()`) can list users
        let (code, v) = get_auth(&app, "/api/v1/admin/users", TEST_BEARER).await;
        assert_eq!(code, StatusCode::OK);
        assert!(v.as_array().unwrap().len() >= 2);

        // logout kills the session
        let res = app
            .clone()
            .oneshot(
                Request::post("/api/v1/auth/logout")
                    .header("authorization", &token)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NO_CONTENT);
        let (code, _) = get_auth(&app, "/api/v1/auth/me", &token).await;
        assert_eq!(code, StatusCode::UNAUTHORIZED);
        cancel.cancel();
    }

    #[tokio::test]
    async fn scoped_search_enforces_allow_and_deny() {
        let dir = tempfile::tempdir().unwrap();
        let (st, app, cancel) = app(dir.path()).await;
        index_test_doc(
            &st,
            "/srv/docs/public/report.txt",
            "quarterly pangolin report",
        )
        .await;
        index_test_doc(&st, "/srv/docs/secret/memo.txt", "pangolin secret memo").await;
        index_test_doc(
            &st,
            "imap://INBOX/42.eml",
            "Subject: pangolin\n\npangolin mail",
        )
        .await;

        // regular user "alice" with allow-grant on /public/** only
        let alice = make_session(&st, "alice", "user");
        let fid = uuid::Uuid::new_v4().to_string();
        st.engine
            .meta
            .create_filter(
                &fid,
                "nur-public",
                Some("alice"),
                &serde_json::json!({"include": {"path": "/srv/docs/public/**"}}).to_string(),
            )
            .unwrap();
        st.engine
            .meta
            .set_grants("alice", &[(fid.clone(), "allow".into())])
            .unwrap();

        // alice searches: only the public doc is visible
        let (code, v) = post_j(
            &app,
            "/api/v1/search",
            serde_json::json!({"query": "pangolin"}),
            &alice,
        )
        .await;
        assert_eq!(code, StatusCode::OK);
        let hits = v.as_array().unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0]["path"], "/srv/docs/public/report.txt");

        // admin sees everything
        let (code, v) = post_j(
            &app,
            "/api/v1/search",
            serde_json::json!({"query": "pangolin"}),
            TEST_BEARER,
        )
        .await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v.as_array().unwrap().len(), 3);

        // ad-hoc deny scope narrows further (admin, exclude imap folder)
        let (code, v) = post_j(
            &app,
            "/api/v1/search",
            serde_json::json!({"query": "pangolin",
                "scope": {"exclude": {"emailFolder": "INBOX"}}}),
            TEST_BEARER,
        )
        .await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v.as_array().unwrap().len(), 2);

        // unknown filterId → 400
        let (code, _) = post_j(
            &app,
            "/api/v1/search",
            serde_json::json!({"query": "pangolin", "filterId": "no-such"}),
            TEST_BEARER,
        )
        .await;
        assert_eq!(code, StatusCode::BAD_REQUEST);

        // stream of a denied doc → 404 for alice (existence must not leak)
        let secret = st
            .engine
            .meta
            .get_document_by_path(&"s1".into(), "/srv/docs/secret/memo.txt")
            .unwrap()
            .unwrap();
        let (code, _) = get_auth(
            &app,
            &format!("/api/v1/documents/{}/stream", secret.id),
            &alice,
        )
        .await;
        assert_eq!(code, StatusCode::NOT_FOUND);
        cancel.cancel();
    }

    /// `ragk_…` service keys: mint (admin-only), authenticate as the
    /// bound user, honor the key-bound scope, revoke → 401.
    #[tokio::test]
    async fn api_keys_lifecycle_and_scope() {
        let dir = tempfile::tempdir().unwrap();
        let (st, app, cancel) = app(dir.path()).await;
        index_test_doc(
            &st,
            "/srv/docs/public/report.txt",
            "quarterly pangolin report",
        )
        .await;
        index_test_doc(&st, "/srv/docs/secret/memo.txt", "pangolin secret memo").await;

        // non-admin cannot mint keys
        let alice = make_session(&st, "alice", "user");
        let (code, _) = post_j(
            &app,
            "/api/v1/apikeys",
            serde_json::json!({"name": "x"}),
            &alice,
        )
        .await;
        assert_eq!(code, StatusCode::FORBIDDEN);

        // admin mints: key comes back once, bound to a chosen user
        let (code, v) = post_j(
            &app,
            "/api/v1/apikeys",
            serde_json::json!({"name": "svc-bot", "user_id": "alice"}),
            TEST_BEARER,
        )
        .await;
        assert_eq!(code, StatusCode::OK, "{v}");
        let key = format!("Bearer {}", v["key"].as_str().unwrap());
        assert!(v["key"].as_str().unwrap().starts_with("ragk_"));
        assert_eq!(v["username"], "alice");

        // key resolves to alice — even though alice has no grants
        // (no grants = unrestricted for plain users) so she searches all
        let (code, m) = get_auth(&app, "/api/v1/auth/me", &key).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(m["username"], "alice");

        // scope-bound key: admin identity but path-narrowed
        let (code, v) = post_j(
            &app,
            "/api/v1/apikeys",
            serde_json::json!({
                "name": "scoped",
                "scope": {"include": {"path": "/srv/docs/public/**"}}
            }),
            TEST_BEARER,
        )
        .await;
        assert_eq!(code, StatusCode::OK);
        let scoped = format!("Bearer {}", v["key"].as_str().unwrap());
        let (code, hits) = post_j(
            &app,
            "/api/v1/search",
            serde_json::json!({"query": "pangolin"}),
            &scoped,
        )
        .await;
        assert_eq!(code, StatusCode::OK);
        let hits = hits.as_array().unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0]["path"], "/srv/docs/public/report.txt");
        // out-of-scope stream → 404, no existence leak
        let secret = st
            .engine
            .meta
            .get_document_by_path(&"s1".into(), "/srv/docs/secret/memo.txt")
            .unwrap()
            .unwrap();
        let (code, _) = get_auth(
            &app,
            &format!("/api/v1/documents/{}/stream", secret.id),
            &scoped,
        )
        .await;
        assert_eq!(code, StatusCode::NOT_FOUND);

        // admin search via the scoped key's sibling (unscoped) sees all
        let (code, hits) = post_j(
            &app,
            "/api/v1/search",
            serde_json::json!({"query": "pangolin"}),
            &key,
        )
        .await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(hits.as_array().unwrap().len(), 2);

        // invalid scope JSON rejected at mint time
        let (code, _) = post_j(
            &app,
            "/api/v1/apikeys",
            serde_json::json!({"name": "bad", "scope": {"bogus_field": 1}}),
            TEST_BEARER,
        )
        .await;
        assert_eq!(code, StatusCode::BAD_REQUEST);

        // list shows prefix + username, never the hash
        let (code, v) = get_auth(&app, "/api/v1/apikeys", TEST_BEARER).await;
        assert_eq!(code, StatusCode::OK);
        let keys = v.as_array().unwrap();
        assert_eq!(keys.len(), 2);
        assert!(keys[0]["prefix"].as_str().unwrap().starts_with("ragk_"));
        assert!(keys[0].get("key_hash").is_none());
        let scoped_id = keys.iter().find(|k| k["name"] == "scoped").unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();

        // revoke → immediate 401 on next use
        let res = app
            .clone()
            .oneshot(
                Request::delete(format!("/api/v1/apikeys/{scoped_id}"))
                    .header("authorization", TEST_BEARER)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NO_CONTENT);
        let (code, _) = get_auth(&app, "/api/v1/auth/me", &scoped).await;
        assert_eq!(code, StatusCode::UNAUTHORIZED);
        // the alice-bound key still works
        let (code, _) = get_auth(&app, "/api/v1/auth/me", &key).await;
        assert_eq!(code, StatusCode::OK);
        cancel.cancel();
    }

    /// `GET /documents`: scope-filtered listing with pagination, path
    /// query, type and source filters, sorting — `total` reflects the
    /// post-scope count.
    #[tokio::test]
    async fn documents_list_scopes_filters_and_paginates() {
        let dir = tempfile::tempdir().unwrap();
        let (st, app, cancel) = app(dir.path()).await;
        index_test_doc(&st, "/srv/docs/public/a-report.txt", "alpha report").await;
        index_test_doc(&st, "/srv/docs/public/b-notes.md", "# beta notes").await;
        index_test_doc(&st, "/srv/docs/secret/c-memo.txt", "gamma memo").await;

        // admin sees all three, sorted by path
        let (code, v) = json_get(&app, "/api/v1/documents").await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["total"], 3);
        let items = v["items"].as_array().unwrap();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0]["path"], "/srv/docs/public/a-report.txt");
        assert_eq!(items[0]["docType"], "text");
        assert!(items[0]["chunkCount"].as_u64().unwrap() >= 1);
        assert!(items[0]["indexedAt"].is_string());

        // path query filter
        let (code, v) = json_get(&app, "/api/v1/documents?q=notes").await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["total"], 1);
        assert_eq!(v["items"][0]["path"], "/srv/docs/public/b-notes.md");

        // doc-type filter
        let (code, v) = json_get(&app, "/api/v1/documents?type=markdown").await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["total"], 1);

        // source filter: docs carry source "s1", no docs under "zzz"
        let (code, v) = json_get(&app, "/api/v1/documents?source=s1").await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["total"], 3);
        let (code, v) = json_get(&app, "/api/v1/documents?source=zzz").await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["total"], 0);

        // pagination: limit=2 + offset=2 → page 2 has the last item
        let (code, v) = json_get(&app, "/api/v1/documents?limit=2&offset=0").await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["items"].as_array().unwrap().len(), 2);
        assert_eq!(v["total"], 3);
        let (code, v) = json_get(&app, "/api/v1/documents?limit=2&offset=2").await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["items"].as_array().unwrap().len(), 1);
        assert_eq!(v["items"][0]["path"], "/srv/docs/secret/c-memo.txt");

        // sort desc by size/indexed order still valid
        let (code, v) = json_get(&app, "/api/v1/documents?sort=indexed_at&dir=desc").await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["items"].as_array().unwrap().len(), 3);

        // invalid sort falls back to a safe default (no 500/injection)
        let (code, _) = json_get(&app, "/api/v1/documents?sort=path;DROP").await;
        assert_eq!(code, StatusCode::OK);

        // scoped user: alice only sees /public/** → secret never listed
        let alice = make_session(&st, "alice", "user");
        let fid = uuid::Uuid::new_v4().to_string();
        st.engine
            .meta
            .create_filter(
                &fid,
                "nur-public",
                Some("alice"),
                &serde_json::json!({"include": {"path": "/srv/docs/public/**"}}).to_string(),
            )
            .unwrap();
        st.engine
            .meta
            .set_grants("alice", &[(fid.clone(), "allow".into())])
            .unwrap();
        let (code, v) = get_auth(&app, "/api/v1/documents", &alice).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["total"], 2);
        assert!(v["items"]
            .as_array()
            .unwrap()
            .iter()
            .all(|d| d["path"].as_str().unwrap().starts_with("/srv/docs/public/")));
        cancel.cancel();
    }

    /// `GET /documents/{id}/chunks`, `DELETE` and `reindex` — scoped
    /// detail access (out-of-scope = 404), admin-gated mutations.
    #[tokio::test]
    async fn document_chunks_delete_and_reindex() {
        let dir = tempfile::tempdir().unwrap();
        let (st, app, cancel) = app(dir.path()).await;
        index_test_doc(
            &st,
            "/srv/docs/public/chunky.txt",
            "one two three four five",
        )
        .await;
        let rec = st
            .engine
            .meta
            .get_document_by_path(&"s1".into(), "/srv/docs/public/chunky.txt")
            .unwrap()
            .unwrap();

        // chunks endpoint returns the indexed chunks
        let (code, v) = json_get(&app, &format!("/api/v1/documents/{}/chunks", rec.id)).await;
        assert_eq!(code, StatusCode::OK);
        let chunks = v["chunks"].as_array().unwrap();
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0]["chunkIndex"], 0);
        assert!(chunks[0]["text"].is_string());

        // unauthenticated → 401
        let res = app
            .clone()
            .oneshot(
                Request::get(format!("/api/v1/documents/{}/chunks", rec.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

        // unknown doc id → 404
        let (code, _) = json_get(&app, "/api/v1/documents/not-a-uuid/chunks").await;
        assert_eq!(code, StatusCode::BAD_REQUEST); // unparseable id
        let fake = uuid::Uuid::new_v4();
        let (code, _) = json_get(&app, &format!("/api/v1/documents/{fake}/chunks")).await;
        assert_eq!(code, StatusCode::NOT_FOUND);

        // scoped user without grant → chunks 404 (no existence leak)
        let bob = make_session(&st, "bob", "user");
        let fid = uuid::Uuid::new_v4().to_string();
        st.engine
            .meta
            .create_filter(
                &fid,
                "nirgends",
                Some("bob"),
                &serde_json::json!({"include": {"path": "/elsewhere/**"}}).to_string(),
            )
            .unwrap();
        st.engine
            .meta
            .set_grants("bob", &[(fid.clone(), "allow".into())])
            .unwrap();
        let (code, _) = get_auth(&app, &format!("/api/v1/documents/{}/chunks", rec.id), &bob).await;
        assert_eq!(code, StatusCode::NOT_FOUND);

        // reindex: non-admin → 403, admin → outcome
        let res = app
            .clone()
            .oneshot(
                Request::post(format!("/api/v1/documents/{}/reindex", rec.id))
                    .header("authorization", &bob)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::FORBIDDEN);
        let res = app
            .clone()
            .oneshot(
                Request::post(format!("/api/v1/documents/{}/reindex", rec.id))
                    .header("authorization", TEST_BEARER)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(res.into_body(), 1 << 16).await.unwrap()).unwrap();
        assert!(v["outcome"].is_string(), "expected outcome, got {v}");

        // delete: non-admin → 403, admin → 204, then doc + chunks are gone
        let res = app
            .clone()
            .oneshot(
                Request::delete(format!("/api/v1/documents/{}", rec.id))
                    .header("authorization", &bob)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::FORBIDDEN);
        let res = app
            .clone()
            .oneshot(
                Request::delete(format!("/api/v1/documents/{}", rec.id))
                    .header("authorization", TEST_BEARER)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NO_CONTENT);
        assert!(st.engine.meta.get_document(rec.id).unwrap().is_none());
        let (code, _) = json_get(&app, &format!("/api/v1/documents/{}/chunks", rec.id)).await;
        assert_eq!(code, StatusCode::NOT_FOUND);
        let res = app
            .clone()
            .oneshot(
                Request::delete(format!("/api/v1/documents/{}", rec.id))
                    .header("authorization", TEST_BEARER)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
        cancel.cancel();
    }

    /// `/inference`: admin-only status + backend persistence + model
    /// download trigger. Uses a provisioner with a fake manifest served
    /// by a local mock — no real model is fetched.
    #[tokio::test]
    async fn inference_status_and_download() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        let dir = tempfile::tempdir().unwrap();
        let (st, app, cancel) = app(dir.path()).await;

        // unauth → 401, non-admin → 403, admin → 200
        let res = app
            .clone()
            .oneshot(
                Request::get("/api/v1/inference")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
        let user = make_session(&st, "u-inf", "user");
        let (code, _) = get_auth(&app, "/api/v1/inference", &user).await;
        assert_eq!(code, StatusCode::FORBIDDEN);
        let (code, v) = json_get(&app, "/api/v1/inference").await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["backend"], "hashing");
        assert!(v.get("onnxCompiled").is_some());
        assert!(v["model"].is_null(), "no provisioner in bare harness");

        // no provisioner → download/put both 400
        let res = app
            .clone()
            .oneshot(
                Request::post("/api/v1/inference/model/download")
                    .header("authorization", TEST_BEARER)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);

        // attach a provisioner backed by a local mock + a config path
        let model_dir = dir.path().join("models");
        std::fs::create_dir_all(&model_dir).unwrap();
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", l.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in l.incoming() {
                let Ok(mut s) = stream else { continue };
                let mut buf = [0u8; 4096];
                let _ = s.read(&mut buf);
                let body = b"fake";
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = s.write_all(head.as_bytes());
                let _ = s.write_all(body);
            }
        });
        let files = vec![
            ragnarok_inference::provision::ModelFile {
                remote: "model.onnx",
                dest: "model.onnx",
                size: 4,
                sha256: None,
                git_sha1: None,
                required: true,
            },
            ragnarok_inference::provision::ModelFile {
                remote: "tokenizer.json",
                dest: "tokenizer.json",
                size: 4,
                sha256: None,
                git_sha1: None,
                required: true,
            },
            // small graph → external-data companion required by ready_in
            ragnarok_inference::provision::ModelFile {
                remote: "model.onnx_data",
                dest: "model.onnx_data",
                size: 4,
                sha256: None,
                git_sha1: None,
                required: false,
            },
        ];
        let prov = ragnarok_inference::Provisioner::with_files(model_dir, Some(base), None, files);
        // a second provisioner covers the reranker model slot
        let reranker_dir = dir.path().join("reranker");
        std::fs::create_dir_all(&reranker_dir).unwrap();
        let rprov = ragnarok_inference::Provisioner::for_reranker(
            reranker_dir,
            Some("http://127.0.0.1:1".into()), // unreachable — no download
            None,
        );
        let st2 = st.with_inference(
            ragnarok_core::InferenceBackend::Hashing,
            prov,
            rprov,
            true,
            dir.path().join("config.toml"),
        );
        let app2 = router(st2);

        let (code, v) = json_get(&app2, "/api/v1/inference").await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["model"]["state"], "missing");
        assert_eq!(v["rerank"]["enabled"], true);
        assert_eq!(v["rerank"]["model"]["state"], "missing");

        // download → 202 → files land → status ready
        let res = app2
            .clone()
            .oneshot(
                Request::post("/api/v1/inference/model/download")
                    .header("authorization", TEST_BEARER)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        tokio::time::sleep(std::time::Duration::from_millis(400)).await;
        let (_, v) = json_get(&app2, "/api/v1/inference").await;
        assert_eq!(v["model"]["state"], "ready");

        // PUT hashing (active backend) → needsRestart false
        let res = app2
            .clone()
            .oneshot(
                Request::put("/api/v1/inference")
                    .header("content-type", "application/json")
                    .header("authorization", TEST_BEARER)
                    .body(Body::from(r#"{"backend":"hashing"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(res.into_body(), 1 << 16).await.unwrap()).unwrap();
        assert_eq!(v["needsRestart"], false);
        let saved = std::fs::read_to_string(dir.path().join("config.toml")).unwrap();
        assert!(saved.contains("backend"), "config not persisted: {saved}");

        // PUT rerank=false → persisted + needsRestart true (stage is on)
        let res = app2
            .clone()
            .oneshot(
                Request::put("/api/v1/inference")
                    .header("content-type", "application/json")
                    .header("authorization", TEST_BEARER)
                    .body(Body::from(r#"{"rerank":false}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(res.into_body(), 1 << 16).await.unwrap()).unwrap();
        assert_eq!(v["rerank"], false);
        assert_eq!(v["needsRestart"], true);
        let saved = std::fs::read_to_string(dir.path().join("config.toml")).unwrap();
        assert!(
            saved.contains("rerank = false"),
            "rerank not persisted: {saved}"
        );

        // PUT rerank=true on a non-onnx binary → rejected
        if !ragnarok_inference::ONNX_COMPILED {
            let res = app2
                .clone()
                .oneshot(
                    Request::put("/api/v1/inference")
                        .header("content-type", "application/json")
                        .header("authorization", TEST_BEARER)
                        .body(Body::from(r#"{"rerank":true}"#))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        }

        cancel.cancel();
    }

    /// Remote (non-copying) sources: `/documents/{id}/stream` re-fetches
    /// the original live from the remote system when neither a blob nor
    /// a local file exists.
    #[tokio::test]
    async fn remote_document_streams_on_demand() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        // tiny canned web server serving the "original" page
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", l.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in l.incoming() {
                let Ok(mut s) = stream else { continue };
                let mut buf = [0u8; 4096];
                let _ = s.read(&mut buf);
                let body = "<html><body>live remote original</body></html>";
                let _ = s.write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                );
            }
        });

        let dir = tempfile::tempdir().unwrap();
        let (st, app, cancel) = app(dir.path()).await;
        // register a Web source + a document record pointing at the mock
        let url = format!("{base}/page");
        let src = ragnarok_core::Source::new(
            "websrc",
            ragnarok_core::SourceConfig::Web {
                urls: vec![url.clone()],
                follow_links: false,
                poll_interval_secs: 3600,
            },
        );
        st.engine.meta.upsert_source(&src).unwrap();
        let doc_id = ragnarok_core::document_id(&src.id, &url);
        st.engine
            .meta
            .upsert_document(&ragnarok_core::DocumentRecord {
                id: doc_id,
                source_id: src.id.clone(),
                path: url.clone(),
                hash: "e".into(),
                doc_type: ragnarok_core::DocType::Html,
                size: 10,
                modified: None,
                indexed_at: chrono::Utc::now(),
                chunk_count: 1,
                metadata: std::collections::HashMap::from([("remote_url".into(), url.clone())]),
            })
            .unwrap();
        // no blob, no fs file → stream must hit the remote
        let res = app
            .clone()
            .oneshot(
                Request::get(format!("/api/v1/documents/{doc_id}/stream"))
                    .header("authorization", TEST_BEARER)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = to_bytes(res.into_body(), 1 << 16).await.unwrap();
        assert!(String::from_utf8_lossy(&body).contains("live remote original"));
        cancel.cancel();
    }

    #[tokio::test]
    async fn chat_threads_are_per_user() {
        let dir = tempfile::tempdir().unwrap();
        let (st, app, cancel) = app(dir.path()).await;
        let alice = make_session(&st, "alice", "user");
        let bob = make_session(&st, "bob", "user");

        // alice creates a thread
        let (code, v) = post_j(&app, "/api/v1/chat/threads", serde_json::json!({}), &alice).await;
        assert_eq!(code, StatusCode::CREATED);
        let tid = v["id"].as_str().unwrap().to_string();

        // bob's list is empty; alice's has 1
        let (_, v) = get_auth(&app, "/api/v1/chat/threads", &bob).await;
        assert_eq!(v["threads"].as_array().unwrap().len(), 0);
        let (_, v) = get_auth(&app, "/api/v1/chat/threads", &alice).await;
        assert_eq!(v["threads"].as_array().unwrap().len(), 1);

        // bob gets 404 on alice's thread — for every verb
        let (code, _) = get_auth(&app, &format!("/api/v1/chat/threads/{tid}/messages"), &bob).await;
        assert_eq!(code, StatusCode::NOT_FOUND);
        let (code, _) = post_j(
            &app,
            &format!("/api/v1/chat/threads/{tid}/stop"),
            serde_json::json!({}),
            &bob,
        )
        .await;
        assert_eq!(code, StatusCode::NOT_FOUND);

        // filter CRUD: alice creates one, bob can't see/touch it
        let (code, v) = post_j(
            &app,
            "/api/v1/filters",
            serde_json::json!({"name": "mine", "expr": {"path": "/a/**"}}),
            &alice,
        )
        .await;
        assert_eq!(code, StatusCode::CREATED);
        let fid = v["id"].as_str().unwrap().to_string();
        let (_, v) = get_auth(&app, "/api/v1/filters", &bob).await;
        assert_eq!(v.as_array().unwrap().len(), 0);
        let res = app
            .clone()
            .oneshot(
                Request::delete(format!("/api/v1/filters/{fid}"))
                    .header("authorization", &bob)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST); // "not your filter" → 400

        // invalid filter expr rejected at write time
        let (code, _) = post_j(
            &app,
            "/api/v1/filters",
            serde_json::json!({"name": "bad", "expr": {"path": "["}}),
            &alice,
        )
        .await;
        assert_eq!(code, StatusCode::BAD_REQUEST);
        cancel.cancel();
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

    /// Non-streaming provider mock for the settings probes: `GET /v1/models`
    /// returns an id list, `POST /v1/chat/completions` a tiny JSON reply.
    /// Captures the request lines for assertions.
    async fn mock_provider() -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!(
            "http://127.0.0.1:{}/v1",
            listener.local_addr().unwrap().port()
        );
        let lines = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        tokio::spawn({
            let lines = lines.clone();
            async move {
                while let Ok((mut sock, _)) = listener.accept().await {
                    let lines = lines.clone();
                    tokio::spawn(async move {
                        let mut buf = Vec::new();
                        let head_end = loop {
                            if let Some(p) = find_sub(&buf, b"\r\n\r\n") {
                                break p;
                            }
                            let mut tmp = [0u8; 8192];
                            match sock.read(&mut tmp).await {
                                Ok(0) | Err(_) => return,
                                Ok(n) => buf.extend_from_slice(&tmp[..n]),
                            }
                        };
                        let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
                        let reqline = head.lines().next().unwrap_or("").to_string();
                        lines.lock().unwrap().push(reqline.clone());
                        let (status, body) = if reqline.starts_with("GET /v1/models") {
                            (
                                "200 OK",
                                r#"{"object":"list","data":[{"id":"m-b"},{"id":"m-a"}]}"#,
                            )
                        } else if reqline.starts_with("POST /v1/chat/completions") {
                            (
                                "200 OK",
                                r#"{"model":"m-a","choices":[{"message":{"role":"assistant","content":"bereit"}}],"usage":{"prompt_tokens":4,"completion_tokens":1}}"#,
                            )
                        } else {
                            ("404 Not Found", "{}")
                        };
                        let resp = format!(
                            "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                            body.len()
                        );
                        let _ = sock.write_all(resp.as_bytes()).await;
                    });
                }
            }
        });
        (url, lines)
    }

    #[tokio::test]
    async fn settings_models_and_test_probes() {
        let dir = tempfile::tempdir().unwrap();
        let (st, app, cancel) = app(dir.path()).await;
        let (url, lines) = mock_provider().await;
        st.engine
            .meta
            .chat_put_settings(&ragnarok_store::metadata::ChatSettingsRow {
                base_url: url.clone(),
                api_key: "k".into(),
                model: "m-a".into(),
                ..Default::default()
            })
            .unwrap();

        // GET /settings/models → id list from the mock
        let (code, v) = json_get(&app, "/api/v1/chat/settings/models").await;
        assert_eq!(code, StatusCode::OK);
        let ids: Vec<&str> = v["models"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| m.as_str())
            .collect();
        assert_eq!(ids, ["m-a", "m-b"]);

        // POST /settings/test → real completion round-trip
        let res = app
            .clone()
            .oneshot(
                Request::post("/api/v1/chat/settings/test")
                    .header("content-type", "application/json")
                    .header("authorization", TEST_BEARER)
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(res.into_body(), 1 << 20).await.unwrap()).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["reply"], "bereit");
        assert_eq!(v["model"], "m-a");
        assert!(v["latencyMs"].as_u64().is_some());

        let calls = lines.lock().unwrap().clone();
        assert!(
            calls.iter().any(|l| l.starts_with("GET /v1/models")),
            "{calls:?}"
        );
        assert!(
            calls
                .iter()
                .any(|l| l.starts_with("POST /v1/chat/completions")),
            "{calls:?}"
        );

        // unauthenticated → 401, non-admin → 403
        let res = app
            .clone()
            .oneshot(
                Request::post("/api/v1/chat/settings/test")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
        let user = make_session(&st, "probe-user", "user");
        let res = app
            .clone()
            .oneshot(
                Request::post("/api/v1/chat/settings/test")
                    .header("content-type", "application/json")
                    .header("authorization", user)
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::FORBIDDEN);

        cancel.cancel();
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
                    .header("authorization", TEST_BEARER)
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
        let reqs = mock.requests.lock().unwrap().clone();
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
                    .header("authorization", TEST_BEARER)
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
                    .header("authorization", TEST_BEARER)
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
                    .header("authorization", TEST_BEARER)
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
                    .header("authorization", TEST_BEARER)
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
                    .header("authorization", TEST_BEARER)
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
                    .header("authorization", TEST_BEARER)
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
        let reqs = mock.requests.lock().unwrap().clone();
        assert_eq!(reqs.len(), 2);
        assert!(reqs[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m["content"]
                .as_str()
                .is_some_and(|c| c.contains("auch Termine prüfen"))));
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
                    .header("authorization", TEST_BEARER)
                    .body(Body::from(r#"{"query":"   "}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }
}
