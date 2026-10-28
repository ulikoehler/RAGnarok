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
                Request::delete(&format!("/api/v1/sources/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NO_CONTENT);
        let res = app
            .clone()
            .oneshot(
                Request::delete(&format!("/api/v1/sources/{id}"))
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
                Request::get(&format!("/api/v1/documents/{}/stream", rec.id))
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
                Request::get(&format!("/api/v1/documents/{}/stream", vrec.id))
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
                Request::get(&format!(
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
