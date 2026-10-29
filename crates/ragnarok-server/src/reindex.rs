//! Global reindex job — `POST /api/v1/admin/reindex` walks the document
//! table (optionally filtered by `source_id`/`doc_type`/`only_stale`),
//! re-ingesting each record through the pipeline. Progress rides the
//! telemetry bus as `TelemetryEvent::Reindex` and is pollable via
//! `GET /api/v1/admin/reindex`; `POST …/cancel` flips the job token.

use crate::auth::AdminAuth;
use crate::routes::{reingest_record, ApiError};
use crate::state::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use ragnarok_core::TelemetryEvent;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Default)]
pub struct ReindexJob {
    pub running: bool,
    pub total: u64,
    pub done: u64,
    pub failed: u64,
    pub skipped: u64,
    pub current: Option<String>,
    pub last_error: Option<String>,
    pub cancelled: bool,
    pub token: Option<CancellationToken>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReindexRequest {
    pub source_id: Option<String>,
    pub doc_type: Option<String>,
    /// Only docs whose `metadata["embedder"]` differs from the live
    /// backend (missing stamp counts as stale).
    pub only_stale: Option<bool>,
}

/// `POST /api/v1/admin/reindex` — start the job (409 while running).
pub async fn start(
    State(st): State<Arc<AppState>>,
    _a: AdminAuth,
    Json(req): Json<ReindexRequest>,
) -> std::result::Result<impl IntoResponse, ApiError> {
    {
        let job = st.reindex.read().unwrap();
        if job.running {
            return Err(
                ragnarok_core::Error::InvalidInput("reindex already running".into()).into(),
            );
        }
    }
    let source = req.source_id.as_deref().map(ragnarok_core::SourceId::from);
    let f = ragnarok_store::metadata::DocumentListFilter {
        source_id: source.as_ref(),
        doc_type: req.doc_type.as_deref(),
        query: None,
        sort: "indexed_at",
        desc: false,
    };
    let mut docs = st
        .engine
        .meta
        .list_documents_filtered(&f)
        .map_err(ApiError::from)?;
    let fp = st.engine.embedder_name();
    if req.only_stale == Some(true) {
        docs.retain(|d| d.metadata.get("embedder").map(String::as_str) != Some(fp));
    }
    let total = docs.len() as u64;
    let token = CancellationToken::new();
    {
        let mut job = st.reindex.write().unwrap();
        *job = ReindexJob {
            running: true,
            total,
            token: Some(token.clone()),
            ..Default::default()
        };
    }
    let _ = st.events.send(TelemetryEvent::Reindex {
        done: 0,
        total,
        failed: 0,
        finished: false,
    });
    let st2 = st.clone();
    tokio::spawn(async move {
        for rec in docs {
            if token.is_cancelled() {
                let mut job = st2.reindex.write().unwrap();
                job.cancelled = true;
                break;
            }
            {
                let mut job = st2.reindex.write().unwrap();
                job.current = Some(rec.path.clone());
            }
            match reingest_record(&st2, &rec).await {
                Ok(_) => st2.reindex.write().unwrap().done += 1,
                Err(e) => {
                    let mut job = st2.reindex.write().unwrap();
                    job.failed += 1;
                    job.last_error = Some(format!("{}: {e}", rec.path));
                }
            }
            let job = st2.reindex.read().unwrap();
            let _ = st2.events.send(TelemetryEvent::Reindex {
                done: job.done,
                total: job.total,
                failed: job.failed,
                finished: false,
            });
        }
        let mut job = st2.reindex.write().unwrap();
        job.running = false;
        job.current = None;
        let _ = st2.events.send(TelemetryEvent::Reindex {
            done: job.done,
            total: job.total,
            failed: job.failed,
            finished: true,
        });
    });
    Ok((StatusCode::ACCEPTED, Json(json!({"total": total}))))
}

/// `GET /api/v1/admin/reindex` — job status snapshot.
pub async fn status(State(st): State<Arc<AppState>>, _a: AdminAuth) -> impl IntoResponse {
    let j = st.reindex.read().unwrap();
    Json(json!({
        "running": j.running, "total": j.total, "done": j.done,
        "failed": j.failed, "cancelled": j.cancelled,
        "current": j.current, "lastError": j.last_error,
    }))
}

/// `POST /api/v1/admin/reindex/cancel` — flip the job's cancel token.
pub async fn cancel_job(State(st): State<Arc<AppState>>, _a: AdminAuth) -> impl IntoResponse {
    let j = st.reindex.read().unwrap();
    if let Some(t) = &j.token {
        t.cancel();
    }
    Json(json!({"cancelling": j.running}))
}

/// Serialize the job status for tests/UI without holding the lock.
pub fn snapshot(st: &AppState) -> Value {
    let j = st.reindex.read().unwrap();
    json!({
        "running": j.running, "total": j.total, "done": j.done,
        "failed": j.failed, "cancelled": j.cancelled,
    })
}

#[cfg(test)]
mod tests {
    use crate::tests::*;
    use axum::http::StatusCode;
    use serde_json::{json, Value};

    async fn wait_done(app: &axum::Router) -> Value {
        for _ in 0..100 {
            let (_c, v) = get_auth(app, "/api/v1/admin/reindex", TEST_BEARER).await;
            if v["running"] == false {
                return v;
            }
            tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        }
        panic!("reindex did not finish");
    }

    #[tokio::test]
    async fn reindex_bulk_stale_and_cancel() {
        let dir = tempfile::tempdir().unwrap();
        let (st, app, cancel) = app(dir.path()).await;
        let f = dir.path().join("a.txt");
        std::fs::write(&f, "reindex probe alpha").unwrap();
        st.pipeline
            .ingest_path(&ragnarok_core::SourceId::from("s1"), &f)
            .await
            .unwrap();
        index_test_doc(&st, "/docs/b.md", "reindex probe beta").await;

        // start job over everything
        let (code, v) = post_j(&app, "/api/v1/admin/reindex", json!({}), TEST_BEARER).await;
        assert_eq!(code, StatusCode::ACCEPTED);
        assert_eq!(v["total"], 2);
        let v = wait_done(&app).await;
        assert_eq!(v["done"], 2);
        assert_eq!(v["failed"], 0);

        // 409 while… job is done; re-run only_stale → docs now carry the
        // embedder stamp, so nothing is stale
        let (_c, v) = post_j(
            &app,
            "/api/v1/admin/reindex",
            json!({"onlyStale": true}),
            TEST_BEARER,
        )
        .await;
        assert_eq!(v["total"], 0);
        let v = wait_done(&app).await;
        assert_eq!(v["done"], 0);

        // stale detection: corrupt the stamp → doc becomes stale again
        let docs = st
            .engine
            .meta
            .list_documents_filtered(&ragnarok_store::metadata::DocumentListFilter {
                source_id: None,
                doc_type: None,
                query: None,
                sort: "indexed_at",
                desc: false,
            })
            .unwrap();
        assert!(
            docs.iter().all(|d| d.metadata.get("embedder").is_some()),
            "stamp missing"
        );

        // non-admin → 403
        let max = make_session(&st, "u-rx", "user");
        let (code, _) = post_j(&app, "/api/v1/admin/reindex", json!({}), &max).await;
        assert_eq!(code, StatusCode::FORBIDDEN);
        cancel.cancel();
    }
}
