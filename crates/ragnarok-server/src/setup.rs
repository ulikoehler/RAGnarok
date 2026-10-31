//! First-run setup surface for the onboarding wizard.
//! `GET /api/v1/setup` reports the checklist the wizard walks through
//! (embedding engine, stale docs, first source, LLM provider) plus the
//! persisted `setup_done` flag; `POST /api/v1/setup/done` marks it
//! complete. `POST /api/v1/admin/restart` re-execs the daemon so a model
//! download or persisted config change takes effect without shell access.

use crate::auth::AdminAuth;
use crate::routes::{audit_admin, ApiError};
use crate::state::AppState;
use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use serde_json::json;
use std::sync::Arc;

/// `GET /api/v1/setup` — everything the wizard needs in one call (admin).
pub async fn status(State(st): State<Arc<AppState>>, _a: AdminAuth) -> impl IntoResponse {
    let meta = &st.engine.meta;
    let done = matches!(meta.get_setting("setup_done"), Ok(Some(v)) if v == "1");
    // stale = docs stamped with a different embedder than the live one
    // (the hashing → onnx switch makes every older doc stale)
    let fp = st.engine.embedder_name();
    let stale_docs = meta
        .list_documents_filtered(&ragnarok_store::metadata::DocumentListFilter {
            source_id: None,
            doc_type: None,
            query: None,
            sort: "indexed_at",
            desc: false,
        })
        .map(|docs| {
            docs.iter()
                .filter(|d| d.metadata.get("embedder").map(String::as_str) != Some(fp))
                .count()
        })
        .unwrap_or(0);
    let sources = meta.list_sources().map(|s| s.len()).unwrap_or(0);
    let llm = meta
        .chat_settings()
        .map(|s| !s.base_url.is_empty() && !s.model.is_empty())
        .unwrap_or(false);
    let model = st
        .provisioner
        .as_ref()
        .map(|p| p.status().read().unwrap().clone());
    Json(json!({
        "done": done,
        "backend": st.inference_backend,
        "onnxCompiled": ragnarok_inference::ONNX_COMPILED,
        "model": model,
        "staleDocs": stale_docs,
        "sources": sources,
        "llmConfigured": llm,
    }))
}

/// `POST /api/v1/setup/done` — persist completion; the wizard stops
/// suggesting itself after logins.
pub async fn done(
    State(st): State<Arc<AppState>>,
    auth: AdminAuth,
) -> std::result::Result<impl IntoResponse, ApiError> {
    st.engine.meta.set_setting("setup_done", "1")?;
    audit_admin(&st, &auth, "setup.done", None, None);
    Ok(Json(json!({ "done": true })))
}

/// `POST /api/v1/admin/restart` — re-exec the daemon: spawn a copy of the
/// running binary with the same arguments, then shut this process down
/// gracefully. The child retries the port bind (up to ~15 s) until the
/// parent's listener is released, so service resumes on its own. Under a
/// supervisor (systemd `Restart=`, docker `--restart`) the spawned copy
/// dies with the unit/container and the plain exit is what restarts us —
/// either way the daemon comes back.
pub async fn restart(State(st): State<Arc<AppState>>, auth: AdminAuth) -> impl IntoResponse {
    audit_admin(&st, &auth, "admin.restart", None, None);
    let cancel = st.cancel.clone();
    tokio::spawn(async move {
        // let the response flush before tearing anything down
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        match std::env::current_exe() {
            Ok(exe) => {
                let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
                match std::process::Command::new(&exe)
                    .args(&args)
                    .stdin(std::process::Stdio::null())
                    .spawn()
                {
                    Ok(child) => tracing::info!(pid = child.id(), "re-exec spawned"),
                    Err(e) => {
                        tracing::error!(%e, "re-exec spawn failed — exiting anyway")
                    }
                }
            }
            Err(e) => tracing::error!(%e, "cannot resolve own binary — exiting anyway"),
        }
        cancel.cancel();
    });
    Json(json!({ "restarting": true }))
}

#[cfg(test)]
mod tests {
    use crate::tests::*;
    use axum::http::StatusCode;
    use serde_json::json;

    #[tokio::test]
    async fn setup_status_done_flag_and_stale_count() {
        let dir = tempfile::tempdir().unwrap();
        let (st, app, cancel) = app(dir.path()).await;

        // fresh state: not done, no sources, no llm, no stale docs
        let (code, v) = get_auth(&app, "/api/v1/setup", TEST_BEARER).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["done"], false);
        assert_eq!(v["sources"], 0);
        assert_eq!(v["llmConfigured"], false);
        assert_eq!(v["staleDocs"], 0);
        assert_eq!(v["backend"], "hashing");

        // a doc stamped with the live embedder is not stale…
        index_test_doc(&st, "/a.md", "setup probe").await;
        let (_c, v) = get_auth(&app, "/api/v1/setup", TEST_BEARER).await;
        assert_eq!(v["staleDocs"], 0);
        // …but a foreign embedder stamp counts
        let mut rec = st.engine.meta.list_documents(None).unwrap()[0].clone();
        rec.metadata.insert("embedder".into(), "onnx-bge-m3".into());
        st.engine.meta.upsert_document(&rec).unwrap();
        let (_c, v) = get_auth(&app, "/api/v1/setup", TEST_BEARER).await;
        assert_eq!(v["staleDocs"], 1);

        // done flag round-trips
        let (code, _) = post_j(&app, "/api/v1/setup/done", json!({}), TEST_BEARER).await;
        assert_eq!(code, StatusCode::OK);
        let (_c, v) = get_auth(&app, "/api/v1/setup", TEST_BEARER).await;
        assert_eq!(v["done"], true);

        // non-admin → 403
        let tok = make_session(&st, "u-setup", "user");
        let (code, _) = get_auth(&app, "/api/v1/setup", &tok).await;
        assert_eq!(code, StatusCode::FORBIDDEN);
        cancel.cancel();
    }
}
