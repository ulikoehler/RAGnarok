//! REST + SPA handlers (spec §4.1).

use crate::auth::{AdminAuth, SessionAuth};
use crate::state::AppState;
use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use ragnarok_core::scope::{Permitted, ScopeFilter};
use ragnarok_core::{DocumentId, Error, SearchRequest, Source, SourceConfig, SystemStats};
use ragnarok_store::metadata::UserRow;
use ragnarok_store::MetadataDb;
use serde::Deserialize;
use std::sync::Arc;

/// Unified API error -> JSON body + status code.
pub struct ApiError(Error);

impl From<Error> for ApiError {
    fn from(e: Error) -> Self {
        ApiError(e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match &self.0 {
            Error::DocumentNotFound(_) | Error::SourceNotFound(_) => StatusCode::NOT_FOUND,
            Error::InvalidInput(_) | Error::UnsupportedType(_) => StatusCode::BAD_REQUEST,
            Error::ResourceExhausted(_) => StatusCode::SERVICE_UNAVAILABLE,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (
            status,
            Json(serde_json::json!({ "error": self.0.to_string() })),
        )
            .into_response()
    }
}

/// `GET /api/v1/system/stats`
pub async fn get_stats(State(st): State<Arc<AppState>>, _a: SessionAuth) -> Json<SystemStats> {
    st.monitor.sample_once(); // force a fresh sample
    Json(st.stats())
}

/// `GET /api/v1/sources`
pub async fn list_sources(State(st): State<Arc<AppState>>, _a: SessionAuth) -> impl IntoResponse {
    Json(st.sources_with_state())
}

/// `POST /api/v1/sources` — register and immediately start crawling.
#[derive(Debug, Deserialize)]
pub struct NewSource {
    pub name: String,
    pub config: SourceConfig,
}

pub async fn add_source(
    State(st): State<Arc<AppState>>,
    _a: AdminAuth,
    Json(req): Json<NewSource>,
) -> std::result::Result<impl IntoResponse, ApiError> {
    if let SourceConfig::FileSystem { path, .. } = &req.config {
        if !path.is_dir() {
            return Err(Error::InvalidInput(format!("not a directory: {}", path.display())).into());
        }
    }
    let source = Source::new(req.name, req.config);
    let id = source.id.clone();
    st.engine.meta.upsert_source(&source)?;
    st.pipeline
        .spawn_all(vec![source.clone()], st.cancel.clone())
        .await;
    Ok((StatusCode::CREATED, Json(serde_json::json!({ "id": id }))))
}

/// `DELETE /api/v1/sources/{id}` — stop the crawl task, unregister the
/// source and remove its indexed documents (metadata + vectors). Without
/// this a deleted source's poller/watcher kept running as a zombie and
/// its documents stayed searchable forever.
pub async fn delete_source(
    State(st): State<Arc<AppState>>,
    _a: AdminAuth,
    Path(id): Path<String>,
) -> std::result::Result<impl IntoResponse, ApiError> {
    let sid = ragnarok_core::SourceId::from(id.as_str());
    let exists = st.engine.meta.list_sources()?.iter().any(|s| s.id == sid);
    if !exists {
        return Err(Error::SourceNotFound(id).into());
    }
    st.pipeline.stop_source(&sid);
    st.engine.meta.delete_source(&sid)?;
    // purge the source's documents from metadata + vector store —
    // deterministic ids make this exact
    for doc in st.engine.meta.list_documents(Some(&sid))? {
        let _ = st.engine.delete_document(doc.id);
    }
    Ok(StatusCode::NO_CONTENT)
}

/// `POST /api/v1/sources/{id}/rescan` — restart the crawl task.
/// Filesystem sources re-scan; remote sources run an immediate
/// inventory-diff cycle in addition to their poll interval.
pub async fn rescan_source(
    State(st): State<Arc<AppState>>,
    _a: AdminAuth,
    Path(id): Path<String>,
) -> std::result::Result<impl IntoResponse, ApiError> {
    let sid = ragnarok_core::SourceId::from(id.as_str());
    let source = st
        .engine
        .meta
        .list_sources()?
        .into_iter()
        .find(|s| s.id == sid)
        .ok_or_else(|| Error::SourceNotFound(id.clone()))?;
    match source.config {
        SourceConfig::Imap { .. } => {
            Err(Error::InvalidInput("imap sources poll on their own interval".into()).into())
        }
        _ => {
            st.pipeline
                .spawn_all(vec![source], st.cancel.child_token())
                .await;
            Ok(StatusCode::ACCEPTED)
        }
    }
}

/// Resolve the caller's effective document scope for a request:
/// `granted(user) AND scope(filterId) AND scope(inline)` — every layer
/// can only narrow. `filterId` must be a filter the user owns or was
/// granted; unknown/inaccessible ids are a 400.
pub fn effective_search_scope(
    meta: &MetadataDb,
    user: &UserRow,
    filter_id: Option<&str>,
    scope: Option<&ScopeFilter>,
) -> ragnarok_core::Result<Permitted> {
    let is_admin = user.role == "admin";
    let mut perm = ragnarok_store::user_permitted(meta, &user.id, is_admin)?;
    if let Some(fid) = filter_id {
        let uid = (!is_admin).then_some(user.id.as_str());
        let row = meta
            .filters_for_user(uid)?
            .into_iter()
            .find(|f| f.id == fid)
            .ok_or_else(|| Error::InvalidInput(format!("unknown or inaccessible filter: {fid}")))?;
        let expr: serde_json::Value = serde_json::from_str(&row.expr)
            .map_err(|e| Error::Config(format!("stored filter '{}': {e}", row.name)))?;
        let f = ScopeFilter::from_value(&expr)?;
        perm = perm.combine(&ragnarok_store::resolve_permitted(meta, &f)?);
    }
    if let Some(f) = scope {
        if !f.is_unrestricted() {
            perm = perm.combine(&ragnarok_store::resolve_permitted(meta, f)?);
        }
    }
    Ok(perm)
}

/// `POST /api/v1/search` — scoped to the caller's permissions; `filterId`
/// and inline `scope` narrow further.
pub async fn search(
    State(st): State<Arc<AppState>>,
    auth: SessionAuth,
    Json(req): Json<SearchRequest>,
) -> std::result::Result<impl IntoResponse, ApiError> {
    if req.query.trim().is_empty() {
        return Err(Error::InvalidInput("empty query".into()).into());
    }
    let perm = effective_search_scope(
        &st.engine.meta,
        &auth.0,
        req.filter_id.as_deref(),
        req.scope.as_ref(),
    )?;
    let hits = st.engine.search_scoped(&req, &perm).await?;
    Ok(Json(hits))
}

/// `GET /api/v1/documents/{document_id}/stream`
///
/// Streams the raw document bytes: from the filesystem for FS sources,
/// from the SQLite blob table for virtual documents (IMAP messages).
/// Out-of-scope documents answer 404 — the id must not leak existence.
pub async fn stream_document(
    State(st): State<Arc<AppState>>,
    auth: SessionAuth,
    Path(id): Path<String>,
) -> std::result::Result<impl IntoResponse, ApiError> {
    let doc_id: DocumentId = id
        .parse()
        .map_err(|_| Error::InvalidInput(format!("bad document id: {id}")))?;
    let perm = effective_search_scope(&st.engine.meta, &auth.0, None, None)?;
    if !perm.allows(&doc_id.to_string()) {
        return Err(Error::DocumentNotFound(id).into());
    }
    let rec = st
        .engine
        .meta
        .get_document(doc_id)?
        .ok_or_else(|| Error::DocumentNotFound(id.clone()))?;

    let (mime, bytes): (String, Vec<u8>) = if let Some(blob) = st.engine.meta.get_blob(doc_id)? {
        (mime_for_path(&rec.path), blob)
    } else {
        let p = std::path::Path::new(&rec.path);
        match tokio::fs::read(p).await {
            Ok(b) => (mime_for_path(&rec.path), b),
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(Error::Io(e).into()),
            Err(_) => {
                // Not on disk — remote sources serve originals on demand.
                let src = st
                    .engine
                    .meta
                    .list_sources()?
                    .into_iter()
                    .find(|s| s.id == rec.source_id)
                    .ok_or_else(|| Error::DocumentNotFound(id.clone()))?;
                let (b, remote_mime) =
                    ragnarok_crawler::remote::fetch_for_record(&src.config, &rec)
                        .await
                        .map_err(|e| match e {
                            Error::Config(_) | Error::Internal(_) => {
                                Error::DocumentNotFound(id.clone())
                            }
                            other => other,
                        })?;
                (remote_mime, b)
            }
        }
    };
    Ok((
        [
            (header::CONTENT_TYPE, inline_mime(&mime)),
            (header::CONTENT_DISPOSITION, "inline".to_string()),
            // remote HTML documents are served inline — the sandbox CSP
            // opaque-origin the document so a stored page cannot run
            // script in the app origin (it would read the session token)
            (header::CONTENT_SECURITY_POLICY, "sandbox".to_string()),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff".to_string()),
        ],
        bytes,
    ))
}

fn mime_for_path(path: &str) -> String {
    let clean = path.split('?').next().unwrap_or(path);
    // virtual imap://…/N.eml paths resolve through the extension as well
    mime_guess::from_path(clean)
        .first_or_octet_stream()
        .to_string()
}

/// Adjusts a fetched/guessed mime for inline display: browsers cannot
/// render `message/rfc822` (blank iframe) and guess Latin-1 for bare
/// `text/*` — mails show their raw source as text, UTF-8 text keeps its
/// encoding.
fn inline_mime(mime: &str) -> String {
    let base = mime.split(';').next().unwrap_or(mime).trim();
    match base {
        "message/rfc822" | "application/vnd.ms-outlook" => "text/plain; charset=utf-8".to_string(),
        b if b.starts_with("text/") && !mime.contains("charset") => {
            format!("{b}; charset=utf-8")
        }
        _ => mime.to_string(),
    }
}

/// SPA handler: embedded assets with `index.html` fallback for client
/// routes.
pub async fn spa(Path(path): Path<String>) -> impl IntoResponse {
    serve_asset(&path)
}

pub async fn spa_root() -> impl IntoResponse {
    serve_asset("")
}

fn serve_asset(path: &str) -> Response {
    if let Some((bytes, mime)) = ragnarok_ui::asset(path) {
        return ([(header::CONTENT_TYPE, mime)], bytes).into_response();
    }
    match ragnarok_ui::index() {
        Some((bytes, mime)) => ([(header::CONTENT_TYPE, mime)], bytes).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::inline_mime;

    #[test]
    fn inline_mime_adjusts_for_display() {
        assert_eq!(inline_mime("message/rfc822"), "text/plain; charset=utf-8");
        assert_eq!(
            inline_mime("application/vnd.ms-outlook"),
            "text/plain; charset=utf-8"
        );
        assert_eq!(inline_mime("text/markdown"), "text/markdown; charset=utf-8");
        assert_eq!(
            inline_mime("text/plain; charset=latin-1"),
            "text/plain; charset=latin-1"
        );
        assert_eq!(inline_mime("application/pdf"), "application/pdf");
    }
}
