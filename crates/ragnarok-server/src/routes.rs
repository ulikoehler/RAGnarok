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
use tokio::sync::broadcast;

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

/// `GET /api/v1/settings/runtime` — the live runtime tunables, read back
/// from the actual components (persisted overrides already applied).
pub async fn get_runtime(State(st): State<Arc<AppState>>, _a: SessionAuth) -> impl IntoResponse {
    Json(st.pipeline.runtime_config())
}

/// `PUT /api/v1/settings/runtime` — validate, persist as the `runtime`
/// JSON blob (survives restart) and apply to the running pipeline:
/// chunker bounds, file cap, OCR toggle/language, pressure thresholds.
pub async fn put_runtime(
    State(st): State<Arc<AppState>>,
    _a: AdminAuth,
    Json(rt): Json<ragnarok_core::RuntimeConfig>,
) -> std::result::Result<impl IntoResponse, ApiError> {
    rt.validate()?;
    st.engine.meta.set_setting(
        "runtime",
        &serde_json::to_string(&rt).map_err(|e| Error::Metadata(e.to_string()))?,
    )?;
    st.pipeline.apply_runtime(&rt);
    Ok(Json(rt))
}

/// `POST /api/v1/sources/{id}/poll` — change a source's poll interval
/// (seconds, ≥30). Persisted into `config_json` AND applied to the live
/// poller, which re-reads the override every cycle.
pub async fn set_source_poll(
    State(st): State<Arc<AppState>>,
    _a: AdminAuth,
    Path(id): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> std::result::Result<impl IntoResponse, ApiError> {
    let secs = body
        .get("secs")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| Error::InvalidInput("body needs {\"secs\": <u64>}".into()))?;
    if secs < 30 {
        return Err(Error::InvalidInput("poll interval must be >= 30s".into()).into());
    }
    let sid = ragnarok_core::SourceId::from(id.as_str());
    let mut source = st
        .engine
        .meta
        .list_sources()?
        .into_iter()
        .find(|s| s.id == sid)
        .ok_or_else(|| Error::SourceNotFound(id.clone()))?;
    match &mut source.config {
        SourceConfig::Imap {
            poll_interval_secs, ..
        }
        | SourceConfig::Paperless {
            poll_interval_secs, ..
        }
        | SourceConfig::WebDav {
            poll_interval_secs, ..
        }
        | SourceConfig::Web {
            poll_interval_secs, ..
        }
        | SourceConfig::Feed {
            poll_interval_secs, ..
        }
        | SourceConfig::S3 {
            poll_interval_secs, ..
        }
        | SourceConfig::Git {
            poll_interval_secs, ..
        }
        | SourceConfig::CalDav {
            poll_interval_secs, ..
        } => {
            *poll_interval_secs = secs;
        }
        SourceConfig::FileSystem { .. } => {
            return Err(Error::InvalidInput(
                "file_system sources watch continuously — no poll interval".into(),
            )
            .into());
        }
    }
    st.engine.meta.upsert_source(&source)?;
    st.pipeline.set_poll_interval(&sid, secs);
    Ok(Json(serde_json::json!({"poll_interval_secs": secs})))
}

/// Resolve the caller's effective document scope for a request:
/// `granted(user) AND scope(apiKey) AND scope(filterId) AND scope(inline)`
/// — every layer can only narrow. `key_scope` is the scope JSON bound to
/// an API key (`SessionAuth.1`); `filterId` must be a filter the user
/// owns or was granted; unknown/inaccessible ids are a 400.
pub fn effective_search_scope(
    meta: &MetadataDb,
    user: &UserRow,
    key_scope: Option<&str>,
    filter_id: Option<&str>,
    scope: Option<&ScopeFilter>,
) -> ragnarok_core::Result<Permitted> {
    let is_admin = user.role == "admin";
    let mut perm = ragnarok_store::user_permitted(meta, &user.id, is_admin)?;
    if let Some(raw) = key_scope {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) {
            if let Ok(f) = ScopeFilter::from_value(&v) {
                if !f.is_unrestricted() {
                    perm = perm.combine(&ragnarok_store::resolve_permitted(meta, &f)?);
                }
            }
        }
    }
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
        auth.1.as_deref(),
        req.filter_id.as_deref(),
        req.scope.as_ref(),
    )?;
    let hits = st.engine.search_scoped(&req, &perm).await?;
    Ok(Json(hits))
}

/// `GET /api/v1/documents` — browsable index inventory for the
/// documents page. Scope-filtered like search: a user only sees docs
/// their grants permit. `total` counts *after* the scope filter.
#[derive(Debug, Deserialize)]
pub struct DocListQuery {
    pub source: Option<String>,
    #[serde(rename = "type")]
    pub doc_type: Option<String>,
    pub q: Option<String>,
    pub sort: Option<String>,
    pub dir: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

pub async fn list_documents(
    State(st): State<Arc<AppState>>,
    auth: SessionAuth,
    axum::extract::Query(q): axum::extract::Query<DocListQuery>,
) -> std::result::Result<impl IntoResponse, ApiError> {
    let perm = effective_search_scope(&st.engine.meta, &auth.0, auth.1.as_deref(), None, None)?;
    let sid: Option<ragnarok_core::SourceId> =
        q.source.as_deref().map(ragnarok_core::SourceId::from);
    let mut docs = st
        .engine
        .meta
        .list_documents_filtered(&ragnarok_store::DocumentListFilter {
            source_id: sid.as_ref(),
            doc_type: q.doc_type.as_deref(),
            query: q.q.as_deref(),
            sort: q.sort.as_deref().unwrap_or("path"),
            desc: q.dir.as_deref() == Some("desc"),
        })?;
    if !perm.is_unrestricted() {
        docs.retain(|d| perm.allows(&d.id.to_string()));
    }
    let total = docs.len();
    let offset = q.offset.unwrap_or(0);
    let limit = q.limit.unwrap_or(50).min(200);
    let names: std::collections::HashMap<String, String> = st
        .engine
        .meta
        .list_sources()?
        .into_iter()
        .map(|s| (s.id.to_string(), s.name))
        .collect();
    let items: Vec<serde_json::Value> = docs
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(|d| {
            serde_json::json!({
                "id": d.id,
                "sourceId": d.source_id,
                "source": names.get(&d.source_id.to_string()).cloned().unwrap_or_default(),
                "path": d.path,
                "docType": d.doc_type,
                "size": d.size,
                "modified": d.modified,
                "indexedAt": d.indexed_at,
                "chunkCount": d.chunk_count,
                "metadata": d.metadata,
            })
        })
        .collect();
    Ok(Json(serde_json::json!({ "items": items, "total": total })))
}

/// `GET /api/v1/documents/{id}/chunks` — chunk list of one document for
/// the detail view. Scoped like the stream: out-of-scope = 404.
pub async fn list_document_chunks(
    State(st): State<Arc<AppState>>,
    auth: SessionAuth,
    Path(id): Path<String>,
) -> std::result::Result<impl IntoResponse, ApiError> {
    let doc_id: DocumentId = id
        .parse()
        .map_err(|_| Error::InvalidInput(format!("bad document id: {id}")))?;
    let perm = effective_search_scope(&st.engine.meta, &auth.0, auth.1.as_deref(), None, None)?;
    if !perm.allows(&doc_id.to_string()) {
        return Err(Error::DocumentNotFound(id).into());
    }
    if st.engine.meta.get_document(doc_id)?.is_none() {
        return Err(Error::DocumentNotFound(id).into());
    }
    let chunks: Vec<serde_json::Value> = st
        .engine
        .meta
        .chunks_for_document(doc_id)?
        .into_iter()
        .map(|c| {
            serde_json::json!({
                "id": c.id,
                "chunkIndex": c.chunk_index,
                "page": c.page,
                "bbox": c.bbox,
                "lineRange": c.line_range,
                "text": c.text,
                "tokenCount": c.token_count,
            })
        })
        .collect();
    Ok(Json(serde_json::json!({ "chunks": chunks })))
}

/// `DELETE /api/v1/documents/{id}` — drop a document from metadata and
/// the vector index (admin). A still-existing file is re-picked on the
/// next scan/watch cycle — that is intentional reindex-on-rescan.
pub async fn delete_document(
    State(st): State<Arc<AppState>>,
    _a: AdminAuth,
    Path(id): Path<String>,
) -> std::result::Result<impl IntoResponse, ApiError> {
    let doc_id: DocumentId = id
        .parse()
        .map_err(|_| Error::InvalidInput(format!("bad document id: {id}")))?;
    let rec = st
        .engine
        .meta
        .get_document(doc_id)?
        .ok_or_else(|| Error::DocumentNotFound(id.clone()))?;
    st.engine.delete_document(doc_id)?;
    let _ = st.events.send(ragnarok_core::TelemetryEvent::Ingest {
        source_id: rec.source_id.to_string(),
        path: rec.path.clone(),
        status: ragnarok_core::IngestStatus::Deleted,
    });
    Ok(StatusCode::NO_CONTENT)
}

/// `POST /api/v1/documents/{id}/reindex` — re-parse + re-embed one
/// document now (admin). Blobs (IMAP) re-parse the stored source, FS
/// docs re-read disk, remote docs re-fetch from the source system.
pub async fn reindex_document(
    State(st): State<Arc<AppState>>,
    _a: AdminAuth,
    Path(id): Path<String>,
) -> std::result::Result<impl IntoResponse, ApiError> {
    let doc_id: DocumentId = id
        .parse()
        .map_err(|_| Error::InvalidInput(format!("bad document id: {id}")))?;
    let rec = st
        .engine
        .meta
        .get_document(doc_id)?
        .ok_or_else(|| Error::DocumentNotFound(id.clone()))?;
    let outcome = if let Some(blob) = st.engine.meta.get_blob(doc_id)? {
        st.pipeline
            .ingest_virtual(&rec.source_id, &rec.path, &blob)
            .await?
    } else if std::path::Path::new(&rec.path).is_file() {
        st.pipeline
            .ingest_path(&rec.source_id, std::path::Path::new(&rec.path))
            .await?
    } else {
        // remote original — rebuild a RemoteDoc from the record; etag
        // left empty so the content hash decides on re-embedding
        let src = st
            .engine
            .meta
            .list_sources()?
            .into_iter()
            .find(|s| s.id == rec.source_id)
            .ok_or_else(|| Error::SourceNotFound(rec.source_id.to_string()))?;
        let doc = ragnarok_crawler::remote::RemoteDoc {
            vpath: rec.path.clone(),
            title: rec.metadata.get("title").cloned().unwrap_or_default(),
            mime: rec
                .metadata
                .get("mime")
                .cloned()
                .unwrap_or_else(|| mime_for_path(&rec.path)),
            modified: rec.modified,
            etag: None,
            size: Some(rec.size),
            ocr_text: None,
            metadata: rec.metadata.clone(),
        };
        st.pipeline
            .ingest_remote(&rec.source_id, &src.config, &doc)
            .await?
    };
    Ok(Json(
        serde_json::json!({ "outcome": format!("{outcome:?}").to_lowercase() }),
    ))
}

/// `GET /api/v1/inference` — active embedding backend, whether the binary
/// was compiled with `onnx`, and the current model provisioning status
/// (admin — displayed on the settings hub's Inference tab).
pub async fn inference_status(State(st): State<Arc<AppState>>, _a: AdminAuth) -> impl IntoResponse {
    let model = st
        .provisioner
        .as_ref()
        .map(|p| p.status().read().unwrap().clone());
    let reranker_model = st
        .reranker_provisioner
        .as_ref()
        .map(|p| p.status().read().unwrap().clone());
    Json(serde_json::json!({
        "backend": st.inference_backend,
        "onnxCompiled": ragnarok_inference::ONNX_COMPILED,
        "model": model,
        "rerank": {
            "enabled": st.inference_rerank,
            "model": reranker_model,
        },
    }))
}

/// Shared spawn for `/inference/*/download` — checks the state, starts a
/// background `download()` on the given provisioner, returns the state.
fn spawn_download(
    p: &Arc<ragnarok_inference::Provisioner>,
    events: &broadcast::Sender<ragnarok_core::TelemetryEvent>,
    cancel: tokio_util::sync::CancellationToken,
) -> std::result::Result<serde_json::Value, ApiError> {
    p.refresh_state();
    let state = p.status().read().unwrap().state;
    match state {
        ragnarok_inference::ModelState::Ready => {
            return Ok(serde_json::json!({ "state": "ready" }))
        }
        ragnarok_inference::ModelState::Downloading => {
            return Err(Error::InvalidInput("download already running".into()).into())
        }
        _ => {}
    }
    let pc = p.clone();
    let ev = events.clone();
    tokio::spawn(async move {
        if let Err(e) = pc.download(Some(&ev), cancel).await {
            tracing::warn!("model provisioning failed: {e}");
        }
    });
    Ok(serde_json::json!({ "state": "downloading" }))
}

/// `POST /api/v1/inference/model/download` — provision the ONNX model in
/// the background (admin). Progress streams over the telemetry WS as
/// `model_download` events; status is polled via `GET /inference`.
pub async fn download_model(
    State(st): State<Arc<AppState>>,
    _a: AdminAuth,
) -> std::result::Result<impl IntoResponse, ApiError> {
    let p = st
        .provisioner
        .clone()
        .ok_or_else(|| Error::InvalidInput("no provisioner configured".into()))?;
    Ok(Json(spawn_download(
        &p,
        &st.events,
        st.cancel.child_token(),
    )?))
}

/// `POST /api/v1/inference/reranker/download` — same for the pinned
/// `bge-reranker-base` cross-encoder used by the post-RRF rerank stage.
pub async fn download_reranker(
    State(st): State<Arc<AppState>>,
    _a: AdminAuth,
) -> std::result::Result<impl IntoResponse, ApiError> {
    let p = st
        .reranker_provisioner
        .clone()
        .ok_or_else(|| Error::InvalidInput("no provisioner configured".into()))?;
    Ok(Json(spawn_download(
        &p,
        &st.events,
        st.cancel.child_token(),
    )?))
}

/// `PUT /api/v1/inference` — persist a backend choice (`hashing`|`onnx`)
/// and/or the rerank toggle to the daemon config file (admin). Both
/// activate on the next restart — embeddings are model-specific, a
/// hot-swap would corrupt the index.
#[derive(Debug, Deserialize)]
pub struct InferencePut {
    #[serde(default)]
    pub backend: Option<ragnarok_core::InferenceBackend>,
    #[serde(default)]
    pub rerank: Option<bool>,
}

pub async fn put_inference(
    State(st): State<Arc<AppState>>,
    _a: AdminAuth,
    Json(req): Json<InferencePut>,
) -> std::result::Result<impl IntoResponse, ApiError> {
    if req.backend == Some(ragnarok_core::InferenceBackend::Onnx)
        && !ragnarok_inference::ONNX_COMPILED
    {
        return Err(Error::InvalidInput(
            "binary built without `onnx` — rebuild with --features onnx".into(),
        )
        .into());
    }
    if req.rerank == Some(true) && !ragnarok_inference::ONNX_COMPILED {
        return Err(Error::InvalidInput(
            "reranker needs the `onnx` feature — rebuild with --features onnx".into(),
        )
        .into());
    }
    let path = st
        .config_path
        .clone()
        .ok_or_else(|| Error::InvalidInput("no config file in use".into()))?;
    let mut cfg = ragnarok_core::RagnarokConfig::load_or_default(&path)?;
    if let Some(b) = req.backend {
        cfg.inference.backend = b;
    }
    if let Some(r) = req.rerank {
        cfg.inference.rerank = r;
    }
    cfg.save(&path)?;
    Ok(Json(serde_json::json!({
        "ok": true,
        "backend": cfg.inference.backend,
        "rerank": cfg.inference.rerank,
        "needsRestart": req.backend.is_some_and(|b| b != st.inference_backend)
            || req.rerank.is_some_and(|r| r != st.inference_rerank),
    })))
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
    let perm = effective_search_scope(&st.engine.meta, &auth.0, auth.1.as_deref(), None, None)?;
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
