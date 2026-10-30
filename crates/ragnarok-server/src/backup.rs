//! Admin backup endpoints — consistent `VACUUM INTO` snapshots of the
//! whole metadata database (documents, chunks, blobs, FTS, users,
//! sessions, chat state) plus list/download/delete management.
//!
//! The vector store is intentionally excluded: it is a derived index and
//! can always be rebuilt from the indexed content via a global reindex.

use crate::auth::AdminAuth;
use crate::routes::{audit_admin, ApiError};
use crate::state::AppState;
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::Json;
use std::path::PathBuf;
use std::sync::Arc;

const PREFIX: &str = "ragnarok-backup-";

fn backups_dir(st: &AppState) -> PathBuf {
    st.engine
        .meta
        .path()
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("backups")
}

/// Validate a client-supplied backup filename: only files we created,
/// no path traversal.
fn safe_name(name: &str) -> Result<String, ApiError> {
    if name.starts_with(PREFIX)
        && name.ends_with(".db")
        && !name.contains(['/', '\\'])
        && !name.contains("..")
    {
        Ok(name.to_owned())
    } else {
        Err(ragnarok_core::Error::InvalidInput(format!("invalid backup name {name:?}")).into())
    }
}

fn entry(dir: &std::path::Path, name: &str) -> Option<serde_json::Value> {
    let md = std::fs::metadata(dir.join(name)).ok()?;
    Some(serde_json::json!({
        "name": name,
        "bytes": md.len(),
        "modified": md.modified().ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64),
    }))
}

/// `POST /api/v1/admin/backup` — snapshot the database to
/// `<data_dir>/backups/ragnarok-backup-<unix_ts>.db` and return its info.
pub async fn create(
    State(st): State<Arc<AppState>>,
    a: AdminAuth,
) -> Result<impl IntoResponse, ApiError> {
    if st.engine.meta.path().as_os_str() == ":memory:" {
        return Err(ragnarok_core::Error::InvalidInput(
            "backup requires a file-backed database".into(),
        )
        .into());
    }
    let dir = backups_dir(&st);
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let name = format!("{PREFIX}{ts}.db");
    let dest = dir.join(&name);
    st.engine.meta.backup_to(&dest)?;
    let bytes = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
    audit_admin(&st, &a, "backup.create", Some(&name), None);
    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({ "name": name, "bytes": bytes })),
    ))
}

/// `GET /api/v1/admin/backups` — newest first.
pub async fn list(
    State(st): State<Arc<AppState>>,
    _a: AdminAuth,
) -> Result<impl IntoResponse, ApiError> {
    let dir = backups_dir(&st);
    let mut items: Vec<serde_json::Value> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.flatten() {
            if let Some(name) = e.file_name().to_str() {
                if name.starts_with(PREFIX) && name.ends_with(".db") {
                    if let Some(v) = entry(&dir, name) {
                        items.push(v);
                    }
                }
            }
        }
    }
    items.sort_by(|a, b| b["name"].as_str().cmp(&a["name"].as_str()));
    Ok(Json(serde_json::json!({ "backups": items })))
}

/// `GET /api/v1/admin/backups/{name}` — download one snapshot file.
pub async fn download(
    State(st): State<Arc<AppState>>,
    Path(name): Path<String>,
    _a: AdminAuth,
) -> Result<impl IntoResponse, ApiError> {
    let name = safe_name(&name)?;
    let path = backups_dir(&st).join(&name);
    if !path.exists() {
        return Err(ragnarok_core::Error::InvalidInput(format!("backup {name} not found")).into());
    }
    let file = tokio::fs::File::open(&path)
        .await
        .map_err(ragnarok_core::Error::from)?;
    let stream = tokio_util::io::ReaderStream::new(file);
    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "application/octet-stream".to_owned()),
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"{name}\""),
            ),
        ],
        Body::from_stream(stream),
    ))
}

/// `DELETE /api/v1/admin/backups/{name}` — remove a snapshot file.
pub async fn delete(
    State(st): State<Arc<AppState>>,
    Path(name): Path<String>,
    a: AdminAuth,
) -> Result<impl IntoResponse, ApiError> {
    let name = safe_name(&name)?;
    let path = backups_dir(&st).join(&name);
    if !path.exists() {
        return Err(ragnarok_core::Error::InvalidInput(format!("backup {name} not found")).into());
    }
    std::fs::remove_file(&path).map_err(ragnarok_core::Error::from)?;
    audit_admin(&st, &a, "backup.delete", Some(&name), None);
    Ok(Json(serde_json::json!({ "deleted": name })))
}
