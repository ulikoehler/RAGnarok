//! Admin & filter management routes.
//!
//! ```text
//! GET    /api/v1/admin/users                      list users (admin)
//! POST   /api/v1/admin/users                      create user (admin)
//! PATCH  /api/v1/admin/users/{id}                 update role/disabled/password (admin)
//! DELETE /api/v1/admin/users/{id}                 delete user (admin)
//! GET    /api/v1/admin/users/{id}/grants          [(filterId, mode)] (admin)
//! PUT    /api/v1/admin/users/{id}/grants          replace grants (admin)
//! GET    /api/v1/filters                          own + granted filters (any user)
//! POST   /api/v1/filters                          create filter (any user)
//! PUT    /api/v1/filters/{id}                     update own filter
//! DELETE /api/v1/filters/{id}                     delete own filter
//! ```

use crate::auth::{hash_password, AdminAuth, SessionAuth};
use crate::routes::ApiError;
use crate::state::AppState;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use ragnarok_core::scope::ScopeFilter;
use ragnarok_core::Error;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Deserialize)]
pub struct NewUser {
    pub username: String,
    pub password: String,
    #[serde(default = "default_role")]
    pub role: String,
}

fn default_role() -> String {
    "user".into()
}

pub async fn list_users(
    State(st): State<Arc<AppState>>,
    _a: AdminAuth,
) -> std::result::Result<impl IntoResponse, ApiError> {
    Ok(Json(st.engine.meta.list_users()?))
}

pub async fn create_user(
    State(st): State<Arc<AppState>>,
    _a: AdminAuth,
    Json(req): Json<NewUser>,
) -> std::result::Result<impl IntoResponse, ApiError> {
    if req.username.trim().is_empty() || req.password.len() < 4 {
        return Err(Error::InvalidInput("username required, password ≥ 4 chars".into()).into());
    }
    if !matches!(req.role.as_str(), "admin" | "user") {
        return Err(Error::InvalidInput("role must be 'admin' or 'user'".into()).into());
    }
    if st.engine.meta.user_by_username(&req.username)?.is_some() {
        return Err(Error::InvalidInput("username already exists".into()).into());
    }
    let id = uuid::Uuid::new_v4().to_string();
    st.engine.meta.create_user(
        &id,
        req.username.trim(),
        &hash_password(&req.password)?,
        &req.role,
    )?;
    Ok((StatusCode::CREATED, Json(serde_json::json!({"id": id}))))
}

#[derive(Deserialize)]
pub struct UpdateUser {
    pub role: Option<String>,
    pub disabled: Option<bool>,
    pub password: Option<String>,
}

pub async fn update_user(
    State(st): State<Arc<AppState>>,
    a: AdminAuth,
    Path(id): Path<String>,
    Json(req): Json<UpdateUser>,
) -> std::result::Result<impl IntoResponse, ApiError> {
    st.engine
        .meta
        .user_by_id(&id)?
        .ok_or_else(|| Error::InvalidInput(format!("no such user: {id}")))?;
    if let Some(r) = &req.role {
        if !matches!(r.as_str(), "admin" | "user") {
            return Err(Error::InvalidInput("role must be 'admin' or 'user'".into()).into());
        }
        // an admin must not demote/disable themselves out of the last admin seat
        if id == a.0.id && r != "admin" {
            return Err(Error::InvalidInput("cannot demote your own admin account".into()).into());
        }
    }
    if req.disabled == Some(true) && id == a.0.id {
        return Err(Error::InvalidInput("cannot disable your own account".into()).into());
    }
    let pass_hash = match &req.password {
        Some(p) if p.len() >= 4 => Some(hash_password(p)?),
        Some(_) => {
            return Err(Error::InvalidInput("password ≥ 4 chars".into()).into());
        }
        None => None,
    };
    st.engine
        .meta
        .update_user(&id, req.role.as_deref(), req.disabled, pass_hash.as_deref())?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn delete_user(
    State(st): State<Arc<AppState>>,
    a: AdminAuth,
    Path(id): Path<String>,
) -> std::result::Result<impl IntoResponse, ApiError> {
    if id == a.0.id {
        return Err(Error::InvalidInput("cannot delete your own account".into()).into());
    }
    st.engine.meta.delete_user(&id)?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn get_grants(
    State(st): State<Arc<AppState>>,
    _a: AdminAuth,
    Path(id): Path<String>,
) -> std::result::Result<impl IntoResponse, ApiError> {
    Ok(Json(st.engine.meta.grants_for_user(&id)?))
}

pub async fn set_grants(
    State(st): State<Arc<AppState>>,
    _a: AdminAuth,
    Path(id): Path<String>,
    Json(grants): Json<Vec<(String, String)>>,
) -> std::result::Result<impl IntoResponse, ApiError> {
    for (_, mode) in &grants {
        if !matches!(mode.as_str(), "allow" | "deny") {
            return Err(Error::InvalidInput("grant mode must be 'allow' or 'deny'".into()).into());
        }
    }
    st.engine
        .meta
        .user_by_id(&id)?
        .ok_or_else(|| Error::InvalidInput(format!("no such user: {id}")))?;
    st.engine.meta.set_grants(&id, &grants)?;
    Ok(StatusCode::NO_CONTENT)
}

// ------------------------------------------------------------------
// filters (user-facing CRUD)
// ------------------------------------------------------------------

#[derive(Deserialize)]
pub struct FilterReq {
    pub name: String,
    /// The `ScopeFilter` JSON object (include/exclude expression).
    pub expr: serde_json::Value,
}

fn parse_expr(v: &serde_json::Value) -> std::result::Result<ScopeFilter, ApiError> {
    let f = ScopeFilter::from_value(v)
        .map_err(|e| ApiError::from(Error::InvalidInput(format!("bad scope filter: {e}"))))?;
    // compile once here so bad globs are rejected at write time
    f.compile()
        .map_err(|e| ApiError::from(Error::InvalidInput(format!("bad scope filter: {e}"))))?;
    Ok(f)
}

pub async fn list_filters(
    State(st): State<Arc<AppState>>,
    auth: SessionAuth,
) -> std::result::Result<impl IntoResponse, ApiError> {
    let uid = (!auth.is_admin()).then_some(auth.0.id.as_str());
    Ok(Json(st.engine.meta.filters_for_user(uid)?))
}

pub async fn create_filter(
    State(st): State<Arc<AppState>>,
    auth: SessionAuth,
    Json(req): Json<FilterReq>,
) -> std::result::Result<impl IntoResponse, ApiError> {
    if req.name.trim().is_empty() {
        return Err(Error::InvalidInput("filter name required".into()).into());
    }
    parse_expr(&req.expr)?;
    let id = uuid::Uuid::new_v4().to_string();
    st.engine.meta.create_filter(
        &id,
        req.name.trim(),
        Some(&auth.0.id),
        &req.expr.to_string(),
    )?;
    Ok((StatusCode::CREATED, Json(serde_json::json!({"id": id}))))
}

pub async fn update_filter(
    State(st): State<Arc<AppState>>,
    auth: SessionAuth,
    Path(id): Path<String>,
    Json(req): Json<FilterReq>,
) -> std::result::Result<impl IntoResponse, ApiError> {
    let row = st
        .engine
        .meta
        .get_filter(&id)?
        .ok_or_else(|| Error::InvalidInput(format!("no such filter: {id}")))?;
    if row.owner_id.as_deref() != Some(auth.0.id.as_str()) && !auth.is_admin() {
        return Err(Error::InvalidInput("not your filter".into()).into());
    }
    parse_expr(&req.expr)?;
    st.engine
        .meta
        .update_filter(&id, req.name.trim(), &req.expr.to_string())?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn delete_filter(
    State(st): State<Arc<AppState>>,
    auth: SessionAuth,
    Path(id): Path<String>,
) -> std::result::Result<impl IntoResponse, ApiError> {
    let row = st
        .engine
        .meta
        .get_filter(&id)?
        .ok_or_else(|| Error::InvalidInput(format!("no such filter: {id}")))?;
    if row.owner_id.as_deref() != Some(auth.0.id.as_str()) && !auth.is_admin() {
        return Err(Error::InvalidInput("not your filter".into()).into());
    }
    st.engine.meta.delete_filter(&id)?;
    Ok(StatusCode::NO_CONTENT)
}
