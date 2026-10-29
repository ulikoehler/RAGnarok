//! Multi-user authentication and authorization.
//!
//! * Passwords are hashed with **argon2id** (PHC string) — the hash never
//!   leaves [`MetadataDb`]; `UserRow` carries no secret material.
//! * Sessions are opaque bearer tokens (`Authorization: Bearer …` or a
//!   `?token=` query param for transports that cannot set headers, i.e.
//!   WebSocket/EventSource) with sliding expiry.
//! * [`SessionAuth`] is an axum extractor: every protected route declares
//!   it as a parameter and gets the resolved [`UserRow`]. [`AdminAuth`]
//!   additionally requires `role == "admin"`.
//! * [`bootstrap_admin`] creates the first admin on an empty user table —
//!   password from `RAGNAROK_ADMIN_PASSWORD` or a generated one logged once.

use crate::state::AppState;
use axum::extract::{FromRequestParts, Query, State};
use axum::http::request::Parts;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use ragnarok_core::{Error, Result};
use ragnarok_store::metadata::UserRow;
use ragnarok_store::MetadataDb;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const SESSION_TTL_SECS: i64 = 30 * 24 * 3600;

pub fn hash_password(pw: &str) -> Result<String> {
    use argon2::password_hash::{rand_core::OsRng, PasswordHasher, SaltString};
    let salt = SaltString::generate(&mut OsRng);
    argon2::Argon2::default()
        .hash_password(pw.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|e| Error::Internal(format!("argon2: {e}")))
}

pub fn verify_password(hash: &str, pw: &str) -> bool {
    use argon2::password_hash::{PasswordHash, PasswordVerifier};
    let Ok(parsed) = PasswordHash::new(hash) else {
        return false;
    };
    argon2::Argon2::default()
        .verify_password(pw.as_bytes(), &parsed)
        .is_ok()
}

fn new_token() -> String {
    use rand::RngCore;
    let mut b = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut b);
    hex::encode(b)
}

/// First-boot: with zero users, create `admin` so the daemon is usable.
/// `RAGNAROK_ADMIN_PASSWORD` wins; otherwise a random password is
/// generated and logged exactly once.
pub fn bootstrap_admin(meta: &MetadataDb) -> Result<()> {
    if meta.user_count()? > 0 {
        return Ok(());
    }
    let password = std::env::var("RAGNAROK_ADMIN_PASSWORD").unwrap_or_else(|_| {
        use rand::RngCore;
        let mut b = [0u8; 12];
        rand::thread_rng().fill_bytes(&mut b);
        hex::encode(b)
    });
    let from_env = std::env::var("RAGNAROK_ADMIN_PASSWORD").is_ok();
    meta.create_user(
        &uuid::Uuid::new_v4().to_string(),
        "admin",
        &hash_password(&password)?,
        "admin",
    )?;
    if from_env {
        tracing::info!("bootstrap admin 'admin' created (password from RAGNAROK_ADMIN_PASSWORD)");
    } else {
        tracing::warn!(password = %password, "bootstrap admin 'admin' created — store this password, it is shown once");
        eprintln!("BOOTSTRAP admin password: {password}");
    }
    Ok(())
}

// ------------------------------------------------------------------
// extractors
// ------------------------------------------------------------------

/// Authenticated user, resolved from `Authorization: Bearer <token>` or
/// a `?token=` query parameter (WS/SSE transports).
#[derive(Debug, Clone)]
pub struct SessionAuth(pub UserRow);

impl SessionAuth {
    pub fn is_admin(&self) -> bool {
        self.0.role == "admin"
    }
}

/// Role check: `SessionAuth` + `role == "admin"`.
#[derive(Debug, Clone)]
pub struct AdminAuth(pub UserRow);

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({"error": "authentication required"})),
    )
        .into_response()
}

fn forbidden() -> Response {
    (
        StatusCode::FORBIDDEN,
        Json(serde_json::json!({"error": "admin role required"})),
    )
        .into_response()
}

#[derive(Deserialize)]
pub struct TokenQuery {
    pub token: Option<String>,
}

/// `Authorization: Bearer <token>` — the auth scheme is case-insensitive
/// per RFC 7235 (`bearer`, `BEARER` are equally valid).
fn bearer_token(v: &header::HeaderValue) -> Option<&str> {
    let s = v.to_str().ok()?;
    let (scheme, rest) = s.split_once(' ')?;
    scheme
        .eq_ignore_ascii_case("bearer")
        .then(|| rest.trim())
        .filter(|t| !t.is_empty())
}

fn extract_token(parts: &mut Parts) -> Option<String> {
    if let Some(v) = parts
        .headers
        .get(header::AUTHORIZATION)
        .and_then(bearer_token)
    {
        return Some(v.to_string());
    }
    // query token fallback (WebSocket/EventSource can't set headers)
    parts
        .uri
        .query()
        .and_then(|q| serde_urlencoded_simple(q).get("token").cloned())
}

fn serde_urlencoded_simple(q: &str) -> std::collections::HashMap<String, String> {
    q.split('&')
        .filter_map(|kv| kv.split_once('='))
        .map(|(k, v)| (k.to_string(), urlencoding_decode(v)))
        .collect()
}

fn urlencoding_decode(s: &str) -> String {
    let mut out = Vec::with_capacity(s.len());
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' if i + 2 < b.len() => {
                let hex = std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or("");
                if let Ok(v) = u8::from_str_radix(hex, 16) {
                    out.push(v);
                    i += 3;
                } else {
                    out.push(b[i]);
                    i += 1;
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn user_for_token(state: &AppState, token: &str) -> Option<UserRow> {
    state.engine.meta.session_user(token).ok().flatten()
}

/// Test fixture: admin user + live session; returns the bearer token.
#[cfg(test)]
pub(crate) fn seed_test_session(state: &AppState) -> &'static str {
    const TOKEN: &str = "test-session-token";
    let meta = &state.engine.meta;
    if meta.user_by_id("u-test").ok().flatten().is_none() {
        meta.create_user("u-test", "tester", "x", "admin").unwrap();
        meta.create_session(TOKEN, "u-test", 3600).unwrap();
    }
    TOKEN
}

#[allow(clippy::result_large_err)]
fn resolve_session(parts: &mut Parts, app: &AppState) -> std::result::Result<UserRow, Response> {
    let token = extract_token(parts).ok_or_else(unauthorized)?;
    user_for_token(app, &token).ok_or_else(unauthorized)
}

// The extractors must be implemented for every router state type they
// run under: the main router's `Arc<AppState>` and the chat sub-router's
// `Arc<ChatState>` (which wraps the app state).

impl FromRequestParts<Arc<AppState>> for SessionAuth {
    type Rejection = Response;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &Arc<AppState>,
    ) -> std::result::Result<Self, Self::Rejection> {
        resolve_session(parts, state).map(SessionAuth)
    }
}

impl FromRequestParts<Arc<AppState>> for AdminAuth {
    type Rejection = Response;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &Arc<AppState>,
    ) -> std::result::Result<Self, Self::Rejection> {
        let user = resolve_session(parts, state)?;
        if user.role == "admin" {
            Ok(AdminAuth(user))
        } else {
            Err(forbidden())
        }
    }
}

impl FromRequestParts<Arc<crate::chat::ChatState>> for SessionAuth {
    type Rejection = Response;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &Arc<crate::chat::ChatState>,
    ) -> std::result::Result<Self, Self::Rejection> {
        resolve_session(parts, &state.app).map(SessionAuth)
    }
}

impl FromRequestParts<Arc<crate::chat::ChatState>> for AdminAuth {
    type Rejection = Response;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &Arc<crate::chat::ChatState>,
    ) -> std::result::Result<Self, Self::Rejection> {
        let user = resolve_session(parts, &state.app)?;
        if user.role == "admin" {
            Ok(AdminAuth(user))
        } else {
            Err(forbidden())
        }
    }
}

// ------------------------------------------------------------------
// routes
// ------------------------------------------------------------------

#[derive(Deserialize)]
pub struct LoginReq {
    pub username: String,
    pub password: String,
}

#[derive(Serialize)]
pub struct SessionOut {
    pub token: String,
    pub user: UserRow,
}

/// `POST /api/v1/auth/login` — public.
#[allow(clippy::result_large_err)]
pub async fn login(
    State(st): State<Arc<AppState>>,
    Json(req): Json<LoginReq>,
) -> std::result::Result<impl IntoResponse, Response> {
    let Some((user, pass_hash)) = st
        .engine
        .meta
        .user_by_username(&req.username)
        .map_err(|e| crate::routes::ApiError::from(e).into_response())?
    else {
        return Err(unauthorized());
    };
    if user.disabled || !verify_password(&pass_hash, &req.password) {
        return Err(unauthorized());
    }
    let token = new_token();
    st.engine
        .meta
        .create_session(&token, &user.id, SESSION_TTL_SECS)
        .map_err(|e| crate::routes::ApiError::from(e).into_response())?;
    Ok(Json(SessionOut { token, user }))
}

/// `POST /api/v1/auth/logout`
pub async fn logout(
    State(st): State<Arc<AppState>>,
    auth: SessionAuth,
    headers: axum::http::HeaderMap,
    Query(q): Query<TokenQuery>,
) -> impl IntoResponse {
    let token = headers
        .get(header::AUTHORIZATION)
        .and_then(bearer_token)
        .map(str::to_string)
        .or(q.token);
    if let Some(t) = token {
        let _ = st.engine.meta.delete_session(&t);
    }
    let _ = auth; // validated by the extractor already
    StatusCode::NO_CONTENT
}

/// `GET /api/v1/auth/me`
pub async fn me(auth: SessionAuth) -> Json<UserRow> {
    Json(auth.0)
}
