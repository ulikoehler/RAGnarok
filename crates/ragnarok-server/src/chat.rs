//! `/api/v1/chat/*` — thread/message persistence plus the SSE surface of
//! the agent loop. Port of RiskMaster's chat handler: claim-then-run with
//! a steal window, `tee_run` replay for late attachers, `pending_state`
//! parking (`awaiting_continue` → `/continue`), cooperative stop
//! (`cancel_requested`) and mid-run steering (`steer_queue`).

// Handlers return `Response` as the error type; it is large but the error
// path is cold and boxing would only obscure the handler signatures.
#![allow(clippy::result_large_err)]

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};

use crate::agent::{self, AgentResume};
use crate::auth::{AdminAuth, SessionAuth};
use crate::state::AppState;

/// Seconds a stale run claim may be reclaimed (covers crashed processes).
const CLAIM_STALE_SECS: i64 = 15 * 60;
const DEFAULT_CHAT_MODEL: &str = "gpt-4o-mini";

type LiveMap = Arc<Mutex<HashMap<String, Arc<Mutex<LiveRun>>>>>;

/// Frames of an in-flight run, replayed to late `…/stream` attachers.
struct LiveRun {
    frames: Vec<Bytes>,
    done: bool,
    notify: tokio::sync::watch::Sender<u64>,
}

#[derive(Clone)]
pub struct ChatState {
    pub app: Arc<AppState>,
    live: LiveMap,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateThread {
    title: Option<String>,
    model: Option<String>,
    reasoning_effort: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PatchThread {
    /// `None` = field absent → keep; `Some(None)` = explicit `null` →
    /// clear to global; `Some(Some(v))` = set.
    #[serde(default)]
    model: Option<Option<String>>,
    #[serde(default)]
    reasoning_effort: Option<Option<String>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PostMessage {
    content: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SteerRequest {
    content: String,
}

#[derive(Deserialize)]
struct ListQuery {
    q: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChatSettingsRequest {
    base_url: String,
    api_key: Option<String>,
    model: Option<String>,
    reasoning_effort: Option<String>,
    max_steps: Option<i64>,
    context_tokens: Option<i64>,
    chat_max_tokens: Option<i64>,
    provider_max_tokens: Option<i64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ThreadDto {
    id: String,
    title: String,
    created_at: String,
    updated_at: String,
    running: bool,
    /// A run paused at the step budget (or on stop) waits for /continue.
    awaiting: bool,
    message_count: i64,
    last_message: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MessageDto {
    id: String,
    seq: i64,
    role: String,
    content: String,
    reasoning: String,
    model: String,
    created_at: String,
}

fn err(status: StatusCode, code: &str, msg: &str) -> Response {
    (
        status,
        Json(json!({"error": {"code": code, "message": msg}})),
    )
        .into_response()
}

fn ts(unix: i64) -> String {
    chrono::DateTime::from_timestamp(unix, 0)
        .map(|t| t.to_rfc3339())
        .unwrap_or_default()
}

/// 404 unless the thread exists and belongs to the caller — admins see
/// everything, legacy threads (empty owner) stay shared. We deliberately
/// return the same "not found" for foreign threads so ids don't leak.
fn own_thread(st: &ChatState, tid: &str, auth: &SessionAuth) -> Result<(), Response> {
    let owner = st
        .app
        .engine
        .meta
        .chat_thread_user(tid)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db", &e.to_string()))?;
    match owner {
        Some(o) if o.is_empty() || o == auth.0.id || auth.is_admin() => Ok(()),
        _ => Err(err(
            StatusCode::NOT_FOUND,
            "not_found",
            "Thread nicht gefunden",
        )),
    }
}

/// Scope snapshot for a new run. The run executes with the *thread
/// owner's* scope — not the caller's: an admin continuing a user's
/// thread must not inject unrestricted results into that user's
/// transcript. Resolved at claim time so a mid-run grant change takes
/// effect on the next `/messages` or `/continue`.
fn thread_perm(st: &ChatState, tid: &str, auth: &SessionAuth) -> ragnarok_core::scope::Permitted {
    let owner = st
        .app
        .engine
        .meta
        .chat_thread_user(tid)
        .ok()
        .flatten()
        .unwrap_or_default();
    let (uid, is_admin) = if owner.is_empty() {
        // legacy/unowned thread — the caller's scope applies
        (auth.0.id.clone(), auth.is_admin())
    } else {
        let role = st
            .app
            .engine
            .meta
            .user_by_id(&owner)
            .ok()
            .flatten()
            .map(|u| u.role)
            .unwrap_or_default();
        (owner, role == "admin")
    };
    let mut perm = st
        .app
        .engine
        .meta
        .effective_scope(&uid, is_admin, None)
        .unwrap_or_else(|_| ragnarok_core::scope::Permitted::all());
    // An API-key-bound scope narrows the caller too — the token defines
    // what this principal may ever see, regardless of thread ownership.
    if let Some(raw) = auth.1.as_deref() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) {
            if let Ok(f) = ragnarok_core::scope::ScopeFilter::from_value(&v) {
                if !f.is_unrestricted() {
                    if let Ok(extra) = ragnarok_store::resolve_permitted(&st.app.engine.meta, &f) {
                        perm = perm.combine(&extra);
                    }
                }
            }
        }
    }
    perm
}

/// `/api/v1/chat/*` — same route shape as RiskMaster's chat handler.
pub fn router(app: Arc<AppState>) -> axum::Router<Arc<AppState>> {
    let st = Arc::new(ChatState {
        app,
        live: Arc::new(Mutex::new(HashMap::new())),
    });
    axum::Router::new()
        .route(
            "/api/v1/chat/threads",
            axum::routing::get(list_threads).post(create_thread),
        )
        .route(
            "/api/v1/chat/threads/{id}",
            axum::routing::get(get_thread)
                .delete(delete_thread)
                .patch(update_thread_model),
        )
        .route(
            "/api/v1/chat/threads/{id}/messages",
            axum::routing::get(list_messages).post(post_message),
        )
        .route(
            "/api/v1/chat/threads/{id}/continue",
            axum::routing::post(continue_thread),
        )
        .route(
            "/api/v1/chat/threads/{id}/stop",
            axum::routing::post(stop_thread),
        )
        .route(
            "/api/v1/chat/threads/{id}/steer",
            axum::routing::post(steer_thread),
        )
        .route(
            "/api/v1/chat/threads/{id}/stream",
            axum::routing::get(attach_thread),
        )
        .route(
            "/api/v1/chat/settings",
            axum::routing::get(get_settings).put(put_settings),
        )
        .route(
            "/api/v1/chat/settings/models",
            axum::routing::get(list_provider_models),
        )
        .route(
            "/api/v1/chat/settings/test",
            axum::routing::post(test_provider),
        )
        .with_state(st)
}

// ---------------------------------------------------------------------------
// thread CRUD
// ---------------------------------------------------------------------------

async fn list_threads(
    State(st): State<Arc<ChatState>>,
    auth: SessionAuth,
    Query(q): Query<ListQuery>,
) -> Result<Json<Value>, Response> {
    let uid = (!auth.is_admin()).then_some(auth.0.id.as_str());
    let rows = st
        .app
        .engine
        .meta
        .chat_threads(&q.q.unwrap_or_default(), uid)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db", &e.to_string()))?;
    let threads: Vec<ThreadDto> = rows
        .into_iter()
        .map(
            |(id, title, ca, ua, running, awaiting, count, last)| ThreadDto {
                id,
                title,
                created_at: ts(ca),
                updated_at: ts(ua),
                running,
                awaiting,
                message_count: count,
                last_message: last,
            },
        )
        .collect();
    Ok(Json(json!({ "threads": threads })))
}

async fn create_thread(
    State(st): State<Arc<ChatState>>,
    auth: SessionAuth,
    Json(req): Json<CreateThread>,
) -> Result<(StatusCode, Json<Value>), Response> {
    let id = uuid::Uuid::new_v4().to_string();
    st.app
        .engine
        .meta
        .chat_create_thread(&id, req.title.as_deref().unwrap_or(""), &auth.0.id)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db", &e.to_string()))?;
    if req.model.is_some() || req.reasoning_effort.is_some() {
        let _ = st.app.engine.meta.chat_update_thread_model(
            &id,
            req.model.as_deref(),
            req.reasoning_effort.as_deref(),
        );
    }
    Ok((StatusCode::CREATED, Json(json!({ "id": id }))))
}

async fn get_thread(
    State(st): State<Arc<ChatState>>,
    auth: SessionAuth,
    Path(id): Path<String>,
) -> Result<Json<Value>, Response> {
    own_thread(&st, &id, &auth)?;
    let Some((title, running, awaiting, model, effort, live_state, todos)) = st
        .app
        .engine
        .meta
        .chat_thread_row(&id)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db", &e.to_string()))?
    else {
        return Err(err(
            StatusCode::NOT_FOUND,
            "not_found",
            "Thread nicht gefunden",
        ));
    };
    Ok(Json(json!({ "thread": {
        "id": id, "title": title, "running": running, "awaiting": awaiting,
        "model": model, "reasoningEffort": effort,
        "liveState": live_state.and_then(|s| serde_json::from_str::<Value>(&s).ok()),
        "todos": todos.and_then(|s| serde_json::from_str::<Value>(&s).ok()),
    }})))
}

/// `PATCH …/threads/{id}` — per-thread model/reasoning-effort overrides
/// (`null` clears → global settings). Port of RiskMaster's
/// `update_thread_model`.
async fn update_thread_model(
    State(st): State<Arc<ChatState>>,
    auth: SessionAuth,
    Path(id): Path<String>,
    Json(req): Json<PatchThread>,
) -> Result<Json<Value>, Response> {
    own_thread(&st, &id, &auth)?;
    st.app
        .engine
        .meta
        .chat_patch_thread_model(
            &id,
            req.model.as_ref().map(|o| o.as_deref()),
            req.reasoning_effort.as_ref().map(|o| o.as_deref()),
        )
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db", &e.to_string()))?;
    // echo the stored values so absent fields don't echo as fake `null`s
    let row = st.app.engine.meta.chat_thread_row(&id).ok().flatten();
    Ok(Json(json!({
        "model": row.as_ref().map(|r| r.3.clone()),
        "reasoningEffort": row.map(|r| r.4),
    })))
}

async fn delete_thread(
    State(st): State<Arc<ChatState>>,
    auth: SessionAuth,
    Path(id): Path<String>,
) -> Result<StatusCode, Response> {
    own_thread(&st, &id, &auth)?;
    st.app
        .engine
        .meta
        .chat_delete_thread(&id)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db", &e.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}

async fn list_messages(
    State(st): State<Arc<ChatState>>,
    auth: SessionAuth,
    Path(id): Path<String>,
) -> Result<Json<Value>, Response> {
    own_thread(&st, &id, &auth)?;
    let rows = st
        .app
        .engine
        .meta
        .chat_messages(&id)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db", &e.to_string()))?;
    let tool_calls = st
        .app
        .engine
        .meta
        .chat_tool_calls(&id)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db", &e.to_string()))?;
    let messages: Vec<MessageDto> = rows
        .into_iter()
        .map(
            |(id, seq, role, content, _cites, reasoning, model, at)| MessageDto {
                id,
                seq,
                role,
                content,
                reasoning,
                model,
                created_at: ts(at),
            },
        )
        .collect();
    let calls: Vec<Value> = tool_calls
        .into_iter()
        .map(|(seq, name, args, status, dur, mid, pos, r)| {
            json!({
                "seq": seq, "name": name, "status": status,
                "durationMs": dur, "messageId": mid, "pos": pos,
                "arguments": args.and_then(|a| serde_json::from_str::<Value>(&a).ok()),
                "ref": r.and_then(|a| serde_json::from_str::<Value>(&a).ok()),
            })
        })
        .collect();
    Ok(Json(json!({ "messages": messages, "toolCalls": calls })))
}

// ---------------------------------------------------------------------------
// settings (single-row provider config; empty apiKey keeps the stored one)
// ---------------------------------------------------------------------------

async fn get_settings(
    State(st): State<Arc<ChatState>>,
    _auth: SessionAuth,
) -> Result<Json<Value>, Response> {
    let s = st
        .app
        .engine
        .meta
        .chat_settings()
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db", &e.to_string()))?;
    // The API key never leaves the server — mask like RiskMaster does.
    let masked = if s.api_key.is_empty() {
        String::new()
    } else {
        format!(
            "…{}",
            s.api_key
                .chars()
                .rev()
                .take(4)
                .collect::<String>()
                .chars()
                .rev()
                .collect::<String>()
        )
    };
    Ok(Json(json!({
        "baseUrl": s.base_url, "apiKey": masked, "model": s.model,
        "reasoningEffort": s.reasoning_effort,
        "maxSteps": s.max_steps, "contextTokens": s.context_tokens,
        "chatMaxTokens": s.chat_max_tokens,
        "providerMaxTokens": s.provider_max_tokens,
        "configured": !s.base_url.trim().is_empty(),
    })))
}

async fn put_settings(
    State(st): State<Arc<ChatState>>,
    _a: AdminAuth,
    Json(req): Json<ChatSettingsRequest>,
) -> Result<Json<Value>, Response> {
    let mut s = st.app.engine.meta.chat_settings().unwrap_or_default();
    s.base_url = req.base_url.trim().trim_end_matches('/').to_owned();
    if let Some(k) = req.api_key {
        let k = k.trim();
        if !k.is_empty() {
            s.api_key = k.to_owned();
        }
    }
    if let Some(m) = req.model {
        s.model = m.trim().to_owned();
    }
    if let Some(e) = req.reasoning_effort {
        s.reasoning_effort = e.trim().to_owned();
    }
    if let Some(v) = req.max_steps {
        s.max_steps = v.clamp(1, 100);
    }
    if let Some(v) = req.context_tokens {
        s.context_tokens = v.clamp(4_096, 1_000_000);
    }
    if let Some(v) = req.chat_max_tokens {
        s.chat_max_tokens = v.clamp(0, 1_000_000);
    }
    if let Some(v) = req.provider_max_tokens {
        s.provider_max_tokens = v.clamp(0, 1_000_000);
    }
    if s.model.is_empty() {
        s.model = DEFAULT_CHAT_MODEL.to_owned();
    }
    if s.max_steps == 0 {
        s.max_steps = 12;
    }
    if s.context_tokens == 0 {
        s.context_tokens = 128_000;
    }
    st.app
        .engine
        .meta
        .chat_put_settings(&s)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db", &e.to_string()))?;
    Ok(Json(
        json!({ "ok": true, "configured": !s.base_url.is_empty(), "model": s.model }),
    ))
}

// ---------------------------------------------------------------------------
// provider probes: model list + connection test (admin only)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProviderProbeQuery {
    /// Overrides the stored base URL — lets the UI probe unsaved input.
    base_url: Option<String>,
    /// Overrides the stored API key (empty/None = use stored).
    api_key: Option<String>,
}

/// Resolve probe credentials: request override wins, stored row is the
/// fallback so the UI can test before saving.
fn probe_credentials(
    st: &ChatState,
    base_url: Option<&str>,
    api_key: Option<&str>,
) -> Result<(String, String), Response> {
    let stored = st.app.engine.meta.chat_settings().unwrap_or_default();
    let base = base_url
        .filter(|b| !b.trim().is_empty())
        .unwrap_or(&stored.base_url)
        .trim()
        .trim_end_matches('/')
        .to_owned();
    if base.is_empty() {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "invalid",
            "Basis-URL fehlt — weder gespeichert noch angegeben.",
        ));
    }
    let key = api_key
        .filter(|k| !k.trim().is_empty())
        .map(str::to_owned)
        .unwrap_or(stored.api_key);
    Ok((base, key))
}

/// `GET …/settings/models?baseUrl=&apiKey=` — cheap `GET {base}/models`
/// probe: validates credentials and returns the model id list for the
/// picker (OpenAI-compatible shape; ids also pulled from `data[].name`
/// for providers that deviate).
async fn list_provider_models(
    State(st): State<Arc<ChatState>>,
    _a: AdminAuth,
    Query(q): Query<ProviderProbeQuery>,
) -> Result<Json<Value>, Response> {
    let (base, key) = probe_credentials(&st, q.base_url.as_deref(), q.api_key.as_deref())?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .build()
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "http", &e.to_string()))?;
    let resp = client
        .get(format!("{base}/models"))
        .bearer_auth(&key)
        .send()
        .await
        .map_err(|e| {
            err(
                StatusCode::BAD_GATEWAY,
                "provider",
                &format!("{base}/models: {e}"),
            )
        })?;
    if !resp.status().is_success() {
        let code = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        return Err(err(
            StatusCode::BAD_GATEWAY,
            "provider",
            &format!(
                "Provider HTTP {code}: {}",
                text.chars().take(300).collect::<String>()
            ),
        ));
    }
    let body: Value = resp.json().await.map_err(|e| {
        err(
            StatusCode::BAD_GATEWAY,
            "provider",
            &format!("bad json: {e}"),
        )
    })?;
    let mut ids: Vec<String> = body["data"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| {
            m["id"]
                .as_str()
                .or_else(|| m["name"].as_str())
                .map(str::to_owned)
        })
        .collect();
    ids.sort();
    ids.dedup();
    Ok(Json(json!({ "models": ids, "count": ids.len() })))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProviderTestRequest {
    base_url: Option<String>,
    api_key: Option<String>,
    model: Option<String>,
    reasoning_effort: Option<String>,
}

/// `POST …/settings/test` — real end-to-end check: one minimal
/// `chat/completions` round-trip (no tools, no stream). 60 s budget —
/// reasoning models spend a while in hidden chain-of-thought even for
/// tiny answers.
async fn test_provider(
    State(st): State<Arc<ChatState>>,
    _a: AdminAuth,
    Json(req): Json<ProviderTestRequest>,
) -> Result<Json<Value>, Response> {
    let (base, key) = probe_credentials(&st, req.base_url.as_deref(), req.api_key.as_deref())?;
    let model = req
        .model
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| st.app.engine.meta.chat_settings().unwrap_or_default().model);
    if model.trim().is_empty() {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "invalid",
            "Modell fehlt — weder gespeichert noch angegeben.",
        ));
    }
    let mut body = json!({
        "model": model.trim(),
        "messages": [{"role": "user", "content": "Antworte mit einem Wort: bereit"}],
        "max_tokens": 32,
        "stream": false,
    });
    let eff = match req.reasoning_effort.as_deref() {
        Some(e) => e.to_owned(),
        None => {
            st.app
                .engine
                .meta
                .chat_settings()
                .unwrap_or_default()
                .reasoning_effort
        }
    };
    crate::agent::apply_thinking_mode(&mut body, &eff);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "http", &e.to_string()))?;
    let t0 = std::time::Instant::now();
    let resp = client
        .post(format!("{base}/chat/completions"))
        .bearer_auth(&key)
        .json(&body)
        .send()
        .await
        .map_err(|e| err(StatusCode::BAD_GATEWAY, "provider", &format!("{e}")))?;
    let latency_ms = t0.elapsed().as_millis() as u64;
    if !resp.status().is_success() {
        let code = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        return Ok(Json(json!({
            "ok": false, "latencyMs": latency_ms,
            "error": format!("HTTP {code}: {}", text.chars().take(300).collect::<String>()),
        })));
    }
    let body: Value = resp.json().await.map_err(|e| {
        err(
            StatusCode::BAD_GATEWAY,
            "provider",
            &format!("bad json: {e}"),
        )
    })?;
    let reply = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .chars()
        .take(120)
        .collect::<String>();
    Ok(Json(json!({
        "ok": true,
        "latencyMs": latency_ms,
        "model": body["model"].as_str().unwrap_or(model.trim()),
        "reply": reply,
        "usage": body["usage"],
    })))
}

// ---------------------------------------------------------------------------
// run lifecycle: message → claim → spawn agent → SSE
// ---------------------------------------------------------------------------

/// Resolve the effective model for a run: thread override → settings →
/// builtin default.
fn resolve_model(st: &ChatState, tid: &str) -> String {
    let thread_model = st
        .app
        .engine
        .meta
        .chat_thread_row(tid)
        .ok()
        .flatten()
        .and_then(|t| t.3)
        .unwrap_or_default();
    let m = thread_model.trim();
    if !m.is_empty() {
        return m.to_owned();
    }
    let s = st.app.engine.meta.chat_settings().unwrap_or_default();
    if s.model.trim().is_empty() {
        DEFAULT_CHAT_MODEL.to_owned()
    } else {
        s.model.trim().to_owned()
    }
}

/// `POST …/messages` — persist the user message, claim the run, spawn the
/// agent loop, stream its frames as SSE. Port of RiskMaster's flow.
async fn post_message(
    State(st): State<Arc<ChatState>>,
    auth: SessionAuth,
    Path(id): Path<String>,
    Json(req): Json<PostMessage>,
) -> Result<Response, Response> {
    own_thread(&st, &id, &auth)?;
    let content = req.content.trim().to_string();
    if content.is_empty() {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "empty_message",
            "Leere Nachricht",
        ));
    }
    let title: String = content
        .split_whitespace()
        .take(14)
        .collect::<Vec<_>>()
        .join(" ");
    let claimed = st
        .app
        .engine
        .meta
        .chat_claim_run(&id, &title, CLAIM_STALE_SECS)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db", &e.to_string()))?;
    if !claimed {
        return Err(err(
            StatusCode::CONFLICT,
            "run_in_progress",
            "In diesem Chat läuft bereits ein Lauf — bitte warten oder später fortsetzen.",
        ));
    }
    // Everything below holds the claim — release it on any error.
    let prepared: Result<
        (
            agent::RunParams,
            tokio::sync::mpsc::Receiver<Result<Bytes, Infallible>>,
        ),
        Response,
    > = (|| {
        let msg_id = uuid::Uuid::new_v4().to_string();
        st.app
            .engine
            .meta
            .chat_add_message(&id, &msg_id, "user", &content, "[]", "", "")
            .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db", &e.to_string()))?;
        let settings = st.app.engine.meta.chat_settings().unwrap_or_default();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, Infallible>>(64);
        let rx = tee_run(&st.live, &id, rx);
        Ok::<_, Response>((
            agent::RunParams {
                thread_id: id.clone(),
                model: resolve_model(&st, &id),
                resume: None,
                max_steps: settings.max_steps,
                context_tokens: settings.context_tokens,
                chat_max_tokens: settings.chat_max_tokens,
                perm: thread_perm(&st, &id, &auth),
                tx,
            },
            rx,
        ))
    })();
    let (params, rx) = match prepared {
        Ok(v) => v,
        Err(e) => {
            let _ = st.app.engine.meta.chat_fail_run(&id);
            return Err(e);
        }
    };
    agent::spawn_agent(st.clone(), params);
    Ok(sse_response(rx))
}

/// `POST …/continue` — resume a run parked in `pending_state` (step
/// budget, stop, or unfinished exit). Claim + context load are atomic.
async fn continue_thread(
    State(st): State<Arc<ChatState>>,
    auth: SessionAuth,
    Path(id): Path<String>,
) -> Result<Response, Response> {
    own_thread(&st, &id, &auth)?;
    let pending = st
        .app
        .engine
        .meta
        .chat_claim_continue(&id, CLAIM_STALE_SECS)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db", &e.to_string()))?;
    let Some(pending) = pending else {
        let (running, has_pending) = st
            .app
            .engine
            .meta
            .chat_thread_flags(&id)
            .unwrap_or((false, false));
        return Err(if running {
            err(
                StatusCode::CONFLICT,
                "run_in_progress",
                "In diesem Chat läuft bereits ein Lauf.",
            )
        } else if !has_pending {
            err(
                StatusCode::BAD_REQUEST,
                "nothing_pending",
                "Kein pausierter Lauf in diesem Chat",
            )
        } else {
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "db",
                "pending_state konnte nicht gelesen werden",
            )
        });
    };
    let prepared: Result<
        (
            agent::RunParams,
            tokio::sync::mpsc::Receiver<Result<Bytes, Infallible>>,
        ),
        Response,
    > = (|| {
        let resume: AgentResume = serde_json::from_str(&pending).map_err(|e| {
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "bad_pending_state",
                &e.to_string(),
            )
        })?;
        let settings = st.app.engine.meta.chat_settings().unwrap_or_default();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, Infallible>>(64);
        let rx = tee_run(&st.live, &id, rx);
        Ok::<_, Response>((
            agent::RunParams {
                thread_id: id.clone(),
                model: resolve_model(&st, &id),
                resume: Some(resume),
                max_steps: settings.max_steps,
                context_tokens: settings.context_tokens,
                chat_max_tokens: settings.chat_max_tokens,
                perm: thread_perm(&st, &id, &auth),
                tx,
            },
            rx,
        ))
    })();
    let (params, rx) = match prepared {
        Ok(v) => v,
        Err(e) => {
            let _ = st.app.engine.meta.chat_fail_run(&id);
            return Err(e);
        }
    };
    agent::spawn_agent(st.clone(), params);
    Ok(sse_response(rx))
}

/// `POST …/stop` — ask the in-flight run to halt. Idempotent; the loop
/// polls the flag between steps and throttled mid-stream, then parks.
async fn stop_thread(
    State(st): State<Arc<ChatState>>,
    auth: SessionAuth,
    Path(id): Path<String>,
) -> Result<Json<Value>, Response> {
    own_thread(&st, &id, &auth)?;
    let stopping = st
        .app
        .engine
        .meta
        .chat_set_cancel(&id)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db", &e.to_string()))?;
    Ok(Json(json!({ "ok": true, "stopping": stopping })))
}

/// `POST …/steer` — inject a user message into an in-flight run: lands in
/// `steer_queue` (drained at the next step boundary) and in chat_messages
/// so the transcript shows it regardless of when the run ends.
async fn steer_thread(
    State(st): State<Arc<ChatState>>,
    auth: SessionAuth,
    Path(id): Path<String>,
    Json(req): Json<SteerRequest>,
) -> Result<Json<Value>, Response> {
    own_thread(&st, &id, &auth)?;
    let content = req.content.trim().to_owned();
    if content.is_empty() {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "empty_content",
            "Leere Nachricht.",
        ));
    }
    let appended = st
        .app
        .engine
        .meta
        .chat_append_steer(&id, &content)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db", &e.to_string()))?;
    if !appended {
        return Err(err(
            StatusCode::CONFLICT,
            "not_running",
            "In diesem Chat läuft gerade kein Lauf — Nachricht normal senden.",
        ));
    }
    let msg_id = uuid::Uuid::new_v4().to_string();
    let _ = st
        .app
        .engine
        .meta
        .chat_add_message(&id, &msg_id, "user", &content, "[]", "", "");
    Ok(Json(json!({ "ok": true, "queued": true })))
}

// ---------------------------------------------------------------------------
// SSE plumbing — verbatim ports of RiskMaster's tee/attach/sse helpers
// ---------------------------------------------------------------------------

/// Tee an in-flight run's SSE output: forwards frames to the initiating
/// client while buffering them for late attachers; the buffer lingers
/// ~30 s after the run ends so a just-late attach still sees `done`.
fn tee_run(
    live: &LiveMap,
    tid: &str,
    mut rx: tokio::sync::mpsc::Receiver<Result<Bytes, Infallible>>,
) -> tokio::sync::mpsc::Receiver<Result<Bytes, Infallible>> {
    let (notify, _) = tokio::sync::watch::channel(0u64);
    let run = Arc::new(Mutex::new(LiveRun {
        frames: Vec::new(),
        done: false,
        notify,
    }));
    live.lock().unwrap().insert(tid.to_string(), run.clone());
    let (out_tx, out_rx) = tokio::sync::mpsc::channel::<Result<Bytes, Infallible>>(64);
    let live = live.clone();
    let tid = tid.to_string();
    tokio::spawn(async move {
        let mut n = 0u64;
        while let Some(item) = rx.recv().await {
            {
                let mut g = run.lock().unwrap();
                if let Ok(b) = &item {
                    g.frames.push(b.clone());
                }
                n += 1;
                let _ = g.notify.send(n);
            }
            if out_tx.send(item).await.is_err() {
                break;
            }
        }
        {
            let mut g = run.lock().unwrap();
            g.done = true;
            let _ = g.notify.send(n + 1);
        }
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            live.lock().unwrap().remove(&tid);
        });
    });
    out_rx
}

/// `GET …/stream` — attach to an in-flight run: replays emitted frames,
/// then follows live. Idle threads answer with a single `done` frame.
async fn attach_thread(
    State(st): State<Arc<ChatState>>,
    auth: SessionAuth,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = own_thread(&st, &id, &auth) {
        return e;
    }
    let run = st.live.lock().unwrap().get(&id).cloned();
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, Infallible>>(64);
    let Some(run) = run else {
        let _ = tx
            .send(Ok(Bytes::from("data: {\"type\":\"done\"}\n\n")))
            .await;
        return sse_response(rx);
    };
    let mut notify = run.lock().unwrap().notify.subscribe();
    tokio::spawn(async move {
        let mut sent = 0usize;
        loop {
            let (frames, done) = {
                let g = run.lock().unwrap();
                (g.frames[sent..].to_vec(), g.done)
            };
            for frame in frames {
                sent += 1;
                if tx.send(Ok(frame)).await.is_err() {
                    return;
                }
            }
            if done || notify.changed().await.is_err() {
                return;
            }
        }
    });
    sse_response(rx)
}

/// `data: {json}\n\n` into the run channel — awaited send applies
/// backpressure; terminal events are never dropped silently.
pub(crate) async fn sse(tx: &tokio::sync::mpsc::Sender<Result<Bytes, Infallible>>, event: &Value) {
    let _ = tx.send(Ok(Bytes::from(format!("data: {event}\n\n")))).await;
}

fn sse_response(rx: tokio::sync::mpsc::Receiver<Result<Bytes, Infallible>>) -> Response {
    let stream = futures::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    });
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(stream))
        .unwrap()
        .into_response()
}
