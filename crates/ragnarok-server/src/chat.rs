//! Retrieval-grounded chat: thread/message persistence in SQLite plus an
//! SSE streaming surface mirroring RiskMaster's chat protocol
//! (`data: {json}\n\n` frames, `token`/`citation`/`done`/`error` events,
//! run-claim with steal window, attach-with-replay for in-flight runs).
//!
//! Instead of an LLM tool-call loop, a run executes the hybrid search
//! pipeline and composes an extractive answer from the top passages —
//! every claim carries a citation back to the indexed document.

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

use crate::state::AppState;

/// Seconds a stale run claim may be reclaimed (covers crashed streams).
const CLAIM_STALE_SECS: i64 = 60;
/// Hits streamed as citations per assistant answer.
const CITATION_LIMIT: usize = 8;
/// Pause between streamed answer fragments — keeps the bubble "typing".
const TOKEN_DELAY_MS: u64 = 18;

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
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PostMessage {
    content: String,
}

#[derive(Deserialize)]
struct ListQuery {
    q: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ThreadDto {
    id: String,
    title: String,
    created_at: String,
    updated_at: String,
    message_count: i64,
    last_message: Option<String>,
    running: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MessageDto {
    id: String,
    seq: i64,
    role: String,
    content: String,
    citations: Value,
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
            axum::routing::get(get_thread).delete(delete_thread),
        )
        .route(
            "/api/v1/chat/threads/{id}/messages",
            axum::routing::get(list_messages).post(post_message),
        )
        .route(
            "/api/v1/chat/threads/{id}/stream",
            axum::routing::get(attach_thread),
        )
        .with_state(st)
}

async fn list_threads(
    State(st): State<Arc<ChatState>>,
    Query(q): Query<ListQuery>,
) -> Result<Json<Value>, Response> {
    let rows = st
        .app
        .engine
        .meta
        .chat_threads(&q.q.unwrap_or_default())
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db", &e.to_string()))?;
    let threads: Vec<ThreadDto> = rows
        .into_iter()
        .map(|(id, title, ca, ua, running, count, last)| ThreadDto {
            id,
            title,
            created_at: ts(ca),
            updated_at: ts(ua),
            running,
            message_count: count,
            last_message: last,
        })
        .collect();
    Ok(Json(json!({ "threads": threads })))
}

async fn create_thread(
    State(st): State<Arc<ChatState>>,
    Json(req): Json<CreateThread>,
) -> Result<(StatusCode, Json<Value>), Response> {
    let id = uuid::Uuid::new_v4().to_string();
    st.app
        .engine
        .meta
        .chat_create_thread(&id, req.title.as_deref().unwrap_or(""))
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db", &e.to_string()))?;
    Ok((StatusCode::CREATED, Json(json!({ "id": id }))))
}

async fn get_thread(
    State(st): State<Arc<ChatState>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, Response> {
    let Some((title, running)) = st
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
    Ok(Json(
        json!({ "thread": { "id": id, "title": title, "running": running } }),
    ))
}

async fn delete_thread(
    State(st): State<Arc<ChatState>>,
    Path(id): Path<String>,
) -> Result<StatusCode, Response> {
    st.app
        .engine
        .meta
        .chat_delete_thread(&id)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db", &e.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}

async fn list_messages(
    State(st): State<Arc<ChatState>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, Response> {
    if !st
        .app
        .engine
        .meta
        .chat_thread_exists(&id)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db", &e.to_string()))?
    {
        return Err(err(
            StatusCode::NOT_FOUND,
            "not_found",
            "Thread nicht gefunden",
        ));
    }
    let rows = st
        .app
        .engine
        .meta
        .chat_messages(&id)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db", &e.to_string()))?;
    let messages: Vec<MessageDto> = rows
        .into_iter()
        .map(|(id, seq, role, content, citations, at)| MessageDto {
            id,
            seq,
            role,
            content,
            citations: serde_json::from_str(&citations).unwrap_or(json!([])),
            created_at: ts(at),
        })
        .collect();
    Ok(Json(json!({ "messages": messages })))
}

/// `POST …/messages` — persist the user message, claim the run, stream
/// the retrieval answer as SSE. Mirrors RiskMaster's claim-then-run flow.
async fn post_message(
    State(st): State<Arc<ChatState>>,
    Path(id): Path<String>,
    Json(req): Json<PostMessage>,
) -> Result<Response, Response> {
    if !st
        .app
        .engine
        .meta
        .chat_thread_exists(&id)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db", &e.to_string()))?
    {
        return Err(err(
            StatusCode::NOT_FOUND,
            "not_found",
            "Thread nicht gefunden",
        ));
    }
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
            "In diesem Chat läuft bereits ein Lauf.",
        ));
    }
    let msg_id = uuid::Uuid::new_v4().to_string();
    let _ = st
        .app
        .engine
        .meta
        .chat_add_message(&id, &msg_id, "user", &content, "[]");

    // live channel: the run task emits frames; tee_run replays them to
    // late attachers and keeps a copy for `…/stream` reconnects.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, Infallible>>(64);
    let rx = tee_run(&st.live, &id, rx);
    let st2 = st.clone();
    let tid = id.clone();
    tokio::spawn(async move {
        run_answer(&st2, &tid, &content, tx).await;
    });
    Ok(sse_response(rx))
}

/// The retrieval "run": hybrid search → citation frames → streamed
/// extractive answer → persisted assistant message → `done`.
async fn run_answer(
    st: &ChatState,
    tid: &str,
    query: &str,
    tx: tokio::sync::mpsc::Sender<Result<Bytes, Infallible>>,
) {
    sse(&tx, &json!({"type": "status", "phase": "embedding"})).await;
    let mut req = ragnarok_core::SearchRequest::new(query);
    req.limit = CITATION_LIMIT;
    sse(&tx, &json!({"type": "status", "phase": "search"})).await;
    let hits = match st.app.engine.search(&req).await {
        Ok(h) => h,
        Err(e) => {
            sse(&tx, &json!({"type": "error", "message": e.to_string()})).await;
            let _ = st.app.engine.meta.chat_finish_run(tid);
            return;
        }
    };

    // citation frames — the UI renders these as source chips with a
    // deep-link into the document viewer.
    let mut cites = Vec::new();
    for (i, h) in hits.iter().enumerate() {
        let cite = json!({
            "n": i + 1,
            "documentId": h.document_id.to_string(),
            "path": h.path,
            "page": h.page,
            "score": h.score,
        });
        cites.push(cite.clone());
        sse(&tx, &json!({"type": "citation", "citation": cite})).await;
    }

    // extractive answer: citations interleaved with the best passages.
    let answer = compose_answer(&hits);
    for frag in answer_fragments(&answer) {
        sse(&tx, &json!({"type": "token", "delta": frag})).await;
        tokio::time::sleep(std::time::Duration::from_millis(TOKEN_DELAY_MS)).await;
    }

    let msg_id = uuid::Uuid::new_v4().to_string();
    let cite_json = serde_json::to_string(&cites).unwrap_or_else(|_| "[]".into());
    let _ = st
        .app
        .engine
        .meta
        .chat_add_message(tid, &msg_id, "assistant", &answer, &cite_json);
    let _ = st.app.engine.meta.chat_finish_run(tid);
    sse(&tx, &json!({"type": "done", "messageId": msg_id})).await;
}

/// Sentence-ish fragments for the streamed answer.
fn answer_fragments(text: &str) -> Vec<String> {
    let mut frags = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        if matches!(ch, '.' | '!' | '?' | '\n') && cur.trim().len() > 24 {
            frags.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        frags.push(cur);
    }
    frags
}

/// Markdown answer with numbered citations back to the documents.
fn compose_answer(hits: &[ragnarok_core::SearchHit]) -> String {
    if hits.is_empty() {
        return "Keine relevanten Passagen in den indexierten Dokumenten gefunden. Versuche eine andere Formulierung oder füge weitere Quellen hinzu.".to_string();
    }
    let mut out = format!("Ich habe {} relevante Passage(n) gefunden:\n\n", hits.len());
    for (i, h) in hits.iter().enumerate() {
        let excerpt: String = h
            .text
            .split_whitespace()
            .take(48)
            .collect::<Vec<_>>()
            .join(" ");
        out.push_str(&format!(
            "**[{}]** `{}` (S. {}) — {:?}\n> {}\n\n",
            i + 1,
            h.path,
            h.page + 1,
            h.doc_type,
            excerpt
        ));
    }
    out.push_str(
        "Die nummerierten Quellen sind unter der Antwort als Chips verlinkt — Klick öffnet das Dokument an der Fundstelle.",
    );
    out
}

/// Copy of RiskMaster's `tee_run`: taps the run's frame stream into the
/// live registry so `…/stream` attachers replay the prefix then follow
/// live until `done`; entries self-clean 30 s after completion.
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
        let tid2 = tid.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            live.lock().unwrap().remove(&tid2);
        });
    });
    out_rx
}

/// `GET …/stream` — attach to an in-flight run: replays emitted frames,
/// then follows live. Idle threads answer with a single `done` frame.
async fn attach_thread(State(st): State<Arc<ChatState>>, Path(id): Path<String>) -> Response {
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

/// `data: {json}\n\n` frame into the run channel (awaited = backpressure,
/// like RiskMaster — terminal events are never dropped silently).
async fn sse(tx: &tokio::sync::mpsc::Sender<Result<Bytes, Infallible>>, event: &Value) {
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
