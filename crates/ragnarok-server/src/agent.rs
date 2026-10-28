//! The model↔tool agent loop behind `POST …/messages` — a direct port of
//! RiskMaster's `run_agent`: streamed OpenAI-compatible chat completions,
//! tool calls executed in-process against the search engine, a step
//! budget with `pending_state` parking (`awaiting_continue`), user
//! steering (`steer_queue`) and cooperative stop (`cancel_requested`).
//!
//! RAGnarök's tool surface replaces RiskMaster's MCP catalogue with the
//! retrieval tools a local archive assistant needs: `search`,
//! `read_document`, `list_documents`, `list_sources`, `system_stats`
//! plus the chat-local `todo_read`/`todo_write`/`task_done`.

use std::sync::Arc;

use axum::body::Bytes;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::convert::Infallible;

use crate::chat::{sse, ChatState};

/// Fallback max_tokens for a step completion (see
/// `effective_max_tokens`).
const DEFAULT_CHAT_MAX_TOKENS: i64 = 100_000;
/// Compact once estimated usage reaches this share of the window.
const COMPACT_AT: f64 = 0.75;
/// Estimated characters per token for the usage heuristic.
const CHARS_PER_TOKEN: usize = 4;
/// Tool results are truncated before going back into the context.
const MAX_TOOL_RESULT_CHARS: usize = 30_000;
/// Messages loaded into a fresh run's context.
const HISTORY_LIMIT: i64 = 60;
/// Connect timeout; the streamed body is bounded by an *idle* timeout so
/// long-but-active generations are not cut off.
const CONNECT_TIMEOUT_SECS: u64 = 60;
/// Max silence between two LLM SSE chunks before the stream is aborted.
const STREAM_IDLE_SECS: u64 = 300;
/// Throttled cancel-poll while tokens stream.
const CONTROL_POLL_MS: u64 = 400;
/// Throttled live-state flush.
const LIVE_FLUSH_SECS: u64 = 2;

/// Chat-local tools — `task_done` is the explicit completion signal: a
/// run ending without it is treated as unfinished and parks for
/// /continue, exactly like RiskMaster.
const TOOL_DONE: &str = "task_done";
const TOOL_TODO_READ: &str = "todo_read";
const TOOL_TODO_WRITE: &str = "todo_write";

/// RAGnarök's retrieval tool surface — all read-only by nature.
const RETRIEVAL_TOOLS: &[&str] = &[
    "search",
    "read_document",
    "list_documents",
    "list_sources",
    "system_stats",
];

fn openai_tools() -> Vec<Value> {
    let v = vec![
        json!({
            "type": "function",
            "function": {
                "name": "search",
                "description": "Hybride Volltext-/Vektorsuche über das indexierte Archiv (dense + sparse + ColBERT-Rescoring). Liefert die besten Passagen mit Dokument-Pfad und Seitenzahl.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Suchanfrage in natürlicher Sprache"},
                        "limit": {"type": "integer", "description": "max. Treffer (Default 8)"},
                    },
                    "required": ["query"],
                },
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "read_document",
                "description": "Volltext eines Dokuments anhand seiner documentId (aus search/list_documents).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "documentId": {"type": "string"},
                    },
                    "required": ["documentId"],
                },
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "list_documents",
                "description": "Indexierte Dokumente auflisten (optional nach Quelle gefiltert).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sourceId": {"type": "string", "description": "optional: nur diese Quelle"},
                        "limit": {"type": "integer", "description": "max. Einträge (Default 50)"},
                    },
                    "required": [],
                },
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "list_sources",
                "description": "Registrierte Datenquellen (Ordner, IMAP-Mailboxen) mit Sync-Status.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "system_stats",
                "description": "Index- und Ressourcen-Statistik: Dokument-/Chunk-Zahlen, RAM-Budget, Backpressure-Zustand.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": TOOL_TODO_READ,
                "description": "Return the current TODO list for this chat run.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": TOOL_TODO_WRITE,
                "description": "Replace the TODO list. Keep it updated as work progresses — it survives pauses and resumes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {"type": "object", "required": ["content", "status"],
                                      "properties": {"content": {"type": "string"}, "status": {"type": "string", "enum": ["open", "in_progress", "done"]}}},
                        }
                    },
                    "required": ["items"],
                },
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": TOOL_DONE,
                "description": "Report that the user's request is fully handled. ALWAYS call this as the final action once the task is complete — a run that ends without it is treated as unfinished and the user is asked to continue it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "1-2 Sätze: was erreicht wurde"},
                    },
                    "required": ["summary"],
                },
            }
        }),
    ];
    v
}

fn allowed_tool_names() -> std::collections::HashSet<&'static str> {
    RETRIEVAL_TOOLS
        .iter()
        .copied()
        .chain([TOOL_TODO_READ, TOOL_TODO_WRITE, TOOL_DONE])
        .collect()
}

fn system_prompt() -> String {
    "Du bist der RAGnarök-Chat-Assistent für ein lokales, datenschutzfreundliches \
     Dokumentenarchiv. Antworte auf Deutsch. Du greifst über die bereitgestellten \
     Werkzeuge direkt auf den Suchindex zu — nutze `search` statt zu raten, und \
     `read_document` für den Volltext einer Fundstelle. Belege Aussagen immer mit \
     der zugehörigen Quelle (Pfad + Seite). Halte Antworten strukturiert und knapp. \
     Verwende todo_read/todo_write, um bei mehrstufiger Arbeit eine TODO-Liste zu \
     führen — sie überlebt Pausen und Fortsetzungen; hake Erledigtes ab. \
     WICHTIG: Melde den Abschluss IMMER mit dem Werkzeug task_done, sobald die \
     Anfrage vollständig bearbeitet ist — endet der Lauf ohne task_done, gilt die \
     Aufgabe als NICHT erledigt und der Nutzer muss manuell fortsetzen."
        .to_owned()
}

// ---------------------------------------------------------------------------
// resumable context
// ---------------------------------------------------------------------------

/// Resumable agent context persisted to `chat_threads.pending_state` —
/// a /continue works from any later session (runs live in spawned tasks;
/// disconnects never abort them). Port of RiskMaster's AgentResume.
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AgentResume {
    messages: Vec<Value>,
    content_acc: String,
    reasoning_acc: String,
    tool_log: Vec<Value>,
    model: String,
    steps_done: usize,
    #[serde(default)]
    todos: Vec<Value>,
    /// When this context was checkpointed (prefix-cache staleness signal).
    #[serde(default)]
    saved_at: Option<i64>,
}

fn resume_snapshot(
    messages: &[Value],
    content_acc: &str,
    reasoning_acc: &str,
    tool_log: &[Value],
    model: &str,
    steps_done: usize,
    todos: &[Value],
) -> AgentResume {
    AgentResume {
        messages: messages.to_vec(),
        content_acc: content_acc.to_owned(),
        reasoning_acc: reasoning_acc.to_owned(),
        tool_log: tool_log.to_vec(),
        model: model.to_owned(),
        steps_done,
        todos: todos.to_vec(),
        saved_at: Some(chrono::Utc::now().timestamp()),
    }
}

/// Display snapshot of the in-flight assistant bubble (`live_state`) —
/// a page reload restores exactly what the user saw.
fn live_state_json(
    content: &str,
    reasoning: &str,
    tool_log: &[Value],
    model: &str,
    inflight_tool: Option<(&str, i64)>,
) -> Value {
    let mut events: Vec<Value> = tool_log
        .iter()
        .map(|e| {
            json!({
                "name": e["name"], "status": e["status"],
                "durationMs": e["durationMs"], "args": e["arguments"],
                "pos": e["pos"], "ref": e["ref"],
            })
        })
        .collect();
    if let Some((name, pos)) = inflight_tool {
        events.push(json!({ "name": name, "status": "running", "pos": pos }));
    }
    json!({
        "content": content, "reasoning": reasoning,
        "model": model, "toolEvents": events,
    })
}

async fn save_checkpoint(st: &ChatState, tid: &str, resume: &AgentResume) {
    let Ok(pending) = serde_json::to_string(resume) else {
        return;
    };
    let live = live_state_json(
        &resume.content_acc,
        &resume.reasoning_acc,
        &resume.tool_log,
        &resume.model,
        None,
    )
    .to_string();
    st.app.engine.meta.chat_checkpoint(tid, &pending, &live);
}

fn save_live(st: &ChatState, tid: &str, live: Value) {
    st.app.engine.meta.chat_save_live(tid, &live.to_string());
}

/// Park a resumable run: persist the context, release the claim and tell
/// the client via `awaiting_continue` — identical exit path for stop,
/// step budget and unfinished runs.
async fn park_run(
    st: &ChatState,
    tx: &tokio::sync::mpsc::Sender<Result<Bytes, Infallible>>,
    tid: &str,
    pending: AgentResume,
    reason: &str,
) {
    let steps_done = pending.steps_done;
    if let Ok(p) = serde_json::to_string(&pending) {
        let live = live_state_json(
            &pending.content_acc,
            &pending.reasoning_acc,
            &pending.tool_log,
            &pending.model,
            None,
        )
        .to_string();
        st.app.engine.meta.chat_park_run(tid, &p, &live);
    }
    sse(
        tx,
        &json!({
            "type": "awaiting_continue",
            "stepsDone": steps_done,
            "message": reason,
        }),
    )
    .await;
}

// ---------------------------------------------------------------------------
// the loop
// ---------------------------------------------------------------------------

/// Everything a spawn needs to resume a run — mirrors RiskMaster's
/// continue path where `resume` replaces the rebuilt history.
pub struct RunParams {
    pub thread_id: String,
    pub model: String,
    pub resume: Option<AgentResume>,
    pub max_steps: i64,
    pub context_tokens: i64,
    pub chat_max_tokens: i64,
    /// Resolved document scope of the thread owner — snapshot at claim
    /// time; every retrieval tool enforces it.
    pub perm: ragnarok_core::scope::Permitted,
    /// The run's frame channel — `post_message`/`continue` create it and
    /// `tee_run` taps it for live attachers.
    pub tx: tokio::sync::mpsc::Sender<Result<Bytes, Infallible>>,
}

/// Spawn the agent task; errors inside are caught and emitted as an
/// `error` event + claim release so a failed run never bricks the thread.
pub fn spawn_agent(st: Arc<ChatState>, params: RunParams) {
    let tx = params.tx.clone();
    tokio::spawn(async move {
        let tid = params.thread_id.clone();
        if let Err(e) = agent_inner(&st, params).await {
            // Release the claim; a failed resume keeps pending_state so
            // /continue can be retried (same guarantee as RiskMaster).
            let _ = st.app.engine.meta.chat_finish_run(&tid, "[]");
            sse(&tx, &json!({ "type": "error", "message": e })).await;
        }
    });
}

async fn agent_inner(st: &Arc<ChatState>, params: RunParams) -> Result<(), String> {
    let meta = &st.app.engine.meta;
    let settings = meta.chat_settings().map_err(|e| e.to_string())?;
    if settings.base_url.trim().is_empty() {
        return Err(
            "Kein LLM-Provider konfiguriert — Basis-URL unter PUT /api/v1/chat/settings setzen."
                .to_owned(),
        );
    }
    let tx = params.tx.clone();
    let tid = params.thread_id.as_str();

    let mut todos: Vec<Value> = params
        .resume
        .as_ref()
        .map(|r| r.todos.clone())
        .unwrap_or_default();
    let perm = params.perm.clone();
    let resume_saved_at = params.resume.as_ref().and_then(|r| r.saved_at);
    let (mut messages, mut content_acc, mut reasoning_acc, mut tool_log, mut steps_done) =
        match params.resume {
            Some(r) => (
                r.messages,
                r.content_acc,
                r.reasoning_acc,
                r.tool_log,
                r.steps_done,
            ),
            None => {
                let history = meta
                    .chat_history(tid, HISTORY_LIMIT)
                    .map_err(|e| e.to_string())?;
                let mut messages = vec![json!({ "role": "system", "content": system_prompt() })];
                for (_id, _seq, role, content, _c, _r, _m, _at) in &history {
                    if role == "user" || role == "assistant" {
                        messages.push(json!({ "role": role, "content": content }));
                    }
                }
                (messages, String::new(), String::new(), Vec::new(), 0usize)
            }
        };

    // Cold-prefix elision on stale resumes — same economics as RiskMaster.
    let stale_prefix = resume_saved_at.is_none_or(|t| t < chrono::Utc::now().timestamp() - 20 * 60);
    if stale_prefix {
        elide_superseded_tool_results(&mut messages);
    }

    let tools = openai_tools();
    let allowed = allowed_tool_names();
    let mut read_dedup = ReadDedup::default();
    let base = settings.base_url.trim().trim_end_matches('/').to_owned();
    let url = format!("{base}/chat/completions");
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(CONNECT_TIMEOUT_SECS))
        .build()
        .map_err(|e| e.to_string())?;

    let max_steps = params.max_steps.max(1) as usize;
    let mut finished = false;
    let mut empty_step_nudged = false;
    let mut task_done_called = false;

    for _step in 0..max_steps {
        steps_done += 1;
        meta.chat_heartbeat(tid);
        save_checkpoint(
            st,
            tid,
            &resume_snapshot(
                &messages,
                &content_acc,
                &reasoning_acc,
                &tool_log,
                &params.model,
                steps_done,
                &todos,
            ),
        )
        .await;
        // --- stop/steer control: drain queued steering into the context;
        // a stop parks the run resumably.
        let (cancelled, steered) = meta.chat_poll_control(tid);
        for s in steered {
            messages.push(json!({
                "role": "user",
                "content": format!("Lenkung des Benutzers während des Laufs: {s}"),
            }));
            sse(&tx, &json!({ "type": "steered" })).await;
        }
        if cancelled {
            park_run(
                st,
                &tx,
                tid,
                resume_snapshot(
                    &messages,
                    &content_acc,
                    &reasoning_acc,
                    &tool_log,
                    &params.model,
                    steps_done,
                    &todos,
                ),
                "Lauf vom Benutzer gestoppt — „Fortsetzen“ setzt ihn fort.",
            )
            .await;
            return Ok(());
        }
        // --- compaction at 75% of the configured window ---
        let est_tokens = serde_json::to_string(&messages)
            .map(|t| t.len() / CHARS_PER_TOKEN)
            .unwrap_or(0);
        if est_tokens > (params.context_tokens as f64 * COMPACT_AT) as usize {
            messages = compact_context(&client, &url, &settings, &params.model, &messages).await;
            sse(
                &tx,
                &json!({"type": "compacted", "message": "Kontext verdichtet"}),
            )
            .await;
        }

        let mut body = json!({
            "model": params.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "stream": true,
            "temperature": 0.3,
            "max_tokens": effective_max_tokens(
                params.chat_max_tokens, settings.provider_max_tokens, DEFAULT_CHAT_MAX_TOKENS),
            "stream_options": { "include_usage": true },
        });
        apply_thinking_mode(&mut body, &settings.reasoning_effort);

        let resp = client
            .post(&url)
            .bearer_auth(&settings.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("LLM request failed: {e}"))?;
        if !resp.status().is_success() {
            let code = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!(
                "LLM Fehler {code}: {}",
                text.chars().take(300).collect::<String>()
            ));
        }

        // --- parse the SSE stream (verbatim port of RiskMaster's parser) ---
        let mut delta_content = String::new();
        let mut delta_reasoning = String::new();
        let mut pending_calls: std::collections::BTreeMap<String, (String, String, String)> =
            Default::default();
        let mut finish_reason = String::new();
        let mut buf = Vec::new();
        let mut byte_stream = resp.bytes_stream();
        let mut last_control_poll = std::time::Instant::now();
        let mut last_live_flush = std::time::Instant::now();
        let mut cancelled_mid_stream = false;
        loop {
            let chunk = match tokio::time::timeout(
                std::time::Duration::from_secs(STREAM_IDLE_SECS),
                byte_stream.next(),
            )
            .await
            {
                Ok(Some(Ok(chunk))) => chunk,
                Ok(Some(Err(e))) => return Err(format!("stream error: {e}")),
                Ok(None) => break,
                Err(_) => {
                    return Err("LLM-Stream: keine Antwort innerhalb von 5 Minuten".to_owned())
                }
            };
            if last_control_poll.elapsed() > std::time::Duration::from_millis(CONTROL_POLL_MS) {
                last_control_poll = std::time::Instant::now();
                if meta.chat_is_cancel(tid) {
                    cancelled_mid_stream = true;
                    break;
                }
            }
            if last_live_flush.elapsed() > std::time::Duration::from_secs(LIVE_FLUSH_SECS) {
                last_live_flush = std::time::Instant::now();
                save_live(
                    st,
                    tid,
                    live_state_json(
                        &(content_acc.clone() + &delta_content),
                        &(reasoning_acc.clone() + &delta_reasoning),
                        &tool_log,
                        &params.model,
                        None,
                    ),
                );
            }
            buf.extend_from_slice(&chunk);
            for event in drain_sse_events(&mut buf) {
                for line in event.lines() {
                    let line = line.trim();
                    let Some(data) = line.strip_prefix("data:") else {
                        continue;
                    };
                    let data = data.trim();
                    if data == "[DONE]" {
                        continue;
                    }
                    let Ok(v) = serde_json::from_str::<Value>(data) else {
                        continue;
                    };
                    if let Some(choice) = v["choices"].get(0) {
                        if let Some(fr) = choice["finish_reason"].as_str() {
                            finish_reason = fr.to_owned();
                        }
                        let delta = &choice["delta"];
                        if let Some(c) = delta["content"].as_str() {
                            delta_content.push_str(c);
                            sse(&tx, &json!({ "type": "token", "delta": c })).await;
                        }
                        for key in ["reasoning_content", "reasoning"] {
                            if let Some(r) = delta[key].as_str() {
                                delta_reasoning.push_str(r);
                                sse(&tx, &json!({ "type": "reasoning", "delta": r })).await;
                            }
                        }
                        if let Some(calls) = delta["tool_calls"].as_array() {
                            for call in calls {
                                let key = pending_call_key(call, pending_calls.keys().last());
                                let entry = pending_calls.entry(key).or_default();
                                if let Some(id) = call["id"].as_str() {
                                    entry.0 = id.to_owned();
                                }
                                if let Some(name) = call["function"]["name"].as_str() {
                                    entry.1.push_str(name);
                                }
                                if let Some(args) = call["function"]["arguments"].as_str() {
                                    entry.2.push_str(args);
                                }
                            }
                        }
                    }
                }
            }
        }

        if cancelled_mid_stream {
            if !delta_content.is_empty() {
                messages.push(json!({"role": "assistant", "content": &delta_content}));
                content_acc.push_str(&delta_content);
            }
            park_run(
                st,
                &tx,
                tid,
                resume_snapshot(
                    &messages,
                    &content_acc,
                    &reasoning_acc,
                    &tool_log,
                    &params.model,
                    steps_done,
                    &todos,
                ),
                "Lauf vom Benutzer gestoppt — „Fortsetzen“ setzt ihn fort.",
            )
            .await;
            return Ok(());
        }

        content_acc.push_str(&delta_content);
        reasoning_acc.push_str(&delta_reasoning);

        let wants_tools = wants_tools(&finish_reason, pending_calls.is_empty());
        if !wants_tools {
            if delta_content.is_empty() && pending_calls.is_empty() {
                // Degenerate step (reasoning burned the budget): nudge
                // once, then fail loudly — same as RiskMaster.
                if empty_step_nudged {
                    return Err(
                        "Das Modell lieferte zwei leere Antworten in Folge — Lauf abgebrochen."
                            .to_owned(),
                    );
                }
                empty_step_nudged = true;
                messages.push(json!({
                    "role": "user",
                    "content": "Deine letzte Antwort war leer. Antworte jetzt direkt — Text oder Werkzeugaufrufe, ohne lange Denkphase.",
                }));
                continue;
            }
            if !task_done_called {
                // Stopped without task_done → unfinished: park (the client
                // shows "Fortsetzen") instead of faking a done.
                if !delta_content.is_empty() {
                    messages.push(json!({"role": "assistant", "content": &delta_content}));
                }
                messages.push(json!({
                    "role": "user",
                    "content": "Aufgabe noch nicht abgeschlossen — arbeite weiter, bis alles erledigt ist, und melde den Abschluss mit task_done.",
                }));
                park_run(
                    st,
                    &tx,
                    tid,
                    AgentResume {
                        messages,
                        content_acc,
                        reasoning_acc,
                        tool_log,
                        model: params.model.clone(),
                        steps_done,
                        todos: todos.clone(),
                        saved_at: Some(chrono::Utc::now().timestamp()),
                    },
                    "Das Modell hat die Aufgabe nicht als abgeschlossen gemeldet — „Fortsetzen“, um weiterzuarbeiten.",
                )
                .await;
                return Ok(());
            }
            finished = true;
            break;
        }

        // --- execute tool calls in-process ---
        let mut assistant_tool_calls = Vec::new();
        let mut tool_messages = Vec::new();
        for (id, name, args) in pending_calls.values() {
            // Stop between executions: run calls keep a schema-valid
            // assistant+tool pair; unrun calls are simply absent.
            if meta.chat_is_cancel(tid) {
                if !assistant_tool_calls.is_empty() {
                    messages.push(json!({
                        "role": "assistant",
                        "content": if delta_content.is_empty() { Value::Null } else { json!(delta_content) },
                        "tool_calls": assistant_tool_calls,
                    }));
                    messages.extend(tool_messages);
                }
                park_run(
                    st,
                    &tx,
                    tid,
                    resume_snapshot(
                        &messages,
                        &content_acc,
                        &reasoning_acc,
                        &tool_log,
                        &params.model,
                        steps_done,
                        &todos,
                    ),
                    "Lauf vom Benutzer gestoppt — „Fortsetzen“ setzt ihn fort.",
                )
                .await;
                return Ok(());
            }
            let arguments: Value = serde_json::from_str(args).unwrap_or_else(|_| json!({}));
            let display_args = truncate_chars(&args.to_string(), 500);
            let pos = content_acc.chars().count() as i64;
            sse(&tx, &json!({ "type": "tool_call", "name": name, "arguments": display_args, "argumentsFull": args, "pos": pos })).await;
            save_live(
                st,
                tid,
                live_state_json(
                    &content_acc,
                    &reasoning_acc,
                    &tool_log,
                    &params.model,
                    Some((name, pos)),
                ),
            );
            if !allowed.contains(name.as_str()) {
                sse(
                    &tx,
                    &json!({ "type": "tool_result", "name": name, "status": "denied" }),
                )
                .await;
                assistant_tool_calls.push(json!({
                    "id": id, "type": "function",
                    "function": { "name": name, "arguments": args },
                }));
                tool_messages.push(json!({
                    "role": "tool", "tool_call_id": id,
                    "content": "TOOL DENIED: Werkzeug ist in dieser Sitzung nicht erlaubt.",
                }));
                tool_log.push(json!({ "name": name, "status": "denied", "pos": pos }));
                continue;
            }
            if name.as_str() == TOOL_DONE {
                task_done_called = true;
                sse(
                    &tx,
                    &json!({ "type": "tool_result", "name": name, "status": "ok" }),
                )
                .await;
                tool_log.push(json!({"name": name, "arguments": arguments, "status": "ok", "durationMs": 0, "pos": pos}));
                assistant_tool_calls.push(
                    json!({"id": id, "type": "function", "function": {"name": name, "arguments": args}}),
                );
                tool_messages.push(
                    json!({"role": "tool", "tool_call_id": id, "content": "{\"status\":\"done\"}"}),
                );
                continue;
            }
            if name.as_str() == TOOL_TODO_READ || name.as_str() == TOOL_TODO_WRITE {
                if name.as_str() == TOOL_TODO_WRITE {
                    todos = arguments["items"].as_array().cloned().unwrap_or_default();
                }
                let result_text =
                    serde_json::to_string_pretty(&json!({"todos": todos})).unwrap_or_default();
                sse(
                    &tx,
                    &json!({ "type": "tool_result", "name": name, "status": "ok" }),
                )
                .await;
                sse(&tx, &json!({ "type": "todos", "todos": todos })).await;
                tool_log.push(json!({"name": name, "arguments": arguments, "status": "ok", "durationMs": 0, "pos": pos}));
                assistant_tool_calls.push(
                    json!({"id": id, "type": "function", "function": {"name": name, "arguments": args}}),
                );
                tool_messages
                    .push(json!({"role": "tool", "tool_call_id": id, "content": result_text}));
                continue;
            }
            // Read-dedup: an identical (tool, args) repeat returns the
            // earlier result by pointer — the full payload would re-enter
            // every following step's prompt otherwise.
            let dedup_key = ReadDedup::key(name.as_str(), &arguments);
            if read_dedup.seen.contains(&dedup_key) {
                sse(
                    &tx,
                    &json!({ "type": "tool_result", "name": name, "status": "ok", "dedup": true }),
                )
                .await;
                tool_log.push(json!({
                    "name": name, "arguments": arguments, "status": "ok",
                    "durationMs": 0, "pos": pos, "dedup": true,
                }));
                assistant_tool_calls.push(json!({
                    "id": id, "type": "function",
                    "function": { "name": name, "arguments": args },
                }));
                tool_messages.push(json!({
                    "role": "tool", "tool_call_id": id,
                    "content": ReadDedup::stub(),
                }));
                continue;
            }
            let call_started = std::time::Instant::now();
            let (result_text, status) =
                exec_tool(&st.app, name.as_str(), &arguments, &tx, &perm).await;
            let duration_ms = call_started.elapsed().as_millis() as i64;
            read_dedup.on_result(&dedup_key, status == "ok");
            sse(
                &tx,
                &json!({ "type": "tool_result", "name": name, "status": status }),
            )
            .await;
            tool_log.push(json!({
                "name": name, "arguments": arguments, "status": status,
                "durationMs": duration_ms, "pos": pos,
            }));
            save_live(
                st,
                tid,
                live_state_json(&content_acc, &reasoning_acc, &tool_log, &params.model, None),
            );
            assistant_tool_calls.push(json!({
                "id": id, "type": "function",
                "function": { "name": name, "arguments": args },
            }));
            tool_messages.push(json!({
                "role": "tool", "tool_call_id": id,
                "content": truncate_chars(&result_text, MAX_TOOL_RESULT_CHARS),
            }));
        }
        messages.push(json!({
            "role": "assistant",
            "content": if delta_content.is_empty() { Value::Null } else { json!(delta_content) },
            "tool_calls": assistant_tool_calls,
        }));
        messages.extend(tool_messages);
        save_checkpoint(
            st,
            tid,
            &resume_snapshot(
                &messages,
                &content_acc,
                &reasoning_acc,
                &tool_log,
                &params.model,
                steps_done,
                &todos,
            ),
        )
        .await;
        if task_done_called {
            finished = true;
            break;
        }
    }

    if !finished {
        park_run(
            st,
            &tx,
            tid,
            AgentResume {
                messages,
                content_acc,
                reasoning_acc,
                tool_log,
                model: params.model.clone(),
                steps_done,
                todos: todos.clone(),
                saved_at: Some(chrono::Utc::now().timestamp()),
            },
            "Schritt-Limit erreicht — „Fortsetzen“, um weiterzuarbeiten.",
        )
        .await;
        return Ok(());
    }

    // --- persist assistant message + tool call rows ---
    let _ = st.app.engine.meta.chat_finish_run(
        tid,
        &serde_json::to_string(&todos).unwrap_or_else(|_| "[]".into()),
    );
    let msg_id = uuid::Uuid::new_v4().to_string();
    let _ = st.app.engine.meta.chat_add_message(
        tid,
        &msg_id,
        "assistant",
        &content_acc,
        "[]",
        &reasoning_acc,
        &params.model,
    );
    for (i, call) in tool_log.iter().enumerate() {
        let _ =
            st.app
                .engine
                .meta
                .chat_add_tool_call(&ragnarok_store::metadata::ChatToolCallInsert {
                    message_id: &msg_id,
                    seq: i as i64,
                    name: call["name"].as_str().unwrap_or_default(),
                    arguments: call.get("arguments").map(|a| a.to_string()),
                    // Results are elided (can be megabytes) — the row keeps the
                    // sentinel, like RiskMaster's TOOL_RESULT_ELIDED.
                    result: Some("{\"elided\":true}".to_owned()),
                    status: call["status"].as_str().unwrap_or("ok"),
                    duration_ms: call["durationMs"].as_i64(),
                    pos: call["pos"].as_i64(),
                    ref_json: call
                        .get("ref")
                        .filter(|r| !r.is_null())
                        .map(|r| r.to_string()),
                });
    }
    sse(
        &tx,
        &json!({
            "type": "done",
            "messageId": msg_id,
            "model": params.model,
            "toolCalls": tool_log.iter().map(|c| json!({"name": c["name"], "status": c["status"]})).collect::<Vec<_>>(),
        }),
    )
    .await;
    Ok(())
}

/// Execute one chat tool against the search engine — scoped to the
/// thread owner's resolved permissions. `search` emits a `citation`
/// frame per hit — the UI renders them as deep-link chips.
async fn exec_tool(
    app: &Arc<crate::state::AppState>,
    name: &str,
    args: &Value,
    tx: &tokio::sync::mpsc::Sender<Result<Bytes, Infallible>>,
    perm: &ragnarok_core::scope::Permitted,
) -> (String, &'static str) {
    match name {
        "search" => {
            let query = args["query"].as_str().unwrap_or_default().to_owned();
            let limit = args["limit"].as_i64().unwrap_or(8).clamp(1, 25) as usize;
            let mut req = ragnarok_core::SearchRequest::new(&query);
            req.limit = limit;
            match app.engine.search_scoped(&req, perm).await {
                Ok(hits) => {
                    let mut out = Vec::new();
                    for (i, h) in hits.iter().enumerate() {
                        let cite = json!({
                            "n": i + 1,
                            "documentId": h.document_id.to_string(),
                            "path": h.path,
                            "page": h.page,
                            "score": h.score,
                        });
                        sse(tx, &json!({"type": "citation", "citation": cite})).await;
                        out.push(json!({
                            "n": i + 1, "documentId": h.document_id.to_string(),
                            "path": h.path, "page": h.page + 1,
                            "docType": format!("{:?}", h.doc_type),
                            "text": truncate_chars(&h.text, 1200),
                            "score": h.score,
                        }));
                    }
                    (
                        serde_json::to_string_pretty(&json!({"hits": out})).unwrap_or_default(),
                        "ok",
                    )
                }
                Err(e) => (format!("Suchfehler: {e}"), "error"),
            }
        }
        "read_document" => {
            let id = args["documentId"].as_str().unwrap_or_default();
            let Some(doc_id) = uuid::Uuid::parse_str(id)
                .ok()
                .map(ragnarok_core::DocumentId::from)
            else {
                return ("Ungültige documentId".to_owned(), "error");
            };
            if !perm.allows(&doc_id.to_string()) {
                return ("Dokument nicht gefunden".to_owned(), "error");
            }
            match app.engine.meta.get_document(doc_id) {
                Ok(Some(rec)) => match app.engine.meta.chunks_for_document(doc_id) {
                    Ok(chunks) => {
                        let text = chunks
                            .iter()
                            .map(|c| c.text.as_str())
                            .collect::<Vec<_>>()
                            .join("\n\n");
                        (
                            serde_json::to_string_pretty(&json!({
                                "path": rec.path, "docType": format!("{:?}", rec.doc_type),
                                "chunkCount": chunks.len(), "text": text,
                            }))
                            .unwrap_or_default(),
                            "ok",
                        )
                    }
                    Err(e) => (e.to_string(), "error"),
                },
                Ok(None) => ("Dokument nicht gefunden".to_owned(), "error"),
                Err(e) => (e.to_string(), "error"),
            }
        }
        "list_documents" => {
            let limit = args["limit"].as_i64().unwrap_or(50).clamp(1, 200) as usize;
            match app.engine.meta.list_documents(None) {
                Ok(docs) => {
                    let rows: Vec<Value> = docs
                        .iter()
                        .filter(|d| perm.allows(&d.id.to_string()))
                        .take(limit)
                        .map(|d| {
                            json!({
                                "documentId": d.id.to_string(), "path": d.path,
                                "docType": format!("{:?}", d.doc_type),
                                "chunks": d.chunk_count,
                            })
                        })
                        .collect();
                    (
                        serde_json::to_string_pretty(&json!({"documents": rows}))
                            .unwrap_or_default(),
                        "ok",
                    )
                }
                Err(e) => (e.to_string(), "error"),
            }
        }
        "list_sources" => match app.engine.meta.list_sources() {
            Ok(srcs) => {
                let rows: Vec<Value> = srcs
                    .iter()
                    .map(|s| {
                        json!({
                            "sourceId": s.id.to_string(), "name": s.name,
                            "config": s.config,
                        })
                    })
                    .collect();
                (
                    serde_json::to_string_pretty(&json!({"sources": rows})).unwrap_or_default(),
                    "ok",
                )
            }
            Err(e) => (e.to_string(), "error"),
        },
        "system_stats" => {
            let st = app.stats();
            (
                serde_json::to_string_pretty(&json!({
                    "indexedDocuments": st.indexed_documents,
                    "indexedChunks": st.indexed_chunks,
                    "budgetBytes": st.budget_bytes,
                    "pressure": format!("{:?}", st.pressure),
                    "cpuPercent": st.cpu_percent,
                }))
                .unwrap_or_default(),
                "ok",
            )
        }
        _ => (format!("Unbekanntes Werkzeug: {name}"), "error"),
    }
}

// ---------------------------------------------------------------------------
// helpers (verbatim ports of RiskMaster's, minus Postgres-specifics)
// ---------------------------------------------------------------------------

/// Apply the configured reasoning effort — OpenAI + Z.AI spelling; the
/// `off`/`auto` pass-through covers vLLM's `enable_thinking` knob.
pub fn apply_thinking_mode(body: &mut Value, effort: &str) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    match effort {
        "" | "auto" => {}
        "off" => {
            obj.insert(
                "chat_template_kwargs".to_owned(),
                json!({ "enable_thinking": false }),
            );
            obj.insert("thinking".to_owned(), json!({ "type": "disabled" }));
            obj.insert("reasoning_effort".to_owned(), json!("none"));
        }
        v => {
            obj.insert("reasoning_effort".to_owned(), json!(v));
            obj.insert("thinking".to_owned(), json!({ "type": "enabled" }));
        }
    }
}

/// `usecase_setting` wins, then the provider default, then the builtin.
fn effective_max_tokens(usecase_setting: i64, provider_max: i64, builtin: i64) -> i64 {
    let v = if usecase_setting > 0 {
        usecase_setting
    } else if provider_max > 0 {
        provider_max
    } else {
        builtin
    };
    v.clamp(1, 1_000_000)
}

/// Run-local dedup for read tools — identical (tool, args) repeats get a
/// pointer stub; all chat tools are reads so there is no write-invalidate.
#[derive(Default)]
struct ReadDedup {
    seen: std::collections::HashSet<String>,
}

impl ReadDedup {
    fn key(name: &str, args: &Value) -> String {
        format!("{name}\u{1f}{args}")
    }
    fn stub() -> &'static str {
        "{\"unchanged\":\"Identischer Aufruf wurde in diesem Lauf bereits beantwortet — das Ergebnis steht weiter oben.\"}"
    }
    fn on_result(&mut self, key: &str, ok: bool) {
        if ok {
            self.seen.insert(key.to_owned());
        }
    }
}

/// Stub out tool results superseded by a later identical call — only
/// safe during compaction/cold resumes (see RiskMaster's comments on the
/// token economics of mid-run elision).
fn elide_superseded_tool_results(messages: &mut [Value]) {
    let mut calls: std::collections::HashMap<String, (String, String)> =
        std::collections::HashMap::new();
    for m in messages.iter() {
        for tc in m["tool_calls"].as_array().into_iter().flatten() {
            let Some(id) = tc["id"].as_str() else {
                continue;
            };
            let name = tc["function"]["name"].as_str().unwrap_or("").to_owned();
            let args = tc["function"]["arguments"].as_str().unwrap_or("");
            let args = serde_json::from_str::<Value>(args)
                .map(|v| v.to_string())
                .unwrap_or_else(|_| args.to_owned());
            calls.insert(id.to_owned(), (name, args));
        }
    }
    let mut last: std::collections::HashMap<(String, String), usize> =
        std::collections::HashMap::new();
    for (i, m) in messages.iter().enumerate() {
        if m["role"].as_str() != Some("tool") {
            continue;
        }
        if let Some(key) = m["tool_call_id"]
            .as_str()
            .and_then(|id| calls.get(id).cloned())
        {
            last.insert(key, i);
        }
    }
    for (i, m) in messages.iter_mut().enumerate() {
        if m["role"].as_str() != Some("tool") {
            continue;
        }
        let Some(key) = m["tool_call_id"]
            .as_str()
            .and_then(|id| calls.get(id).cloned())
        else {
            continue;
        };
        if last.get(&key) != Some(&i) {
            let chars = m["content"].as_str().map(str::len).unwrap_or(0);
            if chars > 200 {
                m["content"] = json!(format!(
                    "{{\"elided\": \"superseded by a later {} call ({} chars)\"}}",
                    key.0, chars
                ));
            }
        }
    }
}

/// Context compaction — the model itself summarizes the older portion;
/// the system prompt and the newest exchanges stay verbatim.
async fn compact_context(
    client: &reqwest::Client,
    url: &str,
    settings: &ragnarok_store::metadata::ChatSettingsRow,
    model: &str,
    messages: &[Value],
) -> Vec<Value> {
    let mut work = messages.to_vec();
    elide_superseded_tool_results(&mut work);
    let boundary = compact_boundary(&work);
    let (head, tail) = work.split_at(boundary);
    if head.len() <= 1 {
        return messages.to_vec();
    }
    let mut out = Vec::new();
    out.extend(head.iter().take(1).cloned()); // system
    let old = serde_json::to_string(&head[1..]).unwrap_or_default();
    let prompt = format!(
        "Fasse den bisherigen Gesprächs- und Werkzeugverlauf kompakt zusammen auf Deutsch. \
         Erhalte: Ziel des Nutzers, konkrete Dokument-IDs/Fakten, was bereits ERLEDIGT ist, \
         offene nächste Schritte und den TODO-Stand. Nur die Zusammenfassung, max. 800 Wörter.\n\n{old}"
    );
    let mut body = json!({
        "model": model, "stream": false, "temperature": 0.2,
        "max_tokens": 30_000,
        "messages": [
            {"role": "system", "content": "Du verdichtest Kontext für einen Agentenlauf."},
            {"role": "user", "content": prompt},
        ],
    });
    apply_thinking_mode(&mut body, &settings.reasoning_effort);
    let summary = match client
        .post(url)
        .bearer_auth(&settings.api_key)
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r.json::<Value>().await.ok().and_then(|v| {
            v["choices"][0]["message"]["content"]
                .as_str()
                .map(str::to_owned)
        }),
        Err(_) => None,
    }
    .unwrap_or_else(|| old.chars().take(4_000).collect::<String>());
    out.push(json!({
        "role": "assistant",
        "content": format!("[Kontextzusammenfassung]\n{summary}"),
    }));
    out.extend(tail.iter().cloned());
    out
}

/// Where compaction may cut: never inside a tool group — `tool` results
/// whose call lands in the summarized head would be orphans and fail
/// schema validation. Verbatim port of RiskMaster's `compact_boundary`.
fn compact_boundary(messages: &[Value]) -> usize {
    const KEEP_TAIL: usize = 10;
    let mut boundary = messages.len().saturating_sub(KEEP_TAIL).max(1);
    while boundary < messages.len() && messages[boundary]["role"].as_str() == Some("tool") {
        boundary += 1;
    }
    boundary.min(messages.len())
}

/// Drain complete SSE events out of `buf` — verbatim port: normalizes
/// `\r\n`/bare `\r` to `\n`, keeps a trailing half-line buffered.
fn drain_sse_events(buf: &mut Vec<u8>) -> Vec<String> {
    let mut work = Vec::with_capacity(buf.len());
    let mut i = 0;
    while i < buf.len() {
        match buf[i] {
            b'\r' if i + 1 < buf.len() => {
                work.push(b'\n');
                i += if buf[i + 1] == b'\n' { 2 } else { 1 };
            }
            b'\r' => break,
            b => {
                work.push(b);
                i += 1;
            }
        }
    }
    let mut events = Vec::new();
    while let Some(pos) = find_subslice(&work, b"\n\n") {
        let event_bytes: Vec<u8> = work.drain(..pos + 2).collect();
        events.push(String::from_utf8_lossy(&event_bytes).into_owned());
    }
    work.extend_from_slice(&buf[i..]);
    *buf = work;
    events
}

/// Accumulation key for streamed `tool_calls` deltas — verbatim port:
/// prefer `index`, fall back to `id`, else append to the last open call.
fn pending_call_key(call: &Value, last_key: Option<&String>) -> String {
    if let Some(i) = call["index"].as_i64() {
        format!("{i:010}")
    } else if let Some(id) = call["id"].as_str() {
        format!("id:{id}")
    } else {
        last_key.cloned().unwrap_or_else(|| "0".to_owned())
    }
}

/// Execute pending tool calls? `finish_reason` is `tool_calls` per spec;
/// empty reason with pending calls runs them too; anything else ends.
fn wants_tools(finish_reason: &str, pending_empty: bool) -> bool {
    !pending_empty && matches!(finish_reason, "" | "tool_calls")
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_owned()
    } else {
        let mut out: String = s.chars().take(max).collect();
        out.push('…');
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str) -> Value {
        json!({ "role": role, "content": "x" })
    }

    #[test]
    fn drain_sse_events_splits_on_blank_line() {
        let mut buf = b"data: {}\n\ndata: {}\n\n".to_vec();
        let events = drain_sse_events(&mut buf);
        assert_eq!(events.len(), 2);
        assert!(buf.is_empty());
    }

    #[test]
    fn drain_sse_events_normalizes_crlf_and_bare_cr() {
        let mut buf = b"data: a\r\n\r\ndata: b\r\rdata: c\n\n".to_vec();
        let events = drain_sse_events(&mut buf);
        assert_eq!(events.len(), 3);
        assert!(events[0].contains("data: a"));
        assert!(events[2].contains("data: c"));
    }

    #[test]
    fn drain_sse_events_keeps_incomplete_tail() {
        let mut buf = b"data: {\"part".to_vec();
        assert!(drain_sse_events(&mut buf).is_empty());
        buf.extend_from_slice(b"1\": 1}\n\n");
        let events = drain_sse_events(&mut buf);
        assert_eq!(events.len(), 1);
        assert!(events[0].contains("\"part1\": 1"));
    }

    #[test]
    fn pending_call_key_prefers_index() {
        let call = json!({ "index": 3, "id": "call_9" });
        assert_eq!(pending_call_key(&call, None), "0000000003");
    }

    #[test]
    fn pending_call_key_falls_back_to_id() {
        let call = json!({ "id": "call_42" });
        assert_eq!(pending_call_key(&call, None), "id:call_42");
    }

    #[test]
    fn pending_call_key_appends_to_last_open_call() {
        let last = "0000000001".to_owned();
        let call = json!({ "function": { "arguments": "..." } });
        assert_eq!(pending_call_key(&call, Some(&last)), "0000000001");
        assert_eq!(pending_call_key(&call, None), "0");
    }

    #[test]
    fn compact_boundary_skips_leading_tool_group() {
        let mut m = vec![msg("system")];
        for _ in 0..8 {
            m.push(msg("user"));
        }
        m.push(msg("assistant"));
        m.extend([msg("tool"), msg("tool")]);
        for _ in 0..8 {
            m.push(msg("user"));
        }
        let b = compact_boundary(&m);
        assert_eq!(b, 12);
        let (head, tail) = m.split_at(b);
        assert_eq!(head.last().unwrap()["role"], "tool");
        assert_eq!(tail[0]["role"], "user");
    }

    #[test]
    fn compact_boundary_never_leaves_tool_at_tail_head() {
        let mut m = vec![msg("system")];
        for i in 0..20 {
            m.push(if i == 9 {
                msg("assistant")
            } else {
                msg("user")
            });
        }
        m.extend([msg("tool"), msg("tool")]);
        let b = compact_boundary(&m);
        let (_head, tail) = m.split_at(b);
        assert_ne!(tail.first().and_then(|v| v["role"].as_str()), Some("tool"));
    }

    #[test]
    fn compact_boundary_small_context_keeps_system_only_head() {
        let m = vec![msg("system"), msg("user"), msg("assistant")];
        assert_eq!(compact_boundary(&m), 1);
    }

    #[test]
    fn truncate_chars_appends_ellipsis_and_counts_chars_not_bytes() {
        assert_eq!(truncate_chars("abc", 10), "abc");
        assert_eq!(truncate_chars("abcdef", 3), "abc…");
        assert_eq!(truncate_chars("🦀🦀🦀🦀", 3), "🦀🦀🦀…");
    }

    #[test]
    fn find_subslice_matches_windows() {
        assert_eq!(find_subslice(b"hello\n\nworld", b"\n\n"), Some(5));
        assert_eq!(find_subslice(b"abc", b"\n\n"), None);
    }

    #[test]
    fn wants_tools_runs_pending_on_empty_or_tool_calls_reason() {
        assert!(wants_tools("", false));
        assert!(wants_tools("tool_calls", false));
        assert!(!wants_tools("stop", false));
        assert!(!wants_tools("length", false));
        assert!(!wants_tools("", true));
        assert!(!wants_tools("tool_calls", true));
    }

    #[test]
    fn tool_surface_is_exactly_the_retrieval_set() {
        let allowed = allowed_tool_names();
        for name in [
            "search",
            "read_document",
            "list_documents",
            "list_sources",
            "system_stats",
            TOOL_TODO_READ,
            TOOL_TODO_WRITE,
            TOOL_DONE,
        ] {
            assert!(allowed.contains(name), "{name} missing");
        }
        let tools = openai_tools();
        let offered: Vec<&str> = tools
            .iter()
            .filter_map(|t| t["function"]["name"].as_str())
            .collect();
        assert_eq!(offered.len(), allowed.len());
        for n in &offered {
            assert!(allowed.contains(n));
        }
    }

    #[test]
    fn apply_thinking_mode_maps_efforts() {
        let mut body = json!({"model": "m"});
        apply_thinking_mode(&mut body, "");
        assert!(body.get("reasoning_effort").is_none());
        apply_thinking_mode(&mut body, "off");
        assert_eq!(body["reasoning_effort"], "none");
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
        let mut body = json!({"model": "m"});
        apply_thinking_mode(&mut body, "low");
        assert_eq!(body["reasoning_effort"], "low");
        assert_eq!(body["thinking"]["type"], "enabled");
    }

    #[test]
    fn elision_stubs_older_duplicate_results() {
        fn tc(id: &str) -> Value {
            json!({"role": "assistant", "tool_calls": [{"id": id, "type": "function",
                  "function": {"name": "search", "arguments": "{}"}}]})
        }
        fn tr(id: &str, c: &str) -> Value {
            json!({"role": "tool", "tool_call_id": id, "content": c})
        }
        let big = "x".repeat(500);
        let mut m = vec![tc("c1"), tr("c1", &big), tc("c2"), tr("c2", &big)];
        elide_superseded_tool_results(&mut m);
        assert!(m[1]["content"].as_str().unwrap().contains("elided"));
        assert_eq!(m[3]["content"].as_str().unwrap(), big);
    }

    #[test]
    fn read_dedup_stubs_repeats() {
        let mut d = ReadDedup::default();
        let k = ReadDedup::key("search", &json!({"query": "a"}));
        assert!(!d.seen.contains(&k));
        d.on_result(&k, true);
        assert!(d.seen.contains(&k));
        d.on_result("other", false); // failed calls don't poison anything
        assert!(d.seen.contains(&k));
        let shuffled = ReadDedup::key("search", &json!({"query": "a"}));
        assert_eq!(k, shuffled);
    }
}
