//! MCP (Model Context Protocol) server surface.
//!
//! Transport: `POST /api/mcp` — JSON-RPC 2.0, single or batch body.
//! Auth: `Authorization: Bearer <session|ragk_…>` via `SessionAuth`, so
//! API keys minted in Settings work directly (Claude Desktop → stdio
//! proxy `ragnarok mcp`). Every `tools/call` re-resolves the caller's
//! document scope — a key-bound scope narrows, never widens.
//!
//! Exposed tools (all read-only by design):
//!   search, read_document, list_documents, list_sources, list_filters
//! Plus protocol methods: initialize, ping, tools/list, tools/call.

use crate::auth::SessionAuth;
use crate::routes::ApiError;
use crate::state::AppState;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use ragnarok_core::scope::Permitted;
use ragnarok_store::metadata::UserRow;
use serde_json::{json, Value};
use std::sync::Arc;

/// JSON-RPC error codes.
const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;

fn err(id: &Value, code: i64, msg: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id.clone(), "error": {"code": code, "message": msg}})
}

fn ok(id: &Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id.clone(), "result": result})
}

fn text_result(id: &Value, payload: Value, is_error: bool) -> Value {
    ok(
        id,
        json!({
            "content": [{"type": "text", "text": serde_json::to_string_pretty(&payload).unwrap_or_default()}],
            "isError": is_error,
        }),
    )
}

/// Advertised tool schemas — kept in lockstep with `agent.rs`'s
/// OpenAI surface (same names/semantics so agents port cleanly).
fn tool_schemas() -> Vec<Value> {
    vec![
        json!({
            "name": "search",
            "description": "Hybrid-Suche über den RAGnarök-Index (dense+sparse+ColBERT+RRF+rerank). Liefert Treffer mit documentId, path, page, score, Text-Snippet.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Suchanfrage (natürliche Sprache oder Schlüsselwörter)"},
                    "limit": {"type": "integer", "description": "max. Treffer (1-25, default 8)"},
                    "filterId": {"type": "string", "description": "optional: gespeicherter Filter zur Einschränkung (aus list_filters)"}
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "read_document",
            "description": "Volltext + Metadaten eines Dokuments anhand seiner documentId (aus search/list_documents).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "documentId": {"type": "string"},
                    "maxChars": {"type": "integer", "description": "Volltext-Kap (default 20000)"}
                },
                "required": ["documentId"]
            }
        }),
        json!({
            "name": "list_documents",
            "description": "Dokumente auflisten (scope-gefiltert). Optional nach Quelle/Typ filtern, paginiert.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sourceId": {"type": "string"},
                    "docType": {"type": "string", "description": "pdf|email|text|markdown|html|code|image|word|spreadsheet|presentation"},
                    "query": {"type": "string", "description": "Pfad-Substring-Filter"},
                    "limit": {"type": "integer", "description": "default 50, max 200"},
                    "offset": {"type": "integer"}
                }
            }
        }),
        json!({
            "name": "list_sources",
            "description": "Alle konfigurierten Quellen mit Typ, Status und Dateizählern.",
            "inputSchema": {"type": "object", "properties": {}}
        }),
        json!({
            "name": "list_filters",
            "description": "Gespeicherte Suchfilter, die der Aufrufer nutzen darf (eigene + geteilte + globale) — IDs für search.filterId.",
            "inputSchema": {"type": "object", "properties": {}}
        }),
        json!({
            "name": "system_stats",
            "description": "Index-/Ressourcen-Status: Dokumente, Chunks, Queue, RAM-Budget, Pressure-Level.",
            "inputSchema": {"type": "object", "properties": {}}
        }),
    ]
}

fn initialize_result() -> Value {
    json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {"listChanged": false}},
        "serverInfo": {"name": "ragnarok", "version": env!("CARGO_PKG_VERSION")},
        "instructions": concat!(
            "RAGnarök embedded vector search. Nutze `search` für jede Wissensfrage — ",
            "rate niemals Dokumentinhalte. Treffer liefern documentId+page; ",
            "`read_document` gibt den Volltext. `list_filters` zeigt nutzbare ",
            "Scope-Filter für search.filterId. Alle Tools respektieren die ",
            "Dokument-Grants des Tokens-Inhabers (API-Keys können weiter einschränken)."
        )
    })
}

/// Caller scope for one MCP call: user grants ∧ ragk_-key scope ∧
/// optional `filterId` argument (must be owned/global/granted).
fn call_scope(
    st: &AppState,
    user: &UserRow,
    key_scope: Option<&str>,
    filter_id: Option<&str>,
) -> ragnarok_core::Result<Permitted> {
    crate::routes::effective_search_scope(&st.engine.meta, user, key_scope, filter_id, None)
}

async fn tool_search(
    st: &AppState,
    perm: &Permitted,
    args: &Value,
) -> std::result::Result<Value, String> {
    let query = args["query"].as_str().unwrap_or_default().to_owned();
    if query.trim().is_empty() {
        return Err("argument 'query' is required".into());
    }
    let limit = args["limit"].as_i64().unwrap_or(8).clamp(1, 25) as usize;
    let mut req = ragnarok_core::SearchRequest::new(&query);
    req.limit = limit;
    let hits = st
        .engine
        .search_scoped(&req, perm)
        .await
        .map_err(|e| e.to_string())?;
    let out: Vec<Value> = hits
        .iter()
        .enumerate()
        .map(|(i, h)| {
            json!({
                "n": i + 1,
                "documentId": h.document_id.to_string(),
                "path": h.path,
                "page": h.page + 1,
                "docType": format!("{:?}", h.doc_type),
                "score": h.score,
                "text": truncate(&h.text, 1500),
            })
        })
        .collect();
    Ok(json!({"hits": out, "count": out.len()}))
}

fn tool_read_document(st: &AppState, perm: &Permitted, args: &Value) -> Result<Value, String> {
    let id = args["documentId"].as_str().unwrap_or_default();
    let doc_id = uuid::Uuid::parse_str(id)
        .ok()
        .map(ragnarok_core::DocumentId::from)
        .ok_or_else(|| "invalid documentId".to_owned())?;
    if !perm.allows(&doc_id.to_string()) {
        return Err("document not found".into());
    }
    let rec = st
        .engine
        .meta
        .get_document(doc_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "document not found".to_owned())?;
    let chunks = st
        .engine
        .meta
        .chunks_for_document(doc_id)
        .map_err(|e| e.to_string())?;
    let cap = args["maxChars"]
        .as_i64()
        .unwrap_or(20000)
        .clamp(500, 200_000) as usize;
    let text = truncate(
        &chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n"),
        cap,
    );
    Ok(json!({
        "documentId": id, "path": rec.path,
        "docType": format!("{:?}", rec.doc_type),
        "sourceId": rec.source_id.to_string(),
        "chunkCount": chunks.len(),
        "metadata": rec.metadata,
        "text": text,
        "truncated": text.len() >= cap,
    }))
}

fn tool_list_documents(st: &AppState, perm: &Permitted, args: &Value) -> Result<Value, String> {
    let limit = args["limit"].as_i64().unwrap_or(50).clamp(1, 200) as usize;
    let offset = args["offset"].as_i64().unwrap_or(0).max(0) as usize;
    let source = args["sourceId"].as_str().map(ragnarok_core::SourceId::from);
    let f = ragnarok_store::metadata::DocumentListFilter {
        source_id: source.as_ref(),
        doc_type: args["docType"].as_str(),
        query: args["query"].as_str(),
        sort: "indexed_at",
        desc: true,
    };
    let docs = st
        .engine
        .meta
        .list_documents_filtered(&f)
        .map_err(|e| e.to_string())?;
    let total = docs.len();
    let rows: Vec<Value> = docs
        .iter()
        .filter(|d| perm.allows(&d.id.to_string()))
        .skip(offset)
        .take(limit)
        .map(|d| {
            json!({
                "documentId": d.id.to_string(), "path": d.path,
                "docType": format!("{:?}", d.doc_type),
                "sourceId": d.source_id.to_string(),
                "chunks": d.chunk_count, "size": d.size,
                "indexedAt": d.indexed_at.to_rfc3339(),
            })
        })
        .collect();
    Ok(json!({"documents": rows, "returned": rows.len(), "total": total}))
}

fn tool_list_sources(st: &AppState, _perm: &Permitted) -> Result<Value, String> {
    let states: std::collections::HashMap<String, ragnarok_core::SourceState> = st
        .pipeline
        .states()
        .into_iter()
        .map(|s| (s.source_id.to_string(), s))
        .collect();
    let rows: Vec<Value> = st
        .engine
        .meta
        .list_sources()
        .map_err(|e| e.to_string())?
        .iter()
        .map(|s| {
            let ty = serde_json::to_value(&s.config)
                .ok()
                .and_then(|c| c["type"].as_str().map(str::to_owned))
                .unwrap_or_default();
            let live = states.get(&s.id.to_string());
            json!({
                "sourceId": s.id.to_string(), "name": s.name, "type": ty,
                "status": live.map(|x| format!("{:?}", x.status)),
                "filesIndexed": live.map(|x| x.files_indexed),
                "filesTotal": live.map(|x| x.files_total),
                "lastError": live.and_then(|x| x.last_error.clone()),
            })
        })
        .collect();
    Ok(json!({"sources": rows}))
}

fn tool_list_filters(st: &AppState, user: &UserRow) -> Result<Value, String> {
    let rows: Vec<Value> = st
        .engine
        .meta
        .filters_for_user(Some(user.id.as_str()))
        .map_err(|e| e.to_string())?
        .iter()
        .map(|f| {
            json!({
                "filterId": f.id, "name": f.name, "expr": f.expr,
            })
        })
        .collect();
    Ok(json!({"filters": rows}))
}

fn tool_system_stats(st: &AppState) -> Value {
    let s = st.stats();
    json!({
        "indexedDocuments": s.indexed_documents,
        "indexedChunks": s.indexed_chunks,
        "budgetBytes": s.budget_bytes,
        "rssBytes": s.process_rss_bytes,
        "ramUsedBytes": s.ram_used_bytes,
        "pressure": format!("{:?}", s.pressure),
        "queueDepth": s.queue_depth,
        "filesPerSec": s.files_per_sec,
        "uptimeSecs": s.uptime_secs,
    })
}

fn truncate(s: &str, cap: usize) -> String {
    if s.len() <= cap {
        return s.to_owned();
    }
    let mut end = cap;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

/// Dispatch one `tools/call`. Errors return MCP `isError` results where
/// the failure is data-level (bad id, not found) — transport/JSON-RPC
/// errors stay on the envelope.
async fn call_tool(
    st: &AppState,
    user: &UserRow,
    key_scope: Option<&str>,
    name: &str,
    args: &Value,
) -> std::result::Result<Value, String> {
    let filter_id = args["filterId"].as_str();
    let perm = call_scope(st, user, key_scope, filter_id).map_err(|e| e.to_string())?;
    match name {
        "search" => tool_search(st, &perm, args).await,
        "read_document" => tool_read_document(st, &perm, args),
        "list_documents" => tool_list_documents(st, &perm, args),
        "list_sources" => tool_list_sources(st, &perm),
        "list_filters" => tool_list_filters(st, user),
        "system_stats" => Ok(tool_system_stats(st)),
        _ => Err(format!("unknown tool '{name}'")),
    }
}

/// Handle a single JSON-RPC message. `None` = notification (no reply).
pub async fn handle_message(
    st: &AppState,
    user: &UserRow,
    key_scope: Option<&str>,
    msg: &Value,
) -> Option<Value> {
    let method = msg["method"].as_str().unwrap_or("");
    // notifications never produce a response (with or without an id,
    // they MUST be silent per spec for `notifications/*`)
    if method.starts_with("notifications/") {
        return None;
    }
    let id = msg.get("id").cloned().unwrap_or(Value::Null);
    let is_notification = msg.get("id").is_none();
    match method {
        "initialize" => Some(ok(&id, initialize_result())),
        "ping" => Some(ok(&id, json!({}))),
        "tools/list" => Some(ok(&id, json!({"tools": tool_schemas()}))),
        "tools/call" => {
            let name = msg["params"]["name"].as_str().unwrap_or("");
            let args = msg["params"]["arguments"].clone();
            if name.is_empty() {
                return Some(err(&id, INVALID_PARAMS, "params.name required"));
            }
            match call_tool(st, user, key_scope, name, &args).await {
                Ok(payload) => Some(text_result(&id, payload, false)),
                Err(e) => Some(text_result(&id, json!({"error": e}), true)),
            }
        }
        "" => Some(err(&id, INVALID_REQUEST, "method required")),
        _ if is_notification => None,
        _ => Some(err(
            &id,
            METHOD_NOT_FOUND,
            &format!("unknown method '{method}'"),
        )),
    }
}

/// `POST /api/mcp` — single or batch JSON-RPC. A body consisting solely
/// of notifications yields `202 Accepted` with no payload.
pub async fn handle(
    State(st): State<Arc<AppState>>,
    a: SessionAuth,
    body: Bytes,
) -> std::result::Result<Response, ApiError> {
    let parsed: std::result::Result<Value, _> = serde_json::from_slice(&body);
    let v = match parsed {
        Ok(v) => v,
        Err(e) => return Ok(Json(err(&Value::Null, PARSE_ERROR, &e.to_string())).into_response()),
    };
    let key_scope = a.1.as_deref();
    match v {
        Value::Array(batch) => {
            let mut out = Vec::new();
            for msg in &batch {
                if let Some(r) = handle_message(&st, &a.0, key_scope, msg).await {
                    out.push(r);
                }
            }
            if out.is_empty() {
                Ok(StatusCode::ACCEPTED.into_response())
            } else {
                Ok(Json(Value::Array(out)).into_response())
            }
        }
        msg => match handle_message(&st, &a.0, key_scope, &msg).await {
            Some(r) => Ok(Json(r).into_response()),
            None => Ok(StatusCode::ACCEPTED.into_response()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;
    use serde_json::json;

    use tower::ServiceExt;
    fn rpc(method: &str, params: Value) -> Value {
        json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    }

    async fn call_tool_http(
        app: &axum::Router,
        bearer: &str,
        name: &str,
        args: Value,
    ) -> (StatusCode, Value) {
        post_j(
            app,
            "/api/mcp",
            rpc("tools/call", json!({"name": name, "arguments": args})),
            bearer,
        )
        .await
    }

    /// Extract the tool's JSON payload from the MCP text-content envelope.
    fn payload(v: &Value) -> Value {
        serde_json::from_str(v["result"]["content"][0]["text"].as_str().unwrap_or("{}"))
            .unwrap_or_default()
    }

    #[tokio::test]
    async fn initialize_ping_and_tools_list() {
        let dir = tempfile::tempdir().unwrap();
        let (_st, app, cancel) = app(dir.path()).await;

        let (code, v) = post_j(&app, "/api/mcp", rpc("initialize", json!({})), TEST_BEARER).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["result"]["protocolVersion"], "2024-11-05");
        assert!(v["result"]["serverInfo"]["name"]
            .as_str()
            .unwrap()
            .contains("agnar"));
        assert!(v["result"]["instructions"].is_string());
        assert!(v["result"]["capabilities"]["tools"].is_object());

        let (_c, v) = post_j(&app, "/api/mcp", rpc("ping", json!({})), TEST_BEARER).await;
        assert_eq!(v["result"], json!({}));

        let (_c, v) = post_j(&app, "/api/mcp", rpc("tools/list", json!({})), TEST_BEARER).await;
        let names: Vec<&str> = v["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();
        for t in [
            "search",
            "read_document",
            "list_documents",
            "list_sources",
            "list_filters",
            "system_stats",
        ] {
            assert!(names.contains(&t), "missing tool {t}");
        }
        cancel.cancel();
    }

    #[tokio::test]
    async fn tools_call_with_fake_docs() {
        let dir = tempfile::tempdir().unwrap();
        let (st, app, cancel) = app(dir.path()).await;
        index_test_doc(
            &st,
            "/docs/alpha.md",
            "needle alpha document about strawberries",
        )
        .await;
        index_test_doc(&st, "/docs/beta.md", "unrelated beta text about forklifts").await;

        // search finds the needle doc
        let (code, v) = call_tool_http(
            &app,
            TEST_BEARER,
            "search",
            json!({"query": "strawberries"}),
        )
        .await;
        assert_eq!(code, StatusCode::OK);
        assert_ne!(v["result"]["isError"], true);
        let p = payload(&v);
        assert!(p["count"].as_u64().unwrap_or(0) >= 1, "search payload: {p}");
        assert!(p["hits"][0]["path"].as_str().unwrap().contains("alpha"));

        // read_document returns joined text + metadata
        let doc_id = p["hits"][0]["documentId"].as_str().unwrap().to_owned();
        let (_c, v) = call_tool_http(
            &app,
            TEST_BEARER,
            "read_document",
            json!({"documentId": doc_id}),
        )
        .await;
        let p = payload(&v);
        assert!(p["text"].as_str().unwrap().contains("strawberries"));
        assert_eq!(p["chunkCount"].as_u64().unwrap(), 1);

        // list_documents paginates
        let (_c, v) =
            call_tool_http(&app, TEST_BEARER, "list_documents", json!({"limit": 1})).await;
        let p = payload(&v);
        assert_eq!(p["returned"], 1);
        assert!(p["total"].as_u64().unwrap() >= 2);

        // list_sources / list_filters / system_stats respond
        let (_c, v) = call_tool_http(&app, TEST_BEARER, "list_sources", json!({})).await;
        assert!(payload(&v)["sources"].is_array());
        let (_c, v) = call_tool_http(&app, TEST_BEARER, "list_filters", json!({})).await;
        assert!(payload(&v)["filters"].is_array());
        let (_c, v) = call_tool_http(&app, TEST_BEARER, "system_stats", json!({})).await;
        assert!(payload(&v)["indexedChunks"].as_u64().unwrap() >= 2);

        // bad args → isError result (not a transport error)
        let (_c, v) = call_tool_http(&app, TEST_BEARER, "search", json!({})).await;
        assert_eq!(v["result"]["isError"], true);
        let (_c, v) = call_tool_http(
            &app,
            TEST_BEARER,
            "read_document",
            json!({"documentId": "no-uuid"}),
        )
        .await;
        assert_eq!(v["result"]["isError"], true);
        let (_c, v) = call_tool_http(&app, TEST_BEARER, "does_not_exist", json!({})).await;
        assert_eq!(v["result"]["isError"], true);
        cancel.cancel();
    }

    #[tokio::test]
    async fn api_key_scope_narrows_user_access() {
        let dir = tempfile::tempdir().unwrap();
        let (st, app, cancel) = app(dir.path()).await;
        index_test_doc(&st, "/allowed/a.md", "scope probe alpha").await;
        index_test_doc(&st, "/secret/b.md", "scope probe beta").await;

        // user max has a grant limiting to /allowed/* (non-admin: grants bind)
        let max = make_session(&st, "u-max", "user");
        st.engine
            .meta
            .create_filter(
                "f1",
                "nur-allowed",
                Some("u-max"),
                r#"{"path": "/allowed/*"}"#,
            )
            .unwrap();
        st.engine
            .meta
            .set_grants("u-max", &[("f1".into(), "allow".into())])
            .unwrap();

        // plain session: grant applies → only /allowed doc visible
        let (_c, v) = call_tool_http(&app, &max, "list_documents", json!({})).await;
        let p = payload(&v);
        let paths: Vec<&str> = p["documents"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|d| d["path"].as_str())
            .collect();
        assert!(
            paths.iter().all(|x| x.starts_with("/allowed")),
            "grant bypass: {paths:?}"
        );

        // mint a ragk_ key for admin (all docs) but scope it to /allowed/*
        let raw = "ragk_scopedtestkey123";
        st.engine
            .meta
            .create_api_key(
                "k1",
                "scoped",
                &ragnarok_core::ids::content_hash(raw.as_bytes()),
                "ragk_scope",
                "u-test",
                Some(r#"{"path": "/allowed/*"}"#),
            )
            .unwrap();
        let (_c, v) =
            call_tool_http(&app, &format!("Bearer {raw}"), "list_documents", json!({})).await;
        let p = payload(&v);
        let paths: Vec<&str> = p["documents"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|d| d["path"].as_str())
            .collect();
        assert_eq!(paths.len(), 1);
        assert!(
            paths[0].starts_with("/allowed"),
            "key scope not applied: {paths:?}"
        );

        // search through the scoped key cannot reach /secret
        let (_c, v) = call_tool_http(
            &app,
            &format!("Bearer {raw}"),
            "search",
            json!({"query": "probe"}),
        )
        .await;
        let p = payload(&v);
        let hits = p["hits"].as_array().unwrap();
        assert!(hits
            .iter()
            .all(|h| h["path"].as_str().unwrap().starts_with("/allowed")));
        cancel.cancel();
    }

    #[tokio::test]
    async fn batch_notifications_and_errors() {
        let dir = tempfile::tempdir().unwrap();
        let (_st, app, cancel) = app(dir.path()).await;

        // batch: initialize + ping + unknown method → 3 results, one -32601
        let batch = json!([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            {"jsonrpc": "2.0", "id": 2, "method": "ping"},
            {"jsonrpc": "2.0", "id": 3, "method": "bogus/method"}
        ]);
        let (code, v) = post_j(&app, "/api/mcp", batch, TEST_BEARER).await;
        assert_eq!(code, StatusCode::OK);
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[2]["error"]["code"], -32601);

        // notification-only body → 202, empty payload
        let res = app
            .clone()
            .oneshot(
                axum::http::Request::post("/api/mcp")
                    .header("content-type", "application/json")
                    .header("authorization", TEST_BEARER)
                    .body(axum::body::Body::from(
                        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::ACCEPTED);

        // garbage JSON → parse error envelope
        let res = app
            .clone()
            .oneshot(
                axum::http::Request::post("/api/mcp")
                    .header("content-type", "application/json")
                    .header("authorization", TEST_BEARER)
                    .body(axum::body::Body::from("{not json"))
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["code"], -32700);

        // unauthenticated → 401
        let res = app
            .clone()
            .oneshot(
                axum::http::Request::post("/api/mcp")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
        cancel.cancel();
    }
}
