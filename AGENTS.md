# RAGnarök v2.0

Privacy-first, embedded hybrid vector-search & RAG engine (Rust/Tokio).
Monitors filesystem sources, PDFs, Office docs, emails and strictly
read-only IMAP mailboxes; produces BGE-M3-style triple embeddings
(dense 1024d + sparse lexical + ColBERT 128d token vectors); stores
them in an embedded vector engine with SQLite metadata; serves a REST
+ WebSocket API and an embedded single-page UI.

## Workspace

- `crates/ragnarok-core` — models, config, ids, telemetry, errors
- `crates/ragnarok-crawler` — FS watcher/scanner/chunker, parsers
  (text/email/office/pdf), strict read-only IMAP crawler, ingest pipeline,
  `remote.rs` — **non-copying remote sources**: `paperless` (Paperless-NGX
  REST, indexes server-side OCR text, `Authorization: Token` from env var),
  `web_dav` (recursive PROPFIND, Basic auth via env var, include/exclude
  globs on decoded paths; hrefs stay percent-encoded for re-fetch),
  `web` (URL list, HTML→text stripper, optional same-host link follow
  depth 1 / 50-page cap), `feed` (RSS/Atom, optional full-text fetch),
  plus `remote_s3.rs`, `remote_git.rs`, `remote_dav.rs`:
  `s3` (ListObjectsV2 + in-house SigV4 signing — KAT-tested against the
  published AWS example signature; path/virtual-host styles, pagination,
  key include/exclude globs), `git` (GitHub/Gitea/Forgejo REST:
  `git/trees/{ref}?recursive=1` inventory, blob sha as etag, `contents`
  endpoint fetch — raw media type or JSON+base64 fallback),
  `cal_dav` (`kind: caldav|carddav` — REPORT `calendar-query` /
  `addressbook-query` embeds full iCalendar/vCard payloads inline;
  `dav_data_to_text` unfolds lines, decodes quoted-printable, unescapes,
  humanizes `DTSTART`-style datetimes and keeps only whitelisted VEVENT/
  VCARD fields — VALARM content stays out of the index).
  Remote docs get `metadata.remote_url`; ETag/change-token is the
  `documents.hash`, so unchanged docs skip and vanished docs are deleted
  from the index. Originals are **never** copied — `/documents/{id}/stream`
  re-fetches them live via `remote::fetch_for_record` when no blob/fs file
  exists.
- `crates/ragnarok-inference` — `Embedder` trait, deterministic
  `HashingEmbedder` (default), optional ONNX backend via `--features onnx`
- `crates/ragnarok-store` — `VectorEngine` wrapping a `qdrant-edge`
  `EdgeShard` (dense 1024d HNSW + int8 quant pinned, `colbert` 128d
  multi-vector MaxSim with m=0 on-disk, `lexical` sparse on-disk;
  WAL-durable, memmap-backed) + `MetadataStore` (SQLite) +
  `SearchEngine` (RRF hybrid → batched MaxSim rescore)
- `crates/ragnarok-resource` — RAM/CPU telemetry, budget math,
  backpressure semaphore
- `crates/ragnarok-server` — Axum REST + `/ws/v1/events` + document stream.
  **Multi-user auth** (`auth.rs`): Argon2 password hashes, opaque session
  tokens (`Authorization: Bearer` or `?token=` for WS/iframes),
  `SessionAuth`/`AdminAuth` extractors for both `AppState` and `ChatState`
  routers. First boot seeds `admin` (password from
  `RAGNAROK_ADMIN_PASSWORD`, else a generated one printed once to stderr).
  **Scoped search** (`ragnarok-core/scope.rs` + `ragnarok-store/scope.rs`):
  `ScopeFilter` = `include`/`exclude` trees of `ScopeExpr` (`any`/`all`/`not`
  + predicates: `docIds`, `sourceIds/Names/Paths`, `path`, `docTypes`,
  `emailFolder/From/To/Subject`; glob semantics, full match). Stored filters
  live in `filters` (owner or global); `grants` map users → filters with
  `allow`/`deny` mode. `effective_scope(user_id)` resolves grants into a
  `Permitted` doc-id set (allow = union, deny = union, deny applied last);
  **no grants = unrestricted**, admins bypass. Every request scope is
  intersected with the grant scope — callers narrow, never widen. The doc-id
  set becomes a qdrant payload filter (`doc_id` match_any) plus a
  post-enrichment `perm.allows` check. Document streaming answers **404**
  for out-of-scope ids (no existence leak). Admin API: `admin.rs` —
  users CRUD + `grants` matrix + filter CRUD (`/api/v1/filters*` user-facing
  list/create-own, `/api/v1/admin/*` admin-only, 401 vs 403 distinguished).
  `/api/v1/chat/*` LLM agent (`agent.rs`): OpenAI-compatible streaming
  client (reqwest), in-process tool dispatch (`search`, `read_document`,
  `list_documents`, `list_sources`, `system_stats`, `todo_read/write`,
  `task_done`), step budget, context compaction at 75%, `pending_state`
  parking. `chat.rs` carries the SSE surface: run-claim with 15 min steal
  window, attach-with-replay live map, `stop` (cancel flag, polled between
  steps/tools/mid-stream), `steer` (queue drained at step boundary),
  `continue` (resumes parked context). `GET/PUT /api/v1/chat/settings`
  configures the provider (apiKey never leaves the server — masked …last4).
  Admin probes: `GET /api/v1/chat/settings/models?baseUrl=&apiKey=` lists
  `GET {base}/models` ids, `POST /api/v1/chat/settings/test` runs a real
  minimal completion — both accept unsaved overrides and fall back to the
  stored row.
  Document browser: `GET /api/v1/documents` (params `source`, `type`, `q`,
  `sort` ∈ path|indexed_at|modified|size|doc_type, `dir`, `limit` ≤200,
  `offset`) returns `{items, total}` — scope-filtered like search, `total`
  is the post-scope count. `GET /documents/{id}/chunks` lists chunk text,
  page/bbox/line_range/token_count (out-of-scope → 404). Admin mutations:
  `DELETE /documents/{id}` drops doc + vectors (a still-existing file is
  re-picked by the next scan/watch cycle), `POST /documents/{id}/reindex`
  re-parses now — blob-backed docs (IMAP) re-parse the stored source, FS
  docs re-read disk, remote docs rebuild a `RemoteDoc` and re-fetch.
  Inference provisioning: `GET/PUT /api/v1/inference` (admin) reports
  `{backend, onnxCompiled, model}` and persists a backend switch to the
  config file (`needsRestart` — embeddings are model-bound, no hot-swap);
  `POST /api/v1/inference/model/download` spawns the provisioner.
  `ragnarok-inference/provision.rs` downloads the pinned `BAAI/bge-m3`
  `onnx/` export (external-data `model.onnx`+`model.onnx_data`+
  `Constant_7_attr__value`+`tokenizer.json`) into
  `data_dir/models/bge-m3/` — streamed to `.part`, SHA-256 for LFS files,
  git-blob SHA-1 for small ones, atomic rename, progress over
  `TelemetryEvent::ModelDownload` + `/inference` status. Config knobs:
  `inference.model_url` (mirror base), `model_sha256` (pin override),
  `auto_provision` (auto-download on `backend=onnx` startup).
  Rerank stage: `rerank.rs` cross-encodes (query, doc) pairs of the top
  `rerank_top` fused candidates (request field, default 20, 0 = off);
  `Reranker` trait + deterministic `LexicalReranker` for tests +
  `OnnxReranker` (bge-reranker-base int8, `Provisioner::for_reranker`
  downloads ~280 MB pinned `Xenova/bge-reranker-base` into
  `models/bge-reranker/`). `inference.rerank` + `reranker_dir/url/sha256`
  gate it; PUT /inference accepts `{backend?, rerank?}`;
  `POST /inference/reranker/download` mirrors the model endpoint.
  Failed/disabled rerank falls back to hybrid order.
  API keys (headless clients): `POST /api/v1/apikeys` (admin) mints a
  `ragk_…` token bound to a user + optional `scope` JSON — the raw key is
  returned once, only its SHA-256 hash (`api_keys.key_hash`) is stored.
  `GET`/`DELETE` list/revoke. Auth: `SessionAuth` is a 2-tuple —
  `.0` the resolved user, `.1` the key's scope JSON (`None` for
  sessions); `effective_search_scope(meta, user, key_scope, filter_id,
  inline)` intersects `grants AND key AND filter AND inline`, chat's
  `thread_perm` narrows by the key scope too. Keys never widen, and a
  disabled bound user makes them dead like expired sessions.
- `crates/ragnarok-ui` — `rust-embed`-bundled Angular 22 SPA (source:
  `frontend/`, built bundle committed to `assets/` because rust-embed
  embeds at compile time — rebuild the bundle before `cargo build`)
- `src/main.rs` — `ragnarok` CLI: `serve`, `index`, `sources`, `search`

## Frontend (`crates/ragnarok-ui/frontend`)

```bash
npm install            # once
npx ng build           # production bundle → ../assets/ (committed!)
npx ng test --watch=false   # Vitest unit tests (jsdom)
```

Angular 22 + TypeScript ~6.0, standalone components + signals.
Auth: `auth.service` (token in localStorage, `/me` restore),
`authInterceptor` (Bearer header + 401 → `/login`), `authGuard`/`adminGuard`.
Login page is the only public route. `chat.service.streamFetch` and the WS
(`?token=`) attach tokens manually — `fetch`/WebSocket bypass interceptors.
Filter UI at `/filters` (JSON editor with presets, admin "global" flag),
grant matrix + user admin at `/admin`, filter selector + inline scope JSON
in search. `/settings` (admin) is a tabbed hub: **LLM-Provider** (18
OpenAI-compatible presets, masked-key field, model picker via
`/settings/models`, live `/settings/test`, effort select matching
`apply_thinking_mode`), **Quellen** (embedded `rag-sources`),
**Monitoring** (embedded `rag-stats`). WS frames are serde-tagged
`{"event":"stats",…}` with fields flattened — read `ev.event`, not `type`.
`/documents` (all users, scope-filtered) is the inventory browser: path/type/
source filters + sort + pagination, expandable chunk rows, admin-only
reindex/delete buttons, filename links into `/doc/:id`. `/settings` also
has an **Embedding** tab: backend select (persisted via `PUT /inference`,
restart-required badge), model state, live download progress.
`/doc/:id` viewer: PDFs render via `pdfjs-dist` on `<canvas>` (lazy chunk,
worker served from the bundle root) so chunk bounding boxes overlay —
`?page=N&bbox=x0,y0,x1,y1` (page 1-based, bbox normalized 0..1 top-left,
the parser already flips PDF y-up → y-down). Search hits and per-chunk ⧉
links carry both params; page/chunk values elsewhere stay **0-based**.
Non-PDF types keep the sandboxed `<iframe>` stream. The component
subscribes to `paramMap` (Angular reuses it across `/doc/:id` switches —
a snapshot keeps showing the first doc).
Chat uses `fetch`/ReadableStream SSE parsing (partial chunks, LF/CRLF/CR,
keep-alives, unterminated final frame) — mirrors RiskMaster's protocol
(`data: {json}\n\n`). Events: `token`, `reasoning`, `status`, `citation`,
`tool_call`, `tool_result`, `todos`, `steered`, `compacted`,
`awaiting_continue`, `done`, `error`. Composer switches to
Lenken/Stop while a run is in flight; `awaiting_continue` arms the
"Fortsetzen" bar; the ⚙ panel edits global provider settings.

Completion is explicit: a run only ends in `done` after the model calls
`task_done`. Ending a step without it parks the context in
`chat_threads.pending_state` and emits `awaiting_continue` — the step
budget and user stops use the same park/resume mechanics.

## Commands

```bash
cargo build --workspace            # build
cargo test --workspace             # all tests
cargo clippy --workspace --all-targets
cargo fmt --all
cargo llvm-cov --workspace --summary-only   # coverage (currently ~94% lines)
```

## CLI

```bash
cargo run -- serve --config ragnarok.toml        # daemon on configured bind
cargo run -- index /path --data-dir ./data       # one-shot index
cargo run -- sources --data-dir ./data           # list sources
cargo run -- search "query" --data-dir ./data    # hybrid search
```

Config file: `--config` TOML (see `ragnarok-core` `Config`); defaults are
sensible. Data dir defaults to `./data` (db + snapshots). Config structs
and `SourceConfig` reject unknown keys — typos like `[storage] data_dir`
or `crawl = true` fail at load instead of being silently ignored.

## Hard requirements

- IMAP is **strictly read-only**: `EXAMINE` + `UID FETCH … BODY.PEEK[]`,
  never `SELECT`/`STORE`/`DELETE`/`EXPUNGE`/etc. A protocol-audit test
  (`imap.rs`, `pipeline.rs`) asserts this on a real socket.
- RAM budget: `min(total*0.70, available-2GiB)`; soft limit 75%, hard 90%
  (ingestion blocks on a semaphore).
- PDF parser is a custom `lopdf` content-stream state machine producing
  normalized top-left-origin bounding boxes for the viewer overlay.
- The edge shard's WAL is **single-writer**: only one `VectorEngine` may
  have a given `data/vectorstore` dir open at a time — drop the old
  instance before reopening.
- The LLM API key in `chat_settings` is server-side only: GET responses
  return a `…last4` mask, an empty `apiKey` on PUT keeps the stored one.
- Every API surface enforces scope: `/search`, document stream, chat agent
  tools (`search`/`read_document`/`list_documents` carry `RunParams.perm`).
  A stored `filterId` must be owned/global/granted — foreign ids are 403.
- Remote sources never persist original bytes: indexed text comes from
  API/OCR/feed summaries or a one-shot fetch; `blob`/`fs` lookups miss by
  design, so streaming must go through `fetch_for_record`. Credentials
  always come from env vars (`token_env`, `password_env`) — never stored.
  `POST /api/v1/sources/{id}/rescan` spawns a cycle for every source kind
  except `imap` (which polls on its own loop → 400).
- `cargo-llvm-cov` line coverage ~94%; target 100% — keep adding tests.
  The agent loop is covered by a scripted mock OpenAI server in
  `ragnarok-server/src/lib.rs` (tool calls, stop/steer/continue, parking).
