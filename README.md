# RAGnarök

Privacy-first, embedded, high-performance hybrid vector-search & RAG engine.
Everything runs locally: one Rust daemon with an embedded SQLite index,
in-process inference, a REST/WebSocket API and an embedded Angular UI —
no external services required.

## Highlights

- **Hybrid retrieval** — dense + sparse + ColBERT (MaxSim) fused by RRF,
  optional cross-encoder reranking, MMR diversification, PRF expansion,
  HyDE, plus a pure keyword mode on SQLite FTS5 (BM25).
- **On-demand sources** — remote originals are never copied; documents are
  fetched at access time via change-token sync: filesystem (with live
  watching), Paperless-NGX, WebDAV/Nextcloud/ownCloud, S3/MinIO, Git REST,
  Maildir (Dovecot/isync/KMail — `new`/`cur`/Maildir++ subfolders),
  CalDAV, CardDAV, websites/crawl, RSS/Atom, read-only IMAP.
- **Broad parsing** — PDF (incl. scanned PDFs via OCR fallback), Office
  formats, ePub, OpenDocument, CSV/TSV/JSONL, archives (ZIP/TAR), EML +
  attachments as searchable child documents, images via Tesseract OCR.
- **RAG without lock-in** — hybrid search needs no LLM at all; chat/agent
  uses any OpenAI-compatible endpoint when configured.
- **Multi-user** — sessions, `ragk_` API keys, per-user scopes/filters,
  audit log, admin console.
- **Ops-ready** — Prometheus `/metrics`, `/healthz` + `/readyz` probes,
  admin backup snapshots, runtime-tunable settings, structured audit.
- **MCP server** — `POST /api/mcp` + `ragnarok mcp` stdio proxy expose
  retrieval tools to external agents.

## Quick start

### Docker (recommended)

```bash
docker build -t ragnarok .
docker run -d -p 7878:7878 -v ragnarok-data:/data \
    -v /path/to/documents:/docs:ro --name ragnarok ragnarok
# or: docker compose up -d
```

Open <http://localhost:7878>. The bootstrap admin password is printed to
the container log on first start. Put TLS in front via your reverse proxy.

### From source

```bash
# frontend (Angular → crates/ragnarok-ui/assets)
cd crates/ragnarok-ui/frontend && npm install && npm run build && cd ../../..

# backend (embeds the UI assets)
cargo build --release
./target/release/ragnarok serve --bind 0.0.0.0:7878 --data-dir ./data
```

Optional system tools: `tesseract` (OCR), `pdftoppm` (scanned PDFs).

## CLI

```
ragnarok serve                      # daemon: REST+WS API, watchers, pollers
ragnarok index <path> [--watch]     # one-shot indexing of a directory
ragnarok search "<query>"           # hybrid search against the local index
ragnarok sources                    # list configured sources
ragnarok mcp                        # stdio MCP proxy (RAGNAROK_API/_TOKEN)
ragnarok passwd <user>              # reset a user password (offline, kills sessions)
```

Global flags: `--config <path>`, `--data-dir <path>`, `--bind <addr>`.

Password reset: `ragnarok passwd admin --password <new>` writes the hash
straight into `ragnarok.db` — works while the daemon is stopped and
invalidates every session of that user. Without `--password` it uses
`RAGNAROK_NEW_PASSWORD` or generates a random one and prints it once.

## Configuration

`config.toml` (default `~/.config/ragnarok/config.toml`):

```toml
data_dir = "~/.local/share/ragnarok"   # ragnarok.db, vectorstore, models

[server]
bind = "127.0.0.1:7878"                # put a TLS proxy in front for LAN use
telemetry_interval_ms = 250

[resources]                            # adaptive backpressure
reserve_fraction = 0.15
cap_fraction = 0.8
soft_fraction = 0.7
hard_fraction = 0.9
base_permits = 8

[crawler]
debounce_ms = 400
chunk_min_tokens = 128
chunk_max_tokens = 512
chunk_overlap = 0.15
max_file_bytes = 104857600             # 100 MiB

[crawler.ocr]
enabled = true
binary = "tesseract"
lang = "eng"                           # install tesseract-ocr-deu for German
dpi = 300
pdf_text_threshold = 32                # below → treat PDF page as scan
max_pages = 50

[inference]
backend = "hashing"                    # or "onnx" (auto-downloads BGE-M3)
provider = "auto"                      # execution provider: auto | cpu |
                                       #   vulkan | webgpu | coreml
auto_provision = true
rerank = true                          # cross-encoder reranker (BGE)
workers = 2
batch_size = 8
```

The `onnx` backend is compiled in by default (`cargo build` — use
`--no-default-features` for a model-free binary). `provider = "auto"`
selects the best available execution provider at session start:
**CoreML** on Apple Silicon, the **WebGPU EP on Dawn (Vulkan)** on
Linux/Windows when a GPU driver is present, and plain CPU otherwise —
GPU providers that fail to register just fall through to CPU. Set
`provider = "vulkan"` to pin the Vulkan backend explicitly, or `"cpu"`
to force CPU. Explicit names are hard-checked: if the provider isn't
usable, model load fails loudly instead of silently degrading.

Sources are added via the UI/API (or `[[sources]]` blocks); chunking, OCR,
file-size caps and resource limits are also tunable at runtime through
`PUT /api/v1/settings/runtime` — changes apply live and persist.

## API overview (v1)

| Area | Endpoints |
|---|---|
| Auth | `POST /auth/login`, `POST /auth/logout`, `GET /auth/me` |
| Search | `POST /search` (modes `hybrid`/`lexical`, filters, MMR/PRF/HyDE, facets) |
| Documents | `GET /documents` (browse/filter/paginate), `GET /documents/{id}`, `GET /documents/{id}/stream`, `GET /documents/{id}/context`, `GET /documents/{id}/chunks`, `DELETE /documents/{id}`, `POST /documents/{id}/reindex` |
| Sources | `GET/POST /sources`, `DELETE /sources/{id}`, `POST /sources/{id}/rescan`, `POST /sources/{id}/poll` |
| Chat | `GET/POST /chat/threads`, `POST /chat/threads/{id}/messages` (SSE), `POST …/stop`, `POST …/continue` |
| Admin | `GET/POST /admin/users`, `PATCH/DELETE /admin/users/{id}`, grants, filters, `POST/GET /admin/reindex`, `GET /audit`, `GET/PUT /settings/runtime`, `POST/GET/DELETE /admin/backup` |
| Keys | `POST/GET/DELETE /apikeys` |
| Ops | `GET /system/stats`, `GET /metrics`, `GET /healthz`, `GET /readyz`, `WS /ws/v1/events` |
| MCP | `POST /api/mcp` (JSON-RPC, tools: search/read/list/context/chunks/filters) |

Auth: session cookie or `Authorization: Bearer ragk_…` API key; scopes are
enforced per request.

## Architecture

```
Angular SPA (embedded)        REST/WS ──► axum server
                                      ├── SearchEngine ──► SQLite (metadata,
                                      │     ├ dense        FTS5, blobs, audit)
                                      │     ├ sparse    ──► VectorEngine (disk)
                                      │     └ colbert
                                      ├── IngestPipeline ──► parsers, chunker,
                                      │     OCR, remote fetchers, watcher
                                      └── backpressure/resource monitor
```

Workspace crates: `ragnarok-core` (config/models/scopes), `ragnarok-store`
(metadata DB + vector engine + hybrid search), `ragnarok-crawler`
(scan/parse/ingest/remotes), `ragnarok-inference` (hashing/ONNX embedders,
provisioning), `ragnarok-resource` (backpressure), `ragnarok-server`
(axum API incl. chat agent + MCP), `ragnarok-ui` (embedded assets).

## Development

```bash
cargo test --workspace                     # backend suite (256 tests)
cd crates/ragnarok-ui/frontend && npx ng test --watch=false   # UI specs
cargo fmt --check && cargo clippy --workspace --all-targets
```

Rebuilt frontend assets are embedded at compile time — rebuild the UI then
`cargo build` after frontend changes. See `AGENTS.md` for contributor notes.

## Backup

`POST /api/v1/admin/backup` snapshots the entire index database
(documents, chunks, blobs, users, sessions) into `data_dir/backups/`
via `VACUUM INTO` — consistent while running. Vectors are a derived index;
rebuild them with `POST /api/v1/admin/reindex` after a restore.
