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
  (text/email/office/pdf), strict read-only IMAP crawler, ingest pipeline
- `crates/ragnarok-inference` — `Embedder` trait, deterministic
  `HashingEmbedder` (default), optional ONNX backend via `--features onnx`
- `crates/ragnarok-store` — `VectorEngine` wrapping a `qdrant-edge`
  `EdgeShard` (dense 1024d HNSW + int8 quant pinned, `colbert` 128d
  multi-vector MaxSim with m=0 on-disk, `lexical` sparse on-disk;
  WAL-durable, memmap-backed) + `MetadataStore` (SQLite) +
  `SearchEngine` (RRF hybrid → batched MaxSim rescore)
- `crates/ragnarok-resource` — RAM/CPU telemetry, budget math,
  backpressure semaphore
- `crates/ragnarok-server` — Axum REST + `/ws/v1/events` + document stream
  + `/api/v1/chat/*` SSE chat (thread/message persistence in SQLite,
  run-claim with 60 s steal window, attach-with-replay live map)
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
Chat uses `fetch`/ReadableStream SSE parsing (partial chunks, LF/CRLF/CR,
keep-alives, unterminated final frame) — mirrors RiskMaster's protocol
(`data: {json}\n\n`, `token`/`status`/`citation`/`done`/`error`).

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
sensible. Data dir defaults to `./data` (db + snapshots).

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
- `cargo-llvm-cov` line coverage ~94%; target 100% — keep adding tests.
