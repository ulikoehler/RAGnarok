//! Ingest pipeline: scan/hash → parse → chunk → embed → index, fully gated
//! by the backpressure controller, plus the per-source FS-watch and IMAP
//! polling tasks.

use crate::chunker::{chunk_document, ChunkerConfig};
use crate::imap;
use crate::parsers::{default_registry, parse_with, Parser};
use crate::scanner::{hash_file, scan_directory};
use crate::watcher::{FsEvent, FsEventKind, FsWatcher};
use chrono::Utc;
use dashmap::DashMap;
use ragnarok_core::{
    DocumentRecord, Error, IngestStatus, ParsedDocument, Result, Source, SourceConfig, SourceId,
    SourceState, SyncStatus, TelemetryEvent,
};
use ragnarok_inference::InferencePool;
use ragnarok_resource::BackpressureController;
use ragnarok_store::SearchEngine;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;

/// Outcome of one ingest attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestOutcome {
    Indexed,
    /// Same content hash already indexed — no work done.
    Skipped,
    Deleted,
    Unsupported,
}

/// Drives all ingestion for all sources.
pub struct IngestPipeline {
    engine: Arc<SearchEngine>,
    pool: Arc<InferencePool>,
    backpressure: Arc<BackpressureController>,
    cfg: parking_lot::RwLock<ChunkerConfig>,
    max_file_bytes: AtomicU64,
    /// Swapped wholesale on OCR runtime-toggle — `parse_with` reads under
    /// the shared lock.
    parsers: parking_lot::RwLock<Vec<Box<dyn Parser>>>,
    /// Boot OCR config — `ocr_enabled`/`lang` are runtime-overridable,
    /// the rest (binaries, dpi, timeouts) is restart-only.
    ocr_base: parking_lot::RwLock<ragnarok_core::OcrConfig>,
    ocr_enabled: std::sync::atomic::AtomicBool,
    /// Runtime poll-interval overrides (secs) — beats the stored source
    /// config until restart. Persisted via `POST /sources/{id}/poll`.
    poll_overrides: DashMap<SourceId, u64>,
    event_tx: broadcast::Sender<TelemetryEvent>,
    /// recent successful ingests for throughput telemetry
    recent: parking_lot::Mutex<VecDeque<Instant>>,
    queue_depth: AtomicUsize,
    states: DashMap<SourceId, SourceState>,
    /// Per-source task cancellation — every spawned crawl task gets a
    /// child token registered here so rescan/delete can stop the previous
    /// task instead of leaking a zombie poller/watcher.
    tasks: DashMap<SourceId, CancellationToken>,
}

impl IngestPipeline {
    pub fn new(
        engine: Arc<SearchEngine>,
        pool: Arc<InferencePool>,
        backpressure: Arc<BackpressureController>,
        crawler: &ragnarok_core::CrawlerConfig,
        event_tx: broadcast::Sender<TelemetryEvent>,
    ) -> Arc<Self> {
        Arc::new(IngestPipeline {
            engine,
            pool,
            backpressure,
            cfg: parking_lot::RwLock::new(ChunkerConfig {
                min_tokens: crawler.chunk_min_tokens,
                max_tokens: crawler.chunk_max_tokens,
                overlap: crawler.chunk_overlap,
            }),
            max_file_bytes: AtomicU64::new(crawler.max_file_bytes),
            parsers: parking_lot::RwLock::new(default_registry(
                crawler.ocr.enabled.then(|| crawler.ocr.clone()),
            )),
            ocr_base: parking_lot::RwLock::new(crawler.ocr.clone()),
            ocr_enabled: std::sync::atomic::AtomicBool::new(crawler.ocr.enabled),
            poll_overrides: DashMap::new(),
            event_tx,
            recent: parking_lot::Mutex::new(VecDeque::new()),
            queue_depth: AtomicUsize::new(0),
            states: DashMap::new(),
            tasks: DashMap::new(),
        })
    }

    pub fn queue_depth(&self) -> usize {
        self.queue_depth.load(Ordering::Relaxed)
    }

    /// Reason the embedder's GPU provider was abandoned, if it was —
    /// surfaced by `GET /inference` so the UI can explain the CPU
    /// fallback.
    pub fn gpu_dead(&self) -> Option<String> {
        self.pool.gpu_dead()
    }

    /// Execution provider actually in use — e.g.
    /// `WebGpuExecutionProvider`, `cpu` — after fallbacks resolved.
    pub fn provider_label(&self) -> Option<String> {
        self.pool.provider_label()
    }

    /// Parse raw bytes through the live parser registry — used by the
    /// document stream endpoint to resolve `parent#att:` children by
    /// re-fetching and re-parsing the container on demand.
    pub fn parse_bytes(&self, path: &Path, bytes: &[u8]) -> Result<ParsedDocument> {
        parse_with(&self.parsers.read(), path, bytes)
    }

    /// Files indexed per second over a rolling 10 s window.
    pub fn files_per_sec(&self) -> f32 {
        let mut recent = self.recent.lock();
        let cutoff = Instant::now() - Duration::from_secs(10);
        while recent.front().is_some_and(|t| *t < cutoff) {
            recent.pop_front();
        }
        recent.len() as f32 / 10.0
    }

    pub fn states(&self) -> Vec<SourceState> {
        self.states.iter().map(|e| e.value().clone()).collect()
    }

    /// Live-tunables for `GET /settings/runtime` — read back from the
    /// actual components, not from a cached copy.
    pub fn runtime_config(&self) -> ragnarok_core::RuntimeConfig {
        let cfg = self.cfg.read();
        ragnarok_core::RuntimeConfig {
            chunk_min_tokens: cfg.min_tokens,
            chunk_max_tokens: cfg.max_tokens,
            chunk_overlap: cfg.overlap,
            max_file_bytes: self.max_file_bytes.load(Ordering::Relaxed),
            ocr_enabled: self.ocr_enabled.load(Ordering::Relaxed),
            ocr_lang: self.ocr_base.read().lang.clone(),
            soft_fraction: self.backpressure.config().soft_fraction,
            hard_fraction: self.backpressure.config().hard_fraction,
            base_permits: self.backpressure.config().base_permits,
            // owned by the server's rate limiter, not the pipeline — the
            // handler merges the live value into this snapshot
            rate_limit_per_min: 0,
        }
    }

    /// Apply a validated runtime config: chunker bounds, file cap,
    /// OCR toggle/language (rebuilds the parser registry) and the
    /// backpressure thresholds. New ingests see the values immediately;
    /// already-indexed chunks keep their original shape.
    pub fn apply_runtime(&self, rt: &ragnarok_core::RuntimeConfig) {
        *self.cfg.write() = ChunkerConfig {
            min_tokens: rt.chunk_min_tokens,
            max_tokens: rt.chunk_max_tokens,
            overlap: rt.chunk_overlap,
        };
        self.max_file_bytes
            .store(rt.max_file_bytes, Ordering::Relaxed);
        let mut ocr = self.ocr_base.read().clone();
        ocr.lang = rt.ocr_lang.clone();
        *self.ocr_base.write() = ocr.clone();
        self.ocr_enabled.store(rt.ocr_enabled, Ordering::Relaxed);
        *self.parsers.write() = default_registry(rt.ocr_enabled.then_some(ocr));
        let mut rc = self.backpressure.config();
        rc.soft_fraction = rt.soft_fraction;
        rc.hard_fraction = rt.hard_fraction;
        rc.base_permits = rt.base_permits;
        self.backpressure.update_config(rc);
    }

    /// Runtime override for a source's poll interval (seconds) — the
    /// remote/imap loops re-read it every cycle, so the change takes
    /// effect on the next sleep. Persisted alongside in `config_json`.
    pub fn set_poll_interval(&self, source_id: &SourceId, secs: u64) {
        self.poll_overrides.insert(source_id.clone(), secs.max(30));
    }

    /// Effective poll interval: runtime override beats stored config.
    fn poll_interval(&self, source_id: &SourceId, stored: u64) -> u64 {
        self.poll_overrides
            .get(source_id)
            .map(|v| *v)
            .unwrap_or(stored)
    }

    fn set_state(&self, id: &SourceId, f: impl FnOnce(&mut SourceState)) {
        let mut e = self
            .states
            .entry(id.clone())
            .or_insert_with(|| SourceState::new(id.clone()));
        f(e.value_mut());
        let st = e.value().clone();
        drop(e);
        let _ = self.event_tx.send(TelemetryEvent::SourceState(st));
    }

    fn emit_ingest(&self, source_id: &SourceId, path: &str, status: IngestStatus) {
        let _ = self.event_tx.send(TelemetryEvent::Ingest {
            source_id: source_id.to_string(),
            path: path.to_string(),
            status,
        });
    }

    fn mark_indexed(&self) {
        self.recent.lock().push_back(Instant::now());
    }

    /// Full ingest for one file on disk.
    pub async fn ingest_path(&self, source_id: &SourceId, path: &Path) -> Result<IngestOutcome> {
        let path_str = path.display().to_string();
        self.emit_ingest(source_id, &path_str, IngestStatus::Started);
        match self.ingest_path_inner(source_id, path, &path_str).await {
            Err(e) => {
                // a Started event without a terminal one leaves the ingest
                // feed hanging — every error path must emit Failed
                self.emit_ingest(source_id, &path_str, IngestStatus::Failed);
                Err(e)
            }
            ok => ok,
        }
    }

    async fn ingest_path_inner(
        &self,
        source_id: &SourceId,
        path: &Path,
        path_str: &str,
    ) -> Result<IngestOutcome> {
        let hash = {
            let p = path.to_path_buf();
            tokio::task::spawn_blocking(move || hash_file(&p))
                .await
                .map_err(|e| Error::Internal(format!("join: {e}")))??
        };

        // dedupe: same path+hash -> skip; same hash elsewhere -> reindex cheaply
        if let Some(existing) = self.engine.meta.get_document_by_path(source_id, path_str)? {
            if existing.hash == hash {
                self.emit_ingest(source_id, path_str, IngestStatus::Skipped);
                return Ok(IngestOutcome::Skipped);
            }
        }

        let bytes = tokio::fs::read(path).await?;
        let doc_id = ragnarok_core::document_id(source_id, path_str);
        self.index_bytes(source_id, doc_id, path_str, path, &bytes, hash)
            .await
    }

    /// Ingest a virtual document (e.g. an IMAP message stored under
    /// `imap://<mailbox>/<uid>`).
    pub async fn ingest_virtual(
        &self,
        source_id: &SourceId,
        virtual_path: &str,
        bytes: &[u8],
    ) -> Result<IngestOutcome> {
        self.emit_ingest(source_id, virtual_path, IngestStatus::Started);
        match self
            .ingest_virtual_inner(source_id, virtual_path, bytes)
            .await
        {
            Err(e) => {
                self.emit_ingest(source_id, virtual_path, IngestStatus::Failed);
                Err(e)
            }
            ok => ok,
        }
    }

    async fn ingest_virtual_inner(
        &self,
        source_id: &SourceId,
        virtual_path: &str,
        bytes: &[u8],
    ) -> Result<IngestOutcome> {
        let hash = crate::scanner::hash_bytes(bytes);
        if let Some(existing) = self
            .engine
            .meta
            .get_document_by_path(source_id, virtual_path)?
        {
            if existing.hash == hash {
                self.emit_ingest(source_id, virtual_path, IngestStatus::Skipped);
                return Ok(IngestOutcome::Skipped);
            }
        }
        let doc_id = ragnarok_core::document_id(source_id, virtual_path);
        let out = self
            .index_bytes(
                source_id,
                doc_id,
                virtual_path,
                Path::new(virtual_path),
                bytes,
                hash,
            )
            .await?;
        // persist raw bytes so the document viewer can stream virtual docs
        if out == IngestOutcome::Indexed {
            self.engine.meta.put_blob(doc_id, bytes)?;
        }
        Ok(out)
    }

    /// Shared tail of both ingest paths: parse → chunk → embed → store.
    async fn index_bytes(
        &self,
        source_id: &SourceId,
        doc_id: ragnarok_core::DocumentId,
        path_str: &str,
        path: &Path,
        bytes: &[u8],
        hash: String,
    ) -> Result<IngestOutcome> {
        // heavy work admitted through the backpressure gate
        let _permit = self
            .backpressure
            .acquire()
            .await
            .ok_or_else(|| Error::ResourceExhausted("shutting down".into()))?;

        let parsed = match parse_with(&self.parsers.read(), path, bytes) {
            Ok(p) => p,
            Err(Error::UnsupportedType(_)) => {
                self.emit_ingest(source_id, path_str, IngestStatus::Skipped);
                return Ok(IngestOutcome::Unsupported);
            }
            Err(e) => {
                self.emit_ingest(source_id, path_str, IngestStatus::Failed);
                return Err(e);
            }
        };
        self.emit_ingest(source_id, path_str, IngestStatus::Parsed);
        self.index_parsed(
            source_id,
            doc_id,
            path_str,
            parsed,
            bytes.len() as u64,
            hash,
            std::fs::metadata(path_str)
                .ok()
                .and_then(|m| m.modified().ok())
                .map(chrono::DateTime::<Utc>::from),
        )
        .await
    }

    /// Chunk + embed + store one parsed document.
    #[allow(clippy::too_many_arguments)]
    async fn index_parsed(
        &self,
        source_id: &SourceId,
        doc_id: ragnarok_core::DocumentId,
        path_str: &str,
        mut parsed: ParsedDocument,
        size: u64,
        hash: String,
        modified: Option<chrono::DateTime<Utc>>,
    ) -> Result<IngestOutcome> {
        // Attachments are lifted out before chunking — they ingest as
        // separate child documents, never mixed into the parent text.
        let attachments = if path_str.contains("#att:") {
            Vec::new() // depth cap 1: children never nest further
        } else {
            std::mem::take(&mut parsed.attachments)
        };
        // normalize parsed text before chunking: markup and opaque
        // base64/hex runs explode the tokenizer without adding anything
        // searchable — a doc that cleans to nothing is skipped below
        for b in &mut parsed.blocks {
            b.text = crate::preprocess::clean(&b.text);
        }
        let chunks = chunk_document(doc_id, &parsed, *self.cfg.read())?;
        if chunks.is_empty() {
            // nothing indexable — and if a previous version WAS indexed,
            // its chunks are stale now: drop the record so an emptied
            // file can't keep answering searches with old content
            if let Some(existing) = self.engine.meta.get_document_by_path(source_id, path_str)? {
                self.engine.delete_document(existing.id)?;
            }
            self.emit_ingest(source_id, path_str, IngestStatus::Skipped);
            return Ok(IngestOutcome::Unsupported);
        }
        self.queue_depth.fetch_add(1, Ordering::Relaxed);
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        // embed + index under the depth counter — the decrement MUST run on
        // the error path too or the gauge leaks permanently
        let result = async {
            let triples = self.pool.embed_batch(&texts).await?;
            let record = DocumentRecord {
                id: doc_id,
                source_id: source_id.clone(),
                path: path_str.to_string(),
                hash,
                doc_type: parsed.doc_type,
                size,
                modified,
                indexed_at: Utc::now(),
                chunk_count: chunks.len() as u32,
                metadata: parsed.metadata.clone(),
            };
            self.engine.index_document(&record, &chunks, &triples).await
        }
        .await;
        self.queue_depth.fetch_sub(1, Ordering::Relaxed);
        // a GPU that died mid-run is reported by the pool — persist the
        // reason so later boots start on CPU instead of re-crashing
        if let Some(reason) = self.pool.gpu_dead() {
            let _ = self.engine.meta.set_setting("gpu_dead", &reason);
        }
        result?;
        self.mark_indexed();
        self.emit_ingest(source_id, path_str, IngestStatus::Indexed);
        // child documents: each attachment ingests under a deterministic
        // `parent#att:{name}` path — same dedup semantics as top-level
        // docs (unchanged content hash → Skipped).
        for (i, att) in attachments.into_iter().enumerate() {
            self.ingest_attachment(source_id, path_str, parsed.doc_type, i, att)
                .await;
        }
        Ok(IngestOutcome::Indexed)
    }

    /// Ingest one attachment as a child document. Per-member failures
    /// never propagate — a broken attachment must not sink its parent.
    async fn ingest_attachment(
        &self,
        source_id: &SourceId,
        parent_path: &str,
        container: ragnarok_core::DocType,
        idx: usize,
        att: ragnarok_core::model::AttachmentRef,
    ) {
        let child_path = format!("{parent_path}#att:{}", att.filename);
        if att.bytes.len() as u64 > self.max_file_bytes.load(Ordering::Relaxed) {
            self.emit_ingest(source_id, &child_path, IngestStatus::Skipped);
            return;
        }
        let hash = crate::scanner::hash_bytes(&att.bytes);
        if let Some(existing) = self
            .engine
            .meta
            .get_document_by_path(source_id, &child_path)
            .ok()
            .flatten()
        {
            if existing.hash == hash {
                return; // unchanged → silent dedup (no event storm)
            }
        }
        // the inner extension decides DocType — unsupported members skip
        // individually without failing the whole container
        let mut child = match parse_with(&self.parsers.read(), Path::new(&att.filename), &att.bytes)
        {
            Ok(p) => p,
            Err(_) => {
                self.emit_ingest(source_id, &child_path, IngestStatus::Skipped);
                return;
            }
        };
        child
            .metadata
            .insert("parent".into(), parent_path.to_owned());
        child
            .metadata
            .insert("container".into(), format!("{container:?}"));
        child
            .metadata
            .insert("attachment_idx".into(), idx.to_string());
        let child_id = ragnarok_core::document_id(source_id, &child_path);
        // boxed: recursion through the async fn is depth-capped by the
        // `#att:` guard but the compiler still needs the indirection
        let _ = Box::pin(self.index_parsed(
            source_id,
            child_id,
            &child_path,
            child,
            att.bytes.len() as u64,
            hash,
            None,
        ))
        .await;
    }

    /// Drop a path from the index (file deleted/renamed away).
    pub async fn remove_path(&self, source_id: &SourceId, path: &Path) -> Result<bool> {
        let path_str = path.display().to_string();
        if let Some(doc) = self
            .engine
            .meta
            .get_document_by_path(source_id, &path_str)?
        {
            self.engine.delete_document(doc.id)?;
            self.emit_ingest(source_id, &path_str, IngestStatus::Deleted);
            return Ok(true);
        }
        Ok(false)
    }

    /// Long-running task for one filesystem source: initial scan, then live
    /// watching. Returns when `cancel` fires or a fatal error occurs.
    pub async fn run_fs_source(self: &Arc<Self>, source: Source, cancel: CancellationToken) {
        let (root, recursive, include, exclude, watch) = match &source.config {
            SourceConfig::FileSystem {
                path,
                recursive,
                include,
                exclude,
                watch,
            } => (
                path.clone(),
                *recursive,
                include.clone(),
                exclude.clone(),
                *watch,
            ),
            _ => {
                self.set_state(&source.id, |s| {
                    s.status = SyncStatus::Error;
                    s.last_error = Some("not a filesystem source".into());
                });
                return;
            }
        };

        // ---- initial scan ----
        self.set_state(&source.id, |s| {
            s.status = SyncStatus::Scanning;
            // reset the counters — a rescan reuses the persisted state
            // and would otherwise accumulate across cycles
            s.files_indexed = 0;
        });
        let files = scan_directory(
            &root,
            recursive,
            &include,
            &exclude,
            self.max_file_bytes.load(Ordering::Relaxed),
        );
        let total = files.len() as u64;
        self.set_state(&source.id, |s| s.files_total = total);
        for path in &files {
            if cancel.is_cancelled() {
                return;
            }
            match self.ingest_path(&source.id, path).await {
                // Indexed = freshly ingested, Skipped = unchanged but
                // already in the index — both count toward files_indexed
                // (same semantic as the remote cycle); Unsupported does
                // not, it never lands in the index
                Ok(IngestOutcome::Indexed | IngestOutcome::Skipped) => {
                    self.set_state(&source.id, |s| {
                        s.files_indexed += 1;
                    })
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "ingest failed");
                    self.set_state(&source.id, |s| {
                        s.last_error = Some(e.to_string());
                    });
                }
            }
        }
        self.set_state(&source.id, |s| {
            s.last_sync = Some(Utc::now());
        });

        // ---- live watch ----
        if !watch {
            self.set_state(&source.id, |s| s.status = SyncStatus::Idle);
            return;
        }
        let (tx, mut rx) = mpsc::channel::<FsEvent>(256);
        let _watcher = match FsWatcher::start(
            &root,
            recursive,
            Duration::from_millis(500),
            tx,
            cancel.clone(),
        ) {
            Ok(w) => w,
            Err(e) => {
                self.set_state(&source.id, |s| {
                    s.status = SyncStatus::Error;
                    s.last_error = Some(e.to_string());
                });
                return;
            }
        };
        self.set_state(&source.id, |s| s.status = SyncStatus::Watching);
        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                ev = rx.recv() => {
                    let Some(ev) = ev else { break };
                    match ev.kind {
                        FsEventKind::Upsert => {
                            // watcher events bypass scan_directory — the same
                            // hidden/include/exclude/size rules apply here or
                            // `node_modules/x.js` would land in the index
                            if !crate::scanner::is_indexable(
                                &root, &ev.path, &include, &exclude, self.max_file_bytes.load(Ordering::Relaxed),
                            ) {
                                // a renamed-away path arrives as Upsert while
                                // the file is already gone — drop its stale
                                // indexed doc (excluded/never-indexed paths
                                // no-op via the lookup)
                                if crate::scanner::was_indexable(
                                    &root, &ev.path, &include, &exclude,
                                ) && matches!(
                                    self.remove_path(&source.id, &ev.path).await,
                                    Ok(true)
                                )
                                {
                                    self.set_state(&source.id, |s| {
                                        s.files_total = s.files_total.saturating_sub(1);
                                        s.files_indexed = s.files_indexed.saturating_sub(1);
                                    });
                                }
                                continue;
                            }
                            // a path not yet in the index grew the inventory
                            let path_str = ev.path.display().to_string();
                            let known = self
                                .engine
                                .meta
                                .get_document_by_path(&source.id, &path_str)
                                .ok()
                                .flatten()
                                .is_some();
                            match self.ingest_path(&source.id, &ev.path).await {
                                Ok(IngestOutcome::Indexed) => self.set_state(&source.id, |s| {
                                    if !known {
                                        s.files_total += 1;
                                        s.files_indexed += 1;
                                    }
                                }),
                                // Unsupported either never enters the index
                                // (new path) or just lost its stale record
                                // (file emptied) — adjust both counters
                                Ok(IngestOutcome::Unsupported) => self.set_state(&source.id, |s| {
                                    if known {
                                        s.files_indexed = s.files_indexed.saturating_sub(1);
                                    } else {
                                        s.files_total += 1;
                                    }
                                }),
                                Ok(_) => {}
                                Err(e) => self.set_state(&source.id, |s| {
                                    s.last_error = Some(e.to_string());
                                }),
                            }
                        }
                        FsEventKind::Removed => {
                            if !crate::scanner::was_indexable(&root, &ev.path, &include, &exclude) {
                                continue;
                            }
                            match self.remove_path(&source.id, &ev.path).await {
                                Ok(true) => self.set_state(&source.id, |s| {
                                    s.files_total = s.files_total.saturating_sub(1);
                                    s.files_indexed = s.files_indexed.saturating_sub(1);
                                }),
                                Ok(false) => {}
                                Err(e) => self.set_state(&source.id, |s| {
                                    s.last_error = Some(e.to_string());
                                }),
                            }
                        }
                    }
                }
            }
        }
    }

    /// Ingest one remote document (Paperless/WebDAV/Web/Feed). Bytes are
    /// fetched on demand — the OCR/API text path never downloads the
    /// original and nothing is persisted to the blob table.
    pub async fn ingest_remote(
        &self,
        source_id: &SourceId,
        cfg: &SourceConfig,
        doc: &crate::remote::RemoteDoc,
    ) -> Result<IngestOutcome> {
        self.emit_ingest(source_id, &doc.vpath, IngestStatus::Started);
        match self.ingest_remote_inner(source_id, cfg, doc).await {
            Err(e) => {
                self.emit_ingest(source_id, &doc.vpath, IngestStatus::Failed);
                Err(e)
            }
            ok => ok,
        }
    }

    async fn ingest_remote_inner(
        &self,
        source_id: &SourceId,
        cfg: &SourceConfig,
        doc: &crate::remote::RemoteDoc,
    ) -> Result<IngestOutcome> {
        // etag is the remote change token — unchanged docs skip everything
        let existing = self
            .engine
            .meta
            .get_document_by_path(source_id, &doc.vpath)?;
        if let (Some(e), Some(h)) = (&existing, &doc.etag) {
            if &e.hash == h {
                self.emit_ingest(source_id, &doc.vpath, IngestStatus::Skipped);
                return Ok(IngestOutcome::Skipped);
            }
        }
        let doc_id = ragnarok_core::document_id(source_id, &doc.vpath);
        // Cheap API text first (Paperless OCR, feed summary, web strip);
        // otherwise download the original and parse it.
        let (parsed, size, hash) = match crate::remote::index_text(cfg, doc).await? {
            Some(text) if !text.trim().is_empty() => {
                let hash = doc
                    .etag
                    .clone()
                    .unwrap_or_else(|| crate::scanner::hash_bytes(text.as_bytes()));
                // same dedupe as the download path: without an etag the
                // content hash decides whether re-embedding is needed
                if let Some(e) = &existing {
                    if e.hash == hash {
                        self.emit_ingest(source_id, &doc.vpath, IngestStatus::Skipped);
                        return Ok(IngestOutcome::Skipped);
                    }
                }
                let mut p = ParsedDocument::new(&doc.vpath, mime_doctype(&doc.mime));
                p.blocks.push(ragnarok_core::TextBlock::plain(text));
                (p, doc.size.unwrap_or(0), hash)
            }
            _ => {
                let (bytes, mime) = crate::remote::fetch_doc(cfg, doc).await?;
                let hash = doc
                    .etag
                    .clone()
                    .unwrap_or_else(|| crate::scanner::hash_bytes(&bytes));
                if let Some(e) = &existing {
                    if e.hash == hash {
                        self.emit_ingest(source_id, &doc.vpath, IngestStatus::Skipped);
                        return Ok(IngestOutcome::Skipped);
                    }
                }
                let parsed = match parse_with(&self.parsers.read(), Path::new(&doc.vpath), &bytes) {
                    Ok(mut p) => {
                        if p.doc_type == ragnarok_core::DocType::Unknown {
                            p.doc_type = mime_doctype(&mime);
                        }
                        p
                    }
                    Err(Error::UnsupportedType(_)) => {
                        self.emit_ingest(source_id, &doc.vpath, IngestStatus::Skipped);
                        return Ok(IngestOutcome::Unsupported);
                    }
                    Err(e) => {
                        self.emit_ingest(source_id, &doc.vpath, IngestStatus::Failed);
                        return Err(e);
                    }
                };
                (parsed, bytes.len() as u64, hash)
            }
        };
        let mut parsed = parsed;
        parsed.metadata.extend(doc.metadata.clone());
        if !doc.title.is_empty() {
            parsed
                .metadata
                .entry("title".into())
                .or_insert(doc.title.clone());
        }
        self.emit_ingest(source_id, &doc.vpath, IngestStatus::Parsed);
        self.index_parsed(
            source_id,
            doc_id,
            &doc.vpath,
            parsed,
            size,
            hash,
            doc.modified,
        )
        .await
    }

    /// Long-running poll loop for any remote (non-copying) source.
    pub async fn run_remote_source(self: &Arc<Self>, source: Source, cancel: CancellationToken) {
        let stored_interval = crate::remote::poll_interval(&source.config);
        loop {
            if cancel.is_cancelled() {
                return;
            }
            let interval = self.poll_interval(&source.id, stored_interval);
            self.set_state(&source.id, |s| s.status = SyncStatus::Syncing);
            match self.remote_cycle(&source, &cancel).await {
                Ok(_n) => self.set_state(&source.id, |s| {
                    s.status = SyncStatus::Idle;
                    s.last_sync = Some(Utc::now());
                    // files_indexed is maintained inside remote_cycle —
                    // adding the return value here would double-count
                }),
                Err(e) => {
                    tracing::warn!(source = %source.name, error = %e, "remote cycle failed");
                    self.set_state(&source.id, |s| {
                        s.status = SyncStatus::Error;
                        s.last_error = Some(e.to_string());
                    });
                }
            }
            tokio::select! {
                _ = cancel.cancelled() => return,
                _ = tokio::time::sleep(Duration::from_secs(interval)) => {}
            }
        }
    }

    /// One inventory-diff cycle: list remote docs, ingest new/changed,
    /// drop deleted. `files_total`/`files_indexed` reflect the last cycle.
    /// Public so one-shot drivers (`ragnarok digest`) can run a single
    /// pass without the polling loop.
    pub async fn remote_cycle(&self, source: &Source, cancel: &CancellationToken) -> Result<usize> {
        let docs = crate::remote::list(&source.config).await?;
        self.set_state(&source.id, |s| {
            s.files_total = docs.len() as u64;
            s.files_indexed = 0;
        });
        let mut seen = std::collections::HashSet::new();
        let mut indexed = 0usize;
        for d in &docs {
            if cancel.is_cancelled() {
                return Ok(indexed);
            }
            seen.insert(d.vpath.clone());
            match self.ingest_remote(&source.id, &source.config, d).await {
                Ok(IngestOutcome::Indexed) => {
                    indexed += 1;
                    self.set_state(&source.id, |s| s.files_indexed += 1);
                }
                // Skipped = already in the index — counts toward the indexed
                // total; Unsupported never lands in the index and must not
                Ok(IngestOutcome::Skipped) => self.set_state(&source.id, |s| s.files_indexed += 1),
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(vpath = %d.vpath, error = %e, "remote ingest failed");
                    self.set_state(&source.id, |s| {
                        s.last_error = Some(e.to_string());
                    });
                }
            }
        }
        // remote deletions propagate: anything indexed under this source
        // that the listing no longer advertises is dropped. Attachment
        // children (`parent#att:*`) are never in the listing — they live
        // or die with their parent (delete_document cascades), so a
        // child whose parent is still advertised must NOT be swept.
        for rec in self.engine.meta.list_documents(Some(&source.id))? {
            let child_of_seen = rec
                .path
                .split_once("#att:")
                .map(|(parent, _)| seen.contains(parent))
                .unwrap_or(false);
            if !seen.contains(&rec.path) && !child_of_seen {
                self.engine.delete_document(rec.id)?;
                self.emit_ingest(&source.id, &rec.path, IngestStatus::Deleted);
            }
        }
        Ok(indexed)
    }

    /// Long-running poll loop for one IMAP source (strictly read-only).
    pub async fn run_imap_source(self: &Arc<Self>, source: Source, cancel: CancellationToken) {
        let (host, port, mailbox, tls, username, password_env, stored_interval) =
            match &source.config {
                SourceConfig::Imap {
                    host,
                    port,
                    mailbox,
                    tls,
                    username,
                    password_env,
                    poll_interval_secs,
                } => (
                    host.clone(),
                    *port,
                    mailbox.clone(),
                    *tls,
                    username.clone(),
                    password_env.clone(),
                    *poll_interval_secs,
                ),
                _ => return,
            };

        let password = match imap::resolve_password(&password_env) {
            Ok(p) => p,
            Err(e) => {
                self.set_state(&source.id, |s| {
                    s.status = SyncStatus::Error;
                    s.last_error = Some(e.to_string());
                });
                return;
            }
        };

        loop {
            if cancel.is_cancelled() {
                return;
            }
            let interval = self.poll_interval(&source.id, stored_interval);
            self.set_state(&source.id, |s| s.status = SyncStatus::Syncing);
            let cycle = self
                .imap_cycle(&source.id, &host, port, tls, &username, &password, &mailbox)
                .await;
            match cycle {
                Ok(_n) => {
                    // files_indexed mirrors the actual index — a restart
                    // cycle ingests nothing new yet the mail docs persist
                    let in_index = self
                        .engine
                        .meta
                        .list_documents(Some(&source.id))
                        .map(|d| d.len() as u64)
                        .unwrap_or(0);
                    self.set_state(&source.id, |s| {
                        s.status = SyncStatus::Idle;
                        s.last_sync = Some(Utc::now());
                        s.files_indexed = in_index;
                    });
                }
                Err(e) => {
                    tracing::warn!(source = %source.name, error = %e, "imap cycle failed");
                    self.set_state(&source.id, |s| {
                        s.status = SyncStatus::Error;
                        s.last_error = Some(e.to_string());
                    });
                }
            }
            tokio::select! {
                _ = cancel.cancelled() => return,
                _ = tokio::time::sleep(imap::poll_duration(interval)) => {}
            }
        }
    }

    /// One incremental sync cycle against the mailbox.
    #[allow(clippy::too_many_arguments)]
    async fn imap_cycle(
        &self,
        source_id: &SourceId,
        host: &str,
        port: u16,
        tls: bool,
        username: &str,
        password: &str,
        mailbox: &str,
    ) -> Result<usize> {
        let client = imap::connect(host, port, tls).await?;
        let (mut session, mb) = imap::open_readonly(client, username, password, mailbox).await?;
        self.set_state(source_id, |s| s.files_total = mb.exists as u64);

        // UIDVALIDITY change invalidates all stored UIDs -> resync from 0
        let state = self.engine.meta.imap_get(source_id, mailbox)?;
        let mut last_uid = match state {
            Some((v, uid)) if v == mb.uid_validity => uid,
            _ => 0,
        };

        let mut indexed = 0usize;
        loop {
            let (msgs, highest) = imap::fetch_since(&mut session, last_uid, 256).await?;
            let batch_len = msgs.len();
            for msg in msgs {
                // `.eml` suffix routes the message through the email parser
                let vpath = format!("imap://{mailbox}/{}.eml", msg.uid);
                match self.ingest_virtual(source_id, &vpath, &msg.bytes).await {
                    Ok(IngestOutcome::Indexed | IngestOutcome::Skipped) => indexed += 1,
                    Ok(_) => {}
                    Err(e) => tracing::warn!(uid = msg.uid, error = %e, "imap ingest failed"),
                }
            }
            // persist whenever the cursor moved — including batches of
            // body-less messages where `msgs` is empty but `highest` ran
            // ahead; otherwise the next cycle refetches them forever
            if highest > last_uid {
                last_uid = highest;
                self.engine
                    .meta
                    .imap_update(source_id, mailbox, mb.uid_validity, last_uid)?;
            }
            if batch_len < 256 {
                break;
            }
        }
        imap::logout(&mut session).await;
        Ok(indexed)
    }

    /// Spawn tasks for every configured source; returns the join handles.
    /// A source that already has a task is first stopped — respawning
    /// must never stack watchers/pollers.
    pub async fn spawn_all(
        self: &Arc<Self>,
        sources: Vec<Source>,
        cancel: CancellationToken,
    ) -> Vec<tokio::task::JoinHandle<()>> {
        let mut handles = Vec::new();
        for src in sources {
            // persist source registration
            let _ = self.engine.meta.upsert_source(&src);
            // stop any task still running for this source (rescan /
            // re-register) — otherwise watchers and pollers stack up
            self.stop_source(&src.id);
            let this = self.clone();
            // child token: global cancel still kills everything, but the
            // source task can be stopped individually
            let child = cancel.child_token();
            self.tasks.insert(src.id.clone(), child.clone());
            match &src.config {
                SourceConfig::FileSystem { .. } => {
                    handles.push(tokio::spawn(async move {
                        this.run_fs_source(src, child).await;
                    }));
                }
                SourceConfig::Imap { .. } => {
                    handles.push(tokio::spawn(async move {
                        this.run_imap_source(src, child).await;
                    }));
                }
                SourceConfig::Paperless { .. }
                | SourceConfig::WebDav { .. }
                | SourceConfig::Web { .. }
                | SourceConfig::Feed { .. }
                | SourceConfig::S3 { .. }
                | SourceConfig::Git { .. }
                | SourceConfig::CalDav { .. }
                | SourceConfig::Maildir { .. } => {
                    handles.push(tokio::spawn(async move {
                        this.run_remote_source(src, child).await;
                    }));
                }
            }
        }
        handles
    }

    /// Stop the crawl task of one source, if any is running.
    pub fn stop_source(&self, id: &SourceId) {
        if let Some((_, tok)) = self.tasks.remove(id) {
            tok.cancel();
        }
    }
}

/// Map an advertised MIME type to a `DocType` for remote documents whose
/// parser path can't be identified from a file extension.
fn mime_doctype(mime: &str) -> ragnarok_core::DocType {
    use ragnarok_core::DocType;
    let m = mime.split(';').next().unwrap_or(mime).trim().to_lowercase();
    match m.as_str() {
        "application/pdf" => DocType::Pdf,
        "message/rfc822" => DocType::Email,
        "text/html" => DocType::Html,
        "text/markdown" => DocType::Markdown,
        "application/epub+zip" => DocType::Text,
        "application/zip" => DocType::Unknown,
        _ if m.contains("opendocument") => DocType::Word,
        _ if m == "text/csv" || m == "text/tab-separated-values" => DocType::Spreadsheet,
        _ if m.contains("jsonl") || m.contains("ndjson") => DocType::Text,
        _ if m.contains("word") => DocType::Word,
        _ if m.contains("sheet") || m.contains("excel") => DocType::Spreadsheet,
        _ if m.contains("presentation") || m.contains("powerpoint") => DocType::Presentation,
        _ if m.starts_with("image/") => DocType::Image,
        _ if m.starts_with("text/") => DocType::Text,
        _ => DocType::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ragnarok_core::{CrawlerConfig, DocType, ResourceConfig};
    use ragnarok_inference::HashingEmbedder;
    use ragnarok_store::{MetadataDb, VectorEngine};
    use std::fs;

    pub(super) async fn pipeline(dir: &Path) -> Arc<IngestPipeline> {
        let meta = Arc::new(MetadataDb::open(&dir.join("m.db")).unwrap());
        let vectors = VectorEngine::open(&dir.join("v"), 1024).unwrap();
        let embedder = Arc::new(HashingEmbedder::default());
        let engine = Arc::new(SearchEngine::new(vectors, meta, embedder.clone()));
        let bp = BackpressureController::new(ResourceConfig::default());
        let pool = InferencePool::new(embedder, 2, 4, bp.clone());
        let (tx, _rx) = broadcast::channel(16);
        IngestPipeline::new(engine, pool, bp, &CrawlerConfig::default(), tx)
    }

    #[tokio::test]
    async fn ingest_file_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let p = pipeline(dir.path()).await;
        let f = dir.path().join("note.txt");
        fs::write(&f, "the quick brown fox jumps").unwrap();
        let src = SourceId::from("s1");
        let out = p.ingest_path(&src, &f).await.unwrap();
        assert_eq!(out, IngestOutcome::Indexed);
        assert!(p.files_per_sec() >= 0.0);
        // second run -> skipped by hash
        let out = p.ingest_path(&src, &f).await.unwrap();
        assert_eq!(out, IngestOutcome::Skipped);
        // change content -> reindex
        fs::write(&f, "completely new content about walruses").unwrap();
        let out = p.ingest_path(&src, &f).await.unwrap();
        assert_eq!(out, IngestOutcome::Indexed);
        // remove
        assert!(p.remove_path(&src, &f).await.unwrap());
        assert!(!p.remove_path(&src, &f).await.unwrap());
    }

    #[tokio::test]
    async fn virtual_ingest_for_emails() {
        let dir = tempfile::tempdir().unwrap();
        let p = pipeline(dir.path()).await;
        let eml = b"From: a@x\r\nSubject: hello\r\n\r\nmail body text\r\n";
        let out = p
            .ingest_virtual(&SourceId::from("imap"), "imap://INBOX/42.eml", eml)
            .await
            .unwrap();
        assert_eq!(out, IngestOutcome::Indexed);
        let out = p
            .ingest_virtual(&SourceId::from("imap"), "imap://INBOX/42.eml", eml)
            .await
            .unwrap();
        assert_eq!(out, IngestOutcome::Skipped);
        // searchable
        let hits = p
            .engine
            .search(&ragnarok_core::SearchRequest::new("mail body"))
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_type, DocType::Email);
    }

    /// Full source lifecycle: initial scan indexes files, watch task picks
    /// up new files, deletion removes them from the index.
    #[tokio::test]
    async fn run_fs_source_scans_and_watches() {
        let dir = tempfile::tempdir().unwrap();
        let p = pipeline(dir.path()).await;
        let docs = tempfile::tempdir().unwrap();
        fs::write(docs.path().join("one.txt"), "alpha bravo charlie").unwrap();

        let src = Source::new(
            "live",
            SourceConfig::FileSystem {
                path: docs.path().to_path_buf(),
                recursive: true,
                include: vec![],
                exclude: vec![],
                watch: true,
            },
        );
        let sid = src.id.clone();
        let cancel = CancellationToken::new();
        let task = tokio::spawn({
            let p = p.clone();
            let c = cancel.clone();
            async move { p.run_fs_source(src, c).await }
        });
        // initial scan picks up the file; the source transitions to
        // Watching only after the scan finishes — poll for both
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let st = p.states().into_iter().find(|s| s.source_id == sid);
                if p.engine.meta.doc_count().unwrap() == 1
                    && st.is_some_and(|s| s.status == SyncStatus::Watching)
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
        .await
        .unwrap();
        let st = p.states().into_iter().find(|s| s.source_id == sid).unwrap();
        assert_eq!(st.files_total, 1);

        // new file lands in the index via the debounced watcher
        fs::write(docs.path().join("two.txt"), "delta echo foxtrot").unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if p.engine.meta.doc_count().unwrap() == 2 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
        .await
        .unwrap();

        // deletion removes it
        fs::remove_file(docs.path().join("two.txt")).unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if p.engine.meta.doc_count().unwrap() == 1 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
        .await
        .unwrap();

        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .unwrap()
            .unwrap();
    }

    /// `watch = false` sources go Idle after the initial scan.
    #[tokio::test]
    async fn run_fs_source_no_watch_goes_idle() {
        let dir = tempfile::tempdir().unwrap();
        let p = pipeline(dir.path()).await;
        let docs = tempfile::tempdir().unwrap();
        fs::write(docs.path().join("a.txt"), "idle scan content").unwrap();
        let src = Source::new(
            "once",
            SourceConfig::FileSystem {
                path: docs.path().to_path_buf(),
                recursive: false,
                include: vec![],
                exclude: vec![],
                watch: false,
            },
        );
        let sid = src.id.clone();
        p.run_fs_source(src, CancellationToken::new()).await;
        let st = p.states().into_iter().find(|s| s.source_id == sid).unwrap();
        assert_eq!(st.status, SyncStatus::Idle);
        assert!(st.last_sync.is_some());
        assert_eq!(p.engine.meta.doc_count().unwrap(), 1);
    }

    /// Unsupported file type is skipped, not an error.
    #[tokio::test]
    async fn ingest_skips_unsupported() {
        let dir = tempfile::tempdir().unwrap();
        let p = pipeline(dir.path()).await;
        let f = dir.path().join("img.xyz");
        fs::write(&f, vec![0u8, 159, 146, 150]).unwrap();
        let out = p.ingest_path(&SourceId::from("s"), &f).await.unwrap();
        assert_eq!(out, IngestOutcome::Unsupported);
        // unparseable but known extension -> error propagates
        let bad = dir.path().join("bad.pdf");
        fs::write(&bad, b"definitely not a pdf").unwrap();
        let r = p.ingest_path(&SourceId::from("s"), &bad).await;
        assert!(r.is_err());
    }

    /// spawn_all registers sources and returns one handle per source.
    #[tokio::test]
    async fn spawn_all_registers_and_spawns() {
        let dir = tempfile::tempdir().unwrap();
        let p = pipeline(dir.path()).await;
        let docs = tempfile::tempdir().unwrap();
        fs::write(docs.path().join("x.txt"), "spawn all test").unwrap();
        let src = Source::new(
            "s",
            SourceConfig::FileSystem {
                path: docs.path().to_path_buf(),
                recursive: false,
                include: vec![],
                exclude: vec![],
                watch: false,
            },
        );
        let handles = p
            .spawn_all(vec![src.clone()], CancellationToken::new())
            .await;
        assert_eq!(handles.len(), 1);
        for h in handles {
            h.await.unwrap();
        }
        assert!(p
            .engine
            .meta
            .list_sources()
            .unwrap()
            .iter()
            .any(|s| s.id == src.id));
    }

    /// A malformed source config drives the state to Error.
    #[tokio::test]
    async fn run_fs_source_wrong_config_type_errors() {
        let dir = tempfile::tempdir().unwrap();
        let p = pipeline(dir.path()).await;
        let src = Source::new(
            "bad",
            SourceConfig::Imap {
                host: "h".into(),
                port: 993,
                mailbox: "INBOX".into(),
                tls: true,
                username: "u".into(),
                password_env: "P".into(),
                poll_interval_secs: 60,
            },
        );
        let sid = src.id.clone();
        p.run_fs_source(src, CancellationToken::new()).await;
        let st = p.states().into_iter().find(|s| s.source_id == sid).unwrap();
        assert_eq!(st.status, SyncStatus::Error);
    }

    /// Full read-only IMAP sync over a real socket: scripted server,
    /// run_imap_source -> imap_cycle -> fetch_since -> ingest_virtual ->
    /// blob storage + imap_state persistence.
    #[tokio::test]
    async fn run_imap_source_syncs_over_socket() {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

        let dir = tempfile::tempdir().unwrap();
        let p = pipeline(dir.path()).await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let transcript = Arc::new(parking_lot::Mutex::new(Vec::<String>::new()));
        let t2 = transcript.clone();

        tokio::spawn(async move {
            loop {
                let Ok((sock, _)) = listener.accept().await else {
                    break;
                };
                let t3 = t2.clone();
                tokio::spawn(async move {
                    let (r, mut w) = sock.into_split();
                    let mut lines = BufReader::new(r).lines();
                    w.write_all(b"* OK ready\r\n").await.unwrap();
                    while let Ok(Some(line)) = lines.next_line().await {
                        t3.lock().push(line.clone());
                        let tag = line.split_whitespace().next().unwrap_or("a1");
                        let up = line.to_uppercase();
                        let resp = if up.contains(" LOGIN ") {
                            format!("{tag} OK\r\n")
                        } else if up.contains(" EXAMINE ") {
                            format!(
                                "* 2 EXISTS\r\n* OK [UIDVALIDITY 55] ok\r\n\
                                 {tag} OK [READ-ONLY] EXAMINE done\r\n"
                            )
                        } else if up.contains("UID FETCH") {
                            let m = b"From: sync@x\r\nSubject: synced\r\n\r\nimap body\r\n";
                            format!(
                                "* 1 FETCH (UID 77 BODY[] {{{}}}\r\n{}\r\n)\r\n{tag} OK\r\n",
                                m.len() + 2,
                                String::from_utf8_lossy(m)
                            )
                        } else if up.contains("LOGOUT") {
                            format!("* BYE\r\n{tag} OK\r\n")
                        } else {
                            format!("{tag} OK\r\n")
                        };
                        if w.write_all(resp.as_bytes()).await.is_err() {
                            break;
                        }
                    }
                });
            }
        });

        unsafe { std::env::set_var("RAGNAROK_TEST_IMAP_PW", "pw") };
        let src = Source::new(
            "mail",
            SourceConfig::Imap {
                host: "127.0.0.1".into(),
                port,
                mailbox: "INBOX".into(),
                tls: false,
                username: "u".into(),
                password_env: "RAGNAROK_TEST_IMAP_PW".into(),
                poll_interval_secs: 3600,
            },
        );
        let sid = src.id.clone();
        let cancel = CancellationToken::new();
        let p2 = p.clone();
        let c = cancel.clone();
        let task = tokio::spawn(async move { p2.run_imap_source(src, c).await });

        // wait until the cycle indexed the message
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if p.engine.meta.doc_count().unwrap() >= 1 {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .expect("imap sync timed out");
        cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(5), task).await;

        // document ingested under the virtual path, blob persisted, uid saved
        let doc = p
            .engine
            .meta
            .get_document_by_path(&sid, "imap://INBOX/77.eml")
            .unwrap()
            .expect("virtual doc indexed");
        assert_eq!(doc.doc_type, DocType::Email);
        assert!(p.engine.meta.get_blob(doc.id).unwrap().is_some());
        assert_eq!(
            p.engine.meta.imap_get(&sid, "INBOX").unwrap(),
            Some((55, 77))
        );

        // searchable + source state observed
        let hits = p
            .engine
            .search(&ragnarok_core::SearchRequest::new("imap body"))
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);

        // protocol audit: only read-only commands on the wire
        let all = transcript.lock().join("\n").to_uppercase();
        assert!(all.contains(" EXAMINE "));
        assert!(all.contains("UID FETCH"));
        for bad in ["SELECT", "STORE", "EXPUNGE", "DELETE", "COPY", "APPEND"] {
            assert!(!all.contains(bad), "{bad} emitted: {all}");
        }
        unsafe { std::env::remove_var("RAGNAROK_TEST_IMAP_PW") };
    }

    /// IMAP source with missing password env fails into Error state.
    #[tokio::test]
    async fn run_imap_source_missing_password() {
        let dir = tempfile::tempdir().unwrap();
        let p = pipeline(dir.path()).await;
        let src = Source::new(
            "mail",
            SourceConfig::Imap {
                host: "127.0.0.1".into(),
                port: 1, // unreachable anyway
                mailbox: "INBOX".into(),
                tls: false,
                username: "u".into(),
                password_env: "RAGNAROK_DEFINITELY_MISSING_PW".into(),
                poll_interval_secs: 3600,
            },
        );
        let sid = src.id.clone();
        let p2 = p.clone();
        let c = CancellationToken::new();
        let task = tokio::spawn(async move { p2.run_imap_source(src, c).await });
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .unwrap()
            .unwrap();
        let st = p.states().into_iter().find(|s| s.source_id == sid).unwrap();
        assert_eq!(st.status, SyncStatus::Error);
        assert!(st
            .last_error
            .unwrap()
            .contains("RAGNAROK_DEFINITELY_MISSING_PW"));
    }

    /// Remote (non-copying) source lifecycle over a real socket:
    /// listing → index via stripped text → etag-dedupe skip → remote
    /// deletion propagates. Nothing lands in the blob table or on disk.
    #[tokio::test]
    async fn run_web_source_cycle_end_to_end() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::sync::{Arc as SArc, Mutex as SMutex};

        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", l.local_addr().unwrap());
        let hits = SArc::new(SMutex::new(Vec::<String>::new()));
        let hits2 = hits.clone();
        std::thread::spawn(move || {
            for stream in l.incoming() {
                let Ok(mut s) = stream else { continue };
                let mut buf = [0u8; 4096];
                let n = s.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]);
                let path = req.split_whitespace().nth(1).unwrap_or("/").to_owned();
                hits2.lock().unwrap().push(path.clone());
                let (body, ctype) = match path.as_str() {
                    "/a" => (
                        "<html><title>Seite A</title><body><p>web seite alpha inhalt</p></body></html>",
                        "text/html",
                    ),
                    "/b" => ("<html><body>seite beta text</body></html>", "text/html"),
                    _ => continue,
                };
                let _ = s.write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nETag: \"e-{path}\"\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                );
            }
        });

        let dir = tempfile::tempdir().unwrap();
        let p = pipeline(dir.path()).await;
        let mut src = Source::new(
            "websrc",
            SourceConfig::Web {
                urls: vec![format!("{base}/a"), format!("{base}/b")],
                follow_links: false,
                poll_interval_secs: 3600,
            },
        );
        let sid = src.id.clone();
        p.engine.meta.upsert_source(&src).unwrap();
        let cancel = CancellationToken::new();

        // cycle 1: both pages indexed, OCR/strip path — no blobs stored
        let n = p.remote_cycle(&src, &cancel).await.unwrap();
        assert_eq!(n, 2);
        assert_eq!(p.engine.meta.doc_count().unwrap(), 2);
        assert!(hits.lock().unwrap().len() >= 2);
        let rec = p
            .engine
            .meta
            .get_document_by_path(&sid, &format!("{base}/a"))
            .unwrap()
            .unwrap();
        assert_eq!(rec.hash, "\"e-/a\""); // remote etag is the hash
        assert!(p.engine.meta.get_blob(rec.id).unwrap().is_none());
        let hits_search = p
            .engine
            .search(&ragnarok_core::SearchRequest::new("alpha inhalt"))
            .await
            .unwrap();
        assert!(!hits_search.is_empty());
        assert_eq!(hits_search[0].doc_type, DocType::Html);
        assert_eq!(hits_search[0].path, format!("{base}/a"));

        // cycle 2: etags unchanged → nothing re-indexed
        let n = p.remote_cycle(&src, &cancel).await.unwrap();
        assert_eq!(n, 0);

        // remote deletion propagates to the index
        if let SourceConfig::Web { urls, .. } = &mut src.config {
            urls.truncate(1);
        }
        let n = p.remote_cycle(&src, &cancel).await.unwrap();
        assert_eq!(n, 0);
        assert_eq!(p.engine.meta.doc_count().unwrap(), 1);
        assert!(p
            .engine
            .meta
            .get_document_by_path(&sid, &format!("{base}/b"))
            .unwrap()
            .is_none());
    }
}

#[cfg(test)]
mod attachment_tests {
    use super::tests::pipeline;
    use ragnarok_core::SourceId;
    use std::io::Write;

    fn eml_with_txt_attachment() -> Vec<u8> {
        // text attachment → child doc with parent metadata
        let inner = b"aW52b2ljZSA0NzExIGRvY3VtZW50IGNvbnRlbnQ="; // base64 "invoice 4711 document content"
        format!(
            "From: a@x\r\nSubject: s\r\nMIME-Version: 1.0\r\n\
Content-Type: multipart/mixed; boundary=\"BB\"\r\n\r\n\
--BB\r\nContent-Type: text/plain\r\n\r\nmail body text\r\n\
--BB\r\nContent-Type: text/plain; name=\"rechnung.txt\"\r\n\
Content-Disposition: attachment; filename=\"rechnung.txt\"\r\n\
Content-Transfer-Encoding: base64\r\n\r\n{}\r\n--BB--\r\n",
            String::from_utf8_lossy(inner)
        )
        .into_bytes()
    }

    #[tokio::test]
    async fn eml_attachment_ingests_as_child() {
        let dir = tempfile::tempdir().unwrap();
        let p = pipeline(dir.path()).await;
        let src = SourceId::from("s1");
        let out = p
            .ingest_virtual(&src, "/mail/m.eml", &eml_with_txt_attachment())
            .await
            .unwrap();
        assert_eq!(out, super::IngestOutcome::Indexed);
        let child = p
            .engine
            .meta
            .get_document_by_path(&src, "/mail/m.eml#att:rechnung.txt")
            .unwrap()
            .expect("child doc missing");
        assert_eq!(child.metadata["parent"], "/mail/m.eml");
        assert_eq!(child.metadata["attachment_idx"], "0");
        // child text is searchable
        let hits = p
            .engine
            .search(&ragnarok_core::SearchRequest::new("invoice 4711"))
            .await
            .unwrap();
        assert!(hits.iter().any(|h| h.path.contains("#att:")), "{hits:?}");
        // dedup: same mail again → no duplicate child
        let _ = p
            .ingest_virtual(&src, "/mail/m.eml", &eml_with_txt_attachment())
            .await
            .unwrap();
        let docs = p.engine.meta.list_documents(Some(&src)).unwrap();
        assert_eq!(docs.iter().filter(|d| d.path.contains("#att:")).count(), 1);
        // cascade: deleting the parent removes the child
        let parent = p
            .engine
            .meta
            .get_document_by_path(&src, "/mail/m.eml")
            .unwrap()
            .unwrap();
        p.engine.delete_document(parent.id).unwrap();
        assert!(p
            .engine
            .meta
            .get_document_by_path(&src, "/mail/m.eml#att:rechnung.txt")
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn zip_members_ingest_and_skip_unsupported() {
        let dir = tempfile::tempdir().unwrap();
        let p = pipeline(dir.path()).await;
        let src = SourceId::from("s1");
        let mut w = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let opts = zip::write::SimpleFileOptions::default();
        w.start_file("notizen.txt", opts).unwrap();
        w.write_all(b"zip member notes walrus content").unwrap();
        w.start_file("binary.bin", opts).unwrap();
        w.write_all(&[0u8, 159, 146, 150][..]).unwrap();
        let z = w.finish().unwrap().into_inner();
        let out = p.ingest_virtual(&src, "/docs/a.zip", &z).await.unwrap();
        assert_eq!(out, super::IngestOutcome::Indexed);
        assert!(p
            .engine
            .meta
            .get_document_by_path(&src, "/docs/a.zip#att:notizen.txt")
            .unwrap()
            .is_some());
        // .bin member has no parser → skipped individually, not fatal
        assert!(p
            .engine
            .meta
            .get_document_by_path(&src, "/docs/a.zip#att:binary.bin")
            .unwrap()
            .is_none());
        // search hits the member content
        let hits = p
            .engine
            .search(&ragnarok_core::SearchRequest::new("walrus"))
            .await
            .unwrap();
        assert!(hits.iter().any(|h| h.path.contains("#att:")));
    }

    #[tokio::test]
    async fn remote_cycle_sweep_keeps_advertised_att_children() {
        // a maildir message with a text attachment: the cycle's delete
        // sweep must not drop `parent#att:` children — they are never in
        // the listing, they live and die with their parent
        let dir = tempfile::tempdir().unwrap();
        let mdir = tempfile::tempdir().unwrap();
        for sub in ["cur", "new", "tmp"] {
            std::fs::create_dir_all(mdir.path().join(sub)).unwrap();
        }
        let msg = mdir.path().join("cur/1001:2,S");
        std::fs::write(&msg, eml_with_txt_attachment()).unwrap();

        let p = pipeline(dir.path()).await;
        let src = ragnarok_core::Source::new(
            "md",
            ragnarok_core::SourceConfig::Maildir {
                path: mdir.path().to_path_buf(),
                include_subfolders: false,
                poll_interval_secs: 300,
            },
        );
        let sid = src.id.clone();
        p.engine.meta.upsert_source(&src).unwrap();
        let cancel = tokio_util::sync::CancellationToken::new();

        let parent_path = "maildir://INBOX/1001.eml";
        let child_path = format!("{parent_path}#att:rechnung.txt");
        p.remote_cycle(&src, &cancel).await.unwrap();
        assert!(p
            .engine
            .meta
            .get_document_by_path(&sid, parent_path)
            .unwrap()
            .is_some());
        assert!(
            p.engine
                .meta
                .get_document_by_path(&sid, &child_path)
                .unwrap()
                .is_some(),
            "attachment child swept by remote_cycle"
        );

        // second cycle: unchanged mail skips ingest — child survives
        p.remote_cycle(&src, &cancel).await.unwrap();
        assert!(p
            .engine
            .meta
            .get_document_by_path(&sid, &child_path)
            .unwrap()
            .is_some());

        // message vanishes → parent is swept and the child cascades
        std::fs::remove_file(&msg).unwrap();
        p.remote_cycle(&src, &cancel).await.unwrap();
        assert!(p
            .engine
            .meta
            .get_document_by_path(&sid, parent_path)
            .unwrap()
            .is_none());
        assert!(p
            .engine
            .meta
            .get_document_by_path(&sid, &child_path)
            .unwrap()
            .is_none());
    }
}
