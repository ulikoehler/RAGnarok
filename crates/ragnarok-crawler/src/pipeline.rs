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
use std::sync::atomic::{AtomicUsize, Ordering};
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
    cfg: ChunkerConfig,
    max_file_bytes: u64,
    parsers: Vec<Box<dyn Parser>>,
    event_tx: broadcast::Sender<TelemetryEvent>,
    /// recent successful ingests for throughput telemetry
    recent: parking_lot::Mutex<VecDeque<Instant>>,
    queue_depth: AtomicUsize,
    states: DashMap<SourceId, SourceState>,
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
            cfg: ChunkerConfig {
                min_tokens: crawler.chunk_min_tokens,
                max_tokens: crawler.chunk_max_tokens,
                overlap: crawler.chunk_overlap,
            },
            max_file_bytes: crawler.max_file_bytes,
            parsers: default_registry(),
            event_tx,
            recent: parking_lot::Mutex::new(VecDeque::new()),
            queue_depth: AtomicUsize::new(0),
            states: DashMap::new(),
        })
    }

    pub fn queue_depth(&self) -> usize {
        self.queue_depth.load(Ordering::Relaxed)
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

        let hash = {
            let p = path.to_path_buf();
            tokio::task::spawn_blocking(move || hash_file(&p))
                .await
                .map_err(|e| Error::Internal(format!("join: {e}")))??
        };

        // dedupe: same path+hash -> skip; same hash elsewhere -> reindex cheaply
        if let Some(existing) = self
            .engine
            .meta
            .get_document_by_path(source_id, &path_str)?
        {
            if existing.hash == hash {
                self.emit_ingest(source_id, &path_str, IngestStatus::Skipped);
                return Ok(IngestOutcome::Skipped);
            }
        }

        let bytes = tokio::fs::read(path).await?;
        let doc_id = ragnarok_core::document_id(source_id, &path_str);
        self.index_bytes(source_id, doc_id, &path_str, path, &bytes, hash)
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

        let parsed = match parse_with(&self.parsers, path, bytes) {
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
        )
        .await
    }

    /// Chunk + embed + store one parsed document.
    async fn index_parsed(
        &self,
        source_id: &SourceId,
        doc_id: ragnarok_core::DocumentId,
        path_str: &str,
        parsed: ParsedDocument,
        size: u64,
        hash: String,
    ) -> Result<IngestOutcome> {
        let chunks = chunk_document(doc_id, &parsed, self.cfg)?;
        if chunks.is_empty() {
            self.emit_ingest(source_id, path_str, IngestStatus::Skipped);
            return Ok(IngestOutcome::Skipped);
        }
        self.queue_depth.fetch_add(1, Ordering::Relaxed);
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let triples = self.pool.embed_batch(&texts).await?;
        let record = DocumentRecord {
            id: doc_id,
            source_id: source_id.clone(),
            path: path_str.to_string(),
            hash,
            doc_type: parsed.doc_type,
            size,
            modified: std::fs::metadata(path_str)
                .ok()
                .and_then(|m| m.modified().ok())
                .map(chrono::DateTime::<Utc>::from),
            indexed_at: Utc::now(),
            chunk_count: chunks.len() as u32,
            metadata: parsed.metadata.clone(),
        };
        self.engine
            .index_document(&record, &chunks, &triples)
            .await?;
        self.queue_depth.fetch_sub(1, Ordering::Relaxed);
        self.mark_indexed();
        self.emit_ingest(source_id, path_str, IngestStatus::Indexed);
        Ok(IngestOutcome::Indexed)
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
        self.set_state(&source.id, |s| s.status = SyncStatus::Scanning);
        let files = scan_directory(&root, recursive, &include, &exclude, self.max_file_bytes);
        let total = files.len() as u64;
        self.set_state(&source.id, |s| s.files_total = total);
        for path in &files {
            if cancel.is_cancelled() {
                return;
            }
            match self.ingest_path(&source.id, path).await {
                Ok(IngestOutcome::Indexed) => self.set_state(&source.id, |s| {
                    s.files_indexed += 1;
                }),
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
                            let _ = self.ingest_path(&source.id, &ev.path).await;
                        }
                        FsEventKind::Removed => {
                            let _ = self.remove_path(&source.id, &ev.path).await;
                        }
                    }
                }
            }
        }
    }

    /// Long-running poll loop for one IMAP source (strictly read-only).
    pub async fn run_imap_source(self: &Arc<Self>, source: Source, cancel: CancellationToken) {
        let (host, port, mailbox, tls, username, password_env, interval) = match &source.config {
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
            self.set_state(&source.id, |s| s.status = SyncStatus::Syncing);
            let cycle = self
                .imap_cycle(&source.id, &host, port, tls, &username, &password, &mailbox)
                .await;
            match cycle {
                Ok(n) => self.set_state(&source.id, |s| {
                    s.status = SyncStatus::Idle;
                    s.last_sync = Some(Utc::now());
                    s.files_indexed += n as u64;
                }),
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
        let (mut session, uidvalidity) =
            imap::open_readonly(client, username, password, mailbox).await?;

        // UIDVALIDITY change invalidates all stored UIDs -> resync from 0
        let state = self.engine.meta.imap_get(mailbox)?;
        let mut last_uid = match state {
            Some((v, uid)) if v == uidvalidity => uid,
            _ => 0,
        };

        let mut indexed = 0usize;
        loop {
            let (msgs, highest) = imap::fetch_since(&mut session, last_uid, 256).await?;
            let batch_len = msgs.len();
            if batch_len == 0 {
                break;
            }
            for msg in msgs {
                // `.eml` suffix routes the message through the email parser
                let vpath = format!("imap://{mailbox}/{}.eml", msg.uid);
                match self.ingest_virtual(source_id, &vpath, &msg.bytes).await {
                    Ok(_) => indexed += 1,
                    Err(e) => tracing::warn!(uid = msg.uid, error = %e, "imap ingest failed"),
                }
            }
            last_uid = highest;
            self.engine
                .meta
                .imap_update(mailbox, uidvalidity, last_uid)?;
            if batch_len < 256 {
                break;
            }
        }
        imap::logout(&mut session).await;
        Ok(indexed)
    }

    /// Spawn tasks for every configured source; returns the join handles.
    pub async fn spawn_all(
        self: &Arc<Self>,
        sources: Vec<Source>,
        cancel: CancellationToken,
    ) -> Vec<tokio::task::JoinHandle<()>> {
        let mut handles = Vec::new();
        for src in sources {
            // persist source registration
            let _ = self.engine.meta.upsert_source(&src);
            let this = self.clone();
            let cancel = cancel.clone();
            match &src.config {
                SourceConfig::FileSystem { .. } => {
                    handles.push(tokio::spawn(async move {
                        this.run_fs_source(src, cancel).await;
                    }));
                }
                SourceConfig::Imap { .. } => {
                    handles.push(tokio::spawn(async move {
                        this.run_imap_source(src, cancel).await;
                    }));
                }
            }
        }
        handles
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ragnarok_core::{CrawlerConfig, DocType, ResourceConfig};
    use ragnarok_inference::HashingEmbedder;
    use ragnarok_store::{MetadataDb, VectorEngine};
    use std::fs;

    async fn pipeline(dir: &Path) -> Arc<IngestPipeline> {
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
}
