use ragnarok_core::{Source, SourceState, SystemStats, TelemetryEvent};
use ragnarok_crawler::IngestPipeline;
use ragnarok_resource::ResourceMonitor;
use ragnarok_store::SearchEngine;
use serde::Serialize;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

/// Everything the handlers need, shared across requests.
pub struct AppState {
    pub engine: Arc<SearchEngine>,
    pub monitor: Arc<ResourceMonitor>,
    pub pipeline: Arc<IngestPipeline>,
    /// Telemetry event bus: ingest progress, backpressure, source states.
    /// The WS task additionally injects periodic `Stats` snapshots.
    pub events: broadcast::Sender<TelemetryEvent>,
    pub cancel: CancellationToken,
    pub started: Instant,
    pub telemetry_interval: Duration,
    /// Live HTTP connection count for the stats endpoint.
    pub connections: AtomicUsize,
}

impl AppState {
    pub fn new(
        engine: Arc<SearchEngine>,
        monitor: Arc<ResourceMonitor>,
        pipeline: Arc<IngestPipeline>,
        events: broadcast::Sender<TelemetryEvent>,
        cancel: CancellationToken,
        telemetry_interval: Duration,
    ) -> Arc<Self> {
        Arc::new(AppState {
            engine,
            monitor,
            pipeline,
            events,
            cancel,
            started: Instant::now(),
            telemetry_interval,
            connections: AtomicUsize::new(0),
        })
    }

    /// Compose the current `SystemStats` snapshot.
    pub fn stats(&self) -> SystemStats {
        let t = self.monitor.latest();
        let bp = self.monitor.controller();
        let est = self
            .engine
            .stats()
            .unwrap_or_else(|_| ragnarok_store::EngineStats {
                documents: 0,
                chunks: 0,
                dense_points: 0,
                sparse_docs: 0,
                colbert_docs: 0,
            });
        SystemStats {
            ram_total_bytes: t.total_ram,
            ram_available_bytes: t.available_ram,
            ram_used_bytes: t.used_ram,
            process_rss_bytes: t.process_rss,
            budget_bytes: bp.budget(),
            cpu_percent: t.cpu_percent,
            thread_count: t.thread_count + bp.in_flight(),
            pressure: bp.level(),
            indexed_documents: est.documents,
            indexed_chunks: est.chunks,
            files_per_sec: self.pipeline.files_per_sec(),
            queue_depth: self.pipeline.queue_depth(),
            uptime_secs: self.started.elapsed().as_secs(),
        }
    }

    /// Registered sources joined with their live sync state.
    pub fn sources_with_state(&self) -> Vec<SourceWithState> {
        let states: std::collections::HashMap<_, _> = self
            .pipeline
            .states()
            .into_iter()
            .map(|s| (s.source_id.clone(), s))
            .collect();
        self.engine
            .meta
            .list_sources()
            .unwrap_or_default()
            .into_iter()
            .map(|s| {
                let state = states
                    .get(&s.id)
                    .cloned()
                    .unwrap_or_else(|| SourceState::new(s.id.clone()));
                SourceWithState {
                    id: s.id,
                    name: s.name,
                    config: s.config,
                    state,
                }
            })
            .collect()
    }
}

/// `GET /api/v1/sources` response element.
#[derive(Debug, Clone, Serialize)]
pub struct SourceWithState {
    pub id: ragnarok_core::SourceId,
    pub name: String,
    pub config: ragnarok_core::SourceConfig,
    pub state: SourceState,
}

impl From<(Source, SourceState)> for SourceWithState {
    fn from((s, st): (Source, SourceState)) -> Self {
        SourceWithState {
            id: s.id,
            name: s.name,
            config: s.config,
            state: st,
        }
    }
}
