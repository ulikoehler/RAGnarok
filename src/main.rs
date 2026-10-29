//! RAGnarök daemon entrypoint — CLI, lifecycle manager, graceful shutdown.
//!
//! ```text
//! ragnarok serve [--config FILE]          run the daemon (default)
//! ragnarok index <path> [--name N]        one-shot: index a directory and exit
//! ragnarok sources                        list configured sources
//! ragnarok search <query> [-n LIMIT]      run a hybrid search and print hits
//! ```

use anyhow::Result;
use clap::{Parser, Subcommand};
use ragnarok_core::{
    default_config_path, InferenceBackend, RagnarokConfig, SearchRequest, Source, SourceConfig,
};
use ragnarok_crawler::IngestPipeline;
use ragnarok_inference::{Embedder, HashingEmbedder, InferencePool, DENSE_DIM};
use ragnarok_resource::ResourceMonitor;
use ragnarok_server::{self, AppState};
use ragnarok_store::SearchEngine;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

#[derive(Parser)]
#[command(
    name = "ragnarok",
    version,
    about = "RAGnarök — embedded hybrid vector search & RAG engine"
)]
struct Cli {
    /// Path to config.toml (default: ~/.config/ragnarok/config.toml)
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    /// Override the data directory
    #[arg(long, global = true)]
    data_dir: Option<PathBuf>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run the daemon: REST + WebSocket API, watchers, IMAP pollers
    Serve,
    /// Index a directory once and exit (no watcher, no server)
    Index {
        path: PathBuf,
        #[arg(long, default_value = "cli-index")]
        name: String,
        /// Keep watching for changes after the initial scan
        #[arg(long)]
        watch: bool,
    },
    /// List registered sources
    Sources,
    /// Run a hybrid search against the local index
    Search {
        query: String,
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ragnarok=info,tower_http=warn".into()),
        )
        .init();

    let cli = Cli::parse();
    let config_path = cli.config.unwrap_or_else(default_config_path);
    let mut cfg = RagnarokConfig::load_or_default(&config_path)?;
    if let Some(d) = cli.data_dir {
        cfg.data_dir = d;
    }
    cfg.validate()?;
    std::fs::create_dir_all(&cfg.data_dir)?;

    match cli.command.unwrap_or(Command::Serve) {
        Command::Serve => serve(cfg, &config_path).await,
        Command::Index { path, name, watch } => index(path, name, watch, cfg).await,
        Command::Sources => sources(cfg).await,
        Command::Search { query, limit } => search(query, limit, cfg).await,
    }
}

/// Build the embedder for the configured backend — provisioning the ONNX
/// model first when it is missing and `auto_provision` allows.
async fn build_embedder(
    cfg: &RagnarokConfig,
    provisioner: &Arc<ragnarok_inference::Provisioner>,
    events: &broadcast::Sender<ragnarok_core::TelemetryEvent>,
    cancel: CancellationToken,
) -> Result<Arc<dyn Embedder>> {
    match cfg.inference.backend {
        InferenceBackend::Hashing => Ok(Arc::new(HashingEmbedder::default())),
        #[cfg(feature = "onnx")]
        InferenceBackend::Onnx => {
            if !ragnarok_inference::Provisioner::ready_in(provisioner.dir()) {
                if !cfg.inference.auto_provision {
                    anyhow::bail!(
                        "onnx model missing in {} and inference.auto_provision is off",
                        provisioner.dir().display()
                    );
                }
                tracing::info!(
                    "onnx model missing — provisioning into {}",
                    provisioner.dir().display()
                );
                provisioner.download(Some(events), cancel).await?;
            }
            Ok(Arc::new(ragnarok_inference::onnx::OnnxBgeM3::load(
                provisioner.dir(),
                cfg.inference.max_colbert_tokens,
            )?))
        }
        #[cfg(not(feature = "onnx"))]
        InferenceBackend::Onnx => {
            let _ = (provisioner, events, cancel); // used only by the onnx arm
            anyhow::bail!(
                "binary built without `onnx` feature — rebuild with --features onnx \
                 or set inference.backend = \"hashing\""
            )
        }
    }
}

/// Provision + load the cross-encoder reranker when the `onnx` feature is
/// compiled and `inference.rerank` is on. Never fatal — a missing/failed
/// reranker just drops the stage (hybrid order is kept).
async fn build_reranker(
    cfg: &RagnarokConfig,
    provisioner: &Arc<ragnarok_inference::Provisioner>,
    events: &broadcast::Sender<ragnarok_core::TelemetryEvent>,
    cancel: CancellationToken,
) -> Option<Arc<dyn ragnarok_inference::Reranker>> {
    #[cfg(feature = "onnx")]
    {
        if !cfg.inference.rerank {
            return None;
        }
        if !ragnarok_inference::Provisioner::ready_in(provisioner.dir()) {
            if !cfg.inference.auto_provision {
                tracing::warn!(
                    "reranker model missing in {} and inference.auto_provision is off — rerank disabled",
                    provisioner.dir().display()
                );
                return None;
            }
            if let Err(e) = provisioner.download(Some(events), cancel).await {
                tracing::warn!("reranker provisioning failed ({e}) — rerank disabled");
                return None;
            }
        }
        match ragnarok_inference::rerank::OnnxReranker::load(provisioner.dir()) {
            Ok(r) => {
                tracing::info!(
                    "cross-encoder reranker ready ({})",
                    provisioner.dir().display()
                );
                Some(r as Arc<dyn ragnarok_inference::Reranker>)
            }
            Err(e) => {
                tracing::warn!("reranker load failed ({e}) — rerank disabled");
                None
            }
        }
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (cfg, provisioner, events, cancel);
        None
    }
}

/// Shared bootstrap: engine + monitor + pool + pipeline.
struct Runtime {
    engine: Arc<SearchEngine>,
    monitor: Arc<ResourceMonitor>,
    pipeline: Arc<IngestPipeline>,
    pool: Arc<ragnarok_inference::InferencePool>,
    events: broadcast::Sender<ragnarok_core::TelemetryEvent>,
    provisioner: Arc<ragnarok_inference::Provisioner>,
    reranker_provisioner: Arc<ragnarok_inference::Provisioner>,
    cancel: CancellationToken,
}

async fn bootstrap(cfg: &RagnarokConfig) -> Result<Runtime> {
    let (tx, _rx) = broadcast::channel(512);
    let cancel = CancellationToken::new();
    let model_dir = cfg
        .inference
        .model_dir
        .clone()
        .unwrap_or_else(|| cfg.data_dir.join("models/bge-m3"));
    let provisioner = ragnarok_inference::Provisioner::new(
        model_dir,
        cfg.inference.model_url.clone(),
        cfg.inference.model_sha256.clone(),
    );
    let reranker_dir = cfg
        .inference
        .reranker_dir
        .clone()
        .unwrap_or_else(|| cfg.data_dir.join("models/bge-reranker"));
    let reranker_provisioner = ragnarok_inference::Provisioner::for_reranker(
        reranker_dir,
        cfg.inference.reranker_url.clone(),
        cfg.inference.reranker_sha256.clone(),
    );
    let embedder = build_embedder(cfg, &provisioner, &tx, cancel.child_token()).await?;
    let mut engine = SearchEngine::open(&cfg.data_dir, embedder.clone(), DENSE_DIM)?;
    if let Some(rk) = build_reranker(cfg, &reranker_provisioner, &tx, cancel.child_token()).await {
        Arc::get_mut(&mut engine)
            .expect("engine not yet shared")
            .set_reranker(rk);
    }
    let monitor = ResourceMonitor::sysinfo(cfg.resources.clone());
    let pool = InferencePool::new(
        embedder,
        cfg.inference.workers,
        cfg.inference.batch_size,
        monitor.controller(),
    );
    let pipeline = IngestPipeline::new(
        engine.clone(),
        pool.clone(),
        monitor.controller(),
        &cfg.crawler,
        tx.clone(),
    );
    Ok(Runtime {
        engine,
        monitor,
        pipeline,
        pool,
        events: tx,
        provisioner,
        reranker_provisioner,
        cancel,
    })
}

/// `ragnarok serve`: full daemon lifecycle.
async fn serve(cfg: RagnarokConfig, config_path: &std::path::Path) -> Result<()> {
    let rt = bootstrap(&cfg).await?;

    // telemetry sampling + backpressure events
    {
        let m = rt.monitor.clone();
        let tx = rt.events.clone();
        let c = rt.cancel.clone();
        tokio::spawn(async move { m.run(Some(tx), c).await });
    }

    // configured + previously-registered sources
    let mut sources = cfg.sources.clone();
    for s in rt.engine.meta.list_sources()? {
        if !sources.iter().any(|x| x.id == s.id) {
            sources.push(s);
        }
    }
    let handles = rt
        .pipeline
        .spawn_all(sources.clone(), rt.cancel.clone())
        .await;
    tracing::info!(n = sources.len(), "sources started");

    let state = AppState::new(
        rt.engine.clone(),
        rt.monitor.clone(),
        rt.pipeline.clone(),
        rt.events.clone(),
        rt.cancel.clone(),
        Duration::from_millis(cfg.server.telemetry_interval_ms.max(50)),
    )
    .with_inference(
        cfg.inference.backend,
        rt.provisioner.clone(),
        rt.reranker_provisioner.clone(),
        cfg.inference.rerank,
        config_path.to_path_buf(),
    );

    // shutdown on ctrl-c / SIGTERM
    let cancel = rt.cancel.clone();
    tokio::spawn(async move {
        shutdown_signal().await;
        tracing::info!("shutdown signal received — draining");
        cancel.cancel();
    });

    let res = ragnarok_server::serve(state, &cfg.server.bind, rt.cancel.clone()).await;

    // graceful drain: release anything parked on the resource gate FIRST —
    // a hard-pressure plateau would otherwise hold ingest/embedding tasks
    // (and their job callers) forever — then stop the pool, let source
    // tasks finish their in-flight writes, and only THEN snapshot so the
    // persisted vector index includes every last write
    rt.monitor.controller().close();
    rt.pool.shutdown().await;
    let _ = tokio::time::timeout(Duration::from_secs(5), async {
        for h in handles {
            let _ = h.await;
        }
    })
    .await;
    let _ = rt.engine.snapshot();
    let _ = config_path; // (kept for future hot-reload)
    res?;
    Ok(())
}

/// `ragnarok index <path>`: synchronous one-shot ingest.
async fn index(path: PathBuf, name: String, watch: bool, cfg: RagnarokConfig) -> Result<()> {
    if !path.is_dir() {
        anyhow::bail!("not a directory: {}", path.display());
    }
    let rt = bootstrap(&cfg).await?;
    let source = Source::new(
        name,
        SourceConfig::FileSystem {
            path: path.canonicalize()?,
            recursive: true,
            include: vec![],
            exclude: vec![],
            watch,
        },
    );
    // persist so `sources` and later `serve` runs see it
    rt.engine.meta.upsert_source(&source)?;
    let cancel = rt.cancel.clone();
    let p = rt.pipeline.clone();
    let src = source.clone();
    let task = tokio::spawn(async move { p.run_fs_source(src, cancel).await });

    // progress reporter until scan phase completes
    let p2 = rt.pipeline.clone();
    let sid = source.id.clone();
    let reporter = tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;
            for s in p2.states() {
                if s.source_id == sid {
                    eprint!("\rindexing… {}/{} files   ", s.files_indexed, s.files_total);
                }
            }
        }
    });
    let _ = task.await;
    reporter.abort();
    eprintln!();
    rt.engine.snapshot()?;
    println!("indexed — {}", rt.engine.stats()?.documents);
    Ok(())
}

/// `ragnarok sources`: print registered sources.
async fn sources(cfg: RagnarokConfig) -> Result<()> {
    let rt = bootstrap(&cfg).await?;
    let list = rt.engine.meta.list_sources()?;
    if list.is_empty() && cfg.sources.is_empty() {
        println!("no sources registered — use the web UI or add one to config.toml");
        return Ok(());
    }
    for s in list.iter().chain(cfg.sources.iter()) {
        let kind = match &s.config {
            SourceConfig::FileSystem { path, .. } => format!("fs  {}", path.display()),
            SourceConfig::Imap { host, mailbox, .. } => format!("imap {host}/{mailbox}"),
            SourceConfig::Paperless { url, .. } => format!("paperless {url}"),
            SourceConfig::WebDav { url, .. } => format!("webdav {url}"),
            SourceConfig::Web { urls, .. } => format!("web {} urls", urls.len()),
            SourceConfig::Feed { url, .. } => format!("feed {url}"),
            SourceConfig::S3 {
                endpoint, bucket, ..
            } => format!("s3 {endpoint}/{bucket}"),
            SourceConfig::Git { api_base, repo, .. } => format!("git {api_base} {repo}"),
            SourceConfig::CalDav { url, kind, .. } => format!("{kind} {url}"),
        };
        println!("{:<12} {:<24} {}", s.id, s.name, kind);
    }
    Ok(())
}

/// `ragnarok search <query>`: CLI hybrid search.
async fn search(query: String, limit: usize, cfg: RagnarokConfig) -> Result<()> {
    let rt = bootstrap(&cfg).await?;
    let mut req = SearchRequest::new(query);
    req.limit = limit;
    let hits = rt.engine.search(&req).await?;
    if hits.is_empty() {
        println!("no hits");
        return Ok(());
    }
    for h in &hits {
        let geo = match (&h.bbox, &h.line_range) {
            (Some(b), _) => format!(" p{} [{:.2},{:.2}]", h.page + 1, b.x0, b.y0),
            (_, Some((a, b))) => format!(" lines {}–{}", a + 1, b),
            _ => String::new(),
        };
        println!(
            "{:>6.3}  {}{}\n        {}",
            h.score,
            h.path,
            geo,
            h.text
                .chars()
                .take(160)
                .collect::<String>()
                .replace('\n', " ")
        );
    }
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        use tokio::signal::unix::{signal, SignalKind};
        if let Ok(mut s) = signal(SignalKind::terminate()) {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
}
