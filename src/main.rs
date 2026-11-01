//! RAGnarök daemon entrypoint — CLI, lifecycle manager, graceful shutdown.
//!
//! ```text
//! ragnarok serve [--config FILE]          run the daemon (default)
//! ragnarok index <path> [--name N]        one-shot: index a directory and exit
//! ragnarok digest <path> [--kind K]       one-shot: digest any directory
//!                                         (auto-detects maildir vs plain fs)
//! ragnarok sources                        list configured sources
//! ragnarok search <query> [-n LIMIT]      run a hybrid search and print hits
//! ragnarok passwd <user> [--password PW]  reset a user password (offline)
//! ```

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use ragnarok_core::{
    default_config_path, InferenceBackend, RagnarokConfig, SearchRequest, Source, SourceConfig,
};
use ragnarok_crawler::IngestPipeline;
use ragnarok_inference::{Embedder, HashingEmbedder, InferencePool};
use ragnarok_resource::ResourceMonitor;
use ragnarok_server::{self, AppState};
use ragnarok_store::SearchEngine;
use std::path::{Path, PathBuf};
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
    /// Override the HTTP bind address (overrides [server] bind)
    #[arg(long, global = true)]
    bind: Option<String>,
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
    /// Digest a directory once and exit — auto-detects Maildir layout
    /// (cur/new subdirs) and ingests it as a maildir source, otherwise
    /// falls back to a plain filesystem scan. Use --kind to override.
    Digest {
        path: PathBuf,
        /// Source kind override (default: auto-detect)
        #[arg(long, value_enum)]
        kind: Option<DigestKind>,
        #[arg(long, default_value = "cli-digest")]
        name: String,
        /// Keep polling (maildir) or watching (fs) after the first pass
        #[arg(long)]
        follow: bool,
    },
    /// Benchmark embedding models on synthetic email-sized texts.
    /// Generates made-up mails in representative size classes, runs them
    /// through the real preprocess→chunk→embed_batch pipeline, and
    /// reports per-model throughput. Tests every model downloaded under
    /// data_dir/models/ (or --models subset); --download fetches missing
    /// ones first.
    Benchmark {
        /// Comma-separated model ids to test (default: all downloaded)
        #[arg(long, value_delimiter = ',')]
        models: Vec<String>,
        /// Provision models that aren't downloaded yet
        #[arg(long)]
        download: bool,
        /// Execution provider (default: auto — each model runs on the
        /// largest Vulkan GPU when its graph fits the device-local
        /// budget, else on CPU; pass a name to force one)
        #[arg(long, default_value = "auto")]
        provider: String,
        /// Synthetic mails per size class
        #[arg(long, default_value = "6")]
        mails: usize,
        /// Minimum benchmark seconds per model (corpus loops until then)
        #[arg(long, default_value = "12")]
        secs: u64,
        /// Texts per embed_batch call (default: inference.batch_size)
        #[arg(long)]
        batch_size: Option<usize>,
    },
    /// List registered sources
    Sources,
    /// Run a hybrid search against the local index
    Search {
        query: String,
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,
    },
    /// MCP stdio proxy: newline-delimited JSON-RPC → POST /api/mcp.
    /// Env: RAGNAROK_API (default http://127.0.0.1:7901), RAGNAROK_TOKEN
    /// (session token or ragk_… API key).
    Mcp,
    /// Reset a user's password — works offline against the data dir,
    /// no running daemon needed. All sessions of the user are
    /// invalidated. Password order: --password flag, then
    /// RAGNAROK_NEW_PASSWORD env, else a random one is generated
    /// and printed once.
    Passwd {
        username: String,
        #[arg(long)]
        password: Option<String>,
    },
}

/// `--kind` for `ragnarok digest`: pin the source type instead of the
/// auto-detection (Maildir layout → maildir, anything else → fs).
#[derive(ValueEnum, Clone, Copy, Debug)]
enum DigestKind {
    /// Maildir when the dir has `new`/`cur`/`tmp` subdirs, else fs scan
    Auto,
    /// Force the maildir source even without the marker subdirs
    Maildir,
    /// Force a plain filesystem scan
    Fs,
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
    if let Some(b) = cli.bind {
        cfg.server.bind = b;
    }
    cfg.validate()?;
    std::fs::create_dir_all(&cfg.data_dir)?;

    match cli.command.unwrap_or(Command::Serve) {
        Command::Serve => serve(cfg, &config_path).await,
        Command::Index { path, name, watch } => index(path, name, watch, cfg).await,
        Command::Digest {
            path,
            kind,
            name,
            follow,
        } => digest(path, kind, name, follow, cfg).await,
        Command::Benchmark {
            models,
            download,
            provider,
            mails,
            secs,
            batch_size,
        } => benchmark(models, download, provider, mails, secs, batch_size, cfg).await,
        Command::Sources => sources(cfg).await,
        Command::Search { query, limit } => search(query, limit, cfg).await,
        Command::Mcp => mcp_stdio().await,
        Command::Passwd { username, password } => passwd(&username, password, &cfg),
    }
}

/// `ragnarok mcp`: newline-delimited JSON-RPC on stdin → POST /api/mcp
/// → response lines on stdout. Lets MCP clients (Claude Desktop, Cursor)
/// reach the daemon's tool surface without holding a session.
async fn mcp_stdio() -> Result<()> {
    let api = std::env::var("RAGNAROK_API")
        .unwrap_or_else(|_| "http://127.0.0.1:7901".into())
        .trim_end_matches('/')
        .to_owned();
    let token = std::env::var("RAGNAROK_TOKEN")
        .map_err(|_| anyhow::anyhow!("RAGNAROK_TOKEN env var required (session or ragk_ key)"))?;
    let client = reqwest::Client::new();
    let url = format!("{api}/api/mcp");
    use std::io::BufRead;
    let mut stdin = std::io::BufReader::new(std::io::stdin());
    let mut line = String::new();
    loop {
        line.clear();
        if stdin.read_line(&mut line)? == 0 {
            return Ok(()); // EOF
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let resp = client
            .post(&url)
            .bearer_auth(&token)
            .header("content-type", "application/json")
            .body(trimmed.to_owned())
            .send()
            .await?;
        if resp.status() == reqwest::StatusCode::ACCEPTED {
            continue; // notification-only → no output
        }
        let text = resp.text().await?;
        println!("{text}");
        use std::io::Write;
        std::io::stdout().flush()?;
    }
}

/// `inference.backend = "auto"` (the default) resolves to `onnx` when the
/// feature is compiled and the model is already provisioned — otherwise
/// `hashing`. `auto` never downloads on its own: the model arrives via
/// `POST /inference/model/download` (setup wizard / settings), and the
/// next restart then silently upgrades to real semantics.
fn resolve_backend(
    cfg: &RagnarokConfig,
    provisioner: &ragnarok_inference::Provisioner,
) -> InferenceBackend {
    match cfg.inference.backend {
        InferenceBackend::Auto => {
            if ragnarok_inference::ONNX_COMPILED
                && ragnarok_inference::Provisioner::ready_in(provisioner.dir())
            {
                InferenceBackend::Onnx
            } else {
                InferenceBackend::Hashing
            }
        }
        b => b,
    }
}

/// Build the embedder for the resolved backend — provisioning the ONNX
/// model first when it is missing and `auto_provision` allows.
async fn build_embedder(
    backend: InferenceBackend,
    cfg: &RagnarokConfig,
    spec: &'static ragnarok_inference::ModelSpec,
    provisioner: &Arc<ragnarok_inference::Provisioner>,
    events: &broadcast::Sender<ragnarok_core::TelemetryEvent>,
    cancel: CancellationToken,
) -> Result<Arc<dyn Embedder>> {
    match backend {
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
            // `auto` honors a remembered GPU death: a device that lost its
            // driver once (OOM → VK_ERROR_DEVICE_LOST) would just crash
            // again — start on CPU instead. Explicit provider names bypass
            // the marker: the user asked for that provider deliberately.
            let mut provider = cfg.inference.provider.clone();
            let mut remembered_death: Option<String> = None;
            if provider.eq_ignore_ascii_case("auto") {
                if let Ok(meta) =
                    ragnarok_store::MetadataDb::open(&cfg.data_dir.join("ragnarok.db"))
                {
                    if let Ok(Some(reason)) = meta.get_setting("gpu_dead") {
                        tracing::warn!(
                            reason = %reason,
                            "GPU provider died in a previous run — starting on CPU \
                             (clear the 'gpu_dead' setting or set inference.provider to retry)"
                        );
                        provider = "cpu".into();
                        remembered_death = Some(reason);
                    }
                }
                // fit check: even a living GPU can't run a model whose
                // graph + workspace exceeds its device-local heap —
                // skip it before the first forward OOMs the driver
                if remembered_death.is_none() {
                    let gpu = ragnarok_inference::gpu_probe::device_local_budget();
                    let bytes =
                        ragnarok_inference::onnx::model_bytes(provisioner.dir()).unwrap_or(0);
                    if let Some(why) = gpu_fit_skip_reason(gpu.as_ref(), bytes) {
                        tracing::warn!(reason = %why, "starting on CPU \
                             (set inference.provider to force the GPU anyway)");
                        provider = "cpu".into();
                    }
                }
            }
            let emb = ragnarok_inference::onnx::OnnxEmbedder::load(
                provisioner.dir(),
                spec,
                cfg.inference.max_colbert_tokens,
                &provider,
                cfg.inference.workers,
            )?;
            if let Some(reason) = remembered_death {
                emb.note_gpu_dead(reason);
            }
            // canary: validates the whole path (tokenizer, EP, output
            // names) at boot instead of letting the first real doc find
            // out — and flips a dead-on-arrival GPU to CPU right away
            if let Err(e) = emb.embed("ragnarok warmup").await {
                tracing::warn!("onnx warmup embed failed: {e}");
            }
            Ok(emb as Arc<dyn Embedder>)
        }
        #[cfg(not(feature = "onnx"))]
        InferenceBackend::Onnx => {
            let _ = (spec, provisioner, events, cancel); // used only by the onnx arm
            anyhow::bail!(
                "binary built without `onnx` feature — rebuild with --features onnx \
                 or set inference.backend = \"hashing\""
            )
        }
        InferenceBackend::Remote => {
            // key precedence: config → env var → server-side runtime
            // setting (the UI writes `remote_embed_key`; it never
            // appears in GET responses or config file dumps)
            let key = cfg
                .inference
                .remote_api_key
                .clone()
                .filter(|k| !k.trim().is_empty())
                .or_else(|| {
                    let var = cfg
                        .inference
                        .remote_api_key_env
                        .clone()
                        .unwrap_or_else(|| "OPENAI_API_KEY".into());
                    std::env::var(&var).ok().filter(|v| !v.is_empty())
                })
                .or_else(|| {
                    ragnarok_store::MetadataDb::open(&cfg.data_dir.join("ragnarok.db"))
                        .ok()
                        .and_then(|m| m.get_setting("remote_embed_key").ok().flatten())
                        .filter(|v| !v.is_empty())
                });
            let url = cfg
                .inference
                .remote_url
                .clone()
                .unwrap_or_else(|| "https://api.openai.com/v1".into());
            let emb = ragnarok_inference::RemoteEmbedder::connect(
                &url,
                cfg.inference.remote_model.as_deref().unwrap_or(""),
                key,
                cfg.inference.remote_dimensions,
                cfg.inference.remote_query_prefix.as_deref(),
                cfg.inference.remote_document_prefix.as_deref(),
                cfg.inference.max_colbert_tokens,
            )
            .await?;
            Ok(Arc::new(emb) as Arc<dyn Embedder>)
        }
        InferenceBackend::Auto => unreachable!("auto is resolved by resolve_backend"),
    }
}

/// Reason a model should skip the GPU, or `None` when it fits / no GPU
/// was detected (callers then keep their configured provider — ORT's
/// own fail-silently chain still lands on CPU).
#[cfg(feature = "onnx")]
fn gpu_fit_skip_reason(
    gpu: Option<&ragnarok_inference::gpu_probe::GpuMemory>,
    model_bytes: u64,
) -> Option<String> {
    let g = gpu?;
    if ragnarok_inference::gpu_probe::model_fits(model_bytes, gpu) == Some(false) {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        Some(format!(
            "model {:.2} GiB + workspace exceeds Vulkan budget {:.2} GiB on '{}'",
            gib(model_bytes),
            gib(g.budget),
            g.device
        ))
    } else {
        None
    }
}

/// Provision + load the cross-encoder reranker when the `onnx` feature is
/// compiled and `inference.rerank` is on. Never fatal — a missing/failed
/// reranker just drops the stage (hybrid order is kept).
async fn build_reranker(
    backend: InferenceBackend,
    cfg: &RagnarokConfig,
    provisioner: &Arc<ragnarok_inference::Provisioner>,
    events: &broadcast::Sender<ragnarok_core::TelemetryEvent>,
    provider: &str,
    cancel: CancellationToken,
) -> Option<Arc<dyn ragnarok_inference::Reranker>> {
    #[cfg(feature = "onnx")]
    {
        if !cfg.inference.rerank || backend != InferenceBackend::Onnx {
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
        match ragnarok_inference::rerank::OnnxReranker::load(provisioner.dir(), provider) {
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
        let _ = (backend, cfg, provisioner, events, cancel);
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
    /// The resolved (`auto` → concrete) embedding backend.
    backend: InferenceBackend,
    cancel: CancellationToken,
}

async fn bootstrap(cfg: &RagnarokConfig) -> Result<Runtime> {
    let (tx, _rx) = broadcast::channel(512);
    let cancel = CancellationToken::new();
    let spec = ragnarok_inference::model_spec(&cfg.inference.model).ok_or_else(|| {
        let known: Vec<&str> = ragnarok_inference::model_registry()
            .iter()
            .map(|s| s.id)
            .collect();
        anyhow::anyhow!(
            "unknown inference.model '{}' — registered: {}",
            cfg.inference.model,
            known.join(", ")
        )
    })?;
    let model_dir = cfg
        .inference
        .model_dir
        .clone()
        .unwrap_or_else(|| cfg.data_dir.join("models").join(spec.id));
    let provisioner = ragnarok_inference::Provisioner::for_model(
        spec,
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
    let backend = resolve_backend(cfg, &provisioner);
    let embedder =
        build_embedder(backend, cfg, spec, &provisioner, &tx, cancel.child_token()).await?;
    let mut engine = SearchEngine::open(&cfg.data_dir, embedder.clone(), embedder.dense_dim())?;
    // a canary-time GPU death is already settled — persist it now instead
    // of waiting for the first ingested doc
    if let Some(reason) = embedder.gpu_dead() {
        let _ = engine.meta.set_setting("gpu_dead", &reason);
    }
    // the reranker follows the embedder's *effective* provider — after a
    // GPU death (persisted marker or explicit cpu config) it must not
    // open a session on the same dead device
    let rerank_provider = match embedder.provider_label().as_deref() {
        Some("cpu") => "cpu",
        _ => cfg.inference.provider.as_str(),
    };
    if let Some(rk) = build_reranker(
        backend,
        cfg,
        &reranker_provisioner,
        &tx,
        rerank_provider,
        cancel.child_token(),
    )
    .await
    {
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
    // persisted runtime overrides (PUT /settings/runtime) beat config.toml
    if let Ok(Some(json)) = engine.meta.get_setting("runtime") {
        if let Ok(rt) = serde_json::from_str::<ragnarok_core::RuntimeConfig>(&json) {
            if rt.validate().is_ok() {
                pipeline.apply_runtime(&rt);
            }
        }
    }
    Ok(Runtime {
        engine,
        monitor,
        pipeline,
        pool,
        events: tx,
        provisioner,
        reranker_provisioner,
        backend,
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
    .with_inference(ragnarok_server::InferenceCtx {
        backend: rt.backend,
        provisioner: rt.provisioner.clone(),
        reranker_provisioner: rt.reranker_provisioner.clone(),
        rerank: cfg.inference.rerank,
        config_path: config_path.to_path_buf(),
        model_id: cfg.inference.model.clone(),
        models_dir: cfg.data_dir.join("models"),
        model_url: cfg.inference.model_url.clone(),
        model_sha256: cfg.inference.model_sha256.clone(),
    });

    // the persisted runtime blob carries the rate limit too — the
    // pipeline half was applied in bootstrap, the limiter lives on state
    if let Ok(Some(json)) = state.engine.meta.get_setting("runtime") {
        if let Ok(rt) = serde_json::from_str::<ragnarok_core::RuntimeConfig>(&json) {
            state.rate_limiter.set_rate(rt.rate_limit_per_min);
        }
    }

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

/// Reuse a registered source targeting the same name + path — `Source::new`
/// mints a random id, so a fresh one every run would re-ingest everything
/// under a new source id and leave the old runs' docs orphaned (dedup and
/// the delete sweep are per-source).
fn matching_source(
    meta: &ragnarok_store::MetadataDb,
    name: &str,
    path: &Path,
) -> Result<Option<Source>> {
    for s in meta.list_sources()? {
        if s.name != name {
            continue;
        }
        let p = match &s.config {
            SourceConfig::FileSystem { path: p, .. } | SourceConfig::Maildir { path: p, .. } => p,
            _ => continue,
        };
        if p == path {
            return Ok(Some(s));
        }
    }
    Ok(None)
}

/// `ragnarok index <path>`: synchronous one-shot ingest.
async fn index(path: PathBuf, name: String, watch: bool, cfg: RagnarokConfig) -> Result<()> {
    if !path.is_dir() {
        anyhow::bail!("not a directory: {}", path.display());
    }
    let rt = bootstrap(&cfg).await?;
    let canon = path.canonicalize()?;
    let source = match matching_source(&rt.engine.meta, &name, &canon)? {
        Some(mut s) => {
            // the stored source takes this run's watch intent
            if let SourceConfig::FileSystem { watch: w, .. } = &mut s.config {
                *w = watch;
            }
            s
        }
        None => Source::new(
            name,
            SourceConfig::FileSystem {
                path: canon,
                recursive: true,
                include: vec![],
                exclude: vec![],
                watch,
            },
        ),
    };
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

/// `ragnarok digest <path>`: one-shot ingest for any local directory.
/// Maildir layout (`new`/`cur`/`tmp` subdirs) goes through the remote
/// maildir cycle — messages stay in place, `maildir://` vpaths, flag
/// renames don't re-embed — anything else is a plain fs scan.
async fn digest(
    path: PathBuf,
    kind: Option<DigestKind>,
    name: String,
    follow: bool,
    cfg: RagnarokConfig,
) -> Result<()> {
    if !path.is_dir() {
        anyhow::bail!("not a directory: {}", path.display());
    }
    let looks_maildir =
        path.join("new").is_dir() || path.join("cur").is_dir() || path.join("tmp").is_dir();
    let use_maildir = match kind.unwrap_or(DigestKind::Auto) {
        DigestKind::Auto => looks_maildir,
        DigestKind::Maildir => true,
        DigestKind::Fs => false,
    };
    if !use_maildir {
        tracing::info!(path = %path.display(), "digesting as filesystem source");
        return index(path, name, follow, cfg).await;
    }

    let rt = bootstrap(&cfg).await?;
    let canon = path.canonicalize()?;
    let source = match matching_source(&rt.engine.meta, &name, &canon)? {
        Some(s) => s,
        None => Source::new(
            name,
            SourceConfig::Maildir {
                path: canon,
                include_subfolders: true,
                poll_interval_secs: 300,
            },
        ),
    };
    rt.engine.meta.upsert_source(&source)?;

    // progress reporter while the cycle runs
    let p = rt.pipeline.clone();
    let sid = source.id.clone();
    let reporter = tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;
            for s in p.states() {
                if s.source_id == sid {
                    eprint!(
                        "\rdigesting… {}/{} messages   ",
                        s.files_indexed, s.files_total
                    );
                }
            }
        }
    });

    if follow {
        let cancel = rt.cancel.clone();
        tokio::spawn(async move {
            shutdown_signal().await;
            cancel.cancel();
        });
        rt.pipeline
            .run_remote_source(source.clone(), rt.cancel.clone())
            .await;
        reporter.abort();
        eprintln!();
        rt.engine.snapshot()?;
        return Ok(());
    }

    let res = rt.pipeline.remote_cycle(&source, &rt.cancel).await;
    reporter.abort();
    eprintln!();
    let fresh = res?;
    let (total, in_index, last_error) = rt
        .pipeline
        .states()
        .into_iter()
        .find(|s| s.source_id == source.id)
        .map(|s| (s.files_total, s.files_indexed, s.last_error))
        .unwrap_or((0, 0, None));
    rt.engine.snapshot()?;
    let failed = total.saturating_sub(in_index);
    println!(
        "digested {fresh} new/changed — {in_index}/{total} messages in index ({failed} failed) — {} docs total",
        rt.engine.stats()?.documents
    );
    if let Some(e) = last_error {
        println!("last error: {e}");
    }
    Ok(())
}

/// `ragnarok benchmark`: per-model embed_batch throughput on synthetic
/// email-sized texts — the honest number is chunks/min through the real
/// preprocess→chunk→batch pipeline, not raw forward time.
async fn benchmark(
    filter: Vec<String>,
    download: bool,
    provider: String,
    mails_per_class: usize,
    min_secs: u64,
    batch_size: Option<usize>,
    cfg: RagnarokConfig,
) -> Result<()> {
    for f in &filter {
        if f != "hashing" && ragnarok_inference::model_spec(f).is_none() {
            anyhow::bail!(
                "unknown model '{f}' — known: hashing, {}",
                ragnarok_inference::model_registry()
                    .iter()
                    .map(|s| s.id)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }

    let corpus = synth_corpus(mails_per_class);
    let chunk_cfg = ragnarok_crawler::ChunkerConfig::default();
    let mut chunks: Vec<String> = Vec::new();
    for (i, (class, text)) in corpus.iter().enumerate() {
        let mut doc = ragnarok_core::ParsedDocument::new(
            format!("bench://{class}/{i}"),
            ragnarok_core::DocType::Email,
        );
        doc.blocks.push(ragnarok_core::TextBlock::plain(
            ragnarok_crawler::preprocess::clean(text),
        ));
        for c in ragnarok_crawler::chunker::chunk_document(
            ragnarok_core::DocumentId::new(),
            &doc,
            chunk_cfg,
        )? {
            chunks.push(c.text);
        }
    }
    let classes: Vec<String> = corpus
        .iter()
        .map(|(c, _)| c.to_string())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let bs = batch_size.unwrap_or(cfg.inference.batch_size).max(1);
    println!(
        "corpus: {} synthetic mails ({}) → {} chunks — batch_size={bs}, provider={provider}",
        corpus.len(),
        classes.join(", "),
        chunks.len(),
    );

    // deterministic hashing embedder as a baseline — no model files
    // needed, shows what the ONNX stage actually costs
    let hashing: Arc<dyn Embedder> = Arc::new(HashingEmbedder::default());
    if filter.is_empty() || filter.iter().any(|f| f == "hashing") {
        let (done, secs, err) = bench_one(&*hashing, &chunks, bs, min_secs).await;
        print_report(&Report {
            id: "hashing".into(),
            info: format!(
                "{}d — deterministic hashing baseline, no model files",
                hashing.dense_dim()
            ),
            provider: "—".into(),
            provider_note: None,
            files: None,
            outcome: Outcome::Timed {
                chunks: done,
                secs,
                err,
            },
        });
    }

    #[cfg(feature = "onnx")]
    {
        // hardware context for `auto`: the biggest Vulkan heap decides
        // which models are even allowed to try the GPU; a persisted
        // death marker vetoes all of them
        let gpu = ragnarok_inference::gpu_probe::device_local_budget();
        let mut dead_reason = ragnarok_store::MetadataDb::open(&cfg.data_dir.join("ragnarok.db"))
            .ok()
            .and_then(|m| m.get_setting("gpu_dead").ok().flatten());
        match &gpu {
            Some(g) => eprintln!(
                "GPU probe: '{}' ({}) — device-local budget {:.2} GiB{}",
                g.device,
                if g.discrete { "dGPU" } else { "integrated" },
                g.budget as f64 / (1u64 << 30) as f64,
                dead_reason
                    .as_ref()
                    .map(|r| format!(" (marked dead: {r})"))
                    .unwrap_or_default(),
            ),
            None => eprintln!("GPU probe: no usable Vulkan device — all models on CPU"),
        }

        let (tx, _rx) = broadcast::channel(8);
        for spec in ragnarok_inference::model_registry()
            .iter()
            .filter(|s| filter.is_empty() || filter.iter().any(|f| f == s.id))
        {
            // inference.model_dir overrides only the configured model's
            // dir; every other model lives at data_dir/models/{id}
            let dir = if spec.id == cfg.inference.model && cfg.inference.model_dir.is_some() {
                cfg.inference.model_dir.clone().unwrap()
            } else {
                cfg.data_dir.join("models").join(spec.id)
            };
            let mut report = Report {
                id: spec.id.into(),
                info: format!(
                    "{}d · {} — {}",
                    spec.dense_dim, spec.languages, spec.description
                ),
                provider: String::new(),
                provider_note: None,
                files: ragnarok_inference::onnx::model_bytes(&dir).map(|b| {
                    format!(
                        "{:.0} MiB graph — {}",
                        b as f64 / (1u64 << 20) as f64,
                        dir.display()
                    )
                }),
                outcome: Outcome::Skipped("not benchmarked".into()),
            };
            if !ragnarok_inference::Provisioner::ready_in(&dir) {
                if !download {
                    report.outcome = Outcome::Skipped(
                        "not downloaded — rerun with --download, or fetch it via \
                         /settings → Embedding"
                            .into(),
                    );
                    print_report(&report);
                    continue;
                }
                let prov = ragnarok_inference::Provisioner::for_model(
                    spec,
                    dir.clone(),
                    cfg.inference.model_url.clone(),
                    cfg.inference.model_sha256.clone(),
                );
                eprintln!("downloading {} → {} …", spec.id, dir.display());
                if let Err(e) = prov.download(Some(&tx), CancellationToken::new()).await {
                    report.outcome = Outcome::Failed(format!("download: {e}"));
                    print_report(&report);
                    continue;
                }
                report.files = ragnarok_inference::onnx::model_bytes(&dir).map(|b| {
                    format!(
                        "{:.0} MiB graph — {}",
                        b as f64 / (1u64 << 20) as f64,
                        dir.display()
                    )
                });
            }
            // per-model provider: `auto` runs on the GPU only when the
            // graph fits its device-local budget and no death marker is
            // set; explicit provider names are honored verbatim
            let bytes = ragnarok_inference::onnx::model_bytes(&dir).unwrap_or(0);
            let (eff_provider, note) = if !provider.eq_ignore_ascii_case("auto") {
                (
                    provider.clone(),
                    format!("forced via --provider {provider}"),
                )
            } else if let Some(r) = dead_reason.as_ref() {
                ("cpu".to_string(), format!("GPU marked dead — {r}"))
            } else if let Some(why) = gpu_fit_skip_reason(gpu.as_ref(), bytes) {
                ("cpu".to_string(), why)
            } else if let Some(g) = gpu.as_ref() {
                (
                    "auto".to_string(),
                    format!(
                        "{:.1} GiB graph + workspace fits {:.2} GiB budget on '{}'",
                        bytes as f64 / (1u64 << 30) as f64,
                        g.budget as f64 / (1u64 << 30) as f64,
                        g.device
                    ),
                )
            } else {
                (
                    "auto".to_string(),
                    "no usable Vulkan device — ORT will land on CPU".to_string(),
                )
            };
            report.provider_note = Some(note);
            let emb = match ragnarok_inference::onnx::OnnxEmbedder::load(
                &dir,
                spec,
                cfg.inference.max_colbert_tokens,
                &eff_provider,
                1, // single session → all intra-op threads, no pool contention
            ) {
                Ok(e) => e,
                Err(e) => {
                    report.provider = short_provider(&eff_provider).into();
                    report.outcome = Outcome::Failed(format!("load: {e}"));
                    print_report(&report);
                    continue;
                }
            };
            report.provider = short_provider(&emb.provider_label()).into();
            // warmup: graph init + first tokenizations
            let _ = emb.embed_batch(&chunks[..chunks.len().min(bs)]).await;
            let (done, secs, err) = bench_one(&*emb, &chunks, bs, min_secs).await;
            // a GPU that died this run stays dead for the remaining
            // models — no point crashing into it repeatedly; the report
            // also explains the flip
            if let Some(r) = emb.gpu_dead() {
                dead_reason.get_or_insert(r.clone());
                report.provider_note = Some(format!("GPU died mid-run ({r}) → CPU fallback"));
            }
            report.outcome = Outcome::Timed {
                chunks: done,
                secs,
                err,
            };
            print_report(&report);
            drop(emb); // free session memory before the next model loads
        }
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (download, provider);
        println!("\n(no onnx feature — only the hashing baseline ran)");
    }
    Ok(())
}

/// One benchmark run: warmup already done, loop the corpus in batches
/// until `min_secs` wall time elapsed. Returns (chunks done, seconds,
/// first error when the run ended early).
async fn bench_one(
    emb: &dyn Embedder,
    chunks: &[String],
    batch_size: usize,
    min_secs: u64,
) -> (usize, f64, Option<String>) {
    let start = std::time::Instant::now();
    let mut done = 0usize;
    while start.elapsed().as_secs() < min_secs || done == 0 {
        for batch in chunks.chunks(batch_size) {
            if let Err(e) = emb.embed_batch(batch).await {
                return (done, start.elapsed().as_secs_f64(), Some(e.to_string()));
            }
            done += batch.len();
            if start.elapsed().as_secs() >= min_secs && done > 0 {
                break;
            }
        }
    }
    (done, start.elapsed().as_secs_f64().max(1e-6), None)
}

/// ORT reports full EP type names — map them to readable labels.
fn short_provider(label: &str) -> &str {
    match label {
        "WebGpuExecutionProvider" => "webgpu",
        "CPUExecutionProvider" => "cpu",
        "CoreMLExecutionProvider" => "coreml",
        "CUDAExecutionProvider" => "cuda",
        "auto" => "auto (gpu-first)",
        other => other,
    }
}

/// Per-model benchmark outcome — gathered while the model runs, then
/// printed as one paragraph so the *why* sits next to the number.
struct Report {
    id: String,
    /// `384d · multilingual — description` after the model id.
    info: String,
    /// Execution provider that actually ran ("webgpu"/"cpu"/"—");
    /// empty when no provider was ever chosen (download skipped).
    provider: String,
    /// Why that provider — fit check, death marker, forced flag, …
    provider_note: Option<String>,
    /// Downloaded graph size + location, when on disk.
    files: Option<String>,
    outcome: Outcome,
}

enum Outcome {
    Timed {
        chunks: usize,
        secs: f64,
        err: Option<String>,
    },
    Skipped(String),
    Failed(String),
}

fn print_report(r: &Report) {
    println!("\n{} ({})", r.id, r.info);
    if let Some(f) = &r.files {
        println!("  model: {f}");
    }
    if !r.provider.is_empty() {
        match &r.provider_note {
            Some(n) => println!("  provider: {} — {n}", r.provider),
            None => println!("  provider: {}", r.provider),
        }
    }
    match &r.outcome {
        Outcome::Timed { chunks, secs, err } => {
            if *chunks == 0 {
                println!(
                    "  failed: {}",
                    err.as_deref().unwrap_or("no chunks embedded")
                );
                return;
            }
            let per_ms = secs * 1000.0 / *chunks as f64;
            let per_min = *chunks as f64 / secs * 60.0;
            let proj_h = 250_000.0 / per_min.max(1e-6) / 60.0;
            let proj = if proj_h >= 48.0 {
                format!("{:.1} d", proj_h / 24.0)
            } else {
                format!("{proj_h:.1} h")
            };
            println!(
                "  result: {:.0} chunks/min · {per_ms:.1} ms/chunk \
                 · {chunks} chunks in {secs:.1} s — ~250k chunks ≈ {proj}",
                per_min
            );
            if let Some(e) = err {
                println!("  warning: run ended early — {e}");
            }
        }
        Outcome::Skipped(why) => println!("  skipped: {why}"),
        Outcome::Failed(e) => println!("  failed: {e}"),
    }
}

/// Deterministic PRNG (xorshift64*) — the corpus must be identical across
/// runs and models, and adding a `rand` dep for ~15 lines isn't worth it.
struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn pick<'a>(&mut self, xs: &'a [&'a str]) -> &'a str {
        xs[(self.next() as usize) % xs.len()]
    }
    fn range(&mut self, n: usize) -> usize {
        (self.next() as usize) % n
    }
}

/// Synthetic email generator — made-up mails in four size classes that
/// match what a real maildir produces (short replies → long threads),
/// with realistic noise (headers, quotes, URLs, PGP blob) so the
/// preprocess stage and chunker see representative input.
fn synth_corpus(per_class: usize) -> Vec<(&'static str, String)> {
    const CLASSES: &[(&str, usize, usize)] = &[
        ("tiny", 40, 0),    // short reply
        ("short", 200, 0),  // typical mail
        ("medium", 700, 0), // newsletter / long mail
        ("long", 1200, 3),  // thread with quoted replies
    ];
    let mut rng = XorShift(0x9E3779B97F4A7C15);
    let mut out = Vec::new();
    for &(class, body_words, depth) in CLASSES {
        for _ in 0..per_class {
            out.push((class, synth_mail(&mut rng, body_words, depth)));
        }
    }
    out
}

fn synth_mail(rng: &mut XorShift, body_words: usize, quote_depth: usize) -> String {
    const NAMES: &[&str] = &[
        "uli",
        "anna.schmidt",
        "j.mueller",
        "kontakt",
        "billing",
        "ops",
    ];
    const WORDS: &[&str] = &[
        "Hallo",
        "danke",
        "für",
        "die",
        "schnelle",
        "Rückmeldung",
        "ich",
        "habe",
        "den",
        "Server",
        "neu",
        "gestartet",
        "und",
        "das",
        "Deployment",
        "läuft",
        "jetzt",
        "wieder",
        "die",
        "Rechnung",
        "vom",
        "Oktober",
        "ist",
        "angehängt",
        "bitte",
        "um",
        "Prüfung",
        "wegen",
        "der",
        "Migration",
        "müssen",
        "wir",
        "noch",
        "klären",
        "the",
        "backup",
        "job",
        "failed",
        "again",
        "last",
        "night",
        "could",
        "you",
        "please",
        "check",
        "the",
        "cron",
        "schedule",
        "Betreff",
        "Angebot",
        "Zahlungserinnerung",
        "Meeting",
        "morgen",
        "um",
        "14",
        "Uhr",
        "Bericht",
        "Logs",
        "Fehlermeldung",
        "Timeout",
        "Datenbank",
        "Index",
        "Abfrage",
        "langsam",
        "Speicherplatz",
        "Update",
        "Version",
        "TLS",
        "Zertifikat",
        "läuft",
        "ab",
    ];
    let mut s = format!(
        "From: {}@example.com\nTo: {}@example.org\nSubject: {} {}\nDate: Tue, 28 Oct 2026 {}:17:42 +0100\n\n",
        rng.pick(NAMES),
        rng.pick(NAMES),
        rng.pick(WORDS),
        rng.pick(WORDS),
        8 + rng.range(10),
    );
    for i in (0..quote_depth).rev() {
        let mut quoted = String::new();
        for _ in 0..40 + rng.range(120) {
            quoted.push_str(rng.pick(WORDS));
            quoted.push(' ');
            if rng.range(14) == 0 {
                quoted.push_str("https://example.org/ticket/");
                quoted.push_str(&rng.range(99999).to_string());
                quoted.push(' ');
            }
        }
        for line in quoted.split(' ').collect::<Vec<_>>().chunks(12) {
            s.push_str(&">".repeat(i + 1));
            s.push_str(&line.join(" "));
            s.push('\n');
        }
        s.push('\n');
    }
    s.push_str(&format!("Hi {},\n\n", rng.pick(NAMES)));
    for _ in 0..body_words {
        s.push_str(rng.pick(WORDS));
        s.push(' ');
        match rng.range(24) {
            0 => {
                s.push_str("https://example.org/docs/");
                s.push_str(&rng.range(99999).to_string());
                s.push(' ');
            }
            1 => {
                s.push_str(&format!("{}@example.com ", rng.pick(NAMES)));
            }
            2 => s.push_str("\n\n"),
            _ => {}
        }
    }
    if quote_depth == 0 && rng.range(3) == 0 {
        // occasional opaque blob like a real mail carries (PGP/JWT/
        // base64 attachment fragment) — preprocess strips it
        s.push_str("\n-----BEGIN PGP SIGNATURE-----\n");
        for _ in 0..4 {
            for _ in 0..64 {
                let c = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
                    [rng.range(64)];
                s.push(c as char);
            }
            s.push('\n');
        }
        s.push_str("-----END PGP SIGNATURE-----\n");
    }
    s.push_str(&format!("\n\nGrüße\n{}\n", rng.pick(NAMES)));
    s
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
            SourceConfig::Maildir { path, .. } => format!("maildir {}", path.display()),
        };
        println!("{:<12} {:<24} {}", s.id, s.name, kind);
    }
    Ok(())
}

/// `ragnarok passwd <user>`: reset a password directly on the metadata
/// store — no engine/model bootstrap, works while the daemon is stopped.
fn passwd(username: &str, password: Option<String>, cfg: &RagnarokConfig) -> Result<()> {
    let meta = ragnarok_store::MetadataDb::open(&cfg.data_dir.join("ragnarok.db"))?;
    let Some((user, _)) = meta.user_by_username(username)? else {
        let known = meta
            .list_users()?
            .iter()
            .map(|u| u.username.clone())
            .collect::<Vec<_>>()
            .join(", ");
        anyhow::bail!("unknown user '{username}' — existing: {known}");
    };
    let (pw, generated) = match password.or_else(|| std::env::var("RAGNAROK_NEW_PASSWORD").ok()) {
        Some(p) => (p, false),
        None => {
            use rand::RngCore;
            let mut b = [0u8; 12];
            rand::thread_rng().fill_bytes(&mut b);
            (hex::encode(b), true)
        }
    };
    if pw.len() < 4 {
        anyhow::bail!("password too short — at least 4 characters");
    }
    meta.update_user(
        &user.id,
        None,
        None,
        Some(&ragnarok_server::auth::hash_password(&pw)?),
    )?;
    let killed = meta.delete_user_sessions(&user.id)?;
    if generated {
        println!("password for '{username}' reset — new password: {pw}");
    } else {
        println!("password for '{username}' updated");
    }
    if killed > 0 {
        println!("{killed} active session(s) invalidated");
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
