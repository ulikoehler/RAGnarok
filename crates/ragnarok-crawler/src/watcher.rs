//! Real-time filesystem monitoring via `notify` with a debounce window
//! (default 500 ms) so write storms collapse into single re-index events.

use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use parking_lot::Mutex;
use ragnarok_core::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// What happened to a path (post-debounce).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsEventKind {
    /// Created or modified — re-index.
    Upsert,
    /// Removed or renamed away — drop from index.
    Removed,
}

#[derive(Debug, Clone)]
pub struct FsEvent {
    pub path: PathBuf,
    pub kind: FsEventKind,
}

/// Watches one directory tree and forwards debounced events to `tx`.
/// Dropping this struct detaches the OS watcher.
pub struct FsWatcher {
    _watcher: RecommendedWatcher,
    _debouncer: tokio::task::JoinHandle<()>,
}

impl FsWatcher {
    /// Start watching `root`; events are debounced by `window` and emitted as
    /// `FsEvent`s on `tx`. Runs until `cancel` fires.
    pub fn start(
        root: &Path,
        recursive: bool,
        window: Duration,
        tx: mpsc::Sender<FsEvent>,
        cancel: CancellationToken,
    ) -> Result<Self> {
        let (event_tx, event_rx) = std_mpsc::channel::<notify::Result<notify::Event>>();
        let mut watcher = RecommendedWatcher::new(
            move |res| {
                let _ = event_tx.send(res);
            },
            notify::Config::default(),
        )
        .map_err(|e| ragnarok_core::Error::Internal(format!("notify init: {e}")))?;
        let mode = if recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };
        watcher
            .watch(root, mode)
            .map_err(|e| ragnarok_core::Error::Internal(format!("notify watch: {e}")))?;

        // forward notify's std-mpsc events into the debouncer state
        let pending: std::sync::Arc<Mutex<HashMap<PathBuf, (FsEventKind, Instant)>>> =
            std::sync::Arc::new(Mutex::new(HashMap::new()));
        let pending2 = pending.clone();
        std::thread::spawn(move || {
            while let Ok(res) = event_rx.recv() {
                match res {
                    Ok(ev) => {
                        let kind = match ev.kind {
                            EventKind::Create(_) | EventKind::Modify(_) => {
                                Some(FsEventKind::Upsert)
                            }
                            EventKind::Remove(_) => Some(FsEventKind::Removed),
                            _ => None,
                        };
                        if let Some(kind) = kind {
                            let mut p = pending2.lock();
                            for path in ev.paths {
                                if path.extension().is_none() && path.is_dir() {
                                    continue; // directories are not indexed
                                }
                                // Removed wins over stale Upsert entries
                                p.insert(path, (kind, Instant::now()));
                            }
                        }
                    }
                    Err(e) => tracing::warn!(error = %e, "notify error"),
                }
            }
        });

        // debounce pump: flush entries whose last event is older than `window`
        let debouncer = tokio::spawn(async move {
            let tick = (window / 2).max(Duration::from_millis(25));
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    _ = tokio::time::sleep(tick) => {}
                }
                let now = Instant::now();
                let ready: Vec<(PathBuf, FsEventKind)> = {
                    let mut p = pending.lock();
                    let keys: Vec<PathBuf> = p
                        .iter()
                        .filter(|(_, (_, t))| now.duration_since(*t) >= window)
                        .map(|(k, _)| k.clone())
                        .collect();
                    keys.into_iter()
                        .filter_map(|k| p.remove(&k).map(|(kind, _)| (k, kind)))
                        .collect()
                };
                for (path, kind) in ready {
                    if tx.send(FsEvent { path, kind }).await.is_err() {
                        return;
                    }
                }
            }
        });

        Ok(FsWatcher {
            _watcher: watcher,
            _debouncer: debouncer,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn watcher_emits_debounced_events() {
        let dir = tempfile::tempdir().unwrap();
        let (tx, mut rx) = mpsc::channel(16);
        let cancel = CancellationToken::new();
        let _w = FsWatcher::start(
            dir.path(),
            true,
            Duration::from_millis(100),
            tx,
            cancel.clone(),
        )
        .unwrap();

        tokio::time::sleep(Duration::from_millis(200)).await;
        let f = dir.path().join("new.txt");
        std::fs::write(&f, "a").unwrap();
        // rapid writes should collapse into one event
        std::fs::write(&f, "b").unwrap();
        std::fs::write(&f, "c").unwrap();

        let ev = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("event")
            .expect("some");
        assert_eq!(ev.kind, FsEventKind::Upsert);
        assert_eq!(ev.path, f);

        // removal event
        std::fs::remove_file(&f).unwrap();
        let mut saw_remove = false;
        for _ in 0..4 {
            match tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
                Ok(Some(e)) if e.kind == FsEventKind::Removed && e.path == f => {
                    saw_remove = true;
                    break;
                }
                Ok(Some(_)) => continue,
                _ => break,
            }
        }
        assert!(saw_remove, "expected removal event");

        cancel.cancel();
    }
}
