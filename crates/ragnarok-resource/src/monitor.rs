use crate::backpressure::BackpressureController;
use crate::telemetry::{HardwareTelemetry, TelemetrySource};
use parking_lot::RwLock;
use ragnarok_core::{ResourceConfig, TelemetryEvent};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

/// Periodically samples hardware telemetry and feeds it to the backpressure
/// controller. Optionally broadcasts `Backpressure` events on level changes.
pub struct ResourceMonitor {
    source: Arc<dyn TelemetrySource>,
    controller: Arc<BackpressureController>,
    config: ResourceConfig,
    latest: RwLock<HardwareTelemetry>,
}

impl ResourceMonitor {
    pub fn new(source: Arc<dyn TelemetrySource>, config: ResourceConfig) -> Arc<Self> {
        let controller = BackpressureController::new(config.clone());
        let mon = Arc::new(ResourceMonitor {
            source,
            controller,
            config,
            latest: RwLock::new(HardwareTelemetry::default()),
        });
        // seed immediately so stats never report a zero budget
        mon.sample_once();
        mon
    }

    /// Convenience constructor using real sysinfo telemetry.
    pub fn sysinfo(config: ResourceConfig) -> Arc<Self> {
        Self::new(Arc::new(crate::telemetry::SysinfoSource::new()), config)
    }

    pub fn controller(&self) -> Arc<BackpressureController> {
        self.controller.clone()
    }

    pub fn latest(&self) -> HardwareTelemetry {
        *self.latest.read()
    }

    /// Take one sample and run the controller assessment. Exposed so tests and
    /// the stats endpoint can force a refresh synchronously.
    pub fn sample_once(&self) -> (HardwareTelemetry, ragnarok_core::PressureLevel, bool) {
        let t = self.source.snapshot();
        *self.latest.write() = t;
        let (level, changed) = self.controller.assess(&t);
        (t, level, changed)
    }

    /// Sampling loop; runs until `cancel` fires. Emits a `Backpressure`
    /// telemetry event on every level transition.
    pub async fn run(
        self: &Arc<Self>,
        event_tx: Option<broadcast::Sender<TelemetryEvent>>,
        cancel: CancellationToken,
    ) {
        let interval = Duration::from_millis(self.config.monitor_interval_ms.max(50));
        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                _ = tokio::time::sleep(interval) => {}
            }
            let (t, level, changed) = self.sample_once();
            if changed {
                tracing::info!(
                    ?level,
                    rss = t.process_rss,
                    budget = self.controller.budget(),
                    "backpressure level changed"
                );
                if let Some(tx) = &event_tx {
                    let _ = tx.send(TelemetryEvent::Backpressure {
                        level,
                        budget_bytes: self.controller.budget(),
                        rss_bytes: t.process_rss,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Scripted telemetry source for deterministic tests.
    struct MockSource {
        rss: AtomicU64,
        available: AtomicU64,
    }

    impl TelemetrySource for MockSource {
        fn snapshot(&self) -> HardwareTelemetry {
            HardwareTelemetry {
                total_ram: 16 * 1024 * 1024 * 1024,
                available_ram: self.available.load(Ordering::Relaxed),
                process_rss: self.rss.load(Ordering::Relaxed),
                ..Default::default()
            }
        }
    }

    const GB: u64 = 1024 * 1024 * 1024;

    fn config() -> ResourceConfig {
        ResourceConfig {
            monitor_interval_ms: 50,
            ..Default::default()
        }
    }

    #[test]
    fn sample_once_updates_state() {
        let src = Arc::new(MockSource {
            rss: AtomicU64::new(100 * 1024 * 1024),
            available: AtomicU64::new(10 * GB),
        });
        let mon = ResourceMonitor::new(src.clone(), config());
        let (t, level, _) = mon.sample_once();
        assert_eq!(t.process_rss, 100 * 1024 * 1024);
        assert_eq!(level, ragnarok_core::PressureLevel::Normal);
        assert_eq!(mon.latest().process_rss, 100 * 1024 * 1024);

        src.rss.store(8 * GB, Ordering::Relaxed);
        let (_, level, changed) = mon.sample_once();
        assert_eq!(level, ragnarok_core::PressureLevel::Hard);
        assert!(changed);
    }

    #[tokio::test(start_paused = true)]
    async fn run_loop_emits_on_transition() {
        let src = Arc::new(MockSource {
            rss: AtomicU64::new(0),
            available: AtomicU64::new(10 * GB),
        });
        let mon = ResourceMonitor::new(src.clone(), config());
        let (tx, mut rx) = broadcast::channel(8);
        let cancel = CancellationToken::new();
        let m2 = mon.clone();
        let c2 = cancel.clone();
        let task = tokio::spawn(async move { m2.run(Some(tx), c2).await });

        // first tick at Normal (no event), then push into Hard
        tokio::time::sleep(Duration::from_millis(60)).await;
        src.rss.store(8 * GB, Ordering::Relaxed);
        tokio::time::sleep(Duration::from_millis(60)).await;
        let ev = rx.try_recv().unwrap();
        match ev {
            TelemetryEvent::Backpressure { level, .. } => {
                assert_eq!(level, ragnarok_core::PressureLevel::Hard)
            }
            _ => panic!("unexpected event"),
        }

        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .unwrap()
            .unwrap();
    }
}
