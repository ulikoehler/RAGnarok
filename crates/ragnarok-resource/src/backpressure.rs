use crate::telemetry::HardwareTelemetry;
use ragnarok_core::{PressureLevel, ResourceConfig};
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{Notify, Semaphore, SemaphorePermit};

/// Compute the daemon memory budget per spec:
/// `Daemon_Cap = min(total_ram * cap_fraction, available_ram - headroom)`.
pub fn compute_budget(config: &ResourceConfig, t: &HardwareTelemetry) -> u64 {
    let by_fraction = (t.total_ram as f64 * config.cap_fraction as f64) as u64;
    let by_headroom = t.available_ram.saturating_sub(config.headroom_bytes);
    by_fraction.min(by_headroom)
}

/// Map daemon RSS vs. budget onto a pressure level.
pub fn level_for(config: &ResourceConfig, rss: u64, budget: u64) -> PressureLevel {
    if budget == 0 {
        return PressureLevel::Hard;
    }
    let ratio = rss as f64 / budget as f64;
    if ratio >= config.hard_fraction as f64 {
        PressureLevel::Hard
    } else if ratio >= config.soft_fraction as f64 {
        PressureLevel::Soft
    } else {
        PressureLevel::Normal
    }
}

fn level_to_u8(l: PressureLevel) -> u8 {
    match l {
        PressureLevel::Normal => 0,
        PressureLevel::Soft => 1,
        PressureLevel::Hard => 2,
    }
}

fn u8_to_level(v: u8) -> PressureLevel {
    match v {
        1 => PressureLevel::Soft,
        2 => PressureLevel::Hard,
        _ => PressureLevel::Normal,
    }
}

/// Result of a non-blocking admission attempt.
pub enum AcquireOutcome<'a> {
    Granted(Permit<'a>),
    /// Backpressure active — caller should retry later or drop the work item.
    Deferred,
}

/// RAII handle; dropping releases the underlying semaphore slot and
/// decrements the in-flight counter.
pub struct Permit<'a> {
    _inner: SemaphorePermit<'a>,
    ctrl: &'a BackpressureController,
}

/// Gates all resource-heavy work (parsing, embedding, indexing).
///
/// * Normal: `base_permits` concurrent workers.
/// * Soft:   semaphore capacity halved.
/// * Hard:   `acquire()` blocks until pressure subsides.
pub struct BackpressureController {
    config: ResourceConfig,
    level: AtomicU8,
    budget: AtomicU64,
    rss: AtomicU64,
    sem: Semaphore,
    /// Wake tasks parked by hard-pressure blocking.
    relaxed: Notify,
    in_flight: AtomicUsize,
}

impl BackpressureController {
    pub fn new(config: ResourceConfig) -> Arc<Self> {
        Arc::new(BackpressureController {
            sem: Semaphore::new(config.base_permits),
            config,
            level: AtomicU8::new(0),
            budget: AtomicU64::new(0),
            rss: AtomicU64::new(0),
            relaxed: Notify::new(),
            in_flight: AtomicUsize::new(0),
        })
    }

    pub fn level(&self) -> PressureLevel {
        u8_to_level(self.level.load(Ordering::Acquire))
    }

    pub fn budget(&self) -> u64 {
        self.budget.load(Ordering::Relaxed)
    }

    pub fn rss(&self) -> u64 {
        self.rss.load(Ordering::Relaxed)
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }

    /// Scaling hint for batch sizing: 1.0 / 0.5 / 0.0.
    pub fn throttle_factor(&self) -> f32 {
        match self.level() {
            PressureLevel::Normal => 1.0,
            PressureLevel::Soft => 0.5,
            PressureLevel::Hard => 0.0,
        }
    }

    /// Target permit count for a level.
    fn target_permits(&self, level: PressureLevel) -> usize {
        match level {
            PressureLevel::Normal => self.config.base_permits,
            PressureLevel::Soft => (self.config.base_permits / 2).max(1),
            // Hard is handled by parking in acquire(); keep semaphore at 1 so
            // released permits do not immediately unblock a full worker pool.
            PressureLevel::Hard => 1,
        }
    }

    /// Re-evaluate pressure from a fresh telemetry snapshot.
    /// Returns `(new_level, transitioned)` — the flag is true only when the
    /// level actually changed, so callers can emit events on transitions.
    pub fn assess(&self, t: &HardwareTelemetry) -> (PressureLevel, bool) {
        let budget = compute_budget(&self.config, t);
        self.budget.store(budget, Ordering::Relaxed);
        self.rss.store(t.process_rss, Ordering::Relaxed);
        let next = level_for(&self.config, t.process_rss, budget);
        let prev = u8_to_level(self.level.swap(level_to_u8(next), Ordering::AcqRel));
        if next != prev {
            self.adjust_semaphore(next);
            if next < PressureLevel::Hard {
                // wake everything parked in acquire() so it can re-check
                self.relaxed.notify_waiters();
            }
        }
        (next, next != prev)
    }

    fn adjust_semaphore(&self, level: PressureLevel) {
        let target = self.target_permits(level);
        let available = self.sem.available_permits();
        if available < target {
            self.sem.add_permits(target - available);
        } else if available > target {
            self.sem.forget_permits(available - target);
        }
    }

    /// Blocking admission: parks while hard pressure is active, then takes a
    /// semaphore slot. `Err` only when the semaphore is closed (shutdown).
    pub async fn acquire(self: &Arc<Self>) -> Option<Permit<'_>> {
        loop {
            if self.level() != PressureLevel::Hard {
                break;
            }
            self.relaxed.notified().await;
        }
        let permit = self.sem.acquire().await.ok()?;
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        Some(Permit {
            _inner: permit,
            ctrl: self,
        })
    }

    /// Non-blocking admission for latency-sensitive paths.
    pub fn try_acquire(&self) -> AcquireOutcome<'_> {
        if self.level() == PressureLevel::Hard {
            return AcquireOutcome::Deferred;
        }
        match self.sem.try_acquire() {
            Ok(p) => {
                self.in_flight.fetch_add(1, Ordering::Relaxed);
                AcquireOutcome::Granted(Permit {
                    _inner: p,
                    ctrl: self,
                })
            }
            Err(_) => AcquireOutcome::Deferred,
        }
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        self.ctrl.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn cfg() -> ResourceConfig {
        ResourceConfig {
            base_permits: 8,
            soft_fraction: 0.75,
            hard_fraction: 0.90,
            cap_fraction: 0.70,
            headroom_bytes: 2 * 1024 * 1024 * 1024,
            ..Default::default()
        }
    }

    fn telem(total: u64, available: u64, rss: u64) -> HardwareTelemetry {
        HardwareTelemetry {
            total_ram: total,
            available_ram: available,
            process_rss: rss,
            ..Default::default()
        }
    }

    const GB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn budget_math() {
        let c = cfg();
        // total*0.7 = 11.2G; available-2G = 8G -> 8G wins
        assert_eq!(compute_budget(&c, &telem(16 * GB, 10 * GB, 0)), 8 * GB);
        // total*0.7 = 11.2G; available-2G = 12G -> 11.2G wins (fraction bound)
        // (cap_fraction is an f32 — mirror that in the expectation)
        assert_eq!(
            compute_budget(&c, &telem(16 * GB, 14 * GB, 0)),
            ((16 * GB) as f64 * 0.7f32 as f64) as u64
        );
        // headroom saturates at 0
        assert_eq!(compute_budget(&c, &telem(16 * GB, GB, 0)), 0);
    }

    #[test]
    fn level_thresholds() {
        let c = cfg();
        let budget = 1000;
        assert_eq!(level_for(&c, 0, budget), PressureLevel::Normal);
        assert_eq!(level_for(&c, 749, budget), PressureLevel::Normal);
        assert_eq!(level_for(&c, 750, budget), PressureLevel::Soft);
        assert_eq!(level_for(&c, 899, budget), PressureLevel::Soft);
        assert_eq!(level_for(&c, 900, budget), PressureLevel::Hard);
        assert_eq!(level_for(&c, 5000, budget), PressureLevel::Hard);
        assert_eq!(level_for(&c, 0, 0), PressureLevel::Hard);
    }

    #[test]
    fn throttle_factor_map() {
        let bp = BackpressureController::new(cfg());
        assert_eq!(bp.throttle_factor(), 1.0);
        bp.assess(&telem(16 * GB, 10 * GB, (8 * GB) * 3 / 4));
        // rss = 6G, budget = 8G -> 0.75 -> soft
        assert_eq!(bp.level(), PressureLevel::Soft);
        assert_eq!(bp.throttle_factor(), 0.5);
        bp.assess(&telem(16 * GB, 10 * GB, 8 * GB));
        assert_eq!(bp.level(), PressureLevel::Hard);
        assert_eq!(bp.throttle_factor(), 0.0);
    }

    #[test]
    fn assess_reports_transitions_once() {
        let bp = BackpressureController::new(cfg());
        let (_, changed) = bp.assess(&telem(16 * GB, 10 * GB, 0));
        assert!(!changed); // Normal -> Normal
        let (lvl, changed) = bp.assess(&telem(16 * GB, 10 * GB, 7 * GB));
        assert_eq!(lvl, PressureLevel::Soft);
        assert!(changed);
        let (_, changed) = bp.assess(&telem(16 * GB, 10 * GB, 7 * GB));
        assert!(!changed);
    }

    #[tokio::test]
    async fn soft_pressure_halves_concurrency() {
        let bp = BackpressureController::new(cfg());
        bp.assess(&telem(16 * GB, 10 * GB, 7 * GB)); // soft -> 4 permits
        assert_eq!(bp.level(), PressureLevel::Soft);
        let mut held = Vec::new();
        for _ in 0..4 {
            match bp.try_acquire() {
                AcquireOutcome::Granted(p) => held.push(p),
                AcquireOutcome::Deferred => panic!("permit should be available"),
            }
        }
        assert!(matches!(bp.try_acquire(), AcquireOutcome::Deferred));
        drop(held);
        match bp.try_acquire() {
            AcquireOutcome::Granted(_) => {}
            AcquireOutcome::Deferred => panic!("released permit should be re-usable"),
        };
    }

    #[tokio::test]
    async fn hard_pressure_blocks_then_recovers() {
        let bp = BackpressureController::new(cfg());
        bp.assess(&telem(16 * GB, 10 * GB, 8 * GB)); // hard
        assert_eq!(bp.level(), PressureLevel::Hard);
        assert!(matches!(bp.try_acquire(), AcquireOutcome::Deferred));

        let bp2 = bp.clone();
        let waiter = tokio::spawn(async move { bp2.acquire().await.is_some() });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!waiter.is_finished());

        // memory freed -> relax -> waiter unblocks
        bp.assess(&telem(16 * GB, 10 * GB, GB));
        assert!(tokio::time::timeout(Duration::from_secs(2), waiter)
            .await
            .unwrap()
            .unwrap());
    }

    #[tokio::test]
    async fn inflight_counted() {
        let bp = BackpressureController::new(cfg());
        {
            let _p = bp.acquire().await.unwrap();
            assert_eq!(bp.in_flight(), 1);
        }
        assert_eq!(bp.in_flight(), 0);
    }
}
