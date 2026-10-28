use parking_lot::Mutex;
use sysinfo::{Pid, ProcessesToUpdate, System};

/// Point-in-time hardware snapshot consumed by the backpressure controller.
#[derive(Debug, Clone, Copy)]
pub struct HardwareTelemetry {
    pub total_ram: u64,
    pub available_ram: u64,
    pub used_ram: u64,
    /// RSS of this daemon process.
    pub process_rss: u64,
    pub cpu_percent: f32,
    pub thread_count: usize,
}

impl Default for HardwareTelemetry {
    fn default() -> Self {
        HardwareTelemetry {
            total_ram: 16 * 1024 * 1024 * 1024,
            available_ram: 8 * 1024 * 1024 * 1024,
            used_ram: 8 * 1024 * 1024 * 1024,
            process_rss: 512 * 1024 * 1024,
            cpu_percent: 0.0,
            thread_count: 1,
        }
    }
}

/// Pluggable telemetry provider — `SysinfoSource` in production, scripted
/// mocks in tests.
pub trait TelemetrySource: Send + Sync {
    fn snapshot(&self) -> HardwareTelemetry;
}

/// Real telemetry via `sysinfo`.
pub struct SysinfoSource {
    sys: Mutex<System>,
    pid: Pid,
}

impl Default for SysinfoSource {
    fn default() -> Self {
        Self::new()
    }
}

impl SysinfoSource {
    pub fn new() -> Self {
        let pid = Pid::from_u32(std::process::id());
        SysinfoSource {
            sys: Mutex::new(System::new()),
            pid,
        }
    }
}

impl TelemetrySource for SysinfoSource {
    fn snapshot(&self) -> HardwareTelemetry {
        let mut sys = self.sys.lock();
        sys.refresh_memory();
        sys.refresh_cpu_usage();
        sys.refresh_processes(ProcessesToUpdate::Some(&[self.pid]), true);

        let (rss, threads) = sys
            .process(self.pid)
            .map(|p| (p.memory(), p.tasks().map(|t| t.len()).unwrap_or(1)))
            .unwrap_or((0, 1));

        HardwareTelemetry {
            total_ram: sys.total_memory(),
            available_ram: sys.available_memory(),
            used_ram: sys.used_memory(),
            process_rss: rss,
            cpu_percent: sys.global_cpu_usage(),
            thread_count: threads,
        }
    }
}
