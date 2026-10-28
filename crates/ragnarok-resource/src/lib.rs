//! ragnarok-resource — dynamic RAM/CPU resource manager and backpressure
//! controller. Prevents the daemon from exhausting memory on consumer
//! hardware by computing a live budget and gating ingestion through an
//! adjustable semaphore.

mod backpressure;
mod monitor;
mod telemetry;

pub use backpressure::{AcquireOutcome, BackpressureController, Permit};
pub use monitor::ResourceMonitor;
pub use telemetry::{HardwareTelemetry, SysinfoSource, TelemetrySource};
