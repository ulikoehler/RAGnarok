use crate::model::{PressureLevel, SourceState};
use serde::{Deserialize, Serialize};

/// Real-time system telemetry snapshot, pushed over `/ws/v1/events`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStats {
    pub ram_total_bytes: u64,
    pub ram_available_bytes: u64,
    pub ram_used_bytes: u64,
    /// Resident set size of this daemon process.
    pub process_rss_bytes: u64,
    /// Computed daemon memory budget (see resource manager).
    pub budget_bytes: u64,
    pub cpu_percent: f32,
    pub thread_count: usize,
    pub pressure: PressureLevel,
    pub indexed_documents: u64,
    pub indexed_chunks: u64,
    /// Rolling ingestion throughput.
    pub files_per_sec: f32,
    pub queue_depth: usize,
    pub uptime_secs: u64,
}

/// Event envelope broadcast to WebSocket subscribers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum TelemetryEvent {
    /// Periodic system snapshot (~250 ms cadence).
    Stats(SystemStats),
    /// One file/message moved through the ingest pipeline.
    Ingest {
        source_id: String,
        path: String,
        status: IngestStatus,
    },
    SourceState(SourceState),
    Backpressure {
        level: PressureLevel,
        budget_bytes: u64,
        rss_bytes: u64,
    },
    /// ONNX model provisioning progress — one event per status change
    /// plus throttled `downloading` updates during a fetch.
    ModelDownload {
        file: String,
        bytes_done: u64,
        bytes_total: Option<u64>,
        status: DownloadStatus,
    },
    Error {
        message: String,
    },
    /// Global reindex job progress — one event on start, then per-doc
    /// increments and a final `finished` event.
    Reindex {
        done: u64,
        total: u64,
        failed: u64,
        finished: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DownloadStatus {
    Started,
    Downloading,
    Verifying,
    Done,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IngestStatus {
    Started,
    Parsed,
    Indexed,
    Skipped,
    Failed,
    Deleted,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_event_serde() {
        let ev = TelemetryEvent::Ingest {
            source_id: "s1".into(),
            path: "/a.pdf".into(),
            status: IngestStatus::Indexed,
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("\"event\":\"ingest\""));
        assert!(json.contains("\"status\":\"indexed\""));
        let back: TelemetryEvent = serde_json::from_str(&json).unwrap();
        match back {
            TelemetryEvent::Ingest { status, .. } => assert_eq!(status, IngestStatus::Indexed),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn stats_serde_roundtrip() {
        let s = SystemStats {
            ram_total_bytes: 1,
            ram_available_bytes: 2,
            ram_used_bytes: 3,
            process_rss_bytes: 4,
            budget_bytes: 5,
            cpu_percent: 6.0,
            thread_count: 7,
            pressure: PressureLevel::Soft,
            indexed_documents: 8,
            indexed_chunks: 9,
            files_per_sec: 1.5,
            queue_depth: 11,
            uptime_secs: 12,
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: SystemStats = serde_json::from_str(&json).unwrap();
        assert_eq!(back.pressure, PressureLevel::Soft);
        assert_eq!(back.thread_count, 7);
    }
}
