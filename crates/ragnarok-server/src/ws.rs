//! `/ws/v1/events` — pushes live telemetry to subscribers.
//!
//! Every connected socket receives:
//! * a `stats` snapshot every `telemetry_interval` (default 250 ms)
//! * every event on the broadcast bus (ingest progress, backpressure
//!   transitions, source state changes)

use crate::state::AppState;
use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::response::Response;
use ragnarok_core::TelemetryEvent;
use std::sync::Arc;

pub async fn ws_events(ws: WebSocketUpgrade, State(st): State<Arc<AppState>>) -> Response {
    ws.on_upgrade(move |socket| serve(socket, st))
}

async fn serve(mut socket: WebSocket, st: Arc<AppState>) {
    let mut rx = st.events.subscribe();
    let mut ticker = tokio::time::interval(st.telemetry_interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        let msg = tokio::select! {
            _ = ticker.tick() => TelemetryEvent::Stats(st.stats()),
            ev = rx.recv() => match ev {
                Ok(e) => e,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => break,
            },
            incoming = socket.recv() => match incoming {
                Some(Ok(Message::Close(_))) | None => break,
                Some(Err(_)) => break,
                _ => continue,
            },
        };
        let Ok(text) = serde_json::to_string(&msg) else {
            continue;
        };
        if socket.send(Message::Text(text.into())).await.is_err() {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router;
    use futures::StreamExt;
    use ragnarok_core::{CrawlerConfig, ResourceConfig};
    use ragnarok_crawler::IngestPipeline;
    use ragnarok_inference::{HashingEmbedder, InferencePool};
    use ragnarok_resource::{BackpressureController, ResourceMonitor};
    use ragnarok_store::{MetadataDb, SearchEngine, VectorEngine};
    use tokio::sync::broadcast;
    use tokio_util::sync::CancellationToken;

    /// Real socket round-trip: bind the router, connect a WS client, expect
    /// a periodic `stats` frame and a broadcast event.
    #[tokio::test]
    async fn ws_pushes_stats_and_events() {
        let dir = tempfile::tempdir().unwrap();
        let meta = Arc::new(MetadataDb::open(&dir.path().join("m.db")).unwrap());
        let vectors = VectorEngine::open(&dir.path().join("v"), 1024).unwrap();
        let embedder = Arc::new(HashingEmbedder::default());
        let engine = Arc::new(SearchEngine::new(vectors, meta, embedder.clone()));
        let bp = BackpressureController::new(ResourceConfig::default());
        let pool = InferencePool::new(embedder, 1, 4, bp);
        let monitor = ResourceMonitor::sysinfo(ResourceConfig::default());
        let (tx, _rx) = broadcast::channel(64);
        let pipeline = IngestPipeline::new(
            engine.clone(),
            pool,
            monitor.controller(),
            &CrawlerConfig::default(),
            tx.clone(),
        );
        let cancel = CancellationToken::new();
        let st = AppState::new(
            engine,
            monitor,
            pipeline,
            tx.clone(),
            cancel.clone(),
            std::time::Duration::from_millis(50),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router(st)).await.unwrap();
        });

        let url = format!("ws://{}/ws/v1/events", addr);
        let (mut ws, _) = tokio_tungstenite::connect_async(&url).await.unwrap();

        // first frame within a second must be a stats snapshot
        let msg = tokio::time::timeout(std::time::Duration::from_secs(2), ws.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&msg.into_text().unwrap()).unwrap();
        assert_eq!(v["event"], "stats");
        assert!(v["budget_bytes"].as_u64().unwrap() > 0);

        // a broadcast event is forwarded too
        tx.send(TelemetryEvent::Error {
            message: "test-error".into(),
        })
        .unwrap();
        let mut saw_error = false;
        for _ in 0..10 {
            let msg = tokio::time::timeout(std::time::Duration::from_secs(2), ws.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            let v: serde_json::Value = serde_json::from_str(&msg.into_text().unwrap()).unwrap();
            if v["event"] == "error" {
                assert_eq!(v["message"], "test-error");
                saw_error = true;
                break;
            }
        }
        assert!(saw_error, "expected the broadcast error event");

        // clean close
        ws.close(None).await.ok();
        cancel.cancel();
        server.abort();
    }
}
