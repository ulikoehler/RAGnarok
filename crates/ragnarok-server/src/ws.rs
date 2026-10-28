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
