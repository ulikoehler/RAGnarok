//! Per-identity token-bucket rate limiting for the API surface.
//!
//! Identity = `Authorization` header when present (session token or
//! `ragk_` key — one bucket per credential) else the client IP.
//! Probes (`/healthz`, `/readyz`, `/metrics`) are exempt. The rate
//! (`requests/minute`, 0 = off) is a live runtime setting.

use crate::state::AppState;
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

const EXEMPT: [&str; 3] = ["/healthz", "/readyz", "/metrics"];
/// Buckets idle longer than this are evicted on the next sweep so a
/// long-running daemon can't accumulate dead keys.
const GC_AFTER_SECS: u64 = 600;
const GC_EVERY: u64 = 1024;

struct Bucket {
    tokens: f64,
    last: Instant,
}

/// Sliding token bucket: `rate` requests/minute per identity.
/// `rate == 0` disables limiting entirely (default).
pub struct RateLimiter {
    rate: AtomicU32,
    buckets: Mutex<HashMap<String, Bucket>>,
    checks: AtomicU32,
}

impl RateLimiter {
    pub fn new(rate_per_min: u32) -> Self {
        Self {
            rate: AtomicU32::new(rate_per_min),
            buckets: Mutex::new(HashMap::new()),
            checks: AtomicU32::new(0),
        }
    }

    pub fn rate(&self) -> u32 {
        self.rate.load(Ordering::Relaxed)
    }

    pub fn set_rate(&self, rate_per_min: u32) {
        self.rate.store(rate_per_min, Ordering::Relaxed);
    }

    /// Consume one token for `key`; `None` = allowed, `Some(secs)` =
    /// retry hint when the bucket is empty.
    pub fn check(&self, key: &str) -> Option<u64> {
        let rate = self.rate.load(Ordering::Relaxed);
        if rate == 0 {
            return None;
        }
        let per_sec = rate as f64 / 60.0;
        let cap = rate as f64;
        let now = Instant::now();
        let mut map = self.buckets.lock();
        // periodic sweep of stale buckets (~1 in GC_EVERY requests)
        if self
            .checks
            .fetch_add(1, Ordering::Relaxed)
            .is_multiple_of(GC_EVERY as u32)
        {
            map.retain(|_, b| now.duration_since(b.last).as_secs() < GC_AFTER_SECS);
        }
        let b = map.entry(key.to_owned()).or_insert(Bucket {
            tokens: cap,
            last: now,
        });
        b.tokens = (b.tokens + now.duration_since(b.last).as_secs_f64() * per_sec).min(cap);
        b.last = now;
        if b.tokens >= 1.0 {
            b.tokens -= 1.0;
            None
        } else {
            Some(((1.0 - b.tokens) / per_sec).ceil().max(1.0) as u64)
        }
    }
}

/// Axum middleware — keyed by Authorization header, falling back to the
/// peer IP (ConnectInfo extension) when unauthenticated — login attempts
/// included.
pub async fn rate_limit(State(st): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    if EXEMPT.contains(&req.uri().path()) {
        return next.run(req).await;
    }
    let key = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
        .or_else(|| {
            req.extensions()
                .get::<ConnectInfo<SocketAddr>>()
                .map(|ConnectInfo(a)| format!("ip:{}", a.ip()))
        })
        .unwrap_or_else(|| "anonymous".into());
    match st.rate_limiter.check(&key) {
        None => next.run(req).await,
        Some(retry) => (
            StatusCode::TOO_MANY_REQUESTS,
            [(header::RETRY_AFTER, retry.to_string())],
            Json(serde_json::json!({
                "error": "rate limit exceeded",
                "retry_after_secs": retry,
            })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_allows_up_to_rate_then_rejects() {
        let l = RateLimiter::new(60); // 1 req/sec
        for _ in 0..60 {
            assert!(l.check("k").is_none());
        }
        assert!(l.check("k").is_some());
        // other identities unaffected
        assert!(l.check("other").is_none());
    }

    #[test]
    fn zero_rate_is_unlimited() {
        let l = RateLimiter::new(0);
        for _ in 0..10_000 {
            assert!(l.check("k").is_none());
        }
    }

    #[test]
    fn retry_hint_is_positive_and_bounded() {
        let l = RateLimiter::new(120); // 2 req/sec
        for _ in 0..120 {
            l.check("k");
        }
        let hint = l.check("k").unwrap();
        assert!((1..=60).contains(&hint));
    }
}
