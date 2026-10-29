//! Prometheus text exposition (`GET /metrics`, format v0.0.4).
//! Hand-rolled format builder — no dependency; label values are escaped
//! per the exposition spec (`\`, `"`, newline).

use crate::auth::SessionAuth;
use crate::state::AppState;
use axum::extract::State;
use axum::http::header;
use axum::response::IntoResponse;
use std::fmt::Write as _;
use std::sync::Arc;

fn esc_label(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn gauge(buf: &mut String, name: &str, help: &str, v: f64) {
    let _ = writeln!(buf, "# HELP {name} {help}");
    let _ = writeln!(buf, "# TYPE {name} gauge");
    let _ = writeln!(buf, "{name} {v}");
}

/// `GET /metrics` — Prometheus-compatible text exposition, bearer-auth.
pub async fn metrics(State(st): State<Arc<AppState>>, _a: SessionAuth) -> impl IntoResponse {
    let s = st.stats();
    let mut b = String::with_capacity(4096);
    gauge(
        &mut b,
        "ragnarok_indexed_documents",
        "Indexed documents",
        s.indexed_documents as f64,
    );
    gauge(
        &mut b,
        "ragnarok_indexed_chunks",
        "Indexed chunks",
        s.indexed_chunks as f64,
    );
    gauge(
        &mut b,
        "ragnarok_queue_depth",
        "Ingest queue depth",
        s.queue_depth as f64,
    );
    gauge(
        &mut b,
        "ragnarok_files_per_sec",
        "Ingest throughput",
        s.files_per_sec as f64,
    );
    gauge(
        &mut b,
        "ragnarok_process_rss_bytes",
        "Process RSS",
        s.process_rss_bytes as f64,
    );
    gauge(
        &mut b,
        "ragnarok_budget_bytes",
        "RAM budget",
        s.budget_bytes as f64,
    );
    gauge(
        &mut b,
        "ragnarok_ram_used_bytes",
        "System RAM used",
        s.ram_used_bytes as f64,
    );
    gauge(
        &mut b,
        "ragnarok_ram_available_bytes",
        "System RAM available",
        s.ram_available_bytes as f64,
    );
    gauge(
        &mut b,
        "ragnarok_cpu_percent",
        "Process CPU %",
        f64::from(s.cpu_percent),
    );
    gauge(
        &mut b,
        "ragnarok_thread_count",
        "Process threads",
        s.thread_count as f64,
    );
    gauge(
        &mut b,
        "ragnarok_pressure",
        "Backpressure level (0=normal..3=critical)",
        s.pressure as u8 as f64,
    );
    gauge(
        &mut b,
        "ragnarok_uptime_secs",
        "Daemon uptime",
        s.uptime_secs as f64,
    );

    let _ = writeln!(b, "# HELP ragnarok_source_files Per-source file counters");
    let _ = writeln!(b, "# TYPE ragnarok_source_files gauge");
    let states: std::collections::HashMap<String, ragnarok_core::SourceState> = st
        .pipeline
        .states()
        .into_iter()
        .map(|x| (x.source_id.to_string(), x))
        .collect();
    if let Ok(sources) = st.engine.meta.list_sources() {
        for src in &sources {
            let ty = serde_json::to_value(&src.config)
                .ok()
                .and_then(|c| c["type"].as_str().map(str::to_owned))
                .unwrap_or_default();
            let labels = format!(
                "source=\"{}\",name=\"{}\",type=\"{}\"",
                esc_label(&src.id.to_string()),
                esc_label(&src.name),
                esc_label(&ty)
            );
            let live = states.get(&src.id.to_string());
            let (idx, tot, err) = live
                .map(|x| (x.files_indexed, x.files_total, x.last_error.is_some() as u8))
                .unwrap_or((0, 0, 0));
            let _ = writeln!(
                b,
                "ragnarok_source_files{{{labels},state=\"indexed\"}} {idx}"
            );
            let _ = writeln!(b, "ragnarok_source_files{{{labels},state=\"total\"}} {tot}");
            let _ = writeln!(b, "ragnarok_source_files{{{labels},state=\"error\"}} {err}");
        }
    }
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        b,
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn metrics_exposition_and_auth() {
        let dir = tempfile::tempdir().unwrap();
        let (st, app, cancel) = app(dir.path()).await;
        index_test_doc(&st, "/docs/m.md", "metrics probe").await;

        // unauthenticated → 401
        let res = app
            .clone()
            .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

        let res = app
            .clone()
            .oneshot(
                Request::get("/metrics")
                    .header("authorization", TEST_BEARER)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let ct = res
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        assert!(ct.contains("version=0.0.4"), "content-type: {ct}");
        let text =
            String::from_utf8(to_bytes(res.into_body(), 1 << 20).await.unwrap().to_vec()).unwrap();
        for series in [
            "ragnarok_indexed_documents",
            "ragnarok_indexed_chunks",
            "ragnarok_queue_depth",
            "ragnarok_files_per_sec",
            "ragnarok_process_rss_bytes",
            "ragnarok_budget_bytes",
            "ragnarok_pressure",
            "ragnarok_uptime_secs",
        ] {
            assert!(
                text.contains(&format!("# TYPE {series} gauge")),
                "missing {series}"
            );
            assert!(
                text.lines().any(|l| l.starts_with(series)),
                "no samples for {series}"
            );
        }
        assert!(text.contains("ragnarok_indexed_documents 1"), "{text}");
        cancel.cancel();
    }

    #[test]
    fn label_escaping() {
        assert_eq!(super::esc_label("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
    }
}
