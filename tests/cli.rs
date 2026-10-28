//! End-to-end tests against the compiled `ragnarok` binary.
//! Covers the real main.rs paths: index -> sources -> search -> serve.

use std::net::TcpListener;
use std::path::Path;
use std::process::{Command, Stdio};

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ragnarok"))
}

fn fixture_docs(dir: &Path) {
    std::fs::write(
        dir.join("alpha.txt"),
        "vector embeddings for semantic search",
    )
    .unwrap();
    std::fs::write(dir.join("beta.txt"), "rust async runtime with tokio tasks").unwrap();
    std::fs::write(
        dir.join("mail.eml"),
        "From: a@x\r\nSubject: hello\r\n\r\ntest mail body\r\n",
    )
    .unwrap();
}

fn get_text(url: &str) -> Result<String, Box<dyn std::error::Error>> {
    Ok(ureq::get(url).call()?.body_mut().read_to_string()?)
}

fn get_text_authed(url: &str, bearer: &str) -> Result<String, Box<dyn std::error::Error>> {
    Ok(ureq::get(url)
        .header("Authorization", bearer)
        .call()?
        .body_mut()
        .read_to_string()?)
}

#[test]
fn index_then_search_then_sources() {
    let data = tempfile::tempdir().unwrap();
    let docs = tempfile::tempdir().unwrap();
    fixture_docs(docs.path());
    let dd = data.path().to_str().unwrap().to_string();

    // index
    let out = bin()
        .args(["--data-dir", &dd, "index"])
        .arg(docs.path())
        .arg("--name")
        .arg("it-docs")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "index failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("indexed"));

    // sources lists the persisted source
    let out = bin().args(["--data-dir", &dd, "sources"]).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("it-docs"), "sources output: {stdout}");

    // search hits
    let out = bin()
        .args(["--data-dir", &dd, "search", "semantic search", "-n", "3"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("alpha.txt"), "search output: {stdout}");

    // no-hit query
    let out = bin()
        .args(["--data-dir", &dd, "search", "xyzzy-nonexistent-term"])
        .output()
        .unwrap();
    assert!(out.status.success());

    // bad path fails cleanly
    let out = bin()
        .args(["--data-dir", &dd, "index", "/nonexistent-dir-xyz"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}

#[test]
fn serve_serves_http_and_shuts_down() {
    let data = tempfile::tempdir().unwrap();
    let docs = tempfile::tempdir().unwrap();
    fixture_docs(docs.path());
    let dd = data.path().to_str().unwrap().to_string();

    // pick a free port
    let port = {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    // config file pointing the server at our port + source dir
    let cfg_path = data.path().join("config.toml");
    std::fs::write(
        &cfg_path,
        format!(
            r#"
data_dir = "{dd}"

[server]
bind = "127.0.0.1:{port}"
telemetry_interval_ms = 50

[[sources]]
name = "cfg-docs"
id = "cfg_src"
[sources.config]
type = "file_system"
path = "{}"
watch = false
"#,
            docs.path().display()
        ),
    )
    .unwrap();

    let mut child = bin()
        .args(["--config"])
        .arg(&cfg_path)
        .args(["serve"])
        .env("RAGNAROK_ADMIN_PASSWORD", "cli-test-pw")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();

    // wait for the listener — login is the only unauthenticated endpoint
    let base = format!("http://127.0.0.1:{port}");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    let token = loop {
        if std::time::Instant::now() > deadline {
            let _ = child.kill();
            panic!("server did not come up");
        }
        match ureq::post(format!("{base}/api/v1/auth/login")).send_json(
            serde_json::json!({"username": "admin", "password": "cli-test-pw"}),
        ) {
            Ok(mut r) => {
                let v: serde_json::Value =
                    serde_json::from_str(&r.body_mut().read_to_string().unwrap()).unwrap();
                break v["token"].as_str().unwrap().to_string();
            }
            Err(_) => std::thread::sleep(std::time::Duration::from_millis(100)),
        }
    };
    let bearer = format!("Bearer {token}");

    // stats endpoint (authenticated)
    let body = get_text_authed(&format!("{base}/api/v1/system/stats"), &bearer).unwrap();
    let stats: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(stats["budget_bytes"].as_u64().unwrap() > 0);

    // unauthenticated requests are rejected
    let resp = ureq::get(format!("{base}/api/v1/system/stats")).call();
    assert!(matches!(resp, Err(ureq::Error::StatusCode(401))));

    // source from config was registered + crawled
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let body = get_text_authed(&format!("{base}/api/v1/sources"), &bearer).unwrap();
        let srcs: serde_json::Value = serde_json::from_str(&body).unwrap();
        if !srcs.as_array().unwrap().is_empty() {
            break;
        }
        if std::time::Instant::now() > deadline {
            panic!("no source registered: {body}");
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // search over the crawled docs
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let resp = ureq::post(format!("{base}/api/v1/search"))
            .header("Authorization", &bearer)
            .send_json(serde_json::json!({"query": "semantic"}))
            .unwrap()
            .body_mut()
            .read_to_string()
            .unwrap();
        let hits: serde_json::Value = serde_json::from_str(&resp).unwrap();
        if !hits.as_array().unwrap().is_empty() {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "search never returned hits"
        );
        std::thread::sleep(std::time::Duration::from_millis(150));
    }

    // SPA served at /
    let resp = ureq::get(format!("{base}/")).call().unwrap();
    assert_eq!(resp.status(), 200);

    // graceful SIGTERM
    #[cfg(unix)]
    unsafe {
        libc_kill(child.id());
    }
    #[cfg(not(unix))]
    child.kill().unwrap();
    let status = child.wait().unwrap();
    #[cfg(unix)]
    assert!(status.success(), "daemon did not exit cleanly: {status}");
}

#[cfg(unix)]
unsafe fn libc_kill(pid: u32) {
    unsafe extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }
    kill(pid as i32, 15);
}
