//! Source connectivity probes — "Test connection" for the add-source form.
//!
//! Each probe performs the lightest real check for its source type and
//! returns a short human-readable detail string on success. Everything is
//! strictly read-only: IMAP uses connect+login+EXAMINE+logout, remote
//! sources run their real `list` call (that's exactly what the poller will
//! do anyway), filesystem checks path readability.

use ragnarok_core::{Error, Result, SourceConfig};
use std::time::Duration;

/// Timeout applied to each probe — a hanging server shouldn't wedge the
/// admin UI for minutes. `remote::list` internally has its own client
/// timeouts; this is the outer cap.
const PROBE_TIMEOUT: Duration = Duration::from_secs(15);

/// Probe a source config. Returns a short detail line (e.g. mailbox size,
/// doc count) suitable for display next to the form's Test button.
pub async fn probe(cfg: &SourceConfig) -> Result<String> {
    tokio::time::timeout(PROBE_TIMEOUT, probe_inner(cfg))
        .await
        .map_err(|_| {
            Error::Internal(format!(
                "probe timed out after {}s",
                PROBE_TIMEOUT.as_secs()
            ))
        })?
}

async fn probe_inner(cfg: &SourceConfig) -> Result<String> {
    match cfg {
        SourceConfig::FileSystem { path, .. } => {
            if !path.is_dir() {
                return Err(Error::InvalidInput(format!(
                    "not a directory: {}",
                    path.display()
                )));
            }
            let mut entries = 0usize;
            let mut rd =
                std::fs::read_dir(path).map_err(|e| Error::Internal(format!("read_dir: {e}")))?;
            // count a small sample — enough to prove readability
            for e in rd.by_ref().take(101) {
                e.map_err(|e| Error::Internal(format!("read_dir entry: {e}")))?;
                entries += 1;
            }
            Ok(format!("Ordner lesbar — {entries} Einträge gefunden"))
        }

        SourceConfig::Imap {
            host,
            port,
            mailbox,
            tls,
            username,
            password_env,
            ..
        } => {
            let password = crate::imap::resolve_password(password_env)?;
            let client = crate::imap::connect(host, *port, *tls).await?;
            let (mut session, info) =
                crate::imap::open_readonly(client, username, &password, mailbox).await?;
            crate::imap::logout(&mut session).await;
            Ok(format!(
                "Login OK — Mailbox „{mailbox}“ enthält {} Nachrichten",
                info.exists
            ))
        }

        // local mailbox directory — list it like the poller does
        SourceConfig::Maildir { .. } => {
            let docs = crate::remote_maildir::list(cfg).await?;
            Ok(format!(
                "Maildir lesbar — {} Nachrichten gefunden",
                docs.len()
            ))
        }

        // every remote kind: run the real list call — same code path the
        // poller uses, so a green probe means the first poll will succeed
        _ => {
            let docs = crate::remote::list(cfg).await?;
            Ok(format!("Verbindung OK — {} Dokumente gefunden", docs.len()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::tests::*;

    #[tokio::test]
    async fn fs_probe_counts_entries_and_rejects_missing() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "x").unwrap();
        std::fs::write(dir.path().join("b.txt"), "y").unwrap();
        let cfg = SourceConfig::FileSystem {
            path: dir.path().to_path_buf(),
            recursive: true,
            watch: false,
            include: vec![],
            exclude: vec![],
        };
        let msg = probe(&cfg).await.unwrap();
        assert!(msg.contains('2'), "{msg}");

        let bad = SourceConfig::FileSystem {
            path: dir.path().join("nope"),
            recursive: true,
            watch: false,
            include: vec![],
            exclude: vec![],
        };
        assert!(probe(&bad).await.is_err());
    }

    #[tokio::test]
    async fn imap_probe_fails_on_missing_password_env() {
        std::env::remove_var("PROBE_NO_SUCH_VAR");
        let cfg = SourceConfig::Imap {
            host: "127.0.0.1".into(),
            port: 1,
            mailbox: "INBOX".into(),
            tls: false,
            username: "u".into(),
            password_env: "PROBE_NO_SUCH_VAR".into(),
            poll_interval_secs: 60,
        };
        let err = probe(&cfg).await.unwrap_err();
        assert!(err.to_string().contains("PROBE_NO_SUCH_VAR"), "{err}");
    }

    #[tokio::test]
    async fn remote_probe_lists_docs_and_surfaces_errors() {
        std::env::set_var("E2E_GIT_TOKEN", "tok");
        let tree = serde_json::json!({
            "truncated": false,
            "tree": [{"path":"a.md","type":"blob","sha":"b1","size":3}]
        });
        let mock = mock_http(vec![(
            "GET".into(),
            "/repos/acme/doku/git/trees/main".into(),
            (200, "application/json", tree.to_string()),
        )]);
        let cfg = SourceConfig::Git {
            api_base: mock.base.clone(),
            repo: "acme/doku".into(),
            ref_name: "main".into(),
            token_env: Some("E2E_GIT_TOKEN".into()),
            include: vec![],
            exclude: vec![],
            poll_interval_secs: 60,
        };
        let msg = probe(&cfg).await.unwrap();
        assert!(msg.contains("Verbindung OK"), "{msg}");

        let dead = SourceConfig::Git {
            api_base: "http://127.0.0.1:1".into(),
            repo: "acme/doku".into(),
            ref_name: "main".into(),
            token_env: Some("E2E_GIT_TOKEN".into()),
            include: vec![],
            exclude: vec![],
            poll_interval_secs: 60,
        };
        assert!(probe(&dead).await.is_err());
    }
}
