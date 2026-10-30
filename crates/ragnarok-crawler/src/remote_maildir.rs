//! Maildir source — the standard one-file-per-message mailbox layout used
//! by Dovecot, Postfix delivery agents, isync/mbsync, OfflineIMAP, KMail,
//! notmuch and friends.
//!
//! Layout: `<path>/{new,cur,tmp}` plus Maildir++ subfolders
//! `<path>/.<folder>/{new,cur}`. Every file in `new`/`cur` is one RFC-822
//! message; `tmp` (partial deliveries) and dotfiles (courier/dovecot
//! state) are never indexed. Messages stay in place — indexing reads
//! them and `/documents/{id}/stream` re-reads on demand, nothing is
//! copied.
//!
//! Identity: `maildir://<folder>/<uniq>.eml` — the Maildir unique name
//! (filename without the `:2,FLAGS` suffix) survives `new`→`cur` moves
//! and flag changes, so a message marked read doesn't reindex as a new
//! document. The etag is `size:mtime` — both preserved across renames —
//! so flag-only updates don't trigger re-embedding either.

use super::remote::RemoteDoc;
use ragnarok_core::{DocumentRecord, Error, Result, SourceConfig};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const MIME: &str = "message/rfc822";

fn cfg_parts(cfg: &SourceConfig) -> Result<(PathBuf, bool)> {
    match cfg {
        SourceConfig::Maildir {
            path,
            include_subfolders,
            ..
        } => Ok((path.clone(), *include_subfolders)),
        _ => Err(Error::Config("not a maildir source".into())),
    }
}

/// Split a Maildir filename into unique name + flag info (`abc:2,RS` →
/// (`abc`, `RS`)). `new/`-messages carry no suffix.
fn split_flags(name: &str) -> (&str, &str) {
    match name.split_once(":2,") {
        Some((u, f)) => (u, f),
        None => (name, ""),
    }
}

/// One Maildir++ folder (`INBOX` for the root, `.Foo` → `Foo`).
fn scan_folder(dir: &Path, folder: &str, out: &mut Vec<RemoteDoc>) -> Result<()> {
    for subdir in ["new", "cur"] {
        let rd = match std::fs::read_dir(dir.join(subdir)) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => {
                return Err(Error::Internal(format!(
                    "maildir read {}: {e}",
                    dir.join(subdir).display()
                )))
            }
        };
        for ent in rd.flatten() {
            if !ent.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let name = ent.file_name().to_string_lossy().to_string();
            if name.starts_with('.') {
                continue;
            }
            let (uniq, flags) = split_flags(&name);
            if uniq.is_empty() {
                continue;
            }
            let md = ent.metadata().ok();
            let (size, mtime) = md
                .as_ref()
                .map(|m| {
                    (
                        m.len(),
                        m.modified()
                            .ok()
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                            .map(|d| d.as_secs())
                            .unwrap_or(0),
                    )
                })
                .unwrap_or((0, 0));
            let mut metadata = HashMap::new();
            metadata.insert("remote_id".into(), uniq.to_string());
            metadata.insert("remote_path".into(), ent.path().display().to_string());
            metadata.insert("maildir_folder".into(), folder.to_string());
            metadata.insert("maildir_subdir".into(), subdir.to_string());
            if !flags.is_empty() {
                metadata.insert("maildir_flags".into(), flags.to_string());
            }
            out.push(RemoteDoc {
                vpath: format!("maildir://{folder}/{uniq}.eml"),
                title: uniq.to_string(),
                mime: MIME.into(),
                modified: (mtime > 0)
                    .then(|| chrono::DateTime::from_timestamp(mtime as i64, 0))
                    .flatten(),
                etag: Some(format!("{size}:{mtime}")),
                size: Some(size),
                ocr_text: None,
                metadata,
            });
        }
    }
    Ok(())
}

fn list_sync(root: &Path, subfolders: bool) -> Result<Vec<RemoteDoc>> {
    if !root.is_dir() {
        return Err(Error::InvalidInput(format!(
            "maildir is not a directory: {}",
            root.display()
        )));
    }
    let mut out = Vec::new();
    scan_folder(root, "INBOX", &mut out)?;
    if subfolders {
        if let Ok(rd) = std::fs::read_dir(root) {
            for ent in rd.flatten() {
                let is_dotdir = ent.file_name().to_string_lossy().starts_with('.');
                if !is_dotdir || !ent.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    continue;
                }
                let folder = ent
                    .file_name()
                    .to_string_lossy()
                    .trim_start_matches('.')
                    .to_string();
                if folder.is_empty() {
                    continue;
                }
                // Maildir++ subfolders are full maildirs themselves —
                // only descend when at least one of new/cur exists.
                if !ent.path().join("new").is_dir() && !ent.path().join("cur").is_dir() {
                    continue;
                }
                scan_folder(&ent.path(), &folder, &mut out)?;
            }
        }
    }
    Ok(out)
}

/// Find a message file by its unique name across every folder —
/// fallback when `remote_path` is stale after a `new`→`cur` move or
/// a subfolder rename between polls.
fn find_message(root: &Path, uniq: &str, subfolders: bool) -> Option<PathBuf> {
    let mut dirs = vec![root.to_path_buf()];
    if subfolders {
        if let Ok(rd) = std::fs::read_dir(root) {
            dirs.extend(rd.flatten().filter_map(|e| {
                (e.file_name().to_string_lossy().starts_with('.')
                    && e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                .then(|| e.path())
            }));
        }
    }
    for dir in dirs {
        for subdir in ["new", "cur"] {
            let p = dir.join(subdir);
            if let Ok(rd) = std::fs::read_dir(&p) {
                for ent in rd.flatten() {
                    let name = ent.file_name().to_string_lossy().to_string();
                    if split_flags(&name).0 == uniq {
                        return Some(ent.path());
                    }
                }
            }
        }
    }
    None
}

/// List every message under `new`/`cur` (root + Maildir++ subfolders).
pub async fn list(cfg: &SourceConfig) -> Result<Vec<RemoteDoc>> {
    let (root, sub) = cfg_parts(cfg)?;
    let r = root.clone();
    tokio::task::spawn_blocking(move || list_sync(&r, sub))
        .await
        .map_err(|e| Error::Internal(format!("maildir list join: {e}")))?
}

/// Read a persisted document's raw RFC-822 bytes — fresh from disk, or
/// relocated via the unique name when the file moved since indexing.
pub async fn fetch(cfg: &SourceConfig, rec: &DocumentRecord) -> Result<(Vec<u8>, String)> {
    let (root, sub) = cfg_parts(cfg)?;
    let stored = rec.metadata.get("remote_path").cloned();
    let uniq = rec
        .metadata
        .get("remote_id")
        .cloned()
        .or_else(|| {
            rec.path
                .strip_prefix("maildir://")
                .and_then(|p| p.rsplit('/').next())
                .map(|f| f.trim_end_matches(".eml").to_string())
        })
        .unwrap_or_default();
    let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
        if let Some(p) = stored {
            match std::fs::read(&p) {
                Ok(b) => return Ok(b),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(Error::Internal(format!("maildir read {p}: {e}"))),
            }
        }
        let Some(path) = find_message(&root, &uniq, sub) else {
            return Err(Error::Internal(format!(
                "maildir message {uniq} no longer exists under {}",
                root.display()
            )));
        };
        std::fs::read(&path)
            .map_err(|e| Error::Internal(format!("maildir read {}: {e}", path.display())))
    })
    .await
    .map_err(|e| Error::Internal(format!("maildir fetch join: {e}")))??;
    Ok((bytes, MIME.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    const MSG: &str = "From: a@x\r\nSubject: hello\r\n\r\nbody text\r\n";

    fn cfg_for(root: &Path) -> SourceConfig {
        SourceConfig::Maildir {
            path: root.to_path_buf(),
            include_subfolders: true,
            poll_interval_secs: 300,
        }
    }

    fn fixture(root: &Path) {
        for d in ["new", "cur", "tmp", ".Sent/new", ".Sent/cur", ".Sent/tmp"] {
            fs::create_dir_all(root.join(d)).unwrap();
        }
        fs::write(root.join("new/1000.host"), MSG).unwrap();
        fs::write(root.join("cur/1001.host:2,RS"), MSG).unwrap();
        fs::write(root.join(".Sent/cur/1002.host:2,S"), MSG).unwrap();
        fs::write(root.join("tmp/partial.msg"), "partial").unwrap();
        fs::write(root.join("new/.courier"), "state").unwrap();
        fs::write(root.join("dovecot-uidlist"), "state").unwrap();
    }

    #[test]
    fn split_flags_parses_suffix() {
        assert_eq!(split_flags("abc"), ("abc", ""));
        assert_eq!(split_flags("abc:2,RS"), ("abc", "RS"));
        assert_eq!(split_flags("a.b.c:2,"), ("a.b.c", ""));
    }

    #[test]
    fn list_scans_new_cur_and_subfolders_not_tmp() {
        let dir = tempfile::tempdir().unwrap();
        fixture(dir.path());
        let docs = list_sync(dir.path(), true).unwrap();
        let paths: Vec<&str> = docs.iter().map(|d| d.vpath.as_str()).collect();
        assert_eq!(paths.len(), 3);
        assert!(paths.contains(&"maildir://INBOX/1000.host.eml"));
        assert!(paths.contains(&"maildir://INBOX/1001.host.eml"));
        assert!(paths.contains(&"maildir://Sent/1002.host.eml"));
        let seen = docs.iter().find(|d| d.vpath.contains("1001")).unwrap();
        assert_eq!(seen.metadata["maildir_flags"], "RS");
        assert_eq!(seen.metadata["maildir_subdir"], "cur");
        assert_eq!(seen.mime, "message/rfc822");
        // tmp/, dotfiles and top-level state files are absent
        assert!(!paths.iter().any(|p| p.contains("partial")));
        assert!(!paths.iter().any(|p| p.contains("courier")));
    }

    #[test]
    fn list_ignores_non_maildir_dotdirs() {
        let dir = tempfile::tempdir().unwrap();
        fixture(dir.path());
        fs::create_dir_all(dir.path().join(".notmail")).unwrap(); // no new/cur
        fs::write(dir.path().join(".notmail/x"), "junk").unwrap();
        let docs = list_sync(dir.path(), true).unwrap();
        assert_eq!(docs.len(), 3);
        // subfolders off → only INBOX
        let docs = list_sync(dir.path(), false).unwrap();
        assert_eq!(docs.len(), 2);
        assert!(!docs.iter().any(|d| d.vpath.contains("/Sent/")));
    }

    #[test]
    fn list_errors_on_missing_dir() {
        let err = list_sync(Path::new("/nonexistent-maildir-xyz"), true).unwrap_err();
        assert!(err.to_string().contains("not a directory"));
    }

    #[tokio::test]
    async fn fetch_reads_stored_path_and_relocates_by_uniq() {
        let dir = tempfile::tempdir().unwrap();
        fixture(dir.path());
        let cfg = cfg_for(dir.path());
        let docs = list(&cfg).await.unwrap();
        let msg = docs.iter().find(|d| d.vpath.contains("1001")).unwrap();
        let rec = DocumentRecord {
            id: ragnarok_core::DocumentId::new(),
            source_id: ragnarok_core::SourceId::new(),
            path: msg.vpath.clone(),
            hash: String::new(),
            doc_type: ragnarok_core::DocType::Email,
            size: 0,
            modified: None,
            indexed_at: chrono::Utc::now(),
            chunk_count: 0,
            metadata: msg.metadata.clone(),
        };
        let (bytes, mime) = fetch(&cfg, &rec).await.unwrap();
        assert_eq!(mime, "message/rfc822");
        assert!(String::from_utf8_lossy(&bytes).contains("Subject: hello"));

        // simulate a new→cur move + flag change: stale remote_path,
        // relocation via unique name
        let moved = dir.path().join("new/1001.host");
        std::fs::rename(dir.path().join("cur/1001.host:2,RS"), &moved).unwrap();
        let (bytes, _) = fetch(&cfg, &rec).await.unwrap();
        assert!(String::from_utf8_lossy(&bytes).contains("Subject: hello"));

        // gone entirely → clean error, not a panic
        std::fs::remove_file(&moved).unwrap();
        assert!(fetch(&cfg, &rec).await.is_err());
    }
}
