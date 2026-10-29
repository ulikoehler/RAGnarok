//! Git host source over REST (GitHub, Gitea, Forgejo — same API shape):
//! the repo tree is listed via `GET /repos/{repo}/git/trees/{ref}
//! ?recursive=1`, individual blobs are fetched on demand through the
//! `contents` endpoint — no local clone, nothing copied to disk.
//!
//! A blob's git `sha` is content-addressed, so it doubles as the
//! document etag: unchanged files skip re-embedding for free.

use crate::remote::{checked_bytes, client, resolve_env, uri_encode, RemoteDoc};
use base64::Engine;
use ragnarok_core::{DocumentRecord, Error, Result, SourceConfig};
use serde_json::Value;
use std::collections::HashMap;

fn encode_path(path: &str) -> String {
    path.split('/')
        .map(|seg| uri_encode(seg, true))
        .collect::<Vec<_>>()
        .join("/")
}

fn auth_request(
    api_base: &str,
    path_and_query: &str,
    token: Option<&str>,
) -> reqwest::RequestBuilder {
    let mut req = client()
        .get(format!(
            "{}/{}",
            api_base.trim_end_matches('/'),
            path_and_query
        ))
        .header(reqwest::header::ACCEPT, "application/vnd.github+json");
    if let Some(t) = token {
        if !t.is_empty() {
            req = req.bearer_auth(t.to_owned());
        }
    }
    req
}

/// `GET /repos/{repo}/git/trees/{ref}?recursive=1` — one request per
/// sync, up to the host's ~100k entry cap (`truncated` is surfaced).
pub async fn git_list(
    api_base: &str,
    repo: &str,
    ref_name: &str,
    token: Option<&str>,
    include: &[String],
    exclude: &[String],
) -> Result<Vec<RemoteDoc>> {
    let resp = auth_request(
        api_base,
        &format!(
            "repos/{repo}/git/trees/{}?recursive=1",
            uri_encode(ref_name, true)
        ),
        token,
    )
    .send()
    .await
    .map_err(|e| Error::Internal(format!("git tree: {e}")))?;
    let body = checked_bytes(resp, "git tree").await?;
    let v: Value = serde_json::from_slice(&body)
        .map_err(|e| Error::Internal(format!("git tree json: {e}")))?;
    let mut docs = Vec::new();
    for e in v["tree"].as_array().into_iter().flatten() {
        if e["type"].as_str() != Some("blob") {
            continue; // trees, submodule commits
        }
        let path = match e["path"].as_str() {
            Some(p) if !p.is_empty() => p,
            _ => continue,
        };
        if !include.is_empty() && !include.iter().any(|p| crate::scanner::glob_match(p, path)) {
            continue;
        }
        if exclude.iter().any(|p| crate::scanner::glob_match(p, path)) {
            continue;
        }
        let mut metadata = HashMap::new();
        metadata.insert("remote_path".to_owned(), path.to_owned());
        metadata.insert("remote_ref".to_owned(), ref_name.to_owned());
        if let Some(u) = e["url"].as_str() {
            metadata.insert("remote_url".to_owned(), u.to_owned());
        }
        docs.push(RemoteDoc {
            vpath: format!("git://{repo}/{ref_name}/{path}"),
            title: path.rsplit('/').next().unwrap_or(path).to_owned(),
            mime: mime_guess::from_path(path)
                .first_or_octet_stream()
                .to_string(),
            modified: None,
            etag: e["sha"].as_str().map(str::to_owned),
            size: e["size"].as_u64(),
            ocr_text: None,
            metadata,
        });
    }
    Ok(docs)
}

/// On-demand blob fetch via the contents endpoint. GitHub honours
/// `Accept: …raw…` and streams raw bytes; Gitea/Forgejo reply JSON with
/// a base64 `content` field — both shapes are handled.
pub async fn git_fetch(cfg: &SourceConfig, rec: &DocumentRecord) -> Result<(Vec<u8>, String)> {
    let SourceConfig::Git {
        api_base,
        repo,
        ref_name,
        token_env,
        ..
    } = cfg
    else {
        return Err(Error::Config("not a git source".into()));
    };
    let path = rec
        .metadata
        .get("remote_path")
        .cloned()
        .ok_or_else(|| Error::Internal("git doc lacks remote_path".into()))?;
    let git_ref = rec
        .metadata
        .get("remote_ref")
        .cloned()
        .unwrap_or_else(|| ref_name.clone());
    let token = token_env
        .as_deref()
        .map(resolve_env)
        .transpose()?
        .filter(|t| !t.is_empty());
    let resp = auth_request(
        api_base,
        &format!(
            "repos/{repo}/contents/{}?ref={}",
            encode_path(&path),
            uri_encode(&git_ref, true)
        ),
        token.as_deref(),
    )
    .header(
        reqwest::header::ACCEPT,
        "application/vnd.github.raw+json, application/vnd.github+json",
    )
    .send()
    .await
    .map_err(|e| Error::Internal(format!("git fetch: {e}")))?;
    let ctype = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.split(';').next().unwrap_or(s).trim().to_owned())
        .unwrap_or_default();
    let body = checked_bytes(resp, "git fetch").await?;
    if ctype.starts_with("application/json") {
        // Gitea/Forgejo (or GitHub fallback): {"content": "<base64>", …}
        let v: Value = serde_json::from_slice(&body)
            .map_err(|e| Error::Internal(format!("git contents json: {e}")))?;
        let b64 = v["content"]
            .as_str()
            .ok_or_else(|| Error::Internal("git contents lacks `content`".into()))?;
        let cleaned: String = b64.chars().filter(|c| !c.is_whitespace()).collect();
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&cleaned)
            .map_err(|e| Error::Internal(format!("git contents base64: {e}")))?;
        return Ok((
            bytes,
            mime_guess::from_path(&path)
                .first_or_octet_stream()
                .to_string(),
        ));
    }
    let mime = if ctype.is_empty() || ctype == "application/octet-stream" {
        mime_guess::from_path(&path)
            .first_or_octet_stream()
            .to_string()
    } else {
        ctype
    };
    Ok((body, mime))
}

pub async fn list(cfg: &SourceConfig) -> Result<Vec<RemoteDoc>> {
    let SourceConfig::Git {
        api_base,
        repo,
        ref_name,
        token_env,
        include,
        exclude,
        ..
    } = cfg
    else {
        return Err(Error::Config("not a git source".into()));
    };
    let token = token_env
        .as_deref()
        .map(resolve_env)
        .transpose()?
        .filter(|t| !t.is_empty());
    git_list(api_base, repo, ref_name, token.as_deref(), include, exclude).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::tests::*;
    use chrono::Utc;

    fn cfg(base: &str) -> SourceConfig {
        SourceConfig::Git {
            api_base: base.to_owned(),
            repo: "acme/doku".into(),
            ref_name: "main".into(),
            token_env: Some("E2E_GIT_TOKEN".into()),
            include: vec![],
            exclude: vec!["*.lock".into()],
            poll_interval_secs: 60,
        }
    }

    fn rec(doc: &RemoteDoc) -> DocumentRecord {
        DocumentRecord {
            id: ragnarok_core::DocumentId::new(),
            source_id: ragnarok_core::SourceId::new(),
            path: doc.vpath.clone(),
            hash: String::new(),
            doc_type: ragnarok_core::DocType::Unknown,
            size: 0,
            modified: None,
            indexed_at: Utc::now(),
            chunk_count: 0,
            metadata: doc.metadata.clone(),
        }
    }

    #[tokio::test]
    async fn git_list_filters_and_fetches_raw() {
        std::env::set_var("E2E_GIT_TOKEN", "tok");
        let tree = serde_json::json!({
            "sha": "t1", "truncated": false,
            "tree": [
                {"path":"README.md","type":"blob","sha":"b1","size":12},
                {"path":"src","type":"tree","sha":"b2"},
                {"path":"src/main.rs","type":"blob","sha":"b3","size":88},
                {"path":"Cargo.lock","type":"blob","sha":"b4","size":999},
                {"path":"sub","type":"commit","sha":"b5"}
            ]
        });
        let mock = mock_http(vec![
            (
                "GET".into(),
                "/repos/acme/doku/git/trees/main".into(),
                (200, "application/json", tree.to_string()),
            ),
            (
                "GET".into(),
                "/repos/acme/doku/contents/README.md".into(),
                (200, "application/vnd.github.raw+json", "# Doku\n".into()),
            ),
        ]);
        let docs = list(&cfg(&mock.base)).await.unwrap();
        // tree entries + submodule + excluded lockfile are gone
        assert_eq!(docs.len(), 2, "{docs:?}");
        let readme = docs.iter().find(|d| d.title == "README.md").unwrap();
        assert_eq!(readme.vpath, "git://acme/doku/main/README.md");
        assert_eq!(readme.etag.as_deref(), Some("b1"));
        assert_eq!(readme.mime, "text/markdown");
        {
            let reqs = mock.requests.lock().unwrap();
            assert!(reqs.iter().any(|r| r.contains("Bearer tok")), "{reqs:?}");
        }

        let (bytes, mime) = git_fetch(&cfg(&mock.base), &rec(readme)).await.unwrap();
        assert_eq!(bytes, b"# Doku\n");
        assert_eq!(mime, "application/vnd.github.raw+json");
    }

    #[tokio::test]
    async fn git_fetch_decodes_json_contents() {
        std::env::set_var("E2E_GIT_TOKEN", "tok");
        let b64 = base64::engine::general_purpose::STANDARD.encode("fn main() {}\n");
        let body = serde_json::json!({
            "name": "main.rs", "path": "src/main.rs", "sha": "b3",
            "type": "file", "encoding": "base64", "content": b64
        });
        let tree = serde_json::json!({
            "truncated": false,
            "tree": [{"path":"src/main.rs","type":"blob","sha":"b3","size":14}]
        });
        let mock = mock_http(vec![
            (
                "GET".into(),
                "/repos/acme/doku/git/trees/main".into(),
                (200, "application/json", tree.to_string()),
            ),
            (
                "GET".into(),
                "/repos/acme/doku/contents/src/main.rs".into(),
                (200, "application/json", body.to_string()),
            ),
        ]);
        let docs = list(&cfg(&mock.base)).await.unwrap();
        assert_eq!(docs.len(), 1);
        let (bytes, mime) = git_fetch(&cfg(&mock.base), &rec(&docs[0])).await.unwrap();
        assert_eq!(bytes, b"fn main() {}\n");
        assert_eq!(mime, "text/x-rust");
    }
}
