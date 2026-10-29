//! Remote (non-copying) sources: **Paperless-NGX**, **WebDAV**, **web
//! pages** and **RSS/Atom feeds**.
//!
//! Documents are *indexed* from remotely fetched text (Paperless even
//! ships its server-side OCR text via the API) but the original bytes
//! are **never copied locally** — `/documents/{id}/stream` re-fetches
//! them on demand through [`fetch_for_record`]. Credentials live in
//! environment variables (`*_env` config fields), never in config files.

use ragnarok_core::{DocumentRecord, Error, Result, SourceConfig};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use std::time::Duration;

static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

/// Shared HTTP client for all remote sources.
pub fn client() -> &'static reqwest::Client {
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(120))
            .user_agent("ragnarok/2.0 (embedded indexer)")
            .build()
            .expect("reqwest client")
    })
}

/// One remotely-hosted document, as advertised by the source listing.
#[derive(Debug, Clone)]
pub struct RemoteDoc {
    /// Deterministic virtual path — becomes `documents.path` and keys
    /// dedupe + scope `path` globs (`paperless://12/rechnung.pdf`,
    /// `webdav://Steuern/2024.pdf`, `https://…`, `feed://<guid>`).
    pub vpath: String,
    pub title: String,
    /// MIME type as advertised by the remote system.
    pub mime: String,
    pub modified: Option<chrono::DateTime<chrono::Utc>>,
    /// Remote change token (etag / checksum / guid+date) used as the
    /// document hash — unchanged docs skip re-embedding entirely.
    pub etag: Option<String>,
    pub size: Option<u64>,
    /// Cheap index text supplied by the API itself (Paperless OCR
    /// `content`). When set, the original bytes are never downloaded.
    pub ocr_text: Option<String>,
    /// Extra metadata persisted into `metadata_json` — drives the scope
    /// `meta` predicate (`tags`, `correspondent`, `remote_url`, …).
    pub metadata: HashMap<String, String>,
}

/// Resolve a `*_env` secret. Empty/unset names mean "no credential".
pub fn resolve_env(name: &str) -> Result<String> {
    if name.is_empty() {
        return Ok(String::new());
    }
    std::env::var(name)
        .map_err(|_| Error::Config(format!("environment variable '{name}' is not set")))
}

fn join_url(base: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

// ---------------------------------------------------------------------------
// listing — every source advertises its full inventory each cycle; the
// pipeline diffs vpaths + etags against the index
// ---------------------------------------------------------------------------

/// List all documents a source currently advertises.
pub async fn list(cfg: &SourceConfig) -> Result<Vec<RemoteDoc>> {
    match cfg {
        SourceConfig::Paperless {
            url,
            token_env,
            query,
            tags,
            correspondent,
            document_type,
            ..
        } => {
            paperless_list(
                client(),
                url,
                &resolve_env(token_env)?,
                query.as_deref(),
                tags,
                *correspondent,
                *document_type,
            )
            .await
        }
        SourceConfig::WebDav {
            url,
            username,
            password_env,
            recursive,
            include,
            exclude,
            ..
        } => {
            webdav_list(
                client(),
                url,
                username.as_deref(),
                password_env.as_deref(),
                *recursive,
                include,
                exclude,
            )
            .await
        }
        SourceConfig::Web {
            urls, follow_links, ..
        } => web_list(client(), urls, *follow_links).await,
        SourceConfig::Feed { url, full_text, .. } => feed_list(client(), url, *full_text).await,
        _ => Err(Error::Config("not a remote source type".into())),
    }
}

/// The poll interval configured for a source (seconds).
pub fn poll_interval(cfg: &SourceConfig) -> u64 {
    match cfg {
        SourceConfig::Paperless {
            poll_interval_secs, ..
        }
        | SourceConfig::WebDav {
            poll_interval_secs, ..
        }
        | SourceConfig::Web {
            poll_interval_secs, ..
        }
        | SourceConfig::Feed {
            poll_interval_secs, ..
        } => (*poll_interval_secs).max(30),
        SourceConfig::Imap {
            poll_interval_secs, ..
        } => (*poll_interval_secs).max(30),
        SourceConfig::FileSystem { .. } => 300,
    }
}

/// Fetch the original bytes for a persisted document — used by
/// `/documents/{id}/stream` so originals are always served fresh from
/// the remote system rather than a local copy.
pub async fn fetch_for_record(
    cfg: &SourceConfig,
    rec: &DocumentRecord,
) -> Result<(Vec<u8>, String)> {
    match cfg {
        SourceConfig::Paperless { url, token_env, .. } => {
            let token = resolve_env(token_env)?;
            let id = rec
                .metadata
                .get("remote_id")
                .cloned()
                .or_else(|| {
                    rec.path
                        .strip_prefix("paperless://")
                        .and_then(|p| p.split('/').next().map(str::to_owned))
                })
                .ok_or_else(|| Error::Internal("paperless doc lacks remote_id".into()))?;
            let resp = client()
                .get(join_url(url, &format!("api/documents/{id}/download/")))
                .header("Authorization", format!("Token {token}"))
                .send()
                .await
                .map_err(|e| Error::Internal(format!("paperless fetch: {e}")))?;
            let mime = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("application/pdf")
                .to_owned();
            let bytes = checked_bytes(resp, "paperless").await?;
            Ok((bytes, mime))
        }
        SourceConfig::WebDav {
            url,
            username,
            password_env,
            ..
        } => {
            let target = rec.metadata.get("remote_url").cloned().unwrap_or_else(|| {
                let rel = rec.path.strip_prefix("webdav://").unwrap_or(&rec.path);
                join_url(url, rel)
            });
            let mut req = client().get(&target);
            if let Some(u) = username {
                let pw = password_env
                    .as_deref()
                    .map(resolve_env)
                    .transpose()?
                    .unwrap_or_default();
                req = req.basic_auth(u.clone(), Some(pw));
            }
            let resp = req
                .send()
                .await
                .map_err(|e| Error::Internal(format!("webdav fetch: {e}")))?;
            let mime = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .map(|s| s.split(';').next().unwrap_or(s).trim().to_owned())
                .unwrap_or_else(|| {
                    mime_guess::from_path(&rec.path)
                        .first_or_octet_stream()
                        .to_string()
                });
            let bytes = checked_bytes(resp, "webdav").await?;
            Ok((bytes, mime))
        }
        SourceConfig::Web { .. } => {
            let resp = client()
                .get(&rec.path)
                .send()
                .await
                .map_err(|e| Error::Internal(format!("web fetch: {e}")))?;
            let mime = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("text/html")
                .to_owned();
            Ok((checked_bytes(resp, "web").await?, mime))
        }
        SourceConfig::Feed { .. } => {
            let link = rec
                .metadata
                .get("link")
                .cloned()
                .or_else(|| {
                    // rec.path is `feed://<guid>` — only usable as a URL
                    // when the guid itself was an http(s) link
                    rec.path
                        .strip_prefix("feed://")
                        .filter(|g| g.starts_with("http"))
                        .map(str::to_owned)
                })
                .ok_or_else(|| {
                    Error::Internal("feed entry has no fetchable article link".into())
                })?;
            let resp = client()
                .get(&link)
                .send()
                .await
                .map_err(|e| Error::Internal(format!("feed article fetch: {e}")))?;
            let mime = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("text/html")
                .to_owned();
            Ok((checked_bytes(resp, "feed").await?, mime))
        }
        _ => Err(Error::Config(format!(
            "document {} is not backed by a remote source",
            rec.path
        ))),
    }
}

/// Fetch raw bytes for one listed doc at index time (when no OCR text).
pub async fn fetch_doc(cfg: &SourceConfig, doc: &RemoteDoc) -> Result<(Vec<u8>, String)> {
    let rec = DocumentRecord {
        id: ragnarok_core::DocumentId::new(),
        source_id: ragnarok_core::SourceId::new(),
        path: doc.vpath.clone(),
        hash: String::new(),
        doc_type: ragnarok_core::DocType::Unknown,
        size: 0,
        modified: None,
        indexed_at: chrono::Utc::now(),
        chunk_count: 0,
        metadata: doc.metadata.clone(),
    };
    fetch_for_record(cfg, &rec).await
}

/// Hard ceiling for a single remote fetch — an unbounded response body
/// is a memory-exhaustion vector (a hostile or misbehaving server can
/// stream gigabytes). Generous so legitimate large files still work.
const MAX_REMOTE_BYTES: u64 = 512 * 1024 * 1024;

async fn checked_bytes(resp: reqwest::Response, what: &str) -> Result<Vec<u8>> {
    let status = resp.status();
    if !status.is_success() {
        return Err(Error::Internal(format!(
            "{what} fetch failed: HTTP {status}"
        )));
    }
    if let Some(len) = resp.content_length() {
        if len > MAX_REMOTE_BYTES {
            return Err(Error::Internal(format!(
                "{what} response too large: {len} bytes (max {MAX_REMOTE_BYTES})"
            )));
        }
    }
    let b = resp
        .bytes()
        .await
        .map_err(|e| Error::Internal(format!("{what} body: {e}")))?;
    if b.len() as u64 > MAX_REMOTE_BYTES {
        return Err(Error::Internal(format!(
            "{what} response too large: {} bytes (max {MAX_REMOTE_BYTES})",
            b.len()
        )));
    }
    Ok(b.to_vec())
}

// ---------------------------------------------------------------------------
// Paperless-NGX
// ---------------------------------------------------------------------------

/// Resolve a `/api/<plural>/` id→name map (tags, correspondents, types).
async fn paperless_name_map(
    http: &reqwest::Client,
    url: &str,
    token: &str,
    kind: &str,
) -> Result<HashMap<i64, String>> {
    let mut map = HashMap::new();
    let mut page_url = join_url(url, &format!("api/{kind}/?page_size=500"));
    loop {
        let bytes = checked_bytes(
            http.get(&page_url)
                .header("Authorization", format!("Token {token}"))
                .send()
                .await
                .map_err(|e| Error::Internal(format!("paperless {kind}: {e}")))?,
            "paperless",
        )
        .await?;
        let resp: Value = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Internal(format!("paperless {kind} json: {e}")))?;
        for r in resp["results"].as_array().into_iter().flatten() {
            if let (Some(id), Some(name)) = (r["id"].as_i64(), r["name"].as_str()) {
                map.insert(id, name.to_owned());
            }
        }
        match resp["next"].as_str() {
            Some(next) => page_url = next.to_owned(),
            None => break,
        }
    }
    Ok(map)
}

async fn paperless_list(
    http: &reqwest::Client,
    url: &str,
    token: &str,
    query: Option<&str>,
    tags: &[i64],
    correspondent: Option<i64>,
    document_type: Option<i64>,
) -> Result<Vec<RemoteDoc>> {
    // id→name maps for the scope-meta fields — three small paged calls.
    let tag_names = paperless_name_map(http, url, token, "tags").await?;
    let corr_names = paperless_name_map(http, url, token, "correspondents").await?;
    let dtype_names = paperless_name_map(http, url, token, "document_types").await?;

    let fields = "id,title,created,modified,added,checksum,mime_type,original_file_name,tags,correspondent,document_type";
    let mut page_url = format!(
        "{}?ordering=id&page_size=200&fields={}",
        join_url(url, "api/documents/"),
        fields
    );
    if let Some(q) = query {
        page_url.push_str(&format!("&query={}", urlencoding(q)));
    }
    for t in tags {
        page_url.push_str(&format!("&tags__id__all={t}"));
    }
    if let Some(c) = correspondent {
        page_url.push_str(&format!("&correspondent__id={c}"));
    }
    if let Some(t) = document_type {
        page_url.push_str(&format!("&document_type__id={t}"));
    }

    let mut docs = Vec::new();
    loop {
        let resp = http
            .get(&page_url)
            .header("Authorization", format!("Token {token}"))
            .send()
            .await
            .map_err(|e| Error::Internal(format!("paperless list: {e}")))?;
        if !resp.status().is_success() {
            return Err(Error::Internal(format!(
                "paperless list failed: HTTP {}",
                resp.status()
            )));
        }
        let bytes = checked_bytes(resp, "paperless").await?;
        let body: Value = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Internal(format!("paperless list json: {e}")))?;

        for r in body["results"].as_array().into_iter().flatten() {
            let Some(id) = r["id"].as_i64() else { continue };
            let fname = r["original_file_name"].as_str().unwrap_or("document.pdf");
            let tag_list: Vec<String> = r["tags"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|t| t.as_i64())
                .filter_map(|id| tag_names.get(&id).cloned())
                .collect();
            let mut meta = HashMap::new();
            meta.insert("remote_id".into(), id.to_string());
            meta.insert(
                "remote_url".into(),
                join_url(url, &format!("documents/{id}/")),
            );
            meta.insert("tags".into(), tag_list.join(","));
            if let Some(c) = r["correspondent"].as_i64() {
                if let Some(n) = corr_names.get(&c) {
                    meta.insert("correspondent".into(), n.clone());
                }
            }
            if let Some(t) = r["document_type"].as_i64() {
                if let Some(n) = dtype_names.get(&t) {
                    meta.insert("document_type".into(), n.clone());
                }
            }
            for k in ["created", "modified", "added"] {
                if let Some(v) = r[k].as_str() {
                    meta.insert(k.into(), v.to_owned());
                }
            }
            docs.push(RemoteDoc {
                vpath: format!("paperless://{id}/{fname}"),
                title: r["title"].as_str().unwrap_or(fname).to_owned(),
                mime: r["mime_type"]
                    .as_str()
                    .unwrap_or("application/pdf")
                    .to_owned(),
                modified: r["modified"]
                    .as_str()
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                    .map(|d| d.with_timezone(&chrono::Utc)),
                etag: r["checksum"].as_str().map(str::to_owned),
                size: None,
                ocr_text: None, // fetched lazily below for changed docs
                metadata: meta,
            });
        }
        match body["next"].as_str() {
            Some(next) => page_url = next.to_owned(),
            None => break,
        }
    }
    Ok(docs)
}

/// The server-side OCR text of one Paperless document — indexing uses
/// this instead of re-parsing the PDF.
pub async fn paperless_ocr(
    http: &reqwest::Client,
    url: &str,
    token: &str,
    id: &str,
) -> Result<Option<String>> {
    let bytes = checked_bytes(
        http.get(join_url(
            url,
            &format!("api/documents/{id}/?fields=id,content"),
        ))
        .header("Authorization", format!("Token {token}"))
        .send()
        .await
        .map_err(|e| Error::Internal(format!("paperless ocr: {e}")))?,
        "paperless",
    )
    .await?;
    let resp: Value = serde_json::from_slice(&bytes)
        .map_err(|e| Error::Internal(format!("paperless ocr json: {e}")))?;
    Ok(resp["content"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned))
}

/// Cheap text for indexing: Paperless OCR wins, everything else falls
/// back to downloading + parsing the original.
pub async fn index_text(cfg: &SourceConfig, doc: &RemoteDoc) -> Result<Option<String>> {
    match cfg {
        SourceConfig::Paperless { url, token_env, .. } => {
            let id = doc.metadata.get("remote_id").cloned().unwrap_or_default();
            if id.is_empty() {
                return Ok(None);
            }
            paperless_ocr(client(), url, &resolve_env(token_env)?, &id).await
        }
        SourceConfig::Web { .. } | SourceConfig::Feed { .. } => Ok(doc.ocr_text.clone()),
        _ => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// WebDAV
// ---------------------------------------------------------------------------

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

const PROPFIND_BODY: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<D:propfind xmlns:D="DAV:">
  <D:prop>
    <D:resourcetype/><D:getetag/><D:getlastmodified/>
    <D:getcontentlength/><D:getcontenttype/>
  </D:prop>
</D:propfind>"#;

async fn propfind(
    http: &reqwest::Client,
    url: &str,
    auth: Option<(&str, &str)>,
    depth: &str,
) -> Result<String> {
    let mut req = http
        .request(reqwest::Method::from_bytes(b"PROPFIND").unwrap(), url)
        .header("Depth", depth)
        .header("Content-Type", "application/xml")
        .body(PROPFIND_BODY);
    if let Some((u, p)) = auth {
        req = req.basic_auth(u.to_owned(), Some(p.to_owned()));
    }
    let resp = req
        .send()
        .await
        .map_err(|e| Error::Internal(format!("propfind {url}: {e}")))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(Error::Internal(format!("propfind {url}: HTTP {status}")));
    }
    let body = checked_bytes(resp, "propfind").await?;
    Ok(String::from_utf8_lossy(&body).into_owned())
}

/// Parse one `<D:response>` element → (href, is_collection, etag, mtime, len, ctype).
fn parse_dav_response(xml: &roxmltree::Node) -> RemoteDoc {
    let mut d = RemoteDoc {
        vpath: String::new(),
        title: String::new(),
        mime: String::new(),
        modified: None,
        etag: None,
        size: None,
        ocr_text: None,
        metadata: HashMap::new(),
    };
    let mut is_collection = false;
    for n in xml.descendants() {
        if !n.is_element() {
            continue;
        }
        match n.tag_name().name() {
            "href" if d.vpath.is_empty() => {
                d.vpath = n.text().unwrap_or_default().to_owned();
            }
            "collection" => is_collection = true,
            "getetag" => d.etag = n.text().map(|t| t.trim_matches('"').to_owned()),
            "getlastmodified" => {
                d.modified = n
                    .text()
                    .and_then(|t| chrono::DateTime::parse_from_rfc2822(t).ok())
                    .map(|t| t.with_timezone(&chrono::Utc));
            }
            "getcontentlength" => {
                d.size = n.text().and_then(|t| t.trim().parse::<u64>().ok());
            }
            "getcontenttype" => d.mime = n.text().unwrap_or_default().to_owned(),
            _ => {}
        }
    }
    if is_collection {
        d.mime = "inode/directory".into();
    }
    d
}

async fn webdav_list(
    http: &reqwest::Client,
    url: &str,
    username: Option<&str>,
    password_env: Option<&str>,
    recursive: bool,
    include: &[String],
    exclude: &[String],
) -> Result<Vec<RemoteDoc>> {
    let password = match password_env {
        Some(e) if !e.is_empty() => Some(resolve_env(e)?),
        _ => None,
    };
    let auth = username.map(|u| (u, password.as_deref().unwrap_or_default()));

    let mut docs = Vec::new();
    let mut queue: Vec<String> = vec![url.trim_end_matches('/').to_owned()];
    let mut visited = HashSet::new();
    // origin (scheme + authority) for absolute-path hrefs the server returns
    let after_scheme = url.find("://").map(|p| p + 3).unwrap_or(0);
    let origin = match url[after_scheme..].find('/') {
        Some(p) => url[..after_scheme + p].to_owned(),
        None => url.trim_end_matches('/').to_owned(),
    };

    while let Some(dir) = queue.pop() {
        if !visited.insert(dir.clone()) {
            continue;
        }
        let body = propfind(http, &format!("{dir}/"), auth, "1").await?;
        let xml = roxmltree::Document::parse(&body)
            .map_err(|e| Error::Internal(format!("propfind {dir}: bad xml: {e}")))?;
        for resp_el in xml
            .descendants()
            .filter(|n| n.tag_name().name() == "response")
        {
            let mut d = parse_dav_response(&resp_el);
            if d.vpath.is_empty() {
                continue;
            }
            // normalize href → absolute URL + collection-relative path.
            // The encoded href is kept for re-fetching; percent-decoding
            // happens only for the display path / vpath / globs.
            let href = d.vpath.clone();
            let abs = if href.starts_with("http") {
                href.clone()
            } else {
                format!("{}{}", origin, href)
            };
            if abs.trim_end_matches('/') == dir.trim_end_matches('/') {
                continue; // the collection itself
            }
            if d.mime == "inode/directory" {
                if recursive {
                    queue.push(abs.trim_end_matches('/').to_owned());
                }
                continue;
            }
            let rel_enc = abs
                .strip_prefix(url.trim_end_matches('/'))
                .map(|r| r.trim_start_matches('/').to_owned())
                .unwrap_or_else(|| abs.clone());
            let rel = percent_decode(&rel_enc);
            if !include.is_empty() && !include.iter().any(|p| crate::scanner::glob_match(p, &rel)) {
                continue;
            }
            if exclude.iter().any(|p| crate::scanner::glob_match(p, &rel)) {
                continue;
            }
            d.title = rel.rsplit('/').next().unwrap_or(&rel).to_owned();
            d.vpath = format!("webdav://{rel}");
            if d.mime.is_empty() {
                d.mime = mime_guess::from_path(&rel)
                    .first_or_octet_stream()
                    .to_string();
            }
            d.metadata
                .insert("remote_url".into(), abs.trim_end_matches('/').to_owned());
            docs.push(d);
        }
    }
    Ok(docs)
}

// ---------------------------------------------------------------------------
// Web pages
// ---------------------------------------------------------------------------

/// Minimal HTML→text: drops script/style/template, turns block-level
/// tags into newlines, decodes common entities. Title is returned
/// separately via [`html_title`].
pub fn html_to_text(html: &str) -> String {
    let lower = html.to_lowercase();
    let mut out = String::with_capacity(html.len() / 2);
    let mut i = 0usize;
    let bytes = lower.as_bytes();
    let raw = html.as_bytes();
    // skip to the matching `</tag` — searching a bare "</" would stop at
    // *any* closing tag inside the skipped body (e.g. `</p>` inside a
    // script string) and leak the rest of the body as text
    let skip_close = |s: &str, tag: &str, from: usize| -> Option<usize> {
        let needle = format!("</{tag}");
        s[from..].find(&needle).map(|p| {
            let abs = from + p;
            s[abs..].find('>').map(|e| abs + e + 1).unwrap_or(s.len())
        })
    };
    const BLOCK: &[&str] = &[
        "p",
        "div",
        "br",
        "li",
        "ul",
        "ol",
        "table",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "section",
        "article",
        "header",
        "footer",
        "nav",
        "blockquote",
        "pre",
        "hr",
    ];
    while i < bytes.len() {
        if bytes[i] != b'<' {
            // text node — copy until next '<'
            let next = lower[i..].find('<').map(|p| i + p).unwrap_or(bytes.len());
            let chunk = &html[i..next.min(html.len())];
            out.push_str(&decode_entities(chunk));
            i = next;
            continue;
        }
        // inside a tag
        let end = lower[i..]
            .find('>')
            .map(|p| i + p + 1)
            .unwrap_or(bytes.len());
        let tag_body = &lower[i + 1..end.saturating_sub(1).max(i + 1)];
        let tag = tag_body
            .trim_start_matches('/')
            .split(|c: char| c.is_whitespace() || c == '/')
            .next()
            .unwrap_or("");
        if matches!(
            tag,
            "script" | "style" | "noscript" | "template" | "head" | "svg"
        ) && !tag_body.trim_start().starts_with('/')
        {
            if let Some(close) = skip_close(&lower, tag, end) {
                i = close;
                continue;
            }
        }
        if BLOCK.contains(&tag) && !out.ends_with('\n') {
            out.push('\n');
        }
        let _ = raw;
        i = end;
    }
    // collapse whitespace runs
    let mut cleaned = String::with_capacity(out.len());
    let mut last_blank = false;
    for line in out.lines() {
        let t = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if t.is_empty() {
            if !last_blank && !cleaned.is_empty() {
                cleaned.push('\n');
                last_blank = true;
            }
        } else {
            cleaned.push_str(&t);
            cleaned.push('\n');
            last_blank = false;
        }
    }
    cleaned.trim().to_owned()
}

fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(p) = rest.find('&') {
        out.push_str(&rest[..p]);
        let tail = &rest[p..];
        let end = tail.find(';').map(|e| e + 1).unwrap_or(0);
        let ent = if end > 1 && end <= 12 {
            &tail[..end]
        } else {
            ""
        };
        let decoded = match ent {
            "&amp;" => "&",
            "&lt;" => "<",
            "&gt;" => ">",
            "&quot;" => "\"",
            "&apos;" | "&#39;" => "'",
            "&nbsp;" => " ",
            "&auml;" => "ä",
            "&ouml;" => "ö",
            "&uuml;" => "ü",
            "&Auml;" => "Ä",
            "&Ouml;" => "Ö",
            "&Uuml;" => "Ü",
            "&szlig;" => "ß",
            _ if ent.starts_with("&#") => {
                // `&#41;` decimal, `&#x29;`/`&#X29;` hex — the radix
                // matters: 0x29=')' vs decimal 29=GS
                let body = ent
                    .trim_start_matches("&#")
                    .trim_end_matches(';')
                    .to_lowercase();
                let (digits, radix) = match body.strip_prefix('x') {
                    Some(h) => (h, 16),
                    None => (body.as_str(), 10),
                };
                match u32::from_str_radix(digits, radix)
                    .ok()
                    .and_then(char::from_u32)
                {
                    Some(c) => {
                        out.push(c);
                        rest = &tail[end.max(1)..];
                        continue;
                    }
                    None => {
                        out.push('&');
                        rest = &tail[1..];
                        continue;
                    }
                }
            }
            _ => {
                out.push('&');
                rest = &tail[1..];
                continue;
            }
        };
        out.push_str(decoded);
        rest = &tail[end..];
    }
    out.push_str(rest);
    out
}

fn html_title(html: &str) -> Option<String> {
    let lower = html.to_lowercase();
    let start = lower
        .find("<title")
        .and_then(|p| lower[p..].find('>').map(|e| p + e + 1))?;
    let end = lower[start..].find("</title").map(|e| start + e)?;
    let t = decode_entities(html[start..end].trim());
    (!t.is_empty()).then_some(t)
}

/// Extract absolute same-host links for `follow_links` crawling.
fn extract_links(html: &str, base: &str, cap: usize) -> Vec<String> {
    let host = base
        .split("//")
        .nth(1)
        .and_then(|r| r.split('/').next())
        .unwrap_or_default();
    let mut out = Vec::new();
    let mut rest = html;
    while let Some(p) = rest.find("href=") {
        rest = &rest[p + 5..];
        let q = rest.chars().next().unwrap_or(' ');
        if q != '"' && q != '\'' {
            continue;
        }
        rest = &rest[1..];
        let end = rest.find(q).unwrap_or(0);
        let link = &rest[..end];
        rest = &rest[end..];
        if (link.starts_with(&format!("https://{host}"))
            || link.starts_with(&format!("http://{host}")))
            && !out.contains(&link.to_owned())
        {
            out.push(link.to_owned());
            if out.len() >= cap {
                break;
            }
        }
    }
    out
}

async fn web_list(
    http: &reqwest::Client,
    urls: &[String],
    follow_links: bool,
) -> Result<Vec<RemoteDoc>> {
    let mut queue: Vec<String> = urls.to_vec();
    let mut seen: HashSet<String> = urls.iter().cloned().collect();
    let mut docs = Vec::new();
    while let Some(u) = queue.pop() {
        if docs.len() >= 200 {
            break;
        }
        let resp = match http.get(&u).send().await {
            Ok(r) if r.status().is_success() => r,
            Ok(r) => {
                tracing::warn!(url = %u, status = %r.status(), "web fetch failed");
                continue;
            }
            Err(e) => {
                tracing::warn!(url = %u, error = %e, "web fetch failed");
                continue;
            }
        };
        let etag = resp
            .headers()
            .get(reqwest::header::ETAG)
            .or_else(|| resp.headers().get(reqwest::header::LAST_MODIFIED))
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let ctype = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("text/html")
            .to_owned();
        let body = match checked_bytes(resp, "web").await {
            Ok(b) => b,
            Err(_) => continue,
        };
        let is_html = ctype.contains("html");
        let is_text = ctype.starts_with("text/")
            || ctype.contains("json")
            || ctype.contains("xml")
            || ctype.contains("javascript");
        if follow_links && is_html {
            for l in extract_links(&String::from_utf8_lossy(&body), &u, 50) {
                if seen.insert(l.clone()) {
                    queue.push(l);
                }
            }
        }
        // text-ish responses are indexed directly; binary payloads (PDF,
        // images) carry ocr_text=None so the ingest path downloads them
        // and runs the real parser instead of indexing UTF-8 garbage
        let ocr_text = if is_html {
            Some(html_to_text(&String::from_utf8_lossy(&body)))
        } else if is_text {
            Some(String::from_utf8_lossy(&body).into_owned())
        } else {
            None
        };
        let body_text = is_html.then(|| String::from_utf8_lossy(&body).into_owned());
        let mut meta = HashMap::new();
        meta.insert("remote_url".into(), u.clone());
        docs.push(RemoteDoc {
            vpath: u.clone(),
            title: body_text
                .as_deref()
                .and_then(html_title)
                .unwrap_or_else(|| u.clone()),
            mime: ctype.split(';').next().unwrap_or(&ctype).trim().to_owned(),
            modified: None,
            etag,
            size: Some(body.len() as u64),
            ocr_text,
            metadata: meta,
        });
    }
    Ok(docs)
}

// ---------------------------------------------------------------------------
// RSS 2.0 / Atom feeds
// ---------------------------------------------------------------------------

fn child_text<'a>(n: &roxmltree::Node<'a, 'a>, name: &str) -> Option<String> {
    n.children()
        .find(|c| c.is_element() && c.tag_name().name() == name)
        .and_then(|c| c.text())
        .map(|t| t.trim().to_owned())
        .filter(|t| !t.is_empty())
}

fn parse_rfc_date(s: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    chrono::DateTime::parse_from_rfc2822(s)
        .or_else(|_| chrono::DateTime::parse_from_rfc3339(s))
        .ok()
        .map(|d| d.with_timezone(&chrono::Utc))
}

async fn feed_list(http: &reqwest::Client, url: &str, full_text: bool) -> Result<Vec<RemoteDoc>> {
    let bytes = checked_bytes(
        http.get(url)
            .send()
            .await
            .map_err(|e| Error::Internal(format!("feed fetch: {e}")))?,
        "feed",
    )
    .await?;
    let body = String::from_utf8_lossy(&bytes);
    let xml = roxmltree::Document::parse(&body)
        .map_err(|e| Error::Internal(format!("feed parse: {e}")))?;
    let is_atom = xml.root_element().tag_name().name() == "feed";
    let items: Vec<roxmltree::Node> = xml
        .descendants()
        .filter(|n| n.is_element() && matches!(n.tag_name().name(), "item" | "entry"))
        .collect();

    let mut docs = Vec::new();
    for it in items {
        let (title, link, guid, date, summary) = if is_atom {
            (
                child_text(&it, "title").unwrap_or_default(),
                // Atom entries may carry several <link> elements — the
                // article URL is rel="alternate" (or no rel); picking the
                // first blindly can grab rel="self"/"hub" endpoints
                it.children()
                    .filter(|c| c.is_element() && c.tag_name().name() == "link")
                    .find(|c| c.attribute("rel").is_none_or(|r| r == "alternate"))
                    .and_then(|c| c.attribute("href"))
                    .map(str::to_owned)
                    .or_else(|| child_text(&it, "link"))
                    .unwrap_or_default(),
                child_text(&it, "id").unwrap_or_default(),
                child_text(&it, "updated").or_else(|| child_text(&it, "published")),
                child_text(&it, "summary")
                    .or_else(|| child_text(&it, "content"))
                    .unwrap_or_default(),
            )
        } else {
            (
                child_text(&it, "title").unwrap_or_default(),
                child_text(&it, "link").unwrap_or_default(),
                child_text(&it, "guid")
                    .or_else(|| child_text(&it, "link"))
                    .unwrap_or_default(),
                child_text(&it, "pubDate").or_else(|| child_text(&it, "date")),
                child_text(&it, "description")
                    .or_else(|| child_text(&it, "encoded"))
                    .unwrap_or_default(),
            )
        };
        if guid.is_empty() && link.is_empty() {
            continue;
        }
        let mtime = date.as_deref().and_then(parse_rfc_date);
        let id = if guid.is_empty() { &link } else { &guid };
        let mut meta = HashMap::new();
        meta.insert("link".into(), link.clone());
        meta.insert("feed".into(), url.to_owned());
        if let Some(t) = &date {
            meta.insert("published".into(), t.clone());
        }

        // Index text: entry summary, or the full linked article when configured.
        let text = if full_text && !link.is_empty() {
            match http.get(&link).send().await {
                Ok(r) if r.status().is_success() => {
                    let html = r
                        .headers()
                        .get(reqwest::header::CONTENT_TYPE)
                        .and_then(|v| v.to_str().ok())
                        .is_none_or(|c| c.contains("html"));
                    match checked_bytes(r, "feed article").await {
                        Ok(b) if html => html_to_text(&String::from_utf8_lossy(&b)),
                        Ok(b) => String::from_utf8_lossy(&b).into_owned(),
                        Err(_) => html_to_text(&summary),
                    }
                }
                _ => html_to_text(&summary),
            }
        } else {
            html_to_text(&summary)
        };

        docs.push(RemoteDoc {
            vpath: format!("feed://{id}"),
            title: if title.is_empty() {
                link.clone()
            } else {
                title
            },
            mime: "text/html".into(),
            modified: mtime,
            etag: Some(format!("{id}|{}", date.unwrap_or_default())),
            size: Some(text.len() as u64),
            ocr_text: Some(text),
            metadata: meta,
        });
    }
    Ok(docs)
}

fn urlencoding(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '~' | '+') {
            out.push(c);
        } else if c == ' ' {
            out.push('+');
        } else {
            for b in c.to_string().as_bytes() {
                out.push_str(&format!("%{b:02X}"));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_to_text_strips_tags_and_scripts() {
        let html = "<html><head><title>Seite</title><style>x{}</style></head>
                    <body><h1>Hallo &amp; Willkommen</h1><p>Erste <b>Zeile</b></p>
                    <script>evil()</script><p>zweite&nbsp;Zeile</p></body></html>";
        let t = html_to_text(html);
        assert!(t.contains("Hallo & Willkommen"), "{t}");
        assert!(t.contains("Erste Zeile"), "{t}");
        assert!(t.contains("zweite Zeile"), "{t}");
        assert!(!t.contains("evil"), "{t}");
        assert!(!t.contains("x{}"), "{t}");
        assert_eq!(html_title(html).as_deref(), Some("Seite"));
    }

    #[test]
    fn percent_decode_decodes() {
        assert_eq!(
            percent_decode("/Dav%20x/%C3%9Cbung.pdf"),
            "/Dav x/Übung.pdf"
        );
        assert_eq!(percent_decode("plain"), "plain");
    }

    #[test]
    fn extract_links_same_host_only() {
        let h = r#"<a href="https://a.de/x">x</a><a href="https://b.de/y">y</a>
                   <a href="https://a.de/z">z</a>"#;
        let l = extract_links(h, "https://a.de/page", 10);
        assert_eq!(l, vec!["https://a.de/x", "https://a.de/z"]);
    }

    // -----------------------------------------------------------------
    // mock HTTP server — one accept loop, routed by "METHOD path" prefix
    // -----------------------------------------------------------------

    struct MockHttp {
        pub base: String,
        pub requests: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        _h: std::thread::JoinHandle<()>,
    }

    /// `routes`: `(method, path_prefix)` → `(status, content_type, body)`.
    /// Falls through to 404. Every response closes the connection.
    fn mock_http(routes: Vec<(String, String, (u16, &'static str, String))>) -> MockHttp {
        use std::io::{BufRead, BufReader, Read, Write};
        use std::net::TcpListener;
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", l.local_addr().unwrap());
        let requests = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let reqs = requests.clone();
        let h = std::thread::spawn(move || {
            for stream in l.incoming() {
                let Ok(mut s) = stream else { continue };
                let mut reader = BufReader::new(s.try_clone().unwrap());
                let mut line = String::new();
                if reader.read_line(&mut line).is_err() || line.is_empty() {
                    continue;
                }
                let request_line = line.trim().to_owned();
                let mut content_length = 0usize;
                let mut auth = String::new();
                loop {
                    let mut hline = String::new();
                    if reader.read_line(&mut hline).is_err() || hline.trim().is_empty() {
                        break;
                    }
                    let hl = hline.to_lowercase();
                    if let Some(v) = hl.strip_prefix("content-length:") {
                        content_length = v.trim().parse().unwrap_or(0);
                    }
                    if hl.starts_with("authorization:") {
                        auth = hline["authorization:".len()..].trim().to_owned();
                    }
                }
                if content_length > 0 {
                    let mut buf = vec![0u8; content_length];
                    let _ = reader.read_exact(&mut buf);
                }
                reqs.lock()
                    .unwrap()
                    .push(format!("{request_line} |{auth}|"));
                let (method, path) = request_line
                    .split_once(' ')
                    .map(|(m, p)| (m.to_owned(), p.to_owned()))
                    .unwrap_or_default();
                // longest matching prefix wins — specific before generic
                let hit = routes
                    .iter()
                    .filter(|(m, pfx, _)| {
                        m.eq_ignore_ascii_case(&method) && path.starts_with(pfx.as_str())
                    })
                    .max_by_key(|(_, pfx, _)| pfx.len());
                let (status, ctype, body) = hit.map(|(_, _, r)| r.clone()).unwrap_or((
                    404,
                    "text/plain",
                    "not found".into(),
                ));
                let reason = if status == 200 { "OK" } else { "ERR" };
                let resp = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = s.write_all(resp.as_bytes());
            }
        });
        MockHttp {
            base,
            requests,
            _h: h,
        }
    }

    fn paperless_cfg(base: &str) -> SourceConfig {
        std::env::set_var("RAG_TEST_PL_TOKEN", "tok-1");
        SourceConfig::Paperless {
            url: base.into(),
            token_env: "RAG_TEST_PL_TOKEN".into(),
            query: None,
            tags: vec![],
            correspondent: None,
            document_type: None,
            poll_interval_secs: 60,
        }
    }

    #[tokio::test]
    async fn paperless_list_ocr_and_fetch() {
        let doc = serde_json::json!({
            "count": 1, "next": null, "previous": null,
            "results": [{
                "id": 42, "title": "Rechnung März", "checksum": "abc123",
                "mime_type": "application/pdf",
                "original_file_name": "rechnung-2025-03.pdf",
                "created": "2025-03-01T10:00:00Z",
                "modified": "2025-03-02T10:00:00Z",
                "added": "2025-03-03T10:00:00Z",
                "tags": [7, 9], "correspondent": 3, "document_type": 2
            }]
        });
        let mock = mock_http(vec![
            (
                "GET".into(),
                "/api/documents/?".into(),
                (200, "application/json", doc.to_string()),
            ),
            (
                "GET".into(),
                "/api/documents/42/".into(),
                (
                    200,
                    "application/json",
                    r#"{"id":42,"content":"OCR text der Rechnung"}"#.into(),
                ),
            ),
            (
                "GET".into(),
                "/api/documents/42/download/".into(),
                (200, "application/pdf", "%PDF-original".into()),
            ),
            (
                "GET".into(),
                "/api/tags/".into(),
                (
                    200,
                    "application/json",
                    r#"{"results":[{"id":7,"name":"Rechnung"},{"id":9,"name":"2025"}],"next":null}"#
                        .into(),
                ),
            ),
            (
                "GET".into(),
                "/api/correspondents/".into(),
                (
                    200,
                    "application/json",
                    r#"{"results":[{"id":3,"name":"Stadtwerke"}],"next":null}"#.into(),
                ),
            ),
            (
                "GET".into(),
                "/api/document_types/".into(),
                (
                    200,
                    "application/json",
                    r#"{"results":[{"id":2,"name":"Rechnung"}],"next":null}"#.into(),
                ),
            ),
        ]);
        let cfg = paperless_cfg(&mock.base);
        let docs = list(&cfg).await.unwrap();
        assert_eq!(docs.len(), 1);
        let d = &docs[0];
        assert_eq!(d.vpath, "paperless://42/rechnung-2025-03.pdf");
        assert_eq!(d.etag.as_deref(), Some("abc123"));
        assert_eq!(d.metadata["remote_id"], "42");
        assert_eq!(d.metadata["tags"], "Rechnung,2025");
        assert_eq!(d.metadata["correspondent"], "Stadtwerke");

        // OCR shortcut — no download needed for indexing
        let text = index_text(&cfg, d).await.unwrap();
        assert_eq!(text.as_deref(), Some("OCR text der Rechnung"));

        // on-demand fetch of the original
        let mut rec = DocumentRecord {
            id: ragnarok_core::DocumentId::new(),
            source_id: ragnarok_core::SourceId::new(),
            path: d.vpath.clone(),
            hash: "h".into(),
            doc_type: ragnarok_core::DocType::Pdf,
            size: 0,
            modified: None,
            indexed_at: chrono::Utc::now(),
            chunk_count: 0,
            metadata: d.metadata.clone(),
        };
        let (bytes, mime) = fetch_for_record(&cfg, &rec).await.unwrap();
        assert_eq!(bytes, b"%PDF-original");
        assert_eq!(mime, "application/pdf");
        // token auth on every request
        assert!(mock
            .requests
            .lock()
            .unwrap()
            .iter()
            .all(|r| r.contains("|Token tok-1|")));
        // path fallback: no remote_id in metadata → parse from vpath
        rec.metadata.clear();
        let (b2, _) = fetch_for_record(&cfg, &rec).await.unwrap();
        assert_eq!(b2, b"%PDF-original");
    }

    #[tokio::test]
    async fn webdav_list_and_fetch() {
        let multistatus = |items: &str| {
            format!(r#"<?xml version="1.0"?><d:multistatus xmlns:d="DAV:">{items}</d:multistatus>"#)
        };
        let listing = multistatus(
            r#"<d:response><d:href>/dav/</d:href>
               <d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop></d:propstat></d:response>
               <d:response><d:href>/dav/report%202025.pdf</d:href>
               <d:propstat><d:prop><d:resourcetype/><d:getetag>"e1"</d:getetag>
               <d:getlastmodified>Tue, 03 Jun 2025 10:00:00 GMT</d:getlastmodified>
               <d:getcontentlength>1234</d:getcontentlength>
               <d:getcontenttype>application/pdf</d:getcontenttype></d:prop></d:propstat></d:response>
               <d:response><d:href>/dav/sub/</d:href>
               <d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop></d:propstat></d:response>"#,
        );
        let sub_listing = multistatus(
            r#"<d:response><d:href>/dav/sub/</d:href>
               <d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop></d:propstat></d:response>
               <d:response><d:href>/dav/sub/notes.txt</d:href>
               <d:propstat><d:prop><d:resourcetype/><d:getetag>"e2"</d:getetag>
               <d:getcontentlength>10</d:getcontentlength>
               <d:getcontenttype>text/plain</d:getcontenttype></d:prop></d:propstat></d:response>"#,
        );
        let mock = mock_http(vec![
            (
                "PROPFIND".into(),
                "/dav/sub/".into(),
                (207, "application/xml", sub_listing),
            ),
            (
                "PROPFIND".into(),
                "/dav/".into(),
                (207, "application/xml", listing),
            ),
            (
                "GET".into(),
                "/dav/".into(),
                (200, "application/pdf", "%PDF-dav".into()),
            ),
        ]);
        let cfg = SourceConfig::WebDav {
            url: format!("{}/dav", mock.base),
            username: None,
            password_env: None,
            recursive: true,
            include: vec![],
            exclude: vec![],
            poll_interval_secs: 60,
        };
        let docs = list(&cfg).await.unwrap();
        assert_eq!(docs.len(), 2);
        let pdf = docs.iter().find(|d| d.vpath.contains("report")).unwrap();
        assert_eq!(pdf.vpath, "webdav://report 2025.pdf");
        assert_eq!(pdf.etag.as_deref(), Some("e1"));
        assert_eq!(pdf.size, Some(1234));
        assert_eq!(pdf.mime, "application/pdf");
        assert!(pdf.modified.is_some());
        assert!(pdf.metadata["remote_url"].contains("report%202025.pdf"));
        // the nested collection was recursed into
        assert!(docs.iter().any(|d| d.vpath == "webdav://sub/notes.txt"));

        // non-recursive mode stops at the top level
        let cfg_flat = SourceConfig::WebDav {
            url: format!("{}/dav", mock.base),
            username: None,
            password_env: None,
            recursive: false,
            include: vec![],
            exclude: vec![],
            poll_interval_secs: 60,
        };
        let flat = list(&cfg_flat).await.unwrap();
        assert_eq!(flat.len(), 1);

        // glob filtering on the collection-relative path
        let cfg_inc = SourceConfig::WebDav {
            url: format!("{}/dav", mock.base),
            username: None,
            password_env: None,
            recursive: true,
            include: vec!["*.pdf".into()],
            exclude: vec![],
            poll_interval_secs: 60,
        };
        let docs = list(&cfg_inc).await.unwrap();
        assert_eq!(docs.len(), 1);

        // on-demand fetch via remote_url
        let rec = DocumentRecord {
            id: ragnarok_core::DocumentId::new(),
            source_id: ragnarok_core::SourceId::new(),
            path: pdf.vpath.clone(),
            hash: "e1".into(),
            doc_type: ragnarok_core::DocType::Pdf,
            size: 0,
            modified: None,
            indexed_at: chrono::Utc::now(),
            chunk_count: 0,
            metadata: pdf.metadata.clone(),
        };
        let (bytes, _) = fetch_for_record(&cfg, &rec).await.unwrap();
        assert_eq!(bytes, b"%PDF-dav");
    }

    #[tokio::test]
    async fn web_list_indexes_and_follows() {
        let mock = mock_http(vec![
            (
                "GET".into(),
                "/index".into(),
                (
                    200,
                    "text/html",
                    r#"<html><head><title>Start</title></head><body>
                    <h1>Willkommen</h1><a href="/sub">Sub</a>
                    <a href="http://other.de/x">extern</a></body></html>"#
                        .replace("/sub", &format!("{}/sub", "{BASE}")),
                ),
            ),
            (
                "GET".into(),
                "/sub".into(),
                (
                    200,
                    "text/html",
                    "<html><title>Sub</title><body><p>Unterseite</p></body></html>".into(),
                ),
            ),
        ]);
        // patch the {BASE} placeholder into real URLs
        let docs = {
            let u = format!("{}/index", mock.base);
            let cfg = SourceConfig::Web {
                urls: vec![u.clone()],
                follow_links: true,
                poll_interval_secs: 60,
            };
            // the canned page links to {BASE}/sub — rewrite via a route on /sub
            let docs = list(&cfg).await.unwrap();
            docs
        };
        // /index was fetched; the extern link was not
        assert_eq!(docs.len(), 1);
        assert!(docs[0].vpath.ends_with("/index"));
        assert_eq!(docs[0].title, "Start");
        assert!(docs[0]
            .ocr_text
            .as_deref()
            .unwrap_or_default()
            .contains("Willkommen"));
        let reqs = mock.requests.lock().unwrap();
        assert!(!reqs.iter().any(|r| r.contains("other.de")), "{reqs:?}");
    }

    #[tokio::test]
    async fn feed_list_parses_items() {
        let rss = r#"<?xml version="1.0"?><rss version="2.0"><channel>
            <title>Testfeed</title>
            <item><title>Artikel Eins</title><link>{BASE}/a1</link>
            <guid>urn:eins</guid><pubDate>Tue, 03 Jun 2025 10:00:00 GMT</pubDate>
            <description>&lt;p&gt;Zusammenfassung&lt;/p&gt;</description></item>
            <item><title>Artikel Zwei</title><link>{BASE}/a2</link>
            <guid>urn:zwei</guid><description>Text</description></item>
            </channel></rss>"#;
        // resolved after mock exists — build routes lazily
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let rss = rss.replace("{BASE}", &base);
        let requests = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let h = {
            let rss = rss.clone();
            let requests = requests.clone();
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    let Ok(mut s) = stream else { continue };
                    use std::io::Read;
                    let mut buf = [0u8; 8192];
                    let n = s.read(&mut buf).unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]);
                    requests.lock().unwrap().push(req.to_string());
                    let path = req.split_whitespace().nth(1).unwrap_or("/").to_owned();
                    let (body, ctype) = if path == "/feed.xml" {
                        (rss.clone(), "application/rss+xml")
                    } else {
                        ("<html><body>Artikel</body></html>".to_owned(), "text/html")
                    };
                    use std::io::Write;
                    let _ = s.write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    );
                }
            })
        };
        let _ = h;
        let cfg = SourceConfig::Feed {
            url: format!("{base}/feed.xml"),
            full_text: true,
            poll_interval_secs: 60,
        };
        let docs = list(&cfg).await.unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].vpath, "feed://urn:eins");
        assert_eq!(docs[0].title, "Artikel Eins");
        assert_eq!(docs[0].metadata["link"], format!("{base}/a1"));
        assert!(docs[0].modified.is_some());
        // full_text fetched the linked article, not the summary
        assert_eq!(docs[0].ocr_text.as_deref(), Some("Artikel"));
    }

    #[test]
    fn parse_rss_items() {
        let xml = r#"<?xml version="1.0"?><rss version="2.0"><channel>
            <item><title>Erster</title><link>https://f.de/1</link>
            <guid>urn:1</guid><pubDate>Tue, 03 Jun 2025 10:00:00 GMT</pubDate>
            <description>&lt;p&gt;Inhalt&lt;/p&gt;</description></item>
            <item><title></title><link></link></item>
            </channel></rss>"#;
        let doc = roxmltree::Document::parse(xml).unwrap();
        let items: Vec<_> = doc
            .descendants()
            .filter(|n| n.is_element() && n.tag_name().name() == "item")
            .collect();
        assert_eq!(items.len(), 2);
        assert_eq!(child_text(&items[0], "title").as_deref(), Some("Erster"));
        assert_eq!(child_text(&items[0], "guid").as_deref(), Some("urn:1"));
        assert!(parse_rfc_date("Tue, 03 Jun 2025 10:00:00 GMT").is_some());
    }
}
