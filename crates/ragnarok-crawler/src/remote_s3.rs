//! S3-compatible source (AWS S3, MinIO, Garage, …): the bucket inventory
//! comes from `ListObjectsV2`, originals are fetched over SigV4-signed
//! `GET` on demand — nothing is copied locally.

use crate::remote::{checked_bytes, client, resolve_env, uri_encode, RemoteDoc};
use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use ragnarok_core::{DocumentRecord, Error, Result, SourceConfig};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

type HmacSha256 = Hmac<Sha256>;

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut m = <HmacSha256 as Mac>::new_from_slice(key).expect("hmac accepts any key length");
    m.update(data);
    m.finalize().into_bytes().to_vec()
}

fn sha256_hex(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

/// One SigV4 signing pass — `service`, the signed header set and `now`
/// are parameters so the whole chain (canonical request → derived key →
/// signature) can be verified against the published AWS doc example
/// (which signs `content-type;host;x-amz-date` on the `iam` service).
#[allow(clippy::too_many_arguments)]
fn sigv4_sign(
    access_key: &str,
    secret_key: &str,
    region: &str,
    service: &str,
    method: &str,
    canonical_uri: &str,
    canonical_query: &str,
    headers: &[(String, String)],
    payload_sha: &str,
    now: DateTime<Utc>,
) -> (String /*authorization*/, String /*amz_date*/) {
    let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
    let date = now.format("%Y%m%d").to_string();
    let mut hdrs = headers.to_vec();
    hdrs.push(("x-amz-date".to_owned(), amz_date.clone()));
    hdrs.sort_by(|a, b| a.0.cmp(&b.0));
    let canonical_headers: String = hdrs.iter().map(|(k, v)| format!("{k}:{v}\n")).collect();
    let signed_headers = hdrs
        .iter()
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>()
        .join(";");
    let canonical_request = format!(
        "{method}\n{canonical_uri}\n{canonical_query}\n{canonical_headers}\n{signed_headers}\n{payload_sha}"
    );
    let scope = format!("{date}/{region}/{service}/aws4_request");
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        sha256_hex(canonical_request.as_bytes())
    );
    let k_date = hmac_sha256(format!("AWS4{secret_key}").as_bytes(), date.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    let k_signing = hmac_sha256(&k_service, b"aws4_request");
    let signature = hex::encode(hmac_sha256(&k_signing, string_to_sign.as_bytes()));
    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={access_key}/{scope}, SignedHeaders={signed_headers}, Signature={signature}"
    );
    (authorization, amz_date)
}

/// Build the request URL + canonical URI for an object operation.
fn s3_url(
    endpoint: &str,
    bucket: &str,
    key: Option<&str>,
    path_style: bool,
) -> Result<(reqwest::Url, String /*canonical_uri*/)> {
    let mut u = reqwest::Url::parse(endpoint)
        .map_err(|e| Error::Config(format!("s3 endpoint '{endpoint}': {e}")))?;
    let base_path = u.path().trim_end_matches('/').to_owned();
    if path_style {
        let mut p = format!("{base_path}/{bucket}");
        if let Some(k) = key {
            p.push('/');
            p.push_str(k);
        }
        u.set_path(&p);
        Ok((u.clone(), uri_encode(&p, false)))
    } else {
        let host = u
            .host_str()
            .ok_or_else(|| Error::Config("s3 endpoint has no host".into()))?;
        u.set_host(Some(&format!("{bucket}.{host}")))
            .map_err(|_| Error::Config("s3 virtual-host rewrite failed".into()))?;
        let p = match key {
            Some(k) => format!("{base_path}/{k}"),
            None => format!("{base_path}/"),
        };
        u.set_path(&p);
        Ok((u.clone(), uri_encode(&p, false)))
    }
}

fn host_header(u: &reqwest::Url) -> String {
    match u.port() {
        Some(p) => format!("{}:{p}", u.host_str().unwrap_or_default()),
        None => u.host_str().unwrap_or_default().to_owned(),
    }
}

/// SigV4-signed GET against the bucket; `query` is a set of
/// `(name, value)` pairs that get sorted + encoded for the signature.
async fn s3_get(
    cfg_args: &S3Args<'_>,
    key: Option<&str>,
    query: &[(String, String)],
) -> Result<reqwest::Response> {
    let (u, canonical_uri) = s3_url(cfg_args.endpoint, cfg_args.bucket, key, cfg_args.path_style)?;
    let mut qs: Vec<(String, String)> = query.to_vec();
    qs.sort();
    let canonical_query = qs
        .iter()
        .map(|(k, v)| format!("{}={}", uri_encode(k, true), uri_encode(v, true)))
        .collect::<Vec<_>>()
        .join("&");
    let payload_sha = sha256_hex(b"");
    let (authorization, amz_date) = sigv4_sign(
        cfg_args.access_key,
        cfg_args.secret_key,
        cfg_args.region,
        "s3",
        "GET",
        &canonical_uri,
        &canonical_query,
        &[
            ("host".to_owned(), host_header(&u)),
            ("x-amz-content-sha256".to_owned(), payload_sha.clone()),
        ],
        &payload_sha,
        Utc::now(),
    );
    // build the request URL as a string so the already-encoded query is
    // not percent-encoded a second time by Url::set_query
    let req_url = format!("{u}?{canonical_query}");
    client()
        .get(&req_url)
        .header("x-amz-date", amz_date)
        .header("x-amz-content-sha256", payload_sha)
        .header("Authorization", authorization)
        .send()
        .await
        .map_err(|e| Error::Internal(format!("s3 request: {e}")))
}

struct S3Args<'a> {
    endpoint: &'a str,
    bucket: &'a str,
    region: &'a str,
    path_style: bool,
    access_key: &'a str,
    secret_key: &'a str,
}

fn xml_text<'a>(node: &roxmltree::Node<'a, 'a>, tag: &str) -> Option<&'a str> {
    node.children()
        .find(|c| c.has_tag_name(tag))
        .and_then(|c| c.text())
}

/// `ListObjectsV2` over the whole (filtered) bucket — paginates via
/// continuation tokens until the server reports `IsTruncated=false`.
async fn s3_list(
    args: &S3Args<'_>,
    prefix: &str,
    include: &[String],
    exclude: &[String],
) -> Result<Vec<RemoteDoc>> {
    let mut docs = Vec::new();
    let mut token: Option<String> = None;
    loop {
        let mut q = vec![
            ("list-type".to_owned(), "2".to_owned()),
            ("max-keys".to_owned(), "1000".to_owned()),
        ];
        if !prefix.is_empty() {
            q.push(("prefix".to_owned(), prefix.to_owned()));
        }
        if let Some(t) = &token {
            q.push(("continuation-token".to_owned(), t.clone()));
        }
        let resp = s3_get(args, None, &q).await?;
        let body = checked_bytes(resp, "s3 list").await?;
        let body = String::from_utf8_lossy(&body);
        let doc = roxmltree::Document::parse(&body)
            .map_err(|e| Error::Internal(format!("s3 list xml: {e}")))?;
        for c in doc.descendants().filter(|n| n.has_tag_name("Contents")) {
            let key = match xml_text(&c, "Key") {
                Some(k) if !k.is_empty() && !k.ends_with('/') => k.to_owned(),
                _ => continue, // skip zero-size "folder" markers + empties
            };
            if !include.is_empty() && !include.iter().any(|p| crate::scanner::glob_match(p, &key)) {
                continue;
            }
            if exclude.iter().any(|p| crate::scanner::glob_match(p, &key)) {
                continue;
            }
            let etag = xml_text(&c, "ETag").map(|e| e.trim_matches('"').to_owned());
            let modified = xml_text(&c, "LastModified").and_then(|m| {
                DateTime::parse_from_rfc3339(m)
                    .map(|d| d.with_timezone(&Utc))
                    .ok()
            });
            let size = xml_text(&c, "Size").and_then(|s| s.parse::<u64>().ok());
            let title = key.rsplit('/').next().unwrap_or(&key).to_owned();
            let mut metadata = HashMap::new();
            metadata.insert("remote_key".to_owned(), key.clone());
            docs.push(RemoteDoc {
                vpath: format!("s3://{}/{key}", args.bucket),
                title,
                mime: mime_guess::from_path(&key)
                    .first_or_octet_stream()
                    .to_string(),
                modified,
                etag,
                size,
                ocr_text: None,
                metadata,
            });
        }
        let truncated = doc
            .descendants()
            .find(|n| n.has_tag_name("IsTruncated"))
            .and_then(|n| n.text())
            .map(|t| t.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if !truncated {
            break;
        }
        token = doc
            .descendants()
            .find(|n| n.has_tag_name("NextContinuationToken"))
            .and_then(|n| n.text())
            .map(str::to_owned);
        if token.is_none() {
            break; // server claims truncation but sent no token
        }
    }
    Ok(docs)
}

/// Signed `GET` for one object — used both for indexing and for the
/// on-demand `/documents/{id}/stream` path.
pub async fn s3_fetch(cfg: &SourceConfig, rec: &DocumentRecord) -> Result<(Vec<u8>, String)> {
    let SourceConfig::S3 {
        endpoint,
        bucket,
        region,
        path_style,
        access_key_env,
        secret_key_env,
        ..
    } = cfg
    else {
        return Err(Error::Config("not an s3 source".into()));
    };
    let key = rec
        .metadata
        .get("remote_key")
        .cloned()
        .or_else(|| {
            rec.path
                .strip_prefix(&format!("s3://{bucket}/"))
                .map(str::to_owned)
        })
        .ok_or_else(|| Error::Internal("s3 doc lacks remote_key".into()))?;
    let args = S3Args {
        endpoint,
        bucket,
        region,
        path_style: *path_style,
        access_key: &resolve_env(access_key_env)?,
        secret_key: &resolve_env(secret_key_env)?,
    };
    let resp = s3_get(&args, Some(&key), &[]).await?;
    let mime = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.split(';').next().unwrap_or(s).trim().to_owned())
        .filter(|m| m != "application/octet-stream" && m != "binary/octet-stream")
        .unwrap_or_else(|| {
            mime_guess::from_path(&key)
                .first_or_octet_stream()
                .to_string()
        });
    Ok((checked_bytes(resp, "s3").await?, mime))
}

pub async fn list(cfg: &SourceConfig) -> Result<Vec<RemoteDoc>> {
    let SourceConfig::S3 {
        endpoint,
        bucket,
        region,
        prefix,
        access_key_env,
        secret_key_env,
        path_style,
        include,
        exclude,
        ..
    } = cfg
    else {
        return Err(Error::Config("not an s3 source".into()));
    };
    let args = S3Args {
        endpoint,
        bucket,
        region,
        path_style: *path_style,
        access_key: &resolve_env(access_key_env)?,
        secret_key: &resolve_env(secret_key_env)?,
    };
    s3_list(&args, prefix, include, exclude).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::tests::*;

    /// AWS Signature Version 4 test suite, `get-vanilla-query-order-key-case`
    /// example adapted to the canonical `iam.amazonaws.com` documentation
    /// request — proves the full signing chain (canonical request, derived
    /// key, signature) byte-exact against the published expected signature.
    #[test]
    fn sigv4_matches_aws_example() {
        let now = DateTime::parse_from_rfc3339("2015-08-30T12:36:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let (auth, amz) = sigv4_sign(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            "us-east-1",
            "iam",
            "GET",
            "/",
            "Action=ListUsers&Version=2010-05-08",
            &[
                ("host".to_owned(), "iam.amazonaws.com".to_owned()),
                (
                    "content-type".to_owned(),
                    "application/x-www-form-urlencoded; charset=utf-8".to_owned(),
                ),
            ],
            &sha256_hex(b""),
            now,
        );
        assert_eq!(amz, "20150830T123600Z");
        assert!(
            auth.contains(
                "Signature=5d672d79c15b13162d9279b0855cfba6789a8edb4c82c400e06b5924a6f2b5d7"
            ),
            "{auth}"
        );
        assert!(auth.contains("Credential=AKIDEXAMPLE/20150830/us-east-1/iam/aws4_request"));
    }

    #[test]
    fn uri_encode_unreserved() {
        assert_eq!(uri_encode("a/b c.pdf", false), "a/b%20c.pdf");
        assert_eq!(uri_encode("übung+1", true), "%C3%BCbung%2B1");
    }

    #[tokio::test]
    async fn s3_list_paginates_and_fetches() {
        std::env::set_var("E2E_S3_AK", "ak");
        std::env::set_var("E2E_S3_SK", "sk");
        let page1 = r#"<?xml version="1.0"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>docs</Name><Prefix/><IsTruncated>true</IsTruncated>
  <NextContinuationToken>tok-2</NextContinuationToken>
  <Contents><Key>docs/report.pdf</Key><LastModified>2025-06-01T10:00:00.000Z</LastModified>
    <ETag>"e111"</ETag><Size>1200</Size></Contents>
  <Contents><Key>tmp/scratch.tmp</Key><LastModified>2025-06-01T10:00:00.000Z</LastModified>
    <ETag>"e222"</ETag><Size>10</Size></Contents>
</ListBucketResult>"#;
        let page2 = r#"<?xml version="1.0"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>docs</Name><Prefix/><IsTruncated>false</IsTruncated>
  <Contents><Key>docs/notes.md</Key><LastModified>2025-06-02T10:00:00.000Z</LastModified>
    <ETag>"e333"</ETag><Size>42</Size></Contents>
  <Contents><Key>docs/sub/</Key><LastModified>2025-06-02T10:00:00.000Z</LastModified>
    <ETag>"d41d"</ETag><Size>0</Size></Contents>
</ListBucketResult>"#;
        let mock = mock_http(vec![
            (
                "GET".into(),
                "/docs/docs/report.pdf".into(),
                (200, "application/pdf", "%PDF-fake".into()),
            ),
            (
                "GET".into(),
                "/docs?continuation-token=tok-2".into(),
                (200, "application/xml", page2.into()),
            ),
            (
                "GET".into(),
                "/docs?list-type=2".into(),
                (200, "application/xml", page1.into()),
            ),
        ]);
        let cfg = SourceConfig::S3 {
            endpoint: mock.base.clone(),
            bucket: "docs".into(),
            region: "us-east-1".into(),
            prefix: String::new(),
            access_key_env: "E2E_S3_AK".into(),
            secret_key_env: "E2E_S3_SK".into(),
            path_style: true,
            include: vec![],
            exclude: vec!["tmp/*".into()],
            poll_interval_secs: 60,
        };
        let docs = list(&cfg).await.unwrap();
        assert_eq!(docs.len(), 2, "{docs:?}");
        let report = docs.iter().find(|d| d.title == "report.pdf").unwrap();
        assert_eq!(report.vpath, "s3://docs/docs/report.pdf");
        assert_eq!(report.etag.as_deref(), Some("e111"));
        assert_eq!(report.size, Some(1200));
        assert_eq!(report.mime, "application/pdf");
        // folder marker keys (trailing /) and excluded globs are skipped
        assert!(!docs.iter().any(|d| d.vpath.contains("scratch.tmp")));
        assert!(!docs.iter().any(|d| d.vpath.ends_with('/')));
        // two list requests: second carried the continuation token
        {
            let reqs = mock.requests.lock().unwrap();
            assert!(reqs.iter().any(|r| r.contains("list-type=2")));
            assert!(
                reqs.iter().any(|r| r.contains("continuation-token=tok-2")),
                "{reqs:?}"
            );
            assert!(
                reqs.iter().all(|r| r.contains("AWS4-HMAC-SHA256")),
                "{reqs:?}"
            );
        }

        let rec = ragnarok_core::DocumentRecord {
            id: ragnarok_core::DocumentId::new(),
            source_id: ragnarok_core::SourceId::new(),
            path: report.vpath.clone(),
            hash: String::new(),
            doc_type: ragnarok_core::DocType::Unknown,
            size: 0,
            modified: None,
            indexed_at: Utc::now(),
            chunk_count: 0,
            metadata: report.metadata.clone(),
        };
        let (bytes, mime) = s3_fetch(&cfg, &rec).await.unwrap();
        assert_eq!(mime, "application/pdf");
        assert_eq!(bytes, b"%PDF-fake");
    }
}
