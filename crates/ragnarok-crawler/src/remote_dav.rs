//! CalDAV calendar (`VEVENT`) and CardDAV addressbook (`VCARD`) source.
//! A single `REPORT` per sync lists every member *with* its iCalendar /
//! vCard payload inline, so indexing never touches the original object
//! — `/documents/{id}/stream` `GET`s the member on demand, read-only.

use crate::remote::{checked_bytes, client, resolve_env, RemoteDoc};
use chrono::Utc;
use ragnarok_core::{DocumentRecord, Error, Result, SourceConfig};
use std::collections::HashMap;

const CAL_REPORT: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<C:calendar-query xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav">
  <D:prop>
    <D:getetag/><D:getlastmodified/>
    <C:calendar-data/>
  </D:prop>
  <C:filter><C:comp-filter name="VCALENDAR"/></C:filter>
</C:calendar-query>"#;

const CARD_REPORT: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<CARD:addressbook-query xmlns:D="DAV:" xmlns:CARD="urn:ietf:params:xml:ns:carddav">
  <D:prop>
    <D:getetag/><D:getlastmodified/>
    <CARD:address-data/>
  </D:prop>
</CARD:addressbook-query>"#;

const VEVENT_FIELDS: &[&str] = &[
    "UID",
    "SUMMARY",
    "DESCRIPTION",
    "LOCATION",
    "DTSTART",
    "DTEND",
    "DTSTAMP",
    "DURATION",
    "CATEGORIES",
    "ORGANIZER",
    "ATTENDEE",
    "STATUS",
    "TRANSP",
    "URL",
    "RRULE",
    "RECURRENCE-ID",
];

const VCARD_FIELDS: &[&str] = &[
    "UID",
    "FN",
    "N",
    "NICKNAME",
    "ORG",
    "TITLE",
    "ROLE",
    "EMAIL",
    "TEL",
    "ADR",
    "URL",
    "NOTE",
    "BDAY",
    "ANNIVERSARY",
    "CATEGORIES",
];

/// iCal/vCard line unfolding: continuation lines start with one space
/// or tab; quoted-printable soft breaks (`=` + CRLF) die first.
fn unfold(data: &str) -> Vec<String> {
    let data = data.replace("=\r\n", "").replace("=\n", "");
    let mut lines: Vec<String> = Vec::new();
    for raw in data.split('\n') {
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        if line.is_empty() {
            continue;
        }
        if (line.starts_with(' ') || line.starts_with('\t')) && !lines.is_empty() {
            lines.last_mut().unwrap().push_str(&line[1..]);
        } else {
            lines.push(line.to_owned());
        }
    }
    lines
}

fn qp_decode(v: &str) -> String {
    let b = v.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'=' && i + 2 < b.len() {
            if let Ok(x) = u8::from_str_radix(&v[i + 1..i + 3], 16) {
                out.push(x);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn unescape_text(v: &str) -> String {
    let mut out = String::with_capacity(v.len());
    let mut it = v.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.next() {
                Some('n') | Some('N') => out.push(' '),
                Some(',') => out.push(','),
                Some(';') => out.push(';'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// `20250610T123000Z` → `2025-06-10 12:30:00 UTC` — the spaced form
/// tokenizes into searchable year/month/day/hour atoms.
fn humanize_dt(v: &str) -> String {
    let t = v.trim();
    let (date, time) = match t.split_once('T') {
        Some((d, t)) => (d, Some(t)),
        None => (t, None),
    };
    if date.len() == 8 && date.bytes().all(|b| b.is_ascii_digit()) {
        let mut s = format!("{}-{}-{}", &date[0..4], &date[4..6], &date[6..8]);
        if let Some(tm) = time {
            let tm = tm.trim_end_matches('Z');
            if tm.len() >= 6 && tm.bytes().all(|b| b.is_ascii_digit()) {
                s.push_str(&format!(
                    " {}:{}:{}{}",
                    &tm[0..2],
                    &tm[2..4],
                    &tm[4..6],
                    if t.ends_with('Z') { " UTC" } else { "" }
                ));
            } else {
                s.push(' ');
                s.push_str(tm);
            }
        }
        return s;
    }
    t.to_owned()
}

fn value_of(line: &str) -> Option<(String, String, String)> {
    let (head, value) = line.split_once(':')?;
    let mut parts = head.split(';');
    let mut name = parts.next()?.to_ascii_uppercase();
    if let Some((_, n)) = name.split_once('.') {
        name = n.to_owned(); // `item1.X-PROP` group prefix
    }
    Some((
        name,
        parts.collect::<Vec<_>>().join(";").to_uppercase(),
        value.to_owned(),
    ))
}

/// Extract searchable text lines from one iCalendar `VCALENDAR` or
/// vCard `VCARD` payload: `FIELD: humanized value` per kept property.
pub fn dav_data_to_text(kind: &str, data: &str) -> String {
    let fields: &[&str] = if kind == "carddav" {
        VCARD_FIELDS
    } else {
        VEVENT_FIELDS
    };
    let mut stack: Vec<String> = Vec::new();
    let mut out: Vec<String> = Vec::new();
    for line in unfold(data) {
        let Some((name, params, value)) = value_of(&line) else {
            continue;
        };
        if name == "BEGIN" {
            stack.push(value.trim().to_ascii_uppercase());
            continue;
        }
        if name == "END" {
            stack.pop();
            continue;
        }
        let target = if kind == "carddav" {
            stack.last().map(|s| s.as_str()) == Some("VCARD") || stack.is_empty()
        } else {
            stack.last().map(|s| s.as_str()) == Some("VEVENT")
        };
        if !target || !fields.contains(&name.as_str()) {
            continue;
        }
        let mut v = value.trim().to_owned();
        if params.contains("QUOTED-PRINTABLE") {
            v = qp_decode(&v);
        }
        v = unescape_text(&v);
        if name.starts_with("DT") || name == "BDAY" || name == "ANNIVERSARY" {
            v = humanize_dt(&v);
        }
        if !v.is_empty() {
            out.push(format!("{name}: {v}"));
        }
    }
    out.join("\n")
}

fn report_body(kind: &str) -> &'static str {
    if kind == "carddav" {
        CARD_REPORT
    } else {
        CAL_REPORT
    }
}

fn basic_auth(
    cfg_user: Option<&String>,
    cfg_pw_env: Option<&String>,
) -> Result<Option<(String, String)>> {
    let Some(u) = cfg_user else { return Ok(None) };
    let pw = cfg_pw_env
        .and_then(|e| {
            if e.is_empty() {
                None
            } else {
                Some(resolve_env(e))
            }
        })
        .transpose()?
        .unwrap_or_default();
    Ok(Some((u.clone(), pw)))
}

/// `REPORT` the collection — each `<response>` carries the member's
/// href, etag and inline iCalendar/vCard payload.
pub async fn dav_list(
    url: &str,
    username: Option<&String>,
    password_env: Option<&String>,
    kind: &str,
) -> Result<Vec<RemoteDoc>> {
    if kind != "caldav" && kind != "carddav" {
        return Err(Error::Config(format!(
            "cal_dav kind must be `caldav` or `carddav`, got `{kind}`"
        )));
    }
    let auth = basic_auth(username, password_env)?;
    let mut req = client()
        .request(reqwest::Method::from_bytes(b"REPORT").unwrap(), url)
        .header("Depth", "1")
        .header("Content-Type", "application/xml; charset=utf-8")
        .body(report_body(kind));
    if let Some((u, p)) = &auth {
        req = req.basic_auth(u.clone(), Some(p.clone()));
    }
    let resp = req
        .send()
        .await
        .map_err(|e| Error::Internal(format!("{kind} report {url}: {e}")))?;
    let body = checked_bytes(resp, "dav report").await?;
    let body = String::from_utf8_lossy(&body);
    let xml = roxmltree::Document::parse(&body)
        .map_err(|e| Error::Internal(format!("{kind} report xml: {e}")))?;

    // collection path for relativizing member hrefs
    let collection_path = reqwest::Url::parse(url)
        .map(|u| u.path().trim_end_matches('/').to_owned())
        .unwrap_or_default();
    let after_scheme = url.find("://").map(|p| p + 3).unwrap_or(0);
    let origin = match url[after_scheme..].find('/') {
        Some(p) => url[..after_scheme + p].to_owned(),
        None => url.trim_end_matches('/').to_owned(),
    };
    let data_tag = if kind == "carddav" {
        "address-data"
    } else {
        "calendar-data"
    };

    let mut docs = Vec::new();
    for resp_el in xml
        .descendants()
        .filter(|n| n.is_element() && n.tag_name().name() == "response")
    {
        let mut href = String::new();
        let mut etag = None;
        let mut modified = None;
        let mut data = None;
        for n in resp_el.descendants() {
            if !n.is_element() {
                continue;
            }
            match n.tag_name().name() {
                "href" if href.is_empty() => href = n.text().unwrap_or_default().to_owned(),
                "getetag" => etag = n.text().map(|t| t.trim_matches('"').to_owned()),
                "getlastmodified" => {
                    modified = n
                        .text()
                        .and_then(|t| chrono::DateTime::parse_from_rfc2822(t).ok())
                        .map(|t| t.with_timezone(&Utc));
                }
                t if t == data_tag => data = n.text().map(str::to_owned),
                _ => {}
            }
        }
        if href.is_empty() {
            continue;
        }
        let abs = if href.starts_with("http") {
            href.clone()
        } else {
            format!("{origin}{href}")
        };
        if abs.trim_end_matches('/') == url.trim_end_matches('/') {
            continue; // the collection member listing itself
        }
        let rel = href
            .strip_prefix(&collection_path)
            .map(|r| r.trim_start_matches('/').to_owned())
            .unwrap_or_else(|| href.trim_start_matches('/').to_owned());
        if rel.is_empty() {
            continue;
        }
        let text = data.as_deref().map(|d| dav_data_to_text(kind, d));
        let mut metadata = HashMap::new();
        metadata.insert("remote_url".to_owned(), abs.clone());
        metadata.insert("kind".to_owned(), kind.to_owned());
        docs.push(RemoteDoc {
            vpath: format!("{kind}://{rel}"),
            title: rel.rsplit('/').next().unwrap_or(&rel).to_owned(),
            mime: if kind == "carddav" {
                "text/vcard".into()
            } else {
                "text/calendar".into()
            },
            modified,
            etag,
            size: data.as_ref().map(|d| d.len() as u64),
            ocr_text: text.filter(|t| !t.trim().is_empty()),
            metadata,
        });
    }
    Ok(docs)
}

/// `GET` one member on demand — `remote_url` was captured at list time.
pub async fn dav_fetch(cfg: &SourceConfig, rec: &DocumentRecord) -> Result<(Vec<u8>, String)> {
    let SourceConfig::CalDav {
        username,
        password_env,
        kind,
        ..
    } = cfg
    else {
        return Err(Error::Config("not a cal_dav source".into()));
    };
    let target = rec
        .metadata
        .get("remote_url")
        .cloned()
        .ok_or_else(|| Error::Internal("dav doc lacks remote_url".into()))?;
    let auth = basic_auth(username.as_ref(), password_env.as_ref())?;
    let mut req = client().get(&target);
    if let Some((u, p)) = auth {
        req = req.basic_auth(u, Some(p));
    }
    let resp = req
        .send()
        .await
        .map_err(|e| Error::Internal(format!("{kind} fetch: {e}")))?;
    let mime = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.split(';').next().unwrap_or(s).trim().to_owned())
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| {
            if kind == "carddav" {
                "text/vcard".into()
            } else {
                "text/calendar".into()
            }
        });
    Ok((checked_bytes(resp, "dav fetch").await?, mime))
}

pub async fn list(cfg: &SourceConfig) -> Result<Vec<RemoteDoc>> {
    let SourceConfig::CalDav {
        url,
        username,
        password_env,
        kind,
        ..
    } = cfg
    else {
        return Err(Error::Config("not a cal_dav source".into()));
    };
    dav_list(url, username.as_ref(), password_env.as_ref(), kind).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::tests::*;

    const ICAL: &str = "BEGIN:VCALENDAR\r\nVERSION:2.0\r\nBEGIN:VEVENT\r\n\
        UID:evt-42@local\r\nSUMMARY:Sprint-Planung Q3\r\n\
        DESCRIPTION:Roadmap f\\, Backlog und Kapazit=C3=A4t besprechen\r\n\
        LOCATION:Raum Neptun\r\nDTSTART:20250610T123000Z\r\n\
        DTEND:20250610T133000Z\r\nCATEGORIES:team\\,planung\r\n\
        BEGIN:VALARM\r\nTRIGGER:-PT15M\r\nACTION:DISPLAY\r\n\
        DESCRIPTION:Erinnerung\r\nEND:VALARM\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n";

    const VCARD: &str = "BEGIN:VCARD\r\nVERSION:3.0\r\nUID:contact-9\r\n\
        FN;ENCODING=QUOTED-PRINTABLE:M=C3=BCller, Andreas\r\n\
        ORG:ACME GmbH\r\nTITLE:Leiter Plattform\r\n\
        EMAIL:andreas@acme.example\r\nTEL;TYPE=WORK:+49 30 12345\r\n\
        NOTE:Kennen gelernt auf der =\r\n Konferenz\r\nEND:VCARD\r\n";

    #[test]
    fn ical_text_extracts_vevent_fields() {
        let t = dav_data_to_text("caldav", ICAL);
        assert!(t.contains("UID: evt-42@local"), "{t}");
        assert!(t.contains("SUMMARY: Sprint-Planung Q3"), "{t}");
        assert!(t.contains("DESCRIPTION: Roadmap f, Backlog"), "{t}");
        assert!(t.contains("LOCATION: Raum Neptun"), "{t}");
        assert!(t.contains("DTSTART: 2025-06-10 12:30:00 UTC"), "{t}");
        assert!(t.contains("DTEND: 2025-06-10 13:30:00 UTC"), "{t}");
        // VALARM contents must not leak into the index text
        assert!(!t.contains("Erinnerung"), "{t}");
        assert!(!t.contains("TRIGGER"), "{t}");
    }

    #[test]
    fn vcard_text_decodes_qp_and_folds() {
        let t = dav_data_to_text("carddav", VCARD);
        assert!(t.contains("FN: Müller, Andreas"), "{t}");
        assert!(t.contains("ORG: ACME GmbH"), "{t}");
        assert!(t.contains("TEL: +49 30 12345"), "{t}");
        // QP soft-break inside NOTE is joined
        assert!(
            t.contains("auf der  Konferenz") || t.contains("auf der Konferenz"),
            "{t}"
        );
    }

    fn cfg(base: &str, kind: &str) -> SourceConfig {
        SourceConfig::CalDav {
            url: format!("{base}/dav/user/{kind}/"),
            username: Some("alice".into()),
            password_env: Some("E2E_DAV_PW".into()),
            kind: kind.into(),
            poll_interval_secs: 60,
        }
    }

    #[tokio::test]
    async fn caldav_report_lists_and_fetches() {
        std::env::set_var("E2E_DAV_PW", "pw");
        let multistatus = format!(
            r#"<?xml version="1.0"?><d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
              <d:response><d:href>/dav/user/caldav/evt-42.ics</d:href>
                <d:propstat><d:prop><d:getetag>"tag-1"</d:getetag>
                <d:getlastmodified>Tue, 03 Jun 2025 10:00:00 GMT</d:getlastmodified>
                <c:calendar-data>{}</c:calendar-data></d:prop></d:propstat></d:response>
              <d:response><d:href>/dav/user/caldav/evt-77.ics</d:href>
                <d:propstat><d:prop><d:getetag>"tag-2"</d:getetag></d:prop>
                <d:status>HTTP/1.1 404 Not Found</d:status></d:propstat></d:response>
            </d:multistatus>"#,
            ICAL.replace('\n', "&#10;")
        );
        let mock = mock_http(vec![
            (
                "REPORT".into(),
                "/dav/user/caldav/".into(),
                (207, "application/xml", multistatus),
            ),
            (
                "GET".into(),
                "/dav/user/caldav/evt-42.ics".into(),
                (200, "text/calendar", ICAL.to_string()),
            ),
        ]);
        let c = cfg(&mock.base, "caldav");
        let docs = list(&c).await.unwrap();
        // 404-propstat entry skipped (no calendar-data → still listed? we
        // keep it — etag exists, payload fetched on demand)
        let evt = docs.iter().find(|d| d.title == "evt-42.ics").unwrap();
        assert_eq!(evt.vpath, "caldav://evt-42.ics");
        assert_eq!(evt.etag.as_deref(), Some("tag-1"));
        assert_eq!(evt.mime, "text/calendar");
        let text = evt.ocr_text.as_deref().unwrap();
        assert!(text.contains("SUMMARY: Sprint-Planung Q3"));
        {
            let reqs = mock.requests.lock().unwrap();
            assert!(
                reqs.iter()
                    .any(|r| r.starts_with("REPORT") && r.contains("Basic")),
                "{reqs:?}"
            );
        }

        let rec = DocumentRecord {
            id: ragnarok_core::DocumentId::new(),
            source_id: ragnarok_core::SourceId::new(),
            path: evt.vpath.clone(),
            hash: String::new(),
            doc_type: ragnarok_core::DocType::Unknown,
            size: 0,
            modified: None,
            indexed_at: Utc::now(),
            chunk_count: 0,
            metadata: evt.metadata.clone(),
        };
        let (bytes, mime) = dav_fetch(&c, &rec).await.unwrap();
        assert_eq!(mime, "text/calendar");
        assert!(String::from_utf8_lossy(&bytes).contains("Sprint-Planung"));
    }

    #[tokio::test]
    async fn carddav_report_lists_vcards() {
        std::env::set_var("E2E_DAV_PW", "pw");
        let multistatus = format!(
            r#"<?xml version="1.0"?><d:multistatus xmlns:d="DAV:" xmlns:card="urn:ietf:params:xml:ns:carddav">
              <d:response><d:href>/dav/user/carddav/contact-9.vcf</d:href>
                <d:propstat><d:prop><d:getetag>"v9"</d:getetag>
                <card:address-data>{}</card:address-data></d:prop></d:propstat></d:response>
            </d:multistatus>"#,
            VCARD.replace('\n', "&#10;")
        );
        let mock = mock_http(vec![(
            "REPORT".into(),
            "/dav/user/carddav/".into(),
            (207, "application/xml", multistatus),
        )]);
        let docs = list(&cfg(&mock.base, "carddav")).await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].vpath, "carddav://contact-9.vcf");
        assert_eq!(docs[0].mime, "text/vcard");
        assert!(docs[0].ocr_text.as_deref().unwrap().contains("Müller"));
    }

    #[tokio::test]
    async fn invalid_kind_rejected() {
        let c = SourceConfig::CalDav {
            url: "http://x/".into(),
            username: None,
            password_env: None,
            kind: "webdav".into(),
            poll_interval_secs: 60,
        };
        let e = list(&c).await.unwrap_err().to_string();
        assert!(e.contains("caldav` or `carddav"), "{e}");
    }
    #[tokio::test]
    async fn dav_list_surfaces_error_status() {
        std::env::set_var("E2E_DAV_PW", "pw");
        let mock = mock_http(vec![(
            "REPORT".into(),
            "/dav/user/caldav/".into(),
            (507, "application/xml", "insufficient storage".into()),
        )]);
        assert!(list(&cfg(&mock.base, "caldav")).await.is_err());
    }

    #[tokio::test]
    async fn dav_list_unauthorized_and_unreachable() {
        std::env::set_var("E2E_DAV_PW", "pw");
        let mock = mock_http(vec![(
            "REPORT".into(),
            "/dav/user/caldav/".into(),
            (401, "text/plain", "auth required".into()),
        )]);
        assert!(list(&cfg(&mock.base, "caldav")).await.is_err());
        assert!(list(&cfg("http://127.0.0.1:1", "caldav")).await.is_err());
    }

    #[tokio::test]
    async fn dav_fetch_surfaces_not_found() {
        std::env::set_var("E2E_DAV_PW", "pw");
        let mock = mock_http(vec![]); // every route → 404
        let rec = DocumentRecord {
            id: ragnarok_core::DocumentId::new(),
            source_id: ragnarok_core::SourceId::new(),
            path: format!("{}/dav/user/caldav/gone.ics", mock.base),
            hash: String::new(),
            doc_type: ragnarok_core::DocType::Unknown,
            size: 0,
            modified: None,
            indexed_at: chrono::Utc::now(),
            chunk_count: 0,
            metadata: std::collections::HashMap::from([(
                "dav_kind".to_owned(),
                "caldav".to_owned(),
            )]),
        };
        assert!(dav_fetch(&cfg(&mock.base, "caldav"), &rec).await.is_err());
    }
}
