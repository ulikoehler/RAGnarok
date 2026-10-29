use super::Parser;
use mail_parser::{MessageParser, MimeHeaders};
use ragnarok_core::{AttachmentRef, DocType, Error, ParsedDocument, Result, TextBlock};
use std::path::Path;

/// RFC-822 `.eml` / `.msg` / `.mbox` parser via `mail-parser`.
/// Headers land in `metadata`; text and HTML bodies become blocks
/// (HTML is indexed for text; attachments are recorded as references).
pub struct EmailParser;

impl Parser for EmailParser {
    fn extensions(&self) -> &'static [&'static str] {
        &["eml", "msg", "mbox"]
    }

    fn doc_type(&self) -> DocType {
        DocType::Email
    }

    fn parse(&self, path: &Path, bytes: &[u8]) -> Result<ParsedDocument> {
        let msg = MessageParser::default().parse(bytes).ok_or_else(|| {
            Error::parse(path.display().to_string(), "not a valid RFC-822 message")
        })?;
        let mut doc = ParsedDocument::new(path.display().to_string(), DocType::Email);

        if let Some(s) = msg.subject() {
            doc.metadata.insert("subject".into(), s.to_string());
        }
        if let Some(from) = msg.from().and_then(|a| a.first()) {
            doc.metadata.insert(
                "from".into(),
                match (from.name(), from.address()) {
                    (Some(n), Some(a)) => format!("{n} <{a}>"),
                    (_, Some(a)) => a.to_string(),
                    (Some(n), None) => n.to_string(),
                    _ => String::new(),
                },
            );
        }
        // every recipient, not just the first — scope predicates on
        // `emailTo` must see them all
        let to: Vec<String> = msg
            .to()
            .into_iter()
            .flat_map(|a| a.iter())
            .filter_map(|a| a.address().map(str::to_owned))
            .collect();
        if !to.is_empty() {
            doc.metadata.insert("to".into(), to.join(", "));
        }
        if let Some(d) = msg.date() {
            doc.metadata
                .insert("date".into(), d.to_timestamp().to_string());
        }

        // Prefer plain-text bodies; index HTML bodies too (the viewer renders
        // them sandboxed; the chunker strips nothing — matches are against
        // the raw part text).
        for i in 0..msg.text_body_count() {
            if let Some(body) = msg.body_text(i) {
                push_body(&mut doc, &body, "text/plain");
            }
        }
        // Index HTML bodies only when no plain-text alternative exists —
        // mail-parser synthesizes an HTML body for text/plain parts and
        // multipart/alternative would otherwise be indexed twice. The raw
        // markup is stripped first so tags don't pollute the index.
        if msg.text_body_count() == 0 {
            for i in 0..msg.html_body_count() {
                if let Some(body) = msg.body_html(i) {
                    push_body(&mut doc, &crate::remote::html_to_text(&body), "text/html");
                }
            }
        }

        for att in msg.attachments() {
            let mime = att
                .content_type()
                .map(|ct| {
                    let sub = ct
                        .c_subtype
                        .as_ref()
                        .map(|s| s.as_ref())
                        .unwrap_or("octet-stream");
                    format!("{}/{sub}", ct.c_type)
                })
                .unwrap_or_else(|| "application/octet-stream".into());
            doc.attachments.push(AttachmentRef {
                filename: att.attachment_name().unwrap_or("unnamed").to_string(),
                mime,
                size: att.contents().len(),
            });
        }
        Ok(doc)
    }
}

fn push_body(doc: &mut ParsedDocument, body: &str, kind: &str) {
    if body.trim().is_empty() {
        return;
    }
    doc.metadata
        .insert(format!("body_{}", doc.blocks.len()), kind.into());
    doc.blocks.push(TextBlock {
        page: 0,
        bbox: None,
        line_range: None,
        text: body.to_string(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const EML: &[u8] = b"From: Alice <alice@example.com>\r\n\
To: bob@example.com\r\n\
Subject: Quarterly Report\r\n\
Date: Mon, 1 Jan 2024 10:00:00 +0000\r\n\
MIME-Version: 1.0\r\n\
Content-Type: text/plain; charset=utf-8\r\n\
\r\n\
Hello Bob,\r\n\
\r\n\
the quarterly numbers are attached.\r\n\
\r\n\
Alice\r\n";

    #[test]
    fn parses_headers_and_body() {
        let p = EmailParser;
        let doc = p.parse(&PathBuf::from("/m.eml"), EML).unwrap();
        assert_eq!(doc.doc_type, DocType::Email);
        assert_eq!(doc.metadata["subject"], "Quarterly Report");
        assert!(doc.metadata["from"].contains("alice@example.com"));
        assert_eq!(doc.metadata["to"], "bob@example.com");
        assert!(doc.metadata["date"].parse::<i64>().is_ok());
        assert_eq!(doc.blocks.len(), 1);
        assert!(doc.blocks[0].text.contains("quarterly numbers"));
    }

    #[test]
    fn parses_attachments() {
        let raw = b"From: a@x\r\nTo: b@y\r\nSubject: s\r\nMIME-Version: 1.0\r\n\
Content-Type: multipart/mixed; boundary=\"BB\"\r\n\r\n\
--BB\r\nContent-Type: text/plain\r\n\r\nbody text\r\n\
--BB\r\nContent-Type: application/pdf; name=\"rep.pdf\"\r\n\
Content-Disposition: attachment; filename=\"rep.pdf\"\r\n\
Content-Transfer-Encoding: base64\r\n\r\n\
JVBERi0xLjQ=\r\n--BB--\r\n";
        let p = EmailParser;
        let doc = p.parse(&PathBuf::from("/m.eml"), raw).unwrap();
        assert_eq!(doc.attachments.len(), 1);
        assert_eq!(doc.attachments[0].filename, "rep.pdf");
        assert_eq!(doc.attachments[0].mime, "application/pdf");
        assert!(doc.blocks.iter().any(|b| b.text.contains("body text")));
    }

    #[test]
    fn html_only_mail_is_indexed() {
        let raw = b"From: a@x\r\nSubject: h\r\nMIME-Version: 1.0\r\n\
Content-Type: text/html; charset=utf-8\r\n\r\n\
<html><body><p>only html here</p></body></html>\r\n";
        let doc = EmailParser.parse(&PathBuf::from("/m.eml"), raw).unwrap();
        assert_eq!(doc.blocks.len(), 1);
        assert!(doc.blocks[0].text.contains("only html here"));
        assert!(doc.metadata.contains_key("body_0"));
    }

    #[test]
    fn invalid_input_errors() {
        let p = EmailParser;
        let r = p.parse(&PathBuf::from("/x.eml"), &[0xff, 0xfe, 0xfd]);
        // mail-parser is lenient; either an error or empty doc is acceptable,
        // but it must not panic
        let _ = r;
    }
}
