//! Archive parser: `.zip` members surface as `AttachmentRef`s so the
//! pipeline ingests each as a child document (`{zip}#att:{member}`).
//! The archive itself indexes a member listing block. Members beyond
//! `MAX_MEMBER_BYTES` are listed but not extracted (zip-bomb cap).

use super::{ParsedDocument, Parser};
use ragnarok_core::model::AttachmentRef;
use ragnarok_core::model::TextBlock;
use ragnarok_core::{DocType, Result};
use std::io::{Cursor, Read};
use std::path::Path;

/// Per-member payload cap — archives can hide arbitrarily big files.
const MAX_MEMBER_BYTES: u64 = 64 * 1024 * 1024;
const MAX_MEMBERS: usize = 512;

pub struct ArchiveParser;

impl Parser for ArchiveParser {
    fn extensions(&self) -> &'static [&'static str] {
        &["zip"]
    }

    fn doc_type(&self) -> DocType {
        DocType::Unknown
    }

    fn parse(&self, path: &Path, bytes: &[u8]) -> Result<ParsedDocument> {
        let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).map_err(|e| {
            ragnarok_core::Error::parse(path.display().to_string(), format!("zip: {e}"))
        })?;
        let mut doc = ParsedDocument::new(path.display().to_string(), DocType::Unknown);
        let mut listing = String::from("ZIP-Inhalt:\n");
        let n = zip.len().min(MAX_MEMBERS);
        for i in 0..n {
            let mut entry = match zip.by_index(i) {
                Ok(e) => e,
                Err(_) => continue, // broken member → skip individually
            };
            let name = entry.name().to_owned();
            if entry.is_dir() {
                continue;
            }
            let size = entry.size();
            listing.push_str(&format!("- {name} ({size} B)\n"));
            if size > MAX_MEMBER_BYTES {
                continue;
            }
            let mut buf = Vec::with_capacity(size as usize);
            if entry.read_to_end(&mut buf).is_err() {
                continue;
            }
            doc.attachments.push(AttachmentRef {
                filename: name.clone(),
                mime: mime_for_name(&name),
                size: buf.len(),
                bytes: buf,
            });
        }
        if zip.len() > MAX_MEMBERS {
            listing.push_str(&format!(
                "… und {} weitere Einträge\n",
                zip.len() - MAX_MEMBERS
            ));
        }
        doc.blocks.push(TextBlock {
            page: 0,
            bbox: None,
            line_range: None,
            text: listing,
        });
        doc.metadata
            .insert("member_count".into(), zip.len().to_string());
        Ok(doc)
    }
}

fn mime_for_name(name: &str) -> String {
    let ext = name
        .rsplit('.')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    match ext.as_str() {
        "pdf" => "application/pdf",
        "txt" | "log" => "text/plain",
        "md" | "markdown" => "text/markdown",
        "csv" => "text/csv",
        "json" | "jsonl" => "application/json",
        "html" | "htm" => "text/html",
        "eml" => "message/rfc822",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        _ => "application/octet-stream",
    }
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_zip(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let opts = zip::write::SimpleFileOptions::default();
        for (name, data) in files {
            w.start_file(*name, opts).unwrap();
            w.write_all(data).unwrap();
        }
        w.finish().unwrap().into_inner()
    }

    #[test]
    fn zip_members_become_attachments() {
        let z = make_zip(&[
            ("rechnung.txt", b"Attachment content invoice 4711"),
            ("ordner/", b""),
            ("notizen.md", b"# Mitglied"),
        ]);
        let doc = ArchiveParser.parse(Path::new("archiv.zip"), &z).unwrap();
        let names: Vec<&str> = doc
            .attachments
            .iter()
            .map(|a| a.filename.as_str())
            .collect();
        assert!(names.contains(&"rechnung.txt"));
        assert!(names.contains(&"notizen.md"));
        assert!(doc.blocks[0].text.contains("ZIP-Inhalt"));
        assert_eq!(doc.metadata["member_count"], "3");
    }
}
