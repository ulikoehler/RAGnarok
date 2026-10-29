//! eBook + OpenDocument parsers: `.epub` (zip → OPF spine → XHTML) and
//! `.odt/.ods/.odp` (zip → `content.xml` paragraph extraction).

use super::{ParsedDocument, Parser};
use ragnarok_core::model::TextBlock;
use ragnarok_core::{DocType, Result};
use std::io::{Cursor, Read};
use std::path::Path;

pub struct EbookParser;

fn zip_member(bytes: &[u8], name: &str) -> Option<Vec<u8>> {
    let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).ok()?;
    let mut f = zip.by_name(name).ok()?;
    let mut buf = Vec::with_capacity(f.size() as usize);
    f.read_to_end(&mut buf).ok()?;
    Some(buf)
}

/// All zip member names (for OPF-href resolution).
fn zip_names(bytes: &[u8]) -> Vec<String> {
    zip::ZipArchive::new(Cursor::new(bytes))
        .map(|z| z.file_names().map(str::to_owned).collect())
        .unwrap_or_default()
}

fn opf_path(bytes: &[u8]) -> Option<String> {
    let container = zip_member(bytes, "META-INF/container.xml")?;
    let doc = roxmltree::Document::parse(std::str::from_utf8(&container).ok()?).ok()?;
    doc.descendants()
        .find(|n| n.has_tag_name("rootfile"))
        .and_then(|n| n.attribute("full-path").map(str::to_owned))
}

fn parse_epub(path: &Path, bytes: &[u8]) -> Result<ParsedDocument> {
    let opf_rel = opf_path(bytes)
        .ok_or_else(|| ragnarok_core::Error::parse(path.display().to_string(), "epub: no OPF"))?;
    let opf_bytes = zip_member(bytes, &opf_rel).ok_or_else(|| {
        ragnarok_core::Error::parse(path.display().to_string(), "epub: OPF unreadable")
    })?;
    let opf = roxmltree::Document::parse(std::str::from_utf8(&opf_bytes).map_err(|e| {
        ragnarok_core::Error::parse(path.display().to_string(), format!("epub OPF utf8: {e}"))
    })?)
    .map_err(|e| {
        ragnarok_core::Error::parse(path.display().to_string(), format!("epub OPF: {e}"))
    })?;

    let mut doc = ParsedDocument::new(path.display().to_string(), DocType::Text);
    doc.metadata.insert("format".into(), "epub".into());
    // Dublin Core metadata
    for n in opf.descendants() {
        match n.tag_name().name() {
            "title" => {
                if let Some(t) = n.text() {
                    doc.metadata.insert("title".into(), t.trim().to_owned());
                }
            }
            "creator" => {
                if let Some(t) = n.text() {
                    doc.metadata.insert("author".into(), t.trim().to_owned());
                }
            }
            "language" => {
                if let Some(t) = n.text() {
                    doc.metadata.insert("language".into(), t.trim().to_owned());
                }
            }
            _ => {}
        }
    }
    // manifest: id → href; spine: ordered idrefs
    let base = opf_rel.rsplit('/').skip(1).collect::<Vec<_>>().join("/");
    let mut manifest = std::collections::HashMap::new();
    let mut spine: Vec<String> = Vec::new();
    for n in opf.descendants() {
        if n.has_tag_name("item") {
            if let (Some(id), Some(href)) = (n.attribute("id"), n.attribute("href")) {
                manifest.insert(id.to_owned(), href.to_owned());
            }
        } else if n.has_tag_name("itemref") {
            if let Some(idref) = n.attribute("idref") {
                spine.push(idref.to_owned());
            }
        }
    }
    let names: std::collections::HashSet<String> = zip_names(bytes).into_iter().collect();
    let mut page = 0u32;
    for idref in &spine {
        let Some(href) = manifest.get(idref) else {
            continue;
        };
        let full = if base.is_empty() {
            href.clone()
        } else {
            format!("{base}/{href}")
        };
        // href may be URL-encoded or nested differently — fall back to
        // a suffix match across member names
        let member = if names.contains(&full) {
            Some(full)
        } else {
            names.iter().find(|n| n.ends_with(href.as_str())).cloned()
        };
        let Some(member) = member else { continue };
        let Some(xhtml) = zip_member(bytes, &member) else {
            continue;
        };
        let text = crate::remote::html_to_text(std::str::from_utf8(&xhtml).unwrap_or_default());
        if text.trim().is_empty() {
            continue;
        }
        doc.blocks.push(TextBlock {
            page,
            bbox: None,
            line_range: None,
            text,
        });
        page += 1;
    }
    Ok(doc)
}

fn parse_odf(path: &Path, bytes: &[u8]) -> Result<ParsedDocument> {
    let content = zip_member(bytes, "content.xml").ok_or_else(|| {
        ragnarok_core::Error::parse(path.display().to_string(), "odf: no content.xml")
    })?;
    let xml = roxmltree::Document::parse(std::str::from_utf8(&content).map_err(|e| {
        ragnarok_core::Error::parse(path.display().to_string(), format!("odf utf8: {e}"))
    })?)
    .map_err(|e| ragnarok_core::Error::parse(path.display().to_string(), format!("odf: {e}")))?;
    let mut doc = ParsedDocument::new(path.display().to_string(), DocType::Word);
    doc.metadata.insert("format".into(), "odf".into());
    // text:h / text:p blocks; table cells join with " | "
    for n in xml.descendants() {
        let tag = n.tag_name().name();
        if tag == "h" || tag == "p" {
            let text: String = n
                .descendants()
                .filter(|d| d.is_text())
                .filter_map(|d| d.text())
                .collect::<Vec<_>>()
                .join(" ");
            let t = text.split_whitespace().collect::<Vec<_>>().join(" ");
            if !t.is_empty() {
                doc.blocks.push(TextBlock {
                    page: 0,
                    bbox: None,
                    line_range: None,
                    text: t,
                });
            }
        }
    }
    Ok(doc)
}

impl Parser for EbookParser {
    fn extensions(&self) -> &'static [&'static str] {
        &["epub", "odt", "ods", "odp"]
    }

    fn doc_type(&self) -> DocType {
        DocType::Text
    }

    fn parse(&self, path: &Path, bytes: &[u8]) -> Result<ParsedDocument> {
        match path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase()
            .as_str()
        {
            "epub" => parse_epub(path, bytes),
            _ => parse_odf(path, bytes),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn zip_of(files: &[(&str, &str)]) -> Vec<u8> {
        let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let opts = zip::write::SimpleFileOptions::default();
        for (n, c) in files {
            w.start_file(*n, opts).unwrap();
            w.write_all(c.as_bytes()).unwrap();
        }
        w.finish().unwrap().into_inner()
    }

    #[test]
    fn epub_spine_blocks_with_dc_metadata() {
        let z = zip_of(&[
            (
                "META-INF/container.xml",
                r#"<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>"#,
            ),
            (
                "OEBPS/content.opf",
                r#"<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/"><metadata><dc:title>Testbuch</dc:title><dc:creator>Autorin</dc:creator></metadata><manifest><item id="c1" href="kap1.xhtml" media-type="application/xhtml+xml"/><item id="c2" href="kap2.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="c1"/><itemref idref="c2"/></spine></package>"#,
            ),
            (
                "OEBPS/kap1.xhtml",
                "<html><body><p>Kapitel eins Text</p></body></html>",
            ),
            (
                "OEBPS/kap2.xhtml",
                "<html><body><p>Kapitel zwei walrus</p></body></html>",
            ),
        ]);
        let doc = EbookParser.parse(Path::new("buch.epub"), &z).unwrap();
        assert_eq!(doc.metadata["title"], "Testbuch");
        assert_eq!(doc.metadata["author"], "Autorin");
        assert_eq!(doc.blocks.len(), 2);
        assert!(doc.blocks[1].text.contains("walrus"));
    }

    #[test]
    fn odt_paragraphs_extract() {
        let z = zip_of(&[(
            "content.xml",
            r#"<?xml version="1.0"?><office:document xmlns:office="o" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><office:body><office:text><text:h>Überschrift</text:h><text:p>Absatz mit   Leerzeichen</text:p></office:text></office:body></office:document>"#,
        )]);
        let doc = EbookParser.parse(Path::new("doc.odt"), &z).unwrap();
        assert_eq!(doc.blocks.len(), 2);
        assert_eq!(doc.doc_type, DocType::Word);
        assert!(doc.blocks[1].text.contains("Leerzeichen"));
    }
}
