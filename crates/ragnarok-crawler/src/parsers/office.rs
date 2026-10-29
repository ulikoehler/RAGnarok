use super::Parser;
use ragnarok_core::{AttachmentRef, DocType, Error, ParsedDocument, Result, TextBlock};
use std::io::{Cursor, Read};
use std::path::Path;
use zip::ZipArchive;

/// OOXML office formats (`.docx`, `.xlsx`, `.pptx` — plus `.csv`, `.odt`
/// fall through to the text parser or a generic XML scrape). Extracts text
/// via zip passthrough + XML text-node collection.
///
/// Page mapping: docx → paragraph blocks (page 0), xlsx → one block per
/// sheet, pptx → one block per slide (page = slide index).
pub struct OfficeParser;

impl Parser for OfficeParser {
    fn extensions(&self) -> &'static [&'static str] {
        &["docx", "xlsx", "pptx"]
    }

    fn doc_type(&self) -> DocType {
        DocType::Word
    }

    fn parse(&self, path: &Path, bytes: &[u8]) -> Result<ParsedDocument> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let mut zip = ZipArchive::new(Cursor::new(bytes)).map_err(|e| {
            Error::parse(path.display().to_string(), format!("not a zip/ooxml: {e}"))
        })?;
        match ext.as_str() {
            "docx" => parse_docx(path, &mut zip),
            "xlsx" => parse_xlsx(path, &mut zip),
            "pptx" => parse_pptx(path, &mut zip),
            _ => Err(Error::UnsupportedType(ext)),
        }
    }
}

fn read_member<R: Read + std::io::Seek>(zip: &mut ZipArchive<R>, name: &str) -> Result<String> {
    let mut buf = String::new();
    zip.by_name(name)
        .map_err(|e| Error::parse(name, format!("member missing: {e}")))?
        .read_to_string(&mut buf)
        .map_err(|e| Error::parse(name, format!("member read: {e}")))?;
    Ok(buf)
}

/// `true` for element names in the OOXML wordprocessing/drawing/spreadsheet
/// main namespaces.
fn is_tag(node: &roxmltree::Node, local: &str, ns_fragment: &str) -> bool {
    node.is_element()
        && node.tag_name().name() == local
        && node
            .tag_name()
            .namespace()
            .is_some_and(|ns| ns.contains(ns_fragment))
}

fn parse_docx<R: Read + std::io::Seek>(
    path: &Path,
    zip: &mut ZipArchive<R>,
) -> Result<ParsedDocument> {
    let xml = read_member(zip, "word/document.xml")?;
    let tree = roxmltree::Document::parse(&xml)
        .map_err(|e| Error::parse(path.display().to_string(), e.to_string()))?;
    let mut doc = ParsedDocument::new(path.display().to_string(), DocType::Word);
    for p in tree
        .descendants()
        .filter(|n| is_tag(n, "p", "wordprocessingml"))
    {
        let mut text = String::new();
        collect_wt(&p, &mut text);
        let text = text.trim().to_string();
        if !text.is_empty() {
            doc.blocks.push(TextBlock {
                page: 0,
                bbox: None,
                line_range: None,
                text,
            });
        }
    }
    Ok(doc)
}

fn collect_wt(node: &roxmltree::Node, out: &mut String) {
    for n in node.children() {
        if is_tag(&n, "t", "wordprocessingml") {
            if let Some(t) = n.text() {
                out.push_str(t);
            }
        } else if is_tag(&n, "tab", "wordprocessingml") {
            out.push('\t');
        } else if is_tag(&n, "br", "wordprocessingml") {
            out.push('\n');
        } else {
            collect_wt(&n, out);
        }
    }
}

fn parse_xlsx<R: Read + std::io::Seek>(
    path: &Path,
    zip: &mut ZipArchive<R>,
) -> Result<ParsedDocument> {
    // shared strings
    let shared: Vec<String> = match read_member(zip, "xl/sharedStrings.xml") {
        Ok(xml) => {
            let tree = roxmltree::Document::parse(&xml)
                .map_err(|e| Error::parse(path.display().to_string(), e.to_string()))?;
            tree.descendants()
                .filter(|n| is_tag(n, "si", "spreadsheetml"))
                .map(|si| {
                    let mut s = String::new();
                    for t in si.descendants().filter(|n| is_tag(n, "t", "spreadsheetml")) {
                        if let Some(x) = t.text() {
                            s.push_str(x);
                        }
                    }
                    s
                })
                .collect()
        }
        Err(_) => Vec::new(),
    };

    let mut doc = ParsedDocument::new(path.display().to_string(), DocType::Spreadsheet);
    let sheet_names: Vec<String> = zip
        .file_names()
        .filter(|n| n.starts_with("xl/worksheets/") && n.ends_with(".xml"))
        .map(|s| s.to_string())
        .collect();
    for (page, name) in sheet_names.iter().enumerate() {
        let xml = read_member(zip, name)?;
        let tree =
            roxmltree::Document::parse(&xml).map_err(|e| Error::parse(name, e.to_string()))?;
        let mut text = String::new();
        for row in tree
            .descendants()
            .filter(|n| is_tag(n, "row", "spreadsheetml"))
        {
            for c in row.children().filter(|n| is_tag(n, "c", "spreadsheetml")) {
                let cell_type = c.attribute("t");
                if cell_type == Some("inlineStr") {
                    // inline strings carry their text in <is><t>…</t></is>
                    // (no <v> child) — skipping them loses whole sheets
                    // written by producers that omit sharedStrings.xml
                    for t in c.descendants().filter(|n| is_tag(n, "t", "spreadsheetml")) {
                        if let Some(s) = t.text() {
                            text.push_str(s.trim());
                            text.push(' ');
                        }
                    }
                    continue;
                }
                let is_shared = cell_type == Some("s");
                for v in c.children().filter(|n| is_tag(n, "v", "spreadsheetml")) {
                    if let Some(raw) = v.text() {
                        if is_shared {
                            if let Some(s) =
                                raw.trim().parse::<usize>().ok().and_then(|i| shared.get(i))
                            {
                                text.push_str(s);
                                text.push(' ');
                            }
                        } else {
                            text.push_str(raw.trim());
                            text.push(' ');
                        }
                    }
                }
            }
            text.push('\n');
        }
        let text = text.trim().to_string();
        if !text.is_empty() {
            doc.blocks.push(TextBlock {
                page: page as u32,
                bbox: None,
                line_range: None,
                text,
            });
        }
    }
    doc.metadata
        .insert("sheets".into(), doc.blocks.len().to_string());
    Ok(doc)
}

fn parse_pptx<R: Read + std::io::Seek>(
    path: &Path,
    zip: &mut ZipArchive<R>,
) -> Result<ParsedDocument> {
    let mut doc = ParsedDocument::new(path.display().to_string(), DocType::Presentation);
    let mut slide_names: Vec<String> = zip
        .file_names()
        .filter(|n| n.starts_with("ppt/slides/slide") && n.ends_with(".xml"))
        .map(|s| s.to_string())
        .collect();
    slide_names.sort_by_key(|n| {
        n.trim_start_matches("ppt/slides/slide")
            .trim_end_matches(".xml")
            .parse::<u32>()
            .unwrap_or(0)
    });
    for (page, name) in slide_names.iter().enumerate() {
        let xml = read_member(zip, name)?;
        let tree =
            roxmltree::Document::parse(&xml).map_err(|e| Error::parse(name, e.to_string()))?;
        let mut text = String::new();
        for t in tree.descendants().filter(|n| is_tag(n, "t", "drawingml")) {
            if let Some(x) = t.text() {
                text.push_str(x);
                text.push(' ');
            }
        }
        let text = text.trim().to_string();
        if !text.is_empty() {
            doc.blocks.push(TextBlock {
                page: page as u32,
                bbox: None,
                line_range: None,
                text,
            });
        }
    }
    doc.metadata
        .insert("slides".into(), doc.blocks.len().to_string());
    // speaker notes are a nice-to-have; attachment refs stay empty
    let _ = AttachmentRef {
        filename: String::new(),
        mime: String::new(),
        size: 0,
        bytes: Vec::new(),
    };
    Ok(doc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    fn make_zip(files: &[(&str, &str)]) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut w = zip::ZipWriter::new(&mut buf);
            let opts = zip::write::SimpleFileOptions::default();
            for (name, content) in files {
                w.start_file(*name, opts).unwrap();
                w.write_all(content.as_bytes()).unwrap();
            }
            w.finish().unwrap();
        }
        buf.into_inner()
    }

    const W_NS: &str = "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"";

    #[test]
    fn docx_extracts_paragraphs() {
        let document = format!(
            "<w:document {W_NS}><w:body>\
             <w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:t> world</w:t></w:r></w:p>\
             <w:p><w:r><w:t>Second para</w:t></w:r></w:p>\
             </w:body></w:document>"
        );
        let zip = make_zip(&[("word/document.xml", &document)]);
        let doc = OfficeParser.parse(&PathBuf::from("/a.docx"), &zip).unwrap();
        assert_eq!(doc.doc_type, DocType::Word);
        assert_eq!(doc.blocks.len(), 2);
        assert_eq!(doc.blocks[0].text, "Hello world");
        assert_eq!(doc.blocks[1].text, "Second para");
    }

    #[test]
    fn xlsx_extracts_shared_strings() {
        let ss = "<sst xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">\
                  <si><t>Revenue</t></si><si><t>Profit</t></si></sst>";
        let sheet =
            "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">\
                     <sheetData><row><c t=\"s\"><v>0</v></c><c><v>42</v></c></row>\
                     <row><c t=\"s\"><v>1</v></c><c><v>7</v></c></row></sheetData></worksheet>";
        let zip = make_zip(&[
            ("xl/sharedStrings.xml", ss),
            ("xl/worksheets/sheet1.xml", sheet),
        ]);
        let doc = OfficeParser.parse(&PathBuf::from("/a.xlsx"), &zip).unwrap();
        assert_eq!(doc.doc_type, DocType::Spreadsheet);
        assert!(doc.blocks[0].text.contains("Revenue"));
        assert!(doc.blocks[0].text.contains("42"));
        assert!(doc.blocks[0].text.contains("Profit"));
    }

    #[test]
    fn xlsx_extracts_inline_strings() {
        // sheets produced without sharedStrings.xml carry t="inlineStr"
        // cells — their text lives in <is><t>, not <v>
        let sheet =
            "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">\
             <sheetData><row>\
             <c t=\"inlineStr\"><is><t>Hello</t></is></c>\
             <c><v>42</v></c>\
             <c t=\"inlineStr\"><is><r><t>Rich</t></r><r><t>Text</t></r></is></c>\
             </row></sheetData></worksheet>";
        let zip = make_zip(&[("xl/worksheets/sheet1.xml", sheet)]);
        let doc = OfficeParser.parse(&PathBuf::from("/a.xlsx"), &zip).unwrap();
        assert!(doc.blocks[0].text.contains("Hello"));
        assert!(doc.blocks[0].text.contains("Rich"));
        assert!(doc.blocks[0].text.contains("42"));
    }

    #[test]
    fn pptx_extracts_slides_as_pages() {
        let mk = |t: &str| {
            format!(
                "<p:sld xmlns:p=\"p\" xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\">\
                 <p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>{t}</a:t></a:r></a:p>\
                 </p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
            )
        };
        let zip = make_zip(&[
            ("ppt/slides/slide2.xml", &mk("Second")),
            ("ppt/slides/slide1.xml", &mk("First")),
        ]);
        let doc = OfficeParser.parse(&PathBuf::from("/a.pptx"), &zip).unwrap();
        assert_eq!(doc.doc_type, DocType::Presentation);
        assert_eq!(doc.blocks.len(), 2);
        assert_eq!(doc.blocks[0].text, "First");
        assert_eq!(doc.blocks[0].page, 0);
        assert_eq!(doc.blocks[1].text, "Second");
        assert_eq!(doc.blocks[1].page, 1);
    }

    #[test]
    fn corrupt_zip_errors() {
        assert!(OfficeParser
            .parse(&PathBuf::from("/a.docx"), b"not a zip")
            .is_err());
    }
}
