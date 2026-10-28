use super::Parser;
use ragnarok_core::{DocType, ParsedDocument, Result, TextBlock};
use std::path::Path;

/// Plain text / code / markdown: paragraphs become blocks with exact
/// 0-indexed line ranges for the text viewer's line highlighting.
pub struct TextParser;

impl Parser for TextParser {
    fn extensions(&self) -> &'static [&'static str] {
        &[
            "txt", "text", "log", "md", "markdown", "rs", "py", "js", "ts", "c", "cpp", "h", "hpp",
            "java", "go", "rb", "sh", "toml", "json", "yaml", "yml", "xml", "sql", "css", "csv",
            "html", "htm", "rtf", "odt",
        ]
    }

    fn doc_type(&self) -> DocType {
        DocType::Text
    }

    fn parse(&self, path: &Path, bytes: &[u8]) -> Result<ParsedDocument> {
        let text = String::from_utf8_lossy(bytes);
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let doc_type = DocType::from_extension(&ext);
        let mut doc = ParsedDocument::new(path.display().to_string(), doc_type);

        let mut block_start: Option<u32> = None;
        let mut buf = String::new();
        for (i, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                if let Some(start) = block_start.take() {
                    doc.blocks.push(TextBlock {
                        page: 0,
                        bbox: None,
                        line_range: Some((start, i as u32)),
                        text: buf.trim_end().to_string(),
                    });
                    buf.clear();
                }
            } else {
                if block_start.is_none() {
                    block_start = Some(i as u32);
                }
                buf.push_str(line);
                buf.push('\n');
            }
        }
        if let Some(start) = block_start {
            doc.blocks.push(TextBlock {
                page: 0,
                bbox: None,
                line_range: Some((start, text.lines().count() as u32)),
                text: buf.trim_end().to_string(),
            });
        }
        doc.metadata
            .insert("lines".into(), text.lines().count().to_string());
        Ok(doc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn paragraphs_have_line_ranges() {
        let p = TextParser;
        let doc = p
            .parse(
                &PathBuf::from("/a.txt"),
                b"first para\nstill first\n\nsecond para\n",
            )
            .unwrap();
        assert_eq!(doc.blocks.len(), 2);
        assert_eq!(doc.blocks[0].line_range, Some((0, 2)));
        assert_eq!(doc.blocks[1].line_range, Some((3, 4)));
        assert_eq!(doc.blocks[0].text, "first para\nstill first");
    }

    #[test]
    fn empty_and_whitespace() {
        let p = TextParser;
        assert!(p
            .parse(&PathBuf::from("/a.txt"), b"")
            .unwrap()
            .blocks
            .is_empty());
        assert!(p
            .parse(&PathBuf::from("/a.txt"), b"\n\n  \n")
            .unwrap()
            .blocks
            .is_empty());
    }

    #[test]
    fn doc_type_from_extension() {
        let p = TextParser;
        let doc = p.parse(&PathBuf::from("/a.rs"), b"fn main(){}").unwrap();
        assert_eq!(doc.doc_type, DocType::Code);
        let doc = p.parse(&PathBuf::from("/a.md"), b"# hi").unwrap();
        assert_eq!(doc.doc_type, DocType::Markdown);
    }
}
