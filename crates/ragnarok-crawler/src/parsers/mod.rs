//! Document parsers dispatching by extension. Each parser produces a
//! `ParsedDocument` whose `TextBlock`s carry page/bbox/line offsets for the
//! universal viewer's highlighting overlay.

mod email;
mod office;
mod pdf;
mod text;

pub use email::EmailParser;
pub use office::OfficeParser;
pub use pdf::PdfParser;
pub use text::TextParser;

use ragnarok_core::{DocType, Error, ParsedDocument, Result};
use std::path::Path;

/// One format-specific parser implementation.
pub trait Parser: Send + Sync {
    /// Lowercase extensions this parser handles (without dot).
    fn extensions(&self) -> &'static [&'static str];
    fn doc_type(&self) -> DocType;
    /// Parse raw file bytes into structural blocks.
    fn parse(&self, path: &Path, bytes: &[u8]) -> Result<ParsedDocument>;
}

/// All built-in parsers.
pub fn default_registry() -> Vec<Box<dyn Parser>> {
    vec![
        Box::new(PdfParser),
        Box::new(EmailParser),
        Box::new(OfficeParser),
        Box::new(TextParser),
    ]
}

/// Dispatch `bytes` to the right parser. UTF-8 text of unknown extension
/// falls back to the text parser; anything else errors.
pub fn parse_with(
    registry: &[Box<dyn Parser>],
    path: &Path,
    bytes: &[u8],
) -> Result<ParsedDocument> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    for p in registry {
        if p.extensions().contains(&ext.as_str()) {
            return p.parse(path, bytes);
        }
    }
    // heuristic fallback: decodable UTF-8 -> treat as text
    if std::str::from_utf8(bytes).is_ok() {
        return TextParser.parse(path, bytes);
    }
    Err(Error::UnsupportedType(format!(
        "{} ({})",
        path.display(),
        ext
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn dispatch_by_extension_and_utf8_fallback() {
        let reg = default_registry();
        // known extension
        let d = parse_with(&reg, &PathBuf::from("/a.txt"), b"hello world").unwrap();
        assert_eq!(d.doc_type, DocType::Text);
        // unknown extension but utf8 -> text fallback (doc_type Unknown)
        let d = parse_with(&reg, &PathBuf::from("/a.weird"), b"plain utf8").unwrap();
        assert!(d.blocks.iter().any(|b| b.text.contains("plain utf8")));
        // binary non-utf8 unknown -> unsupported
        let r = parse_with(&reg, &PathBuf::from("/a.bin"), &[0xff, 0xfe, 0x00]);
        assert!(matches!(r, Err(Error::UnsupportedType(_))));
        // no extension at all, utf8 -> text fallback
        let d = parse_with(&reg, &PathBuf::from("/Makefile"), b"all:").unwrap();
        assert!(d.blocks.iter().any(|b| b.text.contains("all:")));
    }
}
