//! Tabular + line-delimited parsers: CSV/TSV rows become `col: val`
//! text blocks (headers name the columns), JSONL emits one block per
//! JSON line (objects flattened to `key: value` pairs).

use super::{ParsedDocument, Parser};
use ragnarok_core::model::TextBlock;
use ragnarok_core::{DocType, Result};
use std::path::Path;

pub struct TabularParser;

fn split_row(line: &str, sep: char) -> Vec<String> {
    // minimal quoting support: `"a,b"` keeps the comma, `""` → `"`
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut q = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if q && chars.peek() == Some(&'"') => {
                cur.push('"');
                chars.next();
            }
            '"' => q = !q,
            c if c == sep && !q => {
                out.push(cur.trim().to_owned());
                cur.clear();
            }
            c => cur.push(c),
        }
    }
    out.push(cur.trim().to_owned());
    out
}

fn flatten_json(v: &serde_json::Value, prefix: &str, out: &mut Vec<String>) {
    match v {
        serde_json::Value::Object(m) => {
            for (k, val) in m {
                let key = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                flatten_json(val, &key, out);
            }
        }
        serde_json::Value::Array(items) => {
            for (i, val) in items.iter().enumerate() {
                flatten_json(val, &format!("{prefix}[{i}]"), out);
            }
        }
        other => out.push(format!("{prefix}: {}", scalar(other))),
    }
}

fn scalar(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => "null".into(),
        other => other.to_string(),
    }
}

impl Parser for TabularParser {
    fn extensions(&self) -> &'static [&'static str] {
        &["csv", "tsv", "jsonl", "ndjson"]
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
        let mut doc = ParsedDocument::new(path.display().to_string(), DocType::Text);
        doc.metadata.insert("format".into(), ext.clone());

        if ext == "jsonl" || ext == "ndjson" {
            for (i, line) in text.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let block = match serde_json::from_str::<serde_json::Value>(line) {
                    Ok(v) => {
                        let mut pairs = Vec::new();
                        flatten_json(&v, "", &mut pairs);
                        pairs.join("\n")
                    }
                    Err(_) => line.to_owned(), // malformed line indexed verbatim
                };
                doc.blocks.push(TextBlock {
                    page: 0,
                    bbox: None,
                    line_range: Some((i as u32, i as u32)),
                    text: block,
                });
            }
            return Ok(doc);
        }

        let sep = if ext == "tsv" { '\t' } else { ',' };
        let mut lines = text.lines();
        let header: Vec<String> = lines.next().map(|l| split_row(l, sep)).unwrap_or_default();
        doc.metadata.insert("columns".into(), header.join("|"));
        for (i, line) in lines.enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let cells = split_row(line, sep);
            let block = cells
                .iter()
                .enumerate()
                .map(|(j, c)| {
                    let col = header.get(j).filter(|h| !h.is_empty());
                    match col {
                        Some(h) => format!("{h}: {c}"),
                        None => c.clone(),
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            doc.blocks.push(TextBlock {
                page: 0,
                bbox: None,
                line_range: Some(((i + 1) as u32, (i + 1) as u32)),
                text: block,
            });
        }
        Ok(doc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_rows_become_col_val_blocks() {
        let csv = "name,amount,note\nRechnung A,\"1,20\",\"mit, komma\"\nRechnung B,3,x\n";
        let doc = TabularParser
            .parse(std::path::Path::new("f.csv"), csv.as_bytes())
            .unwrap();
        assert_eq!(doc.blocks.len(), 2);
        assert!(doc.blocks[0].text.contains("name: Rechnung A"));
        assert!(doc.blocks[0].text.contains("amount: 1,20"));
        assert!(doc.blocks[0].text.contains("note: mit, komma"));
        assert_eq!(doc.metadata["columns"], "name|amount|note");
    }

    #[test]
    fn tsv_and_jsonl() {
        let tsv = "a\tb\n1\t2\n";
        let doc = TabularParser
            .parse(std::path::Path::new("f.tsv"), tsv.as_bytes())
            .unwrap();
        assert!(doc.blocks[0].text.contains("a: 1"));

        let jsonl = "{\"user\":\"max\",\"addr\":{\"city\":\"Berlin\"}}\n{\"user\":\"ula\"}\n";
        let doc = TabularParser
            .parse(std::path::Path::new("f.jsonl"), jsonl.as_bytes())
            .unwrap();
        assert_eq!(doc.blocks.len(), 2);
        assert!(doc.blocks[0].text.contains("addr.city: Berlin"));
    }
}
