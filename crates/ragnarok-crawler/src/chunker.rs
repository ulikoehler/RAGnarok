//! Paragraph chunker: packs parsed `TextBlock`s into chunks of
//! `chunk_min_tokens..=chunk_max_tokens` with ~15% overlap, preserving page
//! and bounding-box offsets for the universal viewer.

use ragnarok_core::{BoundingBox, Chunk, DocumentId, ParsedDocument, Result, TextBlock};

#[derive(Debug, Clone, Copy)]
pub struct ChunkerConfig {
    pub min_tokens: usize,
    pub max_tokens: usize,
    /// Fractional overlap between consecutive chunks `[0, 1)`.
    pub overlap: f32,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        ChunkerConfig {
            min_tokens: 256,
            max_tokens: 512,
            overlap: 0.15,
        }
    }
}

/// Approximate token count — one whitespace-separated word ≈ one token.
pub fn token_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// A block reference queued for packing.
struct Unit<'a> {
    block: &'a TextBlock,
}

/// Geometry + text of a chunk tail carried into the next chunk (overlap).
type Tail = (u32, Option<BoundingBox>, Option<(u32, u32)>, String);

/// Split a text at a token boundary, returning `(head, tail)` where head has
/// exactly `n` tokens (or all of them if fewer exist).
fn split_at_tokens(text: &str, n: usize) -> (String, String) {
    let mut end_byte = text.len();
    for (i, (idx, _)) in text
        .char_indices()
        .filter(|(_, c)| c.is_whitespace())
        .enumerate()
    {
        if i + 1 == n {
            end_byte = idx;
            break;
        }
    }
    let head = text[..end_byte].trim_end().to_string();
    let tail = text[end_byte..].trim_start().to_string();
    (head, tail)
}

/// Chunk a parsed document. Blocks are packed greedily per page; oversized
/// blocks are hard-split at token boundaries (keeping their geometry).
/// The trailing `overlap` fraction of each chunk seeds the next one.
pub fn chunk_document(
    document_id: DocumentId,
    parsed: &ParsedDocument,
    cfg: ChunkerConfig,
) -> Result<Vec<Chunk>> {
    let units: Vec<Unit> = parsed
        .blocks
        .iter()
        .filter(|b| !b.text.trim().is_empty())
        .map(|b| Unit { block: b })
        .collect();
    if units.is_empty() {
        return Ok(Vec::new());
    }

    let overlap_tokens = (cfg.max_tokens as f32 * cfg.overlap) as usize;
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut cur_text = String::new();
    let mut cur_tokens = 0usize;
    let mut cur_page = units[0].block.page;
    let mut cur_bbox: Option<BoundingBox> = None;
    let mut cur_lines: Option<(u32, u32)> = None;
    let mut last_tail: Option<Tail> = None;

    let flush = |chunks: &mut Vec<Chunk>,
                 cur_text: &mut String,
                 cur_tokens: &mut usize,
                 cur_page: &mut u32,
                 cur_bbox: &mut Option<BoundingBox>,
                 cur_lines: &mut Option<(u32, u32)>,
                 last_tail: &mut Option<Tail>,
                 idx: u32| {
        if cur_text.trim().is_empty() {
            return;
        }
        chunks.push(Chunk {
            id: ragnarok_core::chunk_id(document_id, idx),
            document_id,
            chunk_index: idx,
            page: *cur_page,
            bbox: *cur_bbox,
            line_range: *cur_lines,
            text: cur_text.clone(),
            token_count: *cur_tokens,
        });
        // compute overlap carry: trailing `overlap_tokens` of this chunk
        if overlap_tokens > 0 {
            let (_, tail) = if *cur_tokens > overlap_tokens {
                split_at_tokens(cur_text, *cur_tokens - overlap_tokens)
            } else {
                (String::new(), cur_text.clone())
            };
            if !tail.is_empty() {
                *last_tail = Some((*cur_page, *cur_bbox, *cur_lines, tail));
            }
        }
        cur_text.clear();
        *cur_tokens = 0;
        *cur_bbox = None;
        *cur_lines = None;
    };

    let mut idx = 0u32;
    for unit in &units {
        // hard-split oversized blocks
        let mut segments: Vec<String> = Vec::new();
        let mut rest = unit.block.text.clone();
        while token_count(&rest) > cfg.max_tokens {
            let (head, tail) = split_at_tokens(&rest, cfg.max_tokens);
            if head.is_empty() {
                break;
            }
            segments.push(head);
            rest = tail;
        }
        if !rest.trim().is_empty() {
            segments.push(rest);
        }

        for seg in segments {
            let seg_tokens = token_count(&seg);
            // flush when minimally satisfied and adding would overflow,
            // when the hard cap is hit, or when the page changes
            if (cur_tokens >= cfg.min_tokens && cur_tokens + seg_tokens > cfg.max_tokens)
                || cur_tokens >= cfg.max_tokens
                || (!cur_text.is_empty() && unit.block.page != cur_page)
            {
                flush(
                    &mut chunks,
                    &mut cur_text,
                    &mut cur_tokens,
                    &mut cur_page,
                    &mut cur_bbox,
                    &mut cur_lines,
                    &mut last_tail,
                    idx,
                );
                idx += 1;
                if let Some((page, bbox, lines, text)) = last_tail.take() {
                    // overlap is only carried WITHIN a page: a tail from the
                    // previous page would seed this chunk with the wrong
                    // page number and bounding box — chunks stay
                    // single-page so their geometry stays truthful
                    if page == unit.block.page {
                        cur_page = page;
                        cur_bbox = bbox;
                        cur_lines = lines;
                        cur_tokens = token_count(&text);
                        cur_text = text;
                    }
                }
            }
            if cur_text.is_empty() {
                cur_page = unit.block.page;
                cur_bbox = unit.block.bbox;
                cur_lines = unit.block.line_range;
            } else {
                cur_bbox = match (cur_bbox, unit.block.bbox) {
                    (Some(a), Some(b)) if a_intersects_page(cur_page, unit.block.page) => {
                        Some(a.union(&b))
                    }
                    (Some(a), _) => Some(a),
                    (_, b) => b,
                };
                cur_lines = merge_lines(cur_lines, unit.block.line_range);
                cur_text.push_str("\n\n");
            }
            cur_text.push_str(&seg);
            cur_tokens += seg_tokens;
        }
    }
    flush(
        &mut chunks,
        &mut cur_text,
        &mut cur_tokens,
        &mut cur_page,
        &mut cur_bbox,
        &mut cur_lines,
        &mut last_tail,
        idx,
    );
    Ok(chunks)
}

fn a_intersects_page(a: u32, b: u32) -> bool {
    a == b
}

fn merge_lines(a: Option<(u32, u32)>, b: Option<(u32, u32)>) -> Option<(u32, u32)> {
    match (a, b) {
        (Some((s, e)), Some((s2, e2))) => Some((s.min(s2), e.max(e2))),
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ragnarok_core::{DocType, ParsedDocument, TextBlock};

    fn words(n: usize) -> String {
        (0..n)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn parsed(blocks: Vec<(u32, String)>) -> ParsedDocument {
        let mut d = ParsedDocument::new("/doc.pdf", DocType::Pdf);
        for (page, text) in blocks {
            d.blocks.push(TextBlock {
                page,
                bbox: Some(BoundingBox::new(0.1, 0.1, 0.9, 0.2)),
                line_range: None,
                text,
            });
        }
        d
    }

    #[test]
    fn small_doc_single_chunk() {
        let doc = parsed(vec![(0, words(50))]);
        let chunks = chunk_document(DocumentId::new(), &doc, ChunkerConfig::default()).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].token_count, 50);
        assert_eq!(chunks[0].chunk_index, 0);
    }

    #[test]
    fn multiple_blocks_pack_and_overlap() {
        let cfg = ChunkerConfig {
            min_tokens: 10,
            max_tokens: 20,
            overlap: 0.25, // 5 tokens carry
        };
        let doc = parsed(vec![(0, words(20)), (0, words(20))]);
        let chunks = chunk_document(DocumentId::new(), &doc, cfg).unwrap();
        assert!(chunks.len() >= 2);
        // chunk 2 starts with the tail of chunk 1
        let c0_tail: Vec<&str> = chunks[0].text.split_whitespace().rev().take(5).collect();
        let c1_head: Vec<&str> = chunks[1].text.split_whitespace().take(5).collect();
        let mut c0_tail = c0_tail;
        c0_tail.reverse();
        assert_eq!(c0_tail, c1_head);
        // deterministic ids
        let chunks2 = chunk_document(chunks[0].document_id, &doc, cfg).unwrap();
        assert_eq!(chunks[0].id, chunks2[0].id);
    }

    #[test]
    fn page_break_splits_chunks() {
        let cfg = ChunkerConfig {
            min_tokens: 5,
            max_tokens: 100,
            overlap: 0.0,
        };
        let doc = parsed(vec![(0, words(30)), (1, words(30)), (1, words(20))]);
        let chunks = chunk_document(DocumentId::new(), &doc, cfg).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].page, 0);
        assert_eq!(chunks[1].page, 1);
        // page 1 chunk unions both block bboxes
        assert_eq!(chunks[1].token_count, 50);
    }

    #[test]
    fn page_break_discards_overlap_geometry() {
        // with overlap > 0 the tail of the page-0 chunk must NOT seed the
        // page-1 chunk — the chunk's page/bbox must reflect its content
        let cfg = ChunkerConfig {
            min_tokens: 5,
            max_tokens: 100,
            overlap: 0.5,
        };
        let doc = parsed(vec![(0, words(30)), (1, words(30))]);
        let chunks = chunk_document(DocumentId::new(), &doc, cfg).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].page, 0);
        assert_eq!(chunks[1].page, 1);
        assert_eq!(chunks[1].token_count, 30, "no page-0 tail leaked in");
        // same-page overlap still works
        let doc = parsed(vec![(0, words(60))]);
        let chunks = chunk_document(DocumentId::new(), &doc, cfg).unwrap();
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| c.page == 0));
    }

    #[test]
    fn oversized_block_is_split() {
        let cfg = ChunkerConfig {
            min_tokens: 10,
            max_tokens: 20,
            overlap: 0.0,
        };
        let doc = parsed(vec![(0, words(45))]);
        let chunks = chunk_document(DocumentId::new(), &doc, cfg).unwrap();
        assert_eq!(chunks.len(), 3);
        assert!(chunks.iter().all(|c| c.token_count <= 20));
    }

    #[test]
    fn empty_document() {
        let doc = parsed(vec![]);
        assert!(
            chunk_document(DocumentId::new(), &doc, ChunkerConfig::default())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn split_at_tokens_boundary() {
        let (h, t) = split_at_tokens("a b c d e", 3);
        assert_eq!(h, "a b c");
        assert_eq!(t, "d e");
        let (h, t) = split_at_tokens("one", 10);
        assert_eq!(h, "one");
        assert_eq!(t, "");
    }
}
