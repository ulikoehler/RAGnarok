//! PDF parser via `lopdf` (the spec's `indra` crate does not exist on
//! crates.io — this is the documented fallback: a low-level PDF content
//! stream tokenizer, which the spec allows explicitly).
//!
//! Pipeline per page:
//! 1. decode the content stream into operations
//! 2. run a text-positioning state machine (BT/ET, Tf, Td/TD/Tm/T*, Tj/TJ/'/')
//!    emitting positioned text runs
//! 3. cluster runs into lines, derive bounding boxes
//! 4. normalize boxes to `[0,1]` against the page MediaBox (y-up → y-down)
//!
//! String decoding uses the font's `ToUnicode` CMap when available
//! (bfchar/bfrange, greedy longest-code matching), falling back to UTF-16BE
//! detection then Latin-1.

use super::Parser;
use lopdf::content::Content;
use lopdf::{Dictionary, Document, Object, ObjectId, StringFormat};
use ragnarok_core::{BoundingBox, DocType, Error, ParsedDocument, Result, TextBlock};
use std::collections::HashMap;
use std::path::Path;

/// PDF text extraction. When `ocr` is configured and the text layer is
/// thinner than `pdf_text_threshold` chars, the document is treated as a
/// scan: pages are rasterized via pdftoppm and OCRed with tesseract —
/// the returned blocks then carry real TSV word bboxes.
pub struct PdfParser {
    pub ocr: Option<ragnarok_core::OcrConfig>,
}

impl Parser for PdfParser {
    fn extensions(&self) -> &'static [&'static str] {
        &["pdf"]
    }

    fn doc_type(&self) -> DocType {
        DocType::Pdf
    }

    fn parse(&self, path: &Path, bytes: &[u8]) -> Result<ParsedDocument> {
        let doc = Document::load_mem(bytes)
            .map_err(|e| Error::parse(path.display().to_string(), format!("pdf load: {e}")))?;
        let mut out = parse_pdf_document(&doc, path)?;
        if let Some(ocr) = &self.ocr {
            if super::ocr::needs_ocr(&out, ocr.pdf_text_threshold) {
                out.blocks = super::ocr::ocr_pdf(ocr, bytes)?;
                out.metadata
                    .insert("ocr".to_owned(), "tesseract".to_owned());
            }
        }
        Ok(out)
    }
}

// ------------------------------------------------------------------ cmap ----

/// ToUnicode CMap: maps glyph codes (raw string bytes) to Unicode text.
#[derive(Debug, Default)]
struct CMap {
    /// (byte_length, code) -> replacement text
    map: HashMap<(usize, u64), String>,
    /// sorted distinct code lengths, longest first
    lengths: Vec<usize>,
}

impl CMap {
    fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Decode one code at `pos`, returning (text, consumed_bytes).
    fn decode_at(&self, bytes: &[u8], pos: usize) -> Option<(String, usize)> {
        for &len in &self.lengths {
            if pos + len <= bytes.len() {
                let mut code = 0u64;
                for b in &bytes[pos..pos + len] {
                    code = (code << 8) | *b as u64;
                }
                if let Some(s) = self.map.get(&(len, code)) {
                    return Some((s.clone(), len));
                }
            }
        }
        None
    }

    fn decode(&self, bytes: &[u8]) -> String {
        let mut out = String::new();
        let mut pos = 0;
        while pos < bytes.len() {
            match self.decode_at(bytes, pos) {
                Some((s, n)) => {
                    out.push_str(&s);
                    pos += n;
                }
                None => pos += 1,
            }
        }
        out
    }
}

fn hex_bytes(tok: &str) -> Option<Vec<u8>> {
    let t = tok.trim().trim_start_matches('<').trim_end_matches('>');
    if t.is_empty() || !t.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let mut out = Vec::with_capacity(t.len() / 2);
    let t = if t.len() % 2 == 1 {
        format!("{t}0")
    } else {
        t.to_string()
    };
    for pair in t.as_bytes().chunks(2) {
        out.push(u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()?);
    }
    Some(out)
}

fn hex_u64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0u64, |acc, b| (acc << 8) | *b as u64)
}

/// Parse a ToUnicode CMap stream (bfchar + bfrange sections).
fn parse_cmap(data: &[u8]) -> CMap {
    let text = String::from_utf8_lossy(data);
    let mut cmap = CMap::default();
    let tokens: Vec<String> = text
        .split_whitespace()
        .flat_map(|t| {
            // split `<a><b>` runs into separate tokens
            let mut v = Vec::new();
            let mut cur = String::new();
            let mut in_hex = false;
            for c in t.chars() {
                if c == '<' && !in_hex {
                    if !cur.is_empty() {
                        v.push(cur.clone());
                        cur.clear();
                    }
                    in_hex = true;
                    cur.push(c);
                } else if c == '>' && in_hex {
                    cur.push(c);
                    v.push(cur.clone());
                    cur.clear();
                    in_hex = false;
                } else if !in_hex && (c == '[' || c == ']') {
                    if !cur.is_empty() {
                        v.push(cur.clone());
                        cur.clear();
                    }
                    v.push(c.to_string());
                } else {
                    cur.push(c);
                }
            }
            if !cur.is_empty() {
                v.push(cur);
            }
            v
        })
        .collect();

    let mut i = 0;
    while i < tokens.len() {
        match tokens[i].as_str() {
            "beginbfchar" => {
                i += 1;
                while i + 1 < tokens.len() && tokens[i] != "endbfchar" {
                    if let (Some(src), Some(dst)) =
                        (hex_bytes(&tokens[i]), hex_bytes(&tokens[i + 1]))
                    {
                        let len = src.len();
                        cmap.map.insert((len, hex_u64(&src)), decode_utf16be(&dst));
                    }
                    i += 2;
                }
            }
            "beginbfrange" => {
                i += 1;
                while i + 2 < tokens.len() && tokens[i] != "endbfrange" {
                    if let (Some(lo), Some(hi)) = (hex_bytes(&tokens[i]), hex_bytes(&tokens[i + 1]))
                    {
                        let (lo_v, hi_v, len) = (hex_u64(&lo), hex_u64(&hi), lo.len());
                        if tokens[i + 2] == "[" {
                            // array form: <lo> <hi> [<d0> <d1> ...]
                            let mut j = i + 3;
                            let mut off = 0u64;
                            while j < tokens.len() && tokens[j] != "]" {
                                if let Some(dst) = hex_bytes(&tokens[j]) {
                                    cmap.map.insert((len, lo_v + off), decode_utf16be(&dst));
                                }
                                off += 1;
                                j += 1;
                            }
                            i = j;
                        } else if let Some(dst) = hex_bytes(&tokens[i + 2]) {
                            // sequential form: <lo> <hi> <dst0>
                            for off in 0..=(hi_v - lo_v) {
                                let dst_b = hex_u64(&dst) + off;
                                let dst_len = dst.len().max(2);
                                let mut b = vec![0u8; dst_len];
                                for k in 0..dst_len {
                                    b[dst_len - 1 - k] = ((dst_b >> (8 * k)) & 0xff) as u8;
                                }
                                cmap.map.insert((len, lo_v + off), decode_utf16be(&b));
                            }
                        }
                    }
                    i += 3;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let mut lengths: Vec<usize> = cmap.map.keys().map(|(l, _)| *l).collect();
    lengths.sort_unstable();
    lengths.dedup();
    lengths.reverse();
    cmap.lengths = lengths;
    cmap
}

fn decode_utf16be(bytes: &[u8]) -> String {
    if bytes.len().is_multiple_of(2) && bytes.len() >= 2 {
        let units: Vec<u16> = bytes
            .chunks(2)
            .map(|c| u16::from_be_bytes([c[0], c[1]]))
            .collect();
        String::from_utf16_lossy(&units)
    } else {
        bytes.iter().map(|b| *b as char).collect()
    }
}

// --------------------------------------------------------------- decoder ----

/// Decode a PDF string object to text using the current font's CMap.
fn decode_pdf_string(bytes: &[u8], cmap: Option<&CMap>) -> String {
    if let Some(cmap) = cmap {
        if !cmap.is_empty() {
            return cmap.decode(bytes);
        }
    }
    // UTF-16BE with BOM or heuristic (half the bytes are NUL)
    if bytes.len() >= 2 {
        if bytes.starts_with(&[0xfe, 0xff]) {
            let units: Vec<u16> = bytes[2..]
                .chunks(2)
                .filter_map(|c| {
                    if c.len() == 2 {
                        Some(u16::from_be_bytes([c[0], c[1]]))
                    } else {
                        None
                    }
                })
                .collect();
            return String::from_utf16_lossy(&units);
        }
        let zeros = bytes.iter().filter(|b| **b == 0).count();
        if bytes.len().is_multiple_of(2) && zeros * 2 >= bytes.len() {
            let units: Vec<u16> = bytes
                .chunks(2)
                .map(|c| u16::from_be_bytes([c[0], c[1]]))
                .collect();
            let s = String::from_utf16_lossy(&units);
            if s.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
                return s;
            }
        }
    }
    bytes.iter().map(|b| *b as char).collect()
}

// ------------------------------------------------------------ text state ----

/// One positioned text emission (a `Tj`/`TJ` run) in PDF coordinates.
#[derive(Debug, Clone)]
struct TextRun {
    text: String,
    x: f32,
    /// baseline y in PDF coords (origin bottom-left, y up)
    y: f32,
    font_size: f32,
    /// estimated advance width in PDF units
    width: f32,
}

#[derive(Clone)]
struct TextState {
    tm: [f32; 6],
    tlm: [f32; 6],
    font: String,
    font_size: f32,
    tc: f32,
    tw: f32,
    tl: f32,
    th: f32,
}

impl Default for TextState {
    fn default() -> Self {
        TextState {
            tm: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            tlm: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            font: String::new(),
            font_size: 0.0,
            tc: 0.0,
            tw: 0.0,
            tl: 0.0,
            th: 100.0,
        }
    }
}

fn translate(t: [f32; 6], tx: f32, ty: f32) -> [f32; 6] {
    // T_translate * t  ->  e' = tx*a + ty*c + e ; f' = tx*b + ty*d + f
    [
        t[0],
        t[1],
        t[2],
        t[3],
        tx * t[0] + ty * t[2] + t[4],
        tx * t[1] + ty * t[3] + t[5],
    ]
}

fn num(o: &Object) -> f32 {
    match o {
        Object::Real(f) => *f,
        Object::Integer(i) => *i as f32,
        _ => 0.0,
    }
}

fn string_bytes(o: &Object) -> Option<Vec<u8>> {
    match o {
        Object::String(b, StringFormat::Literal) | Object::String(b, StringFormat::Hexadecimal) => {
            Some(b.clone())
        }
        _ => None,
    }
}

/// Emit a `Tj`-equivalent text run and advance the text matrix.
fn show_run(
    st: &mut TextState,
    fonts: &HashMap<String, CMap>,
    bytes: &[u8],
    runs: &mut Vec<TextRun>,
) {
    let text = decode_pdf_string(bytes, fonts.get(&st.font));
    if text.trim().is_empty() {
        return;
    }
    // estimated advance: ~0.5 em per char plus spacing
    let size = st.font_size.max(1.0);
    let scale = (st.th / 100.0).max(0.01);
    let width = text.chars().count() as f32 * (size * 0.5 * scale + st.tc)
        + st.tw * text.matches(' ').count() as f32;
    runs.push(TextRun {
        text,
        x: st.tm[4],
        y: st.tm[5],
        font_size: size,
        width: width.max(1.0),
    });
    // advance text matrix along x
    st.tm[4] += width;
}

/// Run the text-showing state machine over one page's content operations.
fn extract_runs(ops: &[lopdf::content::Operation], fonts: &HashMap<String, CMap>) -> Vec<TextRun> {
    let mut st = TextState::default();
    let mut runs = Vec::new();

    for op in ops {
        let o = &op.operands;
        match op.operator.as_str() {
            "BT" => st = TextState::default(),
            "Tf" => {
                if let Some(Object::Name(n)) = o.first() {
                    st.font = String::from_utf8_lossy(n).to_string();
                }
                if let Some(sz) = o.get(1) {
                    st.font_size = num(sz);
                }
            }
            "Tc" => st.tc = o.first().map(num).unwrap_or(0.0),
            "Tw" => st.tw = o.first().map(num).unwrap_or(0.0),
            "Tz" => st.th = o.first().map(num).unwrap_or(100.0),
            "TL" => st.tl = o.first().map(num).unwrap_or(0.0),
            "Td" => {
                let (tx, ty) = (
                    o.first().map(num).unwrap_or(0.0),
                    o.get(1).map(num).unwrap_or(0.0),
                );
                st.tlm = translate(st.tlm, tx, ty);
                st.tm = st.tlm;
            }
            "TD" => {
                let (tx, ty) = (
                    o.first().map(num).unwrap_or(0.0),
                    o.get(1).map(num).unwrap_or(0.0),
                );
                st.tl = -ty;
                st.tlm = translate(st.tlm, tx, ty);
                st.tm = st.tlm;
            }
            "Tm" => {
                if o.len() >= 6 {
                    st.tm = [
                        num(&o[0]),
                        num(&o[1]),
                        num(&o[2]),
                        num(&o[3]),
                        num(&o[4]),
                        num(&o[5]),
                    ];
                    st.tlm = st.tm;
                }
            }
            "T*" => {
                st.tlm = translate(st.tlm, 0.0, -st.tl);
                st.tm = st.tlm;
            }
            "Tj" => {
                if let Some(b) = o.first().and_then(string_bytes) {
                    show_run(&mut st, fonts, &b, &mut runs);
                }
            }
            "TJ" => {
                if let Some(Object::Array(items)) = o.first() {
                    for it in items {
                        match it {
                            Object::String(_, _) => {
                                if let Some(b) = string_bytes(it) {
                                    show_run(&mut st, fonts, &b, &mut runs);
                                }
                            }
                            // large negative kerning gap approximates a space
                            Object::Real(f) if *f < -200.0 => {
                                if let Some(last) = runs.last_mut() {
                                    if !last.text.ends_with(' ') {
                                        last.text.push(' ');
                                    }
                                }
                            }
                            Object::Integer(i) if *i < -200 => {
                                if let Some(last) = runs.last_mut() {
                                    if !last.text.ends_with(' ') {
                                        last.text.push(' ');
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            "'" | "\"" => {
                st.tlm = translate(st.tlm, 0.0, -st.tl);
                st.tm = st.tlm;
                let idx = if op.operator == "\"" { 2 } else { 0 };
                if let Some(b) = o.get(idx).and_then(string_bytes) {
                    show_run(&mut st, fonts, &b, &mut runs);
                }
            }
            _ => {}
        }
    }
    runs
}

// ----------------------------------------------------------- line layout ----

/// Cluster runs into lines (y within tolerance), sort left-to-right,
/// merge gaps to spaces, emit blocks with PDF-coordinate boxes.
fn runs_to_blocks(page: u32, mut runs: Vec<TextRun>, mediabox: [f32; 4]) -> Vec<TextBlock> {
    // reading order: y descending (top of page first)
    runs.sort_by(|a, b| b.y.total_cmp(&a.y).then(a.x.total_cmp(&b.x)));

    let mut blocks = Vec::new();
    let mut cur: Vec<TextRun> = Vec::new();
    let mut cur_y = 0.0f32;
    for run in runs {
        let tol = (run.font_size * 0.5).max(2.5);
        if cur.is_empty() || (run.y - cur_y).abs() <= tol {
            if cur.is_empty() {
                cur_y = run.y;
            }
            cur.push(run);
        } else {
            blocks.push(make_line_block(
                page,
                std::mem::take(&mut cur),
                cur_y,
                mediabox,
            ));
            cur_y = run.y;
            cur.push(run);
        }
    }
    if !cur.is_empty() {
        blocks.push(make_line_block(page, cur, cur_y, mediabox));
    }
    blocks
}

fn make_line_block(page: u32, mut runs: Vec<TextRun>, _line_y: f32, mb: [f32; 4]) -> TextBlock {
    runs.sort_by(|a, b| a.x.total_cmp(&b.x));
    let mut text = String::new();
    let mut prev_end: Option<f32> = None;
    for r in &runs {
        if let Some(end) = prev_end {
            let gap = r.x - end;
            if gap > r.font_size * 0.25 && !text.ends_with(' ') && !text.is_empty() {
                text.push(' ');
            }
        }
        text.push_str(&r.text);
        prev_end = Some(r.x + r.width);
    }

    // bbox in PDF coords -> normalize (y flipped: PDF y-up, box y-down)
    let (w, h) = ((mb[2] - mb[0]).max(1.0), (mb[3] - mb[1]).max(1.0));
    let x0 = runs.iter().map(|r| r.x).fold(f32::MAX, f32::min);
    let x1 = runs.iter().map(|r| r.x + r.width).fold(f32::MIN, f32::max);
    let y_base_max = runs.iter().map(|r| r.y).fold(f32::MIN, f32::max);
    let size = runs.iter().map(|r| r.font_size).fold(0.0f32, f32::max);
    // top edge = above the baseline by ~ascent, bottom = below by ~descent
    let y_top_pdf = y_base_max + size * 0.2;
    let y_bot_pdf = y_base_max - size;
    let nx0 = (x0 - mb[0]) / w;
    let nx1 = (x1 - mb[0]) / w;
    let ny0 = 1.0 - (y_top_pdf - mb[1]) / h; // pdf y-up -> viewer y-down
    let ny1 = 1.0 - (y_bot_pdf - mb[1]) / h;

    TextBlock {
        page,
        bbox: Some(BoundingBox::new(nx0, ny0, nx1, ny1)),
        line_range: None,
        text,
    }
}

// ----------------------------------------------------------------- pages ----

fn deref<'a>(doc: &'a Document, obj: &'a Object) -> Option<&'a Object> {
    match obj {
        Object::Reference(id) => doc.get_object(*id).ok(),
        other => Some(other),
    }
}

fn dict_get<'a>(d: &'a Dictionary, key: &[u8]) -> Option<&'a Object> {
    d.get(key).ok()
}

/// Walk page + parents for a MediaBox; default US-Letter.
fn mediabox(doc: &Document, page_id: ObjectId) -> [f32; 4] {
    let mut dict = doc.get_dictionary(page_id).ok();
    let mut hops = 0;
    while let Some(d) = dict {
        if let Some(mb) = dict_get(d, b"MediaBox").and_then(|o| deref(doc, o)) {
            if let Ok(arr) = mb.as_array() {
                if arr.len() >= 4 {
                    return [num(&arr[0]), num(&arr[1]), num(&arr[2]), num(&arr[3])];
                }
            }
        }
        dict = dict_get(d, b"Parent")
            .and_then(|o| o.as_reference().ok())
            .and_then(|id| doc.get_dictionary(id).ok());
        hops += 1;
        if hops > 8 {
            break;
        }
    }
    [0.0, 0.0, 612.0, 792.0]
}

/// Resolve `/Resources/Font/*` -> `ToUnicode` cmaps for one page.
fn load_fonts(doc: &Document, page_dict: &Dictionary) -> HashMap<String, CMap> {
    let mut fonts = HashMap::new();
    let resources = dict_get(page_dict, b"Resources").and_then(|o| deref(doc, o));
    let Some(res) = resources.and_then(|o| o.as_dict().ok()) else {
        return fonts;
    };
    let font_dict = dict_get(res, b"Font").and_then(|o| deref(doc, o));
    let Some(fonts_d) = font_dict.and_then(|o| o.as_dict().ok()) else {
        return fonts;
    };
    for (name, obj) in fonts_d.iter() {
        let fname = String::from_utf8_lossy(name).to_string();
        let font_obj = match deref(doc, obj).and_then(|o| o.as_dict().ok()) {
            Some(d) => d,
            None => continue,
        };
        let cmap = dict_get(font_obj, b"ToUnicode")
            .and_then(|o| o.as_reference().ok())
            .and_then(|id| doc.get_object(id).ok())
            .and_then(|o| o.as_stream().ok())
            .and_then(|s| s.decompressed_content().ok())
            .map(|bytes| parse_cmap(&bytes))
            .unwrap_or_default();
        fonts.insert(fname, cmap);
    }
    fonts
}

fn parse_pdf_document(doc: &Document, path: &Path) -> Result<ParsedDocument> {
    let mut out = ParsedDocument::new(path.display().to_string(), DocType::Pdf);
    let pages = doc.get_pages();
    out.metadata.insert("pages".into(), pages.len().to_string());
    for (page_no, page_id) in pages {
        let page_dict = match doc.get_dictionary(page_id) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let fonts = load_fonts(doc, page_dict);
        let mb = mediabox(doc, page_id);
        let content_bytes = doc.get_page_content(page_id).unwrap_or_default();
        let content = match Content::decode(&content_bytes) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let runs = extract_runs(&content.operations, &fonts);
        let mut blocks = runs_to_blocks(page_no - 1, runs, mb); // 0-indexed pages
        out.blocks.append(&mut blocks);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::content::Content;

    fn ops(stream: &[u8]) -> Vec<lopdf::content::Operation> {
        Content::decode(stream).unwrap().operations
    }

    #[test]
    fn state_machine_extracts_positions() {
        // BT /F1 12 Tf, Tm to (100, 700), show "Hello"
        let ops = ops(b"BT /F1 12 Tf 1 0 0 1 100 700 Tm (Hello) Tj ET");
        let runs = extract_runs(&ops, &HashMap::new());
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].text, "Hello");
        assert_eq!(runs[0].x, 100.0);
        assert_eq!(runs[0].y, 700.0);
        assert_eq!(runs[0].font_size, 12.0);
    }

    #[test]
    fn td_and_tstar_advance_lines() {
        let ops = ops(b"BT /F1 10 Tf 0 750 Td 14 TL (line1) Tj T* (line2) Tj ET");
        let runs = extract_runs(&ops, &HashMap::new());
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].y, 750.0);
        assert_eq!(runs[1].y, 736.0); // 750 - 14 leading
        assert_eq!(runs[0].text, "line1");
        assert_eq!(runs[1].text, "line2");
    }

    #[test]
    fn tj_array_joins_with_gap_spaces() {
        let ops = ops(b"BT /F1 12 Tf 100 700 Td [(Hel) -300 (lo) 50 (World)] TJ ET");
        let runs = extract_runs(&ops, &HashMap::new());
        assert_eq!(runs.len(), 3);
        assert_eq!(runs[0].text, "Hel ");
        assert_eq!(runs[1].text, "lo");
        assert_eq!(runs[2].text, "World");
    }

    /// Build a real single-page PDF via lopdf (MediaBox 612x792, one
    /// Helvetica text run) and serialize it — fixture for the end-to-end
    /// highlight-fidelity test.
    fn build_pdf(content: &[u8], mediabox: [i64; 4]) -> Vec<u8> {
        use lopdf::{dictionary, Stream};
        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let stream = Stream::new(dictionary! {}, content.to_vec());
        let content_id = doc.add_object(Object::Stream(stream));
        let font_id = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
        });
        let resources = dictionary! { "Font" => dictionary! { "F1" => font_id } };
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "MediaBox" => mediabox.iter().map(|v| Object::Integer(*v)).collect::<Vec<_>>(),
            "Resources" => resources,
            "Contents" => content_id,
        });
        let pages = dictionary! {
            "Type" => "Pages",
            "Kids" => vec![Object::Reference(page_id)],
            "Count" => 1i64,
        };
        doc.objects.insert(pages_id, Object::Dictionary(pages));
        let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        doc.trailer.set("Root", catalog_id);
        let mut buf = std::io::Cursor::new(Vec::new());
        doc.save_to(&mut buf).unwrap();
        buf.into_inner()
    }

    /// ACCEPTANCE TEST (spec §6.3 — highlight fidelity): parse a real PDF
    /// end-to-end; the emitted block must carry a normalized bounding box
    /// that covers the placed text.
    #[test]
    fn end_to_end_pdf_parse_yields_precise_bbox() {
        // "Hello" at baseline (100,700), 12pt, on a 612x792 page
        let pdf = build_pdf(
            b"BT /F1 12 Tf 1 0 0 1 100 700 Tm (Hello) Tj ET",
            [0, 0, 612, 792],
        );
        let p = PdfParser { ocr: None };
        let doc = p
            .parse(std::path::Path::new("/doc.pdf"), &pdf)
            .expect("pdf should parse");
        assert_eq!(doc.doc_type, DocType::Pdf);
        assert_eq!(doc.metadata["pages"], "1");
        assert_eq!(doc.blocks.len(), 1);
        let b = doc.blocks[0].bbox.expect("block must carry a bbox");
        assert_eq!(doc.blocks[0].page, 0);
        assert_eq!(doc.blocks[0].text, "Hello");
        // x0 ≈ 100/612 = 0.163 (left edge of the text)
        assert!((b.x0 - 100.0 / 612.0).abs() < 0.01);
        // y0 ≈ 1 - (700+2.4)/792 ≈ 0.113 (top edge, y-down normalized)
        assert!((b.y0 - (1.0 - 702.4 / 792.0)).abs() < 0.01);
        assert!(b.x1 > b.x0);
        assert!(b.y1 > b.y0);
        assert!(b.area() > 0.0);
    }

    #[test]
    fn multi_page_pdf() {
        // two content ops on one page produce two line blocks
        let pdf = build_pdf(
            b"BT /F1 10 Tf 1 0 0 1 50 750 Tm (alpha) Tj ET \
               BT /F1 10 Tf 1 0 0 1 50 720 Tm (beta) Tj ET",
            [0, 0, 612, 792],
        );
        let doc = PdfParser { ocr: None }
            .parse(std::path::Path::new("/d.pdf"), &pdf)
            .unwrap();
        assert_eq!(doc.blocks.len(), 2);
        assert!(doc.blocks[0].bbox.unwrap().y0 < doc.blocks[1].bbox.unwrap().y0);
    }

    #[test]
    fn invalid_pdf_errors() {
        let r = PdfParser { ocr: None }.parse(std::path::Path::new("/bad.pdf"), b"not a pdf");
        assert!(r.is_err());
    }

    #[test]
    fn bbox_normalized_top_left_origin() {
        // mediabox [0,0,600,800]; text at (100,700) size 10
        let ops = ops(b"BT /F1 10 Tf 1 0 0 1 100 700 Tm (Hi) Tj ET");
        let runs = extract_runs(&ops, &HashMap::new());
        let blocks = runs_to_blocks(0, runs, [0.0, 0.0, 600.0, 800.0]);
        assert_eq!(blocks.len(), 1);
        let b = blocks[0].bbox.unwrap();
        // x0 = 100/600, top edge = 1 - (700+2)/800
        assert!((b.x0 - 100.0 / 600.0).abs() < 1e-4);
        assert!((b.y0 - (1.0 - 702.0 / 800.0)).abs() < 1e-4);
        assert!(b.x1 > b.x0);
        assert!(b.y1 > b.y0);
        assert!(b.y0 < 0.15); // near the top of the page
        assert_eq!(blocks[0].page, 0);
    }

    #[test]
    fn line_grouping_same_row() {
        // two runs on same baseline
        let ops = ops(b"BT /F1 10 Tf 1 0 0 1 100 700 Tm (Hello) Tj 50 0 Td (World) Tj ET");
        let runs = extract_runs(&ops, &HashMap::new());
        let blocks = runs_to_blocks(0, runs, [0.0, 0.0, 600.0, 800.0]);
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].text.contains("Hello"));
        assert!(blocks[0].text.contains("World"));
    }

    #[test]
    fn cmap_bfchar_and_bfrange() {
        let cmap_text = b"
/CIDInit /ProcSet findresource begin
1 begincodespacerange <00> <FF>
1 beginbfchar
<41> <0041>
endbfchar
1 beginbfrange
<42> <44> <03B1>
endbfrange
1 beginbfrange
<50> <52> [<0078> <0079> <007A>]
endbfrange
";
        let cmap = parse_cmap(cmap_text);
        assert!(!cmap.is_empty());
        assert_eq!(cmap.decode(b"A"), "A"); // bfchar <41> -> 'A'
        assert_eq!(cmap.decode(b"BCD"), "αβγ"); // bfrange <42>-<44> -> greek
        assert_eq!(cmap.decode(b"PQ"), "xy"); // array form
    }

    #[test]
    fn decode_fallbacks() {
        // UTF-16BE with BOM
        let s = decode_pdf_string(&[0xfe, 0xff, 0x00, 0x48, 0x00, 0x69], None);
        assert_eq!(s, "Hi");
        // Latin-1-ish
        let s = decode_pdf_string(b"hello", None);
        assert_eq!(s, "hello");
        // UTF-16BE heuristic (interleaved zeros)
        let s = decode_pdf_string(&[0x00, 0x41, 0x00, 0x42], None);
        assert_eq!(s, "AB");
    }

    #[test]
    fn hex_helpers() {
        assert_eq!(hex_bytes("<0041>").unwrap(), vec![0x00, 0x41]);
        assert_eq!(hex_bytes("<ABC>").unwrap(), vec![0xAB, 0xC0]); // odd len padded
        assert!(hex_bytes("x").is_none());
        assert_eq!(hex_u64(&[0x01, 0x02]), 0x0102);
    }

    #[test]
    fn translate_composes() {
        let t = translate([1.0, 0.0, 0.0, 1.0, 100.0, 700.0], 10.0, -20.0);
        assert_eq!(t[4], 110.0);
        assert_eq!(t[5], 680.0);
        // with rotation
        let t = translate([0.0, 1.0, -1.0, 0.0, 0.0, 0.0], 10.0, 0.0);
        assert_eq!(t[4], 0.0);
        assert_eq!(t[5], 10.0);
    }
}
