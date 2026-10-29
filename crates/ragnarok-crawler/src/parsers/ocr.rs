//! Tesseract OCR for scans and images: `image/*` documents are OCRed
//! directly; PDFs whose extractable text layer is below a threshold are
//! rasterized via `pdftoppm` and OCRed page by page. Tesseract's TSV
//! output carries word coordinates, so OCRed chunks keep real
//! normalized bounding boxes — the viewer's BBox overlay works on scans.

use super::Parser;
use ragnarok_core::{BoundingBox, DocType, Error, OcrConfig, ParsedDocument, Result, TextBlock};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

/// Run `binary` with `args`, feeding `stdin_bytes`, returning stdout.
/// The child is killed after `timeout_secs` — a pathological image must
/// not wedge an ingest worker forever.
fn run_with_timeout(
    binary: &str,
    args: &[&str],
    stdin_bytes: &[u8],
    timeout_secs: u64,
) -> Result<Vec<u8>> {
    let mut child = Command::new(binary)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| Error::Internal(format!("{binary} spawn: {e}")))?;
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(stdin_bytes);
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut out = Vec::new();
                if let Some(mut stdout) = child.stdout.take() {
                    use std::io::Read;
                    let _ = stdout.read_to_end(&mut out);
                }
                if !status.success() {
                    return Err(Error::Internal(format!("{binary} exited with {status}")));
                }
                return Ok(out);
            }
            Ok(None) => {
                if std::time::Instant::now() > deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(Error::Internal(format!(
                        "{binary} timed out after {timeout_secs}s"
                    )));
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(e) => return Err(Error::Internal(format!("{binary} wait: {e}"))),
        }
    }
}

/// Word row of a tesseract TSV (level 5).
struct Word {
    block: u32,
    par: u32,
    line: u32,
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
    text: String,
}

/// Parse tesseract TSV into per-line `TextBlock`s with normalized bboxes
/// (top-left origin, same convention as the PDF parser).
fn tsv_to_blocks(tsv: &str, page: u32) -> Vec<TextBlock> {
    let mut page_w = 1.0f32;
    let mut page_h = 1.0f32;
    let mut words: Vec<Word> = Vec::new();
    for (i, line) in tsv.lines().enumerate() {
        if i == 0 || line.is_empty() {
            continue; // header
        }
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() < 12 {
            continue;
        }
        let level: u32 = c[0].parse().unwrap_or(0);
        let num = |i: usize| c[i].parse::<f32>().unwrap_or(0.0);
        match level {
            1 => {
                page_w = num(8).max(1.0);
                page_h = num(9).max(1.0);
            }
            5 => {
                let text = c[11].trim();
                if text.is_empty() {
                    continue;
                }
                words.push(Word {
                    block: c[2].parse().unwrap_or(0),
                    par: c[3].parse().unwrap_or(0),
                    line: c[4].parse().unwrap_or(0),
                    left: num(6),
                    top: num(7),
                    right: num(6) + num(8),
                    bottom: num(7) + num(9),
                    text: text.to_owned(),
                });
            }
            _ => {}
        }
    }
    // group by (block, par, line) preserving TSV order
    let mut lines: Vec<(u32, u32, u32)> = Vec::new();
    for w in &words {
        let key = (w.block, w.par, w.line);
        if !lines.contains(&key) {
            lines.push(key);
        }
    }
    lines
        .iter()
        .map(|&(b, p, l)| {
            let ws: Vec<&Word> = words
                .iter()
                .filter(|w| (w.block, w.par, w.line) == (b, p, l))
                .collect();
            let text = ws
                .iter()
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let x0 = ws.iter().map(|w| w.left).fold(f32::INFINITY, f32::min) / page_w;
            let y0 = ws.iter().map(|w| w.top).fold(f32::INFINITY, f32::min) / page_h;
            let x1 = ws.iter().map(|w| w.right).fold(0.0f32, f32::max) / page_w;
            let y1 = ws.iter().map(|w| w.bottom).fold(0.0f32, f32::max) / page_h;
            TextBlock {
                page,
                bbox: Some(BoundingBox {
                    x0: x0.clamp(0.0, 1.0),
                    y0: y0.clamp(0.0, 1.0),
                    x1: x1.clamp(0.0, 1.0),
                    y1: y1.clamp(0.0, 1.0),
                }),
                line_range: None,
                text,
            }
        })
        .collect()
}

/// OCR one raster image (png/jpg/tif/…) → line blocks for `page`.
pub fn ocr_image(cfg: &OcrConfig, image: &[u8], page: u32) -> Result<Vec<TextBlock>> {
    let tsv = run_with_timeout(
        &cfg.binary,
        &["stdin", "stdout", "-l", &cfg.lang, "tsv"],
        image,
        cfg.timeout_secs,
    )?;
    Ok(tsv_to_blocks(&String::from_utf8_lossy(&tsv), page))
}

/// Rasterize a scanned PDF via `pdftoppm` and OCR each page. Returns the
/// blocks plus how many pages were rendered.
pub fn ocr_pdf(cfg: &OcrConfig, bytes: &[u8]) -> Result<Vec<TextBlock>> {
    let dir = std::env::temp_dir().join(format!("rag-ocr-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).map_err(|e| Error::Internal(format!("ocr tempdir: {e}")))?;
    let pdf_path = dir.join("scan.pdf");
    std::fs::write(&pdf_path, bytes)
        .map_err(|e| Error::Internal(format!("ocr temp write: {e}")))?;
    let prefix = dir.join("page");
    let out = Command::new(&cfg.pdftoppm)
        .args([
            "-png",
            "-r",
            &cfg.dpi.to_string(),
            "-l",
            &cfg.max_pages.to_string(),
            pdf_path.to_str().unwrap_or_default(),
            prefix.to_str().unwrap_or_default(),
        ])
        .stdin(Stdio::null())
        .output()
        .map_err(|e| Error::Internal(format!("pdftoppm spawn: {e}")))?;
    if !out.status.success() {
        let _ = std::fs::remove_dir_all(&dir);
        return Err(Error::Internal(format!("pdftoppm: {}", out.status)));
    }
    let mut pages: Vec<_> = std::fs::read_dir(&dir)
        .map_err(|e| Error::Internal(format!("ocr tempdir read: {e}")))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    pages.sort();
    let mut blocks = Vec::new();
    for (i, png) in pages.iter().enumerate() {
        let img = std::fs::read(png).map_err(|e| Error::Internal(format!("ocr png read: {e}")))?;
        match ocr_image(cfg, &img, i as u32) {
            Ok(mut b) => blocks.append(&mut b),
            Err(e) => {
                tracing::warn!(page = i, "ocr page failed: {e}");
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok(blocks)
}

/// Whether a parsed document's text layer is thin enough to treat the
/// document as a scan.
pub fn needs_ocr(parsed: &ParsedDocument, threshold: usize) -> bool {
    parsed.full_text().trim().chars().count() < threshold
}

/// `image/*` files: OCR via tesseract, blocks on page 0.
pub struct OcrParser {
    pub cfg: OcrConfig,
}

impl Parser for OcrParser {
    fn extensions(&self) -> &'static [&'static str] {
        &["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"]
    }
    fn doc_type(&self) -> DocType {
        DocType::Image
    }
    fn parse(&self, path: &Path, bytes: &[u8]) -> Result<ParsedDocument> {
        let mut out = ParsedDocument::new(path.display().to_string(), DocType::Image);
        out.blocks = ocr_image(&self.cfg, bytes, 0)?;
        out.metadata
            .insert("ocr".to_owned(), "tesseract".to_owned());
        Ok(out)
    }
}

impl std::fmt::Debug for OcrParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OcrParser")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TSV: &str = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n\
        1\t1\t0\t0\t0\t0\t0\t0\t1000\t1400\t-1\t\n\
        2\t1\t1\t0\t0\t0\t0\t0\t1000\t1400\t-1\t\n\
        5\t1\t1\t1\t1\t1\t100\t100\t80\t30\t96\tWARTUNG\n\
        5\t1\t1\t1\t1\t2\t200\t100\t60\t30\t95\tWV-2025\n\
        5\t1\t1\t1\t2\t1\t100\t150\t120\t30\t92\tVertrag\n\
        5\t1\t2\t1\t1\t1\t100\t400\t90\t30\t88\tZeile2\n";

    #[test]
    fn tsv_groups_words_to_lines_with_bbox() {
        let blocks = tsv_to_blocks(TSV, 0);
        assert_eq!(blocks.len(), 3, "{blocks:?}");
        assert_eq!(blocks[0].text, "WARTUNG WV-2025");
        let bb = blocks[0].bbox.unwrap();
        assert!((bb.x0 - 0.1).abs() < 1e-6, "{bb:?}");
        assert!((bb.x1 - 0.26).abs() < 1e-6, "{bb:?}"); // right edge of WV-2025
        assert!((bb.y0 - 100.0 / 1400.0).abs() < 1e-6);
        assert_eq!(blocks[1].text, "Vertrag");
        assert_eq!(blocks[2].text, "Zeile2");
        assert_eq!(blocks[0].page, 0);
    }

    #[test]
    fn needs_ocr_threshold() {
        let mut d = ParsedDocument::new("x", DocType::Pdf);
        assert!(needs_ocr(&d, 32));
        d.blocks.push(TextBlock::plain("   "));
        assert!(needs_ocr(&d, 32));
        d.blocks
            .push(TextBlock::plain("langer extrahierter text mit viel inhalt"));
        assert!(!needs_ocr(&d, 32));
    }

    #[test]
    fn real_tesseract_if_available() {
        if Command::new("tesseract").arg("--version").output().is_err() {
            return; // binary missing in CI — parser tests above still cover logic
        }
        // minimal binary PBM (P4) is accepted by leptonica/tesseract
        let cfg = OcrConfig::default();
        // 200x80 all-white P4 pbm — OCR should return empty or near-empty
        // gracefully rather than erroring
        let mut pbm = b"P4\n200 80\n".to_vec();
        pbm.extend(std::iter::repeat_n(0u8, 200 * 80 / 8));
        let blocks = ocr_image(&cfg, &pbm, 0).unwrap();
        assert!(blocks.is_empty() || blocks.iter().all(|b| b.text.len() < 4));
    }
}
