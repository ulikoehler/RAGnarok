//! Text normalization between parsing and chunking — strips content that
//! explodes the tokenizer but carries no searchable semantics.
//!
//! Two dedicated passes run over every parsed block:
//!
//! - [`strip_opaque_blobs`]: removes whitespace-free runs of base64 /
//!   base64url / hex — MIME-encoded attachments that leak into bodies,
//!   PGP armor, JWTs, DKIM/ARC signatures, `data:` URIs, minified blobs.
//!   These tokenize ~1 real token per char (a 20 KB blob ≈ 20 k tokens),
//!   which is how a "414-word" mail once produced a 12.5 k-token chunk.
//! - [`strip_html`]: markup → text via the web remote's `html_to_text`,
//!   gated on a `looks_like_html` check so pasted code/docs quoting a
//!   tag or two survive untouched.

/// Min length for a whitespace-free token to count as an opaque blob.
/// Base64 wraps at 76 chars, SHA-256 hex is 64 — 64 catches both while
/// git SHAs (40), UUIDs (36) and even absurd filenames stay.
const OPAQUE_MIN_CHARS: usize = 64;
/// A token is opaque when ≥80 % of its chars come from the base64url
/// alphabet — letters/digits plus `+`, `/`, `=`, `_`, `-`.
const OPAQUE_FRACTION_NUM: usize = 4;
const OPAQUE_FRACTION_DEN: usize = 5;
/// URL tokens longer than this are truncated — a giant `?utm_…` chain is
/// opaque-equivalent for the tokenizer but the head stays searchable.
const URL_MAX_CHARS: usize = 140;

fn is_opaque_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '=' | '_' | '-')
}

fn is_url_token(tok: &str) -> bool {
    tok.starts_with("http://")
        || tok.starts_with("https://")
        || tok.starts_with("www.")
        || tok.starts_with("mailto:")
}

/// Decide one whitespace-free token: `Some(kept)` or `None` (drop).
fn keep_or_drop(tok: &str) -> Option<String> {
    if tok.starts_with("data:") && tok.len() > 32 {
        return None;
    }
    if is_url_token(tok) {
        if tok.chars().count() > URL_MAX_CHARS {
            let end = tok
                .char_indices()
                .nth(URL_MAX_CHARS)
                .map(|(i, _)| i)
                .unwrap_or(tok.len());
            return Some(format!("{}…", &tok[..end]));
        }
        return Some(tok.to_string());
    }
    let total = tok.chars().count();
    if total >= OPAQUE_MIN_CHARS {
        let opaque = tok.chars().filter(|c| is_opaque_char(*c)).count();
        if opaque * OPAQUE_FRACTION_DEN >= total * OPAQUE_FRACTION_NUM {
            return None;
        }
    }
    Some(tok.to_string())
}

/// Remove opaque whitespace-free runs, preserving the surrounding
/// whitespace structure (a dropped token simply vanishes between its
/// neighboring whitespace).
pub fn strip_opaque_blobs(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = start.take() {
                if let Some(kept) = keep_or_drop(&text[s..i]) {
                    out.push_str(&kept);
                }
            }
            out.push(c);
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        if let Some(kept) = keep_or_drop(&text[s..]) {
            out.push_str(&kept);
        }
    }
    out
}

/// Strong markers (full documents) OR several weak tag hits — a pasted
/// `<div>` inside prose doesn't count, a fragment of real markup does.
fn looks_like_html(text: &str) -> bool {
    let l = text.to_lowercase();
    for marker in ["<!doctype", "<html", "<head", "<body", "<meta "] {
        if l.contains(marker) {
            return true;
        }
    }
    let weak = [
        "<div", "<p>", "<p ", "<br", "<span", "<a href", "<table", "</",
    ];
    weak.iter().map(|m| l.matches(m).count()).sum::<usize>() >= 4
}

/// Strip HTML markup when the text looks like a document made of it.
pub fn strip_html(text: &str) -> String {
    if looks_like_html(text) {
        crate::remote::html_to_text(text)
    } else {
        text.to_string()
    }
}

/// Full preprocessing pass for one parsed block. HTML first — markup
/// attributes contain opaque runs (`href="…very long…"`, inline
/// `data:` images) that the blob stripper would otherwise leave as
/// broken fragments after tag removal.
pub fn clean(text: &str) -> String {
    strip_opaque_blobs(&strip_html(text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_runs_are_dropped() {
        let blob = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVoxMjM0NTY3ODk".repeat(4);
        let input = format!("hello {blob} world");
        assert_eq!(strip_opaque_blobs(&input), "hello  world");
    }

    #[test]
    fn base64url_and_jwt_are_dropped() {
        let jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.\
                   eyJzdWIiOiIxMjM0NTY3ODkwMTIzNDU2Nzg5MCJ9.\
                   SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJVadQssw5c";
        let out = strip_opaque_blobs(&format!("token {jwt} end"));
        assert!(!out.contains("eyJhbGci"));
    }

    #[test]
    fn sha256_hex_dropped_git_sha_kept() {
        let sha256 = "a".repeat(64);
        let git = "b".repeat(40);
        let out = strip_opaque_blobs(&format!("{sha256} {git}"));
        assert!(!out.contains(&sha256));
        assert!(out.contains(&git));
    }

    #[test]
    fn normal_words_survive() {
        let input = "Re: your order #12345 — confirmation_token=abc123";
        assert_eq!(strip_opaque_blobs(input), input);
    }

    #[test]
    fn data_uri_dropped_long_url_truncated() {
        let data = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg".repeat(3);
        assert!(!strip_opaque_blobs(&format!("x {data}")).contains("data:"));
        let url = format!("https://example.com/{}", "p".repeat(300));
        let out = strip_opaque_blobs(&url);
        assert!(out.starts_with("https://example.com/"));
        assert!(out.chars().count() <= URL_MAX_CHARS + 1); // head + '…'
    }

    #[test]
    fn html_document_is_stripped() {
        let html = "<!doctype html><html><body><p>Hi <b>there</b></p>\
                    <script>var x=1;</script></body></html>";
        let out = strip_html(html);
        assert!(out.contains("Hi"));
        assert!(!out.contains("<p>"));
        assert!(!out.contains("var x"));
    }

    #[test]
    fn prose_mentioning_tags_survives() {
        // one stray tag is not a document
        let input = "you write <p> like this in HTML";
        assert_eq!(strip_html(input), input);
    }

    #[test]
    fn clean_composes_both() {
        let blob = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVoxMjM0NTY3ODk".repeat(4);
        let html = format!("<html><body><p>mail</p>{blob}</body></html>");
        let out = clean(&html);
        assert!(out.contains("mail"));
        assert!(!out.contains("QUJD"));
        assert!(!out.contains("<p>"));
    }
}
