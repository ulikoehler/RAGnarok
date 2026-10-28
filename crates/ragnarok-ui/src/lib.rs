//! Compiled single-page application embedded into the daemon binary.
//!
//! The SPA is an Angular app (`frontend/`); its production bundle lives in
//! `assets/` and is embedded at compile time via `rust-embed` — rebuild it
//! with `npx ng build` inside `frontend/` before compiling the binary.

use rust_embed::RustEmbed;

/// All frontend assets under `assets/`.
#[derive(RustEmbed)]
#[folder = "assets"]
pub struct UiAssets;

/// Look up an asset by its request path (e.g. `index.html`, `/app.js`).
/// Returns `(contents, mime)`.
pub fn asset(path: &str) -> Option<(Vec<u8>, String)> {
    let key = path.trim_start_matches('/');
    let key = if key.is_empty() { "index.html" } else { key };
    UiAssets::get(key).map(|f| {
        let mime = mime_guess::from_path(key)
            .first_or_octet_stream()
            .to_string();
        (f.data.into_owned(), mime)
    })
}

/// SPA fallback: any non-asset path serves the shell so client-side
/// routing works on reload.
pub fn index() -> Option<(Vec<u8>, String)> {
    asset("index.html")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_html_is_embedded() {
        let (bytes, mime) = index().expect("index.html must be embedded");
        assert_eq!(mime, "text/html");
        assert!(String::from_utf8_lossy(&bytes).contains("RAGnar"));
    }

    #[test]
    fn asset_lookup_normalizes_slashes() {
        assert!(asset("/index.html").is_some());
        assert!(asset("index.html").is_some());
        assert!(asset("").is_some()); // root -> index.html
        assert!(asset("/no-such-file.bin").is_none());
    }
}
