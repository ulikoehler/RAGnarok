//! Initial recursive directory scan + content hashing.

use ragnarok_core::{content_hash, Result};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Default directories that are never indexed.
const DEFAULT_EXCLUDES: &[&str] = &[
    "node_modules",
    "target",
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".cache",
];

/// Recursively enumerate indexable files under `root`.
///
/// `include`/`exclude` are glob patterns matched against the file name
/// (e.g. `*.pdf`); an empty `include` means "everything".
pub fn scan_directory(
    root: &Path,
    recursive: bool,
    include: &[String],
    exclude: &[String],
    max_bytes: u64,
) -> Vec<PathBuf> {
    if !root.is_dir() {
        return Vec::new();
    }
    let depth = if recursive { usize::MAX } else { 1 };
    let mut out = Vec::new();
    for entry in WalkDir::new(root)
        .max_depth(depth)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| keep_entry(e))
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let name = entry.file_name().to_string_lossy();
        if !include.is_empty() && !matches_any(&name, path, include) {
            continue;
        }
        if matches_any(&name, path, exclude) {
            continue;
        }
        if let Ok(m) = entry.metadata() {
            if m.len() == 0 || m.len() > max_bytes {
                continue;
            }
        }
        out.push(path.to_path_buf());
    }
    out.sort();
    out
}

fn keep_entry(entry: &walkdir::DirEntry) -> bool {
    if entry.file_type().is_dir() && entry.depth() > 0 {
        let name = entry.file_name().to_string_lossy();
        !DEFAULT_EXCLUDES.iter().any(|x| *x == name.as_ref()) && !name.starts_with('.')
    } else {
        true
    }
}

/// Minimal glob matching: `*` matches any sequence, `?` one char. Patterns
/// match against the file name; absolute patterns (`/x/**`) match the path.
fn matches_any(name: &str, path: &Path, patterns: &[String]) -> bool {
    patterns.iter().any(|p| {
        if p.contains('/') {
            glob_match(p, &path.to_string_lossy())
        } else {
            glob_match(p, name)
        }
    })
}

pub fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    glob_rec(&p, 0, &t, 0)
}

fn glob_rec(p: &[char], mut pi: usize, t: &[char], mut ti: usize) -> bool {
    while pi < p.len() {
        match p[pi] {
            '*' => {
                // collapse consecutive stars
                while pi < p.len() && p[pi] == '*' {
                    pi += 1;
                }
                if pi == p.len() {
                    return true;
                }
                for skip in 0..=(t.len() - ti) {
                    if glob_rec(p, pi, t, ti + skip) {
                        return true;
                    }
                }
                return false;
            }
            '?' => {
                if ti >= t.len() {
                    return false;
                }
                pi += 1;
                ti += 1;
            }
            c => {
                if ti >= t.len() || t[ti] != c {
                    return false;
                }
                pi += 1;
                ti += 1;
            }
        }
    }
    ti == t.len()
}

/// SHA-256 of a file's contents (blocking — run inside `spawn_blocking`).
pub fn hash_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex::encode(h.finalize()))
}

pub fn hash_bytes(bytes: &[u8]) -> String {
    content_hash(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn glob_matching() {
        assert!(glob_match("*.pdf", "a.pdf"));
        assert!(glob_match("*.pdf", "a.b.pdf"));
        assert!(!glob_match("*.pdf", "a.txt"));
        assert!(glob_match("doc?.txt", "doc1.txt"));
        assert!(!glob_match("doc?.txt", "doc12.txt"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("/tmp/**", "/tmp/a/b/c"));
        assert!(glob_match("a*b*c", "abc"));
        assert!(glob_match("a*b*c", "axbyc"));
        assert!(!glob_match("a*b*c", "acb"));
    }

    #[test]
    fn scan_finds_files() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::create_dir_all(root.join("sub")).unwrap();
        fs::create_dir_all(root.join("node_modules")).unwrap();
        fs::create_dir_all(root.join(".hidden")).unwrap();
        fs::write(root.join("a.txt"), "x").unwrap();
        fs::write(root.join("b.pdf"), "y").unwrap();
        fs::write(root.join("sub/c.txt"), "z").unwrap();
        fs::write(root.join("node_modules/d.txt"), "w").unwrap();
        fs::write(root.join(".hidden/e.txt"), "v").unwrap();

        let all = scan_directory(root, true, &[], &[], u64::MAX);
        let names: Vec<_> = all
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, ["a.txt", "b.pdf", "c.txt"]); // hidden+node_modules skipped

        let pdfs = scan_directory(root, true, &["*.pdf".into()], &[], u64::MAX);
        assert_eq!(pdfs.len(), 1);

        let non_recursive = scan_directory(root, false, &[], &[], u64::MAX);
        assert_eq!(non_recursive.len(), 2);

        let excluded = scan_directory(root, true, &[], &["*.pdf".into()], u64::MAX);
        assert_eq!(excluded.len(), 2);
    }

    #[test]
    fn hash_is_content_based() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("f.bin");
        fs::write(&f, b"hello world").unwrap();
        assert_eq!(hash_file(&f).unwrap(), hash_bytes(b"hello world"));
        fs::write(&f, b"changed").unwrap();
        assert_ne!(hash_file(&f).unwrap(), hash_bytes(b"hello world"));
    }
}
