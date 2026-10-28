//! Lightweight tokenizer shared by the hashing embedder and tests.
//!
//! Splits on non-alphanumeric boundaries, lowercases, and hashes each token
//! into a fixed lexical vocabulary space used by the sparse representation.

/// Size of the sparse lexical vocabulary (token-id space).
pub const SPARSE_VOCAB: u32 = 30_000;

/// One token with its byte offsets in the source text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub text: String,
    /// Byte offset of the first byte of this token in the input.
    pub start: usize,
    /// Byte offset one past the last byte of this token.
    pub end: usize,
}

/// Stateless whitespace/punctuation tokenizer.
#[derive(Debug, Default, Clone, Copy)]
pub struct Tokenizer;

impl Tokenizer {
    pub fn new() -> Self {
        Tokenizer
    }

    /// Split `text` into lowercase alphanumeric tokens with byte offsets.
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut out = Vec::new();
        let mut start: Option<usize> = None;
        for (i, ch) in text.char_indices() {
            if ch.is_alphanumeric() {
                if start.is_none() {
                    start = Some(i);
                }
            } else if let Some(s) = start.take() {
                out.push(Token {
                    text: text[s..i].to_lowercase(),
                    start: s,
                    end: i,
                });
            }
        }
        if let Some(s) = start {
            out.push(Token {
                text: text[s..].to_lowercase(),
                start: s,
                end: text.len(),
            });
        }
        out
    }

    /// Convenience: token strings only.
    pub fn token_strings(&self, text: &str) -> Vec<String> {
        self.tokenize(text).into_iter().map(|t| t.text).collect()
    }
}

/// FNV-1a hash mapped into the sparse vocabulary space.
pub fn token_id(token: &str) -> u32 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in token.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (h % SPARSE_VOCAB as u64) as u32
}

/// Deterministic 64-bit seed for a token (used to derive its random basis
/// vectors — different domain than [`token_id`] to avoid correlation).
pub(crate) fn token_seed(token: &str, domain: u64) -> u64 {
    let mut h: u64 = 0x9e37_79b9_7f4a_7c15 ^ domain;
    for b in token.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        h ^= h >> 29;
    }
    h
}

/// SplitMix64 PRNG step — deterministic per-seed sequence.
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_basic() {
        let t = Tokenizer::new();
        let toks = t.tokenize("Hello, World! foo_bar 123");
        let strs: Vec<&str> = toks.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(strs, ["hello", "world", "foo", "bar", "123"]);
        // offsets point into the original text
        assert_eq!(&"Hello, World!"[toks[0].start..toks[0].end], "Hello");
    }

    #[test]
    fn tokenize_unicode_and_empty() {
        let t = Tokenizer::new();
        assert!(t.tokenize("").is_empty());
        assert!(t.tokenize(" \t\n").is_empty());
        assert_eq!(t.token_strings("Äpfel!")[0], "äpfel");
    }

    #[test]
    fn token_id_is_bounded_and_deterministic() {
        assert_eq!(token_id("abc"), token_id("abc"));
        assert!(token_id("abc") < SPARSE_VOCAB);
        assert_ne!(token_id("abc"), token_id("abd"));
    }
}
