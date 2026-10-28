use crate::embedder::Embedder;
use crate::tokenizer::{splitmix64, token_id, token_seed, Tokenizer};
use crate::{COLBERT_DIM, DENSE_DIM};
use async_trait::async_trait;
use ragnarok_core::{EmbeddingTriple, SparseVector};

const DENSE_DOMAIN: u64 = 0x64_65_6e_73_65; // "dense"
const COLBERT_DOMAIN: u64 = 0x63_6f_6c_62_65_72_74; // "colbert"

/// Deterministic, dependency-free embedder fulfilling the [`Embedder`]
/// contract.
///
/// Each token owns a fixed pseudo-random basis vector; the dense embedding is
/// the normalized weighted mean over its tokens. This is *not* a semantic
/// model, but it preserves lexical similarity (texts sharing tokens score
/// higher), which keeps the whole indexing/search pipeline exercisable
/// without a multi-GB model download.
pub struct HashingEmbedder {
    tokenizer: Tokenizer,
    /// Cap on ColBERT token vectors per text.
    max_colbert_tokens: usize,
}

impl HashingEmbedder {
    pub fn new(max_colbert_tokens: usize) -> Self {
        HashingEmbedder {
            tokenizer: Tokenizer::new(),
            max_colbert_tokens: max_colbert_tokens.max(1),
        }
    }

    /// Deterministic pseudo-random unit-ish vector for one token.
    fn basis_vector(token: &str, dim: usize, domain: u64, salt: u64) -> Vec<f32> {
        let mut state = token_seed(token, domain) ^ salt;
        (0..dim)
            .map(|_| {
                // map to approx. N(0,1) via sum of two uniforms
                let a = (splitmix64(&mut state) >> 40) as f32 / (1u64 << 24) as f32;
                let b = (splitmix64(&mut state) >> 40) as f32 / (1u64 << 24) as f32;
                a + b - 1.0
            })
            .collect()
    }
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

impl Default for HashingEmbedder {
    fn default() -> Self {
        Self::new(64)
    }
}

#[async_trait]
impl Embedder for HashingEmbedder {
    fn name(&self) -> &'static str {
        "hashing"
    }

    async fn embed(&self, text: &str) -> ragnarok_core::Result<EmbeddingTriple> {
        let tokens = self.tokenizer.token_strings(text);
        if tokens.is_empty() {
            // Keep the contract: return zero-ish but well-formed vectors.
            return Ok(EmbeddingTriple {
                dense: vec![0.0; DENSE_DIM],
                sparse: SparseVector::new(),
                colbert: vec![vec![0.0; COLBERT_DIM]],
            });
        }

        // term frequencies
        let mut tf: SparseVector = SparseVector::new();
        for t in &tokens {
            *tf.entry(token_id(t)).or_insert(0.0) += 1.0;
        }
        let max_tf = tf.values().copied().fold(0.0f32, f32::max).max(1.0);

        // sparse: normalized tf in (0,1]
        let sparse: SparseVector = tf.iter().map(|(k, v)| (*k, *v / max_tf)).collect();

        // dense: weighted mean of per-token basis vectors
        let mut dense = vec![0.0f32; DENSE_DIM];
        for t in &tokens {
            let id = token_id(t);
            let w = tf[&id] / max_tf;
            let basis = Self::basis_vector(t, DENSE_DIM, DENSE_DOMAIN, 0);
            for (d, b) in dense.iter_mut().zip(basis.iter()) {
                *d += b * w;
            }
        }
        let scale = (tokens.len() as f32).sqrt();
        dense.iter_mut().for_each(|x| *x /= scale);
        l2_normalize(&mut dense);

        // colbert: one contextualized vector per token (position salted),
        // capped at max_colbert_tokens.
        let colbert = tokens
            .iter()
            .take(self.max_colbert_tokens)
            .enumerate()
            .map(|(pos, t)| {
                let mut v = Self::basis_vector(t, COLBERT_DIM, COLBERT_DOMAIN, pos as u64);
                l2_normalize(&mut v);
                v
            })
            .collect();

        Ok(EmbeddingTriple {
            dense,
            sparse,
            colbert,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SPARSE_VOCAB;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[tokio::test]
    async fn contract_dimensions_and_norms() {
        let e = HashingEmbedder::default();
        let t = e.embed("the quick brown fox jumps over the lazy dog").await.unwrap();
        assert_eq!(t.dense.len(), DENSE_DIM);
        let norm: f32 = t.dense.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        assert!(!t.sparse.is_empty());
        assert!(t.sparse.keys().all(|k| *k < SPARSE_VOCAB));
        assert!(t.sparse.values().all(|w| *w > 0.0 && *w <= 1.0));
        assert_eq!(t.colbert.len(), 9);
        for row in &t.colbert {
            assert_eq!(row.len(), COLBERT_DIM);
            let n: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-5);
        }
    }

    #[tokio::test]
    async fn deterministic() {
        let e = HashingEmbedder::default();
        let a = e.embed("ragnarok test").await.unwrap();
        let b = e.embed("ragnarok test").await.unwrap();
        assert_eq!(a.dense, b.dense);
        assert_eq!(a.sparse, b.sparse);
        assert_eq!(a.colbert, b.colbert);
    }

    #[tokio::test]
    async fn lexical_similarity_is_preserved() {
        let e = HashingEmbedder::default();
        let a = e.embed("cats are fluffy animals").await.unwrap();
        let b = e.embed("cats are cute animals").await.unwrap();
        let c = e.embed("quantum chromodynamics lecture").await.unwrap();
        let sim_ab = cosine(&a.dense, &b.dense);
        let sim_ac = cosine(&a.dense, &c.dense);
        assert!(sim_ab > sim_ac, "{sim_ab} should exceed {sim_ac}");
        // sparse overlap for shared tokens
        assert!(a.sparse.keys().any(|k| b.sparse.contains_key(k)));
    }

    #[tokio::test]
    async fn empty_text_keeps_contract() {
        let e = HashingEmbedder::default();
        let t = e.embed("   ").await.unwrap();
        assert_eq!(t.dense.len(), DENSE_DIM);
        assert_eq!(t.colbert[0].len(), COLBERT_DIM);
        assert!(t.sparse.is_empty());
    }

    #[tokio::test]
    async fn colbert_cap() {
        let e = HashingEmbedder::new(4);
        let t = e.embed("one two three four five six seven eight").await.unwrap();
        assert_eq!(t.colbert.len(), 4);
    }

    #[tokio::test]
    async fn batch_matches_single() {
        let e = HashingEmbedder::default();
        let texts = vec!["alpha".to_string(), "beta".to_string()];
        let batch = e.embed_batch(&texts).await.unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].dense, e.embed("alpha").await.unwrap().dense);
    }
}
