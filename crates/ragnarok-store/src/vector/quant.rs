use ragnarok_core::{Error, Result};
use serde::{Deserialize, Serialize};

/// int8 scalar-quantized vector: `x ≈ q * scale + offset`, `q ∈ [-127, 127]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizedVec {
    pub data: Vec<i8>,
    pub scale: f32,
    pub offset: f32,
}

/// Per-vector scalar quantization (Qdrant-style "scalar int8, kept in RAM").
pub struct ScalarQuantizer {
    pub dim: usize,
}

impl ScalarQuantizer {
    pub fn new(dim: usize) -> Self {
        ScalarQuantizer { dim }
    }

    pub fn quantize(&self, v: &[f32]) -> Result<QuantizedVec> {
        if v.len() != self.dim {
            return Err(Error::VectorStore(format!(
                "dense dim mismatch: expected {}, got {}",
                self.dim,
                v.len()
            )));
        }
        let (mut min, mut max) = (f32::MAX, f32::MIN);
        for &x in v {
            if x < min {
                min = x;
            }
            if x > max {
                max = x;
            }
        }
        let range = max - min;
        if range <= f32::EPSILON {
            return Ok(QuantizedVec {
                data: vec![0; self.dim],
                scale: 1.0,
                offset: min,
            });
        }
        // map [min, max] -> [-127, 127]
        let scale = range / 254.0;
        let data = v
            .iter()
            .map(|&x| (((x - min) / scale) - 127.0).round().clamp(-127.0, 127.0) as i8)
            .collect();
        Ok(QuantizedVec {
            data,
            scale,
            offset: min + 127.0 * scale,
        })
    }

    pub fn dequantize(&self, q: &QuantizedVec) -> Vec<f32> {
        q.data
            .iter()
            .map(|&x| x as f32 * q.scale + q.offset)
            .collect()
    }

    /// Cosine between a raw query and a quantized doc.
    pub fn cosine(&self, query: &[f32], q: &QuantizedVec) -> f32 {
        let doc = self.dequantize(q);
        let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
        for (a, b) in query.iter().zip(doc.iter()) {
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        let denom = na.sqrt() * nb.sqrt();
        if denom > 0.0 {
            dot / denom
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_error_is_bounded() {
        let q = ScalarQuantizer::new(4);
        let v = vec![-1.0, 0.25, 0.5, 1.0];
        let qv = q.quantize(&v).unwrap();
        let back = q.dequantize(&qv);
        for (a, b) in v.iter().zip(back.iter()) {
            // quantization step is range/254 -> error <= step/2
            assert!((a - b).abs() <= (2.0 / 254.0) + 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn constant_vector() {
        let q = ScalarQuantizer::new(3);
        let qv = q.quantize(&[2.0, 2.0, 2.0]).unwrap();
        assert!(qv.data.iter().all(|&x| x == 0));
        let back = q.dequantize(&qv);
        assert!((back[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn dim_mismatch_rejected() {
        let q = ScalarQuantizer::new(3);
        assert!(q.quantize(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn cosine_on_quantized() {
        let q = ScalarQuantizer::new(2);
        let qv = q.quantize(&[1.0, 0.0]).unwrap();
        let sim = q.cosine(&[1.0, 0.0], &qv);
        assert!((sim - 1.0).abs() < 1e-6);
        let sim_orth = q.cosine(&[0.0, 1.0], &qv);
        assert!(sim_orth.abs() < 0.01);
    }
}
