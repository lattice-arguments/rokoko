//! Helpers on top of `ring_arithmetic`: the shared constants `ZERO`, `ONE`, `TWO` (and their
//! `_FE` field forms), `HALF_WAY_MOD_Q` (a multiple of `q` near `2^62`), centring to `i16`,
//! `inner_product`, `pow_mod` and `inv_mod` in `Z_q`, `field_to_ring_element` (one field
//! element in every slot) and `precompute_structured_values` (a `u64` tensor-product table).

use std::sync::LazyLock;

use crate::ring::{
    config::{DEGREE, MOD_Q, NUM_SLOTS, SLOT_DEGREE},
    kernels::{multiply_mod, sub_mod},
    ring_arithmetic::{incomplete_ntt_multiplication, FieldExtension, Representation, RingElement},
};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::arch::x86_64::{
    __m128i, __m512i, _mm512_cmpgt_epu64_mask, _mm512_cvtepi64_epi16, _mm512_load_si512,
    _mm512_mask_sub_epi64, _mm512_set1_epi64, _mm_store_si128,
};

pub static HALF_WAY_MOD_Q: LazyLock<u64> = LazyLock::new(|| {
    let budget = u64::MAX / (MOD_Q * 4);
    budget * MOD_Q
});

pub static HALF_WAY_MOD_Q_RING_CF: LazyLock<RingElement> =
    LazyLock::new(|| RingElement::all(*HALF_WAY_MOD_Q, Representation::Coefficients));

#[inline(always)]
pub fn centered_i64_from_u64_mod_q_scalar(x: u64) -> i64 {
    let half_q = MOD_Q >> 1;
    if x > half_q {
        x.wrapping_sub(MOD_Q) as i64
    } else {
        x as i64
    }
}

#[inline(always)]
pub fn centered_i16_from_u64_mod_q(dst: &mut [i16; DEGREE], src: &[u64; DEGREE]) {
    #[cfg(feature = "debug-decomp")]
    for &x in src.iter() {
        let c = centered_i64_from_u64_mod_q_scalar(x);
        assert!(
            c >= i16::MIN as i64 && c <= i16::MAX as i64,
            "i16 narrowing overflow: {}",
            c
        );
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe {
        let vq = _mm512_set1_epi64(MOD_Q as i64);
        let vhalfq = _mm512_set1_epi64((MOD_Q >> 1) as i64);
        for k in 0..DEGREE / 8 {
            let a = _mm512_load_si512(src.as_ptr().add(k * 8) as *const __m512i);
            let neg = _mm512_cmpgt_epu64_mask(a, vhalfq);
            let s = _mm512_mask_sub_epi64(a, neg, a, vq);
            let w = _mm512_cvtepi64_epi16(s);
            _mm_store_si128(dst.as_mut_ptr().add(k * 8) as *mut __m128i, w);
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    for (d, &x) in dst.iter_mut().zip(src.iter()) {
        let c = centered_i64_from_u64_mod_q_scalar(x);
        debug_assert!(c >= i16::MIN as i64 && c <= i16::MAX as i64);
        *d = c as i16;
    }
}

pub fn inner_product(a: &Vec<RingElement>, b: &Vec<RingElement>) -> RingElement {
    debug_assert_eq!(a.len(), b.len());
    let mut result = RingElement::zero(Representation::IncompleteNTT);
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    for (x, y) in a.iter().zip(b.iter()) {
        incomplete_ntt_multiplication(&mut temp, x, y);
        result += &temp;
    }
    result
}

#[inline]
pub fn inner_product_into(r: &mut RingElement, a: &Vec<RingElement>, b: &Vec<RingElement>) {
    debug_assert_eq!(a.len(), b.len());
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    for (x, y) in a.iter().zip(b.iter()) {
        incomplete_ntt_multiplication(&mut temp, x, y);
        *r += &temp;
    }
}

pub fn pow_mod(base: u64, mut exp: u64) -> u64 {
    let q = crate::ring::config::MOD_Q as u128;
    let mut acc = 1u128;
    let mut b = base as u128 % q;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = acc * b % q;
        }
        b = b * b % q;
        exp >>= 1;
    }
    acc as u64
}

pub fn inv_mod(a: u64) -> u64 {
    pow_mod(a, crate::ring::config::MOD_Q - 2)
}

#[inline]
pub fn field_to_ring_element(fe: &FieldExtension) -> RingElement {
    let mut result = RingElement::zero(Representation::HomogenizedFieldExtensions);
    for i in 0..SLOT_DEGREE {
        for j in 0..NUM_SLOTS {
            result.v[j + i * NUM_SLOTS] += fe.coeffs[i];
        }
    }
    result
}

#[inline]
pub fn field_to_ring_element_into(r: &mut RingElement, fe: &FieldExtension) {
    for i in 0..SLOT_DEGREE {
        for j in 0..NUM_SLOTS {
            r.v[j + i * NUM_SLOTS] = fe.coeffs[i];
        }
    }
    r.representation = Representation::HomogenizedFieldExtensions;
}

pub static ONE: LazyLock<RingElement> =
    LazyLock::new(|| RingElement::one(Representation::IncompleteNTT));

pub static ALL_ONE_COEFFS: LazyLock<RingElement> =
    LazyLock::new(|| RingElement::all(1, Representation::IncompleteNTT));

pub static TWO: LazyLock<RingElement> =
    LazyLock::new(|| RingElement::constant(2, Representation::IncompleteNTT));

pub static ZERO: LazyLock<RingElement> =
    LazyLock::new(|| RingElement::zero(Representation::IncompleteNTT));

pub static ONE_FE: LazyLock<FieldExtension> = LazyLock::new(|| FieldExtension::from_base(1));
pub static TWO_FE: LazyLock<FieldExtension> = LazyLock::new(|| FieldExtension::from_base(2));
pub static ZERO_FE: LazyLock<FieldExtension> = LazyLock::new(|| FieldExtension::from_base(0));

// this is only for u64
pub fn precompute_structured_values(layers: &[u64]) -> Vec<u64> {
    let size = 1 << layers.len();
    let mut values = vec![1u64; size];

    for (layer_idx, &layer) in layers.iter().rev().enumerate() {
        let layer_complement = unsafe { sub_mod(1, layer, MOD_Q) };

        for i in 0..size {
            if (i >> layer_idx) & 1 == 1 {
                unsafe {
                    values[i] = multiply_mod(values[i], layer, MOD_Q);
                }
            } else {
                unsafe {
                    values[i] = multiply_mod(values[i], layer_complement, MOD_Q);
                }
            }
        }
    }

    values
}

// Vectorized version using eltwise_mult_mod for better performance
pub fn precompute_structured_values_fast(layers: &[u64]) -> Vec<u64> {
    let size = 1 << layers.len();
    let mut values = vec![1u64; size];

    for (layer_idx, &layer) in layers.iter().rev().enumerate() {
        let layer_complement = unsafe { sub_mod(1, layer, MOD_Q) };
        let chunk_size = 1 << (layer_idx + 1);
        let half_chunk = 1 << layer_idx;

        // Process in chunks where bit pattern is uniform
        for chunk_start in (0..size).step_by(chunk_size) {
            // First half of chunk (bit layer_idx = 0): multiply by layer_complement
            let start_0 = chunk_start;
            let end_0 = chunk_start + half_chunk;

            // Second half of chunk (bit layer_idx = 1): multiply by layer
            let start_1 = chunk_start + half_chunk;
            let end_1 = chunk_start + chunk_size;

            // Multiply in-place by scalar
            for i in start_0..end_0 {
                unsafe {
                    values[i] = multiply_mod(values[i], layer_complement, MOD_Q);
                }
            }

            for i in start_1..end_1 {
                unsafe {
                    values[i] = multiply_mod(values[i], layer, MOD_Q);
                }
            }
        }
    }

    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::structured_row::{PreprocessedRow, StructuredRow};

    #[test]
    fn test_precompute_structured_values() {
        use crate::common::hash::HashWrapper;

        // Test with different layer sizes
        for num_layers in 1..=10 {
            let mut hash = HashWrapper::new();
            let layers: Vec<u64> = (0..num_layers).map(|_| hash.sample_u64_mod_q()).collect();

            let result_slow = precompute_structured_values(&layers);
            let result_fast = precompute_structured_values_fast(&layers);

            debug_assert_eq!(
                result_slow.len(),
                result_fast.len(),
                "Length mismatch for {} layers",
                num_layers
            );

            for (i, (slow, fast)) in result_slow.iter().zip(result_fast.iter()).enumerate() {
                debug_assert_eq!(
                    slow, fast,
                    "Mismatch at index {} for {} layers: slow={}, fast={}",
                    i, num_layers, slow, fast
                );
            }
        }
    }

    #[test]
    fn test_precompute_structured_values_properties() {
        use crate::common::hash::HashWrapper;

        let mut hash = HashWrapper::new();
        let layers: Vec<u64> = (0..5).map(|_| hash.sample_u64_mod_q()).collect();
        let values = precompute_structured_values_fast(&layers);

        // Size should be 2^k for k layers
        debug_assert_eq!(values.len(), 1 << layers.len());

        // Test specific properties: values[i] should match the tensor product computation
        // For index i with binary representation b_k...b_1b_0:
        // values[i] = product of (layer[j] if b_j=1, else (1-layer[j]))

        let manual_compute = |index: usize| -> u64 {
            let mut result = 1u64;
            for (bit_pos, &layer) in layers.iter().rev().enumerate() {
                if (index >> bit_pos) & 1 == 1 {
                    unsafe {
                        result = multiply_mod(result, layer, MOD_Q);
                    }
                } else {
                    unsafe {
                        result = multiply_mod(result, sub_mod(1, layer, MOD_Q), MOD_Q);
                    }
                }
            }
            result
        };

        for i in 0..values.len() {
            debug_assert_eq!(
                values[i],
                manual_compute(i),
                "Value mismatch at index {} (binary: {:05b})",
                i,
                i
            );
        }
    }

    #[test]
    fn test_precompute_structured_values_mathces_preprocessed_row() {
        let layers = vec![2u64, 3u64, 5u64];
        let layers_ring = layers
            .iter()
            .map(|&l| RingElement::constant(l, Representation::IncompleteNTT))
            .collect::<Vec<RingElement>>();

        let structure_row = StructuredRow {
            tensor_layers: layers_ring,
        };
        let preprocessed_row = PreprocessedRow::from_structured_row(&structure_row);

        let precomputed_values = precompute_structured_values_fast(&layers);
        let precomputed_values_ring = precomputed_values
            .iter()
            .map(|&v| RingElement::constant(v, Representation::IncompleteNTT))
            .collect::<Vec<RingElement>>();

        debug_assert_eq!(
            preprocessed_row.preprocessed_row.len(),
            precomputed_values_ring.len()
        );
        for i in 0..preprocessed_row.preprocessed_row.len() {
            debug_assert_eq!(
                preprocessed_row.preprocessed_row[i],
                precomputed_values_ring[i],
            );
        }
    }

    #[test]
    fn test_field_to_ring_roundtrip() {
        let fe = FieldExtension {
            coeffs: std::array::from_fn(|k| 123456789 + 864197532 * k as u64),
        };
        let re = field_to_ring_element(&fe);
        let fes = re.split_into_field_extensions();
        for f in fes {
            debug_assert_eq!(f, fe);
        }
    }
}
