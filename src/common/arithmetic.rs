use std::sync::LazyLock;

use crate::{
    common::{
        config::{DEGREE, HALF_DEGREE, MOD_Q},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{
            incomplete_ntt_multiplication, QuadraticExtension, Representation, RingElement,
            SHIFT_FACTORS,
        },
        structured_row::{PreprocessedRow, StructuredRow},
        sumcheck_element::SumcheckElement,
    },
    hexl::bindings::{multiply_mod, sub_mod},
};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::arch::x86_64::{
    __m128i, __m256i, __m512i, __mmask8, _mm256_add_epi16, _mm256_castsi128_si256,
    _mm256_inserti128_si256, _mm256_loadu_si256, _mm256_setzero_si256, _mm256_storeu_si256,
    _mm256_sub_epi16, _mm512_add_epi16, _mm512_cmpgt_epu64_mask, _mm512_cvtepi64_epi16,
    _mm512_cvtsepi64_epi16, _mm512_loadu_si512, _mm512_mask_sub_epi64, _mm512_set1_epi64,
    _mm512_setzero_si512, _mm512_storeu_si512, _mm512_sub_epi16, _mm_storeu_si128,
};

#[inline(always)]
pub fn centered_i64_from_u64_mod_q_scalar(x: u64) -> i64 {
    let half_q = MOD_Q >> 1;
    if x > half_q {
        x.wrapping_sub(MOD_Q) as i64
    } else {
        x as i64
    }
}

// load 16x i64 registers, pack into 16x i16 register with _mm512_cvtsepi64_epi16
// directly loads the low 16 bit of i64 registers (signed truncating)
#[inline(always)]
pub fn pack_i64_to_i16_deg16(dst: &mut [i16], src: &[i64]) {
    assert_eq!(dst.len(), src.len());
    debug_assert!(src.len() % 16 == 0);

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        let mut i = 0usize;
        for k in (0..src.len() / 16) {
            unsafe {
                let i = k * 16;
                // Load 16 i64 (two zmm registers of 8 i64)
                let a0 = _mm512_loadu_si512(src.as_ptr().add(i) as *const __m512i);
                let a1 = _mm512_loadu_si512(src.as_ptr().add(i + 8) as *const __m512i);

                // Narrow 8 i64 -> 8 i16 (signed truncating), result is 128-bit each
                let w0: __m128i = _mm512_cvtepi64_epi16(a0);
                let w1: __m128i = _mm512_cvtepi64_epi16(a1);

                _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, w0);
                _mm_storeu_si128(dst.as_mut_ptr().add(i + 8) as *mut __m128i, w1);
            }
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        debug_assert!(s >= i16::MIN as i64 && s <= i16::MAX as i64);
        *d = s as i16;
    }
    return;
}

// in_u64[i] ∈ [0, Q]
// out_i64[i] ∈ [-Q/2, Q/2)
#[inline(always)]
pub fn centered_coeffs_u64_to_i64_inplace(out_i64: &mut [i64; DEGREE], in_u64: &[u64; DEGREE]) {
    assert_eq!(out_i64.len(), in_u64.len());

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe {
        let half_q = MOD_Q >> 1;
        let vq = _mm512_set1_epi64(MOD_Q as i64);
        let vhalfq = _mm512_set1_epi64(half_q as i64);

        let mut i = 0usize;
        let n = in_u64.len();

        // 8 u64 lanes per __m512i
        for k in 0..(n / 8) {
            let i = k * 8;
            let a = _mm512_loadu_si512(in_u64.as_ptr().add(i) as *const __m512i);

            // neg lanes are ones where x > halfQ
            let neg: __mmask8 = _mm512_cmpgt_epu64_mask(a, vhalfq);

            // if neg: x = x - Q (wraps in u64 domain; interpreting as i64 gives negative)
            let signed = _mm512_mask_sub_epi64(a, neg, a, vq);

            // store as i64
            _mm512_storeu_si512(out_i64.as_mut_ptr().add(i) as *mut __m512i, signed);
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    for (dst, &src) in out_i64.iter_mut().zip(in_u64.iter()) {
        *dst = centered_i64_from_u64_mod_q_scalar(src);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub fn project_one_row_i16_to_u64<const DEGREE: usize>(
    subwitness_i16: &[[i16; DEGREE]], // len = projection_ratio*H
    projection_matrix: &ProjectionMatrix,
    inner_row: usize,
    out_u64: &mut [u64; DEGREE],
) {
    use crate::hexl::bindings::eltwise_reduce_mod;

    let row_len = projection_matrix.projection_ratio * projection_matrix.projection_height;
    debug_assert_eq!(subwitness_i16.len(), row_len);
    debug_assert!(DEGREE % 16 == 0);

    let mut k = 0usize;
    let mut pos_idx = Vec::new();
    let mut neg_idx = Vec::new();
    for i in 0..row_len {
        let (is_positive, is_non_zero) = projection_matrix[(inner_row, i)];
        if !is_non_zero {
            continue;
        }
        if is_positive {
            pos_idx.push(i);
        } else {
            neg_idx.push(i);
        }
    }

    unsafe {
        for j in 0..(DEGREE / 32) {
            let k = j * 32;
            let mut acc = _mm512_setzero_si512();

            for &i in &pos_idx {
                let v = _mm512_loadu_si512(subwitness_i16[i].as_ptr().add(k) as *const __m512i);
                acc = _mm512_add_epi16(acc, v);
            }

            for &i in &neg_idx {
                let v = _mm512_loadu_si512(subwitness_i16[i].as_ptr().add(k) as *const __m512i);
                acc = _mm512_sub_epi16(acc, v);
            }

            convert_i16_as_u64(out_u64.as_mut_ptr().add(k), acc);
        }

        eltwise_reduce_mod(
            out_u64.as_mut_ptr(),
            out_u64.as_ptr(),
            out_u64.len() as u64,
            MOD_Q,
        );
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn convert_i16_as_u64(dst_u64: *mut u64, v16x32: __m512i) {
    unsafe {
        let mut tmp = [0i16; 32];
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, v16x32);

        let q_i64 = MOD_Q as i64;
        for lane in 0..32 {
            let mut r = tmp[lane] as i64;
            if r < 0 {
                r += q_i64;
            }
            *dst_u64.add(lane) = r as u64;
        }
    }
}

#[inline]
pub fn inner_product(a: &Vec<RingElement>, b: &Vec<RingElement>) -> RingElement {
    assert_eq!(a.len(), b.len());
    let mut result = RingElement::zero(Representation::IncompleteNTT);
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    for (x, y) in a.iter().zip(b.iter()) {
        incomplete_ntt_multiplication(&mut temp, x, y);
        result += &temp;
    }
    result
}

#[inline]
pub fn inner_product_into(mut r: &mut RingElement, a: &Vec<RingElement>, b: &Vec<RingElement>) {
    assert_eq!(a.len(), b.len());
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    for (x, y) in a.iter().zip(b.iter()) {
        incomplete_ntt_multiplication(&mut temp, x, y);
        *r += &temp;
    }
}

#[inline]
pub fn field_to_ring_element(fe: &QuadraticExtension) -> RingElement {
    let mut result = RingElement::zero(Representation::HomogenizedFieldExtensions);
    for i in 0..2 {
        for j in 0..HALF_DEGREE {
            result.v[j + i * HALF_DEGREE] += fe.coeffs[i];
        }
    }
    result
}

#[inline]
pub fn field_to_ring_element_into(mut r: &mut RingElement, fe: &QuadraticExtension) {
    for i in 0..2 {
        for j in 0..HALF_DEGREE {
            r.v[j + i * HALF_DEGREE] = fe.coeffs[i];
        }
    }
    r.representation = Representation::HomogenizedFieldExtensions;
}

pub static ONE: LazyLock<RingElement> =
    LazyLock::new(|| RingElement::one(Representation::IncompleteNTT));

pub static TWO: LazyLock<RingElement> =
    LazyLock::new(|| RingElement::constant(2, Representation::IncompleteNTT));

pub static ZERO: LazyLock<RingElement> =
    LazyLock::new(|| RingElement::zero(Representation::IncompleteNTT));

pub static ONE_QUAD: LazyLock<QuadraticExtension> =
    LazyLock::new(|| QuadraticExtension { coeffs: [1, 0] });
pub static TWO_QUAD: LazyLock<QuadraticExtension> =
    LazyLock::new(|| QuadraticExtension { coeffs: [2, 0] });
pub static ZERO_QUAD: LazyLock<QuadraticExtension> =
    LazyLock::new(|| QuadraticExtension { coeffs: [0, 0] });

#[test]
fn test_field_to_ring_roundtrip() {
    let fe = QuadraticExtension {
        coeffs: [123456789, 987654321],
    };
    let re = field_to_ring_element(&fe);
    let fes = re.split_into_quadratic_extensions();
    for f in fes {
        assert_eq!(f, fe);
    }
}

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
    // TODO: can we use preallocated pool here? Does it make sense?
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
                    // TODO: use vectorisation
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

#[test]
fn test_precompute_structured_values() {
    use crate::common::hash::HashWrapper;

    // Test with different layer sizes
    for num_layers in 1..=10 {
        let mut hash = HashWrapper::new();
        let layers: Vec<u64> = (0..num_layers).map(|_| hash.sample_u64_mod_q()).collect();

        let result_slow = precompute_structured_values(&layers);
        let result_fast = precompute_structured_values_fast(&layers);

        assert_eq!(
            result_slow.len(),
            result_fast.len(),
            "Length mismatch for {} layers",
            num_layers
        );

        for (i, (slow, fast)) in result_slow.iter().zip(result_fast.iter()).enumerate() {
            assert_eq!(
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
    assert_eq!(values.len(), 1 << layers.len());

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
        assert_eq!(
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

    assert_eq!(
        preprocessed_row.preprocessed_row.len(),
        precomputed_values_ring.len()
    );
    for i in 0..preprocessed_row.preprocessed_row.len() {
        assert_eq!(
            preprocessed_row.preprocessed_row[i],
            precomputed_values_ring[i],
        );
    }
}

pub static ADDITION_SUBTRACTION_BUDGET: LazyLock<u64> = LazyLock::new(|| u64::MAX / (MOD_Q * 4)); // if we start from number HALF_WAY_MOD_Q how many additions/subtractions (with elements in [0,MOD_Q)) can we do without overflowing u64?

pub static HALF_WAY_MOD_Q: LazyLock<u64> = LazyLock::new(|| {
    let budget = u64::MAX / (MOD_Q * 4);
    budget * MOD_Q
});

pub static HALF_WAY_MOD_Q_RING_CF: LazyLock<RingElement> =
    LazyLock::new(|| RingElement::all(*HALF_WAY_MOD_Q, Representation::Coefficients));
