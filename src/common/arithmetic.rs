use std::sync::LazyLock;

use crate::protocol::project_coarse::Signed16RingElement;
use crate::{
    common::{
        config::{DEGREE, HALF_DEGREE, MOD_Q},
        ring_arithmetic::{
            incomplete_ntt_multiplication, QuadraticExtension, Representation, RingElement,
        },
    },
    hexl::bindings::{multiply_mod, sub_mod},
};

pub static HALF_WAY_MOD_Q: LazyLock<u64> = LazyLock::new(|| {
    let budget = u64::MAX / (MOD_Q * 4);
    budget * MOD_Q
});

pub static HALF_WAY_MOD_Q_RING_CF: LazyLock<RingElement> =
    LazyLock::new(|| RingElement::all(*HALF_WAY_MOD_Q, Representation::Coefficients));

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::arch::x86_64::{
    __m128i, __m512i, _mm512_add_epi16, _mm512_cmpgt_epu64_mask, _mm512_cvtepi16_epi64,
    _mm512_cvtepi64_epi16, _mm512_extracti32x4_epi32, _mm512_load_si512, _mm512_mask_add_epi64,
    _mm512_mask_sub_epi64, _mm512_movepi64_mask, _mm512_set1_epi64, _mm512_setzero_si512,
    _mm512_store_si512, _mm512_sub_epi16, _mm_store_si128,
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

/// Uses `i32` accumulators (to avoid `i16` overflow) and final modular
/// reduction to produce `u64` residues in `[0, Q)`.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub fn project_one_row_i16_to_u64<const DEGREE: usize>(
    subwitness_i16: &[Signed16RingElement],
    pos: &[u16],
    neg: &[u16],
    out_u64: &mut [u64; DEGREE],
) {
    debug_assert!(DEGREE % 16 == 0);

    let mut acc: [i32; DEGREE] = [0; DEGREE];

    for &i in pos {
        let row = &subwitness_i16[i as usize].0;
        for k in 0..DEGREE {
            acc[k] += row[k] as i32;
        }
    }

    for &i in neg {
        let row = &subwitness_i16[i as usize].0;
        for k in 0..DEGREE {
            acc[k] -= row[k] as i32;
        }
    }

    let q = MOD_Q as i64;

    for k in 0..DEGREE {
        let x = acc[k] as i64;
        let mut r = x % q;
        if r < 0 {
            r += q;
        }
        out_u64[k] = r as u64;
    }
}

// Centered i16 lanes -> canonical residues in [0, Q): sign-extend, add Q to negatives.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn convert_i16x32_to_u64_mod_q(dst_u64: *mut u64, v16x32: __m512i) {
    let q = _mm512_set1_epi64(MOD_Q as i64);
    macro_rules! part {
        ($p:literal) => {
            let x = _mm512_extracti32x4_epi32::<$p>(v16x32);
            let w = _mm512_cvtepi16_epi64(x);
            let neg = _mm512_movepi64_mask(w);
            let r = _mm512_mask_add_epi64(w, neg, w, q);
            _mm512_store_si512(dst_u64.add($p * 8) as *mut __m512i, r);
        };
    }
    part!(0);
    part!(1);
    part!(2);
    part!(3);
}

// Computes out[row] = sum of witness elements listed in pos[row] minus those
// in neg[row], coefficient-wise in i16, for one witness block.
//
// The loop nest exists to keep the loads cache-hot (L1, spill L2, never L3):
//   k    - one 32-lane (64 B) slice of the coefficients at a time. Each
//          witness element is 4 such lines; a k-pass touches exactly one
//          line per element.
//   tile - the witness is walked in 256-element windows: 256 x 64 B = 16 KB,
//          which stays L1-resident while...
//   row  - ...ALL output rows consume their entries falling inside the
//          window. So a witness line is fetched from memory once per k and
//          then reused ~H/2 times from L1.
//
// Because rows are revisited per tile, their partial sums cannot live in
// registers; they live in `scratch` (H x 64 B = 16 KB, also L1-resident).
// The offset lists are sorted, so `pos_cur`/`neg_cur` remember per row how
// far its list has been consumed; the next tile continues from there, and
// "entry belongs to this tile" is a single compare against the tile's end.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub fn project_rows_sparse_tiled<const DEGREE: usize>(
    subwitness_i16: &[Signed16RingElement],
    pos: &[u32],
    pos_bounds: &[usize],
    neg: &[u32],
    neg_bounds: &[usize],
    out: &mut [RingElement],
) {
    const TILE: usize = 256;

    #[repr(align(64))]
    #[derive(Clone, Copy)]
    struct Acc([i16; 32]);

    debug_assert!(DEGREE % 32 == 0);
    let h = out.len();
    debug_assert_eq!(pos_bounds.len(), h + 1);
    debug_assert_eq!(neg_bounds.len(), h + 1);
    let row_len = subwitness_i16.len();
    let elem_bytes = core::mem::size_of::<Signed16RingElement>();

    let mut scratch = vec![Acc([0i16; 32]); h];
    let mut pos_cur = vec![0usize; h];
    let mut neg_cur = vec![0usize; h];

    let base = subwitness_i16.as_ptr() as *const u8;

    unsafe {
        for k in 0..DEGREE / 32 {
            // The list entries are byte offsets of ELEMENTS; adding them to
            // `chunk` (base shifted to this k-slice) addresses the element's
            // k-th line directly.
            let chunk = base.add(k * 64);
            // Fresh coefficient slice: clear the partial sums, rewind every
            // row's cursor to the start of its list.
            scratch.fill(Acc([0i16; 32]));
            pos_cur.copy_from_slice(&pos_bounds[..h]);
            neg_cur.copy_from_slice(&neg_bounds[..h]);

            let mut tile_start = 0usize;
            while tile_start < row_len {
                // Tile boundary in the same units as the list entries (bytes).
                let tile_end = ((tile_start + TILE).min(row_len) * elem_bytes) as u32;
                for row in 0..h {
                    // Two independent accumulators so the adds and the
                    // subtracts form separate dependency chains: a0 continues
                    // this row's running sum, a1 collects the negatives, and
                    // a0 - a1 is stored back. i16 adds wrap mod 2^16, which
                    // is exact as long as the true sum fits i16 (the norm
                    // bounds guarantee that; debug-decomp checks it).
                    let mut a0 =
                        _mm512_load_si512(scratch[row].0.as_ptr() as *const __m512i);
                    let mut a1 = _mm512_setzero_si512();

                    // Consume this row's +1 entries that fall inside the
                    // tile. The load fuses into the add (one vpaddw with a
                    // memory operand) and stays cache-hot: the address is
                    // inside the tile (measured: ~78% of kernel loads L1-hit,
                    // the rest L2-hit, ~nothing reaches L3).
                    let end = pos_bounds[row + 1];
                    let mut i = *pos_cur.get_unchecked(row);
                    while i < end && *pos.get_unchecked(i) < tile_end {
                        a0 = _mm512_add_epi16(
                            a0,
                            _mm512_load_si512(
                                chunk.add(*pos.get_unchecked(i) as usize) as *const __m512i
                            ),
                        );
                        i += 1;
                    }
                    *pos_cur.get_unchecked_mut(row) = i;

                    // Same for the -1 entries.
                    let end = neg_bounds[row + 1];
                    let mut i = *neg_cur.get_unchecked(row);
                    while i < end && *neg.get_unchecked(i) < tile_end {
                        a1 = _mm512_add_epi16(
                            a1,
                            _mm512_load_si512(
                                chunk.add(*neg.get_unchecked(i) as usize) as *const __m512i
                            ),
                        );
                        i += 1;
                    }
                    *neg_cur.get_unchecked_mut(row) = i;

                    _mm512_store_si512(
                        scratch[row].0.as_mut_ptr() as *mut __m512i,
                        _mm512_sub_epi16(a0, a1),
                    );
                }
                tile_start += TILE;
            }

            // All tiles done: scratch holds the finished centered i16 sums
            // for this coefficient slice; lift them to residues in [0, Q)
            // and write them into the output elements.
            for row in 0..h {
                let acc = _mm512_load_si512(scratch[row].0.as_ptr() as *const __m512i);
                convert_i16x32_to_u64_mod_q(out[row].v.as_mut_ptr().add(k * 32), acc);
            }
        }
    }
}

/// Wrapper for _mm512_add_epi16 that checks for overflows in debug-decomp, otherwise just adds.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
pub unsafe fn add_epi16_checked(a: __m512i, b: __m512i) -> __m512i {
    #[cfg(feature = "debug-decomp")]
    {
        use std::arch::x86_64::{_mm512_add_epi16, _mm512_cmpgt_epi16_mask, _mm512_set1_epi16};
        let sum = _mm512_add_epi16(a, b);
        let sign_a = _mm512_cmpgt_epi16_mask(a, _mm512_set1_epi16(-1));
        let sign_b = _mm512_cmpgt_epi16_mask(b, _mm512_set1_epi16(-1));
        let sign_sum = _mm512_cmpgt_epi16_mask(sum, _mm512_set1_epi16(-1));
        let same_sign = !(sign_a ^ sign_b); // 1 where same sign
        let overflow = same_sign & (sign_a ^ sign_sum); // 1 where overflow
        if overflow != 0 {
            panic!(
                "add_epi16_checked: overflow detected in SIMD lane(s): {:032b}",
                overflow
            );
        }
        sum
    }
    #[cfg(not(feature = "debug-decomp"))]
    {
        use std::arch::x86_64::_mm512_add_epi16;
        _mm512_add_epi16(a, b)
    }
}

/// Wrapper for _mm512_sub_epi16 that checks for overflows in debug-decomp, otherwise just subtracts.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
pub unsafe fn sub_epi16_checked(a: __m512i, b: __m512i) -> __m512i {
    #[cfg(feature = "debug-decomp")]
    {
        use std::arch::x86_64::{_mm512_cmpgt_epi16_mask, _mm512_set1_epi16, _mm512_sub_epi16};
        let diff = _mm512_sub_epi16(a, b);
        let sign_a = _mm512_cmpgt_epi16_mask(a, _mm512_set1_epi16(-1));
        let sign_b = _mm512_cmpgt_epi16_mask(b, _mm512_set1_epi16(-1));
        let sign_diff = _mm512_cmpgt_epi16_mask(diff, _mm512_set1_epi16(-1));
        let diff_sign = sign_a ^ sign_b; // 1 where different sign
        let overflow = diff_sign & (sign_a ^ sign_diff); // 1 where overflow
        if overflow != 0 {
            panic!(
                "sub_epi16_checked: overflow detected in SIMD lane(s): {:032b}",
                overflow
            );
        }
        diff
    }
    #[cfg(not(feature = "debug-decomp"))]
    {
        use std::arch::x86_64::_mm512_sub_epi16;
        _mm512_sub_epi16(a, b)
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
    let q = crate::common::config::MOD_Q as u128;
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
    pow_mod(a, crate::common::config::MOD_Q - 2)
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
pub fn field_to_ring_element_into(r: &mut RingElement, fe: &QuadraticExtension) {
    for i in 0..2 {
        for j in 0..HALF_DEGREE {
            r.v[j + i * HALF_DEGREE] = fe.coeffs[i];
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

pub static ONE_QUAD: LazyLock<QuadraticExtension> =
    LazyLock::new(|| QuadraticExtension { coeffs: [1, 0] });
pub static TWO_QUAD: LazyLock<QuadraticExtension> =
    LazyLock::new(|| QuadraticExtension { coeffs: [2, 0] });
pub static ZERO_QUAD: LazyLock<QuadraticExtension> =
    LazyLock::new(|| QuadraticExtension { coeffs: [0, 0] });

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
        let fe = QuadraticExtension {
            coeffs: [123456789, 987654321],
        };
        let re = field_to_ring_element(&fe);
        let fes = re.split_into_quadratic_extensions();
        for f in fes {
            debug_assert_eq!(f, fe);
        }
    }
}
