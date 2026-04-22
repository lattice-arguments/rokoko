use crate::common::{
    matrix::new_vec_zero_preallocated,
    ring_arithmetic::{Representation, RingElement},
};

#[cfg(test)]
use crate::common::config::MOD_Q;

impl RingElement {
    pub fn bits_into(&mut self, target: &mut RingElement, from: u64, to: u64) {
        debug_assert!(from < to);

        let mask: u64 = (1u64 << (to - from)) - 1;
        for i in 0..self.v.len() {
            target.v[i] = (self.v[i] >> from) & mask;
        }
    }
}

// Decomposes each element in input into radix parts of base 2^{base_log} using signed (balanced) decomposition.
// Each element x is first shifted by adding k = (b/2) * (1 + b + b^2 + ... + b^{radix-1}) where b = 2^{base_log},
// then decomposed into radix base-b digits, and each digit is shifted back by subtracting b/2.
// This ensures each decomposed part lies in the range [-2^{base_log - 1}, 2^{base_log - 1}).
// Since k = (b/2) * Σ b^i, the recomposition is exact: Σ d_i * b^i = (x + k) - k = x, with zero offset.
pub fn decompose(input: &[RingElement], base_log: u64, radix: usize) -> Vec<RingElement> {
    let mut decomposed = new_vec_zero_preallocated(input.len() * radix);

    if base_log == 1 {
        decomposed.clone_from_slice(input);
        return decomposed;
    }

    let small_shift_val = 1u64 << (base_log - 1);
    let mut big_shift_val: u64 = 0;
    for i in 0..radix {
        big_shift_val += small_shift_val << (i as u64 * base_log);
    }
    let big_shift = RingElement::all(big_shift_val, Representation::EvenOddCoefficients);

    let small_shift = RingElement::all(1u64 << (base_log - 1), Representation::EvenOddCoefficients);

    let mut temp = RingElement::all(0, Representation::EvenOddCoefficients);

    for (index, el) in input.iter().enumerate() {
        temp.set_from(el);
        temp.to_representation(Representation::EvenOddCoefficients);
        temp += &big_shift;
        for i in 0..radix {
            // `bits_into` unconditionally overwrites every `v[..]` entry of
            // the target, so the usual `to_representation(EvenOdd)` NTT
            // inverse on the freshly-pooled (potentially stale) buffer is
            // pure waste: any data the inverse would produce is thrown away
            // on the very next line. Skipping that `HALF_DEGREE`-point NTT
            // twice per output chunk removes `radix` NTT inversions per
            // input element across `decompose`, which is one of the top
            // `ntt_inverse_in_place` callers in `perf`.
            let slot = &mut decomposed[index * radix + i];
            slot.representation = Representation::EvenOddCoefficients;
            temp.bits_into(slot, i as u64 * base_log, (i as u64 + 1) * base_log);
            *slot -= &small_shift;
            slot.to_representation(Representation::IncompleteNTT);
        }
        #[cfg(feature = "debug-decomp")]
        {
            // check that recomposition works
            let mut recomposed = RingElement::all(0, Representation::IncompleteNTT);
            for j in 0..radix {
                let mut term = decomposed[index * radix + j].clone();
                let shift = RingElement::constant(
                    1u64 << (j as u64 * base_log),
                    Representation::IncompleteNTT,
                );
                term *= &shift;
                recomposed += &term;
            }
            let el_incomplete_ntt = {
                let mut temp_el = el.clone();
                temp_el.to_representation(Representation::IncompleteNTT);
                temp_el
            };
            assert_eq!(&recomposed, &el_incomplete_ntt, "Recomposition failed in decomposition. Perhaps base_log and radix are not chosen properly?");
        }
    }

    decomposed
}

// Like decompose, but interleaves by digit index rather than by element.
// decompose([a, b], radix=2)        -> [a0, a1, b0, b1]
// decompose_chunks([a, b], radix=2) -> [a0, b0, a1, b1]
pub fn decompose_chunks_into(
    output: &mut [RingElement],
    input: &[RingElement],
    base_log: u64,
    radix: usize,
) {
    let mut flat = decompose(input, base_log, radix);
    let n = input.len();
    for index in 0..n {
        for i in 0..radix {
            std::mem::swap(&mut output[i * n + index], &mut flat[index * radix + i]);
        }
    }
}

// With the balanced decomposition using k = (b/2) * Σ b^i, the recomposition offset is zero.
// Kept for API compatibility.
pub fn get_composer_offset(_base_log: u64, _radix: usize) -> u64 {
    0
}

// With the balanced decomposition, the offset is zero, so the scaled version is also zero.
// Kept for API compatibility.
pub fn get_decomposed_offset_scaled(_base_log: u64, _radix: usize) -> u64 {
    0
}

pub fn compose_from_decomposed(
    decomposed: &[RingElement],
    base_log: u64,
    radix: usize,
) -> Vec<RingElement> {
    let mut recomposed = new_vec_zero_preallocated(decomposed.len() / radix);

    for i in 0..recomposed.len() {
        recomposed[i] = RingElement::all(0, Representation::IncompleteNTT);
        for j in 0..radix {
            let mut term = decomposed[i * radix + j].clone();
            let shift =
                RingElement::constant(1u64 << (j as u64 * base_log), Representation::IncompleteNTT);
            term *= &shift;
            recomposed[i] += &term;
        }
    }

    recomposed
}

#[test]
fn test_decompose() {
    let mut input = vec![RingElement::all(37, Representation::IncompleteNTT)];
    let base_log = 3; // base 8
    let radix = 4;
    let decomposed = decompose(&mut input, base_log, radix);
    debug_assert_eq!(
        input[0],
        RingElement::all(37, Representation::IncompleteNTT)
    );
    debug_assert_eq!(decomposed.len(), radix * 1);
    // k = 4 * (1 + 8 + 64 + 512) = 2340
    // 37 is shifted to 37 + 2340 = 2377
    // base 8 representation of 2377 = 4 * 8^3 + 5 * 8^2 + 1 * 8^1 + 1 * 8^0
    // so the decomposed elements should be [1, 1, 5, 4]
    // after removing the shift, they should be [1 - 4, 1 - 4, 5 - 4, 4 - 4] = [-3, -3, 1, 0]
    debug_assert_eq!(
        decomposed[0],
        RingElement::all(MOD_Q - 3, Representation::IncompleteNTT)
    );
    debug_assert_eq!(
        decomposed[1],
        RingElement::all(MOD_Q - 3, Representation::IncompleteNTT)
    );
    debug_assert_eq!(
        decomposed[2],
        RingElement::all(1, Representation::IncompleteNTT)
    );
    debug_assert_eq!(
        decomposed[3],
        RingElement::all(0, Representation::IncompleteNTT)
    );

    let mut recomposed = RingElement::all(0, Representation::IncompleteNTT);
    for i in 0..radix {
        let mut term = decomposed[i].clone();
        let shift =
            RingElement::constant(1u64 << (i as u64 * base_log), Representation::IncompleteNTT);
        term *= &shift;
        recomposed += &term;
    }
    debug_assert_eq!(recomposed, input[0]);
}

#[test]
fn test_random_mod_q() {
    let r = RingElement::random(Representation::IncompleteNTT);
    let data = vec![r];
    let base_log = 13; // do we cover 52 bits?
    let radix = 4;

    let mut decomposed = decompose(&data, base_log, radix);

    let mut recomposed = RingElement::all(0, Representation::IncompleteNTT);
    for i in 0..radix {
        let mut term = decomposed[i].clone();
        let shift =
            RingElement::constant(1u64 << (i as u64 * base_log), Representation::IncompleteNTT);
        term *= &shift;
        recomposed += &term;
    }
    debug_assert_eq!(recomposed, data[0]);

    let mut inf_norm = 0;
    for d in decomposed.iter_mut() {
        d.from_incomplete_ntt_to_even_odd_coefficients();
        for &v in d.v.iter() {
            let abs_v = if v > MOD_Q / 2 { MOD_Q - v } else { v };
            if abs_v > inf_norm {
                inf_norm = abs_v;
            }
        }
    }

    debug_assert_eq!(inf_norm < (1u64 << (base_log - 1)), true);
}

#[test]
fn test_compose_from_decomposed() {
    let mut input = vec![RingElement::all(37, Representation::IncompleteNTT)];
    let base_log = 3; // base 8
    let radix = 4;
    let decomposed = decompose(&mut input, base_log, radix);
    let recomposed = compose_from_decomposed(&decomposed, base_log, radix);
    debug_assert_eq!(recomposed[0], input[0]);
}
