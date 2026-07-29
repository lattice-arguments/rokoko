use crate::common::{
    arithmetic::pow_mod,
    config::{DEGREE, MOD_Q},
    ring_arithmetic::{Representation, RingElement},
};

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
    let mut decomposed = vec![RingElement::zero(Representation::IncompleteNTT); input.len() * radix];

    if base_log == 1 {
        assert_eq!(
            radix, 1,
            "balanced base-2 decomposition is not supported; use decompose_bits"
        );
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

    #[cfg(feature = "debug-decomp")]
    let mut call_max: i64 = 0;

    for (index, el) in input.iter().enumerate() {
        temp.set_from(el);
        temp.to_representation(Representation::EvenOddCoefficients);
        #[cfg(feature = "debug-decomp")]
        {
            let q = crate::common::config::MOD_Q;
            for &c in temp.v.iter() {
                let s = if c > q / 2 { c as i64 - q as i64 } else { c as i64 };
                call_max = call_max.max(s.abs());
            }
        }
        temp += &big_shift;
        for i in 0..radix {
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
            if recomposed != el_incomplete_ntt {
                let mut coeffs = el.clone();
                coeffs.to_representation(Representation::EvenOddCoefficients);
                let centered: Vec<i64> = coeffs
                    .v
                    .iter()
                    .map(|&c| {
                        if c > crate::common::config::MOD_Q / 2 {
                            c as i64 - crate::common::config::MOD_Q as i64
                        } else {
                            c as i64
                        }
                    })
                    .collect();
                let max_abs = centered.iter().map(|c| c.abs()).max().unwrap();
                panic!(
                    "Recomposition failed: index={} of {} base_log={} radix={} max|coeff|={}",
                    index,
                    input.len(),
                    base_log,
                    radix,
                    max_abs
                );
            }
        }
    }

    #[cfg(feature = "debug-decomp")]
    {
        let capacity_neg = (big_shift_val as i64).min(1 << 62);
        let capacity_pos = ((1u64 << (base_log * radix as u64)).wrapping_sub(big_shift_val))
            .min(1 << 62) as i64
            - 1;
        println!(
            "  [debug] decompose base={} radix={} n={} max|coeff|={} window=[-{}, {}]",
            base_log,
            radix,
            input.len(),
            call_max,
            capacity_neg,
            capacity_pos
        );
    }

    decomposed
}

// Decomposes each element in input into radix unsigned binary digits of its representative in [0, q).
// output[index * radix + i] holds, coefficient-wise, bit i of input[index], so every digit is in {0, 1}
// and Σ d_i * 2^i = x. radix must cover the bit length of q; digits beyond it are zero padding.
pub fn decompose_bits(input: &[RingElement], radix: usize) -> Vec<RingElement> {
    assert!(
        radix as u32 > MOD_Q.ilog2(),
        "radix must cover the whole [0, q) representative"
    );
    assert!(radix <= 64, "radix must fit into a u64 representative");

    // Representatives lie in [0, q), so digits n_bits.. are zero, as is their NTT: leave those slots.
    let n_bits = ((MOD_Q.ilog2() + 1) as usize).min(radix);

    let mut decomposed =
        vec![RingElement::zero(Representation::IncompleteNTT); input.len() * radix];

    let mut temp = RingElement::all(0, Representation::EvenOddCoefficients);

    for (index, el) in input.iter().enumerate() {
        temp.set_from(el);
        // Bit extraction is per coefficient, so the even/odd ordering is irrelevant here.
        temp.to_representation(Representation::EvenOddCoefficients);
        debug_assert!(temp.v.iter().all(|&c| c < (1u64 << n_bits)));

        let digits = &mut decomposed[index * radix..index * radix + n_bits];
        for (i, d) in digits.iter_mut().enumerate() {
            d.representation = Representation::EvenOddCoefficients;

            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
            unsafe {
                use std::arch::x86_64::*;

                let count = _mm_cvtsi32_si128(i as i32);
                let one = _mm512_set1_epi64(1);
                let src = temp.v.as_ptr() as *const i64;
                let dst = d.v.as_mut_ptr() as *mut i64;
                for b in (0..DEGREE).step_by(8) {
                    let x = _mm512_srl_epi64(_mm512_load_epi64(src.add(b)), count);
                    _mm512_store_epi64(dst.add(b), _mm512_and_si512(x, one));
                }
            }
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
            for c in 0..DEGREE {
                d.v[c] = (temp.v[c] >> i) & 1;
            }

            d.to_representation(Representation::IncompleteNTT);
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
    let mut recomposed = vec![RingElement::zero(Representation::IncompleteNTT); decomposed.len() / radix];

    for i in 0..recomposed.len() {
        recomposed[i] = RingElement::all(0, Representation::IncompleteNTT);
        for j in 0..radix {
            let mut term = decomposed[i * radix + j].clone();
            // Reduced mod q: 1u64 << k overflows the modulus (and then u64) once k >= 50.
            let shift = RingElement::constant(
                pow_mod(2, j as u64 * base_log),
                Representation::IncompleteNTT,
            );
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

#[test]
fn test_decompose_bits_roundtrip() {
    let input: Vec<RingElement> = (0..3)
        .map(|_| RingElement::random(Representation::IncompleteNTT))
        .collect();

    // 64 exercises the composer at shifts >= 50, where an unreduced 2^k would wrap.
    for radix in [MOD_Q.ilog2() as usize + 1, 64] {
        let decomposed = decompose_bits(&input, radix);
        debug_assert_eq!(decomposed.len(), input.len() * radix);

        let recomposed = compose_from_decomposed(&decomposed, 1, radix);
        for i in 0..input.len() {
            debug_assert_eq!(recomposed[i], input[i]);
        }
    }
}

#[test]
fn test_decompose_bits_digits_are_binary() {
    let input: Vec<RingElement> = (0..2)
        .map(|_| RingElement::random(Representation::IncompleteNTT))
        .collect();
    let radix = 64;

    let mut decomposed = decompose_bits(&input, radix);

    for d in decomposed.iter_mut() {
        d.from_incomplete_ntt_to_even_odd_coefficients();
        for &v in d.v.iter() {
            debug_assert_eq!(v == 0 || v == 1, true);
        }
    }
}

#[test]
fn test_decompose_bits_padding_is_zero() {
    let input = vec![RingElement::random(Representation::IncompleteNTT)];
    let radix = 64;

    let decomposed = decompose_bits(&input, radix);

    let zero = RingElement::all(0, Representation::IncompleteNTT);
    for i in (MOD_Q.ilog2() as usize + 1)..radix {
        debug_assert_eq!(decomposed[i], zero);
    }
}

#[test]
fn test_decompose_bits_edge_values() {
    let input = vec![
        RingElement::all(MOD_Q - 1, Representation::IncompleteNTT),
        RingElement::all(0, Representation::IncompleteNTT),
    ];
    let radix = 64;

    let decomposed = decompose_bits(&input, radix);

    for i in 0..radix {
        debug_assert_eq!(
            decomposed[i],
            RingElement::all(((MOD_Q - 1) >> i) & 1, Representation::IncompleteNTT)
        );
        debug_assert_eq!(
            decomposed[radix + i],
            RingElement::all(0, Representation::IncompleteNTT)
        );
    }

    let recomposed = compose_from_decomposed(&decomposed, 1, radix);
    debug_assert_eq!(recomposed[0], input[0]);
    debug_assert_eq!(recomposed[1], input[1]);
}

#[test]
fn test_decompose_bits_non_uniform_coefficients() {
    let mut el = RingElement::zero(Representation::EvenOddCoefficients);
    for j in 0..el.v.len() {
        el.v[j] = (j as u64).wrapping_mul(0x9E3779B97F4A7C15) % MOD_Q;
    }
    el.to_representation(Representation::IncompleteNTT);
    let input = vec![el];
    let radix = 64;

    let decomposed = decompose_bits(&input, radix);

    let mut coefficients = input[0].clone();
    coefficients.to_representation(Representation::EvenOddCoefficients);
    for i in 0..radix {
        let mut digit = decomposed[i].clone();
        digit.to_representation(Representation::EvenOddCoefficients);
        for j in 0..digit.v.len() {
            debug_assert_eq!(digit.v[j], (coefficients.v[j] >> i) & 1);
        }
    }

    let recomposed = compose_from_decomposed(&decomposed, 1, radix);
    debug_assert_eq!(recomposed[0], input[0]);
}

#[test]
fn test_decompose_bits_preserves_input() {
    let input = vec![
        RingElement::random(Representation::IncompleteNTT),
        RingElement::random(Representation::EvenOddCoefficients),
    ];
    let before = input.clone();

    let _ = decompose_bits(&input, 64);

    debug_assert_eq!(input, before);
}
