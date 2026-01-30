use crate::{
    common::{
        config::MOD_Q,
        matrix::new_vec_zero_preallocated,
        ring_arithmetic::{Representation, RingElement},
    },
    hexl::bindings::{inv_mod, multiply_mod},
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

// Decomposes each element in input into radix parts of base 2^{base_log}.
// The decomposition is smart in the sense that it's not wasteful for negative numbers.
// Each element x is first shifted by adding 2^{base_log * radix - 1} to it, then decomposed into radix parts,
// and then each part is shifted back by subtracting 2^{base_log - 1}.
// This way, we ensure that each decomposed part lies in the range [-2^{base_log - 1}, 2^{base_log - 1}).
// The input elements are not modified in the end (i.e., they retain their original values).
pub fn decompose(input: &Vec<RingElement>, base_log: u64, radix: usize) -> Vec<RingElement> {
    let mut decomposed = new_vec_zero_preallocated(input.len() * radix);

    if base_log == 1 {
        decomposed.clone_from_slice(input);
        return decomposed;
    }

    let big_shift = RingElement::all(
        1u64 << (base_log * radix as u64 - 1),
        Representation::EvenOddCoefficients,
    );

    let small_shift = RingElement::all(1u64 << (base_log - 1), Representation::EvenOddCoefficients);

    let mut temp = RingElement::all(0, Representation::EvenOddCoefficients);

    for (index, el) in input.iter().enumerate() {
        temp.set_from(el);
        // TODO: mainly clone??
        temp.to_representation(Representation::EvenOddCoefficients);
        temp += &big_shift;
        for i in 0..radix {
            decomposed[index * radix + i].to_representation(Representation::EvenOddCoefficients);
            temp.bits_into(
                &mut decomposed[index * radix + i],
                i as u64 * base_log,
                (i as u64 + 1) * base_log,
            );
            decomposed[index * radix + i] -= &small_shift;
            decomposed[index * radix + i].to_representation(Representation::IncompleteNTT);
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
            let offset = get_composer_offset(base_log, radix);
            recomposed -= &RingElement::all(offset, Representation::IncompleteNTT);
            let el_incomplete_ntt = {
                let mut temp_el = el.clone();
                temp_el.to_representation(Representation::IncompleteNTT);
                temp_el
            };
            debug_assert_eq!(&recomposed, &el_incomplete_ntt, "Recomposition failed in decomposition. Perhaps base_log and radix are not chosen properly?");
        }
    }

    decomposed
}

// a + big_shift = \sum_{i \in [radix]} (decomposed_i + small_shift) * (2^{i * base_log})
// => a = \sum_{i \in [radix]} decomposed_i * (2^{i * base_log}) + (small_shift * \sum_{i \in [radix]} (2^{i * base_log}) - big_shift
pub fn get_composer_offset(base_log: u64, radix: usize) -> u64 {
    let small_shift = 1u64 << (base_log - 1);
    let big_shift = 1u64 << (base_log * radix as u64 - 1);
    let mut offset = MOD_Q + big_shift;
    for i in 0..radix {
        let shift = 1u64 << (i as u64 * base_log);
        offset -= small_shift * shift;
    }
    offset
}

// similar to get_composer_offset but scaled by 1/radix mod MOD_Q
// this is done so that we can write sumcheck claims on the recomposed values divided by radix (as the radix will correspond to unused variables used in other context for composition)
pub fn get_decomposed_offset_scaled(base_log: u64, radix: usize) -> u64 {
    let mut offset = get_composer_offset(base_log, radix);
    unsafe {
        // no need to cache as this is only used in preprocessing
        let inv_radix = inv_mod(radix as u64, MOD_Q);
        multiply_mod(offset, inv_radix, MOD_Q)
    }
}

pub fn compose_from_decomposed(
    decomposed: &Vec<RingElement>,
    base_log: u64,
    radix: usize,
) -> Vec<RingElement> {
    let mut recomposed = new_vec_zero_preallocated(decomposed.len() / radix);

    let offset = get_composer_offset(base_log, radix);

    for i in 0..recomposed.len() {
        recomposed[i] = RingElement::all(0, Representation::IncompleteNTT);
        for j in 0..radix {
            let mut term = decomposed[i * radix + j].clone();
            let shift =
                RingElement::constant(1u64 << (j as u64 * base_log), Representation::IncompleteNTT);
            term *= &shift;
            recomposed[i] += &term;
        }
        recomposed[i] -= &RingElement::all(offset, Representation::IncompleteNTT);
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
    // 37 is shifted to 37 + (8^4) / 2 = 2085
    // base 8 representation of 2085 = 4 * 8^3 + 0 * 8^2 + 4 * 8^1 + 5 * 8^0
    // so the decomposed elements should be [5, 4, 0, 4]
    // after removing the shift, they should be [5 - 4, 4 - 4, 0 - 4, 4 - 4] = [1, 0, -4, 0]
    debug_assert_eq!(
        decomposed[0],
        RingElement::all(1, Representation::IncompleteNTT)
    );
    debug_assert_eq!(
        decomposed[1],
        RingElement::all(0, Representation::IncompleteNTT)
    );
    debug_assert_eq!(
        decomposed[2],
        RingElement::all(MOD_Q - 4, Representation::IncompleteNTT)
    );
    debug_assert_eq!(
        decomposed[3],
        RingElement::all(0, Representation::IncompleteNTT)
    );

    let offset = get_composer_offset(base_log, radix);

    let mut recomposed = RingElement::all(0, Representation::IncompleteNTT);
    for i in 0..radix {
        let mut term = decomposed[i].clone();
        let shift =
            RingElement::constant(1u64 << (i as u64 * base_log), Representation::IncompleteNTT);
        term *= &shift;
        recomposed += &term;
    }
    recomposed -= &RingElement::all(offset, Representation::IncompleteNTT);
    debug_assert_eq!(recomposed, input[0]);
}

#[test]
fn test_random_mod_q() {
    let r = RingElement::random(Representation::IncompleteNTT);
    let data = vec![r];
    let base_log = 13; // do we cover 52 bits?
    let radix = 4;

    let mut decomposed = decompose(&data, base_log, radix);

    let offset = get_composer_offset(base_log, radix);

    let mut recomposed = RingElement::all(0, Representation::IncompleteNTT);
    for i in 0..radix {
        let mut term = decomposed[i].clone();
        let shift =
            RingElement::constant(1u64 << (i as u64 * base_log), Representation::IncompleteNTT);
        term *= &shift;
        recomposed += &term;
    }
    recomposed -= &RingElement::all(offset, Representation::IncompleteNTT);
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
