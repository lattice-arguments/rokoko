//! Packing of TFHE-ring elements into proof-ring elements (PR4 groundwork).
//!
//! R' = Z_q[Y]/(Y^{N'}+1) with N' = 16*128 factors as prod_j F_{q^2}[Y]/(Y^16 - x_j)
//! over the same 64 twiddles psi_j as the proof ring R = Z_q[X]/(X^128+1):
//! both twiddle sets are the roots of psi^64 = -1. A TFHE poly packs into
//! N'/128 RingElements, element t holding the Y^t-coordinate of every slot,
//! so that R'-multiplication becomes a short convolution of native
//! R-multiplications (with one X-twist for the wrapped half).

use crate::common::ring_arithmetic::{Representation, RingElement};

/// Per-slot twiddles psi_i of R in slot order, read off as the even part of
/// X*X computed in the NTT domain.
pub fn slot_twiddles() -> Vec<u64> {
    use crate::common::config::HALF_DEGREE;
    let mut x = RingElement::zero(Representation::EvenOddCoefficients);
    x.v[HALF_DEGREE] = 1; // X: even part 0, odd part 1
    x.from_even_odd_coefficients_to_incomplete_ntt_representation();
    let mut x2 = x.clone();
    x2 *= &x;
    x2.v[..HALF_DEGREE].to_vec()
}

use crate::common::config::{DEGREE, HALF_DEGREE};
use super::poly::{add_q, mul_q, Poly};

/// pack(p)[t], for t < N'/64/2: RingElement whose slot j holds, as its
/// (even, odd) entries, the residue coefficients (c^(j)_t, c^(j)_{t+L}) of
/// p mod (Y^{2L} - psi_j), L = N'/128. Multiplication in R' becomes
/// c_t = sum_{t1+t2=t} a_t1 b_t2 + X * sum_{t1+t2=t+L} a_t1 b_t2 over R.
pub fn pack(p: &Poly) -> Vec<RingElement> {
    let n_prime = p.n();
    debug_assert!(n_prime % DEGREE == 0);
    let l = n_prime / DEGREE; // Y-degree over F_{q^2} per slot
    let psis = slot_twiddles();
    let mut out = vec![RingElement::zero(Representation::IncompleteNTT); l];
    for (j, &psi) in psis.iter().enumerate() {
        // residue mod (Y^{2l} - psi): c^(j)_t = sum_r coeff[2l*r + t] * psi^r
        let mut psi_pow = 1u64;
        let mut residue = vec![0u64; 2 * l];
        for r in 0..n_prime / (2 * l) {
            for t in 0..2 * l {
                residue[t] = add_q(residue[t], mul_q(p.coeffs[2 * l * r + t], psi_pow));
            }
            psi_pow = mul_q(psi_pow, psi);
        }
        for t in 0..l {
            out[t].v[j] = residue[t];
            out[t].v[j + HALF_DEGREE] = residue[t + l];
        }
    }
    out
}

/// Product in R' computed on packed representations: a length-l negacyclic-
/// with-X-twist convolution of native R-multiplications.
pub fn packed_mul(a: &[RingElement], b: &[RingElement]) -> Vec<RingElement> {
    let l = a.len();
    debug_assert_eq!(l, b.len());
    let x = {
        let mut x = RingElement::zero(Representation::EvenOddCoefficients);
        x.v[HALF_DEGREE] = 1;
        x.from_even_odd_coefficients_to_incomplete_ntt_representation();
        x
    };
    let mut out = vec![RingElement::zero(Representation::IncompleteNTT); l];
    let mut tmp = RingElement::zero(Representation::IncompleteNTT);
    for t1 in 0..l {
        for t2 in 0..l {
            tmp *= (&a[t1], &b[t2]);
            if t1 + t2 < l {
                out[t1 + t2] += &tmp;
            } else {
                tmp *= &x;
                out[t1 + t2 - l] += &tmp;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::config::{HALF_DEGREE, MOD_Q};

    fn pow_mod(mut b: u64, mut e: u64) -> u64 {
        let mut r = 1u128;
        let mut bb = b as u128;
        while e > 0 {
            if e & 1 == 1 {
                r = r * bb % MOD_Q as u128;
            }
            bb = bb * bb % MOD_Q as u128;
            e >>= 1;
        }
        b = r as u64;
        b
    }

    #[test]
    fn test_slot_twiddles_are_roots_of_minus_one() {
        crate::common::init_common();
        let psis = slot_twiddles();
        assert_eq!(psis.len(), HALF_DEGREE);
        let mut distinct = psis.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 64, "expected 64 distinct twiddles");
        for &psi in &psis {
            assert_eq!(pow_mod(psi, 64), MOD_Q - 1, "psi^64 != -1");
        }
    }

    #[test]
    fn test_packed_mul_matches_tfhe_ring_mul() {
        use super::super::poly::{Poly, Q};
        use rand::{Rng, SeedableRng};
        crate::common::init_common();
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        for n_prime in [256usize, 2048] {
            let a = Poly {
                coeffs: (0..n_prime).map(|_| rng.random_range(0..Q)).collect(),
            };
            let b = Poly {
                coeffs: (0..n_prime).map(|_| rng.random_range(0..Q)).collect(),
            };
            let expected = pack(&a.mul(&b));
            let got = packed_mul(&pack(&a), &pack(&b));
            assert_eq!(got.len(), expected.len());
            for (g, e) in got.iter().zip(expected.iter()) {
                assert_eq!(g.v, e.v, "packed product mismatch at N'={}", n_prime);
            }
        }
    }
}
