//! Negacyclic Z_q[X]/(X^N + 1) for the TFHE layer; N is independent of the
//! proof system's ring degree.

use crate::common::config::MOD_Q;

pub const Q: u64 = MOD_Q;

#[inline]
pub fn add_q(a: u64, b: u64) -> u64 {
    let s = a + b;
    if s >= Q {
        s - Q
    } else {
        s
    }
}

#[inline]
pub fn sub_q(a: u64, b: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a + Q - b
    }
}

#[inline]
pub fn mul_q(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % Q as u128) as u64
}

#[inline]
pub fn neg_q(a: u64) -> u64 {
    if a == 0 {
        0
    } else {
        Q - a
    }
}

/// Centered representative in (-q/2, q/2].
#[inline]
pub fn center(a: u64) -> i64 {
    if a > Q / 2 {
        a as i64 - Q as i64
    } else {
        a as i64
    }
}

#[inline]
pub fn from_i64(v: i64) -> u64 {
    v.rem_euclid(Q as i64) as u64
}

#[derive(Clone, Debug, PartialEq)]
pub struct Poly {
    pub coeffs: Vec<u64>,
}

impl Poly {
    pub fn zero(n: usize) -> Self {
        Poly {
            coeffs: vec![0; n],
        }
    }

    pub fn n(&self) -> usize {
        self.coeffs.len()
    }

    pub fn add_assign(&mut self, other: &Poly) {
        for (a, b) in self.coeffs.iter_mut().zip(other.coeffs.iter()) {
            *a = add_q(*a, *b);
        }
    }

    pub fn sub_assign(&mut self, other: &Poly) {
        for (a, b) in self.coeffs.iter_mut().zip(other.coeffs.iter()) {
            *a = sub_q(*a, *b);
        }
    }

    /// self * X^e in the negacyclic ring, e in [0, 2N).
    pub fn monomial_mul(&self, e: usize) -> Poly {
        let n = self.n();
        debug_assert!(e < 2 * n);
        let mut out = Poly::zero(n);
        for (i, &c) in self.coeffs.iter().enumerate() {
            let mut j = i + e;
            let mut v = c;
            if j >= 2 * n {
                j -= 2 * n;
            } else if j >= n {
                j -= n;
                v = neg_q(v);
            }
            out.coeffs[j] = add_q(out.coeffs[j], v);
        }
        out
    }

    pub fn mul(&self, other: &Poly) -> Poly {
        let n = self.n();
        debug_assert_eq!(n, other.n());
        let mut pos = vec![0u128; n];
        let mut neg = vec![0u128; n];
        // between reductions each entry gains < 60000/n terms of < 2^100 each
        const REDUCE_EVERY: usize = 60_000;
        let mut since_reduce = 0usize;
        for (i, &a) in self.coeffs.iter().enumerate() {
            if a == 0 {
                continue;
            }
            let a = a as u128;
            for (j, &b) in other.coeffs.iter().enumerate() {
                if b == 0 {
                    continue;
                }
                let t = a * b as u128;
                let k = i + j;
                if k < n {
                    pos[k] += t;
                } else {
                    neg[k - n] += t;
                }
            }
            since_reduce += n;
            if since_reduce >= REDUCE_EVERY {
                for k in 0..n {
                    pos[k] %= Q as u128;
                    neg[k] %= Q as u128;
                }
                since_reduce = 0;
            }
        }
        let mut out = Poly::zero(n);
        for k in 0..n {
            let p = (pos[k] % Q as u128) as u64;
            let m = (neg[k] % Q as u128) as u64;
            out.coeffs[k] = sub_q(p, m);
        }
        out
    }

    pub fn scalar_mul_assign(&mut self, s: u64) {
        for c in self.coeffs.iter_mut() {
            *c = mul_q(*c, s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_mul(a: &Poly, b: &Poly) -> Poly {
        let n = a.n();
        let mut out = Poly::zero(n);
        for i in 0..n {
            for j in 0..n {
                let t = mul_q(a.coeffs[i], b.coeffs[j]);
                let k = (i + j) % n;
                if i + j < n {
                    out.coeffs[k] = add_q(out.coeffs[k], t);
                } else {
                    out.coeffs[k] = sub_q(out.coeffs[k], t);
                }
            }
        }
        out
    }

    #[test]
    fn test_negacyclic_mul_matches_naive() {
        use rand::Rng;
        let mut rng = rand::rng();
        for n in [8usize, 64] {
            let a = Poly {
                coeffs: (0..n).map(|_| rng.random_range(0..Q)).collect(),
            };
            let b = Poly {
                coeffs: (0..n).map(|_| rng.random_range(0..Q)).collect(),
            };
            assert_eq!(a.mul(&b), naive_mul(&a, &b));
        }
    }

    #[test]
    fn test_monomial_mul_wraps_negacyclically() {
        let n = 8;
        let mut a = Poly::zero(n);
        a.coeffs[6] = 5;
        let b = a.monomial_mul(3); // X^6 * X^3 = X^9 = -X^1
        assert_eq!(b.coeffs[1], neg_q(5));
        let c = a.monomial_mul(n + 2); // X^6 * X^10 = X^16 = +X^0
        assert_eq!(c.coeffs[0], 5);
    }
}
