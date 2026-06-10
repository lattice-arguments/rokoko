//! Negacyclic Z_q[X]/(X^N + 1) for the TFHE layer; N is independent of the
//! proof system's ring degree.

use crate::common::config::MOD_Q;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};

pub const Q: u64 = MOD_Q;

const TWO_Q: u64 = 2 * Q;
const M50: u128 = (1u128 << 50) - 1;
const R50: u128 = (1u128 << 50) - Q as u128;

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

#[inline]
fn reduce_u128(x: u128) -> u64 {
    let x = (x >> 50) * R50 + (x & M50);
    let x = (x >> 50) * R50 + (x & M50);
    let x = ((x >> 50) * R50 + (x & M50)) as u64;
    if x >= Q {
        x - Q
    } else {
        x
    }
}

#[inline]
fn mul_lazy(x: u64, w: u64, w_precon: u64) -> u64 {
    let h = ((x as u128 * w_precon as u128) >> 64) as u64;
    w.wrapping_mul(x).wrapping_sub(h.wrapping_mul(Q))
}

fn precon(w: u64) -> u64 {
    (((w as u128) << 64) / Q as u128) as u64
}

fn pow_q(mut b: u64, mut e: u64) -> u64 {
    let mut r = 1u64;
    while e > 0 {
        if e & 1 == 1 {
            r = mul_q(r, b);
        }
        b = mul_q(b, b);
        e >>= 1;
    }
    r
}

fn primitive_root_128() -> u64 {
    let e = (Q - 1) / 128;
    let mut g = 2u64;
    loop {
        let c = pow_q(g, e);
        if pow_q(c, 64) == Q - 1 {
            return c;
        }
        g += 1;
    }
}

struct MulTables {
    levels: u32,
    block: usize,
    fwd: Vec<(u64, u64)>,
    inv: Vec<(u64, u64)>,
    rho: Vec<u64>,
    scale: (u64, u64),
}

fn build_tables(n: usize) -> MulTables {
    debug_assert!(n >= 2 && n.is_power_of_two());
    let levels = 6.min(n.trailing_zeros());
    let blocks = 1usize << levels;
    let psi = primitive_root_128();
    let mut fwd = vec![(0u64, 0u64); blocks];
    let mut inv = vec![(0u64, 0u64); blocks];
    for idx in 1..blocks {
        let w = pow_q(psi, ((idx as u64).reverse_bits() >> 58) as u64);
        let wi = pow_q(w, Q - 2);
        fwd[idx] = (w, precon(w));
        inv[idx] = (wi, precon(wi));
    }
    let mut rho = vec![0u64; blocks];
    let half = blocks / 2;
    for i in 0..half {
        rho[2 * i] = fwd[half + i].0;
        rho[2 * i + 1] = neg_q(fwd[half + i].0);
    }
    let s_inv = pow_q(pow_q(2, levels as u64), Q - 2);
    MulTables {
        levels,
        block: n >> levels,
        fwd,
        inv,
        rho,
        scale: (s_inv, precon(s_inv)),
    }
}

static TABLES: LazyLock<RwLock<HashMap<usize, Arc<MulTables>>>> = LazyLock::new(Default::default);

fn tables(n: usize) -> Arc<MulTables> {
    if let Some(t) = TABLES.read().unwrap().get(&n) {
        return t.clone();
    }
    let t = Arc::new(build_tables(n));
    TABLES.write().unwrap().entry(n).or_insert(t).clone()
}

fn forward(v: &mut [u64], tb: &MulTables) {
    let mut len = v.len() >> 1;
    let mut m = 1usize;
    while m < (1usize << tb.levels) {
        for i in 0..m {
            let (w, wp) = tb.fwd[m + i];
            let (xs, ys) = v[2 * len * i..2 * len * (i + 1)].split_at_mut(len);
            for (x, y) in xs.iter_mut().zip(ys.iter_mut()) {
                let tx = if *x >= TWO_Q { *x - TWO_Q } else { *x };
                let u = mul_lazy(*y, w, wp);
                *x = tx + u;
                *y = tx + TWO_Q - u;
            }
        }
        len >>= 1;
        m <<= 1;
    }
}

fn inverse(v: &mut [u64], tb: &MulTables) {
    let mut len = tb.block;
    let mut m = 1usize << (tb.levels - 1);
    while m > 0 {
        for i in 0..m {
            let (w, wp) = tb.inv[m + i];
            let (xs, ys) = v[2 * len * i..2 * len * (i + 1)].split_at_mut(len);
            for (x, y) in xs.iter_mut().zip(ys.iter_mut()) {
                let s = *x + *y;
                let d = *x + TWO_Q - *y;
                *x = if s >= TWO_Q { s - TWO_Q } else { s };
                *y = mul_lazy(d, w, wp);
            }
        }
        len <<= 1;
        m >>= 1;
    }
    let (sc, scp) = tb.scale;
    for x in v.iter_mut() {
        let r = mul_lazy(*x, sc, scp);
        *x = if r >= Q { r - Q } else { r };
    }
}

fn block_products(a: &[u64], b: &[u64], tb: &MulTables, out: &mut [u64]) {
    let l = tb.block;
    let mut acc = vec![0u128; 2 * l];
    for (bi, &rho) in tb.rho.iter().enumerate() {
        let off = bi * l;
        let ab = &a[off..off + l];
        let bb = &b[off..off + l];
        acc.fill(0);
        for (i, &ai) in ab.iter().enumerate() {
            let ai = ai as u128;
            for (s, &bj) in acc[i..i + l].iter_mut().zip(bb.iter()) {
                *s += ai * bj as u128;
            }
        }
        for t in 0..l {
            let hi = reduce_u128(acc[t + l]) as u128;
            out[off + t] = reduce_u128(acc[t] + rho as u128 * hi);
        }
    }
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
        let tb = tables(n);
        let mut a = self.coeffs.clone();
        let mut b = other.coeffs.clone();
        forward(&mut a, &tb);
        forward(&mut b, &tb);
        let mut out = Poly::zero(n);
        block_products(&a, &b, &tb, &mut out.coeffs);
        inverse(&mut out.coeffs, &tb);
        out
    }

    #[cfg(test)]
    pub fn mul_schoolbook(&self, other: &Poly) -> Poly {
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
    fn test_mul_matches_schoolbook_across_sizes_and_densities() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for n in [64usize, 256, 2048] {
            for density in [1usize, 8, 0] {
                let sample = |rng: &mut rand::rngs::StdRng| Poly {
                    coeffs: (0..n)
                        .map(|_| {
                            if density != 0 && rng.random_range(0..n) >= density {
                                0
                            } else {
                                rng.random_range(0..Q)
                            }
                        })
                        .collect(),
                };
                let a = sample(&mut rng);
                let b = sample(&mut rng);
                assert_eq!(
                    a.mul(&b),
                    a.mul_schoolbook(&b),
                    "n={} density={}",
                    n,
                    density
                );
            }
        }
    }

    #[test]
    #[ignore = "benchmark: prints schoolbook vs partial-NTT timings at N=2048"]
    fn bench_mul_schoolbook_vs_ntt() {
        use rand::{Rng, SeedableRng};
        use std::time::Instant;
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let n = 2048;
        let a = Poly {
            coeffs: (0..n).map(|_| rng.random_range(0..Q)).collect(),
        };
        let b = Poly {
            coeffs: (0..n).map(|_| rng.random_range(0..Q)).collect(),
        };
        let _ = a.mul(&b);
        let reps = 20;
        let t0 = Instant::now();
        let mut c_new = Poly::zero(n);
        for _ in 0..reps {
            c_new = a.mul(&b);
        }
        let t_new = t0.elapsed() / reps;
        let t1 = Instant::now();
        let mut c_old = Poly::zero(n);
        for _ in 0..3 {
            c_old = a.mul_schoolbook(&b);
        }
        let t_old = t1.elapsed() / 3;
        assert_eq!(c_new, c_old);
        println!(
            "N=2048 mul: schoolbook {:?}, partial-NTT {:?}, speedup {:.1}x",
            t_old,
            t_new,
            t_old.as_secs_f64() / t_new.as_secs_f64()
        );
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
