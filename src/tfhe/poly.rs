//! Negacyclic Z_q[X]/(X^N + 1) for the TFHE layer; N is independent of the
//! proof system's ring degree.

use crate::common::config::MOD_Q;
use incomplete_rexl::{eltwise_mult_mod, ntt_forward_in_place, ntt_inverse_in_place};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};

pub const Q: u64 = MOD_Q;

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

struct MulTables {
    ntt_size: usize,
    block: usize,
    rho: Vec<u64>,
}

fn build_tables(n: usize) -> MulTables {
    debug_assert!(n >= 2 && n.is_power_of_two());
    let ntt_size = 64.min(n);
    let mut rho = vec![0u64; ntt_size];
    rho[1] = 1;
    ntt_forward_in_place(&mut rho, ntt_size, Q);
    MulTables {
        ntt_size,
        block: n / ntt_size,
        rho,
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

fn transpose(src: &[u64], dst: &mut [u64], rows: usize, cols: usize) {
    for r in 0..rows {
        for (c, &v) in src[r * cols..(r + 1) * cols].iter().enumerate() {
            dst[c * rows + r] = v;
        }
    }
}

fn columns_apply(v: &mut [u64], tb: &MulTables, f: impl Fn(&mut [u64], usize, u64)) {
    let (c, l) = (tb.ntt_size, tb.block);
    if l == 1 {
        f(v, c, Q);
        return;
    }
    let mut cols = vec![0u64; v.len()];
    transpose(v, &mut cols, c, l);
    for t in 0..l {
        f(&mut cols[t * c..(t + 1) * c], c, Q);
    }
    transpose(&cols, v, l, c);
}

#[cfg(target_arch = "x86_64")]
fn mul_ifma(a0: &[u64], b0: &[u64], tb: &MulTables, out: &mut [u64]) {
    let (c, l) = (tb.ntt_size, tb.block);
    let n = a0.len();
    let mut a = vec![0u64; n];
    let mut b = vec![0u64; n];
    transpose(a0, &mut a, c, l);
    transpose(b0, &mut b, c, l);
    for t in 0..l {
        ntt_forward_in_place(&mut a[t * c..(t + 1) * c], c, Q);
        ntt_forward_in_place(&mut b[t * c..(t + 1) * c], c, Q);
    }
    let mut rb = vec![0u64; n];
    for t in 0..l {
        eltwise_mult_mod(&mut rb[t * c..(t + 1) * c], &b[t * c..(t + 1) * c], &tb.rho, Q);
    }
    let mut prod = vec![0u64; n];
    unsafe { block_products_ifma(&a, &b, &rb, l, &mut prod) };
    for t in 0..l {
        ntt_inverse_in_place(&mut prod[t * c..(t + 1) * c], c, Q);
    }
    transpose(&prod, out, l, c);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512ifma")]
unsafe fn block_products_ifma(a: &[u64], b: &[u64], rb: &[u64], l: usize, out: &mut [u64]) {
    use std::arch::x86_64::*;
    let mut lo = [0u64; 8];
    let mut hi = [0u64; 8];
    for base in (0..64).step_by(8) {
        for t in 0..l {
            let mut alo = _mm512_setzero_si512();
            let mut ahi = _mm512_setzero_si512();
            for i in 0..=t {
                let va = _mm512_loadu_si512(a.as_ptr().add(i * 64 + base) as *const __m512i);
                let vb =
                    _mm512_loadu_si512(b.as_ptr().add((t - i) * 64 + base) as *const __m512i);
                alo = _mm512_madd52lo_epu64(alo, va, vb);
                ahi = _mm512_madd52hi_epu64(ahi, va, vb);
            }
            for i in t + 1..l {
                let va = _mm512_loadu_si512(a.as_ptr().add(i * 64 + base) as *const __m512i);
                let vb =
                    _mm512_loadu_si512(rb.as_ptr().add((l + t - i) * 64 + base) as *const __m512i);
                alo = _mm512_madd52lo_epu64(alo, va, vb);
                ahi = _mm512_madd52hi_epu64(ahi, va, vb);
            }
            _mm512_storeu_si512(lo.as_mut_ptr() as *mut __m512i, alo);
            _mm512_storeu_si512(hi.as_mut_ptr() as *mut __m512i, ahi);
            for lane in 0..8 {
                out[t * 64 + base + lane] =
                    reduce_u128(lo[lane] as u128 + ((hi[lane] as u128) << 52));
            }
        }
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
        let mut out = Poly::zero(n);
        #[cfg(target_arch = "x86_64")]
        if tb.ntt_size == 64 && *incomplete_rexl::cpu_features::HAS_AVX512IFMA {
            mul_ifma(&self.coeffs, &other.coeffs, &tb, &mut out.coeffs);
            return out;
        }
        let mut a = self.coeffs.clone();
        let mut b = other.coeffs.clone();
        columns_apply(&mut a, &tb, ntt_forward_in_place);
        columns_apply(&mut b, &tb, ntt_forward_in_place);
        block_products(&a, &b, &tb, &mut out.coeffs);
        columns_apply(&mut out.coeffs, &tb, ntt_inverse_in_place);
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
