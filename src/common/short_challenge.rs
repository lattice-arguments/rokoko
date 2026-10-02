//! Short ternary challenge sampler for R = Z[X]/(X^N + 1).
//!
//! Each challenge has a fixed weight and operator norm bounded by a rejection
//! threshold, TAU and T_OP_NORM_BOUND by default. For a sparse `c = sum_k eps_k * X^{p_k}`,
//!
//! ```text
//! c(zeta^{2j+1}) = sum_k eps_k * T[((2j+1) * p_k) mod 2N]
//! ```
//!
//! with `T[m] = exp(i*pi*m/N)`. The hot path uses this directly via a length-2N
//! phase LUT instead of a full FFT.

use crate::common::config::{DEGREE, MOD_Q};
use crate::common::hash::HashWrapper;
use crate::common::ring_arithmetic::{Representation, RingElement};
use num::Complex;
use std::sync::LazyLock;

include!(concat!(env!("OUT_DIR"), "/challenge.rs"));

const N: usize = DEGREE;
const LOG_N: usize = N.trailing_zeros() as usize;
const PHASE_LEN: usize = 2 * N;
const PHASE_MASK: usize = PHASE_LEN - 1;

static PRE_TWIST: LazyLock<[Complex<f64>; N]> = LazyLock::new(|| {
    let mut arr = [Complex::new(0.0, 0.0); N];
    for j in 0..N {
        let angle = std::f64::consts::PI * (j as f64) / (N as f64);
        arr[j] = Complex::new(angle.cos(), angle.sin());
    }
    arr
});

static PHASE_RE: LazyLock<[f64; PHASE_LEN]> = LazyLock::new(|| {
    let mut arr = [0.0f64; PHASE_LEN];
    for m in 0..PHASE_LEN {
        let angle = std::f64::consts::PI * (m as f64) / (N as f64);
        arr[m] = angle.cos();
    }
    arr
});
static PHASE_IM: LazyLock<[f64; PHASE_LEN]> = LazyLock::new(|| {
    let mut arr = [0.0f64; PHASE_LEN];
    for m in 0..PHASE_LEN {
        let angle = std::f64::consts::PI * (m as f64) / (N as f64);
        arr[m] = angle.sin();
    }
    arr
});

static BIT_REVERSE: LazyLock<[usize; N]> = LazyLock::new(|| {
    let mut arr = [0usize; N];
    for i in 0..N {
        let mut x = i;
        let mut r = 0usize;
        for _ in 0..LOG_N {
            r = (r << 1) | (x & 1);
            x >>= 1;
        }
        arr[i] = r;
    }
    arr
});

fn fft_in_place(a: &mut [Complex<f64>; N]) {
    let br = &*BIT_REVERSE;
    for i in 0..N {
        let j = br[i];
        if i < j {
            a.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= N {
        let half = len / 2;
        let theta = 2.0 * std::f64::consts::PI / (len as f64);
        let w_step = Complex::new(theta.cos(), theta.sin());
        let mut start = 0usize;
        while start < N {
            let mut w = Complex::new(1.0_f64, 0.0_f64);
            for j in 0..half {
                let t = w * a[start + half + j];
                let u = a[start + j];
                a[start + j] = u + t;
                a[start + half + j] = u - t;
                w *= w_step;
            }
            start += len;
        }
        len <<= 1;
    }
}

pub fn op_norm(c: &[i8; N]) -> f64 {
    let pt = &*PRE_TWIST;
    let mut a = [Complex::new(0.0, 0.0); N];
    for j in 0..N {
        if c[j] != 0 {
            let s = c[j] as f64;
            a[j] = pt[j] * s;
        }
    }
    fft_in_place(&mut a);
    let mut max_sq = 0.0f64;
    for k in 0..(N / 2) {
        let m = a[k].norm_sqr();
        if m > max_sq {
            max_sq = m;
        }
    }
    max_sq.sqrt()
}

#[inline(always)]
pub fn op_norm_sq_sparse<const W: usize>(positions: &[u8; W], signs: &[i8; W]) -> f64 {
    let in_range = positions.iter().all(|&p| (p as usize) < N);
    assert!(in_range, "position out of range");
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe {
        return op_norm_sq_sparse_avx512(positions, signs);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    op_norm_sq_sparse_scalar(positions, signs)
}

pub fn op_norm_sq_sparse_scalar<const W: usize>(positions: &[u8; W], signs: &[i8; W]) -> f64 {
    let phase_re: &[f64; PHASE_LEN] = &PHASE_RE;
    let phase_im: &[f64; PHASE_LEN] = &PHASE_IM;

    let mut v_re = [0.0f64; N / 2];
    let mut v_im = [0.0f64; N / 2];

    for k in 0..W {
        let p = positions[k] as usize;
        let s = signs[k] as f64;
        let step = (2 * p) & PHASE_MASK;
        let mut idx = p & PHASE_MASK;
        for j in 0..(N / 2) {
            let pr = unsafe { *phase_re.get_unchecked(idx) };
            let pi = unsafe { *phase_im.get_unchecked(idx) };
            v_re[j] += s * pr;
            v_im[j] += s * pi;
            idx = (idx + step) & PHASE_MASK;
        }
    }

    let mut max_sq = 0.0f64;
    for j in 0..(N / 2) {
        let m = v_re[j] * v_re[j] + v_im[j] * v_im[j];
        if m > max_sq {
            max_sq = m;
        }
    }
    max_sq
}

/// Row `p` holds `T[p (2j + 1) mod 2N]` for `j < N/2`, real parts then imaginary parts: the
/// phases [`op_norm_sq_sparse_scalar`] steps through for a nonzero at position `p`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
static PHASE_ROWS: LazyLock<Vec<f64>> = LazyLock::new(|| {
    let mut rows = vec![0.0f64; N * N];
    for (p, row) in rows.chunks_exact_mut(N).enumerate() {
        for j in 0..N / 2 {
            let idx = (p * (2 * j + 1)) & PHASE_MASK;
            row[j] = PHASE_RE[idx];
            row[N / 2 + j] = PHASE_IM[idx];
        }
    }
    rows
});

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn op_norm_sq_sparse_avx512<const W: usize>(positions: &[u8; W], signs: &[i8; W]) -> f64 {
    use std::arch::x86_64::*;

    const NUM_BATCHES: usize = N / 16;
    const _: () = assert!(N % 16 == 0 && NUM_BATCHES * 2 <= 32);

    let rows = PHASE_ROWS.as_ptr();

    let mut vr = [_mm512_setzero_pd(); NUM_BATCHES];
    let mut vi = [_mm512_setzero_pd(); NUM_BATCHES];

    for k in 0..W {
        let s_v = _mm512_set1_pd(signs[k] as f64);
        let row = rows.add(N * positions[k] as usize);
        for b in 0..NUM_BATCHES {
            let pre = _mm512_loadu_pd(row.add(8 * b));
            let pim = _mm512_loadu_pd(row.add(N / 2 + 8 * b));
            vr[b] = _mm512_fmadd_pd(s_v, pre, vr[b]);
            vi[b] = _mm512_fmadd_pd(s_v, pim, vi[b]);
        }
    }

    let mut mm = _mm512_fmadd_pd(vr[0], vr[0], _mm512_mul_pd(vi[0], vi[0]));
    for b in 1..NUM_BATCHES {
        let m = _mm512_fmadd_pd(vr[b], vr[b], _mm512_mul_pd(vi[b], vi[b]));
        mm = _mm512_max_pd(mm, m);
    }
    _mm512_reduce_max_pd(mm)
}

pub fn op_norm_direct(c: &[i8; N]) -> f64 {
    let mut max_sq = 0.0f64;
    let two_n = 2 * N;
    for j in (1..two_n).step_by(2) {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for k in 0..N {
            if c[k] == 0 {
                continue;
            }
            let angle = std::f64::consts::PI * (j as f64) * (k as f64) / (N as f64);
            let s = c[k] as f64;
            re += s * angle.cos();
            im += s * angle.sin();
        }
        let m = re * re + im * im;
        if m > max_sq {
            max_sq = m;
        }
    }
    max_sq.sqrt()
}

pub const ATTEMPT_LABEL: &[u8] = b"short-ternary-challenge-attempt";
const ATTEMPT_BUF_LEN: usize = 128;

const _: () = assert!(N <= 256, "Fisher-Yates uses one byte per index sample");

#[allow(unused_assignments)]
fn sample_attempt<const W: usize>(hasher: &mut HashWrapper, label: &[u8]) -> ([u8; W], [i8; W]) {
    let mut buf = [0u8; ATTEMPT_BUF_LEN];
    hasher.fill_from_xof(label, &mut buf);
    let mut idx: usize = 0;

    macro_rules! next_byte {
        () => {{
            if idx >= ATTEMPT_BUF_LEN {
                hasher.fill_from_xof(label, &mut buf);
                idx = 0;
            }
            let b = buf[idx];
            idx += 1;
            b
        }};
    }

    let mut perm: [u8; N] = std::array::from_fn(|i| i as u8);
    for i in 0..W {
        let range = (N - i) as u32;
        let cutoff: u32 = 256 - (256 % range);
        let r = loop {
            let b = next_byte!() as u32;
            if b < cutoff {
                break b;
            }
        };
        let j = i + (r % range) as usize;
        perm.swap(i, j);
    }

    let mut positions = [0u8; W];
    let mut signs = [0i8; W];
    let mut sign_byte = 0u8;
    for i in 0..W {
        if i % 8 == 0 {
            sign_byte = next_byte!();
        }
        positions[i] = perm[i];
        let bit = (sign_byte >> (i % 8)) & 1;
        signs[i] = if bit == 1 { 1 } else { -1 };
    }
    (positions, signs)
}

pub fn sample_fixed_weight_challenge<const W: usize>(
    hasher: &mut HashWrapper,
    op_norm_bound: f64,
    label: &[u8],
) -> ([i8; N], usize) {
    assert!(0 < W && W <= N);
    let bound_sq = op_norm_bound * op_norm_bound;
    let mut attempts: usize = 0;
    loop {
        attempts += 1;
        let (positions, signs) = sample_attempt::<W>(hasher, label);
        if op_norm_sq_sparse(&positions, &signs) <= bound_sq {
            let mut c = [0i8; N];
            for i in 0..W {
                c[positions[i] as usize] = signs[i];
            }
            return (c, attempts);
        }
    }
}

pub fn sample_fixed_weight_challenge_into<const W: usize>(
    hasher: &mut HashWrapper,
    op_norm_bound: f64,
    label: &[u8],
    output: &mut RingElement,
) -> usize {
    let (c, attempts) = sample_fixed_weight_challenge::<W>(hasher, op_norm_bound, label);
    output.representation = Representation::Coefficients;
    for j in 0..N {
        output.v[j] = match c[j] {
            -1 => MOD_Q - 1,
            0 => 0,
            1 => 1,
            _ => unreachable!(),
        };
    }
    output.to_representation(Representation::IncompleteNTT);
    attempts
}

pub fn sample_short_challenge(hasher: &mut HashWrapper) -> ([i8; N], usize) {
    sample_fixed_weight_challenge::<TAU>(hasher, T_OP_NORM_BOUND, ATTEMPT_LABEL)
}

pub fn sample_short_challenge_into(hasher: &mut HashWrapper, output: &mut RingElement) -> usize {
    sample_fixed_weight_challenge_into::<TAU>(hasher, T_OP_NORM_BOUND, ATTEMPT_LABEL, output)
}

pub fn repetition_rate() -> f64 {
    let mut hasher = HashWrapper::new();
    let mut total_attempts: usize = 0;
    let trials = 10_000;
    for _ in 0..trials {
        let (_, attempts) = sample_short_challenge(&mut hasher);
        total_attempts += attempts;
    }
    (total_attempts as f64) / (trials as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_op_norm_rejects_positions_past_the_degree() {
        if N > u8::MAX as usize {
            return;
        }
        let mut positions = [0u8; TAU];
        positions[TAU - 1] = N as u8;
        let run = || op_norm_sq_sparse(&positions, &[1i8; TAU]);
        assert!(std::panic::catch_unwind(run).is_err());
    }

    fn weight(c: &[i8; N]) -> usize {
        c.iter().filter(|&&x| x != 0).count()
    }

    fn fixed_test_vector() -> ([u8; TAU], [i8; TAU]) {
        let mut positions = [0u8; TAU];
        let mut signs = [0i8; TAU];
        for i in 0..TAU {
            positions[i] = ((i * 13 + 1) % N) as u8;
            signs[i] = if i % 3 == 0 { -1 } else { 1 };
        }
        (positions, signs)
    }

    #[test]
    fn fft_op_norm_matches_direct_on_fixed_vector() {
        let (positions, signs) = fixed_test_vector();
        let mut c = [0i8; N];
        for i in 0..TAU {
            c[positions[i] as usize] = signs[i];
        }
        let fft = op_norm(&c);
        let direct = op_norm_direct(&c);
        assert!(
            (fft - direct).abs() < 1e-10,
            "fft={} direct={}",
            fft,
            direct
        );
    }

    #[test]
    fn sparse_op_norm_matches_dense_fft() {
        let mut hasher = HashWrapper::new();
        let top: [u8; TAU] = std::array::from_fn(|i| (N - 1 - i) as u8);
        for attempt in 0..2_000 {
            let (positions, signs) = match attempt {
                0 => fixed_test_vector(),
                1 => (top, [1i8; TAU]),
                2 => (top, [-1i8; TAU]),
                _ => sample_attempt::<TAU>(&mut hasher, b"sparse-vs-dense"),
            };
            let mut c = [0i8; N];
            for i in 0..TAU {
                c[positions[i] as usize] += signs[i];
            }
            let dense = op_norm(&c);
            let sparse = op_norm_sq_sparse(&positions, &signs).sqrt();
            assert!(
                (dense - sparse).abs() < 1e-10,
                "dense={dense} sparse={sparse}"
            );
        }
    }

    #[test]
    fn samples_have_correct_weight_and_op_norm() {
        let mut hasher = HashWrapper::new();
        for _ in 0..10_000 {
            let (c, _) = sample_short_challenge(&mut hasher);
            assert_eq!(weight(&c), TAU);
            for &x in &c {
                assert!(x == -1 || x == 0 || x == 1);
            }
            let n = op_norm(&c);
            assert!(n <= T_OP_NORM_BOUND, "op_norm {} > {}", n, T_OP_NORM_BOUND);
        }
    }

    #[test]
    fn parametrized_samples_have_correct_weight_and_op_norm() {
        const W: usize = 23;
        const BOUND: f64 = 8.357;
        let label = b"parametrized-challenge-test";
        let mut hasher = HashWrapper::new();
        let mut total_attempts = 0usize;
        for _ in 0..1_000 {
            let (c, attempts) = sample_fixed_weight_challenge::<W>(&mut hasher, BOUND, label);
            total_attempts += attempts;
            assert_eq!(weight(&c), W);
            for &x in &c {
                assert!(x == -1 || x == 0 || x == 1);
            }
            let n = op_norm(&c);
            assert!(n <= BOUND, "op_norm {} > {}", n, BOUND);
        }
        assert!(total_attempts > 1_000);
    }

    #[test]
    fn parametrized_sampling_is_deterministic_and_label_separated() {
        const W: usize = 23;
        const BOUND: f64 = 8.357;
        let mut h1 = HashWrapper::new();
        let mut h2 = HashWrapper::new();
        let mut h3 = HashWrapper::new();
        for _ in 0..20 {
            let (c1, _) = sample_fixed_weight_challenge::<W>(&mut h1, BOUND, b"label-a");
            let (c2, _) = sample_fixed_weight_challenge::<W>(&mut h2, BOUND, b"label-a");
            let (c3, _) = sample_fixed_weight_challenge::<W>(&mut h3, BOUND, b"label-b");
            assert_eq!(c1, c2);
            assert_ne!(c1, c3);
        }
        assert_eq!(h1.sample_bytes(64), h2.sample_bytes(64));
    }

    #[test]
    fn full_weight_bound_accepts_first_attempt() {
        const W: usize = 21;
        let mut hasher = HashWrapper::new();
        for _ in 0..100 {
            let (c, attempts) = sample_fixed_weight_challenge::<W>(&mut hasher, W as f64, b"t");
            assert_eq!(attempts, 1);
            assert_eq!(weight(&c), W);
        }
    }

    #[test]
    fn sampling_is_deterministic_and_post_state_matches() {
        let mut h1 = HashWrapper::new();
        let mut h2 = HashWrapper::new();
        for _ in 0..20 {
            let (c1, _) = sample_short_challenge(&mut h1);
            let (c2, _) = sample_short_challenge(&mut h2);
            assert_eq!(c1, c2);
        }
        assert_eq!(h1.sample_bytes(64), h2.sample_bytes(64));
    }

    #[test]
    fn challenge_stream_fingerprint() {
        let mut hasher = HashWrapper::new();
        let mut digest = blake3::Hasher::new();
        for _ in 0..100 {
            let (c, _) = sample_short_challenge(&mut hasher);
            let bytes: [u8; N] = std::array::from_fn(|i| c[i] as u8);
            digest.update(&bytes);
        }
        let actual = digest.finalize().to_hex().to_string();
        assert_eq!(
            actual, EXPECTED_FINGERPRINT,
            "challenge stream changed; update EXPECTED_FINGERPRINT if intentional"
        );
    }

    const EXPECTED_FINGERPRINT: &str =
        "5670da5c3578390a8e449b8d42526135dcdb708c95054a31c5767281218c42b1";

    #[test]
    fn ring_output_encoding_is_consistent() {
        let mut hasher = HashWrapper::new();
        let mut elem = RingElement::new(Representation::Coefficients);
        let _ = sample_short_challenge_into(&mut hasher, &mut elem);
        assert_eq!(elem.representation, Representation::IncompleteNTT);
        elem.to_representation(Representation::Coefficients);
        let mut nonzero = 0;
        for &v in &elem.v {
            assert!(v == 0 || v == 1 || v == MOD_Q - 1);
            if v != 0 {
                nonzero += 1;
            }
        }
        assert_eq!(nonzero, TAU);
    }
}
