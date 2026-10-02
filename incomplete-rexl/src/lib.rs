#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

mod aligned_vec;
mod avx512_util;
pub mod cpu_features;
mod eltwise;
mod ntt;
mod ntt_avx512_util;
mod number_theory;
mod util;

pub use eltwise::{
    eltwise_add_mod, eltwise_fma_mod, eltwise_mult_mod, eltwise_reduce_mod, eltwise_sub_mod,
    fused_incomplete_ntt_mult_inner,
};
pub use number_theory::{add_uint_mod, inverse_mod, multiply_mod, pow_mod, sub_uint_mod};

pub fn power_mod(a: u64, b: u64, modulus: u64) -> u64 {
    pow_mod(a, b, modulus)
}

pub fn add_mod(a: u64, b: u64, modulus: u64) -> u64 {
    add_uint_mod(a, b, modulus)
}

pub fn sub_mod(a: u64, b: u64, modulus: u64) -> u64 {
    sub_uint_mod(a, b, modulus)
}

pub fn inv_mod(a: u64, modulus: u64) -> u64 {
    inverse_mod(a, modulus)
}

static NTT_CACHE: LazyLock<Mutex<HashMap<(usize, u64), Arc<ntt::Ntt>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

thread_local! {
    static NTT_LAST: std::cell::RefCell<Option<((usize, u64), Arc<ntt::Ntt>)>> =
        std::cell::RefCell::new(None);
}

fn get_ntt(n: usize, modulus: u64) -> Arc<ntt::Ntt> {
    let key = (n, modulus);
    if let Some(hit) = NTT_LAST.with(|cell| {
        cell.borrow().as_ref().and_then(|(cached_key, cached)| {
            if *cached_key == key {
                Some(cached.clone())
            } else {
                None
            }
        })
    }) {
        return hit;
    }

    let mut cache = NTT_CACHE.lock().expect("NTT cache poisoned");
    let ntt = if let Some(existing) = cache.get(&key) {
        existing.clone()
    } else {
        let ntt = Arc::new(ntt::Ntt::new(n as u64, modulus));
        cache.insert(key, ntt.clone());
        ntt
    };

    NTT_LAST.with(|cell| {
        *cell.borrow_mut() = Some((key, ntt.clone()));
    });
    ntt
}

fn with_ntt<F: FnOnce(&ntt::Ntt)>(n: usize, modulus: u64, f: F) {
    let mut f = Some(f);
    let hit = NTT_LAST.with(|cell| {
        let borrow = cell.borrow();
        if let Some((cached_key, cached)) = borrow.as_ref() {
            if *cached_key == (n, modulus) {
                if let Some(f) = f.take() {
                    f(cached);
                }
                return true;
            }
        }
        false
    });
    if hit {
        return;
    }
    let ntt = get_ntt(n, modulus);
    if let Some(f) = f {
        f(&ntt);
    }
}

pub fn get_roots(n: usize, modulus: u64) -> *const u64 {
    let ntt = get_ntt(n, modulus);
    ntt.root_of_unity_powers().as_ptr()
}

pub fn get_inv_roots(n: usize, modulus: u64) -> *const u64 {
    let ntt = get_ntt(n, modulus);
    ntt.inv_root_of_unity_powers().as_ptr()
}

pub fn ntt_forward_in_place(data: &mut [u64], n: usize, modulus: u64) {
    let operand = unsafe { std::slice::from_raw_parts(data.as_ptr(), data.len()) };
    with_ntt(n, modulus, |ntt| {
        ntt.compute_forward(data, operand, 1, 1);
    });
}

pub fn ntt_inverse_in_place(data: &mut [u64], n: usize, modulus: u64) {
    let operand = unsafe { std::slice::from_raw_parts(data.as_ptr(), data.len()) };
    with_ntt(n, modulus, |ntt| {
        ntt.compute_inverse(data, operand, 1, 1);
    });
}

pub fn ntt_inverse(result: &mut [u64], operand: &[u64], n: usize, modulus: u64) {
    with_ntt(n, modulus, |ntt| {
        ntt.compute_inverse(result, operand, 1, 1);
    });
}

/// Convert a polynomial from coefficient representation to the even-odd
/// incomplete-NTT representation used by [`fused_incomplete_ntt_mult`].
///
/// Given a degree-`2n` polynomial stored in `data[0..2n]`, this function:
/// 1. De-interleaves even/odd coefficients: `data = [a0,a1,a2,…] → [a0,a2,…,a1,a3,…]`
/// 2. Applies a forward NTT of size `n` to each half independently.
///
/// The result is a `2n`-element vector in the "incomplete NTT" layout where
/// `data[0..n]` holds the NTT of the even-indexed coefficients and `data[n..2n]`
/// holds the NTT of the odd-indexed coefficients.  This is exactly the format
/// expected by [`fused_incomplete_ntt_mult`].
///
/// # Parameters
/// - `data`: mutable slice of length `2 * n` (polynomial coefficients, modified in-place)
/// - `n`: half-degree (must be a power of two, ≥ 8)
/// - `modulus`: NTT-friendly prime modulus
pub fn incomplete_ntt_forward_in_place(data: &mut [u64], n: usize, modulus: u64) {
    assert!(data.len() >= 2 * n, "data.len() must be >= 2*n");

    // Step 1: de-interleave even/odd coefficients in-place.
    // [a0, a1, a2, a3, …, a_{2n-2}, a_{2n-1}]
    //   → even half: [a0, a2, a4, …]
    //   → odd  half: [a1, a3, a5, …]
    let mut tmp = vec![0u64; 2 * n];
    for i in 0..n {
        tmp[i] = data[2 * i];
        tmp[n + i] = data[2 * i + 1];
    }
    data[..2 * n].copy_from_slice(&tmp);

    // Step 2: forward NTT each half independently.
    ntt_forward_in_place(&mut data[..n], n, modulus);
    ntt_forward_in_place(&mut data[n..2 * n], n, modulus);
}

/// Inverse of [`incomplete_ntt_forward_in_place`].
///
/// Takes a `2n`-element vector in incomplete-NTT (even-odd) layout and converts
/// it back to coefficient representation.
///
/// # Parameters
/// - `data`: mutable slice of length `2 * n` (incomplete-NTT form, modified in-place)
/// - `n`: half-degree
/// - `modulus`: NTT-friendly prime modulus
pub fn incomplete_ntt_inverse_in_place(data: &mut [u64], n: usize, modulus: u64) {
    assert!(data.len() >= 2 * n, "data.len() must be >= 2*n");

    // Step 1: inverse NTT each half.
    ntt_inverse_in_place(&mut data[..n], n, modulus);
    ntt_inverse_in_place(&mut data[n..2 * n], n, modulus);

    // Step 2: re-interleave even/odd → coefficient order.
    let mut tmp = vec![0u64; 2 * n];
    for i in 0..n {
        tmp[2 * i] = data[i];
        tmp[2 * i + 1] = data[n + i];
    }
    data[..2 * n].copy_from_slice(&tmp);
}

/// Fused incomplete-NTT ring multiplication with internally cached shift factors.
///
/// Computes the product of two polynomials that are already in incomplete-NTT
/// (even-odd) representation.  For each `i` in `0..n`:
///
/// ```text
/// result[i]   = op1[i]*op2[i] + shift[i] * (op1[n+i]*op2[n+i])   (mod modulus)
/// result[n+i] = op1[n+i]*op2[i] + op1[i]*op2[n+i]                (mod modulus)
/// ```
///
/// The shift factors are derived from the NTT tables for `(n, modulus)` and are
/// cached internally—callers do **not** need to supply them.
///
/// # Parameters
/// - `result`: output slice of length `>= 2*n`
/// - `operand1`, `operand2`: input slices of length `>= 2*n` (incomplete-NTT form)
/// - `n`: half-degree (must be a power of two, ≥ 8, divisible by 8)
/// - `modulus`: NTT-friendly prime modulus
pub fn fused_incomplete_ntt_mult(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    n: usize,
    modulus: u64,
) {
    assert!(result.len() >= 2 * n && operand1.len() >= 2 * n && operand2.len() >= 2 * n);
    with_ntt(n, modulus, |ntt| unsafe {
        slot_mult(
            ntt,
            2,
            modulus,
            ifma_tables(ntt),
            result.as_mut_ptr(),
            operand1.as_ptr(),
            operand2.as_ptr(),
        )
    });
}

/// Degree `d` of the irreducible factors `X^d - zeta` of `X^N + 1` over `Z_q`:
/// `d = N / 2^(v2(q-1) - 1)` when `v2(q-1) <= log2 N`, else `1` (full splitting).
pub fn slot_degree(ring_degree: usize, modulus: u64) -> usize {
    assert!(
        ring_degree.is_power_of_two(),
        "ring degree {ring_degree} is not a power of two"
    );
    let v = (modulus - 1).trailing_zeros();
    (2 * ring_degree).checked_shr(v).unwrap_or(0).max(1)
}

fn supported_slot_degree(ring_degree: usize, modulus: u64) -> usize {
    let d = slot_degree(ring_degree, modulus);
    assert!(
        d == 2 || d == 4,
        "N = {ring_degree}, q = {modulus}, v2(q-1) = {}: slot degree {d} is not supported (only 2 and 4)",
        (modulus - 1).trailing_zeros()
    );
    d
}

/// Strided layout with `d = slot_degree(N, q)`: block `j` in `0..d` is
/// `result[j*N/d..(j+1)*N/d] = (a_j, a_{j+d}, a_{j+2d}, …)`.
pub fn coefficients_to_strided(
    result: &mut [u64],
    coefficients: &[u64],
    ring_degree: usize,
    modulus: u64,
) {
    let d = supported_slot_degree(ring_degree, modulus);
    let n = ring_degree / d;
    for (i, chunk) in coefficients[..ring_degree].chunks_exact(d).enumerate() {
        for (j, &c) in chunk.iter().enumerate() {
            result[j * n + i] = c;
        }
    }
}

pub fn strided_to_coefficients(
    result: &mut [u64],
    strided: &[u64],
    ring_degree: usize,
    modulus: u64,
) {
    let d = supported_slot_degree(ring_degree, modulus);
    let n = ring_degree / d;
    for (i, chunk) in result[..ring_degree].chunks_exact_mut(d).enumerate() {
        for (j, c) in chunk.iter_mut().enumerate() {
            *c = strided[j * n + i];
        }
    }
}

/// Length-`N/d` negacyclic NTT of each block of a strided element. Slot `i` is then
/// `(block_0[i], …, block_{d-1}[i]) = a mod (X^d - zeta_i)`, `zeta_i = NTT_{N/d}(X)[i]`.
pub fn strided_ntt_forward_in_place(data: &mut [u64], ring_degree: usize, modulus: u64) {
    let d = supported_slot_degree(ring_degree, modulus);
    with_ntt(ring_degree / d, modulus, |ntt| {
        ntt.compute_forward_blocks(data, d)
    });
}

pub fn strided_ntt_inverse_in_place(data: &mut [u64], ring_degree: usize, modulus: u64) {
    let d = supported_slot_degree(ring_degree, modulus);
    with_ntt(ring_degree / d, modulus, |ntt| {
        ntt.compute_inverse_blocks(data, d)
    });
}

/// Ring multiplication of two outputs of [`strided_ntt_forward_in_place`], slot by slot
/// modulo `Y^d - zeta_i`; `d = 2` is [`fused_incomplete_ntt_mult`].
pub fn fused_slot_mult(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    ring_degree: usize,
    modulus: u64,
) {
    let d = supported_slot_degree(ring_degree, modulus);
    assert!(
        (ring_degree / d) % 8 == 0,
        "N/d = {} is not divisible by 8",
        ring_degree / d
    );
    assert!(
        result.len() >= ring_degree
            && operand1.len() >= ring_degree
            && operand2.len() >= ring_degree
    );
    with_ntt(ring_degree / d, modulus, |ntt| unsafe {
        slot_mult(
            ntt,
            d,
            modulus,
            ifma_tables(ntt),
            result.as_mut_ptr(),
            operand1.as_ptr(),
            operand2.as_ptr(),
        )
    });
}

fn ifma_tables(ntt: &ntt::Ntt) -> Option<&ntt::Ifma52> {
    #[cfg(target_arch = "x86_64")]
    if *cpu_features::HAS_AVX512IFMA {
        return ntt.ifma52();
    }
    None
}

/// Slot product of `d n` u64s with inputs below `modulus`; `result` may alias `op1`.
unsafe fn slot_mult(
    ntt: &ntt::Ntt,
    d: usize,
    modulus: u64,
    ifma: Option<&ntt::Ifma52>,
    result: *mut u64,
    op1: *const u64,
    op2: *const u64,
) {
    let n = ntt.degree();
    let (zetas, precon) = (ntt.shift_factors(), ntt.shift_factors_precon52());
    #[cfg(target_arch = "x86_64")]
    if let Some(ifma) = ifma {
        let kernel = match d {
            2 => eltwise::fused_slot_mult_avx512_ifma::<2>,
            _ => eltwise::fused_slot_mult_avx512_ifma::<4>,
        };
        return kernel(result, op1, op2, zetas, precon, n, ifma);
    }
    let result = std::slice::from_raw_parts_mut(result, d * n);
    let op1 = std::slice::from_raw_parts(op1, d * n);
    let op2 = std::slice::from_raw_parts(op2, d * n);
    match d {
        2 => fused_incomplete_ntt_mult_inner(
            result,
            op1,
            op2,
            zetas,
            ntt.shift_factors_f64(),
            n,
            modulus,
        ),
        _ => eltwise::fused_slot4_mult_inner(result, op1, op2, ntt, n, modulus),
    }
}

/// [`fused_slot_mult`] for one `(N, q)`, with the tables and kernel looked up once.
pub struct SlotRing {
    ntt: Arc<ntt::Ntt>,
    ring_degree: usize,
    slot_degree: usize,
    modulus: u64,
    ifma: Option<ntt::Ifma52>,
}

impl SlotRing {
    pub fn new(ring_degree: usize, modulus: u64) -> Self {
        let slot_degree = supported_slot_degree(ring_degree, modulus);
        let n = ring_degree / slot_degree;
        assert!(n % 8 == 0, "N/d = {n} is not divisible by 8");
        let ntt = get_ntt(n, modulus);
        let ifma = ifma_tables(&ntt).copied();
        Self {
            ntt,
            ring_degree,
            slot_degree,
            modulus,
            ifma,
        }
    }

    /// `result` may alias `op1`.
    #[inline]
    pub unsafe fn mult(&self, result: *mut u64, op1: *const u64, op2: *const u64) {
        slot_mult(
            &self.ntt,
            self.slot_degree,
            self.modulus,
            self.ifma.as_ref(),
            result,
            op1,
            op2,
        );
    }

    /// [`strided_ntt_forward_in_place`] of the `N` u64s at `data`.
    pub fn ntt_forward(&self, data: &mut [u64]) {
        self.ntt
            .compute_forward_blocks(&mut data[..self.ring_degree], self.slot_degree);
    }

    pub fn ntt_inverse(&self, data: &mut [u64]) {
        self.ntt
            .compute_inverse_blocks(&mut data[..self.ring_degree], self.slot_degree);
    }

    /// [`Self::dot`] for `outputs` results at once: `results + o result_stride` (+)=
    /// `sum_k (op1 + k stride1) (op2 + k stride2 + o output_stride)`.
    pub unsafe fn dot_many(
        &self,
        results: *mut u64,
        result_stride: usize,
        outputs: usize,
        op1: *const u64,
        stride1: usize,
        op2: *const u64,
        stride2: usize,
        output_stride: usize,
        count: usize,
        accumulate: bool,
    ) {
        #[cfg(target_arch = "x86_64")]
        if let Some(ifma) = &self.ifma {
            let kernel = match self.slot_degree {
                2 => eltwise::slot_dot_avx512_ifma::<2>,
                _ => eltwise::slot_dot_avx512_ifma::<4>,
            };
            kernel(
                results,
                result_stride,
                outputs,
                op1,
                stride1,
                op2,
                stride2,
                output_stride,
                count,
                accumulate,
                self.ntt.shift_factors(),
                self.ntt.shift_factors_precon52(),
                self.ntt.degree(),
                ifma,
            );
            return;
        }
        let mut stack = [std::mem::MaybeUninit::<u64>::uninit(); 512];
        let mut heap = Vec::new();
        let product: *mut u64 = if self.ring_degree <= stack.len() {
            stack.as_mut_ptr().cast()
        } else {
            heap.resize(self.ring_degree, 0);
            heap.as_mut_ptr()
        };
        for o in 0..outputs {
            let result =
                std::slice::from_raw_parts_mut(results.add(o * result_stride), self.ring_degree);
            if !accumulate {
                result.fill(0);
            }
            for k in 0..count {
                self.mult(
                    product,
                    op1.add(k * stride1),
                    op2.add(k * stride2 + o * output_stride),
                );
                let product = std::slice::from_raw_parts(product, self.ring_degree);
                for (r, &x) in result.iter_mut().zip(product) {
                    *r = add_uint_mod(*r, x, self.modulus);
                }
            }
        }
    }

    /// `result (+)= sum_k (op1 + k stride1) (op2 + k stride2)`, strides in u64s; `result` must
    /// not overlap the operands.
    pub unsafe fn dot(
        &self,
        result: *mut u64,
        op1: *const u64,
        stride1: usize,
        op2: *const u64,
        stride2: usize,
        count: usize,
        accumulate: bool,
    ) {
        self.dot_many(
            result, 0, 1, op1, stride1, op2, stride2, 0, count, accumulate,
        );
    }
}
