#![feature(target_feature_inline_always)]
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
    let ntt = get_ntt(n, modulus);
    fused_incomplete_ntt_mult_inner(
        result,
        operand1,
        operand2,
        ntt.shift_factors(),
        ntt.shift_factors_f64(),
        n,
        modulus,
    );
}
