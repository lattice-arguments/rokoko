#![feature(target_feature_inline_always)]
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

mod avx512_util;
mod aligned_vec;
mod cpu_features;
mod eltwise;
mod ntt;
mod ntt_avx512_util;
mod number_theory;
mod util;

pub use eltwise::{
    eltwise_add_mod, eltwise_fma_mod, eltwise_mult_mod, eltwise_reduce_mod, eltwise_sub_mod,
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
