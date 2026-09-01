use crate::common::{config::MOD_Q, ring_arithmetic::RingElement};

pub fn inf_norm(vec: &[RingElement]) -> u64 {
    vec.iter()
        .map(|el| {
            let mut el_cloned = el.clone();
            el_cloned.from_incomplete_ntt_to_even_odd_coefficients();
            el_cloned
                .v
                .map(|x| x)
                .iter()
                .map(|&x| {
                    if x > MOD_Q / 2 {
                        MOD_Q - x as u64
                    } else {
                        x as u64
                    }
                })
                .max()
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0)
}

/// Adds the squared centered coefficients to `sum`, reporting `false` if the sum leaves `u128`.
///
/// A centered coefficient is up to `MOD_Q / 2`, so a single square already exceeds `u64` once the
/// coefficient passes `2^32`, and the sum of many smaller ones exceeds it too. Both callers turn
/// a `false` here into an infinite norm, so a vector too large to measure is rejected by every
/// bound rather than accepted on a wrapped value.
fn accumulate_squares(sum: &mut u128, coefficients: &[u64]) -> bool {
    for &x in coefficients {
        let centered = (if x > MOD_Q / 2 { MOD_Q - x } else { x }) as u128;
        match sum.checked_add(centered * centered) {
            Some(next) => *sum = next,
            None => return false,
        }
    }
    true
}

pub fn l2_norm(vec: &[RingElement]) -> f64 {
    let mut sum = 0u128;
    for el in vec {
        let mut el_cloned = el.clone();
        el_cloned.from_incomplete_ntt_to_even_odd_coefficients();
        if !accumulate_squares(&mut sum, &el_cloned.v) {
            return f64::INFINITY;
        }
    }
    (sum as f64).sqrt()
}

pub fn l2_norm_coeffs(vec: &[RingElement]) -> f64 {
    let mut sum = 0u128;
    for el in vec {
        if !accumulate_squares(&mut sum, &el.v) {
            return f64::INFINITY;
        }
    }
    (sum as f64).sqrt()
}

pub fn assert_norm_bounded(label: &str, value: f64, bound: f64) {
    tracing::debug!("L2 norm of {label}: {value} (bound {bound})");
    assert!(
        value <= bound,
        "L2 norm of {label} = {value} exceeds the registered bound {bound}"
    );
}
