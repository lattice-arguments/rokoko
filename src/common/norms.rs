use crate::common::{config::MOD_Q, ring_arithmetic::RingElement};

pub fn inf_norm(vec: &Vec<RingElement>) -> u64 {
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

pub fn l2_norm(vec: &Vec<RingElement>) -> f64 {
    let mut sum = 0u64;
    for el in vec {
        let mut el_cloned = el.clone();
        el_cloned.from_incomplete_ntt_to_even_odd_coefficients();
        for &x in el_cloned.v.map(|x| x).iter() {
            let centered = if x < MOD_Q / 2 { x } else { MOD_Q - x };
            sum += centered * centered;
        }
    }
    (sum as f64).sqrt() as f64
}

pub fn l2_norm_coeffs(vec: &Vec<RingElement>) -> f64 {
    let mut sum = 0u64;
    for el in vec {
        for &x in el.v.map(|x| x).iter() {
            let centered = if x < MOD_Q / 2 { x } else { MOD_Q - x };
            sum += centered * centered;
        }
    }
    (sum as f64).sqrt() as f64
}
