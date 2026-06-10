use super::glwe::{cmux, GgswCiphertext, GlweCiphertext};
use super::lwe::{sample_extract_constant, LweCiphertext};
use super::poly::{mul_q, neg_q, Poly, Q};

/// round(x * 2N / q) mod 2N (tfhe-rs Standard modulus switch).
pub fn mod_switch(x: u64, two_n: usize) -> usize {
    let r = ((x as u128 * two_n as u128 + (Q as u128) / 2) / (Q as u128)) as usize;
    r % two_n
}

/// Redundant LUT as in tfhe-rs generate_programmable_bootstrap_glwe_lut:
/// `entries` = message*carry boxes of f(i)*delta covering [0, N) (the padding
/// bit is the negacyclic wrap), rotated left by half a box to center boxes on
/// the mod-switched phases.
pub fn generate_lut(n: usize, entries: usize, delta: u64, f: impl Fn(u64) -> u64) -> Poly {
    let box_size = n / entries;
    let mut lut = Poly::zero(n);
    for i in 0..entries {
        let v = mul_q(f(i as u64) % (2 * entries) as u64, delta);
        for j in 0..box_size {
            lut.coeffs[i * box_size + j] = v;
        }
    }
    let half_box = box_size / 2;
    let mut out = Poly::zero(n);
    for i in 0..n - half_box {
        out.coeffs[i] = lut.coeffs[i + half_box];
    }
    for i in 0..half_box {
        out.coeffs[n - half_box + i] = neg_q(lut.coeffs[i]);
    }
    out
}

pub struct BootstrapKey {
    pub ggsws: Vec<GgswCiphertext>, // GGSW(s_i) per input LWE key bit
}

/// Blind rotation, the tfhe-rs loop: ACC = X^{-b~} * LUT (trivial GLWE), then
/// per step ACC = CMUX(GGSW(s_i), ACC, X^{a~_i} * ACC), skipping a~_i = 0.
pub fn blind_rotate(bsk: &BootstrapKey, ct: &LweCiphertext, lut: &Poly, k: usize) -> GlweCiphertext {
    let n = lut.n();
    let two_n = 2 * n;

    let b_switched = mod_switch(ct.body, two_n);
    let mut acc = GlweCiphertext::zero(k, n);
    acc.body = lut.monomial_mul((two_n - b_switched) % two_n);

    for (a, ggsw) in ct.mask.iter().zip(bsk.ggsws.iter()) {
        let a_switched = mod_switch(*a, two_n);
        if a_switched == 0 {
            continue;
        }
        let rotated = acc.monomial_mul(a_switched);
        acc = cmux(ggsw, &acc, &rotated);
    }
    acc
}

/// Outputs LWE(f(m)*delta) under the flattened GLWE key.
pub fn programmable_bootstrap(
    bsk: &BootstrapKey,
    ct: &LweCiphertext,
    lut: &Poly,
    k: usize,
) -> LweCiphertext {
    sample_extract_constant(&blind_rotate(bsk, ct, lut, k))
}
