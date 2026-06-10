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

/// Everything the SNARK witness needs about one blind rotation: the surviving
/// steps (a~_i = 0 steps are skipped publicly), per-step gadget digits of
/// X^{a~_i}*ACC - ACC, and all intermediate ACC states.
pub struct BlindRotationTrace {
    pub b_switched: usize,
    pub steps: Vec<BlindRotationStep>,
    pub acc_states: Vec<GlweCiphertext>, // acc_0 .. acc_final
}

pub struct BlindRotationStep {
    pub index: usize,
    pub a_switched: usize,
    pub digits: Vec<Vec<Poly>>, // [k+1][levels]
}

pub fn blind_rotate_traced(
    bsk: &BootstrapKey,
    ct: &LweCiphertext,
    lut: &Poly,
    k: usize,
) -> (GlweCiphertext, BlindRotationTrace) {
    use super::glwe::{decompose_poly, external_product_with_digits};

    let n = lut.n();
    let two_n = 2 * n;

    let b_switched = mod_switch(ct.body, two_n);
    let mut acc = GlweCiphertext::zero(k, n);
    acc.body = lut.monomial_mul((two_n - b_switched) % two_n);

    let mut steps = Vec::new();
    let mut acc_states = vec![acc.clone()];

    for (i, (a, ggsw)) in ct.mask.iter().zip(bsk.ggsws.iter()).enumerate() {
        let a_switched = mod_switch(*a, two_n);
        if a_switched == 0 {
            continue;
        }
        let mut diff = acc.monomial_mul(a_switched);
        diff.sub_assign(&acc);
        let digits: Vec<Vec<Poly>> = diff
            .mask
            .iter()
            .chain(std::iter::once(&diff.body))
            .map(|p| decompose_poly(p, ggsw.base_log, ggsw.levels))
            .collect();
        acc.add_assign(&external_product_with_digits(ggsw, &digits, k, n));
        steps.push(BlindRotationStep {
            index: i,
            a_switched,
            digits,
        });
        acc_states.push(acc.clone());
    }

    (
        acc.clone(),
        BlindRotationTrace {
            b_switched,
            steps,
            acc_states,
        },
    )
}
