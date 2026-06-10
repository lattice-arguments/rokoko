use super::poly::{add_q, center, from_i64, mul_q, sub_q, Poly, Q};
use rand::Rng;

#[derive(Clone, Debug)]
pub struct GlweSecretKey {
    pub polys: Vec<Poly>, // k binary polynomials
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlweCiphertext {
    pub mask: Vec<Poly>, // k polynomials a_i
    pub body: Poly,      // b = sum a_i s_i + m + e
}

impl GlweCiphertext {
    pub fn zero(k: usize, n: usize) -> Self {
        GlweCiphertext {
            mask: vec![Poly::zero(n); k],
            body: Poly::zero(n),
        }
    }

    pub fn add_assign(&mut self, other: &GlweCiphertext) {
        for (a, b) in self.mask.iter_mut().zip(other.mask.iter()) {
            a.add_assign(b);
        }
        self.body.add_assign(&other.body);
    }

    pub fn sub_assign(&mut self, other: &GlweCiphertext) {
        for (a, b) in self.mask.iter_mut().zip(other.mask.iter()) {
            a.sub_assign(b);
        }
        self.body.sub_assign(&other.body);
    }

    pub fn monomial_mul(&self, e: usize) -> GlweCiphertext {
        GlweCiphertext {
            mask: self.mask.iter().map(|p| p.monomial_mul(e)).collect(),
            body: self.body.monomial_mul(e),
        }
    }
}

pub fn glwe_sample_binary_key(k: usize, n: usize, rng: &mut impl Rng) -> GlweSecretKey {
    GlweSecretKey {
        polys: (0..k)
            .map(|_| Poly {
                coeffs: (0..n).map(|_| rng.random_range(0..2u64)).collect(),
            })
            .collect(),
    }
}

/// TUniform(b): uniform over [-2^b, 2^b].
pub fn tuniform(bound_log2: u32, rng: &mut impl Rng) -> i64 {
    let b = 1i64 << bound_log2;
    rng.random_range(-b..=b)
}

pub fn glwe_encrypt(
    key: &GlweSecretKey,
    plaintext: &Poly,
    noise_bound_log2: u32,
    rng: &mut impl Rng,
) -> GlweCiphertext {
    let n = plaintext.n();
    let k = key.polys.len();
    let mask: Vec<Poly> = (0..k)
        .map(|_| Poly {
            coeffs: (0..n).map(|_| rng.random_range(0..Q)).collect(),
        })
        .collect();
    let mut body = plaintext.clone();
    for (a, s) in mask.iter().zip(key.polys.iter()) {
        body.add_assign(&a.mul(s));
    }
    for c in body.coeffs.iter_mut() {
        *c = add_q(*c, from_i64(tuniform(noise_bound_log2, rng)));
    }
    GlweCiphertext { mask, body }
}

pub fn glwe_decrypt(key: &GlweSecretKey, ct: &GlweCiphertext) -> Poly {
    let mut m = ct.body.clone();
    for (a, s) in ct.mask.iter().zip(key.polys.iter()) {
        m.sub_assign(&a.mul(s));
    }
    m
}

/// Exact balanced base-2^base_log decomposition of a centered value into
/// `levels` digits, least significant first: v = sum_j d_j (2^base_log)^j.
pub fn decompose_scalar(v: u64, base_log: u32, levels: usize) -> Vec<i64> {
    let base = 1i64 << base_log;
    let half = base / 2;
    let mut rest = center(v);
    let mut digits = Vec::with_capacity(levels);
    for _ in 0..levels - 1 {
        let mut d = rest % base;
        rest /= base;
        if d > half {
            d -= base;
            rest += 1;
        } else if d < -half {
            d += base;
            rest -= 1;
        }
        digits.push(d);
    }
    digits.push(rest);
    digits
}

pub fn decompose_poly(p: &Poly, base_log: u32, levels: usize) -> Vec<Poly> {
    let n = p.n();
    let mut out = vec![Poly::zero(n); levels];
    for (i, &c) in p.coeffs.iter().enumerate() {
        for (j, d) in decompose_scalar(c, base_log, levels).into_iter().enumerate() {
            out[j].coeffs[i] = from_i64(d);
        }
    }
    out
}

/// GGSW(m): (k+1) * levels GLWE rows; row (u, j) encrypts m * B^j at slot u
/// (mask slot u for u < k, body for u = k).
#[derive(Clone, Debug)]
pub struct GgswCiphertext {
    pub rows: Vec<Vec<GlweCiphertext>>, // [k+1][levels]
    pub base_log: u32,
    pub levels: usize,
}

pub fn ggsw_encrypt(
    key: &GlweSecretKey,
    message: i64,
    base_log: u32,
    levels: usize,
    noise_bound_log2: u32,
    rng: &mut impl Rng,
) -> GgswCiphertext {
    let k = key.polys.len();
    let n = key.polys[0].n();
    let m = from_i64(message);
    let mut rows = Vec::with_capacity(k + 1);
    for u in 0..=k {
        let mut level_rows = Vec::with_capacity(levels);
        for j in 0..levels {
            let scale = mul_q(m, mod_pow2(base_log as u64 * j as u64));
            let mut ct = glwe_encrypt(key, &Poly::zero(n), noise_bound_log2, rng);
            if u < k {
                // m B^j on mask slot u; the row decrypts to -m B^j s_u
                ct.mask[u].coeffs[0] = add_q(ct.mask[u].coeffs[0], scale);
            } else {
                ct.body.coeffs[0] = add_q(ct.body.coeffs[0], scale);
            }
            level_rows.push(ct);
        }
        rows.push(level_rows);
    }
    GgswCiphertext {
        rows,
        base_log,
        levels,
    }
}

fn mod_pow2(e: u64) -> u64 {
    let mut r = 1u64;
    for _ in 0..e {
        r = add_q(r, r);
    }
    r
}

/// External product GGSW(m) ⊡ GLWE(mu) -> GLWE(m * mu).
pub fn external_product(ggsw: &GgswCiphertext, ct: &GlweCiphertext) -> GlweCiphertext {
    let k = ct.mask.len();
    let n = ct.body.n();
    let mut out = GlweCiphertext::zero(k, n);
    for u in 0..=k {
        let poly = if u < k { &ct.mask[u] } else { &ct.body };
        let digits = decompose_poly(poly, ggsw.base_log, ggsw.levels);
        for (j, digit) in digits.iter().enumerate() {
            let row = &ggsw.rows[u][j];
            for v in 0..k {
                out.mask[v].add_assign(&digit.mul(&row.mask[v]));
            }
            out.body.add_assign(&digit.mul(&row.body));
        }
    }
    out
}

/// CMUX(C, d0, d1) = d0 + C ⊡ (d1 - d0).
pub fn cmux(ggsw: &GgswCiphertext, d0: &GlweCiphertext, d1: &GlweCiphertext) -> GlweCiphertext {
    let mut diff = d1.clone();
    diff.sub_assign(d0);
    let mut out = external_product(ggsw, &diff);
    out.add_assign(d0);
    out
}

#[allow(dead_code)]
pub fn noise_of(p: &Poly) -> i64 {
    p.coeffs.iter().map(|&c| center(c).abs()).max().unwrap_or(0)
}

#[allow(dead_code)]
pub fn assert_close(a: u64, b: u64, tolerance_log2: u32) {
    let d = center(sub_q(a, b)).abs();
    assert!(
        d <= 1i64 << tolerance_log2,
        "values differ by 2^{:.1} > 2^{}",
        (d as f64).log2(),
        tolerance_log2
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_decompose_recompose_and_digit_bounds() {
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        for _ in 0..1000 {
            let v = rng.random_range(0..Q);
            let digits = decompose_scalar(v, 25, 2);
            let mut acc = 0i128;
            let mut pow = 1i128;
            for &d in &digits {
                assert!(d.abs() <= (1 << 24) + 1, "digit {} out of range", d);
                acc += d as i128 * pow;
                pow <<= 25;
            }
            assert_eq!(acc.rem_euclid(Q as i128) as u64, v);
        }
    }

    #[test]
    fn test_glwe_roundtrip_external_product_cmux() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        let (k, n) = (1, 256);
        let delta = Q / 32;
        let key = glwe_sample_binary_key(k, n, &mut rng);

        let mut pt = Poly::zero(n);
        pt.coeffs[0] = mul_q(5, delta);
        pt.coeffs[3] = mul_q(11, delta);

        let ct = glwe_encrypt(&key, &pt, 3, &mut rng);
        let dec = glwe_decrypt(&key, &ct);
        assert_close(dec.coeffs[0], pt.coeffs[0], 10);
        assert_close(dec.coeffs[3], pt.coeffs[3], 10);

        // external product by GGSW(1) preserves the plaintext, GGSW(0) kills it
        for (bit, expect0) in [(1i64, mul_q(5, delta)), (0, 0)] {
            let ggsw = ggsw_encrypt(&key, bit, 25, 2, 3, &mut rng);
            let out = external_product(&ggsw, &ct);
            let dec = glwe_decrypt(&key, &out);
            assert_close(dec.coeffs[0], expect0, 40);
        }

        // cmux selects d0 / d1 by the encrypted bit
        let mut pt2 = Poly::zero(n);
        pt2.coeffs[0] = mul_q(9, delta);
        let ct2 = glwe_encrypt(&key, &pt2, 3, &mut rng);
        for (bit, expect) in [(0i64, mul_q(5, delta)), (1, mul_q(9, delta))] {
            let ggsw = ggsw_encrypt(&key, bit, 25, 2, 3, &mut rng);
            let out = cmux(&ggsw, &ct, &ct2);
            let dec = glwe_decrypt(&key, &out);
            assert_close(dec.coeffs[0], expect, 40);
        }
    }
}
