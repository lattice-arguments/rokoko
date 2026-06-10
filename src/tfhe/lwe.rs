use super::glwe::{tuniform, GlweCiphertext, GlweSecretKey};
use super::poly::{add_q, from_i64, mul_q, neg_q, sub_q, Q};
use rand::Rng;

#[derive(Clone, Debug)]
pub struct LweSecretKey {
    pub bits: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct LweCiphertext {
    pub mask: Vec<u64>,
    pub body: u64, // b = <a, s> + m + e
}

pub fn lwe_sample_binary_key(n: usize, rng: &mut impl Rng) -> LweSecretKey {
    LweSecretKey {
        bits: (0..n).map(|_| rng.random_range(0..2u64)).collect(),
    }
}

pub fn lwe_encrypt(
    key: &LweSecretKey,
    message: u64,
    noise_bound_log2: u32,
    rng: &mut impl Rng,
) -> LweCiphertext {
    let mask: Vec<u64> = key.bits.iter().map(|_| rng.random_range(0..Q)).collect();
    let mut body = message;
    for (a, s) in mask.iter().zip(key.bits.iter()) {
        body = add_q(body, mul_q(*a, *s));
    }
    body = add_q(body, from_i64(tuniform(noise_bound_log2, rng)));
    LweCiphertext { mask, body }
}

pub fn lwe_decrypt(key: &LweSecretKey, ct: &LweCiphertext) -> u64 {
    let mut m = ct.body;
    for (a, s) in ct.mask.iter().zip(key.bits.iter()) {
        m = sub_q(m, mul_q(*a, *s));
    }
    m
}

/// The flattened GLWE key as an LWE key of dimension k*N (sample extract target).
pub fn glwe_key_as_lwe(key: &GlweSecretKey) -> LweSecretKey {
    LweSecretKey {
        bits: key
            .polys
            .iter()
            .flat_map(|p| p.coeffs.iter().copied())
            .collect(),
    }
}

/// LWE sample of the constant coefficient of a GLWE ciphertext, under the
/// flattened key: a'_{uN} = mask_u[0], a'_{uN+i} = -mask_u[N-i] (i >= 1).
pub fn sample_extract_constant(ct: &GlweCiphertext) -> LweCiphertext {
    let n = ct.body.n();
    let mut mask = Vec::with_capacity(ct.mask.len() * n);
    for poly in &ct.mask {
        mask.push(poly.coeffs[0]);
        for i in 1..n {
            mask.push(neg_q(poly.coeffs[n - i]));
        }
    }
    LweCiphertext {
        mask,
        body: ct.body.coeffs[0],
    }
}

/// Key switching key: ksk[i][j] encrypts s_in[i] * B^j under the output key.
pub struct LweKeyswitchKey {
    pub rows: Vec<Vec<LweCiphertext>>,
    pub base_log: u32,
    pub levels: usize,
}

pub fn keyswitch_key_gen(
    key_in: &LweSecretKey,
    key_out: &LweSecretKey,
    base_log: u32,
    levels: usize,
    noise_bound_log2: u32,
    rng: &mut impl Rng,
) -> LweKeyswitchKey {
    let rows = key_in
        .bits
        .iter()
        .map(|&s| {
            (0..levels)
                .map(|j| {
                    let mut scale = s;
                    for _ in 0..(base_log as usize * j) {
                        scale = add_q(scale, scale);
                    }
                    lwe_encrypt(key_out, scale, noise_bound_log2, rng)
                })
                .collect()
        })
        .collect();
    LweKeyswitchKey {
        rows,
        base_log,
        levels,
    }
}

/// out = (b, 0) - sum_{i,j} d_j(a_i) * ksk[i][j], which decrypts under
/// key_out to b - <a, s_in> up to noise.
pub fn keyswitch(ksk: &LweKeyswitchKey, ct: &LweCiphertext) -> LweCiphertext {
    let n_out = ksk.rows[0][0].mask.len();
    let mut out = LweCiphertext {
        mask: vec![0; n_out],
        body: ct.body,
    };
    for (a, row) in ct.mask.iter().zip(ksk.rows.iter()) {
        for (j, d) in super::glwe::decompose_scalar(*a, ksk.base_log, ksk.levels)
            .into_iter()
            .enumerate()
        {
            let d = from_i64(d);
            if d == 0 {
                continue;
            }
            for (o, m) in out.mask.iter_mut().zip(row[j].mask.iter()) {
                *o = sub_q(*o, mul_q(d, *m));
            }
            out.body = sub_q(out.body, mul_q(d, row[j].body));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_lwe_roundtrip_and_keyswitch() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let delta = Q / 32;
        let key_in = lwe_sample_binary_key(512, &mut rng);
        let key_out = lwe_sample_binary_key(128, &mut rng);

        fn round_to(dec: u64, p: u64) -> u64 {
            (((dec as u128 * p as u128) + (Q as u128) / 2) / (Q as u128)) as u64 % p
        }

        let ksk = keyswitch_key_gen(&key_in, &key_out, 2, 25, 3, &mut rng);
        for m in [0u64, 1, 7, 15, 31] {
            let ct = lwe_encrypt(&key_in, mul_q(m, delta), 3, &mut rng);
            assert_eq!(round_to(lwe_decrypt(&key_in, &ct), 32), m);

            let ct2 = keyswitch(&ksk, &ct);
            assert_eq!(round_to(lwe_decrypt(&key_out, &ct2), 32), m);
        }
    }
}
