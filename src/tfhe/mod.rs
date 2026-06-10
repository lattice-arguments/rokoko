//! CGGI/TFHE bootstrapping implemented natively over the proof system's
//! modulus q = MOD_Q (no auxiliary 2^64 torus). Parameter shapes follow
//! tfhe-rs 1.6.2 PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128 with noise
//! bounds rescaled by q/2^64 and the gadget re-chosen for prime q (exact
//! balanced base-2^25, 2 levels). The blind rotation loop, modulus switch and
//! redundant LUT match the tfhe-rs implementation (Standard rounding; the
//! centered-mean body correction is not implemented).

pub mod bootstrap;
pub mod embed;
pub mod glwe;
pub mod lwe;
pub mod poly;

use rand::Rng;

#[derive(Clone, Copy, Debug)]
pub struct TfheParams {
    pub lwe_dimension: usize,     // n: blind rotation steps
    pub glwe_dimension: usize,    // k
    pub polynomial_size: usize,   // N
    pub pbs_base_log: u32,        // gadget base for GGSW
    pub pbs_levels: usize,
    pub ks_base_log: u32,
    pub ks_levels: usize,
    pub lwe_noise_bound_log2: u32,  // TUniform bound, q-scale
    pub glwe_noise_bound_log2: u32,
    pub message_modulus: u64,
    pub carry_modulus: u64,
}

impl TfheParams {
    /// Plaintext slots incl. the padding bit: p = 2 * message * carry.
    pub fn plaintext_modulus(&self) -> u64 {
        2 * self.message_modulus * self.carry_modulus
    }

    pub fn delta(&self) -> u64 {
        poly::Q / self.plaintext_modulus()
    }
}

/// Zama's default set with noise rescaled from 2^64 to q (45 -> 31, 17 -> 3)
/// and the PBS gadget re-chosen for prime q.
pub const SCALED_ZAMA_2_2: TfheParams = TfheParams {
    lwe_dimension: 918,
    glwe_dimension: 1,
    polynomial_size: 2048,
    pbs_base_log: 25,
    pbs_levels: 2,
    ks_base_log: 2,
    ks_levels: 25,
    lwe_noise_bound_log2: 31,
    glwe_noise_bound_log2: 3,
    message_modulus: 4,
    carry_modulus: 4,
};

pub const TOY: TfheParams = TfheParams {
    lwe_dimension: 8,
    glwe_dimension: 1,
    polynomial_size: 256,
    pbs_base_log: 25,
    pbs_levels: 2,
    ks_base_log: 2,
    ks_levels: 25,
    lwe_noise_bound_log2: 3,
    glwe_noise_bound_log2: 3,
    message_modulus: 4,
    carry_modulus: 4,
};

pub struct TfheKeys {
    pub lwe_key: lwe::LweSecretKey,
    pub glwe_key: glwe::GlweSecretKey,
    pub extracted_key: lwe::LweSecretKey, // flattened GLWE key (PBS output key)
    pub bsk: bootstrap::BootstrapKey,
    pub ksk: lwe::LweKeyswitchKey, // extracted_key -> lwe_key
}

pub fn keygen(params: &TfheParams, rng: &mut impl Rng) -> TfheKeys {
    let lwe_key = lwe::lwe_sample_binary_key(params.lwe_dimension, rng);
    let glwe_key =
        glwe::glwe_sample_binary_key(params.glwe_dimension, params.polynomial_size, rng);
    let extracted_key = lwe::glwe_key_as_lwe(&glwe_key);

    let ggsws = lwe_key
        .bits
        .iter()
        .map(|&s| {
            glwe::ggsw_encrypt(
                &glwe_key,
                s as i64,
                params.pbs_base_log,
                params.pbs_levels,
                params.glwe_noise_bound_log2,
                rng,
            )
        })
        .collect();

    let ksk = lwe::keyswitch_key_gen(
        &extracted_key,
        &lwe_key,
        params.ks_base_log,
        params.ks_levels,
        params.lwe_noise_bound_log2,
        rng,
    );

    TfheKeys {
        lwe_key,
        glwe_key,
        extracted_key,
        bsk: bootstrap::BootstrapKey { ggsws },
        ksk,
    }
}

pub fn encrypt(
    params: &TfheParams,
    keys: &TfheKeys,
    message: u64,
    rng: &mut impl Rng,
) -> lwe::LweCiphertext {
    lwe::lwe_encrypt(
        &keys.lwe_key,
        poly::mul_q(message % (params.message_modulus * params.carry_modulus), params.delta()),
        params.lwe_noise_bound_log2,
        rng,
    )
}

pub fn decrypt(params: &TfheParams, key: &lwe::LweSecretKey, ct: &lwe::LweCiphertext) -> u64 {
    let p = params.plaintext_modulus();
    let dec = lwe::lwe_decrypt(key, ct);
    (((dec as u128 * p as u128) + (poly::Q as u128) / 2) / (poly::Q as u128)) as u64 % p
}

/// KS_PBS order as in tfhe-rs EncryptionKeyChoice::Big: keyswitch from the
/// extracted key down to the LWE key, then bootstrap with the LUT for f.
pub fn bootstrap_with_lut(
    params: &TfheParams,
    keys: &TfheKeys,
    ct: &lwe::LweCiphertext,
    f: impl Fn(u64) -> u64,
) -> lwe::LweCiphertext {
    let lut = bootstrap::generate_lut(
        params.polynomial_size,
        (params.message_modulus * params.carry_modulus) as usize,
        params.delta(),
        f,
    );
    bootstrap::programmable_bootstrap(&keys.bsk, ct, &lut, params.glwe_dimension)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn pbs_roundtrip(params: &TfheParams, f: impl Fn(u64) -> u64 + Copy, seed: u64) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let keys = keygen(params, &mut rng);
        let p = params.message_modulus * params.carry_modulus;
        for m in 0..p {
            let ct = encrypt(params, &keys, m, &mut rng);
            let out = bootstrap_with_lut(params, &keys, &ct, f);
            let dec = decrypt(params, &keys.extracted_key, &out);
            assert_eq!(dec, f(m) % params.plaintext_modulus(), "message {}", m);

            // chain back to the small key, as before a next bootstrap
            let switched = lwe::keyswitch(&keys.ksk, &out);
            assert_eq!(
                decrypt(params, &keys.lwe_key, &switched),
                f(m) % params.plaintext_modulus(),
                "after keyswitch, message {}",
                m
            );
        }
    }

    #[test]
    fn test_toy_pbs_identity_and_lut() {
        pbs_roundtrip(&TOY, |x| x, 1);
        pbs_roundtrip(&TOY, |x| (3 * x + 1) % 16, 2);
    }

    #[test]
    fn test_traced_blind_rotation_matches_and_replays() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let params = &TOY;
        let keys = keygen(params, &mut rng);
        let ct = encrypt(params, &keys, 3, &mut rng);
        let lut = bootstrap::generate_lut(
            params.polynomial_size,
            (params.message_modulus * params.carry_modulus) as usize,
            params.delta(),
            |x| x,
        );

        let plain = bootstrap::blind_rotate(&keys.bsk, &ct, &lut, params.glwe_dimension);
        let (traced, trace) =
            bootstrap::blind_rotate_traced(&keys.bsk, &ct, &lut, params.glwe_dimension);
        assert_eq!(plain, traced);

        // replay: every recorded step satisfies
        // acc_{s+1} = acc_s + sum_{u,j} digits[u][j] * ggsw_rows[u][j]
        // and the digits recompose to X^{a~} acc_s - acc_s
        for (s, step) in trace.steps.iter().enumerate() {
            let acc = &trace.acc_states[s];
            let ggsw = &keys.bsk.ggsws[step.index];

            let mut diff = acc.monomial_mul(step.a_switched);
            diff.sub_assign(acc);
            for (u, poly) in diff
                .mask
                .iter()
                .chain(std::iter::once(&diff.body))
                .enumerate()
            {
                let mut recomposed = poly::Poly::zero(params.polynomial_size);
                for (j, d) in step.digits[u].iter().enumerate() {
                    let mut term = d.clone();
                    let mut base_pow = 1u64;
                    for _ in 0..(params.pbs_base_log as usize * j) {
                        base_pow = poly::add_q(base_pow, base_pow);
                    }
                    term.scalar_mul_assign(base_pow);
                    recomposed.add_assign(&term);
                }
                assert_eq!(&recomposed, poly, "digit recomposition, step {}", s);
            }

            let mut next = acc.clone();
            next.add_assign(&glwe::external_product_with_digits(
                ggsw,
                &step.digits,
                params.glwe_dimension,
                params.polynomial_size,
            ));
            assert_eq!(&next, &trace.acc_states[s + 1], "step equation, step {}", s);
        }
    }

    #[test]
    #[ignore = "minutes-long: full scaled-Zama bootstrap (N=2048, n=918)"]
    fn test_scaled_zama_pbs() {
        pbs_roundtrip(&SCALED_ZAMA_2_2, |x| (x * x) % 16, 3);
    }
}
