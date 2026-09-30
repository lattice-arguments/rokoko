use std::sync::LazyLock;

pub mod arithmetic;
pub mod config;
pub mod decomposition;
pub mod estimator;
pub mod hash;
pub mod matrix;
pub mod norms;
pub mod projection_matrix;
pub mod ring_arithmetic;
pub mod sampling;
pub mod short_challenge;
pub mod structured_row;
pub mod sumcheck_element;
use crate::common::{
    arithmetic::{HALF_WAY_MOD_Q, HALF_WAY_MOD_Q_RING_CF, ONE, ONE_FE, TWO, TWO_FE, ZERO, ZERO_FE},
    ring_arithmetic::*,
};

pub fn init_common() {
    seed_rng("Widziałem lotne w powietrzu bociany długim szeregiem");

    LazyLock::force(&SHIFT_FACTORS);
    LazyLock::force(&FIELD_SHIFT_FACTOR);
    LazyLock::force(&INV_HALF_DEGREE);
    LazyLock::force(&TWO_INV_HALF_DEGREE);
    LazyLock::force(&CONJUGATION_NTT_TRANSFORM);
    LazyLock::force(&NORMALIZE_INCOMPLETE_NTT_FACTORS);
    LazyLock::force(&NORMALIZE_INCOMPLETE_NTT_FACTORS_INVERSE);
    LazyLock::force(&ONE);
    LazyLock::force(&ONE_FE);
    LazyLock::force(&ZERO);
    LazyLock::force(&ZERO_FE);
    LazyLock::force(&TWO);
    LazyLock::force(&TWO_FE);
    LazyLock::force(&HALF_WAY_MOD_Q);
    LazyLock::force(&HALF_WAY_MOD_Q_RING_CF);
    LazyLock::force(&CONSTANT_TERM_FACTORS);

    // init some caches of HEXL
    let mut a = RingElement::new(Representation::EvenOddCoefficients);
    let mut b = RingElement::new(Representation::IncompleteNTT);
    a.from_even_odd_coefficients_to_incomplete_ntt_representation();
    incomplete_ntt_multiplication(&mut b, &a, &a);
    a.from_incomplete_ntt_to_even_odd_coefficients();
}
