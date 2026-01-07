use std::sync::LazyLock;

pub mod arithmetic;
pub mod config;
pub mod decomposition;
pub mod hash;
pub mod matrix;
pub mod projection_matrix;
pub mod ring_arithmetic;
pub mod sampling;
pub mod structured_row;
pub mod sumcheck_element;
use crate::common::{
    arithmetic::{ONE, ZERO},
    ring_arithmetic::*,
};

pub fn init_common() {
    LazyLock::force(&SHIFT_FACTORS);
    LazyLock::force(&NORMALIZE_INCOMPLETE_NTT_FACTORS);
    LazyLock::force(&NORMALIZE_INCOMPLETE_NTT_FACTORS_INVERSE);
    LazyLock::force(&ONE);
    LazyLock::force(&ZERO);
    unsafe { LazyLock::force_mut(&mut crate::common::ring_arithmetic::temp_buffer) };

    // init some caches of HEXL
    let mut a = RingElement::new(Representation::EvenOddCoefficients);
    let mut b = RingElement::new(Representation::IncompleteNTT);
    a.from_even_odd_coefficients_to_incomplete_ntt_representation();
    incomplete_ntt_multiplication(&mut b, &a, &a);
    a.from_incomplete_ntt_to_even_odd_coefficients();
}
