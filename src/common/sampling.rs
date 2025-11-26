use super::ring_arithmetic::Representation;
use crate::common::{matrix::Matrix, ring_arithmetic::RingElement};

pub fn sample_random_vector(size: usize) -> Vec<RingElement> {
    let mut vec = Vec::with_capacity(size);
    unsafe {
        vec.set_len(size);
    }
    for i in 0..size {
        vec[i] = RingElement::random(Representation::EvenOddCoefficients);
        vec[i].from_even_odd_coefficients_to_incomplete_ntt_representation();
    }
    vec
}

pub fn sample_random_short_mat(n: usize, m: usize, bound: u64) -> Matrix<RingElement> {
    let mut m = Matrix::new(m, n);
    for i in m.data.iter_mut() {
        *i = RingElement::random_bounded(Representation::EvenOddCoefficients, bound);

        i.from_even_odd_coefficients_to_incomplete_ntt_representation();
    }
    m
}
