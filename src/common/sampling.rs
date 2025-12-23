use super::ring_arithmetic::Representation;
use crate::common::{ring_arithmetic::RingElement, witness::WitnessMatrix};

pub fn sample_random_vector(size: usize, representation: Representation) -> Vec<RingElement> {
    let mut vec = Vec::with_capacity(size);
    unsafe {
        vec.set_len(size);
    }
    for i in 0..size {
        vec[i] = RingElement::random(representation);
    }
    vec
}

pub fn sample_random_short_mat(n: usize, m: usize, bound: u64) -> WitnessMatrix<RingElement> {
    let mut m = WitnessMatrix::new(m, n);
    for i in m.data.iter_mut() {
        *i = RingElement::random_bounded(Representation::EvenOddCoefficients, bound);

        i.from_even_odd_coefficients_to_incomplete_ntt_representation();
    }
    m
}
