use super::ring_arithmetic::Representation;
use crate::common::{
    ring_arithmetic::RingElement, vertically_aligned_matrix::VerticallyAlignedMatrix,
};

pub fn sample_random_vector(size: usize, representation: Representation) -> Vec<RingElement> {
    let mut vec = Vec::with_capacity(size);
    for i in 0..size {
        vec.push(RingElement::random(representation));
    }
    vec
}

pub fn sample_random_short_vector(
    size: usize,
    bound: u64,
    representation: Representation,
) -> Vec<RingElement> {
    let mut vec = Vec::with_capacity(size);

    for i in 0..size {
        vec.push(RingElement::random_bounded(representation, bound)); // Sample from {-1, 0, 1}
    }
    vec
}
