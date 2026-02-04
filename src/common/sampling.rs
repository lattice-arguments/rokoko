use super::ring_arithmetic::Representation;
use crate::common::ring_arithmetic::RingElement;

pub fn sample_random_vector(size: usize, representation: Representation) -> Vec<RingElement> {
    let mut vec = Vec::with_capacity(size);
    for _i in 0..size {
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

    for _i in 0..size {
        vec.push(RingElement::random_bounded(representation, bound));
    }
    vec
}
