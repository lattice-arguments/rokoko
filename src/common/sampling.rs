use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake128,
};

use super::ring_arithmetic::Representation;
use crate::common::config::{DEGREE, MOD_Q};
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

pub const PUBLIC_CRS_SEED: &[u8] = b"rokoko-CRS-v1/SHAKE128 public seed";

pub struct ShakePublicSampler {
    reader: sha3::Shake128Reader,
    threshold: u64,
}

impl ShakePublicSampler {
    pub fn from_seed(seed: &[u8]) -> Self {
        let mut hasher = Shake128::default();
        hasher.update(seed);
        let threshold = u64::MAX - (u64::MAX % MOD_Q);
        Self {
            reader: hasher.finalize_xof(),
            threshold,
        }
    }

    #[inline]
    fn next_u64_mod_q(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        loop {
            self.reader.read(&mut buf);
            let x = u64::from_le_bytes(buf);
            if x < self.threshold {
                return x % MOD_Q;
            }
        }
    }

    pub fn fill_ring_element(&mut self, element: &mut RingElement, representation: Representation) {
        element.representation = Representation::IncompleteNTT;
        for i in 0..DEGREE {
            element.v[i] = self.next_u64_mod_q();
        }
        element.to_representation(representation);
    }
}

pub fn sample_public_vector_from_seed(
    seed: &[u8],
    size: usize,
    representation: Representation,
) -> Vec<RingElement> {
    let mut sampler = ShakePublicSampler::from_seed(seed);
    let mut vec = Vec::with_capacity(size);
    for _ in 0..size {
        let mut element = RingElement::new(representation);
        sampler.fill_ring_element(&mut element, representation);
        vec.push(element);
    }
    vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shake_sampler_is_deterministic() {
        let v1 = sample_public_vector_from_seed(b"seed", 3, Representation::IncompleteNTT);
        let v2 = sample_public_vector_from_seed(b"seed", 3, Representation::IncompleteNTT);
        assert_eq!(v1, v2);
    }

    #[test]
    fn shake_sampler_changes_with_seed() {
        let v1 = sample_public_vector_from_seed(b"seed-a", 3, Representation::IncompleteNTT);
        let v2 = sample_public_vector_from_seed(b"seed-b", 3, Representation::IncompleteNTT);
        assert_ne!(v1, v2);
    }

    #[test]
    fn shake_sampler_outputs_in_range() {
        let v =
            sample_public_vector_from_seed(PUBLIC_CRS_SEED, 4, Representation::IncompleteNTT);
        for el in &v {
            for &c in &el.v {
                assert!(c < MOD_Q, "coefficient {} not reduced mod q", c);
            }
        }
    }
}
