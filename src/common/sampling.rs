use aes::cipher::{KeyIvInit, StreamCipher};

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

pub const PUBLIC_CRS_SEED: &[u8] = b"rokoko-CRS-v1/AES-256-CTR public seed";

type Aes256Ctr = ctr::Ctr64BE<aes::Aes256>;

const AES_BUF_U64: usize = 1024;

pub struct AesCtrPublicSampler {
    cipher: Aes256Ctr,
    buf: Box<[u64; AES_BUF_U64]>,
    pos: usize,
    threshold: u64,
}

impl AesCtrPublicSampler {
    pub fn from_seed(seed: &[u8]) -> Self {
        let key: [u8; 32] = *blake3::hash(seed).as_bytes();
        let iv = [0u8; 16];
        let cipher = Aes256Ctr::new(&key.into(), &iv.into());
        let mut s = Self {
            cipher,
            buf: Box::new([0u64; AES_BUF_U64]),
            pos: AES_BUF_U64,
            threshold: u64::MAX - (u64::MAX % MOD_Q),
        };
        s.refill();
        s
    }

    fn refill(&mut self) {
        for w in self.buf.iter_mut() {
            *w = 0;
        }
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                self.buf.as_mut_ptr() as *mut u8,
                AES_BUF_U64 * 8,
            )
        };
        self.cipher.apply_keystream(bytes);
        self.pos = 0;
    }

    #[inline(always)]
    fn next_u64_mod_q(&mut self) -> u64 {
        loop {
            if self.pos >= AES_BUF_U64 {
                self.refill();
            }
            let x = unsafe { *self.buf.get_unchecked(self.pos) };
            self.pos += 1;
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
    let mut sampler = AesCtrPublicSampler::from_seed(seed);
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
    fn sampler_is_deterministic() {
        let v1 = sample_public_vector_from_seed(b"seed", 3, Representation::IncompleteNTT);
        let v2 = sample_public_vector_from_seed(b"seed", 3, Representation::IncompleteNTT);
        assert_eq!(v1, v2);
    }

    #[test]
    fn sampler_changes_with_seed() {
        let v1 = sample_public_vector_from_seed(b"seed-a", 3, Representation::IncompleteNTT);
        let v2 = sample_public_vector_from_seed(b"seed-b", 3, Representation::IncompleteNTT);
        assert_ne!(v1, v2);
    }

    #[test]
    fn sampler_outputs_in_range() {
        let v =
            sample_public_vector_from_seed(PUBLIC_CRS_SEED, 4, Representation::IncompleteNTT);
        for el in &v {
            for &c in &el.v {
                assert!(c < MOD_Q);
            }
        }
    }
}
