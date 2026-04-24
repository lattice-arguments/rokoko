use crate::common::arithmetic::field_to_ring_element_into;
use crate::common::config::{DEGREE, MOD_Q};
use crate::common::ring_arithmetic::*;
use crate::common::sumcheck_element::SumcheckElement;
use crate::hexl::bindings::eltwise_reduce_mod;
use blake3::Hasher;

/// Incremental transcript hash suitable for Fiat–Shamir style challenges.
/// The transcript absorbs ring elements and can deterministically derive
/// pseudorandom bytes for challenges.
#[derive(Clone)]
pub struct HashWrapper {
    transcript: Hasher,
    sample_counter: u64,
}

impl Default for HashWrapper {
    fn default() -> Self {
        Self::new()
    }
}

static SEED: &[u8] = b"Smutno mi, Boze";
impl HashWrapper {
    pub fn new() -> Self {
        let mut transcript = Hasher::new();
        transcript.update(SEED);
        Self {
            transcript,
            sample_counter: 0,
        }
    }

    /// Absorb a ring element into the transcript.
    pub fn update_with_ring_element(&mut self, element: &RingElement) {
        // hack: treat the u64 slice as raw bytes (native endianness)
        let ptr = element.v.as_ptr() as *const u8;
        let len = element.v.len() * std::mem::size_of::<u64>();
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        self.transcript.update(bytes);
    }

    pub fn update_with_ring_element_slice(&mut self, elements: &[RingElement]) {
        for element in elements {
            self.update_with_ring_element(element);
        }
    }

    pub fn update_with_quadratic_extension_element(&mut self, element: &QuadraticExtension) {
        // hack: treat the coeffs slice as raw bytes (native endianness)
        let ptr = element.coeffs.as_ptr() as *const u8;
        let len = element.coeffs.len() * std::mem::size_of::<u64>();
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        self.transcript.update(bytes);
    }

    pub fn update_with_quadratic_extension_slice(&mut self, elements: &[QuadraticExtension]) {
        for element in elements {
            self.update_with_quadratic_extension_element(element);
        }
    }

    pub fn update_with_bytes(&mut self, bytes: &[u8]) {
        self.transcript.update(bytes);
    }

    pub fn update_with_u64(&mut self, value: u64) {
        self.transcript.update(&value.to_le_bytes());
    }

    /// Derive `len` bytes of pseudorandomness from the current transcript.
    pub fn sample_bytes(&mut self, len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        self.fill_from_xof(b"bytes", &mut out);
        out
    }

    /// Convenience helper to sample a u64 challenge.
    pub fn sample_u64(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        self.fill_from_xof(b"u64", &mut buf);
        u64::from_le_bytes(buf)
    }

    pub fn sample_u64_mod_q(&mut self) -> u64 {
        self.sample_u64() % MOD_Q
    }

    pub fn fill_from_xof(&mut self, label: &[u8], out: &mut [u8]) {
        let mut state = self.transcript.clone();
        state.update(&self.sample_counter.to_le_bytes());
        state.update(label);
        self.sample_counter += 1;

        let mut xof = state.finalize_xof();
        xof.fill(out);
    }

    #[allow(dead_code)]
    fn sample_binary_ring_element(&mut self) -> RingElement {
        let mut buf = [0u8; DEGREE / 8];
        self.fill_from_xof(b"binary-ring-element", &mut buf);
        let mut element = RingElement::new(Representation::Coefficients);
        for i in 0..DEGREE / 8 {
            let b = buf[i];
            for j in 0..8 {
                element.v[i * 8 + j] = ((b >> j) & 1) as u64;
            }
        }
        element
    }
    #[allow(dead_code)]
    fn sample_biased_ternary_ring_element(&mut self) -> RingElement {
        let el1 = self.sample_binary_ring_element();
        let el2 = self.sample_binary_ring_element();
        let mut result = RingElement::new(Representation::Coefficients);
        subtraction(&mut result, &el1, &el2);
        result
    }

    fn sample_biased_ternary_ring_element_into(&mut self, output: &mut RingElement) {
        let mut buf = [0u8; DEGREE / 4];
        self.fill_from_xof(b"biased-ternary-ring-element", &mut buf);
        output.to_representation(Representation::Coefficients);
        for i in 0..DEGREE / 4 {
            let b = buf[i];
            for j in 0..4 {
                let bits = (b >> (j * 2)) & 0b11;
                let coeff = match bits {
                    0b01 => 1,
                    0b10 => MOD_Q - 1,
                    _ => 0,
                };
                output.v[i * 4 + j] = coeff;
            }
        }
        output.to_representation(Representation::IncompleteNTT);
    }

    pub fn sample_ring_element_into(&mut self, output: &mut RingElement) {
        let buf = output.v.as_mut_ptr() as *mut u8;
        let len = output.v.len() * std::mem::size_of::<u64>();
        self.fill_from_xof(b"ring-element", unsafe {
            std::slice::from_raw_parts_mut(buf, len)
        });
        unsafe {
            eltwise_reduce_mod(
                output.v.as_mut_ptr(),
                output.v.as_mut_ptr(),
                output.v.len() as u64,
                MOD_Q,
            );
        }
    }

    pub fn sample_field_element_into(&mut self, output: &mut QuadraticExtension) {
        let buf = output.coeffs.as_mut_ptr() as *mut u8;
        let len = output.coeffs.len() * std::mem::size_of::<u64>();
        self.fill_from_xof(b"field-element", unsafe {
            std::slice::from_raw_parts_mut(buf, len)
        });
        unsafe {
            eltwise_reduce_mod(
                output.coeffs.as_mut_ptr(),
                output.coeffs.as_mut_ptr(),
                output.coeffs.len() as u64,
                MOD_Q,
            );
        }
    }
    pub fn sample_ring_element_vec_into(&mut self, output: &mut [RingElement]) {
        for element in output.iter_mut() {
            self.sample_ring_element_into(element);
        }
        // output[5].set_from(&ZERO);
        // output[7].set_from(&ZERO);
    }

    pub fn sample_ring_element_ntt_slots_same_vec_into(&mut self, output: &mut [RingElement]) {
        let mut f = QuadraticExtension::zero();
        for element in output.iter_mut() {
            self.sample_field_element_into(&mut f);
            field_to_ring_element_into(&mut *element, &f);
            element.from_homogenized_field_extensions_to_incomplete_ntt();
        }
    }

    pub fn sample_ring_element_ntt_slots_into(&mut self, output: &mut RingElement) {
        let mut f = QuadraticExtension::zero();
        self.sample_field_element_into(&mut f);
        field_to_ring_element_into(output, &f);
        output.from_homogenized_field_extensions_to_incomplete_ntt();
    }

    pub fn sample_biased_ternary_ring_element_vec_into(&mut self, output: &mut [RingElement]) {
        for element in output.iter_mut() {
            self.sample_biased_ternary_ring_element_into(element);
        }
    }

    #[allow(dead_code)]
    fn sample_ternary_ring_element(&mut self) -> RingElement {
        // we need 2 bits per coefficient to sample uniformly from {-1, 0, 1}
        // 0b11 is discarded so we take (two times) more bytes to account for the rejections
        let mut buf = [0u8; DEGREE / 2];
        self.fill_from_xof(b"ternary-ring-element", &mut buf);
        let mut element = RingElement::new(Representation::Coefficients);
        let mut coeff_index = 0;
        for i in 0..DEGREE / 2 {
            let b = buf[i];
            for j in 0..4 {
                if coeff_index >= DEGREE {
                    break;
                }
                let bits = (b >> (j * 2)) & 0b11;
                if bits == 0b11 {
                    continue; // reject
                }
                let coeff = match bits {
                    0b00 => 0,
                    0b01 => 1,
                    0b10 => MOD_Q - 1,
                    _ => unreachable!(),
                };
                element.v[coeff_index] = coeff;
                coeff_index += 1;
            }
        }
        if coeff_index < DEGREE {
            // we didn't fill all coefficients due to rejections, sample recursively
            // practically this will almost never recurse
            return self.sample_ternary_ring_element();
        }
        element
    }
}

#[cfg(test)]
mod tests {
    use crate::common::config::MOD_Q;

    use super::*;

    #[test]
    fn deterministic_given_same_transcript() {
        let mut t1 = HashWrapper::new();
        let mut t2 = HashWrapper::new();

        let mut e = RingElement::new(Representation::Coefficients);
        e.v[0] = 42;
        e.v[5] = 7;

        t1.update_with_ring_element(&e);
        t2.update_with_ring_element(&e);

        let c1 = t1.sample_bytes(32);
        let c2 = t2.sample_bytes(32);
        debug_assert_eq!(c1, c2);
    }

    #[test]
    fn deterministic_given_different_transcript_order() {
        let mut t1 = HashWrapper::new();
        let mut t2 = HashWrapper::new();

        let mut e1 = RingElement::new(Representation::Coefficients);
        e1.v[0] = 1;
        let mut e2 = RingElement::new(Representation::Coefficients);
        e2.v[0] = 2;

        t1.update_with_ring_element(&e1);
        t1.update_with_ring_element(&e2);

        t2.update_with_ring_element(&e2);
        t2.update_with_ring_element(&e1);

        let c1 = t1.sample_bytes(32);
        let c2 = t2.sample_bytes(32);
        debug_assert_ne!(c1, c2);
    }

    #[test]
    fn same_transcript_sampled_twice_yield_different_outputs() {
        let mut t1 = HashWrapper::new();

        let challenge1 = t1.sample_biased_ternary_ring_element();
        let challenge2 = t1.sample_biased_ternary_ring_element();
        debug_assert_ne!(challenge1, challenge2);
    }

    #[test]
    fn transcript_update_changes_output() {
        let mut t1 = HashWrapper::new();
        let mut t2 = HashWrapper::new();

        let mut e = RingElement::new(Representation::Coefficients);
        e.v[0] = 1;
        t1.update_with_ring_element(&e);

        let challenge1 = t1.sample_bytes(16);
        let challenge2 = t2.sample_bytes(16);
        debug_assert_ne!(challenge1, challenge2);
    }

    #[test]
    fn counter_produces_fresh_challenges() {
        let mut transcript = HashWrapper::new();
        let first = transcript.sample_bytes(8);
        let second = transcript.sample_bytes(8);
        debug_assert_ne!(first, second);
    }

    #[test]
    fn sample_binary_ring_element_works() {
        let mut transcript = HashWrapper::new();
        let element = transcript.sample_binary_ring_element();
        for &coeff in element.v.iter() {
            debug_assert!(coeff == 0 || coeff == 1);
        }
    }

    #[test]
    fn sample_biased_ternary_ring_element_works() {
        let mut transcript = HashWrapper::new();
        let element = transcript.sample_biased_ternary_ring_element();
        for &coeff in element.v.iter() {
            debug_assert!(coeff == MOD_Q - 1 || coeff == 0 || coeff == 1);
        }
    }

    #[test]
    fn sample_ternary_ring_element_works() {
        let mut transcript = HashWrapper::new();
        let element = transcript.sample_ternary_ring_element();
        for &coeff in element.v.iter() {
            debug_assert!(coeff == MOD_Q - 1 || coeff == 0 || coeff == 1);
        }
    }

    #[test]
    fn sample_ring_element_works() {
        let mut transcript = HashWrapper::new();
        let mut element = RingElement::new(Representation::IncompleteNTT);
        transcript.sample_ring_element_into(&mut element);
        let mut found_nonzero = false;
        for &coeff in element.v.iter() {
            if coeff != 0 {
                found_nonzero = true;
                break;
            }
        }
        debug_assert!(found_nonzero);
    }
}
