use crate::common::config::*;
use crate::hexl::bindings::*;
use crate::protocol::config::SizeableProof;
use rand::{Rng, SeedableRng};
use std::cell::RefCell;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};
use std::sync::LazyLock;

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum Representation {
    Coefficients, // This should not be used almost ever. Use only for printing or debugging.
    StridedCoefficients, // Coefficients grouped by index modulo the stride SLOT_DEGREE (part r holds c_r, c_{r+D}, ...), so that each part gets its own NTT.
    IncompleteNTT, // Incomplete NTT representation, where each part is separately transformed; slot i is a mod (X^D - zeta_i).
    HomogenizedFieldExtensions, // We use that reprentation so that "Incomplete NTT slots" are homogenized, i.e. they are all of
                                // the structure Zq[Y] / <Y^D - \alpha>, i.e. \alpha is the same for each slot.
}

// DO NOT derive Copy here, as RingElement is large.
// align(64) ensures `v` starts on a cache-line boundary so the AVX-512
// kernel can use aligned 512-bit loads/stores without crossing cache lines.
#[derive(PartialEq, Clone, Debug)]
#[repr(C, align(64))]
pub struct RingElement {
    pub v: [u64; DEGREE],
    pub representation: Representation,
}

thread_local! {
    static RNG: RefCell<rand::rngs::StdRng> =
        RefCell::new(rand::rngs::StdRng::from_os_rng());
}

pub fn seed_rng(seed: &str) {
    let seed_bytes: [u8; 32] = *blake3::hash(seed.as_bytes()).as_bytes();
    RNG.with(|cell| {
        *cell.borrow_mut() = rand::rngs::StdRng::from_seed(seed_bytes);
    });
}

impl RingElement {
    pub const fn new(representation: Representation) -> Self {
        Self {
            v: [0; DEGREE],
            representation,
        }
    }

    pub fn random(representation: Representation) -> Self {
        let mut element = Self {
            v: [0; DEGREE],
            representation,
        };

        RNG.with(|cell| {
            let mut rng = cell.borrow_mut();
            for i in 0..DEGREE {
                element.v[i] = rng.random_range(0..MOD_Q);
            }
        });

        element
    }

    pub fn one(representation: Representation) -> Self {
        let mut element = Self {
            v: [0; DEGREE],
            representation: Representation::StridedCoefficients,
        };
        element.v[0] = 1;

        element.to_representation(representation);

        element
    }

    pub fn all(value: u64, representation: Representation) -> Self {
        let mut element = Self {
            v: [0; DEGREE],
            representation: Representation::StridedCoefficients,
        };
        for i in 0..DEGREE {
            element.v[i] = value;
        }

        element.to_representation(representation);

        element
    }

    pub fn zero(representation: Representation) -> Self {
        let mut element = Self {
            v: [0; DEGREE],
            representation,
        };
        element.v[0] = 0;

        element
    }

    pub fn constant(value: u64, representation: Representation) -> Self {
        let mut element = Self {
            v: [0; DEGREE],
            representation: Representation::StridedCoefficients,
        };

        element.v[0] = value;

        element.to_representation(representation);

        element
    }

    pub fn random_bounded(representation: Representation, bound: u64) -> Self {
        let mut element = Self {
            v: [0; DEGREE],
            representation: Representation::Coefficients,
        };

        RNG.with(|cell| {
            let mut rng = cell.borrow_mut();
            for i in 0..DEGREE {
                let val = rng.random_range(0..bound);
                // Use a single random u64 bit to decide sign
                element.v[i] = if (rng.random::<u8>() & 1) == 0 {
                    val
                } else {
                    MOD_Q - val
                };
            }
        });
        unsafe {
            eltwise_reduce_mod(
                element.v.as_mut_ptr(),
                element.v.as_mut_ptr(),
                element.v.len() as u64,
                MOD_Q,
            );
        }

        element.to_representation(representation);

        element
    }

    pub fn random_bounded_unsigned(representation: Representation, bound: u64) -> Self {
        let mut element = Self {
            v: [0; DEGREE],
            representation: Representation::Coefficients,
        };

        RNG.with(|cell| {
            let mut rng = cell.borrow_mut();
            for i in 0..DEGREE {
                element.v[i] = rng.random_range(0..bound);
            }
        });

        element.to_representation(representation);

        element
    }

    pub fn from_strided_coefficients_to_incomplete_ntt_representation(&mut self) {
        debug_assert!(
            self.representation == Representation::StridedCoefficients,
            "Already in Incomplete NTT representation"
        );

        for part in 0..SLOT_DEGREE {
            unsafe {
                ntt_forward_in_place(self.v.as_mut_ptr().add(part * NUM_SLOTS), NUM_SLOTS, MOD_Q);
            }
        }

        self.representation = Representation::IncompleteNTT;
    }

    pub fn from_incomplete_ntt_to_strided_coefficients(&mut self) {
        debug_assert!(
            self.representation == Representation::IncompleteNTT,
            "Not in Incomplete NTT representation"
        );

        for part in 0..SLOT_DEGREE {
            unsafe {
                ntt_inverse_in_place(self.v.as_mut_ptr().add(part * NUM_SLOTS), NUM_SLOTS, MOD_Q);
            }
        }

        self.representation = Representation::StridedCoefficients;
    }

    pub fn from_coefficients_to_strided_coefficients(&mut self) {
        debug_assert!(
            self.representation == Representation::Coefficients,
            "Not in Coefficients representation"
        );

        let mut temp = [0u64; DEGREE];

        for i in 0..NUM_SLOTS {
            for part in 0..SLOT_DEGREE {
                temp[i + part * NUM_SLOTS] = self.v[SLOT_DEGREE * i + part];
            }
        }

        self.v = temp;
        self.representation = Representation::StridedCoefficients;
    }

    pub fn from_strided_coefficients_to_coefficients(&mut self) {
        debug_assert!(
            self.representation == Representation::StridedCoefficients,
            "Not in Strided Coefficients representation"
        );

        let mut temp = [0u64; DEGREE];

        for i in 0..NUM_SLOTS {
            for part in 0..SLOT_DEGREE {
                temp[SLOT_DEGREE * i + part] = self.v[i + part * NUM_SLOTS];
            }
        }

        self.v = temp;
        self.representation = Representation::Coefficients;
    }

    pub fn from_incomplete_ntt_to_homogenized_field_extensions(&mut self) {
        debug_assert!(
            self.representation == Representation::IncompleteNTT,
            "Not in Incomplete NTT representation"
        );

        if SLOT_DEGREE == 4 {
            self.swap_inverted_slots();
        }
        unsafe {
            eltwise_mult_mod(
                self.v.as_mut_ptr().add(NUM_SLOTS),
                self.v.as_ptr().add(NUM_SLOTS),
                NORMALIZE_INCOMPLETE_NTT_FACTORS.as_ptr(),
                (DEGREE - NUM_SLOTS) as u64,
                MOD_Q,
            );
        }
        self.representation = Representation::HomogenizedFieldExtensions;
    }

    pub fn from_homogenized_field_extensions_to_incomplete_ntt(&mut self) {
        debug_assert!(
            self.representation == Representation::HomogenizedFieldExtensions,
            "Not in Homogenized Field Extensions representation"
        );

        unsafe {
            eltwise_mult_mod(
                self.v.as_mut_ptr().add(NUM_SLOTS),
                self.v.as_ptr().add(NUM_SLOTS),
                NORMALIZE_INCOMPLETE_NTT_FACTORS_INVERSE.as_ptr(),
                (DEGREE - NUM_SLOTS) as u64,
                MOD_Q,
            );
        }
        if SLOT_DEGREE == 4 {
            self.swap_inverted_slots();
        }
        self.representation = Representation::IncompleteNTT;
    }

    /// Parts 1 and 3 of the slots whose homogenization inverts `X` trade places.
    fn swap_inverted_slots(&mut self) {
        for &slot in INVERTED_SLOTS.iter() {
            self.v.swap(NUM_SLOTS + slot, 3 * NUM_SLOTS + slot);
        }
    }

    pub fn to_representation(&mut self, representation: Representation) {
        match (self.representation, representation) {
            (Representation::Coefficients, Representation::StridedCoefficients) => {
                self.from_coefficients_to_strided_coefficients()
            }
            (Representation::Coefficients, Representation::IncompleteNTT) => {
                self.from_coefficients_to_strided_coefficients();
                self.from_strided_coefficients_to_incomplete_ntt_representation();
            }
            (Representation::Coefficients, Representation::HomogenizedFieldExtensions) => {
                self.from_coefficients_to_strided_coefficients();
                self.from_strided_coefficients_to_incomplete_ntt_representation();
                self.from_incomplete_ntt_to_homogenized_field_extensions();
            }
            (Representation::StridedCoefficients, Representation::IncompleteNTT) => {
                self.from_strided_coefficients_to_incomplete_ntt_representation()
            }
            (Representation::StridedCoefficients, Representation::HomogenizedFieldExtensions) => {
                self.from_strided_coefficients_to_incomplete_ntt_representation();
                self.from_incomplete_ntt_to_homogenized_field_extensions();
            }
            (Representation::IncompleteNTT, Representation::HomogenizedFieldExtensions) => {
                self.from_incomplete_ntt_to_homogenized_field_extensions()
            }
            (Representation::HomogenizedFieldExtensions, Representation::IncompleteNTT) => {
                self.from_homogenized_field_extensions_to_incomplete_ntt()
            }
            (Representation::HomogenizedFieldExtensions, Representation::StridedCoefficients) => {
                self.from_homogenized_field_extensions_to_incomplete_ntt();
                self.from_incomplete_ntt_to_strided_coefficients();
            }
            (Representation::HomogenizedFieldExtensions, Representation::Coefficients) => {
                self.from_homogenized_field_extensions_to_incomplete_ntt();
                self.from_incomplete_ntt_to_strided_coefficients();
                self.from_strided_coefficients_to_coefficients();
            }
            (Representation::IncompleteNTT, Representation::StridedCoefficients) => {
                self.from_incomplete_ntt_to_strided_coefficients();
            }
            (Representation::IncompleteNTT, Representation::Coefficients) => {
                self.from_incomplete_ntt_to_strided_coefficients();
                self.from_strided_coefficients_to_coefficients();
            }
            (Representation::StridedCoefficients, Representation::Coefficients) => {
                self.from_strided_coefficients_to_coefficients();
            }
            _ => {
                // nothing to do
            }
        }
    }

    // Probably should never be used
    pub fn split_into_field_extensions(&self) -> [FieldExtension; NUM_SLOTS] {
        debug_assert!(
            self.representation == Representation::HomogenizedFieldExtensions,
            "RingElement not in Homogenized Field Extensions representation"
        );

        let mut result = [FieldExtension::from_base(0); NUM_SLOTS];

        for i in 0..NUM_SLOTS {
            for part in 0..SLOT_DEGREE {
                result[i].coeffs[part] = self.v[i + part * NUM_SLOTS];
            }
        }

        result
    }

    pub fn combine_from_field_extensions(&mut self, extensions: &[FieldExtension; NUM_SLOTS]) {
        debug_assert!(
            self.representation == Representation::HomogenizedFieldExtensions,
            "RingElement not in Homogenized Field Extensions representation"
        );

        for i in 0..NUM_SLOTS {
            for part in 0..SLOT_DEGREE {
                self.v[i + part * NUM_SLOTS] = extensions[i].coeffs[part];
            }
        }
    }

    /// Slot 0 of an IncompleteNTT element as a field element: homogenization leaves slot 0
    /// unchanged, so this is also slot 0 of its homogenized form.
    #[inline]
    pub fn slot_zero(&self) -> FieldExtension {
        FieldExtension {
            coeffs: std::array::from_fn(|part| self.v[part * NUM_SLOTS]),
        }
    }

    #[inline]
    pub fn set_zero(&mut self) {
        self.v.fill(0);
    }

    fn conjugate_in_place_ref(&mut self) {
        // True Galois conjugation: X -> X^{-1} = -X^{n-1} in Z_q[X]/(X^n + 1)
        // In coefficient form: [c_0, c_1, ..., c_{n-1}] -> [c_0, -c_{n-1}, -c_{n-2}, ..., -c_1]
        // Reference implementation used for deriving NTT-domain transformations
        debug_assert_eq!(self.representation, Representation::IncompleteNTT);
        self.from_incomplete_ntt_to_strided_coefficients();
        self.from_strided_coefficients_to_coefficients();

        // Reverse and negate coefficients 1 to n-1
        for i in 1..(DEGREE / 2 + 1) {
            let temp = self.v[i];
            self.v[i] = MOD_Q - self.v[DEGREE - i];
            self.v[DEGREE - i] = MOD_Q - temp;
        }

        self.from_coefficients_to_strided_coefficients();
        self.from_strided_coefficients_to_incomplete_ntt_representation();
    }

    #[inline]
    pub fn set_from(&mut self, other: &RingElement) {
        self.v.copy_from_slice(&other.v);
        self.representation = other.representation;
    }

    pub fn conjugate_in_place(&mut self) {
        // True Galois conjugation: X -> X^{-1} = -X^{n-1} in Z_q[X]/(X^n + 1)
        // Pure NTT-domain implementation using empirically derived transformations
        //
        // PERFORMANCE: O(n) vs O(n log n)
        // This implementation: 0 NTT transforms, pure element-wise permutation and multiplication
        // Reference (coefficient space): 4 NTT transforms required
        //
        // MATHEMATICAL FOUNDATION:
        // =======================
        // Conjugation in coefficient space: [c_0, c_1, ..., c_{n-1}] -> [c_0, -c_{n-1}, ..., -c_1]
        // This reverses and negates all non-constant coefficients.
        //
        // In slot j, X^{-1} = zeta_j^{-1} X^{D-1}, so the slot takes its value from the slot i
        // with zeta_i = zeta_j^{-1}, and part r of slot i lands in part (D - r) mod D of slot j,
        // scaled by a power of zeta_j.
        //
        // NTT-DOMAIN TRANSFORMATION:
        // =========================
        // Rather than analytically deriving how conjugation acts on NTT coefficients
        // (which requires deep knowledge of HEXL's root ordering and evaluation points),
        // we use PRECOMPUTED PERMUTATIONS AND FACTORS derived empirically.
        //
        // The empirical approach:
        // 1. For each basis vector e_i in IncompleteNTT space
        // 2. Apply conjugation via coefficient space (ground truth)
        // 3. Observe where e_i maps to and what scaling factor is applied
        // 4. Build lookup tables: CONJUGATION_NTT_TRANSFORM
        //
        // This gives us, for every part r:
        // - permutation[r][i]: where slot i of part r goes after conjugation
        // - factors[r][i]: scaling factor for slot i of part r (1 for part 0)
        //
        // IMPLEMENTATION:
        // ==============
        // Apply the precomputed transformation directly in NTT space:
        // - new[(D - r) mod D][permutation[r][i]] = old[r][i] * factors[r][i]
        //
        // Benefits:
        // - No NTT transforms needed (pure O(n) operation)
        // - Provably correct (matches reference implementation by construction)
        // - Robust to HEXL implementation details

        debug_assert_eq!(self.representation, Representation::IncompleteNTT);
        let v = self.v.as_mut_ptr();
        unsafe { conjugate_slots(v, v) };
    }

    #[inline]
    pub fn conjugate_into(&self, result: &mut RingElement) {
        debug_assert_eq!(self.representation, Representation::IncompleteNTT);
        result.representation = self.representation;
        unsafe { conjugate_slots(self.v.as_ptr(), result.v.as_mut_ptr()) };
    }

    pub fn conjugate(&self) -> RingElement {
        let mut result = RingElement::new(self.representation);
        self.conjugate_into(&mut result);
        result
    }

    pub fn negate(&self) -> RingElement {
        let zero = RingElement::zero(self.representation);
        &zero - self
    }

    // 1 us
    pub fn inverse(&self) -> RingElement {
        assert_eq!(
            self.representation,
            Representation::HomogenizedFieldExtensions
        );
        if SLOT_DEGREE == 4 {
            return self.inverse_degree4();
        }

        // Each slot is Z_q[X]/(X^2 - beta) where beta = FIELD_SHIFT_FACTOR.
        // Slot i represents a_i + b_i*X with a_i = v[i], b_i = v[i + NUM_SLOTS].
        // Inverse: (a + bX)^{-1} = (a - bX) / (a^2 - beta * b^2)
        //
        // We use Montgomery's batch inversion trick to compute all norm inverses
        // with a single inv_mod call.

        let beta = *FIELD_SHIFT_FACTOR;
        let mut result = RingElement::new(Representation::HomogenizedFieldExtensions);

        // Step 1: Compute norms n_i = a_i^2 - beta * b_i^2
        let mut norms = [0u64; NUM_SLOTS];
        let mut temp = [0u64; NUM_SLOTS];

        unsafe {
            // norms[i] = a_i^2
            eltwise_mult_mod(
                norms.as_mut_ptr(),
                self.v.as_ptr(),
                self.v.as_ptr(),
                NUM_SLOTS as u64,
                MOD_Q,
            );

            // temp[i] = b_i^2
            eltwise_mult_mod(
                temp.as_mut_ptr(),
                self.v.as_ptr().add(NUM_SLOTS),
                self.v.as_ptr().add(NUM_SLOTS),
                NUM_SLOTS as u64,
                MOD_Q,
            );

            // norms[i] = -beta * b_i^2 + a_i^2
            eltwise_fma_mod(
                norms.as_mut_ptr(),
                temp.as_ptr(),
                MOD_Q - beta,
                norms.as_ptr(),
                NUM_SLOTS as u64,
                MOD_Q,
            );
        }

        // Step 2: Montgomery batch inversion of norms
        let mut prefix_products = [0u64; NUM_SLOTS];
        prefix_products[0] = norms[0];
        for i in 1..NUM_SLOTS {
            prefix_products[i] = unsafe { multiply_mod(prefix_products[i - 1], norms[i], MOD_Q) };
        }

        let mut inv = unsafe { inv_mod(prefix_products[NUM_SLOTS - 1], MOD_Q) };

        let mut norm_inverses = [0u64; NUM_SLOTS];
        for i in (1..NUM_SLOTS).rev() {
            norm_inverses[i] = unsafe { multiply_mod(inv, prefix_products[i - 1], MOD_Q) };
            inv = unsafe { multiply_mod(inv, norms[i], MOD_Q) };
        }
        norm_inverses[0] = inv;

        // Step 3: result = (a - bX) * n^{-1}
        // result_even[i] = a_i * n_i^{-1}
        // result_odd[i]  = -b_i * n_i^{-1}
        unsafe {
            eltwise_mult_mod(
                result.v.as_mut_ptr(),
                self.v.as_ptr(),
                norm_inverses.as_ptr(),
                NUM_SLOTS as u64,
                MOD_Q,
            );

            eltwise_mult_mod(
                result.v.as_mut_ptr().add(NUM_SLOTS),
                self.v.as_ptr().add(NUM_SLOTS),
                norm_inverses.as_ptr(),
                NUM_SLOTS as u64,
                MOD_Q,
            );

            // Negate the odd part: result_odd = 0 - result_odd
            temp.fill(0);
            eltwise_sub_mod(
                result.v.as_mut_ptr().add(NUM_SLOTS),
                temp.as_ptr(),
                result.v.as_ptr().add(NUM_SLOTS),
                NUM_SLOTS as u64,
                MOD_Q,
            );
        }

        result
    }

    /// Slot by slot in `Z_q[Y]/(Y^4 - alpha)` through the tower `Z = Y^2`: for
    /// `a = A0 + Y A1`, `A0 = a0 + a2 Z`, `A1 = a1 + a3 Z`, the norm to `Z_q[Z]/(Z^2 - alpha)` is
    /// `n = A0^2 - Z A1^2 = n0 + n1 Z`, and `a^{-1} = (A0 - Y A1)(n0 - n1 Z) / (n0^2 - alpha n1^2)`.
    fn inverse_degree4(&self) -> RingElement {
        let alpha = *FIELD_SHIFT_FACTOR;
        let mul = |a: u64, b: u64| unsafe { multiply_mod(a, b, MOD_Q) };
        let add = |a: u64, b: u64| unsafe { add_mod(a, b, MOD_Q) };
        let sub = |a: u64, b: u64| unsafe { sub_mod(a, b, MOD_Q) };
        let part = |r: usize, i: usize| self.v[r * NUM_SLOTS + i];

        let mut n0 = [0u64; NUM_SLOTS];
        let mut n1 = [0u64; NUM_SLOTS];
        let mut denominators = [0u64; NUM_SLOTS];
        for i in 0..NUM_SLOTS {
            let (a0, a1, a2, a3) = (part(0, i), part(1, i), part(2, i), part(3, i));
            let a1a3 = mul(a1, a3);
            n0[i] = sub(
                add(mul(a0, a0), mul(alpha, mul(a2, a2))),
                mul(alpha, add(a1a3, a1a3)),
            );
            let a0a2 = mul(a0, a2);
            n1[i] = sub(sub(add(a0a2, a0a2), mul(a1, a1)), mul(alpha, mul(a3, a3)));
            denominators[i] = sub(mul(n0[i], n0[i]), mul(alpha, mul(n1[i], n1[i])));
        }

        let mut prefix_products = [0u64; NUM_SLOTS];
        prefix_products[0] = denominators[0];
        for i in 1..NUM_SLOTS {
            prefix_products[i] = mul(prefix_products[i - 1], denominators[i]);
        }
        let mut inv = unsafe { inv_mod(prefix_products[NUM_SLOTS - 1], MOD_Q) };
        let mut inverses = [0u64; NUM_SLOTS];
        for i in (1..NUM_SLOTS).rev() {
            inverses[i] = mul(inv, prefix_products[i - 1]);
            inv = mul(inv, denominators[i]);
        }
        inverses[0] = inv;

        let mut result = RingElement::new(Representation::HomogenizedFieldExtensions);
        for i in 0..NUM_SLOTS {
            let (a0, a1, a2, a3) = (part(0, i), part(1, i), part(2, i), part(3, i));
            let coefficients = [
                sub(mul(a0, n0[i]), mul(alpha, mul(a2, n1[i]))),
                sub(mul(alpha, mul(a3, n1[i])), mul(a1, n0[i])),
                sub(mul(a2, n0[i]), mul(a0, n1[i])),
                sub(mul(a1, n1[i]), mul(a3, n0[i])),
            ];
            for (r, c) in coefficients.into_iter().enumerate() {
                result.v[r * NUM_SLOTS + i] = mul(c, inverses[i]);
            }
        }
        result
    }

    pub fn constant_term_from_incomplete_ntt(&self) -> u64 {
        debug_assert_eq!(self.representation, Representation::IncompleteNTT);
        let mut buf = [0u64; DEGREE];
        buf.copy_from_slice(&self.v);
        unsafe {
            eltwise_mult_mod(
                buf.as_mut_ptr(),
                self.v.as_ptr(),
                CONSTANT_TERM_FACTORS.as_ptr(),
                NUM_SLOTS as u64,
                MOD_Q,
            );
        }
        let mut sum = 0u64;
        for i in 0..NUM_SLOTS {
            sum += buf[i];
        }

        // we call it once so it's probably fine
        sum % MOD_Q
    }
}

pub static CONSTANT_TERM_FACTORS: LazyLock<[u64; NUM_SLOTS]> = LazyLock::new(|| {
    let scale = unsafe { inv_mod(NUM_SLOTS as u64, MOD_Q) };
    let mut factors = RingElement::one(Representation::IncompleteNTT);
    unsafe {
        for i in 0..NUM_SLOTS {
            factors.v[i] = multiply_mod(scale, inv_mod(factors.v[i], MOD_Q), MOD_Q);
        }
    }
    factors.v[..NUM_SLOTS].try_into().unwrap()
});

pub static SHIFT_FACTORS: LazyLock<[u64; NUM_SLOTS]> = LazyLock::new(|| {
    let mut factors = [0u64; NUM_SLOTS];
    factors[1] = 1;
    unsafe { ntt_forward_in_place(factors.as_mut_ptr(), factors.len(), MOD_Q) };
    factors
});

pub static FIELD_SHIFT_FACTOR: LazyLock<u64> = LazyLock::new(|| SHIFT_FACTORS[0]);

pub static INV_HALF_DEGREE: LazyLock<u64> =
    LazyLock::new(|| unsafe { power_mod(HALF_DEGREE as u64, MOD_Q - 2, MOD_Q) });

pub static TWO_INV_HALF_DEGREE: LazyLock<u64> =
    LazyLock::new(|| unsafe { multiply_mod(2, *INV_HALF_DEGREE, MOD_Q) });

/// Precomputed permutation and factors for NTT-domain conjugation
/// Generated by analyzing how conjugation transforms NTT coefficients empirically
pub static CONJUGATION_NTT_TRANSFORM: LazyLock<ConjugationTransform> =
    LazyLock::new(|| derive_conjugation_transform());

#[derive(Clone, Debug)]
pub struct ConjugationTransform {
    pub permutation: [[usize; NUM_SLOTS]; SLOT_DEGREE],
    pub factors: [[u64; NUM_SLOTS]; SLOT_DEGREE],
}

/// Conjugates the IncompleteNTT element at `source` into `result`, reading all of `source`
/// before writing, so the two may alias.
#[inline(always)]
unsafe fn conjugate_slots(source: *const u64, result: *mut u64) {
    let transform = &*CONJUGATION_NTT_TRANSFORM;
    let mut temp = [0u64; DEGREE];
    for i in 0..NUM_SLOTS {
        temp[transform.permutation[0][i]] = *source.add(i);
    }
    for part in 1..SLOT_DEGREE {
        eltwise_mult_mod(
            temp.as_mut_ptr().add((SLOT_DEGREE - part) * NUM_SLOTS),
            source.add(part * NUM_SLOTS),
            transform.factors[part].as_ptr(),
            NUM_SLOTS as u64,
            MOD_Q,
        );
    }
    std::ptr::copy_nonoverlapping(temp.as_ptr(), result, NUM_SLOTS);
    for part in 1..SLOT_DEGREE {
        let at = (SLOT_DEGREE - part) * NUM_SLOTS;
        for i in 0..NUM_SLOTS {
            *result.add(at + transform.permutation[part][i]) = temp[at + i];
        }
    }
}

/// Empirically derive the conjugation transformation in NTT domain
///
/// METHODOLOGY:
/// ============
/// We cannot analytically derive the NTT-domain conjugation without deep knowledge
/// of HEXL's internal NTT implementation details. Instead, we use an empirical approach:
///
/// 1. Generate test basis vectors in IncompleteNTT (one-hot encoded)
/// 2. For each basis vector:
///    a. Convert to coefficient space
///    b. Apply conjugation (reverse and negate non-constant coefficients)
///    c. Convert back to IncompleteNTT
///    d. Observe where the value moved and what factor was applied
/// 3. Build permutation tables and factor arrays from observations
///
/// This approach is robust because:
/// - It directly measures HEXL's actual behavior
/// - No assumptions about root ordering or evaluation points
/// - Automatically handles any HEXL implementation details
/// - Will continue working even if HEXL internals change (as long as we regenerate)
fn derive_conjugation_transform() -> ConjugationTransform {
    let mut permutation = [[0usize; NUM_SLOTS]; SLOT_DEGREE];
    let mut factors = [[0u64; NUM_SLOTS]; SLOT_DEGREE];

    for part in 0..SLOT_DEGREE {
        let target = (SLOT_DEGREE - part) % SLOT_DEGREE * NUM_SLOTS;
        for i in 0..NUM_SLOTS {
            let mut test_vec = RingElement::new(Representation::IncompleteNTT);
            test_vec.v[part * NUM_SLOTS + i] = 1;

            let mut conjugated = test_vec.clone();
            conjugated.conjugate_in_place_ref();

            // The value lands in one slot of the mirrored part, and since we started with 1
            // the value there is the factor.
            let j = (0..NUM_SLOTS)
                .find(|&j| conjugated.v[target + j] != 0)
                .expect("conjugation maps part r into part (D - r) mod D");
            debug_assert_eq!(conjugated.v.iter().filter(|&&x| x != 0).count(), 1);
            permutation[part][i] = j;
            factors[part][i] = conjugated.v[target + j];
        }
    }

    ConjugationTransform {
        permutation,
        factors,
    }
}

///// Helpers

pub fn addition(result: &mut RingElement, operand1: &RingElement, operand2: &RingElement) {
    debug_assert!(
        operand1.representation == operand2.representation,
        "Operands have different representations"
    );
    debug_assert!(
        result.representation == operand1.representation,
        "Result has different representation than operands"
    );

    unsafe {
        eltwise_add_mod(
            result.v.as_mut_ptr(),
            operand1.v.as_ptr(),
            operand2.v.as_ptr(),
            DEGREE as u64,
            MOD_Q,
        );
    }
}
pub fn addition_in_place(result_op1: &mut RingElement, operand2: &RingElement) {
    debug_assert!(
        result_op1.representation == operand2.representation,
        "Operands have different representations"
    );

    unsafe {
        eltwise_add_mod(
            result_op1.v.as_mut_ptr(),
            result_op1.v.as_ptr(),
            operand2.v.as_ptr(),
            DEGREE as u64,
            MOD_Q,
        );
    }
}

pub fn subtraction(result: &mut RingElement, operand1: &RingElement, operand2: &RingElement) {
    debug_assert!(
        operand1.representation == operand2.representation,
        "Operands have different representations"
    );
    debug_assert!(
        result.representation == operand1.representation,
        "Result has different representation than operands"
    );

    unsafe {
        eltwise_sub_mod(
            result.v.as_mut_ptr(),
            operand1.v.as_ptr(),
            operand2.v.as_ptr(),
            DEGREE as u64,
            MOD_Q,
        );
    }
}

pub fn subtraction_in_place(result_op1: &mut RingElement, operand2: &RingElement) {
    debug_assert!(
        result_op1.representation == operand2.representation,
        "Operands have different representations"
    );

    unsafe {
        eltwise_sub_mod(
            result_op1.v.as_mut_ptr(),
            result_op1.v.as_ptr(),
            operand2.v.as_ptr(),
            DEGREE as u64,
            MOD_Q,
        );
    }
}

#[inline(always)]
pub fn incomplete_ntt_multiplication(
    result: &mut RingElement,
    operand1: &RingElement,
    operand2: &RingElement,
) {
    debug_assert!(
        operand1.representation == Representation::IncompleteNTT,
        "Operand1 not in Incomplete NTT representation"
    );
    debug_assert!(
        operand2.representation == Representation::IncompleteNTT,
        "Operand2 not in Incomplete NTT representation"
    );
    debug_assert!(
        result.representation == Representation::IncompleteNTT,
        "Result not in Incomplete NTT representation"
    );

    incomplete_ntt_multiplication_inner(result, operand1, operand2, false);
}

#[inline(always)]
pub fn incomplete_ntt_multiplication_in_place(result: &mut RingElement, operand: &RingElement) {
    debug_assert!(
        operand.representation == Representation::IncompleteNTT,
        "Operand not in Incomplete NTT representation"
    );
    debug_assert!(
        result.representation == Representation::IncompleteNTT,
        "Result not in Incomplete NTT representation"
    );

    // The fused AVX512 kernels load all inputs into registers before any store
    // within each 8-element iteration, so result can safely alias operand1.
    unsafe {
        slot_mult(result.v.as_mut_ptr(), result.v.as_ptr(), operand.v.as_ptr());
    }
}

/// Slot-wise product modulo `X^D - zeta_i` of two IncompleteNTT elements.
#[inline(always)]
unsafe fn slot_mult(result: *mut u64, operand1: *const u64, operand2: *const u64) {
    if SLOT_DEGREE == 2 {
        fused_incomplete_ntt_mult(
            result,
            operand1,
            operand2,
            SHIFT_FACTORS.as_ptr(),
            HALF_DEGREE,
            MOD_Q,
        );
    } else {
        fused_slot_mult(result, operand1, operand2, DEGREE, MOD_Q);
    }
}

pub fn incomplete_ntt_multiplication_homogenized(
    result: &mut RingElement,
    operand1: &RingElement,
    operand2: &RingElement,
) {
    debug_assert!(
        operand1.representation == Representation::HomogenizedFieldExtensions,
        "Operand1 not in Homogenized Field Extensions representation"
    );
    debug_assert!(
        operand2.representation == Representation::HomogenizedFieldExtensions,
        "Operand2 not in Homogenized Field Extensions representation"
    );
    debug_assert!(
        result.representation == Representation::HomogenizedFieldExtensions,
        "Result not in Homogenized Field Extensions representation"
    );
    incomplete_ntt_multiplication_inner(result, operand1, operand2, true);
}

#[inline(always)]
pub fn incomplete_ntt_multiplication_inner(
    result: &mut RingElement,
    operand1: &RingElement,
    operand2: &RingElement,
    homogenized: bool,
) {
    let op1_data = &operand1.v;
    let op2_data = &operand2.v;

    if !homogenized {
        // Fused path: all 5 mults + 2 adds in a single AVX512 pass.
        // Eliminates per-call dispatch overhead, redundant int↔float
        // conversions, and intermediate memory traffic.
        unsafe {
            slot_mult(result.v.as_mut_ptr(), op1_data.as_ptr(), op2_data.as_ptr());
        }
        return;
    }

    if SLOT_DEGREE == 4 {
        let a = operand1.split_into_field_extensions();
        let b = operand2.split_into_field_extensions();
        let product: [FieldExtension; NUM_SLOTS] = std::array::from_fn(|i| a[i] * b[i]);
        result.combine_from_field_extensions(&product);
        return;
    }

    // Homogenized path: keep original separate-call implementation
    let mut temp = [0u64; DEGREE];

    unsafe {
        // result_even = op1_even * op2_even
        eltwise_mult_mod(
            result.v.as_mut_ptr(),
            op1_data.as_ptr(),
            op2_data.as_ptr(),
            NUM_SLOTS as u64,
            MOD_Q,
        );

        // result_odd = op1_odd * op2_even
        eltwise_mult_mod(
            result.v.as_mut_ptr().add(NUM_SLOTS),
            op1_data.as_ptr().add(NUM_SLOTS),
            op2_data.as_ptr(),
            NUM_SLOTS as u64,
            MOD_Q,
        );

        // temp = op1_odd * op2_odd
        eltwise_mult_mod(
            temp.as_mut_ptr(),
            op1_data.as_ptr().add(NUM_SLOTS),
            op2_data.as_ptr().add(NUM_SLOTS),
            NUM_SLOTS as u64,
            MOD_Q,
        );

        // result_even += temp * SHIFT_FACTORS[0]
        eltwise_fma_mod(
            result.v.as_mut_ptr(),
            temp.as_ptr(),
            SHIFT_FACTORS[0],
            result.v.as_ptr(),
            NUM_SLOTS as u64,
            MOD_Q,
        );

        // Reuse temp for op1_even * op2_odd
        eltwise_mult_mod(
            temp.as_mut_ptr(),
            op1_data.as_ptr(),
            op2_data.as_ptr().add(NUM_SLOTS),
            NUM_SLOTS as u64,
            MOD_Q,
        );

        // result_odd += temp
        eltwise_add_mod(
            result.v.as_mut_ptr().add(NUM_SLOTS),
            result.v.as_ptr().add(NUM_SLOTS),
            temp.as_ptr(),
            NUM_SLOTS as u64,
            MOD_Q,
        );
    }
}

pub fn naive_polynomial_multiplication(
    result: &mut RingElement,
    operand1: &RingElement,
    operand2: &RingElement,
) {
    debug_assert!(
        operand1.representation == Representation::Coefficients,
        "Operand1 not in Coefficients representation"
    );
    debug_assert!(
        operand2.representation == Representation::Coefficients,
        "Operand2 not in Coefficients representation"
    );
    debug_assert!(
        result.representation == Representation::Coefficients,
        "Result not in Coefficients representation"
    );

    for i in 0..DEGREE {
        result.v[i] = 0;
    }

    for i in 0..DEGREE {
        for j in 0..DEGREE {
            let index = (i + j) % DEGREE;
            let prod = (operand1.v[i] as u128 * operand2.v[j] as u128) % (MOD_Q as u128);
            if i + j >= DEGREE {
                // Negation in modular arithmetic: MOD_Q - value
                let neg_prod = (MOD_Q as u128 - prod) % (MOD_Q as u128);
                result.v[index] = (result.v[index] + neg_prod as u64) % MOD_Q;
            } else {
                result.v[index] = (result.v[index] + prod as u64) % MOD_Q;
            }
        }
    }
}

pub static NORMALIZE_INCOMPLETE_NTT_FACTORS: LazyLock<[u64; DEGREE - NUM_SLOTS]> =
    LazyLock::new(|| get_roots_of_unity_trans().0);

pub static NORMALIZE_INCOMPLETE_NTT_FACTORS_INVERSE: LazyLock<[u64; DEGREE - NUM_SLOTS]> =
    LazyLock::new(|| get_roots_of_unity_trans().1);

/// Slots whose homogenization sends `Y` to a multiple of `X^{-1}`; only degree-4 slots have them.
pub static INVERTED_SLOTS: LazyLock<Vec<usize>> = LazyLock::new(|| match SLOT_DEGREE {
    4 => degree4_homogenization().1,
    _ => Vec::new(),
});

/// Factors of parts `1..D`, part by part, taking slot `i` of an IncompleteNTT element to
/// `Z_q[Y]/(Y^D - alpha)`, `alpha = zeta_0`, and their inverses.
pub fn get_roots_of_unity_trans() -> ([u64; DEGREE - NUM_SLOTS], [u64; DEGREE - NUM_SLOTS]) {
    let roots_translations = match SLOT_DEGREE {
        2 => degree2_homogenization(),
        _ => degree4_homogenization().0,
    };

    let mut roots_translations_inv = [0u64; DEGREE - NUM_SLOTS];

    for i in 0..DEGREE - NUM_SLOTS {
        roots_translations_inv[i] = unsafe { inv_mod(roots_translations[i], MOD_Q) };
    }

    (roots_translations, roots_translations_inv)
}

/// Degree-2 slots: the odd part of slot `i` scales by the smallest power `lambda` of `zeta_i`
/// with `alpha * lambda^2 = zeta_i`.
fn degree2_homogenization() -> [u64; DEGREE - NUM_SLOTS] {
    let mut roots_translations = [0u64; DEGREE - NUM_SLOTS];
    for i in 0..NUM_SLOTS {
        let mut t = 0;
        while (|| {
            let mut ex = RingElement::new(Representation::IncompleteNTT);
            ex.v[NUM_SLOTS + i] = 1;
            let mut ex_0 = RingElement::new(Representation::IncompleteNTT);
            incomplete_ntt_multiplication_inner(&mut ex_0, &ex, &ex, false);
            let mut ex_1 = RingElement::new(Representation::HomogenizedFieldExtensions);

            ex.v[NUM_SLOTS + i] = unsafe { power_mod(SHIFT_FACTORS[i], t, MOD_Q) };
            incomplete_ntt_multiplication_inner(&mut ex_1, &ex, &ex, true);
            ex_0.v != ex_1.v
        })() {
            t = t + 1;
        }
        roots_translations[i] = unsafe { power_mod(SHIFT_FACTORS[i], t, MOD_Q) };
    }
    roots_translations
}

/// Degree-4 slots. `alpha = zeta_0` has order `2 * NUM_SLOTS` and generates the 2-Sylow subgroup
/// of `Z_q^*`, so `zeta_i = alpha^{e_i}` with `e_i` odd. If `e_i = 1 mod 4`, `Y -> mu X` with
/// `mu^4 = alpha / zeta_i` scales part `r` by `mu^{-r}`. If `e_i = 3 mod 4`, `Y -> c X^{-1}` with
/// `c^4 = alpha * zeta_i` swaps parts 1 and 3 and then scales part `r` by `zeta_i / c^r`.
fn degree4_homogenization() -> ([u64; DEGREE - NUM_SLOTS], Vec<usize>) {
    let order = 2 * NUM_SLOTS;
    let mut powers = vec![1u64; order];
    for k in 1..order {
        powers[k] = unsafe { multiply_mod(powers[k - 1], SHIFT_FACTORS[0], MOD_Q) };
    }

    let mut factors = [0u64; DEGREE - NUM_SLOTS];
    let mut inverted = Vec::new();
    for i in 0..NUM_SLOTS {
        let zeta = SHIFT_FACTORS[i];
        let e = powers
            .iter()
            .position(|&p| p == zeta)
            .expect("every zeta_i is a power of zeta_0");
        assert_eq!(e % 2, 1, "zeta_{i} is not a root of Y^NUM_SLOTS + 1");
        let swap = e % 4 == 3;
        let t = match swap {
            true => (e + 1) % order / 4,
            false => (order + 1 - e) % order / 4,
        };
        for r in 1..SLOT_DEGREE {
            let scale = powers[(order - r * t % order) % order];
            factors[(r - 1) * NUM_SLOTS + i] = match swap {
                true => unsafe { multiply_mod(zeta, scale, MOD_Q) },
                false => scale,
            };
        }
        if swap {
            inverted.push(i);
        }
    }
    (factors, inverted)
}

impl Add for &RingElement {
    type Output = RingElement;

    fn add(self, other: Self) -> Self::Output {
        let mut result = RingElement::new(self.representation);
        addition(&mut result, &self, &other);
        result
    }
}

impl AddAssign<&RingElement> for RingElement {
    fn add_assign(&mut self, other: &Self) {
        addition_in_place(self, other);
    }
}

impl Mul for &RingElement {
    type Output = RingElement;

    fn mul(self, other: Self) -> Self::Output {
        let mut result = RingElement::new(self.representation);
        incomplete_ntt_multiplication(&mut result, self, other);
        result
    }
}

impl MulAssign<&RingElement> for RingElement {
    fn mul_assign(&mut self, other: &Self) {
        incomplete_ntt_multiplication_in_place(self, other);
    }
}

impl Sub for &RingElement {
    type Output = RingElement;

    fn sub(self, other: Self) -> Self::Output {
        let mut result = RingElement::new(self.representation);
        subtraction(&mut result, self, other);
        result
    }
}

impl SubAssign<&RingElement> for RingElement {
    fn sub_assign(&mut self, other: &Self) {
        subtraction_in_place(self, other);
    }
}

// Methods below are a bit unorthodox, but they allow to avoid cloning when using
// addition with references.
// In this case, a += (&b, &c) means a = b + c, but without cloning b and c.

impl AddAssign<(&RingElement, &RingElement)> for RingElement {
    fn add_assign(&mut self, other: (&RingElement, &RingElement)) {
        let (op1, op2) = other;
        addition(self, op1, op2);
    }
}

impl SubAssign<(&RingElement, &RingElement)> for RingElement {
    fn sub_assign(&mut self, other: (&RingElement, &RingElement)) {
        let (op1, op2) = other;
        subtraction(self, op1, op2);
    }
}

impl MulAssign<(&RingElement, &RingElement)> for RingElement {
    fn mul_assign(&mut self, other: (&RingElement, &RingElement)) {
        let (op1, op2) = other;
        incomplete_ntt_multiplication(self, op1, op2);
    }
}

// They are small so we can store them on stack.
/// An element of `Z_q[Y]/(Y^D - alpha)`, `alpha = FIELD_SHIFT_FACTOR`: one homogenized slot.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
// Transparent so a slice of field elements can be read as `SLOT_DEGREE * len` contiguous `u64`s.
#[repr(transparent)]
pub struct FieldExtension {
    pub coeffs: [u64; SLOT_DEGREE],
}

impl FieldExtension {
    #[inline]
    pub fn from_base(x: u64) -> Self {
        let mut coeffs = [0u64; SLOT_DEGREE];
        coeffs[0] = x;
        Self { coeffs }
    }

    #[inline]
    pub fn is_base(&self) -> bool {
        self.coeffs[1..].iter().all(|&c| c == 0)
    }
}

/// `x mod q` for `x < 2^(bits(q) + 61)`: Barrett with `floor(2^(64 + s) / q)`, `s = bits(q) - 2`,
/// leaves the quotient at most one short.
#[inline(always)]
fn reduce_wide(x: u128) -> u64 {
    const S: u32 = 62 - MOD_Q.leading_zeros();
    const BARRETT: u64 = ((1u128 << (64 + S)) / MOD_Q as u128) as u64;
    let quotient = (((x >> S) as u64 as u128 * BARRETT as u128) >> 64) as u64;
    let r = (x as u64).wrapping_sub(quotient.wrapping_mul(MOD_Q));
    if r >= MOD_Q {
        r - MOD_Q
    } else {
        r
    }
}

/// Schoolbook product modulo `Y^4 - alpha`, the products accumulated in `u128` and reduced once
/// per coefficient.
#[inline(always)]
fn degree4_mul(a: &[u64; SLOT_DEGREE], b: &[u64; SLOT_DEGREE]) -> [u64; SLOT_DEGREE] {
    let product = |i: usize, j: usize| a[i] as u128 * b[j] as u128;
    let alpha = *FIELD_SHIFT_FACTOR as u128;
    let low = [
        product(0, 0),
        product(0, 1) + product(1, 0),
        product(0, 2) + product(1, 1) + product(2, 0),
        product(0, 3) + product(1, 2) + product(2, 1) + product(3, 0),
    ];
    let high = [
        product(1, 3) + product(2, 2) + product(3, 1),
        product(2, 3) + product(3, 2),
        product(3, 3),
        0,
    ];
    std::array::from_fn(|k| reduce_wide(low[k] + reduce_wide(high[k]) as u128 * alpha))
}

impl Add for FieldExtension {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let coeffs =
            std::array::from_fn(|k| unsafe { add_mod(self.coeffs[k], other.coeffs[k], MOD_Q) });
        Self { coeffs }
    }
}

impl Mul for FieldExtension {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        if SLOT_DEGREE == 4 {
            return Self {
                coeffs: degree4_mul(&self.coeffs, &other.coeffs),
            };
        }
        let a = self.coeffs[0];
        let b = self.coeffs[1];
        let c = other.coeffs[0];
        let d = other.coeffs[1];

        let mut coeffs = [0u64; SLOT_DEGREE];
        unsafe {
            coeffs[0] = add_mod(
                multiply_mod(a, c, MOD_Q as u64),
                multiply_mod(
                    *FIELD_SHIFT_FACTOR,
                    multiply_mod(b, d, MOD_Q as u64),
                    MOD_Q as u64,
                ),
                MOD_Q as u64,
            );
            coeffs[1] = add_mod(
                multiply_mod(a, d, MOD_Q as u64),
                multiply_mod(b, c, MOD_Q as u64),
                MOD_Q as u64,
            );
        }
        Self { coeffs }
    }
}

impl<'a> AddAssign<&'a FieldExtension> for FieldExtension {
    fn add_assign(&mut self, other: &'a FieldExtension) {
        for k in 0..SLOT_DEGREE {
            self.coeffs[k] = unsafe { add_mod(self.coeffs[k], other.coeffs[k], MOD_Q) };
        }
    }
}

impl<'a> AddAssign<(&'a FieldExtension, &'a FieldExtension)> for FieldExtension {
    fn add_assign(&mut self, other: (&'a FieldExtension, &'a FieldExtension)) {
        let (op1, op2) = other;
        for k in 0..SLOT_DEGREE {
            self.coeffs[k] = unsafe { add_mod(op1.coeffs[k], op2.coeffs[k], MOD_Q) };
        }
    }
}
impl<'a> SubAssign<&'a FieldExtension> for FieldExtension {
    fn sub_assign(&mut self, other: &'a FieldExtension) {
        for k in 0..SLOT_DEGREE {
            self.coeffs[k] = unsafe { sub_mod(self.coeffs[k], other.coeffs[k], MOD_Q) };
        }
    }
}

impl<'a> MulAssign<&'a FieldExtension> for FieldExtension {
    fn mul_assign(&mut self, other: &'a FieldExtension) {
        if SLOT_DEGREE == 4 {
            self.coeffs = degree4_mul(&self.coeffs, &other.coeffs);
            return;
        }
        let a = self.coeffs[0];
        let b = self.coeffs[1];
        let c = other.coeffs[0];
        let d = other.coeffs[1];
        unsafe {
            self.coeffs[0] = add_mod(
                multiply_mod(a, c, MOD_Q),
                multiply_mod(*FIELD_SHIFT_FACTOR, multiply_mod(b, d, MOD_Q), MOD_Q),
                MOD_Q,
            );
            self.coeffs[1] = add_mod(multiply_mod(a, d, MOD_Q), multiply_mod(b, c, MOD_Q), MOD_Q);
        }
    }
}

impl<'a> MulAssign<(&'a FieldExtension, &'a FieldExtension)> for FieldExtension {
    fn mul_assign(&mut self, other: (&'a FieldExtension, &'a FieldExtension)) {
        let (lhs, rhs) = other;
        *self = *lhs * *rhs;
    }
}

impl RingElement {
    pub fn compact_size_in_bits(&self) -> usize {
        let resident = self.size_in_bits();
        let mut other = self.clone();
        match self.representation {
            Representation::IncompleteNTT => {
                other.from_incomplete_ntt_to_strided_coefficients();
                other.from_strided_coefficients_to_coefficients();
            }
            Representation::Coefficients => {
                other.from_coefficients_to_strided_coefficients();
                other.from_strided_coefficients_to_incomplete_ntt_representation();
            }
            _ => return resident,
        }
        resident.min(other.size_in_bits())
    }
}

impl SizeableProof for RingElement {
    fn size_in_bits(&self) -> usize {
        let mut size = 0;
        for v in &self.v {
            if *v > MOD_Q {
                panic!("Value exceeds modulus in size_in_bits calculation");
            }
            if *v % MOD_Q == 0 {
                continue; // zero contributes 0 bits
            }
            let size_0 = v.ilog2() as usize + 1;
            let centered = if *v > MOD_Q / 2 { MOD_Q - *v } else { *v };
            let size_1 = centered.ilog2() as usize + 2; // +1 for the sign bit
            size += size_0.min(size_1);
        }
        size
    }
}

impl SizeableProof for FieldExtension {
    fn size_in_bits(&self) -> usize {
        let mut size = 0;
        for v in &self.coeffs {
            let centered = if *v > MOD_Q / 2 { MOD_Q - *v } else { *v };
            if centered == 0 {
                continue; // zero contributes 0 bits
            }
            size += centered.ilog2() as usize + 1; // +1 for the sign bit
        }
        size
    }
}

#[cfg(test)]
mod tests {
    use crate::common::init_common;
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn test_ntt_multiplication_matches_naive() {
        init_common();
        let mut a = RingElement::random(Representation::Coefficients);
        let mut b = RingElement::random(Representation::Coefficients);
        let mut c = RingElement::new(Representation::Coefficients);

        naive_polynomial_multiplication(&mut c, &a, &b);

        a.from_coefficients_to_strided_coefficients();
        b.from_coefficients_to_strided_coefficients();
        a.from_strided_coefficients_to_incomplete_ntt_representation();
        b.from_strided_coefficients_to_incomplete_ntt_representation();

        let mut d = RingElement::new(Representation::IncompleteNTT);
        incomplete_ntt_multiplication(&mut d, &a, &b);
        d.from_incomplete_ntt_to_strided_coefficients();
        d.from_strided_coefficients_to_coefficients();

        debug_assert_eq!(c.v, d.v);
    }

    #[test]
    fn test_homogenized_field_extension_conversion_roundtrip() {
        init_common();
        let mut b = RingElement::random(Representation::Coefficients);
        b.from_coefficients_to_strided_coefficients();
        b.from_strided_coefficients_to_incomplete_ntt_representation();

        let mut b_c = b.clone();
        b_c.from_incomplete_ntt_to_homogenized_field_extensions();
        b_c.from_homogenized_field_extensions_to_incomplete_ntt();

        debug_assert_eq!(b.v, b_c.v);
    }

    #[test]
    fn test_field_extension_split_combine_roundtrip() {
        init_common();
        let mut b = RingElement::random(Representation::Coefficients);
        b.from_coefficients_to_strided_coefficients();
        b.from_strided_coefficients_to_incomplete_ntt_representation();
        b.from_incomplete_ntt_to_homogenized_field_extensions();

        let ext_b: [FieldExtension; NUM_SLOTS] = b.split_into_field_extensions();
        let mut b_reconstructed = RingElement::new(Representation::HomogenizedFieldExtensions);
        b_reconstructed.combine_from_field_extensions(&ext_b);

        debug_assert_eq!(b.v, b_reconstructed.v);
    }

    #[test]
    fn test_hadamard_multiplication_in_field_extensions() {
        init_common();
        let mut a = RingElement::random(Representation::Coefficients);
        let mut b = RingElement::random(Representation::Coefficients);
        let mut c = RingElement::new(Representation::Coefficients);

        naive_polynomial_multiplication(&mut c, &a, &b);

        a.from_coefficients_to_strided_coefficients();
        b.from_coefficients_to_strided_coefficients();
        a.from_strided_coefficients_to_incomplete_ntt_representation();
        b.from_strided_coefficients_to_incomplete_ntt_representation();
        a.from_incomplete_ntt_to_homogenized_field_extensions();
        b.from_incomplete_ntt_to_homogenized_field_extensions();

        let ext_a: [FieldExtension; NUM_SLOTS] = a.split_into_field_extensions();
        let ext_b: [FieldExtension; NUM_SLOTS] = b.split_into_field_extensions();

        let field_extensions_hadamard: [FieldExtension; NUM_SLOTS] = ext_a
            .iter()
            .zip(ext_b.iter())
            .map(|(x, y)| *x * *y)
            .collect::<Vec<FieldExtension>>()
            .try_into()
            .unwrap();

        let mut c_c = RingElement::new(Representation::HomogenizedFieldExtensions);
        c_c.combine_from_field_extensions(&field_extensions_hadamard);
        c_c.from_homogenized_field_extensions_to_incomplete_ntt();
        c_c.from_incomplete_ntt_to_strided_coefficients();
        c_c.from_strided_coefficients_to_coefficients();

        debug_assert_eq!(c.v, c_c.v);
    }

    #[test]
    fn test_homogenized_multiplication_matches_naive() {
        init_common();
        let mut a = RingElement::random(Representation::Coefficients);
        let mut b = RingElement::random(Representation::Coefficients);
        let mut c = RingElement::new(Representation::Coefficients);

        naive_polynomial_multiplication(&mut c, &a, &b);

        a.from_coefficients_to_strided_coefficients();
        b.from_coefficients_to_strided_coefficients();
        a.from_strided_coefficients_to_incomplete_ntt_representation();
        b.from_strided_coefficients_to_incomplete_ntt_representation();
        a.from_incomplete_ntt_to_homogenized_field_extensions();
        b.from_incomplete_ntt_to_homogenized_field_extensions();

        let mut e = RingElement::new(Representation::HomogenizedFieldExtensions);
        incomplete_ntt_multiplication_homogenized(&mut e, &a, &b);
        e.from_homogenized_field_extensions_to_incomplete_ntt();
        e.from_incomplete_ntt_to_strided_coefficients();
        e.from_strided_coefficients_to_coefficients();

        debug_assert_eq!(c.v, e.v);
    }

    #[test]
    fn test_strided_coefficients_conversion_roundtrip() {
        init_common();
        let original = RingElement::random(Representation::Coefficients);
        let mut a = original.clone();

        a.from_coefficients_to_strided_coefficients();
        a.from_strided_coefficients_to_coefficients();

        debug_assert_eq!(original.v, a.v);
    }

    /// Schoolbook product modulo `Y^D - alpha` with `%` reductions.
    fn field_extension_schoolbook(a: &FieldExtension, b: &FieldExtension) -> FieldExtension {
        let q = MOD_Q as u128;
        let mut coeffs = [0u128; SLOT_DEGREE];
        for i in 0..SLOT_DEGREE {
            for j in 0..SLOT_DEGREE {
                let term = a.coeffs[i] as u128 * b.coeffs[j] as u128 % q;
                let (k, term) = match i + j < SLOT_DEGREE {
                    true => (i + j, term),
                    false => (i + j - SLOT_DEGREE, term * *FIELD_SHIFT_FACTOR as u128 % q),
                };
                coeffs[k] = (coeffs[k] + term) % q;
            }
        }
        FieldExtension {
            coeffs: coeffs.map(|c| c as u64),
        }
    }

    #[test]
    fn test_field_extension_multiplication() {
        init_common();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let extremes = [
            FieldExtension {
                coeffs: [MOD_Q - 1; SLOT_DEGREE],
            },
            FieldExtension::from_base(MOD_Q - 1),
        ];
        let random = (0..200).map(|_| FieldExtension {
            coeffs: std::array::from_fn(|_| rng.random_range(0..MOD_Q)),
        });
        let values: Vec<FieldExtension> = extremes.into_iter().chain(random).collect();
        for pair in values.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let expected = field_extension_schoolbook(&a, &b);
            assert_eq!(a * b, expected);
            let mut c = a;
            c *= &b;
            assert_eq!(c, expected);
        }
        let top = extremes[0];
        assert_eq!(top * top, field_extension_schoolbook(&top, &top));
    }

    #[test]
    fn test_strided_layout() {
        init_common();
        let original = RingElement::random(Representation::Coefficients);
        let mut strided = original.clone();
        strided.from_coefficients_to_strided_coefficients();
        for part in 0..SLOT_DEGREE {
            for j in 0..NUM_SLOTS {
                let coefficient = original.v[SLOT_DEGREE * j + part];
                assert_eq!(strided.v[part * NUM_SLOTS + j], coefficient);
            }
        }
    }

    #[test]
    fn test_incomplete_ntt_roundtrip() {
        init_common();
        let original = RingElement::random(Representation::StridedCoefficients);
        let mut a = original.clone();
        a.from_strided_coefficients_to_incomplete_ntt_representation();
        assert_ne!(a.v, original.v);
        a.from_incomplete_ntt_to_strided_coefficients();
        assert_eq!(a.v, original.v);
    }

    /// Slot `i` of an IncompleteNTT element is `a mod (X^D - zeta_i)`, `zeta_i = SHIFT_FACTORS[i]`.
    #[test]
    fn test_slots_are_residues() {
        init_common();
        let mut a = RingElement::random(Representation::Coefficients);
        let coefficients = a.v;
        a.to_representation(Representation::IncompleteNTT);
        for i in 0..NUM_SLOTS {
            let zeta = SHIFT_FACTORS[i] as u128;
            let q = MOD_Q as u128;
            let mut residue = [0u128; SLOT_DEGREE];
            let mut power = 1u128;
            for chunk in coefficients.chunks_exact(SLOT_DEGREE) {
                for (r, c) in chunk.iter().enumerate() {
                    residue[r] = (residue[r] + *c as u128 * power) % q;
                }
                power = power * zeta % q;
            }
            for r in 0..SLOT_DEGREE {
                let value = a.v[r * NUM_SLOTS + i] as u128;
                assert_eq!(value, residue[r], "slot {i}, part {r}");
            }
        }
    }

    #[test]
    fn test_ring_multiplication_in_place_matches_naive() {
        init_common();
        for _ in 0..4 {
            let mut a = RingElement::random(Representation::Coefficients);
            let mut b = RingElement::random(Representation::Coefficients);
            let mut c = RingElement::new(Representation::Coefficients);
            naive_polynomial_multiplication(&mut c, &a, &b);
            a.to_representation(Representation::IncompleteNTT);
            b.to_representation(Representation::IncompleteNTT);
            a *= &b;
            a.to_representation(Representation::Coefficients);
            assert_eq!(a.v, c.v);
        }
    }

    /// Homogenization is a ring isomorphism slot by slot, and the identity on slot 0.
    #[test]
    fn test_homogenization_is_multiplicative() {
        init_common();
        assert!(NORMALIZE_INCOMPLETE_NTT_FACTORS
            .iter()
            .step_by(NUM_SLOTS)
            .all(|&f| f == 1));
        if SLOT_DEGREE == 4 {
            let inverted = INVERTED_SLOTS.len();
            assert!(0 < inverted && inverted < NUM_SLOTS, "{inverted} inverted");
        }
        for _ in 0..8 {
            let a = RingElement::random(Representation::IncompleteNTT);
            let b = RingElement::random(Representation::IncompleteNTT);
            let mut product = &a * &b;
            product.from_incomplete_ntt_to_homogenized_field_extensions();
            let (mut a_h, mut b_h) = (a.clone(), b.clone());
            a_h.from_incomplete_ntt_to_homogenized_field_extensions();
            b_h.from_incomplete_ntt_to_homogenized_field_extensions();
            let a_s = a_h.split_into_field_extensions();
            let b_s = b_h.split_into_field_extensions();
            let expected = product.split_into_field_extensions();
            for i in 0..NUM_SLOTS {
                assert_eq!(a_s[i] * b_s[i], expected[i], "slot {i}");
            }
            assert_eq!(a_h.slot_zero(), a.slot_zero());
            a_h.from_homogenized_field_extensions_to_incomplete_ntt();
            assert_eq!(a_h.v, a.v);
        }
    }

    #[test]
    fn test_conjugation_ref() {
        init_common();
        let a = RingElement::random(Representation::IncompleteNTT);
        let mut a_conj = a.clone();
        let b = RingElement::new(Representation::IncompleteNTT);
        let mut b_conj = b.clone();

        a_conj.conjugate_in_place_ref();
        // a.conjugate_in_place_ref();

        b_conj.conjugate_in_place_ref();
        // debug_assert_eq!(a.v, a_conj.v);

        let a_plus_b = &a + &b;
        let a_times_b = &a * &b;

        let mut a_plus_b_conj = a_plus_b.clone();
        a_plus_b_conj.conjugate_in_place_ref();

        let mut a_times_b_conj = a_times_b.clone();
        a_times_b_conj.conjugate_in_place_ref();

        debug_assert_eq!(&a_conj + &b_conj, a_plus_b_conj);
        debug_assert_eq!(&a_conj * &b_conj, a_times_b_conj);
    }

    #[test]
    fn test_conjugation() {
        init_common();
        let a = RingElement::random(Representation::IncompleteNTT);
        let mut a_conj = a.clone();
        let mut a_conj_ref = a.clone();
        a_conj.conjugate_in_place();
        a_conj_ref.conjugate_in_place_ref();
        debug_assert_eq!(a_conj.v, a_conj_ref.v);
    }

    #[test]
    fn test_norm_squared_via_conjugation() {
        init_common();
        let mut vector: Vec<RingElement> = vec![
            RingElement::random_bounded(Representation::Coefficients, 10),
            RingElement::random_bounded(Representation::Coefficients, 10),
            RingElement::random_bounded(Representation::Coefficients, 10),
            RingElement::random_bounded(Representation::Coefficients, 10),
        ];

        let mut two_norm_squared = 0u64;
        for e in vector.iter_mut() {
            for coeff in e.v.iter() {
                let centered = if *coeff > MOD_Q / 2 {
                    MOD_Q - *coeff // Interpret as negative
                } else {
                    *coeff
                };
                two_norm_squared = unsafe {
                    add_mod(
                        two_norm_squared,
                        multiply_mod(centered, centered, MOD_Q),
                        MOD_Q,
                    )
                };
            }
            e.to_representation(Representation::IncompleteNTT);
        }

        let mut vector_conj = vector.clone();
        for e in vector_conj.iter_mut() {
            e.conjugate_in_place();
        }

        let mut inner_product = RingElement::new(Representation::IncompleteNTT);
        for (e1, e2) in vector.iter().zip(vector_conj.iter()) {
            let mut prod = RingElement::new(Representation::IncompleteNTT);
            prod *= (e1, e2);
            inner_product += &prod;
        }
        inner_product.from_incomplete_ntt_to_strided_coefficients();
        inner_product.from_strided_coefficients_to_coefficients();
        let ct = inner_product.v[0];

        debug_assert_eq!(ct, two_norm_squared);
    }

    #[test]
    fn test_conjugate_into_matches_in_place() {
        init_common();

        let mut a = RingElement::random(Representation::Coefficients);
        a.from_coefficients_to_strided_coefficients();
        a.from_strided_coefficients_to_incomplete_ntt_representation();

        let original = a.clone();

        let mut result = RingElement::new(Representation::IncompleteNTT);
        a.conjugate_into(&mut result);

        let mut expected = a.clone();
        expected.conjugate_in_place();

        debug_assert_eq!(result, expected);
        debug_assert_eq!(a, original);
    }

    #[test]
    fn test_constant_term_from_incomplete_ntt() {
        init_common();

        let mut a = RingElement::random(Representation::IncompleteNTT);
        let computed_constant_term = a.constant_term_from_incomplete_ntt();
        a.from_incomplete_ntt_to_strided_coefficients();
        a.from_strided_coefficients_to_coefficients();
        let expected_constant_term = a.v[0];

        debug_assert_eq!(expected_constant_term, computed_constant_term % MOD_Q);
    }

    /// Verifies that the fused incomplete-NTT multiplication kernel produces
    /// bit-identical results to the original separate-call implementation.
    #[test]
    fn test_fused_incomplete_ntt_mult_matches_separate() {
        init_common();
        if SLOT_DEGREE != 2 {
            return; // the separate calls are the degree-2 slot product
        }

        for _ in 0..20 {
            let op1 = RingElement::random(Representation::IncompleteNTT);
            let op2 = RingElement::random(Representation::IncompleteNTT);

            // --- Reference: separate eltwise calls (the original algorithm) ---
            let mut ref_result = RingElement::new(Representation::IncompleteNTT);
            let temp = &mut [0u64; DEGREE];
            unsafe {
                // ref_even = op1_even * op2_even
                eltwise_mult_mod(
                    ref_result.v.as_mut_ptr(),
                    op1.v.as_ptr(),
                    op2.v.as_ptr(),
                    HALF_DEGREE as u64,
                    MOD_Q,
                );
                // ref_odd = op1_odd * op2_even
                eltwise_mult_mod(
                    ref_result.v.as_mut_ptr().add(HALF_DEGREE),
                    op1.v.as_ptr().add(HALF_DEGREE),
                    op2.v.as_ptr(),
                    HALF_DEGREE as u64,
                    MOD_Q,
                );
                // temp = op1_odd * op2_odd
                eltwise_mult_mod(
                    temp.as_mut_ptr(),
                    op1.v.as_ptr().add(HALF_DEGREE),
                    op2.v.as_ptr().add(HALF_DEGREE),
                    HALF_DEGREE as u64,
                    MOD_Q,
                );
                // temp *= shift_factors
                eltwise_mult_mod(
                    temp.as_mut_ptr(),
                    temp.as_ptr(),
                    SHIFT_FACTORS.as_ptr(),
                    HALF_DEGREE as u64,
                    MOD_Q,
                );
                // ref_even += temp
                eltwise_add_mod(
                    ref_result.v.as_mut_ptr(),
                    ref_result.v.as_ptr(),
                    temp.as_ptr(),
                    HALF_DEGREE as u64,
                    MOD_Q,
                );
                // temp2 = op1_even * op2_odd
                eltwise_mult_mod(
                    temp.as_mut_ptr(),
                    op1.v.as_ptr(),
                    op2.v.as_ptr().add(HALF_DEGREE),
                    HALF_DEGREE as u64,
                    MOD_Q,
                );
                // ref_odd += temp2
                eltwise_add_mod(
                    ref_result.v.as_mut_ptr().add(HALF_DEGREE),
                    ref_result.v.as_ptr().add(HALF_DEGREE),
                    temp.as_ptr(),
                    HALF_DEGREE as u64,
                    MOD_Q,
                );
            }

            // --- Fused path ---
            let mut fused_result = RingElement::new(Representation::IncompleteNTT);
            unsafe {
                fused_incomplete_ntt_mult(
                    fused_result.v.as_mut_ptr(),
                    op1.v.as_ptr(),
                    op2.v.as_ptr(),
                    SHIFT_FACTORS.as_ptr(),
                    HALF_DEGREE,
                    MOD_Q,
                );
            }

            assert_eq!(
                ref_result.v, fused_result.v,
                "Fused ring mult diverged from reference"
            );
        }
    }

    #[test]
    fn test_inverse_times_self_is_one() {
        init_common();

        for _ in 0..10 {
            let a = RingElement::random(Representation::HomogenizedFieldExtensions);
            let a_inv = a.inverse();

            let mut product = RingElement::new(Representation::HomogenizedFieldExtensions);
            incomplete_ntt_multiplication_homogenized(&mut product, &a, &a_inv);

            let one = RingElement::one(Representation::HomogenizedFieldExtensions);
            assert_eq!(product.v, one.v, "a * a^{{-1}} should equal 1");
        }
    }

    #[test]
    fn test_inverse_slot_by_slot() {
        init_common();

        let a = RingElement::random(Representation::HomogenizedFieldExtensions);
        let a_inv = a.inverse();

        let slots = a.split_into_field_extensions();
        let inv_slots = a_inv.split_into_field_extensions();

        let one = FieldExtension::from_base(1);
        for i in 0..NUM_SLOTS {
            let product = slots[i] * inv_slots[i];
            assert_eq!(
                product, one,
                "Slot {i} inverse incorrect: {:?} * {:?} = {:?}",
                slots[i], inv_slots[i], product
            );
        }
    }

    #[test]
    fn test_inverse_of_one_is_one() {
        init_common();

        let one = RingElement::one(Representation::HomogenizedFieldExtensions);
        let one_inv = one.inverse();
        assert_eq!(one.v, one_inv.v, "1^{{-1}} should equal 1");
    }
}
