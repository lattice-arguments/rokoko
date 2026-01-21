use crate::common::config::*;
use crate::hexl::bindings::*;
use crate::protocol::config::SizeableProof;
use num::traits::ops::inv;
use rand::Rng;
use std::cell::RefCell;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};
use std::sync::LazyLock;

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum Representation {
    Coefficients, // This should not be used almost ever. Use only for printing or debugging.
    EvenOddCoefficients, // In this representation, coefficients are stored as even part followed by odd part. This is so that NTT can be applied more easily.
    IncompleteNTT, // Incomplete NTT representation, where even and odd parts are separately transformed.
    HomogenizedFieldExtensions, // We use that reprentation so that "Incomplete NTT slots" are homogenized, i.e. they are all of
                                // the structure Zq[X] / <X^2 + \alpha>, i.e. \alpha is the same for each slot.
}

// DO NOT derive Copy here, as RingElement is large.
#[derive(PartialEq, Clone, Debug)]
pub struct RingElement {
    pub v: [u64; DEGREE],
    pub representation: Representation,
}

thread_local! {
    static RNG: RefCell<rand::rngs::ThreadRng> =
        RefCell::new(rand::thread_rng());
}

// TODO: handle better the case when ring element is zero.
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
                element.v[i] = rng.gen_range(0..MOD_Q);
            }
        });

        element
    }

    pub fn one(representation: Representation) -> Self {
        let mut element = Self {
            v: [0; DEGREE],
            representation: Representation::EvenOddCoefficients,
        };
        element.v[0] = 1;

        element.to_representation(representation);

        element
    }

    pub fn all(value: u64, representation: Representation) -> Self {
        let mut element = Self {
            v: [0; DEGREE],
            representation: Representation::EvenOddCoefficients,
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
            representation: Representation::EvenOddCoefficients,
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
                element.v[i] = rng.random_range(0..bound);
                if rng.random_bool(0.5) {
                    element.v[i] = MOD_Q - element.v[i];
                }
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

    pub fn from_even_odd_coefficients_to_incomplete_ntt_representation(&mut self) {
        debug_assert!(
            self.representation == Representation::EvenOddCoefficients,
            "Already in Incomplete NTT representation"
        );

        unsafe {
            ntt_forward_in_place(self.v.as_mut_ptr(), HALF_DEGREE, MOD_Q);
            ntt_forward_in_place(self.v.as_mut_ptr().add(HALF_DEGREE), HALF_DEGREE, MOD_Q);
        }

        self.representation = Representation::IncompleteNTT;
    }

    pub fn from_incomplete_ntt_to_even_odd_coefficients(&mut self) {
        debug_assert!(
            self.representation == Representation::IncompleteNTT,
            "Not in Incomplete NTT representation"
        );

        unsafe {
            ntt_inverse_in_place(self.v.as_mut_ptr(), HALF_DEGREE, MOD_Q);
            ntt_inverse_in_place(self.v.as_mut_ptr().add(HALF_DEGREE), HALF_DEGREE, MOD_Q);
        }

        self.representation = Representation::EvenOddCoefficients;
    }

    pub fn from_coefficients_to_even_odd_coefficients(&mut self) {
        debug_assert!(
            self.representation == Representation::Coefficients,
            "Not in Coefficients representation"
        );

        let mut temp = [0u64; DEGREE];

        for i in 0..(DEGREE / 2) {
            temp[i] = self.v[2 * i];
            temp[i + (DEGREE / 2)] = self.v[2 * i + 1];
        }

        self.v = temp;
        self.representation = Representation::EvenOddCoefficients;
    }

    pub fn from_even_odd_coefficients_to_coefficients(&mut self) {
        debug_assert!(
            self.representation == Representation::EvenOddCoefficients,
            "Not in Even-Odd Coefficients representation"
        );

        let mut temp = [0u64; DEGREE];

        for i in 0..(DEGREE / 2) {
            temp[2 * i] = self.v[i];
            temp[2 * i + 1] = self.v[i + (DEGREE / 2)];
        }

        self.v = temp;
        self.representation = Representation::Coefficients;
    }

    pub fn from_incomplete_ntt_to_homogenized_field_extensions(&mut self) {
        debug_assert!(
            self.representation == Representation::IncompleteNTT,
            "Not in Incomplete NTT representation"
        );

        unsafe {
            eltwise_mult_mod(
                self.v.as_mut_ptr().add(HALF_DEGREE),
                self.v.as_ptr().add(HALF_DEGREE),
                NORMALIZE_INCOMPLETE_NTT_FACTORS.as_ptr(),
                (HALF_DEGREE) as u64,
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
                self.v.as_mut_ptr().add(HALF_DEGREE),
                self.v.as_ptr().add(HALF_DEGREE),
                NORMALIZE_INCOMPLETE_NTT_FACTORS_INVERSE.as_ptr(),
                (HALF_DEGREE) as u64,
                MOD_Q,
            );
        }
        self.representation = Representation::IncompleteNTT;
    }

    pub fn to_representation(&mut self, representation: Representation) {
        match (self.representation, representation) {
            (Representation::Coefficients, Representation::EvenOddCoefficients) => {
                self.from_coefficients_to_even_odd_coefficients()
            }
            (Representation::Coefficients, Representation::IncompleteNTT) => {
                self.from_coefficients_to_even_odd_coefficients();
                self.from_even_odd_coefficients_to_incomplete_ntt_representation();
            }
            (Representation::Coefficients, Representation::HomogenizedFieldExtensions) => {
                self.from_coefficients_to_even_odd_coefficients();
                self.from_even_odd_coefficients_to_incomplete_ntt_representation();
                self.from_incomplete_ntt_to_homogenized_field_extensions();
            }
            (Representation::EvenOddCoefficients, Representation::IncompleteNTT) => {
                self.from_even_odd_coefficients_to_incomplete_ntt_representation()
            }
            (Representation::EvenOddCoefficients, Representation::HomogenizedFieldExtensions) => {
                self.from_even_odd_coefficients_to_incomplete_ntt_representation();
                self.from_incomplete_ntt_to_homogenized_field_extensions();
            }
            (Representation::IncompleteNTT, Representation::HomogenizedFieldExtensions) => {
                self.from_incomplete_ntt_to_homogenized_field_extensions()
            }
            (Representation::HomogenizedFieldExtensions, Representation::IncompleteNTT) => {
                self.from_homogenized_field_extensions_to_incomplete_ntt()
            }
            (Representation::HomogenizedFieldExtensions, Representation::EvenOddCoefficients) => {
                self.from_homogenized_field_extensions_to_incomplete_ntt();
                self.from_incomplete_ntt_to_even_odd_coefficients();
            }
            (Representation::HomogenizedFieldExtensions, Representation::Coefficients) => {
                self.from_homogenized_field_extensions_to_incomplete_ntt();
                self.from_incomplete_ntt_to_even_odd_coefficients();
                self.from_even_odd_coefficients_to_coefficients();
            }
            (Representation::IncompleteNTT, Representation::EvenOddCoefficients) => {
                self.from_incomplete_ntt_to_even_odd_coefficients();
            }
            (Representation::IncompleteNTT, Representation::Coefficients) => {
                self.from_incomplete_ntt_to_even_odd_coefficients();
                self.from_even_odd_coefficients_to_coefficients();
            }
            (Representation::EvenOddCoefficients, Representation::Coefficients) => {
                self.from_even_odd_coefficients_to_coefficients();
            }
            _ => {
                // nothing to do
            }
        }
    }

    // Probably should never be used
    pub fn split_into_quadratic_extensions(&self) -> [QuadraticExtension; HALF_DEGREE] {
        debug_assert!(
            self.representation == Representation::HomogenizedFieldExtensions,
            "RingElement not in Homogenized Field Extensions representation"
        );

        let mut result = [QuadraticExtension { coeffs: [0u64; 2] }; HALF_DEGREE];

        for i in 0..HALF_DEGREE {
            result[i].coeffs[0] = self.v[i];
            result[i].coeffs[1] = self.v[i + HALF_DEGREE];
        }

        result
    }

    pub fn combine_from_quadratic_extensions(
        &mut self,
        extensions: &[QuadraticExtension; HALF_DEGREE],
    ) {
        debug_assert!(
            self.representation == Representation::HomogenizedFieldExtensions,
            "RingElement not in Homogenized Field Extensions representation"
        );

        for i in 0..HALF_DEGREE {
            self.v[i] = extensions[i].coeffs[0];
            self.v[i + HALF_DEGREE] = extensions[i].coeffs[1];
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
        self.from_incomplete_ntt_to_even_odd_coefficients();
        self.from_even_odd_coefficients_to_coefficients();

        // Reverse and negate coefficients 1 to n-1
        for i in 1..(DEGREE / 2 + 1) {
            let temp = self.v[i];
            self.v[i] = MOD_Q - self.v[DEGREE - i];
            self.v[DEGREE - i] = MOD_Q - temp;
        }

        self.from_coefficients_to_even_odd_coefficients();
        self.from_even_odd_coefficients_to_incomplete_ntt_representation();
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
        // In IncompleteNTT representation: f(X) = f_even(X^2) + X·f_odd(X^2)
        // Conjugation: f(X^{-1}) = f_even(X^{-2}) - X^{n-1}·f_odd(X^{-2})
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
        // This gives us:
        // - even_permutation[i]: where even[i] goes after conjugation
        // - odd_permutation[i]: where odd[i] goes after conjugation
        // - odd_factors[i]: scaling factor for odd[i] (accounts for X -> -X^{n-1})
        //
        // IMPLEMENTATION:
        // ==============
        // Apply the precomputed transformation directly in NTT space:
        // - even_new[even_permutation[i]] = even_old[i]
        // - odd_new[odd_permutation[i]] = odd_old[i] * odd_factors[i]
        //
        // Benefits:
        // - No NTT transforms needed (pure O(n) operation)
        // - Provably correct (matches reference implementation by construction)
        // - Robust to HEXL implementation details

        debug_assert_eq!(self.representation, Representation::IncompleteNTT);

        let transform = &*CONJUGATION_NTT_TRANSFORM;
        let mut temp = get_temp_buffer();

        // Apply even part permutation
        for i in 0..HALF_DEGREE {
            temp[transform.even_permutation[i]] = self.v[i];
        }
        self.v[..HALF_DEGREE].copy_from_slice(&temp[..HALF_DEGREE]);

        // Apply odd part: multiply by factors, then permute
        unsafe {
            eltwise_mult_mod(
                temp.as_mut_ptr(),
                self.v.as_ptr().add(HALF_DEGREE),
                transform.odd_factors.as_ptr(),
                HALF_DEGREE as u64,
                MOD_Q,
            );
        }
        for i in 0..HALF_DEGREE {
            self.v[HALF_DEGREE + transform.odd_permutation[i]] = temp[i];
        }
    }

    #[inline]
    pub fn conjugate_into(&self, result: &mut RingElement) {
        debug_assert_eq!(self.representation, Representation::IncompleteNTT);
        result.representation = self.representation;

        let transform = &*CONJUGATION_NTT_TRANSFORM;
        let mut temp = get_temp_buffer();
        for i in 0..HALF_DEGREE {
            temp[transform.even_permutation[i]] = self.v[i];
        }
        result.v[..HALF_DEGREE].copy_from_slice(&temp[..HALF_DEGREE]);
        unsafe {
            eltwise_mult_mod(
                temp.as_mut_ptr(),
                self.v.as_ptr().add(HALF_DEGREE),
                transform.odd_factors.as_ptr(),
                HALF_DEGREE as u64,
                MOD_Q,
            );
        }
        for i in 0..HALF_DEGREE {
            result.v[HALF_DEGREE + transform.odd_permutation[i]] = temp[i];
        }
    }

    pub fn conjugate(&self) -> RingElement {
        let mut result = RingElement::new(self.representation);
        self.conjugate_into(&mut result);
        result
    }

    pub fn constant_term_from_incomplete_ntt(&self) -> u64 {
        debug_assert_eq!(self.representation, Representation::IncompleteNTT);
        let buf = &mut *get_temp_buffer();
        buf.copy_from_slice(&self.v);
        unsafe {
            eltwise_mult_mod(
                buf.as_mut_ptr(),
                self.v.as_ptr(),
                CONSTANT_TERM_FACTORS.as_ptr(),
                HALF_DEGREE as u64,
                MOD_Q,
            );
        }
        let mut sum = 0u64;
        for i in 0..HALF_DEGREE {
            // TODO: vectorized sum
            sum += buf[i];
        }

        // we call it once so it's probably fine
        sum % MOD_Q
    }
}

pub static CONSTANT_TERM_FACTORS: LazyLock<[u64; HALF_DEGREE]> = LazyLock::new(|| {
    let scale = unsafe { inv_mod(HALF_DEGREE as u64, MOD_Q) };
    let mut factors = RingElement::one(Representation::IncompleteNTT);
    unsafe {
        for i in 0..HALF_DEGREE {
            factors.v[i] = multiply_mod(scale, inv_mod(factors.v[i], MOD_Q), MOD_Q);
        }
    }
    factors.v[..HALF_DEGREE].try_into().unwrap()
});

pub static SHIFT_FACTORS: LazyLock<[u64; HALF_DEGREE]> = LazyLock::new(|| {
    let mut factors = [0u64; HALF_DEGREE];
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
    pub even_permutation: [usize; HALF_DEGREE],
    pub odd_permutation: [usize; HALF_DEGREE],
    pub odd_factors: [u64; HALF_DEGREE],
}

pub static mut temp_buffer: LazyLock<[u64; DEGREE]> = LazyLock::new(|| [0u64; DEGREE]);

#[inline(always)]
fn get_temp_buffer() -> &'static mut [u64; DEGREE] {
    unsafe { &mut temp_buffer }
}

pub static mut aux: LazyLock<RingElement> =
    LazyLock::new(|| RingElement::new(Representation::IncompleteNTT));

#[inline(always)]
fn get_aux() -> &'static mut RingElement {
    unsafe { &mut aux }
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
    let mut even_permutation = [0usize; HALF_DEGREE];
    let mut odd_permutation = [0usize; HALF_DEGREE];
    let mut odd_factors = [0u64; HALF_DEGREE];

    // Derive even part permutation
    // Test each position in the even part
    for i in 0..HALF_DEGREE {
        let mut test_vec = RingElement::new(Representation::IncompleteNTT);
        test_vec.v[i] = 1; // One-hot encode position i in even part

        // Apply reference conjugation (via coefficient space)
        let mut conjugated = test_vec.clone();
        conjugated.conjugate_in_place_ref();

        // Find where the 1 moved to in the even part
        for j in 0..HALF_DEGREE {
            if conjugated.v[j] != 0 {
                even_permutation[i] = j;
                break;
            }
        }
    }

    // Derive odd part permutation and factors
    // Test each position in the odd part
    for i in 0..HALF_DEGREE {
        let mut test_vec = RingElement::new(Representation::IncompleteNTT);
        test_vec.v[HALF_DEGREE + i] = 1; // One-hot encode position i in odd part

        // Apply reference conjugation
        let mut conjugated = test_vec.clone();
        conjugated.conjugate_in_place_ref();

        // Find where the value moved to and what factor was applied
        for j in 0..HALF_DEGREE {
            if conjugated.v[HALF_DEGREE + j] != 0 {
                odd_permutation[i] = j;
                // The factor is the value at the new position
                // (since we started with 1)
                odd_factors[i] = conjugated.v[HALF_DEGREE + j];
                break;
            }
        }
    }

    ConjugationTransform {
        even_permutation,
        odd_permutation,
        odd_factors,
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

    // TODO: We need a copy of the original result because the in-place routine
    // overwrites `result` while still reading from it. Without cloning, the
    // computation produces incorrect values.
    // Remove this cloning by implementing a proper in-place algorithm.
    // incomplete_ntt_multiplication_in_place_inner(result, operand, false);
    // seems to be broken for in-place multiplication.
    let original = get_aux();
    original.set_from(result);
    incomplete_ntt_multiplication(result, &original, operand);
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
    let mut temp = get_temp_buffer();

    let op1_data = &operand1.v;
    let op2_data = &operand2.v;

    unsafe {
        // result_even = op1_even * op2_even
        eltwise_mult_mod(
            result.v.as_mut_ptr(),
            op1_data.as_ptr(),
            op2_data.as_ptr(),
            HALF_DEGREE as u64,
            MOD_Q,
        );

        // result_odd = op1_odd * op2_even
        eltwise_mult_mod(
            result.v.as_mut_ptr().add(HALF_DEGREE),
            op1_data.as_ptr().add(HALF_DEGREE),
            op2_data.as_ptr(),
            HALF_DEGREE as u64,
            MOD_Q,
        );

        // temp = op1_odd * op2_odd
        eltwise_mult_mod(
            temp.as_mut_ptr(),
            op1_data.as_ptr().add(HALF_DEGREE),
            op2_data.as_ptr().add(HALF_DEGREE),
            HALF_DEGREE as u64,
            MOD_Q,
        );

        if homogenized {
            // result_even += temp * SHIFT_FACTORS[0]
            eltwise_fma_mod(
                result.v.as_mut_ptr(),
                temp.as_ptr(),
                SHIFT_FACTORS[0],
                result.v.as_ptr(),
                HALF_DEGREE as u64,
                MOD_Q,
            );
        } else {
            // Apply shift factors
            eltwise_mult_mod(
                temp.as_mut_ptr(),
                temp.as_ptr(),
                SHIFT_FACTORS.as_ptr(),
                HALF_DEGREE as u64,
                MOD_Q,
            );

            // result_even += temp
            eltwise_add_mod(
                result.v.as_mut_ptr(),
                result.v.as_ptr(),
                temp.as_ptr(),
                HALF_DEGREE as u64,
                MOD_Q,
            );
        }

        // Reuse temp for op1_even * op2_odd
        eltwise_mult_mod(
            temp.as_mut_ptr(),
            op1_data.as_ptr(),
            op2_data.as_ptr().add(HALF_DEGREE),
            HALF_DEGREE as u64,
            MOD_Q,
        );

        // result_odd += temp
        eltwise_add_mod(
            result.v.as_mut_ptr().add(HALF_DEGREE),
            result.v.as_ptr().add(HALF_DEGREE),
            temp.as_ptr(),
            HALF_DEGREE as u64,
            MOD_Q,
        );
    }
}

#[inline(always)]
pub fn incomplete_ntt_multiplication_in_place_inner(
    result: &mut RingElement,
    operand1: &RingElement,
    homogenized: bool,
) {
    let mut temp = get_temp_buffer();

    let op1_data = &operand1.v;

    unsafe {
        // result_even = op1_even * op2_even
        eltwise_mult_mod(
            result.v.as_mut_ptr(),
            op1_data.as_ptr(),
            result.v.as_ptr(),
            HALF_DEGREE as u64,
            MOD_Q,
        );

        // result_odd = op1_odd * op2_even
        eltwise_mult_mod(
            result.v.as_mut_ptr().add(HALF_DEGREE),
            op1_data.as_ptr().add(HALF_DEGREE),
            result.v.as_ptr(),
            HALF_DEGREE as u64,
            MOD_Q,
        );

        // temp = op1_odd * op2_odd
        eltwise_mult_mod(
            temp.as_mut_ptr(),
            op1_data.as_ptr().add(HALF_DEGREE),
            result.v.as_ptr().add(HALF_DEGREE),
            HALF_DEGREE as u64,
            MOD_Q,
        );

        if homogenized {
            // result_even += temp * SHIFT_FACTORS[0]
            eltwise_fma_mod(
                result.v.as_mut_ptr(),
                temp.as_ptr(),
                SHIFT_FACTORS[0],
                result.v.as_ptr(),
                HALF_DEGREE as u64,
                MOD_Q,
            );
        } else {
            // Apply shift factors
            eltwise_mult_mod(
                temp.as_mut_ptr(),
                temp.as_ptr(),
                SHIFT_FACTORS.as_ptr(),
                HALF_DEGREE as u64,
                MOD_Q,
            );

            // result_even += temp
            eltwise_add_mod(
                result.v.as_mut_ptr(),
                result.v.as_ptr(),
                temp.as_ptr(),
                HALF_DEGREE as u64,
                MOD_Q,
            );
        }

        // Reuse temp for op1_even * op2_odd
        eltwise_mult_mod(
            temp.as_mut_ptr(),
            op1_data.as_ptr(),
            result.v.as_ptr().add(HALF_DEGREE),
            HALF_DEGREE as u64,
            MOD_Q,
        );

        // result_odd += temp
        eltwise_add_mod(
            result.v.as_mut_ptr().add(HALF_DEGREE),
            result.v.as_ptr().add(HALF_DEGREE),
            temp.as_ptr(),
            HALF_DEGREE as u64,
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

pub static NORMALIZE_INCOMPLETE_NTT_FACTORS: LazyLock<[u64; HALF_DEGREE]> =
    LazyLock::new(|| get_roots_of_unity_trans().0);

pub static NORMALIZE_INCOMPLETE_NTT_FACTORS_INVERSE: LazyLock<[u64; HALF_DEGREE]> =
    LazyLock::new(|| get_roots_of_unity_trans().1);

pub fn get_roots_of_unity_trans() -> ([u64; HALF_DEGREE], [u64; HALF_DEGREE]) {
    let mut roots_translations = [0u64; HALF_DEGREE];
    for i in 0..HALF_DEGREE {
        let mut t = 0;
        while (|| {
            let mut ex = RingElement::new(Representation::IncompleteNTT);
            ex.v[HALF_DEGREE + i] = 1;
            let mut ex_0 = RingElement::new(Representation::IncompleteNTT);
            incomplete_ntt_multiplication_inner(&mut ex_0, &ex, &ex, false);
            let mut ex_1 = RingElement::new(Representation::HomogenizedFieldExtensions);

            ex.v[HALF_DEGREE + i] = unsafe { power_mod(SHIFT_FACTORS[i], t, MOD_Q) };
            incomplete_ntt_multiplication_inner(&mut ex_1, &ex, &ex, true);
            ex_0.v != ex_1.v
        })() {
            t = t + 1;
        }
        roots_translations[i] = unsafe { power_mod(SHIFT_FACTORS[i], t, MOD_Q) };
    }

    let mut roots_translations_inv = [0u64; HALF_DEGREE];

    for i in 0..HALF_DEGREE {
        roots_translations_inv[i] = unsafe { inv_mod(roots_translations[i], MOD_Q) };
    }

    (roots_translations, roots_translations_inv)
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
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct QuadraticExtension {
    pub coeffs: [u64; 2],
}

impl Add for QuadraticExtension {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let coeffs = unsafe {
            [
                add_mod(self.coeffs[0], other.coeffs[0], MOD_Q as u64),
                add_mod(self.coeffs[1], other.coeffs[1], MOD_Q as u64),
            ]
        };
        Self { coeffs }
    }
}

impl Mul for QuadraticExtension {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let a = self.coeffs[0];
        let b = self.coeffs[1];
        let c = other.coeffs[0];
        let d = other.coeffs[1];

        let coeffs = unsafe {
            [
                add_mod(
                    multiply_mod(a, c, MOD_Q as u64),
                    multiply_mod(
                        *FIELD_SHIFT_FACTOR,
                        multiply_mod(b, d, MOD_Q as u64),
                        MOD_Q as u64,
                    ),
                    MOD_Q as u64,
                ),
                add_mod(
                    multiply_mod(a, d, MOD_Q as u64),
                    multiply_mod(b, c, MOD_Q as u64),
                    MOD_Q as u64,
                ),
            ]
        };
        Self { coeffs }
    }
}

impl<'a> AddAssign<&'a QuadraticExtension> for QuadraticExtension {
    fn add_assign(&mut self, other: &'a QuadraticExtension) {
        unsafe {
            self.coeffs[0] = add_mod(self.coeffs[0], other.coeffs[0], MOD_Q);
            self.coeffs[1] = add_mod(self.coeffs[1], other.coeffs[1], MOD_Q);
        }
    }
}

impl<'a> SubAssign<&'a QuadraticExtension> for QuadraticExtension {
    fn sub_assign(&mut self, other: &'a QuadraticExtension) {
        unsafe {
            self.coeffs[0] = sub_mod(self.coeffs[0], other.coeffs[0], MOD_Q);
            self.coeffs[1] = sub_mod(self.coeffs[1], other.coeffs[1], MOD_Q);
        }
    }
}

impl<'a> MulAssign<&'a QuadraticExtension> for QuadraticExtension {
    fn mul_assign(&mut self, other: &'a QuadraticExtension) {
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

impl<'a> MulAssign<(&'a QuadraticExtension, &'a QuadraticExtension)> for QuadraticExtension {
    fn mul_assign(&mut self, other: (&'a QuadraticExtension, &'a QuadraticExtension)) {
        let (lhs, rhs) = other;
        *self = *lhs * *rhs;
    }
}

impl SizeableProof for RingElement {
    fn size_in_bits(&self) -> usize {
        let mut size = 0;
        for v in &self.v {
            if *v > MOD_Q {
                panic!("Value exceeds modulus in size_in_bits calculation");
            }
            let centered = if *v > MOD_Q / 2 { MOD_Q - *v } else { *v };
            if centered == 0 {
                continue; // zero contributes 0 bits
            }
            size += centered.ilog2() as usize + 1; // +1 for the sign bit
        }
        size
    }
}

impl SizeableProof for QuadraticExtension {
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
    use crate::common::{init_common, norms};

    use super::*;

    #[test]
    fn test_ntt_multiplication_matches_naive() {
        init_common();
        let mut a = RingElement::random(Representation::Coefficients);
        let mut b = RingElement::random(Representation::Coefficients);
        let mut c = RingElement::new(Representation::Coefficients);

        naive_polynomial_multiplication(&mut c, &a, &b);

        a.from_coefficients_to_even_odd_coefficients();
        b.from_coefficients_to_even_odd_coefficients();
        a.from_even_odd_coefficients_to_incomplete_ntt_representation();
        b.from_even_odd_coefficients_to_incomplete_ntt_representation();

        let mut d = RingElement::new(Representation::IncompleteNTT);
        incomplete_ntt_multiplication(&mut d, &a, &b);
        d.from_incomplete_ntt_to_even_odd_coefficients();
        d.from_even_odd_coefficients_to_coefficients();

        debug_assert_eq!(c.v, d.v);
    }

    #[test]
    fn test_homogenized_field_extension_conversion_roundtrip() {
        init_common();
        let mut b = RingElement::random(Representation::Coefficients);
        b.from_coefficients_to_even_odd_coefficients();
        b.from_even_odd_coefficients_to_incomplete_ntt_representation();

        let mut b_c = b.clone();
        b_c.from_incomplete_ntt_to_homogenized_field_extensions();
        b_c.from_homogenized_field_extensions_to_incomplete_ntt();

        debug_assert_eq!(b.v, b_c.v);
    }

    #[test]
    fn test_quadratic_extension_split_combine_roundtrip() {
        init_common();
        let mut b = RingElement::random(Representation::Coefficients);
        b.from_coefficients_to_even_odd_coefficients();
        b.from_even_odd_coefficients_to_incomplete_ntt_representation();
        b.from_incomplete_ntt_to_homogenized_field_extensions();

        let ext_b: [QuadraticExtension; HALF_DEGREE] = b.split_into_quadratic_extensions();
        let mut b_reconstructed = RingElement::new(Representation::HomogenizedFieldExtensions);
        b_reconstructed.combine_from_quadratic_extensions(&ext_b);

        debug_assert_eq!(b.v, b_reconstructed.v);
    }

    #[test]
    fn test_hadamard_multiplication_in_quadratic_extensions() {
        init_common();
        let mut a = RingElement::random(Representation::Coefficients);
        let mut b = RingElement::random(Representation::Coefficients);
        let mut c = RingElement::new(Representation::Coefficients);

        naive_polynomial_multiplication(&mut c, &a, &b);

        a.from_coefficients_to_even_odd_coefficients();
        b.from_coefficients_to_even_odd_coefficients();
        a.from_even_odd_coefficients_to_incomplete_ntt_representation();
        b.from_even_odd_coefficients_to_incomplete_ntt_representation();
        a.from_incomplete_ntt_to_homogenized_field_extensions();
        b.from_incomplete_ntt_to_homogenized_field_extensions();

        let ext_a: [QuadraticExtension; HALF_DEGREE] = a.split_into_quadratic_extensions();
        let ext_b: [QuadraticExtension; HALF_DEGREE] = b.split_into_quadratic_extensions();

        let quadratic_fields_hadamard: [QuadraticExtension; HALF_DEGREE] = ext_a
            .iter()
            .zip(ext_b.iter())
            .map(|(x, y)| *x * *y)
            .collect::<Vec<QuadraticExtension>>()
            .try_into()
            .unwrap();

        let mut c_c = RingElement::new(Representation::HomogenizedFieldExtensions);
        c_c.combine_from_quadratic_extensions(&quadratic_fields_hadamard);
        c_c.from_homogenized_field_extensions_to_incomplete_ntt();
        c_c.from_incomplete_ntt_to_even_odd_coefficients();
        c_c.from_even_odd_coefficients_to_coefficients();

        debug_assert_eq!(c.v, c_c.v);
    }

    #[test]
    fn test_homogenized_multiplication_matches_naive() {
        init_common();
        let mut a = RingElement::random(Representation::Coefficients);
        let mut b = RingElement::random(Representation::Coefficients);
        let mut c = RingElement::new(Representation::Coefficients);

        naive_polynomial_multiplication(&mut c, &a, &b);

        a.from_coefficients_to_even_odd_coefficients();
        b.from_coefficients_to_even_odd_coefficients();
        a.from_even_odd_coefficients_to_incomplete_ntt_representation();
        b.from_even_odd_coefficients_to_incomplete_ntt_representation();
        a.from_incomplete_ntt_to_homogenized_field_extensions();
        b.from_incomplete_ntt_to_homogenized_field_extensions();

        let mut e = RingElement::new(Representation::HomogenizedFieldExtensions);
        incomplete_ntt_multiplication_homogenized(&mut e, &a, &b);
        e.from_homogenized_field_extensions_to_incomplete_ntt();
        e.from_incomplete_ntt_to_even_odd_coefficients();
        e.from_even_odd_coefficients_to_coefficients();

        debug_assert_eq!(c.v, e.v);
    }

    #[test]
    fn test_even_odd_coefficients_conversion_roundtrip() {
        init_common();
        let original = RingElement::random(Representation::Coefficients);
        let mut a = original.clone();

        a.from_coefficients_to_even_odd_coefficients();
        a.from_even_odd_coefficients_to_coefficients();

        debug_assert_eq!(original.v, a.v);
    }

    #[test]
    fn test_quadratic_extension_multiplication() {
        let qe1 = QuadraticExtension { coeffs: [2, 3] };
        let qe2 = QuadraticExtension { coeffs: [4, 5] };
        let result = qe1 * qe2;

        // (2 + 3X)(4 + 5X) = 8 + 10X + 12X + 15X^2 = 8 + 22X + 15*shift
        let expected_c0 = unsafe {
            add_mod(
                multiply_mod(2, 4, MOD_Q),
                multiply_mod(*FIELD_SHIFT_FACTOR, multiply_mod(3, 5, MOD_Q), MOD_Q),
                MOD_Q,
            )
        };
        let expected_c1 =
            unsafe { add_mod(multiply_mod(2, 5, MOD_Q), multiply_mod(3, 4, MOD_Q), MOD_Q) };

        debug_assert_eq!(result.coeffs[0], expected_c0);
        debug_assert_eq!(result.coeffs[1], expected_c1);
    }

    #[test]
    fn test_conjugation_ref() {
        init_common();
        let mut a = RingElement::random(Representation::IncompleteNTT);
        let mut a_conj = a.clone();
        let mut b = RingElement::new(Representation::IncompleteNTT);
        let mut b_conj = b.clone();

        a_conj.conjugate_in_place_ref();
        // a.conjugate_in_place_ref();

        b_conj.conjugate_in_place_ref();
        // debug_assert_eq!(a.v, a_conj.v);

        let mut a_plus_b = &a + &b;
        let mut a_times_b = &a * &b;

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
        let mut a = RingElement::random(Representation::IncompleteNTT);
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
        inner_product.from_incomplete_ntt_to_even_odd_coefficients();
        inner_product.from_even_odd_coefficients_to_coefficients();
        let ct = inner_product.v[0];

        debug_assert_eq!(ct, two_norm_squared);
    }

    #[test]
    fn test_conjugate_into_matches_in_place() {
        init_common();

        let mut a = RingElement::random(Representation::Coefficients);
        a.from_coefficients_to_even_odd_coefficients();
        a.from_even_odd_coefficients_to_incomplete_ntt_representation();

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
        a.from_incomplete_ntt_to_even_odd_coefficients();
        a.from_even_odd_coefficients_to_coefficients();
        let expected_constant_term = a.v[0];

        debug_assert_eq!(expected_constant_term, computed_constant_term % MOD_Q);
    }
}

// #[test]
// fn test_conjugation() {
//     use crate::common::init_common;

//     init_common();
//     let mut a = RingElement::random(Representation::IncompleteNTT);
//     let mut a_conj = a.clone();
//     a_conj.conjugate_in_place_ref();
//     a.conjugate_in_place_ref();
//     debug_assert_eq!(a.v, a_conj.v);
// }
