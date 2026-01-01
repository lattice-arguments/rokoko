use crate::common::config::*;
use crate::hexl::bindings::*;
use num::pow::Pow;
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
        assert!(
            self.representation == Representation::HomogenizedFieldExtensions,
            "RingElement not in Homogenized Field Extensions representation"
        );

        let mut result = [QuadraticExtension {
            coeffs: [0u64; 2],
            shift: 0,
        }; HALF_DEGREE];

        for i in 0..HALF_DEGREE {
            result[i].coeffs[0] = self.v[i];
            result[i].coeffs[1] = self.v[i + HALF_DEGREE];
            result[i].shift = SHIFT_FACTORS[0];
        }

        result
    }

    // Probably should never be used
    pub fn combine_from_quadratic_extensions(
        &mut self,
        extensions: &[QuadraticExtension; HALF_DEGREE],
    ) {
        assert!(
            self.representation == Representation::HomogenizedFieldExtensions,
            "RingElement not in Homogenized Field Extensions representation"
        );

        for i in 0..HALF_DEGREE {
            self.v[i] = extensions[i].coeffs[0];
            self.v[i + HALF_DEGREE] = extensions[i].coeffs[1];
        }
    }

    pub fn set_zero(&mut self) {
        // TODO: optimize with memset (or Rust's equivalent)
        for i in 0..DEGREE {
            self.v[i] = 0;
        }
    }

    pub fn conjugate_in_place(&mut self) {
        // TODO: implement
    }

    pub fn set_from(&mut self, other: &RingElement) {
        self.v.copy_from_slice(&other.v);
        self.representation = other.representation;
    }
}

pub static SHIFT_FACTORS: LazyLock<[u64; HALF_DEGREE]> = LazyLock::new(|| {
    let mut factors = [0u64; HALF_DEGREE];
    factors[1] = 1;
    unsafe { ntt_forward_in_place(factors.as_mut_ptr(), factors.len(), MOD_Q) };
    factors
});

pub static mut temp_buffer: LazyLock<[u64; DEGREE]> = LazyLock::new(|| [0u64; DEGREE]);

fn get_temp_buffer() -> &'static mut [u64; DEGREE] {
    unsafe { &mut temp_buffer }
}

///// Helpers

pub fn addition(result: &mut RingElement, operand1: &RingElement, operand2: &RingElement) {
    assert!(
        operand1.representation == operand2.representation,
        "Operands have different representations"
    );
    assert!(
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
    assert!(
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
    assert!(
        operand1.representation == operand2.representation,
        "Operands have different representations"
    );
    assert!(
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
    assert!(
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

pub fn incomplete_ntt_multiplication(
    result: &mut RingElement,
    operand1: &RingElement,
    operand2: &RingElement,
) {
    assert!(
        operand1.representation == Representation::IncompleteNTT,
        "Operand1 not in Incomplete NTT representation"
    );
    assert!(
        operand2.representation == Representation::IncompleteNTT,
        "Operand2 not in Incomplete NTT representation"
    );
    assert!(
        result.representation == Representation::IncompleteNTT,
        "Result not in Incomplete NTT representation"
    );

    incomplete_ntt_multiplication_inner(result, operand1, operand2, false);
}

pub fn incomplete_ntt_multiplication_in_place(result: &mut RingElement, operand: &RingElement) {
    assert!(
        operand.representation == Representation::IncompleteNTT,
        "Operand not in Incomplete NTT representation"
    );
    assert!(
        result.representation == Representation::IncompleteNTT,
        "Result not in Incomplete NTT representation"
    );

    // TODO: We need a copy of the original result because the in-place routine
    // overwrites `result` while still reading from it. Without cloning, the
    // computation produces incorrect values.
    // Remove this cloning by implementing a proper in-place algorithm.
    // incomplete_ntt_multiplication_in_place_inner(result, operand, false);
    // seems to be broken for in-place multiplication.
    let original = result.clone();
    incomplete_ntt_multiplication(result, &original, operand);
}

pub fn incomplete_ntt_multiplication_homogenized(
    result: &mut RingElement,
    operand1: &RingElement,
    operand2: &RingElement,
) {
    assert!(
        operand1.representation == Representation::HomogenizedFieldExtensions,
        "Operand1 not in Homogenized Field Extensions representation"
    );
    assert!(
        operand2.representation == Representation::HomogenizedFieldExtensions,
        "Operand2 not in Homogenized Field Extensions representation"
    );
    assert!(
        result.representation == Representation::HomogenizedFieldExtensions,
        "Result not in Homogenized Field Extensions representation"
    );
    incomplete_ntt_multiplication_inner(result, operand1, operand2, true);
}

#[inline]
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

#[inline]
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
    shift: u64,
}

impl Add for QuadraticExtension {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert!(
            self.shift == other.shift,
            "Shifts must be the same for addition"
        );
        let coeffs = unsafe {
            [
                add_mod(self.coeffs[0], other.coeffs[0], MOD_Q as u64),
                add_mod(self.coeffs[1], other.coeffs[1], MOD_Q as u64),
            ]
        };
        Self {
            coeffs,
            shift: self.shift, // Assuming shift remains unchanged
        }
    }
}

impl Mul for QuadraticExtension {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        assert!(
            self.shift == other.shift,
            "Shifts must be the same for multiplication"
        );
        let a = self.coeffs[0];
        let b = self.coeffs[1];
        let c = other.coeffs[0];
        let d = other.coeffs[1];

        let coeffs = unsafe {
            [
                add_mod(
                    multiply_mod(a, c, MOD_Q as u64),
                    multiply_mod(self.shift, multiply_mod(b, d, MOD_Q as u64), MOD_Q as u64),
                    MOD_Q as u64,
                ),
                add_mod(
                    multiply_mod(a, d, MOD_Q as u64),
                    multiply_mod(b, c, MOD_Q as u64),
                    MOD_Q as u64,
                ),
            ]
        };
        Self {
            coeffs,
            shift: self.shift, // Assuming shift remains unchanged
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::common::init_common;

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

        assert_eq!(c.v, d.v);
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

        assert_eq!(b.v, b_c.v);
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

        assert_eq!(b.v, b_reconstructed.v);
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

        assert_eq!(c.v, c_c.v);
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

        assert_eq!(c.v, e.v);
    }

    #[test]
    fn test_even_odd_coefficients_conversion_roundtrip() {
        init_common();
        let original = RingElement::random(Representation::Coefficients);
        let mut a = original.clone();

        a.from_coefficients_to_even_odd_coefficients();
        a.from_even_odd_coefficients_to_coefficients();

        assert_eq!(original.v, a.v);
    }

    #[test]
    fn test_quadratic_extension_addition() {
        let qe1 = QuadraticExtension {
            coeffs: [1, 2],
            shift: 5,
        };
        let qe2 = QuadraticExtension {
            coeffs: [3, 4],
            shift: 5,
        };
        let result = qe1 + qe2;

        assert_eq!(result.coeffs[0], (1 + 3) % MOD_Q);
        assert_eq!(result.coeffs[1], (2 + 4) % MOD_Q);
    }

    #[test]
    fn test_quadratic_extension_multiplication() {
        let qe1 = QuadraticExtension {
            coeffs: [2, 3],
            shift: SHIFT_FACTORS[0],
        };
        let qe2 = QuadraticExtension {
            coeffs: [4, 5],
            shift: SHIFT_FACTORS[0],
        };
        let result = qe1 * qe2;

        // (2 + 3X)(4 + 5X) = 8 + 10X + 12X + 15X^2 = 8 + 22X + 15*shift
        let expected_c0 = unsafe {
            add_mod(
                multiply_mod(2, 4, MOD_Q),
                multiply_mod(SHIFT_FACTORS[0], multiply_mod(3, 5, MOD_Q), MOD_Q),
                MOD_Q,
            )
        };
        let expected_c1 =
            unsafe { add_mod(multiply_mod(2, 5, MOD_Q), multiply_mod(3, 4, MOD_Q), MOD_Q) };

        assert_eq!(result.coeffs[0], expected_c0);
        assert_eq!(result.coeffs[1], expected_c1);
    }
}
