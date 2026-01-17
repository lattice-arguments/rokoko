use std::ops::{AddAssign, MulAssign, SubAssign};

use crate::common::{
    arithmetic::{ONE, ONE_QUAD, TWO, TWO_QUAD, ZERO, ZERO_QUAD},
    matrix::new_vec_zero_preallocated,
    ring_arithmetic::{Representation, RingElement},
    QuadraticExtension,
};

/// Minimal operations Sumcheck and Polynomial need from a field-like element.
/// Designed to be implementable by `RingElement` now and other element types (e.g. `QuadraticExtension`) later.
pub trait SumcheckElement:
    Clone
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + for<'a> MulAssign<(&'a Self, &'a Self)>
{
    fn zero() -> Self;
    fn zero_ref() -> &'static Self;
    fn one() -> Self;
    fn one_ref() -> &'static Self;
    fn two_ref() -> &'static Self;
    fn set_zero(&mut self);
    fn set_from(&mut self, other: &Self);

    /// Allocate a zero vector. Implementors can override to use preallocated pools.
    fn allocate_zero_vec(len: usize) -> Vec<Self>
    where
        Self: Sized,
    {
        vec![Self::zero(); len]
    }
}

impl SumcheckElement for RingElement {
    fn zero() -> Self {
        RingElement::zero(Representation::IncompleteNTT)
    }

    fn zero_ref() -> &'static Self {
        &ZERO
    }

    fn one() -> Self {
        RingElement::one(Representation::IncompleteNTT)
    }

    fn one_ref() -> &'static Self {
        &ONE
    }

    fn two_ref() -> &'static Self {
        &TWO
    }

    fn set_zero(&mut self) {
        RingElement::set_zero(self);
    }

    fn allocate_zero_vec(len: usize) -> Vec<Self> {
        new_vec_zero_preallocated(len)
    }

    fn set_from(&mut self, other: &Self) {
        self.set_from(other);
    }
}

impl SumcheckElement for QuadraticExtension {
    fn zero() -> Self {
        QuadraticExtension { coeffs: [0, 0] }
    }

    fn one() -> Self {
        QuadraticExtension { coeffs: [1, 0] }
    }

    fn one_ref() -> &'static Self {
        &ONE_QUAD
    }

    fn two_ref() -> &'static Self {
        &TWO_QUAD
    }

    fn zero_ref() -> &'static Self {
        &ZERO_QUAD
    }

    fn set_zero(&mut self) {
        self.coeffs = [0, 0];
    }

    fn allocate_zero_vec(len: usize) -> Vec<Self> {
        vec![QuadraticExtension { coeffs: [0, 0] }; len]
    }

    fn set_from(&mut self, other: &Self) {
        self.coeffs = other.coeffs;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct U64Wrapper(pub u64);

impl AddAssign<&U64Wrapper> for U64Wrapper {
    fn add_assign(&mut self, rhs: &U64Wrapper) {
        self.0 += rhs.0;
    }
}

impl SubAssign<&U64Wrapper> for U64Wrapper {
    fn sub_assign(&mut self, rhs: &U64Wrapper) {
        self.0 -= rhs.0;
    }
}

impl MulAssign<&U64Wrapper> for U64Wrapper {
    fn mul_assign(&mut self, rhs: &U64Wrapper) {
        self.0 *= rhs.0;
    }
}

impl MulAssign<(&U64Wrapper, &U64Wrapper)> for U64Wrapper {
    fn mul_assign(&mut self, (a, b): (&U64Wrapper, &U64Wrapper)) {
        self.0 = a.0 * b.0;
    }
}
