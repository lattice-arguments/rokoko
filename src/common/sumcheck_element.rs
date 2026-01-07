use std::ops::{AddAssign, MulAssign, SubAssign};

use crate::common::{
    matrix::new_vec_zero_preallocated,
    ring_arithmetic::{Representation, RingElement},
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
    fn one() -> Self;
    fn set_zero(&mut self);

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

    fn one() -> Self {
        RingElement::one(Representation::IncompleteNTT)
    }

    fn set_zero(&mut self) {
        RingElement::set_zero(self);
    }

    fn allocate_zero_vec(len: usize) -> Vec<Self> {
        new_vec_zero_preallocated(len)
    }
}
