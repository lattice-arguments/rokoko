use std::cmp::max;

use crate::common::ring_arithmetic::RingElement;

/// Dense polynomial representation used throughout the sumcheck routines.
/// The storage is fixed to four coefficients because, at the moment,
/// the protocol only needs up to cubic polynomials.
pub struct Polynomial {
    // coefficients[i] corresponds to x^i.
    pub coefficients: [RingElement; 4],
    /// How many coefficients are actually active (degree + 1).
    pub num_coefficients: usize,
}

impl Polynomial {
    pub fn new(
        num_coefficients: usize,
        representation: crate::common::ring_arithmetic::Representation,
    ) -> Self {
        assert!(
            num_coefficients <= 4,
            "Only up to cubic polynomials are supported for now"
        );
        Polynomial {
            coefficients: [
                RingElement::zero(representation),
                RingElement::zero(representation),
                RingElement::zero(representation),
                RingElement::zero(representation),
            ],
            num_coefficients,
        }
    }

    /// Evaluate at x = 0.
    pub fn at_zero(&self) -> RingElement {
        self.coefficients[0].clone()
    }

    /// Evaluate at x = 1 by summing all coefficients.
    pub fn at_one(&self) -> RingElement {
        let mut result = RingElement::zero(self.coefficients[0].representation);
        for i in 0..self.num_coefficients {
            result += &self.coefficients[i];
        }
        result
    }

    /// Evaluate using straightforward power accumulation.
    pub fn at(&self, point: &RingElement) -> RingElement {
        let mut result = RingElement::zero(self.coefficients[0].representation);
        let mut power = RingElement::one(self.coefficients[0].representation);
        for i in 0..self.num_coefficients {
            let mut term = self.coefficients[i].clone();
            term *= &power;
            result += &term;
            power *= point;
        }
        result
    }

    pub fn set_zero(&mut self) {
        for coeff in self.coefficients.iter_mut() {
            coeff.set_zero();
        }
        self.num_coefficients = 0;
    }
}

/// Multiply two polynomials and store the result in `result`.
pub fn mul_poly_into(result: &mut Polynomial, poly_0: &Polynomial, poly_1: &Polynomial) {
    assert!(
        poly_0.num_coefficients + poly_1.num_coefficients - 1 <= 4,
        "Resulting polynomial degree exceeds supported maximum"
    );

    for i in 0..poly_0.num_coefficients {
        for j in 0..poly_1.num_coefficients {
            result.coefficients[i + j] += &(&poly_0.coefficients[i] * &poly_1.coefficients[j]);
        }
    }
    result.num_coefficients = poly_0.num_coefficients + poly_1.num_coefficients - 1;
}

/// Add two polynomials and store the sum in `result`.
pub fn add_poly_into(result: &mut Polynomial, poly_0: &Polynomial, poly_1: &Polynomial) {
    for i in 0..poly_0.num_coefficients {
        result.coefficients[i] = &poly_0.coefficients[i] + &poly_1.coefficients[i];
    }
    result.num_coefficients = max(poly_0.num_coefficients, poly_1.num_coefficients);
}

/// Add `poly` into `result` in place.
pub fn add_poly_in_place(result: &mut Polynomial, poly: &Polynomial) {
    for i in 0..poly.num_coefficients {
        result.coefficients[i] += &poly.coefficients[i];
    }

    result.num_coefficients = max(result.num_coefficients, poly.num_coefficients);
}

/// Subtract `poly` from `result` in place.
pub fn sub_poly_in_place(result: &mut Polynomial, poly: &Polynomial) {
    for i in 0..poly.num_coefficients {
        result.coefficients[i] -= &poly.coefficients[i];
    }

    result.num_coefficients = max(result.num_coefficients, poly.num_coefficients);
}
