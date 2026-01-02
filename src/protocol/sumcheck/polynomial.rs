use std::{cmp::max, usize::MAX};

use crate::common::ring_arithmetic::RingElement;

pub struct Polynomial {
    pub coefficients: [RingElement; 4], // we support up to cubic polynomials for now, idk if we need more. maybe we can make this dynamic later
    pub nof_coefficients: usize,
}

impl Polynomial {
    pub fn new(
        nof_coefficients: usize,
        representation: crate::common::ring_arithmetic::Representation,
    ) -> Self {
        assert!(
            nof_coefficients <= 4,
            "Only up to cubic polynomials are supported for now"
        );
        Polynomial {
            coefficients: [
                RingElement::zero(representation),
                RingElement::zero(representation),
                RingElement::zero(representation),
                RingElement::zero(representation),
            ],
            nof_coefficients,
        }
    }

    pub fn at_zero(&self) -> RingElement {
        self.coefficients[0].clone()
    }

    pub fn at_one(&self) -> RingElement {
        let mut result = RingElement::zero(self.coefficients[0].representation);
        for i in 0..self.nof_coefficients {
            result += &self.coefficients[i];
        }
        result
    }

    pub fn at(&self, point: &RingElement) -> RingElement {
        let mut result = RingElement::zero(self.coefficients[0].representation);
        let mut power = RingElement::one(self.coefficients[0].representation);
        for i in 0..self.nof_coefficients {
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
        self.nof_coefficients = 0;
    }
}

pub fn mul_poly_into(result: &mut Polynomial, poly_0: &Polynomial, poly_1: &Polynomial) {
    assert!(
        poly_0.nof_coefficients + poly_1.nof_coefficients - 1 <= 4,
        "Resulting polynomial degree exceeds supported maximum"
    );

    for i in 0..poly_0.nof_coefficients {
        for j in 0..poly_1.nof_coefficients {
            result.coefficients[i + j] += &(&poly_0.coefficients[i] * &poly_1.coefficients[j]);
        }
    }
    result.nof_coefficients = poly_0.nof_coefficients + poly_1.nof_coefficients - 1;
}

pub fn add_poly_into(result: &mut Polynomial, poly_0: &Polynomial, poly_1: &Polynomial) {
    for i in 0..poly_0.nof_coefficients {
        result.coefficients[i] = &poly_0.coefficients[i] + &poly_1.coefficients[i];
    }
    result.nof_coefficients = max(poly_0.nof_coefficients, poly_1.nof_coefficients);
}

pub fn add_poly_in_place(result: &mut Polynomial, poly: &Polynomial) {
    for i in 0..poly.nof_coefficients {
        result.coefficients[i] += &poly.coefficients[i];
    }

    result.nof_coefficients = max(result.nof_coefficients, poly.nof_coefficients);
}
