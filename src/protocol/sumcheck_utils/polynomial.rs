use std::cmp::max;

use crate::common::{ring_arithmetic::RingElement, sumcheck_element::SumcheckElement};

/// Dense polynomial representation used throughout the sumcheck routines.
/// The storage is fixed to four coefficients because, at the moment,
/// the protocol only needs up to cubic polynomials.
#[derive(Clone, Debug)]
pub struct Polynomial<E: SumcheckElement = RingElement> {
    // coefficients[i] corresponds to x^i.
    pub coefficients: [E; 4],
    /// How many coefficients are actually active (degree + 1).
    pub num_coefficients: usize,
}

impl<E: SumcheckElement> Polynomial<E> {
    pub fn new(num_coefficients: usize) -> Self {
        debug_assert!(
            num_coefficients <= 3,
            "Only up to cubic polynomials are supported for now"
        );
        Polynomial {
            coefficients: std::array::from_fn(|_| E::zero()),
            num_coefficients,
        }
    }

    /// Evaluate at x = 0.
    pub fn at_zero(&self) -> E {
        self.coefficients[0].clone()
    }

    /// Evaluate at x = 1 by summing all coefficients.
    pub fn at_one(&self) -> E {
        let mut result = E::zero();
        for i in 0..self.num_coefficients {
            result += &self.coefficients[i];
        }
        result
    }

    /// Evaluate using straightforward power accumulation.
    pub fn at(&self, point: &E) -> E {
        let mut result = E::zero();
        let mut power = E::one();
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

    /// Copy the contents of `other` into `self`.
    #[inline]
    pub fn copy_from(&mut self, other: &Polynomial<E>) {
        self.num_coefficients = other.num_coefficients;
        for i in 0..other.num_coefficients {
            self.coefficients[i].set_from(&other.coefficients[i]);
        }
    }
}

/// Multiply two polynomials and store the result in `result`.
#[inline]
pub fn mul_poly_into<E: SumcheckElement>(
    result: &mut Polynomial<E>,
    poly_0: &Polynomial<E>,
    poly_1: &Polynomial<E>,
) {
    debug_assert!(
        poly_0.num_coefficients + poly_1.num_coefficients - 1 <= 4,
        "Resulting polynomial degree exceeds supported maximum"
    );

    // result.set_zero();

    if poly_0.num_coefficients == 2 && poly_1.num_coefficients == 2 {
        // Both are linear. Use Karatsuba with 3 multiplications:
        // z0 = a0*b0, z2 = a1*b1, z1 = (a0+a1)(b0+b1) - z0 - z2.

        let (first, rest) = result.coefficients.split_at_mut(1);
        let (second, third) = rest.split_at_mut(1);

        // first = a0 + a1
        first[0].set_from(&poly_0.coefficients[0]);
        first[0] += &poly_0.coefficients[1];

        // third = b0 + b1
        third[0].set_from(&poly_1.coefficients[0]);
        third[0] += &poly_1.coefficients[1];

        // second = (a0 + a1) * (b0 + b1)
        second[0] *= (&first[0], &third[0]);

        // first = a0 * b0, third = a1 * b1
        first[0] *= (&poly_0.coefficients[0], &poly_1.coefficients[0]);
        third[0] *= (&poly_0.coefficients[1], &poly_1.coefficients[1]);

        // second = z1 = second - first - third
        second[0] -= &first[0];
        second[0] -= &third[0];

        result.num_coefficients = 3;
        return;
    }

    // we handle the case of one linear and one quadratic polynomial separately for efficiency
    if poly_1.num_coefficients == 3 && poly_0.num_coefficients == 2 {
        // First is linear, second is quadratic: (a0 + a1*x) * (b0 + b1*x + b2*x^2) = a0*b0 + (a0*b1 + a1*b0)*x + (a0*b2 + a1*b1)*x^2 + a1*b2*x^3

        let (first, rest) = result.coefficients.split_at_mut(1);
        let (second, rest) = rest.split_at_mut(1);
        let (third, fourth) = rest.split_at_mut(1);

        // // first, the second
        first[0] *= (&poly_0.coefficients[0], &poly_1.coefficients[1]); // buffer
        second[0] *= (&poly_0.coefficients[1], &poly_1.coefficients[0]);
        second[0] += &first[0];

        // third
        first[0] *= (&poly_0.coefficients[0], &poly_1.coefficients[2]);
        third[0] *= (&poly_0.coefficients[1], &poly_1.coefficients[1]);
        third[0] += &first[0];

        // fourth
        fourth[0] *= (&poly_0.coefficients[1], &poly_1.coefficients[2]);

        // first
        first[0] *= (&poly_0.coefficients[0], &poly_1.coefficients[0]);

        result.num_coefficients = 4;
        return;
    }

    if poly_0.num_coefficients == 3 && poly_1.num_coefficients == 2 {
        panic!("Not implemented yet");
    }

    // This works only in one poly is constant
    for i in 0..poly_0.num_coefficients {
        for j in 0..poly_1.num_coefficients {
            result.coefficients[i + j] *= (&poly_0.coefficients[i], &poly_1.coefficients[j])
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

#[inline]
/// Add `poly` into `result` in place.
pub fn add_poly_in_place<E: SumcheckElement>(result: &mut Polynomial<E>, poly: &Polynomial<E>) {
    for i in 0..poly.num_coefficients {
        result.coefficients[i] += &poly.coefficients[i];
    }

    result.num_coefficients = max(result.num_coefficients, poly.num_coefficients);
}

#[inline]
/// Subtract `poly` from `result` in place.
pub fn sub_poly_in_place<E: SumcheckElement>(result: &mut Polynomial<E>, poly: &Polynomial<E>) {
    for i in 0..poly.num_coefficients {
        result.coefficients[i] -= &poly.coefficients[i];
    }

    result.num_coefficients = max(result.num_coefficients, poly.num_coefficients);
}
