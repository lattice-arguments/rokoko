use std::ops::Index;

use crate::{
    common::ring_arithmetic::{Representation, RingElement},
    protocol::sumcheck::{
        hypercube_point::HypercubePoint,
        polynomial::{add_poly_in_place, Polynomial},
    },
};

pub trait SumcheckBaseData: HighOrderSumcheckData {
    fn partial_evaluate(&mut self, value: &RingElement);
    fn final_evaluations(&self) -> &RingElement;
}

pub trait HighOrderSumcheckData {
    fn nof_polynomial_coefficients(&self) -> usize;
    fn variable_count(&self) -> usize;
    // this is the univariate polynomial for the current variable with the other variables summed out
    // i.e. let a = f(x_0, x_1, ..., x_{n-1}) then this function returns g(x) = sum_{x_1, ..., x_{n-1}} f(x, x_1, ..., x_{n-1})
    fn univariate_polynomial_into(&self, polynomial: &mut Polynomial) {
        // TODO: optimize this to avoid allocating a temp polynomial each time
        let mut temp = Polynomial::new(2, Representation::IncompleteNTT);

        polynomial.set_zero();
        polynomial.nof_coefficients = self.nof_polynomial_coefficients();

        let len = 1 << self.variable_count();
        let half = len / 2;

        for i in 0..half {
            self.univariate_polynomial_at_point_into(HypercubePoint::new(i), &mut temp);
            add_poly_in_place(polynomial, &temp);
        }
    }

    // this is similar to univariate_polynomial_into but evaluates the polynomial at a given point.
    // We ruturn `false` is the polynomial is identically zero (for efficiency in some cases)
    fn univariate_polynomial_at_point_into(
        &self,
        point: HypercubePoint, // this is just the usize so we pass it by value
        polynomial: &mut Polynomial,
    ) -> bool;
}
