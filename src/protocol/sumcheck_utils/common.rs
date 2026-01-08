use std::cell::RefCell;

use crate::common::sumcheck_element::SumcheckElement;
use crate::protocol::sumcheck_utils::{
    hypercube_point::HypercubePoint,
    polynomial::{add_poly_in_place, Polynomial},
};

/// Marker trait for data that can be consumed by the sumcheck protocol.
/// Implementors must also provide the higher-order hooks so they can be
/// composed with other sumchecks (products, differences, etc.).
pub trait HighOrderSumcheckData {
    type Element: SumcheckElement;
    /// Degree + 1 of the univariate polynomial produced at each round.
    fn max_num_polynomial_coefficients(&self) -> usize;
    fn variable_count(&self) -> usize;
    /// Mutable scratch polynomial to avoid allocations between rounds.
    fn get_scratch_poly(&self) -> &RefCell<Polynomial<Self::Element>>;
    // this is the univariate polynomial for the current variable with the other variables summed out
    // i.e. let a = f(x_0, x_1, ..., x_{n-1}) then this function returns g(x) = sum_{x_1, ..., x_{n-1}} f(x, x_1, ..., x_{n-1})
    fn univariate_polynomial_into(&self, polynomial: &mut Polynomial<Self::Element>) {
        let temp = self.get_scratch_poly();

        polynomial.set_zero();
        polynomial.num_coefficients = 0; // will be updated as we add terms

        let hypercube_size = 1 << self.variable_count();
        let half_hypercube = hypercube_size / 2;

        // Enumerate over the first half of the hypercube; the polynomial at the
        // corresponding point in the second half is handled by the callee.
        for i in 0..half_hypercube {
            if self.is_univariate_polynomial_zero_at_point(HypercubePoint::new(i)) {
                continue;
            }
            self.univariate_polynomial_at_point_into(
                HypercubePoint::new(i),
                &mut temp.borrow_mut(),
            );
            add_poly_in_place(polynomial, &temp.borrow());
        }
    }

    // this is similar to univariate_polynomial_into but evaluates the polynomial at a given point.
    fn univariate_polynomial_at_point_into(
        &self,
        point: HypercubePoint, // this is just the usize so we pass it by value
        polynomial: &mut Polynomial<Self::Element>,
    );

    fn is_univariate_polynomial_zero_at_point(&self, point: HypercubePoint) -> bool;
}

pub trait SumcheckBaseData: HighOrderSumcheckData {
    /// Fold the multilinear extension along the current challenge `value`.
    fn partial_evaluate(&mut self, value: &Self::Element);
    /// Final aggregated evaluations after all variables have been folded.
    fn final_evaluations(&self) -> &Self::Element;
}
