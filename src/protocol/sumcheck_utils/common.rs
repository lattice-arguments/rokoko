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
        polynomial.num_coefficients = 1; // will be updated as we add terms

        let hypercube_size = 1 << self.variable_count();
        let half_hypercube = hypercube_size / 2;

        // Enumerate over the first half of the hypercube; the polynomial at the
        // corresponding point in the second half is handled by the callee.
        for i in 0..half_hypercube {
            let constant = self
                .constant_univariate_polynomial_at_point_available_by_ref(HypercubePoint::new(i));

            if let Some(constant) = constant {
                polynomial.coefficients[0] += constant;
                continue;
            }

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

    fn claim(&self) -> Self::Element {
        // let mut poly = self.get_scratch_poly().borrow_mut();
        let mut poly = Polynomial::new(0);
        self.univariate_polynomial_into(&mut poly);
        let mut res = poly.at_one();
        res += &poly.at_zero();
        res
    }

    // this is similar to univariate_polynomial_into but evaluates the polynomial at a given point.
    fn univariate_polynomial_at_point_into(
        &self,
        point: HypercubePoint, // this is just the usize so we pass it by value
        polynomial: &mut Polynomial<Self::Element>,
    );

    fn is_univariate_polynomial_zero_at_point(&self, point: HypercubePoint) -> bool;

    fn constant_univariate_polynomial_at_point_available_by_ref(
        &self,
        _: HypercubePoint,
    ) -> Option<&Self::Element> {
        None
    }

    /// Expose the raw data as (low_half, high_half) slices for batched
    /// inner-product computation.  Only `LinearSumcheck` with no prefix/suffix
    /// variables overrides this.
    fn as_data_slices(&self) -> Option<(&[Self::Element], &[Self::Element])> {
        None
    }

    /// Return the contiguous half-hypercube range `[start, end)` outside which
    /// `is_univariate_polynomial_zero_at_point` is guaranteed to return `true`.
    /// Nodes that carry a sparse selector (e.g. `SelectorEq`) override this so
    /// that callers can iterate only the relevant points.
    fn non_zero_range(&self) -> Option<(usize, usize)> {
        None
    }

    /// When `bypass = true`, per-point cache stores in
    /// `univariate_polynomial_at_point_into` are skipped.  Used by
    /// `univariate_polynomial_into` during sweeps where each point is visited
    /// once and the cache would never hit.  Implementations that maintain a
    /// per-point cache should override this and propagate to children.
    fn set_cache_bypass(&self, _bypass: bool) {}

    fn final_evaluations_test_only(&self) -> Self::Element;
}

pub trait SumcheckBaseData: HighOrderSumcheckData {
    /// Fold the multilinear extension along the current challenge `value`.
    fn partial_evaluate(&mut self, value: &Self::Element);
    /// Final aggregated evaluations after all variables have been folded.
    fn final_evaluations(&self) -> &Self::Element;
}

// This trait is different from SumcheckBaseData in that it only requires the ability to evaluate the polynomial at a point.
// This is useful for verifier, who only needs to check evaluations and does not need to perform folding.
pub trait EvaluationSumcheckData {
    // This evaluates the polynomial at the given point and writes the result into `point`.
    type Element: SumcheckElement;
    fn evaluate(&mut self, point: &Vec<Self::Element>) -> &Self::Element;
}
