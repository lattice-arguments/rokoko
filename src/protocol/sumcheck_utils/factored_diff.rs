use std::{cell::RefCell, cmp::max};

use crate::{
    common::{ring_arithmetic::RingElement, sumcheck_element::SumcheckElement},
    protocol::sumcheck_utils::{
        common::HighOrderSumcheckData,
        elephant_cell::ElephantCell,
        hypercube_point::HypercubePoint,
        polynomial::{mul_poly_into, sub_poly_in_place, Polynomial},
    },
};

/// Difference of two products that share a common factor.
///
/// Encodes a constraint of the form
///
/// ```text
///   (lhs_rest · shared) - (rhs_rest · shared) = 0
/// ```
///
/// where `shared` is a multilinear extension that appears as a factor on both
/// sides — typically the combined witness `w`. Pointwise distributivity
///
/// ```text
///   A(p, X)·w(p, X) - B(p, X)·w(p, X) = (A(p, X) - B(p, X))·w(p, X)
/// ```
///
/// applied inside the half-hypercube loop saves exactly one polynomial
/// multiplication per point relative to wrapping a [`super::diff::DiffSumcheck`]
/// around two [`super::product::ProductSumcheck`] trees that each pre-multiply
/// `shared` into LHS and RHS. See
/// `notes/rokoko_notes/notes/factored_diffsumcheck.tex` for the algebraic
/// derivation and end-to-end magnitude estimate.
pub struct FactoredDiffSumcheck<E: SumcheckElement = RingElement> {
    pub shared: ElephantCell<dyn HighOrderSumcheckData<Element = E>>,
    pub lhs_rest: ElephantCell<dyn HighOrderSumcheckData<Element = E>>,
    pub rhs_rest: ElephantCell<dyn HighOrderSumcheckData<Element = E>>,

    /// Holds `lhs_rest(p, X) - rhs_rest(p, X)` between the difference and the
    /// final multiplication by `shared`.
    diff_rest_poly: RefCell<Polynomial<E>>,
    /// Scratch for `rhs_rest(p, X)` when feeding the subtraction.
    rhs_rest_poly: RefCell<Polynomial<E>>,
    /// Scratch for `shared(p, X)` before multiplying into the result.
    shared_poly: RefCell<Polynomial<E>>,
    /// Required by `HighOrderSumcheckData::get_scratch_poly`.
    scratch_poly: RefCell<Polynomial<E>>,
}

impl<E: SumcheckElement> FactoredDiffSumcheck<E> {
    pub fn new(
        shared: ElephantCell<dyn HighOrderSumcheckData<Element = E>>,
        lhs_rest: ElephantCell<dyn HighOrderSumcheckData<Element = E>>,
        rhs_rest: ElephantCell<dyn HighOrderSumcheckData<Element = E>>,
    ) -> FactoredDiffSumcheck<E> {
        let vc = shared.get_ref().variable_count();
        debug_assert_eq!(
            vc,
            lhs_rest.get_ref().variable_count(),
            "FactoredDiffSumcheck: shared and lhs_rest must share variable_count"
        );
        debug_assert_eq!(
            vc,
            rhs_rest.get_ref().variable_count(),
            "FactoredDiffSumcheck: shared and rhs_rest must share variable_count"
        );
        FactoredDiffSumcheck {
            shared,
            lhs_rest,
            rhs_rest,
            diff_rest_poly: RefCell::new(Polynomial::new(0)),
            rhs_rest_poly: RefCell::new(Polynomial::new(0)),
            shared_poly: RefCell::new(Polynomial::new(0)),
            scratch_poly: RefCell::new(Polynomial::new(0)),
        }
    }
}

impl<E: SumcheckElement> HighOrderSumcheckData for FactoredDiffSumcheck<E> {
    type Element = E;

    fn gadget_span(&self) -> tracing::Span {
        tracing::info_span!("sumcheck::gadget::factored_diff")
    }

    fn get_scratch_poly(&self) -> &RefCell<Polynomial<E>> {
        &self.scratch_poly
    }

    fn variable_count(&self) -> usize {
        self.shared.get_ref().variable_count()
    }

    fn max_num_polynomial_coefficients(&self) -> usize {
        // Result polynomial = (lhs_rest - rhs_rest) · shared.
        // Multiplying length-a by length-b gives length a + b - 1.
        let rest_max = max(
            self.lhs_rest.get_ref().max_num_polynomial_coefficients(),
            self.rhs_rest.get_ref().max_num_polynomial_coefficients(),
        );
        let shared_max = self.shared.get_ref().max_num_polynomial_coefficients();
        rest_max + shared_max - 1
    }

    fn is_univariate_polynomial_zero_at_point(&self, point: HypercubePoint) -> bool {
        // (lhs_rest - rhs_rest) · shared is zero at point if shared is zero,
        // or if both lhs_rest and rhs_rest are zero (so their difference is).
        // Conservative: we don't try to detect lhs == rhs without computing.
        if self
            .shared
            .get_ref()
            .is_univariate_polynomial_zero_at_point(point)
        {
            return true;
        }
        self.lhs_rest
            .get_ref()
            .is_univariate_polynomial_zero_at_point(point)
            && self
                .rhs_rest
                .get_ref()
                .is_univariate_polynomial_zero_at_point(point)
    }

    fn univariate_polynomial_at_point_into(
        &self,
        point: HypercubePoint,
        polynomial: &mut Polynomial<E>,
    ) {
        polynomial.set_zero();

        // Skip the per-point work entirely when shared is zero.
        if self
            .shared
            .get_ref()
            .is_univariate_polynomial_zero_at_point(point)
        {
            return;
        }

        let lhs_zero = self
            .lhs_rest
            .get_ref()
            .is_univariate_polynomial_zero_at_point(point);
        let rhs_zero = self
            .rhs_rest
            .get_ref()
            .is_univariate_polynomial_zero_at_point(point);
        if lhs_zero && rhs_zero {
            return;
        }

        let mut diff_rest_poly = self.diff_rest_poly.borrow_mut();
        let mut rhs_rest_poly = self.rhs_rest_poly.borrow_mut();
        diff_rest_poly.set_zero();
        rhs_rest_poly.set_zero();

        // diff_rest_poly = lhs_rest(p, X) - rhs_rest(p, X).
        if !lhs_zero {
            self.lhs_rest
                .get_ref()
                .univariate_polynomial_at_point_into(point, &mut diff_rest_poly);
        }
        if !rhs_zero {
            self.rhs_rest
                .get_ref()
                .univariate_polynomial_at_point_into(point, &mut rhs_rest_poly);
            sub_poly_in_place(&mut diff_rest_poly, &rhs_rest_poly);
        }

        // shared(p, X).
        let mut shared_poly = self.shared_poly.borrow_mut();
        self.shared
            .get_ref()
            .univariate_polynomial_at_point_into(point, &mut shared_poly);

        // (lhs_rest - rhs_rest) · shared.
        mul_poly_into(polynomial, &diff_rest_poly, &shared_poly);
    }

    fn final_evaluations_test_only(&self) -> Self::Element {
        let lhs = self.lhs_rest.get_ref().final_evaluations_test_only();
        let rhs = self.rhs_rest.get_ref().final_evaluations_test_only();
        let shared = self.shared.get_ref().final_evaluations_test_only();
        let mut diff = lhs;
        diff -= &rhs;
        diff *= &shared;
        diff
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::ring_arithmetic::Representation;
    use crate::protocol::sumcheck_utils::{
        common::SumcheckBaseData, diff::DiffSumcheck, linear::LinearSumcheck,
        product::ProductSumcheck,
    };

    /// The factored gadget produces the same round polynomial as wrapping a
    /// `DiffSumcheck` around two pre-multiplied `ProductSumcheck` trees that
    /// each include the shared factor. Verifies the algebraic identity at
    /// the round-polynomial level over multiple folding rounds.
    #[test]
    fn factored_matches_unfactored() {
        let repr = Representation::IncompleteNTT;
        let n: usize = 8; // 3 variables; fold twice to test rounds 1, 2, and 3.

        let w_data: Vec<RingElement> = (0..n)
            .map(|i| RingElement::constant((3 * i + 7) as u64, repr))
            .collect();
        let a_data: Vec<RingElement> = (0..n)
            .map(|i| RingElement::constant((5 * i + 11) as u64, repr))
            .collect();
        let b_data: Vec<RingElement> = (0..n)
            .map(|i| RingElement::constant((7 * i + 2) as u64, repr))
            .collect();

        let make_lin = |data: &[RingElement]| {
            let cell = ElephantCell::new(LinearSumcheck::<RingElement>::new(data.len()));
            cell.borrow_mut().load_from(data);
            cell
        };

        // Unfactored: DiffSumcheck(A·w, B·w).
        let w_u = make_lin(&w_data);
        let a_u = make_lin(&a_data);
        let b_u = make_lin(&b_data);
        let aw = ElephantCell::new(ProductSumcheck::new(a_u.clone(), w_u.clone()));
        let bw = ElephantCell::new(ProductSumcheck::new(b_u.clone(), w_u.clone()));
        let unfactored = DiffSumcheck::new(aw, bw);

        // Factored: FactoredDiffSumcheck(w, A, B).
        let w_f = make_lin(&w_data);
        let a_f = make_lin(&a_data);
        let b_f = make_lin(&b_data);
        let factored = FactoredDiffSumcheck::new(w_f.clone(), a_f.clone(), b_f.clone());

        let challenges = [
            RingElement::constant(13, repr),
            RingElement::constant(29, repr),
        ];

        let mut poly_u = Polynomial::new(0);
        let mut poly_f = Polynomial::new(0);

        // Round 1: full hypercube (3 variables).
        unfactored.univariate_polynomial_into(&mut poly_u);
        factored.univariate_polynomial_into(&mut poly_f);
        assert_eq!(poly_u.at_zero(), poly_f.at_zero(), "round 1 X=0 mismatch");
        assert_eq!(poly_u.at_one(), poly_f.at_one(), "round 1 X=1 mismatch");

        // Fold once and check round 2.
        for cell in [&w_u, &a_u, &b_u, &w_f, &a_f, &b_f] {
            cell.borrow_mut().partial_evaluate(&challenges[0]);
        }
        unfactored.univariate_polynomial_into(&mut poly_u);
        factored.univariate_polynomial_into(&mut poly_f);
        assert_eq!(poly_u.at_zero(), poly_f.at_zero(), "round 2 X=0 mismatch");
        assert_eq!(poly_u.at_one(), poly_f.at_one(), "round 2 X=1 mismatch");

        // Fold once more and check round 3 (last round before vc = 0).
        for cell in [&w_u, &a_u, &b_u, &w_f, &a_f, &b_f] {
            cell.borrow_mut().partial_evaluate(&challenges[1]);
        }
        unfactored.univariate_polynomial_into(&mut poly_u);
        factored.univariate_polynomial_into(&mut poly_f);
        assert_eq!(poly_u.at_zero(), poly_f.at_zero(), "round 3 X=0 mismatch");
        assert_eq!(poly_u.at_one(), poly_f.at_one(), "round 3 X=1 mismatch");
    }
}
