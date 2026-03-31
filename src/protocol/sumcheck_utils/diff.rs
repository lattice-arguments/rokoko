use std::{cell::RefCell, cmp::max};

use crate::{
    common::{
        ring_arithmetic::{Representation, RingElement},
        sumcheck_element::SumcheckElement,
    },
    protocol::sumcheck_utils::{
        common::{EvaluationSumcheckData, HighOrderSumcheckData},
        elephant_cell::ElephantCell,
        hypercube_point::HypercubePoint,
        polynomial::{sub_poly_in_place, Polynomial},
    },
};

/// Sumcheck data that represents the difference between two other sumchecks.
/// Useful for enforcing equality constraints between two multilinear extensions.
pub struct DiffSumcheck<E: SumcheckElement = RingElement> {
    pub lhs_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = E>>,
    pub rhs_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = E>>,

    rhs_eval_poly: RefCell<Polynomial<E>>,
    scratch_poly: RefCell<Polynomial<E>>,

    // Cached constant difference when both children are constant at a point.
    const_cache: RefCell<E>,
}

impl<E: SumcheckElement> DiffSumcheck<E> {
    pub fn new(
        lhs_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = E>>,
        rhs_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = E>>,
    ) -> DiffSumcheck<E> {
        debug_assert_eq!(
            lhs_sumcheck.get_ref().variable_count(),
            rhs_sumcheck.get_ref().variable_count(),
            "Diff sumcheck: both sumchecks must have the same variable count"
        );

        DiffSumcheck {
            lhs_sumcheck,
            rhs_sumcheck,
            rhs_eval_poly: RefCell::new(Polynomial::new(0)),
            scratch_poly: RefCell::new(Polynomial::new(0)),
            const_cache: RefCell::new(E::zero()),
        }
    }
}

impl<E: SumcheckElement> HighOrderSumcheckData for DiffSumcheck<E> {
    type Element = E;

    fn get_scratch_poly(&self) -> &RefCell<Polynomial<E>> {
        &self.scratch_poly
    }
    fn max_num_polynomial_coefficients(&self) -> usize {
        max(
            self.lhs_sumcheck
                .get_ref()
                .max_num_polynomial_coefficients(),
            self.rhs_sumcheck
                .get_ref()
                .max_num_polynomial_coefficients(),
        )
    }
    fn variable_count(&self) -> usize {
        self.lhs_sumcheck.get_ref().variable_count()
    }

    #[inline]
    fn is_univariate_polynomial_zero_at_point(&self, point: HypercubePoint) -> bool {
        self.lhs_sumcheck
            .get_ref()
            .is_univariate_polynomial_zero_at_point(point)
            && self
                .rhs_sumcheck
                .get_ref()
                .is_univariate_polynomial_zero_at_point(point)
    }

    #[inline]
    fn constant_univariate_polynomial_at_point_available_by_ref(
        &self,
        point: HypercubePoint,
    ) -> Option<&Self::Element> {
        let lhs_const = self
            .lhs_sumcheck
            .get_ref()
            .constant_univariate_polynomial_at_point_available_by_ref(point);
        let rhs_const = self
            .rhs_sumcheck
            .get_ref()
            .constant_univariate_polynomial_at_point_available_by_ref(point);
        match (lhs_const, rhs_const) {
            (Some(lc), Some(rc)) => {
                let cache = unsafe { &mut *self.const_cache.as_ptr() };
                cache.set_from(lc);
                *cache -= rc;
                Some(cache)
            }
            // If only one side is constant and the other is zero, we can still
            // propagate constants.
            (Some(lc), None) => {
                if self
                    .rhs_sumcheck
                    .get_ref()
                    .is_univariate_polynomial_zero_at_point(point)
                {
                    Some(lc)
                } else {
                    None
                }
            }
            (None, Some(_rc)) => {
                // LHS is not constant; if it's zero, result is -rc, but we'd need
                // to negate, which requires cache storage. Skip for now.
                None
            }
            (None, None) => None,
        }
    }

    /// Override: Σ_p (LHS(p) - RHS(p)) = Σ_p LHS(p) - Σ_p RHS(p).
    ///
    /// The default implementation iterates every half-hypercube point and
    /// performs expensive constant-availability checks that traverse the entire
    /// Product tree only to return `None`.  By delegating directly to each
    /// child's `univariate_polynomial_into`, we let them handle their own
    /// zero-skipping internally, eliminating redundant tree traversals.
    fn univariate_polynomial_into(&self, polynomial: &mut Polynomial<Self::Element>) {
        self.lhs_sumcheck
            .get_ref()
            .univariate_polynomial_into(polynomial);

        let mut rhs_poly = self.rhs_eval_poly.borrow_mut();
        rhs_poly.set_zero();
        self.rhs_sumcheck
            .get_ref()
            .univariate_polynomial_into(&mut rhs_poly);
        sub_poly_in_place(polynomial, &rhs_poly);
    }

    #[inline]
    fn univariate_polynomial_at_point_into(
        &self,
        point: HypercubePoint,
        polynomial: &mut Polynomial<E>,
    ) {
        let lhs_sumcheck = &self.lhs_sumcheck;
        if !lhs_sumcheck
            .get_ref()
            .is_univariate_polynomial_zero_at_point(point)
        {
            lhs_sumcheck
                .get_ref()
                .univariate_polynomial_at_point_into(point, polynomial);
        } else {
            polynomial.set_zero();
        }

        let mut rhs_eval_poly = self.rhs_eval_poly.borrow_mut();
        let rhs_sumcheck = &self.rhs_sumcheck;
        if !rhs_sumcheck
            .get_ref()
            .is_univariate_polynomial_zero_at_point(point)
        {
            rhs_sumcheck
                .get_ref()
                .univariate_polynomial_at_point_into(point, &mut rhs_eval_poly);
            sub_poly_in_place(polynomial, &rhs_eval_poly);
        }
    }

    fn final_evaluations_test_only(&self) -> Self::Element {
        let mut result = self
            .lhs_sumcheck
            .get_ref()
            .final_evaluations_test_only()
            .clone();
        result -= &self.rhs_sumcheck.get_ref().final_evaluations_test_only();
        result
    }
}

/// Evaluation-only version of DiffSumcheck that evaluates the difference of two sumchecks at a point.
pub struct DiffSumcheckEvaluation {
    lhs_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
    rhs_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
    result: RingElement,
}

impl DiffSumcheckEvaluation {
    pub fn new(
        lhs_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
        rhs_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
    ) -> Self {
        DiffSumcheckEvaluation {
            lhs_evaluation,
            rhs_evaluation,
            result: RingElement::zero(Representation::IncompleteNTT),
        }
    }
}

impl EvaluationSumcheckData for DiffSumcheckEvaluation {
    type Element = RingElement;

    fn evaluate(&mut self, point: &Vec<Self::Element>) -> &Self::Element {
        self.result -= (
            self.lhs_evaluation.borrow_mut().evaluate(&point),
            self.rhs_evaluation.borrow_mut().evaluate(&point),
        );
        &self.result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::sumcheck_utils::{
        common::SumcheckBaseData, linear::LinearSumcheck, selector_eq::SelectorEq,
    };

    #[test]
    fn test_diff_sumcheck_basic() {
        let data_0 = vec![
            RingElement::constant(8, Representation::IncompleteNTT),
            RingElement::constant(7, Representation::IncompleteNTT),
            RingElement::constant(6, Representation::IncompleteNTT),
            RingElement::constant(5, Representation::IncompleteNTT),
        ];

        let data_1 = vec![
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(2, Representation::IncompleteNTT),
            RingElement::constant(3, Representation::IncompleteNTT),
            RingElement::constant(4, Representation::IncompleteNTT),
        ];

        let sumcheck_0 = ElephantCell::new(LinearSumcheck::new(data_0.len()));
        let sumcheck_1 = ElephantCell::new(LinearSumcheck::new(data_1.len()));

        sumcheck_0.borrow_mut().load_from(&data_0);
        sumcheck_1.borrow_mut().load_from(&data_1);

        let diff_sumcheck = DiffSumcheck::new(sumcheck_0.clone(), sumcheck_1.clone());

        let mut poly = Polynomial::new(0);
        diff_sumcheck.univariate_polynomial_into(&mut poly);

        // Sum(data_0) - Sum(data_1) = 26 - 10 = 16
        let claim = RingElement::constant(16, Representation::IncompleteNTT);
        debug_assert_eq!(&poly.at_zero() + &poly.at_one(), claim);

        // Check evaluation at a random point stays consistent.
        let r0 = RingElement::constant(5, Representation::IncompleteNTT);
        let claim_r0 = poly.at(&r0);

        sumcheck_0.borrow_mut().partial_evaluate(&r0);
        sumcheck_1.borrow_mut().partial_evaluate(&r0);

        diff_sumcheck.univariate_polynomial_into(&mut poly);

        debug_assert_eq!(&poly.at_zero() + &poly.at_one(), claim_r0);
    }

    #[test]
    fn diff_with_eqs() {
        let lhs = ElephantCell::new(SelectorEq::<RingElement>::new(0b101, 3, 5));
        let rhs = ElephantCell::new(SelectorEq::<RingElement>::new(0b011, 3, 5));

        let diff = DiffSumcheck::new(lhs.clone(), rhs.clone());
        let claim = RingElement::constant(0, Representation::IncompleteNTT);

        // Initial claim: the difference of the two selectors over the full hypercube is zero.
        // Both selectors are 1 at exactly 4 points, so their difference sums to zero.

        let mut poly = Polynomial::new(0);
        diff.univariate_polynomial_into(&mut poly);

        debug_assert_eq!(&poly.at_zero() + &poly.at_one(), claim);
    }

    #[test]
    fn test_diff_evaluation() {
        use crate::protocol::sumcheck_utils::linear::BasicEvaluationLinearSumcheck;

        let lhs_data = vec![
            RingElement::constant(8, Representation::IncompleteNTT),
            RingElement::constant(7, Representation::IncompleteNTT),
            RingElement::constant(6, Representation::IncompleteNTT),
            RingElement::constant(5, Representation::IncompleteNTT),
        ];

        let rhs_data = vec![
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(2, Representation::IncompleteNTT),
            RingElement::constant(3, Representation::IncompleteNTT),
            RingElement::constant(4, Representation::IncompleteNTT),
        ];

        let mut lhs_eval_impl = BasicEvaluationLinearSumcheck::new(lhs_data.len());
        lhs_eval_impl.load_from(&lhs_data);
        let lhs_eval: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>> =
            ElephantCell::new(lhs_eval_impl);

        let mut rhs_eval_impl = BasicEvaluationLinearSumcheck::new(rhs_data.len());
        rhs_eval_impl.load_from(&rhs_data);
        let rhs_eval: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>> =
            ElephantCell::new(rhs_eval_impl);

        let mut diff_eval = DiffSumcheckEvaluation::new(lhs_eval, rhs_eval);

        let point = vec![
            RingElement::constant(5, Representation::IncompleteNTT),
            RingElement::constant(7, Representation::IncompleteNTT),
        ];

        // Create reference using the folding implementation
        let ref_lhs_data = vec![
            RingElement::constant(8, Representation::IncompleteNTT),
            RingElement::constant(7, Representation::IncompleteNTT),
            RingElement::constant(6, Representation::IncompleteNTT),
            RingElement::constant(5, Representation::IncompleteNTT),
        ];
        let ref_rhs_data = vec![
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(2, Representation::IncompleteNTT),
            RingElement::constant(3, Representation::IncompleteNTT),
            RingElement::constant(4, Representation::IncompleteNTT),
        ];

        let sumcheck_0 = ElephantCell::new(LinearSumcheck::new(ref_lhs_data.len()));
        sumcheck_0.borrow_mut().load_from(&ref_lhs_data);
        let sumcheck_1 = ElephantCell::new(LinearSumcheck::new(ref_rhs_data.len()));
        sumcheck_1.borrow_mut().load_from(&ref_rhs_data);

        for r in &point {
            sumcheck_0.borrow_mut().partial_evaluate(r);
            sumcheck_1.borrow_mut().partial_evaluate(r);
        }

        let expected =
            sumcheck_0.get_ref().final_evaluations() - sumcheck_1.get_ref().final_evaluations();

        debug_assert_eq!(diff_eval.evaluate(&point), &expected);
    }
}
