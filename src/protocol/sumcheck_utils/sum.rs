use crate::{
    common::{
        ring_arithmetic::{Representation, RingElement},
        sumcheck_element::SumcheckElement,
    },
    protocol::sumcheck_utils::{
        common::{EvaluationSumcheckData, HighOrderSumcheckData},
        elephant_cell::ElephantCell,
        hypercube_point::HypercubePoint,
        polynomial::{add_poly_in_place, Polynomial},
    },
};
use std::{cell::RefCell, cmp::max};

#[cfg(test)]
use crate::protocol::sumcheck_utils::{
    common::SumcheckBaseData, linear::LinearSumcheck, selector_eq::SelectorEq,
};

/// Sumcheck data that represents the sum of two other sumchecks.
pub struct SumSumcheck<E: SumcheckElement = RingElement> {
    pub lhs_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = E>>,
    pub rhs_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = E>>,

    lhs_eval_poly: RefCell<Polynomial<E>>,
    rhs_eval_poly: RefCell<Polynomial<E>>,
    scratch_poly: RefCell<Polynomial<E>>,

    const_cache: RefCell<E>,
}

impl<E: SumcheckElement> SumSumcheck<E> {
    pub fn new(
        lhs_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = E>>,
        rhs_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = E>>,
    ) -> SumSumcheck<E> {
        debug_assert_eq!(
            lhs_sumcheck.get_ref().variable_count(),
            rhs_sumcheck.get_ref().variable_count(),
            "Sum sumcheck: both sumchecks must have the same variable count"
        );

        SumSumcheck {
            lhs_sumcheck,
            rhs_sumcheck,
            lhs_eval_poly: RefCell::new(Polynomial::new(0)),
            rhs_eval_poly: RefCell::new(Polynomial::new(0)),
            scratch_poly: RefCell::new(Polynomial::new(0)),
            const_cache: RefCell::new(E::zero()),
        }
    }
}

impl<E: SumcheckElement> HighOrderSumcheckData for SumSumcheck<E> {
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

        // Both constant and both non-zero → sum
        if let (Some(lc), Some(rc)) = (lhs_const, rhs_const) {
            let cache = unsafe { &mut *self.const_cache.as_ptr() };
            cache.set_from(lc);
            *cache += rc;
            return Some(cache);
        }

        // One constant and other is zero → propagate the constant
        if let Some(lc) = lhs_const {
            if self
                .rhs_sumcheck
                .get_ref()
                .is_univariate_polynomial_zero_at_point(point)
            {
                return Some(lc);
            }
        }
        if let Some(rc) = rhs_const {
            if self
                .lhs_sumcheck
                .get_ref()
                .is_univariate_polynomial_zero_at_point(point)
            {
                return Some(rc);
            }
        }
        None
    }

    #[inline]
    /// Override: Σ_p (LHS(p) + RHS(p)) = Σ_p LHS(p) + Σ_p RHS(p).
    ///
    /// Avoids the per-point constant-availability tree traversal and lets
    /// each child handle its own zero-skipping internally.
    fn univariate_polynomial_into(&self, polynomial: &mut Polynomial<Self::Element>) {
        self.lhs_sumcheck
            .get_ref()
            .univariate_polynomial_into(polynomial);

        let mut rhs_poly = self.rhs_eval_poly.borrow_mut();
        rhs_poly.set_zero();
        self.rhs_sumcheck
            .get_ref()
            .univariate_polynomial_into(&mut rhs_poly);
        add_poly_in_place(polynomial, &rhs_poly);
    }

    fn univariate_polynomial_at_point_into(
        &self,
        point: HypercubePoint,
        polynomial: &mut Polynomial<E>,
    ) {
        // Compute the per-round polynomial as the sum of the two inputs.
        polynomial.set_zero();

        let mut lhs_eval_poly = self.lhs_eval_poly.borrow_mut();
        let lhs_sumcheck = &self.lhs_sumcheck;
        if !lhs_sumcheck
            .get_ref()
            .is_univariate_polynomial_zero_at_point(point)
        {
            lhs_sumcheck
                .get_ref()
                .univariate_polynomial_at_point_into(point, &mut lhs_eval_poly);
            add_poly_in_place(polynomial, &lhs_eval_poly);
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
            add_poly_in_place(polynomial, &rhs_eval_poly);
        }
    }

    fn final_evaluations_test_only(&self) -> Self::Element {
        let mut result = self
            .lhs_sumcheck
            .get_ref()
            .final_evaluations_test_only()
            .clone();
        result += &self.rhs_sumcheck.get_ref().final_evaluations_test_only();
        result
    }
}

/// Evaluation-only version of SumSumcheck that evaluates the sum of two sumchecks at a point.
pub struct SumSumcheckEvaluation {
    lhs_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
    rhs_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
    result: RingElement,
}

impl SumSumcheckEvaluation {
    pub fn new(
        lhs_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
        rhs_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
    ) -> Self {
        SumSumcheckEvaluation {
            lhs_evaluation,
            rhs_evaluation,
            result: RingElement::zero(Representation::IncompleteNTT),
        }
    }
}

impl EvaluationSumcheckData for SumSumcheckEvaluation {
    type Element = RingElement;

    fn evaluate(&mut self, point: &Vec<Self::Element>) -> &Self::Element {
        self.result += (
            self.lhs_evaluation.borrow_mut().evaluate(&point),
            self.rhs_evaluation.borrow_mut().evaluate(&point),
        );
        &self.result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_sumcheck_basic() {
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

        let sum_sumcheck = SumSumcheck::new(sumcheck_0.clone(), sumcheck_1.clone());

        let mut poly = Polynomial::new(0);
        sum_sumcheck.univariate_polynomial_into(&mut poly);

        // Sum(data_0) + Sum(data_1) = 26 + 10 = 36
        let claim = RingElement::constant(36, Representation::IncompleteNTT);
        debug_assert_eq!(&poly.at_zero() + &poly.at_one(), claim);

        // Check evaluation at a random point stays consistent.
        let r0 = RingElement::constant(5, Representation::IncompleteNTT);
        let claim_r0 = poly.at(&r0);

        sumcheck_0.borrow_mut().partial_evaluate(&r0);
        sumcheck_1.borrow_mut().partial_evaluate(&r0);

        sum_sumcheck.univariate_polynomial_into(&mut poly);

        debug_assert_eq!(&poly.at_zero() + &poly.at_one(), claim_r0);
    }

    #[test]
    fn sum_with_eqs() {
        let lhs = ElephantCell::new(SelectorEq::<RingElement>::new(0b101, 3, 5));
        let rhs = ElephantCell::new(SelectorEq::<RingElement>::new(0b011, 3, 5));

        let sum = SumSumcheck::new(lhs.clone(), rhs.clone());
        let claim = RingElement::constant(8, Representation::IncompleteNTT);

        // Initial claim: each selector is 1 at 4 points, so their sum is 8.

        let mut poly = Polynomial::new(0);
        sum.univariate_polynomial_into(&mut poly);

        debug_assert_eq!(&poly.at_zero() + &poly.at_one(), claim);
    }

    #[test]
    fn test_sum_evaluation() {
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

        let mut sum_eval = SumSumcheckEvaluation::new(lhs_eval, rhs_eval);

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
            sumcheck_0.get_ref().final_evaluations() + sumcheck_1.get_ref().final_evaluations();

        debug_assert_eq!(sum_eval.evaluate(&point), &expected);
    }
}
