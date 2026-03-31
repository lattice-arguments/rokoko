use std::cell::RefCell;

use crate::{
    common::{ring_arithmetic::RingElement, sumcheck_element::SumcheckElement},
    protocol::sumcheck_utils::{
        common::{EvaluationSumcheckData, HighOrderSumcheckData},
        elephant_cell::ElephantCell,
        hypercube_point::HypercubePoint,
        polynomial::{add_poly_in_place, Polynomial},
    },
};

#[cfg(test)]
use crate::{
    common::{config::MOD_Q, ring_arithmetic::Representation, structured_row::PreprocessedRow},
    protocol::{
        open::evaluation_point_to_structured_row,
        sumcheck_utils::{common::SumcheckBaseData, linear::LinearSumcheck},
    },
};

pub struct Combiner<E: SumcheckElement = RingElement> {
    sumchecks: Vec<ElephantCell<dyn HighOrderSumcheckData<Element = E>>>,
    challenges: Vec<E>,
    temp_poly: RefCell<Polynomial<E>>,
    scratch_poly: RefCell<Polynomial<E>>,
    // Scratch space for the output-first loop in univariate_polynomial_into.
    output_poly: RefCell<Polynomial<E>>,
}

impl<E: SumcheckElement> Combiner<E> {
    pub fn new(sumchecks: Vec<ElephantCell<dyn HighOrderSumcheckData<Element = E>>>) -> Self {
        let sumchecks_len = sumchecks.len();
        // debug_assert all variables counts are the same
        let var_count = sumchecks[0].get_ref().variable_count();
        for sumcheck in &sumchecks {
            debug_assert_eq!(sumcheck.get_ref().variable_count(), var_count);
        }
        Self {
            sumchecks,
            challenges: E::allocate_zero_vec(sumchecks_len),
            scratch_poly: RefCell::new(super::polynomial::Polynomial::new(0)),
            temp_poly: RefCell::new(super::polynomial::Polynomial::new(0)),
            output_poly: RefCell::new(super::polynomial::Polynomial::new(0)),
        }
    }

    pub fn load_challenges_from(&mut self, challenges: &[E]) {
        debug_assert_eq!(
            challenges.len(),
            self.sumchecks.len(),
            "Combiner: number of challenges must match number of sumchecks"
        );
        self.challenges.clone_from_slice(challenges);
    }

    pub fn sumchecks_count(&self) -> usize {
        self.sumchecks.len()
    }
}

impl<E: SumcheckElement> HighOrderSumcheckData for Combiner<E> {
    type Element = E;

    fn max_num_polynomial_coefficients(&self) -> usize {
        let mut max_coeffs = 0;
        for sumcheck in &self.sumchecks {
            let sumcheck_borrowed = sumcheck.get_ref();
            let coeffs = sumcheck_borrowed.max_num_polynomial_coefficients();
            if coeffs > max_coeffs {
                max_coeffs = coeffs;
            }
        }
        max_coeffs
    }

    fn variable_count(&self) -> usize {
        self.sumchecks[0].get_ref().variable_count()
    }

    fn get_scratch_poly(&self) -> &RefCell<super::polynomial::Polynomial<E>> {
        &self.scratch_poly
    }

    /// Instead of the default point-first loop (iterate H half-hypercube points,
    /// and for each point iterate all N outputs with an is_zero check), we flip
    /// the loop order: iterate outputs first, let each output compute its own
    /// full univariate polynomial, scale by the batching challenge, accumulate.
    ///
    /// This matters because (a) the point-first path does N*H ~3M vtable
    /// dispatches for is_zero alone (N=48, H=65536), and (b) flipping the order
    /// lets each child's univariate_polynomial_into override fire -- that's
    /// where the batched Karatsuba, algebraic delegation, non_zero_range
    /// skipping etc. live. Without this, the default path only ever calls
    /// univariate_polynomial_at_point_into, so those bulk overrides never fire.
    fn univariate_polynomial_into(&self, polynomial: &mut Polynomial<E>) {
        polynomial.set_zero();
        polynomial.num_coefficients = 0;

        let mut output_poly = self.output_poly.borrow_mut();

        for (sumcheck, challenge) in self.sumchecks.iter().zip(self.challenges.iter()) {
            output_poly.set_zero();
            output_poly.num_coefficients = 0;

            sumcheck
                .get_ref()
                .univariate_polynomial_into(&mut output_poly);

            // Scale by the batching challenge and accumulate
            for j in 0..output_poly.num_coefficients {
                output_poly.coefficients[j] *= challenge;
            }
            add_poly_in_place(polynomial, &output_poly);
        }
    }

    #[inline]
    fn univariate_polynomial_at_point_into(
        &self,
        point: HypercubePoint, // this is just the usize so we pass it by value
        polynomial: &mut Polynomial<E>,
    ) {
        polynomial.set_zero();
        polynomial.num_coefficients = 0; // will be updated as we add terms

        let mut temp_poly = self.temp_poly.borrow_mut();
        let nof_sumchecks = self.sumchecks.len();
        for i in 0..nof_sumchecks {
            let sumcheck = &self.sumchecks[i];

            if sumcheck
                .borrow()
                .is_univariate_polynomial_zero_at_point(point)
            {
                continue;
            }

            let challenge = &self.challenges[i];
            sumcheck
                .borrow()
                .univariate_polynomial_at_point_into(point, &mut temp_poly);
            // multiply temp_poly by challenge
            for j in 0..temp_poly.num_coefficients {
                temp_poly.coefficients[j] *= challenge;
            }
            // add to polynomial
            add_poly_in_place(polynomial, &temp_poly);
        }
    }

    fn is_univariate_polynomial_zero_at_point(
        &self,
        _point: super::hypercube_point::HypercubePoint,
    ) -> bool {
        false
    }

    fn final_evaluations_test_only(&self) -> Self::Element {
        let mut result = E::zero();
        let nof_sumchecks = self.sumchecks.len();
        for i in 0..nof_sumchecks {
            let sumcheck = &self.sumchecks[i];
            let mut term = sumcheck.get_ref().final_evaluations_test_only().clone();
            term *= &self.challenges[i];
            result += &term;
        }
        result
    }
}

/// Evaluation-only version of Combiner that evaluates a linear combination of sumchecks at a point.
pub struct CombinerEvaluation<E: SumcheckElement = RingElement> {
    evaluations: Vec<ElephantCell<dyn EvaluationSumcheckData<Element = E>>>,
    challenges: Vec<E>,
    result: E,
    scratch: E,
}

impl<E: SumcheckElement> CombinerEvaluation<E> {
    pub fn new(evaluations: Vec<ElephantCell<dyn EvaluationSumcheckData<Element = E>>>) -> Self {
        let evaluations_len = evaluations.len();
        CombinerEvaluation {
            evaluations,
            challenges: E::allocate_zero_vec(evaluations_len),
            result: E::zero(),
            scratch: E::zero(),
        }
    }

    pub fn load_challenges_from(&mut self, challenges: &[E]) {
        debug_assert_eq!(
            challenges.len(),
            self.evaluations.len(),
            "CombinerEvaluation: number of challenges must match number of evaluations"
        );
        self.challenges.clone_from_slice(challenges);
    }

    pub fn sumchecks_count(&self) -> usize {
        self.evaluations.len()
    }
}

impl<E: SumcheckElement> EvaluationSumcheckData for CombinerEvaluation<E> {
    type Element = E;

    fn evaluate(&mut self, point: &Vec<Self::Element>) -> &Self::Element {
        // Compute the linear combination: sum of (evaluation[i] * challenge[i])
        self.result.set_zero();

        for i in 0..self.evaluations.len() {
            self.scratch *= (
                self.evaluations[i].borrow_mut().evaluate(&point),
                &self.challenges[i],
            );
            self.result += &self.scratch;
        }

        &self.result
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combiner() {
        let data0 = (0..8)
            .map(|i| RingElement::constant(i + 1 as u64, Representation::IncompleteNTT))
            .collect::<Vec<RingElement>>();

        let data1 = (8..16)
            .map(|i| RingElement::constant((i + 1) as u64, Representation::IncompleteNTT))
            .collect::<Vec<RingElement>>();

        let data2 = (16..24)
            .map(|i| RingElement::constant((i + 1) as u64, Representation::IncompleteNTT))
            .collect::<Vec<RingElement>>();

        let data3 = (24..32)
            .map(|i| RingElement::constant((i + 1) as u64, Representation::IncompleteNTT))
            .collect::<Vec<RingElement>>();

        let combiner = evaluation_point_to_structured_row(&vec![
            RingElement::constant(32, Representation::IncompleteNTT),
            RingElement::constant(33, Representation::IncompleteNTT),
        ]);

        // vector is (1 - 32) * (1 - 33), (1 - 32) * 33, 32 * (1 - 33), 32 * 33

        let preprocessed_challenges = PreprocessedRow::from_structured_row(&combiner);

        let mut sumcheck0 = LinearSumcheck::new(data0.len());

        sumcheck0.load_from(&data0);

        let mut sumcheck1 = LinearSumcheck::new(data1.len());
        sumcheck1.load_from(&data1);

        let mut sumcheck2 = LinearSumcheck::new(data2.len());

        sumcheck2.load_from(&data2);

        let mut sumcheck3 = LinearSumcheck::new(data3.len());
        sumcheck3.load_from(&data3);

        let sumcheck0_ref = ElephantCell::new(sumcheck0);
        let sumcheck1_ref = ElephantCell::new(sumcheck1);
        let sumcheck2_ref = ElephantCell::new(sumcheck2);
        let sumcheck3_ref = ElephantCell::new(sumcheck3);

        let mut combiner = Combiner::new(vec![
            sumcheck0_ref.clone(),
            sumcheck1_ref.clone(),
            sumcheck2_ref.clone(),
            sumcheck3_ref.clone(),
        ]);

        combiner.load_challenges_from(&preprocessed_challenges.preprocessed_row);

        let mut poly = Polynomial::new(0);

        combiner.univariate_polynomial_into(&mut poly);

        // manually compute expected coefficients
        // 1 + 2 + ... + 8 = 36
        // 9 + 10 + ... + 16 = 100
        // 17 + ... + 24 = 164
        // 25 + ... + 32 = 228

        let claim = RingElement::constant(
            (MOD_Q as i64
                + 36 * (1 - 32) * (1 - 33)
                + 100 * (1 - 32) * 33
                + 164 * 32 * (1 - 33)
                + 228 * 32 * 33) as u64,
            Representation::IncompleteNTT,
        );

        debug_assert_eq!(&poly.at_zero() + &poly.at_one(), claim);

        let r0 = RingElement::constant(2, Representation::IncompleteNTT);

        let claim = poly.at(&r0);

        sumcheck0_ref.borrow_mut().partial_evaluate(&r0);
        sumcheck1_ref.borrow_mut().partial_evaluate(&r0);
        sumcheck2_ref.borrow_mut().partial_evaluate(&r0);
        sumcheck3_ref.borrow_mut().partial_evaluate(&r0);

        combiner.univariate_polynomial_into(&mut poly);
        debug_assert_eq!(&poly.at_zero() + &poly.at_one(), claim);

        let r1 = RingElement::constant(3, Representation::IncompleteNTT);
        let claim = poly.at(&r1);

        sumcheck0_ref.borrow_mut().partial_evaluate(&r1);
        sumcheck1_ref.borrow_mut().partial_evaluate(&r1);
        sumcheck2_ref.borrow_mut().partial_evaluate(&r1);
        sumcheck3_ref.borrow_mut().partial_evaluate(&r1);
        combiner.univariate_polynomial_into(&mut poly);
        debug_assert_eq!(&poly.at_zero() + &poly.at_one(), claim);

        let r2 = RingElement::constant(5, Representation::IncompleteNTT);
        let final_claim = poly.at(&r2);
        sumcheck0_ref.borrow_mut().partial_evaluate(&r2);
        sumcheck1_ref.borrow_mut().partial_evaluate(&r2);
        sumcheck2_ref.borrow_mut().partial_evaluate(&r2);
        sumcheck3_ref.borrow_mut().partial_evaluate(&r2);

        let mut final_eval = RingElement::zero(Representation::IncompleteNTT);

        let mut term = sumcheck0_ref.get_ref().final_evaluations().clone();
        term *= &combiner.challenges[0];
        final_eval += &term;

        term = sumcheck1_ref.get_ref().final_evaluations().clone();
        term *= &combiner.challenges[1];
        final_eval += &term;

        term = sumcheck2_ref.get_ref().final_evaluations().clone();
        term *= &combiner.challenges[2];
        final_eval += &term;

        term = sumcheck3_ref.get_ref().final_evaluations().clone();
        term *= &combiner.challenges[3];
        final_eval += &term;

        debug_assert_eq!(final_eval, final_claim);
    }

    #[test]
    fn test_combiner_evaluation() {
        use crate::protocol::sumcheck_utils::linear::BasicEvaluationLinearSumcheck;

        let data0 = vec![
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(2, Representation::IncompleteNTT),
            RingElement::constant(3, Representation::IncompleteNTT),
            RingElement::constant(4, Representation::IncompleteNTT),
        ];

        let data1 = vec![
            RingElement::constant(5, Representation::IncompleteNTT),
            RingElement::constant(6, Representation::IncompleteNTT),
            RingElement::constant(7, Representation::IncompleteNTT),
            RingElement::constant(8, Representation::IncompleteNTT),
        ];

        let mut eval0_impl = BasicEvaluationLinearSumcheck::new(data0.len());
        eval0_impl.load_from(&data0);
        let eval0: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>> =
            ElephantCell::new(eval0_impl);

        let mut eval1_impl = BasicEvaluationLinearSumcheck::new(data1.len());
        eval1_impl.load_from(&data1);
        let eval1: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>> =
            ElephantCell::new(eval1_impl);

        let challenges = vec![
            RingElement::constant(3, Representation::IncompleteNTT),
            RingElement::constant(5, Representation::IncompleteNTT),
        ];

        let mut combiner_eval = CombinerEvaluation::new(vec![eval0, eval1]);
        combiner_eval.load_challenges_from(&challenges);

        let point = vec![
            RingElement::constant(7, Representation::IncompleteNTT),
            RingElement::constant(11, Representation::IncompleteNTT),
        ];

        // Create reference using the folding implementation
        let sumcheck0 = ElephantCell::new(LinearSumcheck::new(data0.len()));
        sumcheck0.borrow_mut().load_from(&data0);
        let sumcheck1 = ElephantCell::new(LinearSumcheck::new(data1.len()));
        sumcheck1.borrow_mut().load_from(&data1);

        for r in &point {
            sumcheck0.borrow_mut().partial_evaluate(r);
            sumcheck1.borrow_mut().partial_evaluate(r);
        }

        let mut expected = sumcheck0.get_ref().final_evaluations().clone();
        expected *= &challenges[0];
        let mut term = sumcheck1.get_ref().final_evaluations().clone();
        term *= &challenges[1];
        expected += &term;

        debug_assert_eq!(combiner_eval.evaluate(&point), &expected);
    }
}
