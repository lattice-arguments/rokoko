use std::cell::RefCell;

use crate::{
    common::{
        config::HALF_DEGREE,
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        sumcheck_element::SumcheckElement,
    },
    protocol::sumcheck_utils::{
        common::{EvaluationSumcheckData, HighOrderSumcheckData},
        elephant_cell::ElephantCell,
        polynomial::Polynomial,
    },
};

#[cfg(test)]
use crate::protocol::sumcheck_utils::{common::SumcheckBaseData, linear::LinearSumcheck};

pub struct RingToFieldCombiner {
    sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
    challenge_vec: [QuadraticExtension; HALF_DEGREE],
    temp_poly: RefCell<Polynomial<RingElement>>,
    scratch_poly: RefCell<Polynomial<QuadraticExtension>>,
}

impl RingToFieldCombiner {
    pub fn new(sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>) -> Self {
        Self {
            sumcheck,
            challenge_vec: [QuadraticExtension::zero(); HALF_DEGREE],
            scratch_poly: RefCell::new(Polynomial::new(0)),
            temp_poly: RefCell::new(Polynomial::new(0)),
        }
    }

    pub fn load_challenges_from(&mut self, challenge: [QuadraticExtension; HALF_DEGREE]) {
        self.challenge_vec = challenge;
    }
}

impl HighOrderSumcheckData for RingToFieldCombiner {
    type Element = QuadraticExtension;

    fn max_num_polynomial_coefficients(&self) -> usize {
        self.sumcheck.get_ref().max_num_polynomial_coefficients()
    }

    fn variable_count(&self) -> usize {
        self.sumcheck.get_ref().variable_count()
    }

    fn get_scratch_poly(&self) -> &RefCell<Polynomial<Self::Element>> {
        &self.scratch_poly
    }

    /// Compute the ring-level polynomial first in one pass, then convert only
    /// the resulting coefficients (typically 3–4) to field-level.
    ///
    /// The default impl iterates over all half-hypercube points and for each
    /// point calls `univariate_polynomial_at_point_into` which performs a
    /// field conversion (`from_incomplete_ntt_to_homogenized_field_extensions`
    /// + HALF_DEGREE QE multiplications) per coefficient per point.  With
    /// H=65536 and 3 coefficients that is ~200K expensive conversions.
    ///
    /// Since the field conversion is linear, we can instead:
    ///  1. Ask the ring-level sumcheck to produce its polynomial once.
    ///  2. Convert each coefficient (≤4 of them) to field.
    ///
    /// This cuts conversions from O(H) to O(1).
    fn univariate_polynomial_into(&self, polynomial: &mut Polynomial<Self::Element>) {
        let mut ring_poly = self.temp_poly.borrow_mut();
        ring_poly.set_zero();
        ring_poly.num_coefficients = 0;

        // Compute the full ring-level polynomial (Combiner's output-first loop)
        self.sumcheck
            .get_ref()
            .univariate_polynomial_into(&mut ring_poly);

        // Convert the result to field
        polynomial.set_zero();
        for i in 0..ring_poly.num_coefficients {
            ring_poly.coefficients[i].from_incomplete_ntt_to_homogenized_field_extensions();
            let mut coeff = ring_poly.coefficients[i].split_into_quadratic_extensions();
            for j in 0..HALF_DEGREE {
                coeff[j] *= &self.challenge_vec[j];
                polynomial.coefficients[i] += &coeff[j];
            }
            ring_poly.coefficients[i].representation = Representation::IncompleteNTT;
        }
        polynomial.num_coefficients = ring_poly.num_coefficients;
    }

    fn univariate_polynomial_at_point_into(
        &self,
        point: super::hypercube_point::HypercubePoint, // this is just the usize so we pass it by value
        polynomial: &mut Polynomial<Self::Element>,
    ) {
        let temp = &mut self.temp_poly.borrow_mut();
        self.sumcheck
            .get_ref()
            .univariate_polynomial_at_point_into(point, temp);

        polynomial.set_zero();
        for i in 0..temp.num_coefficients {
            temp.coefficients[i].from_incomplete_ntt_to_homogenized_field_extensions();
            let mut coeff = temp.coefficients[i].split_into_quadratic_extensions();
            for j in 0..HALF_DEGREE {
                coeff[j] *= &self.challenge_vec[j];
                polynomial.coefficients[i] += &coeff[j];
            }
            // this will be zeroed anyway so no need to keep it in the final representation
            temp.coefficients[i].representation = Representation::IncompleteNTT;
        }
        polynomial.num_coefficients = temp.num_coefficients;
    }

    fn is_univariate_polynomial_zero_at_point(
        &self,
        _point: super::hypercube_point::HypercubePoint,
    ) -> bool {
        false
    }
    fn final_evaluations_test_only(&self) -> Self::Element {
        let mut result = QuadraticExtension::zero();
        let final_ring_eval = self.sumcheck.get_ref().final_evaluations_test_only();
        let mut temp = final_ring_eval.clone();
        temp.from_incomplete_ntt_to_homogenized_field_extensions();
        let mut coeff = temp.split_into_quadratic_extensions();

        for j in 0..HALF_DEGREE {
            coeff[j] *= &self.challenge_vec[j];
            result += &coeff[j];
        }
        result
    }
}

/// Evaluation-only version of RingToFieldCombiner that evaluates a ring element sumcheck
/// and combines it into a field extension element.
/// Note: This takes RingElement points but implements EvaluationSumcheckData<Element=QuadraticExtension>
/// because it converts the ring evaluation to field extensions.
#[allow(dead_code)]
pub struct RingToFieldCombinerEvaluation {
    evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
    challenge_vec: [QuadraticExtension; HALF_DEGREE],
    result: QuadraticExtension,
    // Store the point converted to QuadraticExtension for trait compatibility
    qe_point: Vec<QuadraticExtension>,
}

impl RingToFieldCombinerEvaluation {
    pub fn new(
        evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
    ) -> Self {
        RingToFieldCombinerEvaluation {
            evaluation,
            challenge_vec: [QuadraticExtension::zero(); HALF_DEGREE],
            result: QuadraticExtension::zero(),
            qe_point: Vec::new(),
        }
    }

    pub fn load_challenges_from(&mut self, challenge: [QuadraticExtension; HALF_DEGREE]) {
        self.challenge_vec = challenge;
    }

    /// Evaluate at a RingElement point (convenience method)
    #[allow(unused_mut)]
    pub fn evaluate_at_ring_point(&mut self, point: &Vec<RingElement>) -> &QuadraticExtension {
        // Evaluate the inner sumcheck at the given point
        let ring_eval = {
            let mut eval_borrow = self.evaluation.borrow_mut();
            eval_borrow.evaluate(point).clone()
        };

        // Convert to field extensions and combine with challenges
        let mut temp = ring_eval.clone();
        temp.from_incomplete_ntt_to_homogenized_field_extensions();
        let mut coeff = temp.split_into_quadratic_extensions();

        self.result = QuadraticExtension::zero();
        for j in 0..HALF_DEGREE {
            coeff[j] *= &self.challenge_vec[j];
            self.result += &coeff[j];
        }

        &self.result
    }
}

impl EvaluationSumcheckData for RingToFieldCombinerEvaluation {
    type Element = QuadraticExtension;

    fn evaluate(&mut self, point: &Vec<QuadraticExtension>) -> &Self::Element {
        // Convert QuadraticExtension point to RingElement
        let mut ring_point = Vec::with_capacity(point.len());
        for qe in point {
            let mut r = RingElement::constant(0, Representation::HomogenizedFieldExtensions);
            r.combine_from_quadratic_extensions(&[*qe; HALF_DEGREE]);
            r.from_homogenized_field_extensions_to_incomplete_ntt();
            ring_point.push(r);
        }

        self.evaluate_at_ring_point(&ring_point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_to_field_combiner() {
        let data = vec![
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(2, Representation::IncompleteNTT),
            RingElement::constant(3, Representation::IncompleteNTT),
            RingElement::constant(4, Representation::IncompleteNTT),
            RingElement::constant(5, Representation::IncompleteNTT),
            RingElement::constant(6, Representation::IncompleteNTT),
            RingElement::constant(7, Representation::IncompleteNTT),
            RingElement::constant(8, Representation::IncompleteNTT),
        ];

        let sumcheck = ElephantCell::new(LinearSumcheck::<RingElement>::new(data.len()));
        sumcheck.borrow_mut().load_from(&data);

        let mut challenge_qe = vec![];
        for i in 0..HALF_DEGREE {
            challenge_qe.push(QuadraticExtension {
                coeffs: [i as u64 + 1, 0],
            });
        }

        let mut combiner = RingToFieldCombiner::new(sumcheck.clone());

        combiner.load_challenges_from(challenge_qe.try_into().unwrap());

        let claim = (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) * (HALF_DEGREE + 1) * (HALF_DEGREE) / 2;

        let mut poly = Polynomial::<QuadraticExtension>::new(0);

        combiner.univariate_polynomial_into(&mut poly);

        debug_assert_eq!(
            poly.at_zero() + poly.at_one(),
            QuadraticExtension {
                coeffs: [claim as u64, 0],
            }
        );

        let r0qe = QuadraticExtension { coeffs: [7, 3] };

        let mut r0 = RingElement::constant(0, Representation::HomogenizedFieldExtensions);

        r0.combine_from_quadratic_extensions(&[r0qe; HALF_DEGREE]);

        r0.from_homogenized_field_extensions_to_incomplete_ntt();

        let claim_after_r0 = poly.at(&r0qe);

        sumcheck.borrow_mut().partial_evaluate(&r0);
        combiner.univariate_polynomial_into(&mut poly);

        debug_assert_eq!(poly.at_zero() + poly.at_one(), claim_after_r0);

        let r1qe = QuadraticExtension { coeffs: [21, 37] };

        let mut r1 = RingElement::constant(0, Representation::HomogenizedFieldExtensions);

        r1.combine_from_quadratic_extensions(&[r1qe; HALF_DEGREE]);

        r1.from_homogenized_field_extensions_to_incomplete_ntt();

        let claim_after_r1 = poly.at(&r1qe);

        sumcheck.borrow_mut().partial_evaluate(&r1);
        combiner.univariate_polynomial_into(&mut poly);

        debug_assert_eq!(poly.at_zero() + poly.at_one(), claim_after_r1);

        let r2qe = QuadraticExtension { coeffs: [53, 89] };

        let mut r2 = RingElement::constant(0, Representation::HomogenizedFieldExtensions);

        r2.combine_from_quadratic_extensions(&[r2qe; HALF_DEGREE]);

        r2.from_homogenized_field_extensions_to_incomplete_ntt();

        let final_claim = poly.at(&r2qe);

        sumcheck.borrow_mut().partial_evaluate(&r2);

        // this API is bit awkward

        let mut final_qe = sumcheck.borrow().final_evaluations().clone();

        final_qe.from_incomplete_ntt_to_homogenized_field_extensions();

        let mut final_qes = final_qe.split_into_quadratic_extensions();
        let mut final_eval = QuadraticExtension::zero();
        for i in 0..HALF_DEGREE {
            final_qes[i] *= &combiner.challenge_vec[i];
            final_eval += &final_qes[i];
        }
        debug_assert_eq!(final_eval, final_claim);
    }

    #[test]
    fn test_ring_to_field_combiner_evaluation() {
        use crate::protocol::sumcheck_utils::linear::BasicEvaluationLinearSumcheck;

        let data = vec![
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(2, Representation::IncompleteNTT),
            RingElement::constant(3, Representation::IncompleteNTT),
            RingElement::constant(4, Representation::IncompleteNTT),
        ];

        let mut eval_impl = BasicEvaluationLinearSumcheck::new(data.len());
        eval_impl.load_from(&data);
        let eval: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>> =
            ElephantCell::new(eval_impl);

        let mut challenge_qe = [QuadraticExtension::zero(); HALF_DEGREE];
        for i in 0..HALF_DEGREE {
            challenge_qe[i] = QuadraticExtension {
                coeffs: [i as u64 + 1, 0],
            };
        }

        let mut combiner_eval = RingToFieldCombinerEvaluation::new(eval);

        combiner_eval.load_challenges_from(challenge_qe);

        let point = vec![
            RingElement::constant(7, Representation::IncompleteNTT),
            RingElement::constant(11, Representation::IncompleteNTT),
        ];

        // Create reference using the folding implementation
        let sumcheck = ElephantCell::new(LinearSumcheck::<RingElement>::new(data.len()));
        sumcheck.borrow_mut().load_from(&data);

        for r in &point {
            sumcheck.borrow_mut().partial_evaluate(r);
        }

        let mut final_qe = sumcheck.borrow().final_evaluations().clone();
        final_qe.from_incomplete_ntt_to_homogenized_field_extensions();
        let mut final_qes = final_qe.split_into_quadratic_extensions();
        let mut expected = QuadraticExtension::zero();
        for i in 0..HALF_DEGREE {
            final_qes[i] *= &challenge_qe[i];
            expected += &final_qes[i];
        }

        debug_assert_eq!(combiner_eval.evaluate_at_ring_point(&point), &expected);
    }
}
