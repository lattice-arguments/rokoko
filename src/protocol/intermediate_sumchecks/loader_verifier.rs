use crate::common::{
    config::HALF_DEGREE,
    ring_arithmetic::{QuadraticExtension, RingElement},
    structured_row::StructuredRow,
};

use super::context_verifier::IntermediateVerifierSumcheckContext;

pub fn load_intermediate_verifier_sumcheck_data(
    verifier_sumcheck_context: &mut IntermediateVerifierSumcheckContext,
    claim_over_witness: &RingElement,
    claim_over_witness_conjugate: &RingElement,
    evaluation_points_inner: &[StructuredRow],
    combination: &[RingElement],
    qe: &[QuadraticExtension; HALF_DEGREE],
) {
    verifier_sumcheck_context
        .witness_evaluation
        .borrow_mut()
        .set_result(claim_over_witness.clone());
    verifier_sumcheck_context
        .conjugated_witness_evaluation
        .borrow_mut()
        .set_result(claim_over_witness_conjugate.clone());
    verifier_sumcheck_context
        .type5evaluation
        .conjugated_witness_evaluation
        .borrow_mut()
        .set_result(claim_over_witness_conjugate.clone());

    for (type1_eval, point) in verifier_sumcheck_context
        .type1evaluations
        .iter()
        .zip(evaluation_points_inner.iter())
    {
        type1_eval
            .inner_evaluation
            .borrow_mut()
            .load_from(point.clone());
    }

    verifier_sumcheck_context
        .combiner_evaluation
        .borrow_mut()
        .load_challenges_from(combination);

    verifier_sumcheck_context
        .field_combiner_evaluation
        .borrow_mut()
        .load_challenges_from(*qe);
}
