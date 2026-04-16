use crate::common::{
    config::HALF_DEGREE,
    ring_arithmetic::{QuadraticExtension, Representation, RingElement},
    structured_row::StructuredRow,
};
use crate::protocol::project_2::BatchedProjectionChallengesSuccinct;

use super::context_verifier::IntermediateVerifierSumcheckContext;

pub fn load_intermediate_verifier_sumcheck_data(
    verifier_sumcheck_context: &mut IntermediateVerifierSumcheckContext,
    claim_over_witness: &RingElement,
    claim_over_witness_conjugate: &RingElement,
    evaluation_points_inner: &[StructuredRow],
    combination: &[RingElement],
    challenges_batching_projection_1: &[BatchedProjectionChallengesSuccinct; 2],
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

    for (type3_1_eval, challenge) in verifier_sumcheck_context
        .type3_1evaluations
        .iter()
        .zip(challenges_batching_projection_1.iter())
    {
        type3_1_eval.c_0_evaluation.borrow_mut().load_from(StructuredRow {
            tensor_layers: challenge
                .c_0_layers
                .iter()
                .map(|&val| RingElement::constant(val, Representation::IncompleteNTT))
                .collect(),
        });
        type3_1_eval
            .j_batched_evaluation
            .borrow_mut()
            .load_from(&challenge.j_batched);
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
