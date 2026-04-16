use crate::common::{
    config::HALF_DEGREE,
    ring_arithmetic::{QuadraticExtension, RingElement},
};

use super::context_verifier::IntermediateVerifierSumcheckContext;

pub fn load_intermediate_verifier_sumcheck_data(
    verifier_sumcheck_context: &mut IntermediateVerifierSumcheckContext,
    claim_over_witness: &RingElement,
    combination: &[RingElement],
    qe: &[QuadraticExtension; HALF_DEGREE],
) {
    verifier_sumcheck_context
        .witness_evaluation
        .borrow_mut()
        .set_result(claim_over_witness.clone());

    verifier_sumcheck_context
        .combiner_evaluation
        .borrow_mut()
        .load_challenges_from(combination);

    verifier_sumcheck_context
        .field_combiner_evaluation
        .borrow_mut()
        .load_challenges_from(*qe);
}
