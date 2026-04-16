use crate::{
    common::{
        config::HALF_DEGREE,
        ring_arithmetic::{QuadraticExtension, RingElement},
    },
    protocol::config::IntermediateConfig,
};

use super::context::IntermediateSumcheckContext;

pub fn load_intermediate_sumcheck_data(
    sumcheck_context: &mut IntermediateSumcheckContext,
    config: &IntermediateConfig,
    combined_witness: &[RingElement],
    combination: &[RingElement],
    qe: &[QuadraticExtension; HALF_DEGREE],
) {
    let expected_witness_len = config.witness_height * config.witness_decomposition_chunks;
    assert_eq!(
        combined_witness.len(),
        expected_witness_len,
        "Intermediate witness length mismatch, expected {}, got {}",
        expected_witness_len,
        combined_witness.len()
    );

    sumcheck_context
        .witness_sumcheck
        .borrow_mut()
        .load_from(combined_witness);

    sumcheck_context
        .combiner
        .borrow_mut()
        .load_challenges_from(combination);

    sumcheck_context
        .field_combiner
        .borrow_mut()
        .load_challenges_from(*qe);
}
