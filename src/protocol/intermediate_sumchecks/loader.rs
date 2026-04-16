use crate::{
    common::{
        config::HALF_DEGREE,
        ring_arithmetic::{QuadraticExtension, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    protocol::config::IntermediateConfig,
};

use super::context::IntermediateSumcheckContext;

pub fn load_intermediate_sumcheck_data(
    sumcheck_context: &mut IntermediateSumcheckContext,
    config: &IntermediateConfig,
    combined_witness: &[RingElement],
    conjugated_combined_witness: &[RingElement],
    evaluation_points_inner: &[StructuredRow],
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
    assert_eq!(
        conjugated_combined_witness.len(),
        expected_witness_len,
        "Intermediate conjugated witness length mismatch, expected {}, got {}",
        expected_witness_len,
        conjugated_combined_witness.len()
    );

    sumcheck_context
        .witness_sumcheck
        .borrow_mut()
        .load_from(combined_witness);
    sumcheck_context
        .type5sumcheck
        .conjugated_witness_sumcheck
        .borrow_mut()
        .load_from(conjugated_combined_witness);

    for (type1_sc, eval_point) in sumcheck_context
        .type1sumchecks
        .iter()
        .zip(evaluation_points_inner.iter())
    {
        type1_sc
            .inner_evaluation_sumcheck
            .borrow_mut()
            .load_from(&PreprocessedRow::from_structured_row(eval_point).preprocessed_row);
    }

    sumcheck_context
        .combiner
        .borrow_mut()
        .load_challenges_from(combination);

    sumcheck_context
        .field_combiner
        .borrow_mut()
        .load_challenges_from(*qe);
}
