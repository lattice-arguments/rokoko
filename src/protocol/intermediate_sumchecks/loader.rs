use crate::{
    common::{
        config::{HALF_DEGREE, NOF_BATCHES},
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    protocol::{config::IntermediateConfig, project_fine::BatchedProjectionChallenges},
};

use super::context::IntermediateSumcheckContext;

pub fn load_intermediate_sumcheck_data(
    sumcheck_context: &mut IntermediateSumcheckContext,
    config: &IntermediateConfig,
    combined_witness: &[RingElement],
    conjugated_combined_witness: &[RingElement],
    evaluation_points_inner: &[StructuredRow],
    combination: &[RingElement],
    challenges_batching_projection_1: &[BatchedProjectionChallenges; NOF_BATCHES],
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
        .norm_check_sumcheck
        .conjugated_witness_sumcheck
        .borrow_mut()
        .load_from(conjugated_combined_witness);

    for (inner_eval_fold_sc, eval_point) in sumcheck_context
        .inner_eval_fold_sumchecks
        .iter()
        .zip(evaluation_points_inner.iter())
    {
        inner_eval_fold_sc
            .inner_evaluation_sumcheck
            .borrow_mut()
            .load_from(&PreprocessedRow::from_structured_row(eval_point).preprocessed_row);
    }

    for (challenge, fine_proj_sc) in challenges_batching_projection_1
        .iter()
        .zip(sumcheck_context.fine_proj_sumchecks.iter())
    {
        let c_0_ring: Vec<RingElement> = challenge
            .c_0_values
            .iter()
            .map(|&val| RingElement::constant(val, Representation::IncompleteNTT))
            .collect();

        fine_proj_sc.c_0_sumcheck.borrow_mut().load_from(&c_0_ring);

        fine_proj_sc
            .j_batched_sumcheck
            .borrow_mut()
            .load_from(&challenge.j_batched);
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
