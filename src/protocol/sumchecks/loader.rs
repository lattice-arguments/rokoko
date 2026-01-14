use crate::{
    common::{
        arithmetic::field_to_ring_element_into,
        config::{HALF_DEGREE, NOF_BATCHES},
        matrix::new_vec_zero_preallocated,
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    protocol::{
        config::Config,
        open::Opening,
        project_2::BatchedProjectionChallenges,
        sumchecks::helpers::{projection_flatter_1_times_matrix, split_projection_flatter},
    },
};

use super::context::SumcheckContext;

use super::helpers::{projection_coefficients, tensor_product};

/// Loads all data into the sumcheck context.
///
/// This function encapsulates all the `load_from` calls that populate the sumcheck
/// gadgets with their actual input values. By extracting this logic, we separate
/// data preparation from the main sumcheck execution flow.
///
/// # Arguments
///
/// * `sumcheck_context` - The initialized sumcheck context to load data into
/// * `config` - Protocol configuration
/// * `combined_witness` - The full witness vector
/// * `folding_challenges` - Random weights for folding multiple witnesses
/// * `opening` - Opening proofs with evaluation points
/// * `projection_matrix` - The structured projection matrix
/// * `projection_matrix_flatter_structured` - Structured row for flattening projection
/// * `projection_matrix_flatter_preprocessed` - Preprocessed flattening point for projection
pub fn load_sumcheck_data(
    sumcheck_context: &mut SumcheckContext,
    config: &Config,
    combined_witness: &Vec<RingElement>,
    conjugated_combined_witness: &Vec<RingElement>,
    folding_challenges: &Vec<RingElement>,
    challenges_batching_projection_1: &Option<&[BatchedProjectionChallenges; NOF_BATCHES]>,
    opening: &Opening,
    projection_matrix: &ProjectionMatrix,
    projection_matrix_flatter_structured: &StructuredRow,
    projection_matrix_flatter_preprocessed: &PreprocessedRow,
    combination: &Vec<RingElement>,
    qe: &[QuadraticExtension; HALF_DEGREE],
) {
    // Load combined witness
    sumcheck_context
        .combined_witness_sumcheck
        .borrow_mut()
        .load_from(combined_witness);

    // Load folding challenges
    sumcheck_context
        .folding_challenges_sumcheck
        .borrow_mut()
        .load_from(&folding_challenges);

    sumcheck_context
        .type5sumcheck
        .conjugated_combined_witness
        .borrow_mut()
        .load_from(&conjugated_combined_witness);

    // Load inner evaluation points (type1)
    for (type1_sc, eval_point) in sumcheck_context
        .type1sumchecks
        .iter()
        .zip(opening.evaluation_points_inner.iter())
    {
        type1_sc
            .inner_evaluation_sumcheck
            .borrow_mut()
            .load_from(&eval_point.preprocessed_row);
    }

    // Load outer evaluation points (type2)
    for (type2_sc, eval_point) in sumcheck_context
        .type2sumchecks
        .iter()
        .zip(opening.evaluation_points_outer.iter())
    {
        type2_sc
            .outer_evaluation_sumcheck
            .borrow_mut()
            .load_from(&eval_point.preprocessed_row);
    }

    // Load projection data (type3)
    // LHS: Split into flatter_0 (elder/block variables) and flatter_1·matrix (LS/within-block variables)
    if let Some(type3_sc) = &mut sumcheck_context.type3sumcheck {
        let (projection_flatter_0_structured, projection_flatter_1_structured) =
            split_projection_flatter(
                projection_matrix_flatter_structured,
                projection_matrix.projection_height,
            );

        // Load flatter_0 (block-level weights)
        let projection_flatter_0_preprocessed =
            PreprocessedRow::from_structured_row(&projection_flatter_0_structured);
        type3_sc
            .lhs_flatter_0_sumcheck
            .borrow_mut()
            .load_from(&projection_flatter_0_preprocessed.preprocessed_row);

        // Load flatter_1 · projection_matrix (within-block coefficients)
        let projection_flatter_1_preprocessed =
            PreprocessedRow::from_structured_row(&projection_flatter_1_structured);
        let flatter_1_times_matrix = projection_flatter_1_times_matrix(
            projection_matrix,
            &projection_flatter_1_preprocessed,
        );

        let mut flatter_1_times_matrix_ring =
            new_vec_zero_preallocated(flatter_1_times_matrix.len());

        for i in 0..flatter_1_times_matrix.len() {
            field_to_ring_element_into(
                &mut flatter_1_times_matrix_ring[i],
                &flatter_1_times_matrix[i],
            );
            flatter_1_times_matrix_ring[i].from_homogenized_field_extensions_to_incomplete_ntt();
        }

        type3_sc
            .lhs_flatter_1_times_matrix_sumcheck
            .borrow_mut()
            .load_from(&flatter_1_times_matrix_ring);

        // RHS: Split into fold_challenge and projection_flatter (Product)
        type3_sc
            .rhs_fold_challenge_sumcheck
            .borrow_mut()
            .load_from(folding_challenges);

        type3_sc
            .rhs_projection_flatter_sumcheck
            .borrow_mut()
            .load_from(&projection_matrix_flatter_preprocessed.preprocessed_row);
    }

    // Load type3_1_a_sumchecks if present (batched projections)
    if let Some(type3_1_a_contexts) = &mut sumcheck_context.type3_1_a_sumchecks {
        if let Some(challenges) = challenges_batching_projection_1 {
            // Each batch gets its own (c_0_values, c_1_values, j_batched) tuple
            for (batch_idx, (type3_1_a_ctx, challenges)) in type3_1_a_contexts
                .sumchecks
                .iter_mut()
                .zip(challenges.iter())
                .enumerate()
            {
                // let c_0_values = challenges.c_0_values;
                // let c_1_values = challenges.c_1_values;
                // let j_batched = challenges.j_batched;
                // Lift c_0_values from u64 to RingElement and load into lhs_flatter_0
                let c_0_ring: Vec<RingElement> = challenges
                    .c_0_values
                    .iter()
                    .map(|&val| RingElement::constant(val, Representation::IncompleteNTT))
                    .collect();

                type3_1_a_ctx
                    .lhs_flatter_0_sumcheck
                    .borrow_mut()
                    .load_from(&c_0_ring);

                println!(
                    "len, c_0_values: {}, len j_batched: {}",
                    c_0_ring.len(),
                    challenges.j_batched.len()
                );

                type3_1_a_ctx
                    .lhs_flatter_1_times_matrix_sumcheck
                    .borrow_mut()
                    .load_from(&challenges.j_batched);
            }
            // RHS: fold_challenge (same for all batches, already loaded in folding_challenges_sumcheck)
            type3_1_a_contexts
                .rhs_fold_challenge_sumcheck
                .borrow_mut()
                .load_from(folding_challenges);
        }
    }

    sumcheck_context
        .combiner
        .borrow_mut()
        .load_challenges_from(&combination);

    sumcheck_context
        .field_combiner
        .borrow_mut()
        .load_challenges_from(qe.clone());
}
