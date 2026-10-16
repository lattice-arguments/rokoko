use crate::{
    common::{
        arithmetic::{field_to_ring_element_into, precompute_structured_values_fast},
        config::{DEGREE, NOF_BATCHES},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    protocol::{
        config::SumcheckConfig,
        open::Opening,
        project_fine::BatchedProjectionChallenges,
        snark::{conjugated_oracle, witness_oracle},
        sumcheck_utils::{elephant_cell::ElephantCell, linear::LinearSumcheck},
        sumchecks::helpers::{
            projection_flatter_1_times_matrix, split_projection_flatter, tensor_product_u64,
        },
    },
};

use super::context::SumcheckContext;

pub(crate) struct Borrows<'a> {
    tables: Vec<ElephantCell<LinearSumcheck<RingElement>>>,
    lifetime: std::marker::PhantomData<&'a [RingElement]>,
}

impl Drop for Borrows<'_> {
    fn drop(&mut self) {
        for table in &self.tables {
            table.borrow_mut().release_caller_data();
        }
    }
}

pub(crate) fn borrow_caller_data<'a>(
    sumcheck_context: &SumcheckContext,
    config: &SumcheckConfig,
    combined_witness: &'a [RingElement],
) -> Borrows<'a> {
    let witness = &sumcheck_context.combined_witness_sumcheck;
    let conjugate = &sumcheck_context
        .norm_check_sumcheck
        .conjugated_combined_witness;
    let borrows = Borrows {
        tables: vec![witness.clone(), conjugate.clone()],
        lifetime: std::marker::PhantomData,
    };
    assert_eq!(combined_witness.len(), config.composed_witness_length);
    let non_zero_end = used_length(config, combined_witness);
    *witness.borrow_mut() = witness_oracle(combined_witness, non_zero_end);
    *conjugate.borrow_mut() = conjugated_oracle(combined_witness, non_zero_end);
    borrows
}

pub(crate) fn used_length(config: &SumcheckConfig, combined_witness: &[RingElement]) -> usize {
    let used_columns =
        (config.composed_witness_length as f64 * config.next_level_usage_ratio).ceil() as usize;
    used_columns
        .min(config.composed_witness_length)
        .min(combined_witness.len())
}

/// Loads all data into the sumcheck context.
///
/// This function encapsulates all the `load_from` calls that populate the sumcheck
/// gadgets with their actual input values. By extracting this logic, we separate
/// data preparation from the main sumcheck execution flow.
pub fn load_sumcheck_data(
    sumcheck_context: &mut SumcheckContext,
    folding_challenges: &Vec<RingElement>,
    fine_proj_batching_challenges: &Option<&[BatchedProjectionChallenges; NOF_BATCHES]>,
    opening: &Opening,
    projection_matrix: &ProjectionMatrix,
    projection_matrix_flatter: &Option<(PreprocessedRow, StructuredRow)>,
) {
    // Load folding challenges
    sumcheck_context
        .folding_challenges_sumcheck
        .borrow_mut()
        .load_from(&folding_challenges);

    // Load inner evaluation points (inner_eval_fold)
    for (inner_eval_fold_sc, eval_point) in sumcheck_context
        .inner_eval_fold_sumchecks
        .iter()
        .zip(opening.evaluation_points_inner.iter())
    {
        inner_eval_fold_sc
            .inner_evaluation_sumcheck
            .borrow_mut()
            .load_from(&eval_point.preprocessed_row);
    }

    // Load outer evaluation points (outer_eval_claim)
    for (outer_eval_claim_sc, eval_point) in sumcheck_context
        .outer_eval_claim_sumchecks
        .iter()
        .zip(opening.evaluation_points_outer.iter())
    {
        outer_eval_claim_sc
            .outer_evaluation_sumcheck
            .borrow_mut()
            .load_from(&eval_point.preprocessed_row);
    }

    // Load projection data (coarse_proj)
    // LHS: Split into flatter_0 (elder/block variables) and flatter_1·matrix (LS/within-block variables)
    if let Some(coarse_proj_sc) = &mut sumcheck_context.coarse_proj_sumcheck {
        let (projection_flatter_0_structured, projection_flatter_1_structured) =
            split_projection_flatter(
                &projection_matrix_flatter.as_ref().unwrap().1,
                projection_matrix.projection_height,
            );

        // Load flatter_0 (block-level weights)
        let projection_flatter_0_preprocessed =
            PreprocessedRow::from_structured_row(&projection_flatter_0_structured);
        coarse_proj_sc
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
            vec![RingElement::zero(Representation::IncompleteNTT); flatter_1_times_matrix.len()];

        for i in 0..flatter_1_times_matrix.len() {
            field_to_ring_element_into(
                &mut flatter_1_times_matrix_ring[i],
                &flatter_1_times_matrix[i],
            );
            flatter_1_times_matrix_ring[i].from_homogenized_field_extensions_to_incomplete_ntt();
        }

        coarse_proj_sc
            .lhs_flatter_1_times_matrix_sumcheck
            .borrow_mut()
            .load_from(&flatter_1_times_matrix_ring);

        // RHS: Split into fold_challenge and projection_flatter (Product)
        coarse_proj_sc
            .rhs_fold_challenge_sumcheck
            .borrow_mut()
            .load_from(folding_challenges);

        coarse_proj_sc
            .rhs_projection_flatter_sumcheck
            .borrow_mut()
            .load_from(
                &projection_matrix_flatter
                    .as_ref()
                    .unwrap()
                    .0
                    .preprocessed_row,
            );
    }

    // Load fine_proj_sumchecks if present (batched projections)
    if let Some(fine_proj_contexts) = &mut sumcheck_context.fine_proj_sumchecks {
        if let Some(challenges) = fine_proj_batching_challenges {
            // Each batch gets its own (c_0_values, c_1_values, j_batched) tuple
            for (_batch_idx, (fine_proj_ctx, challenges)) in fine_proj_contexts
                .sumchecks
                .iter_mut()
                .zip(challenges.iter())
                .enumerate()
            {
                // Lift c_0_values from u64 to RingElement and load into lhs_flatter_0
                let c_0_ring: Vec<RingElement> = challenges
                    .c_0_values
                    .iter()
                    .map(|&val| RingElement::constant(val, Representation::IncompleteNTT))
                    .collect();

                fine_proj_ctx
                    .lhs_flatter_0_sumcheck
                    .borrow_mut()
                    .load_from(&c_0_ring);

                fine_proj_ctx
                    .lhs_flatter_1_times_matrix_sumcheck
                    .borrow_mut()
                    .load_from(&challenges.j_batched);

                // consistency

                let (e_0_values, e_1_values) = {
                    let mut e_0_layers = Vec::new();
                    let mut e_1_layers = Vec::new();
                    for (i, &layer) in challenges.c_1_layers.iter().enumerate() {
                        if i < challenges.c_1_layers.len() - DEGREE.ilog2() as usize {
                            e_0_layers.push(layer);
                        } else {
                            e_1_layers.push(layer);
                        }
                    }
                    (
                        precompute_structured_values_fast(&e_0_layers),
                        precompute_structured_values_fast(&e_1_layers),
                    )
                };

                let lhs_multipier_ring = challenges
                    .c_2_values
                    .iter()
                    .map(|&x| RingElement::constant(x, Representation::IncompleteNTT))
                    .collect::<Vec<RingElement>>();

                let rhs_multipier_ring: Vec<RingElement> = {
                    // c_2 \otimes c_0 \otimes e_0
                    // first over u64
                    let values_0 =
                        tensor_product_u64(&challenges.c_2_values, &challenges.c_0_values);
                    let values_1 = tensor_product_u64(&values_0, &e_0_values);
                    let vals_over_ring = values_1
                        .iter()
                        .map(|&x| RingElement::constant(x, Representation::IncompleteNTT))
                        .collect::<Vec<RingElement>>();
                    vals_over_ring
                };
                let e = {
                    let mut e = RingElement::zero(Representation::Coefficients);
                    for (i, &val) in e_1_values.iter().enumerate() {
                        e.v[i as usize] = val;
                    }
                    e.from_coefficients_to_strided_coefficients();
                    e.from_strided_coefficients_to_incomplete_ntt_representation();
                    e.conjugate_in_place();
                    e
                };

                fine_proj_ctx
                    .lhs_consistency_flatter_sumcheck
                    .borrow_mut()
                    .load_from(&lhs_multipier_ring);
                fine_proj_ctx
                    .rhs_consistency_flatter_sumcheck
                    .borrow_mut()
                    .load_from(&rhs_multipier_ring);
                fine_proj_ctx
                    .rhs_scalar_consistency_sumcheck
                    .borrow_mut()
                    .load_from(&vec![e]);
            }
            // RHS: fold_challenge (same for all batches, already loaded in folding_challenges_sumcheck)
            fine_proj_contexts
                .rhs_fold_challenge_sumcheck
                .borrow_mut()
                .load_from(folding_challenges);
        }
    }
}
