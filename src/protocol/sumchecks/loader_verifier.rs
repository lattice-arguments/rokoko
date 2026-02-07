use crate::{
    common::{
        arithmetic::precompute_structured_values_fast,
        config::{DEGREE, HALF_DEGREE, NOF_BATCHES},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    protocol::{
        project_2::BatchedProjectionChallengesSuccinct,
        sumchecks::helpers::{projection_flatter_1_times_matrix, split_projection_flatter},
    },
};

use super::context_verifier::VerifierSumcheckContext;

/// Loads verifier-side evaluation gadgets with public claims and evaluation points.
///
/// Unlike the prover loader, the verifier only sees folded claims rather than the
/// full witness, so we seed the fake linear evaluations with those claims and load
/// evaluation points in their structured form.
pub fn load_verifier_sumcheck_data(
    verifier_sumcheck_context: &mut VerifierSumcheckContext,
    folding_challenges: &[RingElement],
    claim_over_witness: &RingElement,
    claim_over_witness_conjugate: &RingElement,
    evaluation_points_inner: &[StructuredRow],
    evaluation_points_outer: &[StructuredRow],
    projection_matrix: &ProjectionMatrix,
    projection_matrix_flatter_structured: &Option<StructuredRow>, // Only needed for type0 projection
    challenges_3_1: &Option<[BatchedProjectionChallengesSuccinct; NOF_BATCHES]>,
    combination: &[RingElement],
    qe: &[QuadraticExtension; HALF_DEGREE],
) {
    verifier_sumcheck_context
        .combined_witness_evaluation
        .borrow_mut()
        .set_result(claim_over_witness.clone());

    verifier_sumcheck_context
        .type5evaluation
        .conjugated_combined_witness_evaluation
        .borrow_mut()
        .set_result(claim_over_witness_conjugate.clone());

    verifier_sumcheck_context
        .folding_challenges_evaluation
        .borrow_mut()
        .load_from(folding_challenges);

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

    for (type2_eval, point) in verifier_sumcheck_context
        .type2evaluations
        .iter()
        .zip(evaluation_points_outer.iter())
    {
        type2_eval
            .outer_evaluation
            .borrow_mut()
            .load_from(point.clone());
    }
    if let Some(type3_eval) = &mut verifier_sumcheck_context.type3evaluation {
        // Type3: projection image consistency with split LHS structure
        // LHS: Prod(flatter_0, flatter_1·matrix) where flatter_0 covers elder (block) variables
        // and flatter_1·matrix covers LS (within-block) variables

        let (projection_flatter_0_structured, projection_flatter_1_structured) =
            split_projection_flatter(
                projection_matrix_flatter_structured.as_ref().unwrap(),
                projection_matrix.projection_height,
            );

        type3_eval
            .lhs_flatter_0_evaluation
            .borrow_mut()
            .load_from(projection_flatter_0_structured.clone());

        // Load flatter_1 · projection_matrix (within-block coefficients)
        let projection_flatter_1_preprocessed =
            PreprocessedRow::from_structured_row(&projection_flatter_1_structured); // this is over field actually

        let flatter_1_times_matrix = projection_flatter_1_times_matrix(
            projection_matrix,
            &projection_flatter_1_preprocessed,
        );

        type3_eval
            .lhs_flatter_1_times_matrix_evaluation_field
            .borrow_mut()
            .load_from(&flatter_1_times_matrix);

        // RHS: Split into projection_flatter and fold_challenge (Product)
        type3_eval
            .rhs_projection_flatter_evaluation
            .borrow_mut()
            .load_from(
                projection_matrix_flatter_structured
                    .as_ref()
                    .unwrap()
                    .clone(),
            );

        type3_eval
            .rhs_fold_challenge_evaluation
            .borrow_mut()
            .load_from(folding_challenges);
    }

    if let Some(type3_1_eval) = &mut verifier_sumcheck_context.type3_1_evaluations {
        // Type3_1_A: projection image consistency for type1.1 projection

        type3_1_eval
            .rhs_fold_challenge_evaluation
            .borrow_mut()
            .load_from(folding_challenges);

        for (batch_idx, challenges) in challenges_3_1.as_ref().unwrap().iter().enumerate() {
            type3_1_eval.sumchecks[batch_idx]
                .lhs_flatter_1_times_matrix_evaluation
                .borrow_mut()
                .load_from(&challenges.j_batched);

            let c_0_field = StructuredRow {
                tensor_layers: challenges
                    .c_0_layers
                    .iter()
                    .map(|e| QuadraticExtension { coeffs: [*e, 0] })
                    .collect::<Vec<_>>(),
            };
            type3_1_eval.sumchecks[batch_idx]
                .lhs_flatter_0_evaluation_field
                .borrow_mut()
                .load_from(c_0_field);

            // consistency cehck between embedded constant terms
            let (e_0_layers, e) = {
                let mut e_0_layers = Vec::new();
                let mut e_1_layers = Vec::new();
                for (i, &layer) in challenges.c_1_layers.iter().enumerate() {
                    if i < challenges.c_1_layers.len() - DEGREE.ilog2() as usize {
                        e_0_layers.push(layer);
                    } else {
                        e_1_layers.push(layer);
                    }
                }

                let e_1_values = precompute_structured_values_fast(&e_1_layers);
                let mut e = RingElement::zero(Representation::Coefficients);
                for (i, &val) in e_1_values.iter().enumerate() {
                    e.v[i as usize] = val;
                }
                e.from_coefficients_to_even_odd_coefficients();
                e.from_even_odd_coefficients_to_incomplete_ntt_representation();
                e.conjugate_in_place();
                (e_0_layers, e)
            };

            let lhs_layers_fields = StructuredRow {
                tensor_layers: challenges
                    .c_2_layers
                    .iter()
                    .map(|&x| QuadraticExtension { coeffs: [x, 0] })
                    .collect::<Vec<QuadraticExtension>>(),
            };

            let rhs_layers_field = {
                let mut layers = Vec::new();
                for c_2 in &challenges.c_2_layers {
                    layers.push(QuadraticExtension { coeffs: [*c_2, 0] });
                }

                for c_0 in &challenges.c_0_layers {
                    layers.push(QuadraticExtension { coeffs: [*c_0, 0] });
                }

                for layer in &e_0_layers {
                    layers.push(QuadraticExtension {
                        coeffs: [*layer, 0],
                    });
                }
                StructuredRow {
                    tensor_layers: layers,
                }
            };

            type3_1_eval.sumchecks[batch_idx]
                .lhs_consistency_flatter_evaluation_field
                .borrow_mut()
                .load_from(lhs_layers_fields);

            type3_1_eval.sumchecks[batch_idx]
                .rhs_consistency_flatter_evaluation_field
                .borrow_mut()
                .load_from(rhs_layers_field);

            type3_1_eval.sumchecks[batch_idx]
                .rhs_scalar_consistency_evaluation
                .borrow_mut()
                .load_from(&vec![e]);
        }
    }
    // Load combiner challenges
    verifier_sumcheck_context
        .combiner_evaluation
        .borrow_mut()
        .load_challenges_from(combination);

    verifier_sumcheck_context
        .field_combiner_evaluation
        .borrow_mut()
        .load_challenges_from(qe.clone());

    verifier_sumcheck_context
        .combiner_evaluation
        .borrow_mut()
        .load_challenges_from(&combination);

    verifier_sumcheck_context
        .field_combiner_evaluation
        .borrow_mut()
        .load_challenges_from(qe.clone());
}
