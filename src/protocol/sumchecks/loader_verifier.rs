use crate::{
    common::{
        arithmetic::field_to_ring_element_into,
        config::HALF_DEGREE,
        matrix::new_vec_zero_preallocated,
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{QuadraticExtension, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    protocol::{
        config::Config,
        sumchecks::helpers::{
            projection_coefficients, projection_flatter_1_times_matrix, split_projection_flatter,
            tensor_product,
        },
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
    folding_challenges: &Vec<RingElement>,
    claim_over_witness: &RingElement,
    claim_over_witness_conjugate: &RingElement,
    evaluation_points_inner: &Vec<StructuredRow>,
    evaluation_points_outer: &Vec<StructuredRow>,
    projection_matrix: &ProjectionMatrix,
    projection_matrix_flatter_structured: &StructuredRow,
    combination: &Vec<RingElement>,
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
    {
        // Type3: projection image consistency with split LHS structure
        // LHS: Prod(flatter_0, flatter_1·matrix) where flatter_0 covers elder (block) variables
        // and flatter_1·matrix covers LS (within-block) variables

        let (projection_flatter_0_structured, projection_flatter_1_structured) =
            split_projection_flatter(projection_matrix_flatter_structured);

        verifier_sumcheck_context
            .type3evaluation
            .lhs_flatter_0_evaluation
            .borrow_mut()
            .load_from(projection_flatter_0_structured.clone());

        // Load flatter_1 · projection_matrix (within-block coefficients)
        let projection_flatter_1_preprocessed =
            PreprocessedRow::from_structured_row(&projection_flatter_1_structured);
        let flatter_1_times_matrix = projection_flatter_1_times_matrix(
            projection_matrix,
            &projection_flatter_1_preprocessed,
        );

        verifier_sumcheck_context
            .type3evaluation
            .lhs_flatter_1_times_matrix_evaluation_field
            .borrow_mut()
            .load_from(&flatter_1_times_matrix);

        // RHS: Split into projection_flatter and fold_challenge (Product)
        verifier_sumcheck_context
            .type3evaluation
            .rhs_projection_flatter_evaluation
            .borrow_mut()
            .load_from(projection_matrix_flatter_structured.clone());

        verifier_sumcheck_context
            .type3evaluation
            .rhs_fold_challenge_evaluation
            .borrow_mut()
            .load_from(folding_challenges);
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
