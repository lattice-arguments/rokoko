use crate::{
    common::{
        config::HALF_DEGREE,
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{QuadraticExtension, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    protocol::{
        config::Config,
        sumchecks::helpers::{projection_coefficients, tensor_product},
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
    config: &Config,
    folding_challenges: &Vec<RingElement>,
    claim_over_witness: &RingElement,
    claim_over_witness_conjugate: &RingElement,
    evaluation_points_inner: &Vec<StructuredRow>,
    evaluation_points_outer: &Vec<StructuredRow>,
    projection_matrix: &ProjectionMatrix,
    projection_matrix_flatter_structured: &StructuredRow,
    projection_matrix_flatter_preprocessed: &PreprocessedRow,
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

    let projection_coeffs = projection_coefficients(
        projection_matrix,
        projection_matrix_flatter_structured,
        config.witness_height,
        config.projection_ratio,
    );
    verifier_sumcheck_context
        .type3evaluation
        .lhs_evaluation
        .borrow_mut()
        .load_from(&projection_coeffs);

    let fold_tensor = tensor_product(
        folding_challenges,
        &projection_matrix_flatter_preprocessed.preprocessed_row,
    );
    verifier_sumcheck_context
        .type3evaluation
        .rhs_evaluation
        .borrow_mut()
        .load_from(&fold_tensor);

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
