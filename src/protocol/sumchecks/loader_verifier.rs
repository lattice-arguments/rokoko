use crate::{
    common::{
        config::HALF_DEGREE,
        ring_arithmetic::{QuadraticExtension, RingElement},
        structured_row::StructuredRow,
    },
    protocol::{
        config::Config,
        crs::CRS,
        open::Opening,
        sumchecks::context_verifier::VerifierSumcheckContext,
    },
};

use super::runner::Proof;

/// Load data into the verifier context - ONLY loads into leaf nodes
/// The structure has already been built by init_context_verifier
pub fn load_verifier_data(
    context: &mut VerifierSumcheckContext,
    proof: &Proof,
    _config: &Config,
    crs: &CRS,
    opening: &Opening,
    folding_challenges: &Vec<RingElement>,
    projection_matrix_flatter: &crate::common::structured_row::PreprocessedRow,
    _combination: &Vec<RingElement>,
    _combination_to_field: &[QuadraticExtension; HALF_DEGREE],
) {
    // Load prover-provided claims
    context.combined_witness_evaluation.set_result(proof.claim_over_witness.clone());
    context.type5evaluation.conjugated_combined_witness_evaluation
        .set_result(proof.claim_over_witness_conjugate.clone());

    // Load folded witness combiner data (radix weights)
    // TODO: compute and load

    // Load witness combiner constant data (signed-digit offset)
    // TODO: compute and load

    // Load basic commitment combiner data
    // TODO: compute and load

    // Load basic commitment combiner constant data
    // TODO: compute and load

    // Load opening combiner data
    // TODO: compute and load

    // Load opening combiner constant data
    // TODO: compute and load

    // Load projection combiner data
    // TODO: compute and load

    // Load projection combiner constant data
    // TODO: compute and load

    // Load folding challenges
    context.folding_challenges_evaluation.load_from(folding_challenges);

    // Load commitment key rows from CRS
    for (i, ck_eval) in context.commitment_key_rows_evaluation.iter_mut().enumerate() {
        // Get the structured row from CRS and load it
        // TODO: load from crs.structured_cks
    }

    // Load Type1 inner evaluation points
    for (i, type1_ctx) in context.type1evaluations.iter_mut().enumerate() {
        if i < opening.evaluation_points_inner.len() {
            type1_ctx.inner_evaluation.load_from(&opening.evaluation_points_inner[i].preprocessed_row);
        }
    }

    // Load Type2 outer evaluation points
    for (i, type2_ctx) in context.type2evaluations.iter_mut().enumerate() {
        if i < opening.evaluation_points_outer.len() {
            type2_ctx.outer_evaluation.load_from(&opening.evaluation_points_outer[i].preprocessed_row);
        }
    }

    // Load Type3 projection coefficients
    // TODO: compute projection_coefficients and fold_tensor, then load

    // Load Type4 CK rows for all three recursive trees
    // TODO: load from CRS for each layer

    // Note: Selectors don't need data loaded - they compute from their prefix
    // Note: Products and Diffs don't need data loaded - they reference other evaluations
    // Note: FakeEvaluations already have their results set above
}
