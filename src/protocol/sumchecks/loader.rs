use crate::{
    common::{
        matrix::new_vec_zero_preallocated,
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::RingElement,
        structured_row::{PreprocessedRow, StructuredRow},
    },
    protocol::{config::Config, open::Opening},
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
    folding_challenges: &Vec<RingElement>,
    opening: &Opening,
    projection_matrix: &ProjectionMatrix,
    projection_matrix_flatter_structured: &StructuredRow,
    projection_matrix_flatter_preprocessed: &PreprocessedRow,
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

    // Compute and load conjugated witness for norm check (type5)
    let mut conjugated_combined_witness = new_vec_zero_preallocated(combined_witness.len());
    combined_witness
        .iter()
        .zip(conjugated_combined_witness.iter_mut())
        .for_each(|(orig, conj)| {
            orig.conjugate_into(conj);
        });

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
    let type3_sc = &mut sumcheck_context.type3sumcheck;
    {
        let projection_coeffs = projection_coefficients(
            projection_matrix,
            projection_matrix_flatter_structured,
            config.witness_height,
            config.projection_ratio,
        );
        type3_sc
            .lhs_sumcheck
            .borrow_mut()
            .load_from(&projection_coeffs);

        let fold_tensor = tensor_product(
            folding_challenges,
            &projection_matrix_flatter_preprocessed.preprocessed_row,
        );
        type3_sc.rhs_sumcheck.borrow_mut().load_from(&fold_tensor);
    }
}
