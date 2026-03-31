use crate::{
    common::{
        arithmetic::{field_to_ring_element_into, inner_product},
        config::NOF_BATCHES,
        hash::HashWrapper,
        matrix::new_vec_zero_preallocated,
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        structured_row::PreprocessedRow,
        sumcheck_element::SumcheckElement,
    },
    protocol::{
        config::{Projection, SumcheckConfig},
        open::{evaluation_point_to_structured_row, Opening},
        project_2::BatchedProjectionChallenges,
        sumcheck::SumcheckContext,
        sumcheck_utils::{
            common::{HighOrderSumcheckData, SumcheckBaseData},
            polynomial::Polynomial,
        },
    },
};

use super::loader::load_sumcheck_data;

/// Executes the sumcheck protocol for all constraint types.
///
/// **Flow:**
/// 1. Sample projection flattener from Fiat-Shamir and compute conjugated witness for norm check
/// 2. Sample random batching coefficients for combining all constraints
/// 3. Load all data via `load_sumcheck_data` (witness, challenges, evaluation points, projection coefficients)
/// 4. Run sumcheck loop: for each variable, extract univariate polynomial, sample challenge, fold all gadgets
/// 5. Return final evaluations at the random point
///
/// **Constraints:**
/// - **Type0**: `CK · folded_witness = commitment · fold_challenge`
/// - **Type1**: `<inner_eval, folded_witness> = opening.rhs · fold_challenge`
/// - **Type2**: `<outer_eval, opening.rhs> = claimed_evaluation`
/// - **Type3**: `<projection_coeffs, folded_witness> = <fold_tensor, projection_image>` (block-diagonal projection)
/// - **Type3_1**: `c^T (I ⊗ projection_matrix) · folded_witness = c^T projection_image · fold_challenge` (Kronecker projection)
/// - **Type4**: Recursive commitment trees (commitment, opening, projection recursions)
///   - Internal layers: `CK_i · selected_witness_i = compose(child_commitment_{i+1})`
///   - Output layer: `selector · (CK_leaf · witness) = public_commitment`
/// - **Type5**: `<combined_witness, conjugated_combined_witness> = norm_claim`
pub fn sumcheck(
    config: &SumcheckConfig,
    combined_witness: &Vec<RingElement>,
    projection_matrix: &ProjectionMatrix,
    folding_challenges: &Vec<RingElement>,
    challenges_batching_projection_1: &Option<&[BatchedProjectionChallenges; NOF_BATCHES]>,
    opening: &Opening,
    sumcheck_context: &mut SumcheckContext,
    hash_wrapper: &mut HashWrapper,
) -> (
    RingElement,
    RingElement,
    RingElement,
    RingElement,
    Vec<Polynomial<QuadraticExtension>>,
    Vec<RingElement>,
    Option<Vec<RingElement>>,
) {
    // Removed: let mut hash_wrapper_clone = hash_wrapper.clone(); - unused
    let projection_matrix_flatter = match config.projection_recursion {
        Projection::Type0(_) => {
            let projection_height_flat = config.witness_height / config.projection_ratio;
            let mut projection_matrix_flatter_base =
                new_vec_zero_preallocated(projection_height_flat.ilog2() as usize);
            hash_wrapper
                .sample_ring_element_ntt_slots_same_vec_into(&mut projection_matrix_flatter_base);

            let projection_matrix_flatter_structured =
                evaluation_point_to_structured_row(&projection_matrix_flatter_base);
            let projection_matrix_flatter =
                PreprocessedRow::from_structured_row(&projection_matrix_flatter_structured);

            Some((
                projection_matrix_flatter,
                projection_matrix_flatter_structured,
            ))
        }
        Projection::Type1(_) => None,
        Projection::Skip => None,
    };

    let mut conjugated_combined_witness = new_vec_zero_preallocated(combined_witness.len());
    combined_witness
        .iter()
        .zip(conjugated_combined_witness.iter_mut())
        .for_each(|(orig, conj)| {
            orig.conjugate_into(conj);
        });

    let norm_claim = inner_product(&combined_witness, &conjugated_combined_witness);

    hash_wrapper.update_with_ring_element(&norm_claim);

    // Sample random batching coefficients from Fiat-Shamir
    let num_sumchecks = sumcheck_context.combiner.borrow().sumchecks_count();
    let mut combination = new_vec_zero_preallocated(num_sumchecks);
    hash_wrapper.sample_ring_element_vec_into(&mut combination);

    let mut combination_to_field = RingElement::zero(Representation::IncompleteNTT);
    hash_wrapper.sample_ring_element_into(&mut combination_to_field);
    combination_to_field.from_incomplete_ntt_to_homogenized_field_extensions();
    let qe = combination_to_field.split_into_quadratic_extensions();

    // Load all data into the sumcheck context
    let t_load = std::time::Instant::now();
    load_sumcheck_data(
        sumcheck_context,
        config,
        combined_witness,
        &conjugated_combined_witness,
        folding_challenges,
        challenges_batching_projection_1,
        opening,
        projection_matrix,
        &projection_matrix_flatter,
        &combination,
        &qe,
    );
    println!(
        "    load_sumcheck_data: {} ms",
        t_load.elapsed().as_millis()
    );

    let norm_inner_norm_claim = sumcheck_context.type5sumcheck.output_2.borrow_mut().claim();

    sumcheck_context
        .combiner
        .borrow_mut()
        .load_challenges_from(&combination);

    sumcheck_context
        .field_combiner
        .borrow_mut()
        .load_challenges_from(qe.clone());

    let mut num_vars = sumcheck_context.combiner.borrow().variable_count();
    println!(
        "    sumcheck num_vars: {}, hypercube_size: {}",
        num_vars,
        1u64 << (num_vars - 1)
    );

    let constant_term_claims =
        sumcheck_context
            .type3_1_sumchecks
            .as_ref()
            .map(|type3_1_sumchecks| {
                type3_1_sumchecks
                    .sumchecks
                    .iter()
                    .map(|type3_1_sc| type3_1_sc.output_2.borrow().claim())
                    .collect::<Vec<_>>()
            });

    // Collect evaluation points during sumcheck
    let mut evaluation_points: Vec<RingElement> = vec![];

    let mut polys: Vec<Polynomial<QuadraticExtension>> = vec![];
    let t_loop = std::time::Instant::now();
    let mut time_poly = 0;
    let mut time_eval = 0;

    while num_vars > 0 {
        num_vars -= 1;

        let t1 = std::time::Instant::now();
        let mut poly_over_field = Polynomial::<QuadraticExtension>::new(0);

        sumcheck_context
            .field_combiner
            .borrow_mut()
            .univariate_polynomial_into(&mut poly_over_field);
        time_poly += t1.elapsed().as_millis();

        hash_wrapper.update_with_quadratic_extension_slice(&poly_over_field.coefficients);

        let mut r = RingElement::zero(Representation::IncompleteNTT);
        let mut f = QuadraticExtension::zero();

        hash_wrapper.sample_field_element_into(&mut f);

        field_to_ring_element_into(&mut r, &f);
        r.from_homogenized_field_extensions_to_incomplete_ntt();

        evaluation_points.push(r.clone());

        let t2 = std::time::Instant::now();
        sumcheck_context.partial_evaluate_all(&r);
        time_eval += t2.elapsed().as_millis();

        polys.push(poly_over_field);
    }
    println!(
        "    sumcheck loop: {} ms (poly: {} ms, eval: {} ms)",
        t_loop.elapsed().as_millis(),
        time_poly,
        time_eval
    );

    // final round
    debug_assert_eq!(sumcheck_context.field_combiner.borrow().variable_count(), 0);

    let claim_over_witness = sumcheck_context
        .combined_witness_sumcheck
        .borrow()
        .final_evaluations()
        .clone();

    let claim_over_witness_conjugate = sumcheck_context
        .type5sumcheck
        .conjugated_combined_witness
        .borrow()
        .final_evaluations()
        .clone();

    evaluation_points.reverse();

    (
        claim_over_witness,
        claim_over_witness_conjugate,
        norm_claim,
        norm_inner_norm_claim,
        polys,
        evaluation_points,
        constant_term_claims,
    )
}
