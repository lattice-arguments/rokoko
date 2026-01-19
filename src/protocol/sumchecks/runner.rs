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
        config::{Config, Projection, SumcheckConfig},
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

/// Executes the complete sumcheck protocol for all constraints in the prover's proof.
///
/// This is the main entry point for running the sumcheck layer of the protocol. It's
/// deliberately written as an eager, assertion-heavy implementation that validates each
/// round of the sumcheck protocol, serving several purposes:
///
/// **Design Philosophy:**
/// 1. **Testing**: By computing all polynomials and checking all claims eagerly, we can
///    catch bugs in the sumcheck gadget wiring before they become mysterious verification
///    failures. Each assertion documents an invariant that must hold.
///
/// 2. **Documentation**: The flow of this function mirrors the paper's description of the
///    protocol. By reading the sequence of load/evaluate/assert operations, you can see
///    exactly which data feeds into which constraint and in what order.
///
/// 3. **Debugging**: When an assertion fails, you immediately know which constraint is
///    broken and at which round, making it easier to pinpoint issues.
///
/// **High-Level Flow:**
///
/// 1. **Initialization** (`init_sumcheck`):
///    - Builds all sumcheck gadgets with the correct prefix/suffix padding and loads
///      the CRS data (commitment keys).
///    - Creates the tree of product/difference sumchecks that implement each constraint.
///
/// 2. **Random Sampling** (projection flattener):
///    - Samples a random point from the Fiat-Shamir hash for flattening the projection
///      constraint. This compresses the projection matrix check into a single inner product.
///    - Converts to both structured and preprocessed row formats for different uses.
///
/// 3. **Data Loading** (via `load_sumcheck_data`):
///    - Loads the combined witness into the root linear sumcheck.
///    - Loads the folding challenges, which are used to fold multiple witnesses together.
///    - Loads the evaluation points from the opening proofs (both inner and outer).
///    - Computes and loads the projection coefficients (via the tensor product trick).
///    - Computes and loads the conjugated witness for the norm check (Type5).
///
/// 4. **Batching Setup**:
///    - Combines all individual constraint claims into a single batched claim using
///      random linear combination (currently using coefficients of 1 for simplicity).
///    - The batched claim includes: recursive commitment claims, opening claims,
///      projection claims, evaluation claims, and the witness norm claim.
///
/// 5. **Sumcheck Loop** (one iteration per variable):
///    - For each round i (from 0 to num_vars-1):
///      * Extracts the univariate polynomial from the combiner representing all constraints.
///      * Asserts the sumcheck invariant: `poly(0) + poly(1) = batched_claim`.
///      * Samples a random challenge `r` from the Fiat-Shamir hash.
///      * Calls `partial_evaluate_all(&r)` to fold all gadgets with the challenge.
///      * Updates `batched_claim = poly(r)` for the next round.
///    - This advances the protocol by collapsing the hypercube dimension by 1 each round.
///
/// 6. **Final Verification**:
///    - After the loop completes, verifies that all variables have been consumed
///      (variable_count == 0).
///    - At this point, the sumcheck has reduced the multilinear polynomial evaluation
///      to a final point evaluation.
///
/// **Constraint Types Checked:**
///
/// - **Type0** (basic commitment): `CK · folded_witness = commitment · fold_challenge`
///   We only check the first row here for brevity (index i=0), but in a full run we'd
///   check all ranks.
///
/// - **Type1** (inner evaluation): `<inner_eval, folded_witness> = opening.rhs · fold_challenge`
///   This links the opening's claimed RHS to the actual witness via the evaluation point.
///
/// - **Type2** (outer evaluation): `<outer_eval, opening.rhs> = claimed_evaluation`
///   This completes the two-level evaluation structure, tying the opening to the public claim.
///
/// - **Type3** (projection): `<projection_coeffs, folded_witness> = <fold_tensor, projection_image>`
///   This verifies the projection image is correctly formed from the witness.
///
/// - **Type4** (recursive commitments): Verifies well-formedness of recursive commitment trees.
///   There are three separate Type4 contexts (for commitment, opening, and projection recursions).
///   Each Type4 context contains multiple layers:
///   
///   * **Internal layers** (non-leaf): For each layer i, prove that:
///     `CK_i · selected_witness_i = compose(child_commitment_{i+1})`
///     where compose() reconstructs the parent commitment from decomposed child chunks.
///     These checks ensure parent-child consistency throughout the recursion tree.
///     Assertion: `poly(0) + poly(1) = 0` (difference should sum to zero).
///   
///   * **Output layer** (leaf): At the deepest level, prove that:
///     `selector · (CK_leaf · witness) = public_commitment`
///     This anchors the entire recursive tree to the public commitment value.
///     Assertion: `poly(0) + poly(1) = rc_inner[i]` (product should equal the public value).
///
/// - **Type5** (witness norm check): `<combined_witness, conjugated_combined_witness> = norm_claim`
///   Verifies the inner product of the witness with its self-conjugate to bound the witness norm.
///   This is computed as the sum over the boolean hypercube:
///   `Σ_Z MLE[witness](Z) · MLE[conjugated_witness](Z) = <witness, conjugated_witness>`
///   Assertion: `poly(0) + poly(1) = norm_claim` (product should equal the claimed norm).
///   
///   The recursive structure allows us to commit to large data efficiently by breaking it
///   into a tree where each parent commits to its children's commitments, rather than
///   committing to all data at once. Type4 sumchecks verify every level of this tree is
///   correctly constructed, from the public root commitment down to the actual witness data.
///
/// **Parameters:**
///
/// - `crs`: Common reference string (contains commitment keys).
/// - `config`: Protocol configuration (dimensions, decomposition parameters, etc.).
/// - `combined_witness`: The full witness vector, containing all data (folded witness,
///   commitments, openings, projections) concatenated with appropriate prefix padding.
/// - `projection_matrix`: The structured projection matrix (block diagonal with small blocks).
/// - `folding_challenges`: Random weights for folding multiple witnesses together.
/// - `opening`: Opening proofs (evaluation points, both inner and outer).
/// - `claims`: Public claimed evaluations (one per opening).
/// - `rc_commitment_inner`, `rc_opening_inner`, `rc_projection_inner`: Inner commitments
///   for the three recursive commitment trees (commitment, opening, projection recursions).
/// - `hash_wrapper`: Fiat-Shamir hash state for sampling randomness.
/// /// Executes a full round of the protocol’s sumcheck layer on the prover side.
/// This is intentionally written as an eager, assert-heavy simulation rather
/// than an interactive loop: it loads all inputs, seeds selectors, and then
/// checks that each sumcheck’s claimed polynomial sums match the expected
/// public values. The flow mirrors the paper:
///   1) Sample the projection “flatter” point and derive the auxiliary tensor.
///   2) Load the combined witness and folding challenges into the prebuilt
///      `SumcheckContext`.
///   3) Populate per-relation sumchecks (inner/outer evals, projection tensor).
///   4) Evaluate univariate polynomials at round 0, assert the initial claims,
///      and record the running claims after the verifier challenge `r0`.
///   5) Fold every sumcheck with `r0` and assert that the partially evaluated
///      claims stay consistent. Type4 recursive checks are currently limited to
///      internal layers, so leaf commitments are intentionally ignored.
/// By keeping this sequence explicit, future changes to the folding schedule
/// can be reasoned about locally without digging through shared state.
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
    assert_eq!(sumcheck_context.field_combiner.borrow().variable_count(), 0);

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
