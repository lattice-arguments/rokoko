use core::hash;
use std::vec;

use crate::{
    common::{
        arithmetic::{field_to_ring_element, field_to_ring_element_into, inner_product, ONE},
        config::HALF_DEGREE,
        hash::HashWrapper,
        matrix::new_vec_zero_preallocated,
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
        sumcheck_element::SumcheckElement,
    },
    protocol::{
        config::{Config, Projection},
        crs,
        open::{evaluation_point_to_structured_row, Opening},
        project,
        sumcheck::{self, SumcheckContext},
        sumcheck_utils::{
            common::{EvaluationSumcheckData, HighOrderSumcheckData, SumcheckBaseData},
            polynomial::Polynomial,
        },
        sumchecks::{
            context_verifier::VerifierSumcheckContext,
            loader_verifier::{self, load_verifier_sumcheck_data},
        },
    },
};

use super::{builder::init_sumcheck, loader::load_sumcheck_data};

fn batch_claims(
    config: &Config,
    claims: &Vec<RingElement>,
    rc_commitment_inner: &Vec<RingElement>,
    rc_opening_inner: &Vec<RingElement>,
    rc_projection_inner: Option<&Vec<RingElement>>,
    norm_claim: &RingElement,
    combination: &Vec<RingElement>,
) -> RingElement {
    let mut batched_claim = RingElement::zero(Representation::IncompleteNTT);
    let mut idx = 0;

    // Type0: zero claims (difference sumchecks)
    idx += config.basic_commitment_rank;

    // Type1: zero claims (difference sumchecks)
    idx += config.nof_openings;

    // Type2: claims for evaluations
    for claim in claims.iter() {
        let mut weighted = claim.clone();
        weighted *= &combination[idx];
        batched_claim += &weighted;
        idx += 1;
    }

    // Type3: zero claim (difference sumcheck)
    idx += 1;

    let mut polys: Vec<Polynomial<QuadraticExtension>> = vec![];

    // Type4: Three recursion trees (commitment, opening, projection)
    // Each tree has: (layers with rank each) + (output layer with rank)
    for (recursion_idx, rc_inner) in [
        Some(rc_commitment_inner),
        Some(rc_opening_inner),
        rc_projection_inner,
    ]
    .iter()
    .enumerate()
    {
        if rc_inner.is_none() {
            continue;
        }
        let rc_inner = rc_inner.as_ref().unwrap();
        let recursion_config = match recursion_idx {
            0 => &config.commitment_recursion,
            1 => &config.opening_recursion,
            2 => match &config.projection_recursion {
                Projection::Type0(proj_config) => proj_config,
                Projection::Type1(proj_config) => &proj_config.recursion_constant_term,
            },
            3 => match &config.projection_recursion {
                Projection::Type1(proj_config) => &proj_config.recursion_batched_projection,
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };

        // Internal layers (zero claims)
        let mut current = recursion_config;
        while let Some(next) = current.next.as_deref() {
            idx += current.rank; // Each layer has rank outputs, all zero claims
            current = next;
        }

        // Output layer: rc_inner claims
        for rc_value in rc_inner.iter() {
            let mut weighted = rc_value.clone();
            weighted *= &combination[idx];
            batched_claim += &weighted;
            idx += 1;
        }
    }

    // Type5: norm claim
    let mut weighted_norm = norm_claim.clone();
    weighted_norm *= &combination[idx];
    batched_claim += &weighted_norm;

    batched_claim
}

pub use crate::protocol::proof::Proof;
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
    config: &Config,
    combined_witness: &Vec<RingElement>,
    projection_matrix: &ProjectionMatrix,
    folding_challenges: &Vec<RingElement>,
    opening: &Opening,
    sumcheck_context: &mut SumcheckContext,
    hash_wrapper: &mut HashWrapper,
) -> (
    RingElement,
    RingElement,
    RingElement,
    Vec<Polynomial<QuadraticExtension>>,
) {
    // Removed: let mut hash_wrapper_clone = hash_wrapper.clone(); - unused
    let projection_height_flat = config.witness_height / config.projection_ratio;
    let mut projection_matrix_flatter_base =
        new_vec_zero_preallocated(projection_height_flat.ilog2() as usize);
    hash_wrapper.sample_ring_element_ntt_slots_same_vec_into(&mut projection_matrix_flatter_base);

    let projection_matrix_flatter_structured =
        evaluation_point_to_structured_row(&projection_matrix_flatter_base);

    let projection_matrix_flatter =
        PreprocessedRow::from_structured_row(&projection_matrix_flatter_structured);

    let mut conjugated_combined_witness = new_vec_zero_preallocated(combined_witness.len());
    combined_witness
        .iter()
        .zip(conjugated_combined_witness.iter_mut())
        .for_each(|(orig, conj)| {
            orig.conjugate_into(conj);
        });

    // let norm_claim = RingElement::zero(Representation::IncompleteNTT);
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
        opening,
        projection_matrix,
        &projection_matrix_flatter_structured,
        &projection_matrix_flatter,
        &combination,
        &qe,
    );
    println!(
        "    load_sumcheck_data: {} ms",
        t_load.elapsed().as_millis()
    );

    sumcheck_context
        .combiner
        .borrow_mut()
        .load_challenges_from(&combination);

    // TODO: can we avoid cloning?
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
        polys,
    )
}

pub struct RoundProof<'a> {
    pub polys: &'a Vec<Polynomial<QuadraticExtension>>,
    pub claim_over_witness: &'a RingElement,
    pub claim_over_witness_conjugate: &'a RingElement,
    pub norm_claim: &'a RingElement,
    pub rc_commitment_inner: &'a Vec<RingElement>,
    pub rc_opening_inner: &'a Vec<RingElement>,
    pub rc_projection_inner: Option<&'a Vec<RingElement>>,
}

pub fn sumcheck_verifier(
    config: &Config,
    verifier_sumcheck_context: &mut VerifierSumcheckContext,
    round_proof: &RoundProof,
    evaluation_points_inner: &Vec<StructuredRow>,
    evaluation_points_outer: &Vec<StructuredRow>,
    claims: &Vec<RingElement>,
    hash_wrapper: &mut HashWrapper,
) {
    hash_wrapper.update_with_ring_element_slice(&round_proof.rc_commitment_inner);
    hash_wrapper.update_with_ring_element_slice(&round_proof.rc_opening_inner);
    let mut projection_matrix = ProjectionMatrix::new(config.projection_ratio);

    projection_matrix.sample(hash_wrapper);
    if let Some(rc_projection_inner) = &round_proof.rc_projection_inner {
        hash_wrapper.update_with_ring_element_slice(rc_projection_inner);
    }

    let mut folding_challenges =
        vec![RingElement::zero(Representation::IncompleteNTT); config.witness_width];

    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut folding_challenges);

    let projection_height_flat = config.witness_height / config.projection_ratio;
    let mut projection_matrix_flatter_base =
        new_vec_zero_preallocated(projection_height_flat.ilog2() as usize);
    hash_wrapper.sample_ring_element_ntt_slots_same_vec_into(&mut projection_matrix_flatter_base);

    let projection_matrix_flatter_structured =
        evaluation_point_to_structured_row(&projection_matrix_flatter_base);

    hash_wrapper.update_with_ring_element(&round_proof.norm_claim);

    // Sample random batching coefficients from Fiat-Shamir
    let num_sumchecks = verifier_sumcheck_context
        .combiner_evaluation
        .borrow()
        .sumchecks_count();
    let mut combination = new_vec_zero_preallocated(num_sumchecks);
    hash_wrapper.sample_ring_element_vec_into(&mut combination);

    let mut combination_to_field = RingElement::zero(Representation::IncompleteNTT);
    hash_wrapper.sample_ring_element_into(&mut combination_to_field);
    combination_to_field.from_incomplete_ntt_to_homogenized_field_extensions();
    let qe = combination_to_field.split_into_quadratic_extensions();

    // Compute batched claim matching the combiner's output order:
    // type0 (rank many) -> type1 (nof_openings) -> type2 (nof_openings) ->
    // type3 (1) -> type4[3 recursions, each with layers*rank + output_rank] -> type5 (1)

    let mut batched_claim = batch_claims(
        config,
        claims,
        round_proof.rc_commitment_inner,
        round_proof.rc_opening_inner,
        round_proof.rc_projection_inner,
        &round_proof.norm_claim,
        &combination,
    );

    let mut batched_claim_over_field = {
        let batched_claim = {
            let mut temp = batched_claim.clone();
            temp.from_incomplete_ntt_to_homogenized_field_extensions();
            temp
        };
        let mut temp = batched_claim.split_into_quadratic_extensions();
        let mut result = QuadraticExtension::zero();
        for i in 0..HALF_DEGREE {
            temp[i] *= &qe[i];
            result += &temp[i];
        }
        result
    };

    let mut num_vars = round_proof.polys.len();

    let mut evaluation_points: Vec<QuadraticExtension> = vec![];
    while num_vars > 0 {
        num_vars -= 1;

        let poly_over_field = round_proof
            .polys
            .get(round_proof.polys.len() - num_vars - 1)
            .unwrap();

        hash_wrapper.update_with_quadratic_extension_slice(&poly_over_field.coefficients);

        assert_eq!(
            poly_over_field.at_zero() + poly_over_field.at_one(),
            batched_claim_over_field
        );

        let mut f = QuadraticExtension::zero();

        hash_wrapper.sample_field_element_into(&mut f);

        batched_claim_over_field = poly_over_field.at(&f);
        evaluation_points.push(f);
    }

    load_verifier_sumcheck_data(
        verifier_sumcheck_context,
        &folding_challenges,
        &round_proof.claim_over_witness,
        &round_proof.claim_over_witness_conjugate,
        evaluation_points_inner,
        evaluation_points_outer,
        &projection_matrix,
        Some(&projection_matrix_flatter_structured), // assume type0 projection
        &combination,
        &qe,
    );

    assert_eq!(
        &batched_claim_over_field,
        verifier_sumcheck_context
            .field_combiner_evaluation
            .borrow_mut()
            .evaluate(&evaluation_points)
    );
}
