use core::hash;

use crate::{
    common::{
        arithmetic::{ONE, inner_product},
        hash::HashWrapper,
        matrix::new_vec_zero_preallocated,
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{Representation, RingElement},
        structured_row::PreprocessedRow,
    },
    protocol::{
        config::Config,
        crs,
        open::{Opening, evaluation_point_to_structured_row},
        sumcheck_utils::{
            common::HighOrderSumcheckData,
            polynomial::Polynomial,
        },
    },
};

use super::{
    builder::init_sumcheck,
    loader::load_sumcheck_data,
};
/// Executes a full sumcheck protocol round for all constraints in the prover's proof.
///
/// This is the main entry point for running the sumcheck layer of the protocol. It's
/// deliberately written as an eager, assertion-heavy simulation rather than an interactive
/// loop, which serves several purposes:
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
///    broken. In an interactive protocol, failures might not manifest until the final
///    verification step, making it hard to pinpoint the issue.
///
/// **High-Level Flow:**
///
/// 1. **Initialization** (`init_sumcheck`):
///    - Builds all sumcheck gadgets with the correct prefix/suffix padding and loads
///      the CRS data (commitment keys).
///    - Creates the tree of product/difference sumchecks that implement each constraint.
///
/// 2. **Random Sampling** (projection flattener):
///    - Samples a random point for flattening the projection constraint. This is the only
///      randomness we need from the Fiat-Shamir hash, and it's used to compress the
///      projection matrix check into a single inner product.
///
/// 3. **Data Loading**:
///    - Loads the combined witness into the root linear sumcheck.
///    - Loads the folding challenges, which are used to fold multiple witnesses together.
///    - Loads the evaluation points from the opening proofs (both inner and outer).
///    - Computes and loads the projection coefficients (via the tensor product trick).
///
/// 4. **Initial Evaluation (Round 0)**:
///    - For each constraint type (type0, type1, type2, type3, type4, type5), extracts the
///      univariate polynomial that the verifier will see in round 0.
///    - Asserts that the polynomial sums correctly:
///      * For zero-claims: `poly(0) + poly(1) = 0`
///      * For public claims: `poly(0) + poly(1) = claim`
///    - Records the claim after folding with the verifier's challenge `r0` (here, hardcoded
///      to 7 for testing, but in a real protocol this would come from Fiat-Shamir).
///
/// 5. **Partial Evaluation (Folding)**:
///    - Calls `partial_evaluate_all(&r0)` to fold every sumcheck gadget with the challenge.
///    - This advances the protocol by one round: the hypercube dimension decreases by 1,
///      and each gadget's internal state is updated to reflect the evaluation at `r0`.
///
/// 6. **Post-Fold Verification**:
///    - Re-extracts the univariate polynomials (now for round 1) and asserts that they
///      sum to the recorded claims from round 0.
///    - This confirms that the folding was done correctly and that the constraints are
///      still consistent after the fold.
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
    crs: &crs::CRS,
    config: &Config,
    combined_witness: &Vec<RingElement>,
    projection_matrix: &ProjectionMatrix,
    folding_challenges: &Vec<RingElement>,
    opening: &Opening,
    claims: &Vec<RingElement>,
    rc_commitment_inner: &Vec<RingElement>,
    rc_opening_inner: &Vec<RingElement>,
    rc_projection_inner: &Vec<RingElement>,
    hash_wrapper: &mut HashWrapper,
) {
    let mut sumcheck_context = init_sumcheck(crs, config);

    let projection_height_flat = config.witness_height / config.projection_ratio;
    let mut projection_matrix_flatter_base =
        new_vec_zero_preallocated(projection_height_flat.ilog2() as usize);
    hash_wrapper.sample_ring_element_vec_into(&mut projection_matrix_flatter_base);

    let projection_matrix_flatter_structured =
        evaluation_point_to_structured_row(&projection_matrix_flatter_base);

    let projection_matrix_flatter =
        PreprocessedRow::from_structured_row(&projection_matrix_flatter_structured);

    // Load all data into the sumcheck context
    load_sumcheck_data(
        &mut sumcheck_context,
        config,
        combined_witness,
        folding_challenges,
        opening,
        projection_matrix,
        &projection_matrix_flatter_structured,
        &projection_matrix_flatter,
    );


    let mut conjugated_combined_witness = new_vec_zero_preallocated(combined_witness.len());
    combined_witness
        .iter()
        .zip(conjugated_combined_witness.iter_mut())
        .for_each(|(orig, conj)| {
            orig.conjugate_into(conj);
        });

    // let norm_claim = RingElement::zero(Representation::IncompleteNTT);
    let norm_claim = inner_product(&combined_witness, &conjugated_combined_witness);


    let mut poly = Polynomial::new(0);

    let combination = vec![ONE.clone(); sumcheck_context.combiner.borrow().sumchecks_count()]; 
    
    sumcheck_context.combiner.borrow_mut().load_challenges_from(&combination);

    
    let mut num_vars = sumcheck_context
        .combiner
        .borrow()
        .variable_count();

    let mut batched_claim = RingElement::zero(Representation::IncompleteNTT);
    // we need to add all claims (assuming for now that we combine with 1s)
    for rc_inner_i in rc_commitment_inner.iter() {
        batched_claim += rc_inner_i;
    }

    for rc_opening_i in rc_opening_inner.iter() {
        batched_claim += rc_opening_i;
    }

    for rc_projection_i in rc_projection_inner.iter() {
        batched_claim += rc_projection_i;
    }

    for claim in claims.iter() {
        batched_claim += claim;
    }


    batched_claim += &norm_claim;

    print!("Num vars before sumcheck: {}\n", num_vars);

    while num_vars > 0 {
        num_vars -= 1;
        // round 0

        sumcheck_context
            .combiner
            .borrow_mut()
            .univariate_polynomial_into(&mut poly);
        
        assert_eq!(&poly.at_zero() + &poly.at_one(), batched_claim);

        let mut r = RingElement::zero(Representation::IncompleteNTT);
        hash_wrapper.sample_ring_element_into(&mut r);
        sumcheck_context.partial_evaluate_all(&r);

        batched_claim = poly.at(&r);
    }

    // final round
    assert_eq!(sumcheck_context
        .combiner
        .borrow().variable_count(), 0);

    
}
