use crate::{
    common::{config::NOF_BATCHES, ring_arithmetic::RingElement},
    protocol::sumcheck_utils::{
        combiner::Combiner,
        common::{HighOrderSumcheckData, SumcheckBaseData},
        diff::DiffSumcheck,
        elephant_cell::ElephantCell,
        linear::LinearSumcheck,
        product::ProductSumcheck,
        ring_to_field_combiner::RingToFieldCombiner,
        selector_eq::SelectorEq,
    },
};

/// Grouping of all per-relation sumchecks we need to track during a round.
/// Each type corresponds to a semantic constraint in the protocol:
/// - type0 enforces linear commitments against the folded witness.
/// - type1 wires inner evaluation points into the opening.
/// - type2 does the same for outer evaluations.
/// - type3 ties projection images to the folded witness via a selector.
/// - type4 recursively enforces that decomposed commitments match the public
///   recursive commitments at every internal layer. The recursive structure
///   means we keep a stack of layers so we can fold every selector/combiner
///   consistently when the verifier provides a challenge.
/// - type5 verifies the inner product between the combined witness and its
///   conjugate, which is used to check witness norm constraints.
/// By keeping everything together here, folding a single random challenge
/// becomes a single method call, which reduces the risk of forgetting to
/// advance some sub-check when modifying the protocol.
pub struct SumcheckContext {
    pub combined_witness_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub folded_witness_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub folded_witness_combiner_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub witness_combiner_constant_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub folding_challenges_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub basic_commitment_combiner_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub basic_commitment_combiner_constant_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub commitment_key_rows_sumcheck: Vec<ElephantCell<LinearSumcheck<RingElement>>>,
    pub opening_combiner_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub opening_combiner_constant_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub type0sumchecks: Vec<Type0SumcheckContext>,
    pub type1sumchecks: Vec<Type1SumcheckContext>,
    pub type2sumchecks: Vec<Type2SumcheckContext>,
    pub type3sumcheck: Option<Type3SumcheckContext>,
    pub type4sumchecks: Vec<Type4SumcheckContext>,
    pub type5sumcheck: Type5SumcheckContext,
    pub type3_1_a_sumchecks: Option<Type3_1ASumcheckContextWrapper>, // it should never go together with type3sumcheck TODO: refactor for enum I guess
    pub combiner: ElephantCell<Combiner<RingElement>>,
    pub field_combiner: ElephantCell<RingToFieldCombiner>,
    pub next: Option<Box<SumcheckContext>>,
}

/// Encapsulates the bookkeeping required to fold every tracked sumcheck with
/// a single verifier challenge. The folding order mirrors the logical flow of
/// constraints so that intermediate claims stay consistent across types.
impl SumcheckContext {
    pub fn partial_evaluate_all(&mut self, r: &RingElement) {
        self.combined_witness_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.folded_witness_selector_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.folded_witness_combiner_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.witness_combiner_constant_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.folding_challenges_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.basic_commitment_combiner_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.basic_commitment_combiner_constant_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        for ck_row_sc in self.commitment_key_rows_sumcheck.iter() {
            ck_row_sc.borrow_mut().partial_evaluate(r);
        }
        for type0_sc in self.type0sumchecks.iter() {
            type0_sc
                .basic_commitment_row_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }
        self.opening_combiner_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.opening_combiner_constant_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        for type1_sc in self.type1sumchecks.iter() {
            type1_sc
                .inner_evaluation_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            type1_sc
                .opening_selector_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }
        for type2_sc in self.type2sumchecks.iter() {
            type2_sc
                .outer_evaluation_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }

        if let Some(type3_sc) = &mut self.type3sumcheck {
            type3_sc
                .projection_combiner_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            type3_sc
                .projection_combiner_constant_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            type3_sc
                .lhs_flatter_0_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            type3_sc
                .lhs_flatter_1_times_matrix_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            type3_sc
                .rhs_fold_challenge_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            type3_sc
                .rhs_projection_flatter_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            type3_sc
                .projection_selector_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }

        if let Some(type3_1_a_sumchecks) = &mut self.type3_1_a_sumchecks {
            type3_1_a_sumchecks
                .projection_combiner_constant_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            type3_1_a_sumchecks
                .projection_combiner_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            type3_1_a_sumchecks
                .rhs_fold_challenge_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            for type3_1_a_sc in type3_1_a_sumchecks.sumchecks.iter_mut() {
                type3_1_a_sc
                    .lhs_flatter_0_sumcheck
                    .borrow_mut()
                    .partial_evaluate(r);
                type3_1_a_sc
                    .lhs_flatter_1_times_matrix_sumcheck
                    .borrow_mut()
                    .partial_evaluate(r);
                type3_1_a_sc
                    .projection_selector_sumcheck
                    .borrow_mut()
                    .partial_evaluate(r);
                type3_1_a_sc
                    .lhs_consistency_flatter_sumcheck
                    .borrow_mut()
                    .partial_evaluate(r);
                type3_1_a_sc
                    .rhs_consistency_flatter_sumcheck
                    .borrow_mut()
                    .partial_evaluate(r);
                type3_1_a_sc
                    .lhs_scalar_consistency_sumcheck
                    .borrow_mut()
                    .partial_evaluate(r);
                type3_1_a_sc
                    .rhs_scalar_consistency_sumcheck
                    .borrow_mut()
                    .partial_evaluate(r);
            }
        }

        for type4_sc in self.type4sumchecks.iter_mut() {
            partial_evaluate_type4(type4_sc, r);
        }
        self.type5sumcheck
            .conjugated_combined_witness
            .borrow_mut()
            .partial_evaluate(r);
    }
}

/// Type0 sumcheck context: Basic commitment correctness constraint.
///
/// This sumcheck proves that the basic (non-recursive) commitment to the folded witness
/// is consistent with how the folded witness is actually computed. Concretely, we prove:
///
///   CK · folded_witness = commitment · fold_challenge
///
/// where:
///   - CK is a commitment key row (one row per Type0 context, rank many contexts total)
///   - folded_witness is the recomposed (from decomposed chunks) version of the witness
///     slice that we're committing to
///   - commitment is the claimed commitment value (part of the proof)
///   - fold_challenge is the random linear combination weight for folding multiple witnesses
///
/// **Why This Matters:**
/// The basic commitment is the "anchor" of the entire proof. It's what gets posted publicly
/// and what the verifier uses to check consistency. If the prover could cheat here—claim
/// a commitment that doesn't match the actual witness—the entire protocol would be broken.
///
/// **Structure of the Constraint:**
/// The output is a DiffSumcheck that computes:
///   LHS: selector · (recomposed_folded_witness · CK_row)
///   RHS: commitment_selector · (recomposed_commitment · fold_challenge)
///
/// Both sides are selected by their respective prefixes (folded witness lives at one prefix,
/// commitment lives at another in the combined witness vector), recomposed from decomposed
/// form (to handle large coefficients), and then multiplied by their respective challenge
/// vectors. The difference should sum to zero over the entire hypercube.
///
/// **Field Details:**
///   - `basic_commitment_row_sumcheck`: Selector for picking out this commitment row's slice
///     from the combined witness. Each row gets its own prefix to avoid collisions.
///   - `output`: The final DiffSumcheck that the verifier will check. This is the composition
///     of multiple product and difference sumchecks that implement the full constraint.
pub struct Type0SumcheckContext {
    pub basic_commitment_row_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub output: ElephantCell<DiffSumcheck<RingElement>>,
}

/// Type1 sumcheck context: Inner evaluation point consistency for openings.
///
/// This sumcheck proves that the opening's claimed inner evaluation is consistent with
/// the folded witness. Specifically, we prove:
///
///   <inner_evaluation_points, folded_witness> = opening.rhs · fold_challenge
///
/// where:
///   - inner_evaluation_points is a vector defining a linear evaluation of the witness
///     (typically computed from the MLE of some challenge point)
///   - folded_witness is the recomposed witness slice (same as in Type0)
///   - opening.rhs is the claimed evaluation result stored in the opening proof
///   - fold_challenge is the folding weight (same as in Type0)
///
/// **Why This Matters:**
/// Openings are how the prover reveals evaluations of the committed polynomial at
/// specific points chosen by the verifier. The verifier can't recompute these evaluations
/// directly (the witness is too large), so the prover must prove they're correct. Type1
/// sumchecks enforce that the opening's claimed RHS value actually matches the inner
/// product of the evaluation point with the witness.
///
/// **Structure of the Constraint:**
/// The output is a DiffSumcheck that computes:
///   LHS: folded_witness_selector · (recomposed_folded_witness · inner_eval_points)
///   RHS: opening_selector · (recomposed_opening_rhs · fold_challenge)
///
/// The pattern is similar to Type0: both sides are selected, recomposed, and then multiplied
/// by challenge vectors. The selector for the opening ensures we're checking the correct
/// opening (we may have multiple openings, each at a different point).
///
/// **Field Details:**
///   - `inner_evaluation_sumcheck`: Linear sumcheck loaded with the inner evaluation point's
///     coefficients. This defines "which" linear combination of the witness we're evaluating.
///   - `opening_selector_sumcheck`: Selector that picks out this opening's RHS value from
///     the combined witness. Each opening gets its own prefix.
///   - `output`: The final DiffSumcheck that the verifier checks.
pub struct Type1SumcheckContext {
    pub inner_evaluation_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub opening_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub output: ElephantCell<DiffSumcheck<RingElement>>,
}

/// Type2 sumcheck context: Outer evaluation point consistency for openings.
///
/// This sumcheck proves that the opening's RHS value, when further evaluated at the outer
/// evaluation point, matches the publicly claimed evaluation. Specifically, we prove:
///
///   <outer_evaluation_points, opening.rhs> = claimed_evaluation (public)
///
/// where:
///   - outer_evaluation_points is a vector defining a second-level evaluation (the opening
///     itself is a vector, and we're taking an inner product with it)
///   - opening.rhs is the RHS value from the opening (whose consistency was checked in Type1)
///   - claimed_evaluation is the final scalar value that the verifier sees (part of the
///     public input)
///
/// **Why This Matters:**
/// Our opening mechanism is two-level: the witness is first evaluated to produce a vector
/// (opening.rhs), and then that vector is evaluated at a second point to produce a scalar.
/// This two-level structure is essential for efficiency—it allows us to batch multiple
/// point queries and to exploit the structure of the witness (which is often a matrix or
/// a similarly structured object).
///
/// Type1 checks the first level (witness → opening.rhs), and Type2 checks the second level
/// (opening.rhs → scalar). Together, they ensure the full evaluation is correct.
///
/// **Structure of the Constraint:**
/// The output is a ProductSumcheck that computes:
///   opening_selector · (recomposed_opening_rhs · outer_eval_points)
///
/// Unlike Type0 and Type1, this is a product rather than a difference, because the result
/// should equal the public claimed_evaluation (which the verifier adds directly), not zero.
///
/// **Field Details:**
///   - `outer_evaluation_sumcheck`: Linear sumcheck loaded with the outer evaluation point's
///     coefficients.
///   - `output`: The final ProductSumcheck that the verifier checks. The verifier will
///     compute the sum of this product and compare it to the claimed_evaluation.
pub struct Type2SumcheckContext {
    pub outer_evaluation_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
}

/// Type3 sumcheck context: Projection image consistency constraint.
///
/// This sumcheck proves that the projection image (a compressed version of the witness)
/// is correctly formed. Specifically, we prove:
///
///   <projection_coeffs, folded_witness> = <fold_tensor, projection_image>
///
/// where:
///   - projection_coeffs is a vector derived from the projection matrix and a random
///     flattening point (computed by `projection_coefficients`)
///   - folded_witness is the recomposed witness slice (same as before)
///   - fold_tensor is the tensor product of fold_challenge and the projection flattener
///   - projection_image is the claimed projected value (part of the proof)
///
/// **Why This Matters:**
/// Projection is a dimension-reduction technique that lets us work with a smaller witness
/// for efficiency. The projection matrix is structured (e.g., it might be a block matrix
/// with copies of a smaller projection matrix along the diagonal), and we need to prove
/// that applying this matrix to the witness gives the claimed projection image.
///
/// However, we can't just prove the projection row-by-row (too many constraints). Instead,
/// we sample a random linear combination (the projection flattener) of the rows, which
/// gives us a single inner product constraint. This is sound because if the projection
/// is wrong in any entry, it will (with high probability) be detected by the random
/// linear combination.
///
/// **Structure of the Constraint:**
/// The output is a DiffSumcheck that computes:
///   LHS: folded_witness_selector · (recomposed_folded_witness · projection_coeffs)
///   RHS: projection_selector · (recomposed_projection_image · fold_tensor)
///
/// The fold_tensor arises because we're folding multiple witnesses, and the projection
/// image needs to fold compatibly. The tensor product structure ensures that the folded
/// projection image equals the projection of the folded witness.
///
/// **Field Details:**
///   - `lhs_flatter_0_sumcheck`: Block-level flattener (elder variables)
///   - `lhs_flatter_1_times_matrix_sumcheck`: Within-block coefficients (LS variables)
///   - `rhs_sumcheck`: Linear sumcheck loaded with the fold tensor (fold_challenge ⊗
///     projection_flattener).
///   - `projection_selector_sumcheck`: Selector that picks out the projection image's slice
///     from the combined witness.
///   - `output`: The final DiffSumcheck that the verifier checks.
pub struct Type3SumcheckContext {
    pub projection_combiner_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub projection_combiner_constant_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub lhs_flatter_0_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub lhs_flatter_1_times_matrix_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub rhs_fold_challenge_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub rhs_projection_flatter_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub projection_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub output: ElephantCell<DiffSumcheck<RingElement>>,
}

/// Type4 layer sumcheck context: One layer in a recursive commitment tree.
///
/// This structure holds all the sumcheck gadgets needed to verify one internal layer
/// of a recursive commitment. Each layer proves that the commitment at this level is
/// correctly derived from the child layer's commitment via decomposition and composition.
///
/// **Recursive Commitment Background:**
/// A recursive commitment is a tree structure where:
///   - Leaves: commitments to chunks of the original witness
///   - Internal nodes: commitments to the concatenation of their children's commitments
///   - Root: the final public commitment
///
/// Each internal layer needs to prove: "The data at my prefix, when committed with CK,
/// equals the composed version of my child's commitment."
///
/// **Field-by-Field Explanation:**
///
/// - `selector_sumcheck`: Picks out this layer's data slice from the combined witness.
///   This ensures we're only looking at the portion of the witness that belongs to this
///   layer, ignoring all other prefixes.
///
/// - `child_selector_sumcheck`: Picks out the child layer's data slice. This is Some(...)
///   for all non-leaf layers (because there's a child to check against) and None for the
///   leaf layer (which has no child and serves as the base case).
///
/// - `combiner_sumcheck`: Linear sumcheck holding the radix weights for recomposing the
///   child's decomposed value. For example, if we decompose base 2^8 into 4 chunks, this
///   holds [1, 2^8, 2^16, 2^24]. Multiplying the decomposed chunks by these weights and
///   summing gives us the original value. This is Some(...) for non-leaf layers and None
///   for leaf layers (which don't need recomposition).
///
/// - `combiner_constant_sumcheck`: Holds the constant offset that needs to be subtracted
///   after recomposition. This arises from signed-digit representation: to keep decomposed
///   chunks small, we allow negative digits, which introduces a known constant bias that
///   must be corrected. This is Some(...) for non-leaf layers and None for leaf layers.
///
/// - `data_selected_sumcheck`: A product sumcheck that combines the selector with the
///   combined witness to produce "this layer's data, selected and ready to use." This is
///   a helper gadget that gets reused across all CK row checks, so we compute it once and
///   share references.
///
/// - `rhs_sumcheck`: The "right-hand side" of the constraint, representing the composed
///   child commitment. For non-leaf layers, this is a DiffSumcheck that computes:
///     child_selector · (recomposed_child_commitment)
///   For leaf layers, this could be a LinearSumcheck directly if we're checking against
///   a public value. The type is boxed as `dyn HighOrderSumcheckData` to allow for this
///   flexibility.
///
/// - `commitment_sumcheck`: Optional linear sumcheck for the public commitment value at
///   this layer. This is Some(...) for leaf layers (where we check against the public
///   commitment) and None for non-leaf layers (where we check against the child's
///   recomposed commitment instead).
///
/// - `ck_sumchecks`: A vector of linear sumchecks, one per rank. Each holds one row of
///   the commitment key (CK) for this layer. The rank determines how many constraints
///   we're batching together—higher rank means more CK rows, which gives stronger binding
///   but increases proof size.
///
/// - `outputs`: A vector of DiffSumchecks, one per CK row. Each output proves:
///     CK_row_i · selected_data = composed_child_commitment
///   These are the constraints that the verifier will check. All outputs share references
///   to the same underlying selectors, data, and child commitment, so we're efficiently
///   reusing computation across rows.
///
/// **Why So Many Fields?**
/// The complexity arises from the need to handle both non-leaf and leaf layers uniformly
/// while allowing for different constraint types (difference vs. product sumchecks), and
/// from the sharing of sub-computations across multiple CK rows to minimize prover work.
pub struct Type4LayerSumcheckContext {
    pub selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub child_selector_sumcheck: Option<ElephantCell<SelectorEq<RingElement>>>,
    pub combiner_sumcheck: Option<ElephantCell<LinearSumcheck<RingElement>>>,
    pub combiner_constant_sumcheck: Option<ElephantCell<LinearSumcheck<RingElement>>>,
    pub data_selected_sumcheck: ElephantCell<ProductSumcheck<RingElement>>,
    pub rhs_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
    pub commitment_sumcheck: Option<ElephantCell<LinearSumcheck<RingElement>>>,
    pub ck_sumchecks: Vec<ElephantCell<LinearSumcheck<RingElement>>>,
    pub outputs: Vec<ElephantCell<DiffSumcheck<RingElement>>>,
}

/// Type4 output layer sumcheck context: The leaf layer in a recursive commitment tree.
///
/// This structure handles the base case of the recursive commitment tree: the leaf layer
/// that directly checks the public commitment value. Unlike internal layers (which check
/// that a parent layer matches the composed child layer), the leaf layer checks that the
/// selected witness data matches the public commitment when multiplied by the CK.
///
/// **Difference from Internal Layers:**
/// - Internal layers: `CK · selected_data = compose(child_commitment)` (DiffSumcheck)
/// - Leaf layer: `selector · (CK · witness)` should equal public commitment (ProductSumcheck)
///
/// The outputs here are ProductSumchecks rather than DiffSumchecks because we're checking
/// against a known public value (the commitment at the leaf level), not against a composed
/// child commitment.
///
/// **Field Details:**
/// - `selector_sumcheck`: Picks out the leaf layer's data slice from the combined witness.
/// - `ck_sumchecks`: CK rows for the leaf layer, one per rank.
/// - `outputs`: ProductSumchecks that compute `selector · (witness · CK_row)` for each row.
///   The verifier will check these against the public leaf commitments.
pub struct Type4OutputLayerSumcheckContext {
    pub selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub ck_sumchecks: Vec<ElephantCell<LinearSumcheck<RingElement>>>,
    pub outputs: Vec<ElephantCell<ProductSumcheck<RingElement>>>,
}

/// Type4 sumcheck context: Complete recursive commitment verification structure.
///
/// This is the top-level container for all Type4 (recursive commitment) sumchecks.
/// It holds a vector of internal layers plus an output layer for the leaf level.
///
/// **Structure:**
/// - `layers`: A vector of `Type4LayerSumcheckContext`, one per internal layer. The
///   vector is ordered from the outermost (closest to root) to the innermost (closest
///   to leaves). These layers enforce parent-child consistency via DiffSumchecks.
/// - `output_layer`: The leaf layer (`Type4OutputLayerSumcheckContext`) that checks
///   the base level against the public commitment via ProductSumchecks.
///
/// **Why Multiple Type4 Contexts?**
/// Our protocol actually has THREE separate recursive commitment trees:
///   1. Commitment recursion: for the basic commitments to the witness
///   2. Opening recursion: for the opening proofs (which are themselves committed)
///   3. Projection recursion: for the projection images (also committed recursively)
///
/// Each tree has its own recursion config (depth, rank, decomposition base, etc.), so
/// we build three separate `Type4SumcheckContext` instances. The `init_sumcheck` function
/// constructs all three and stores them in `SumcheckContext.type4sumchecks`.
///
/// **Folding:**
/// When the verifier provides a random challenge, we need to fold all layers in all three
/// trees, including both the internal layers and the output layer. The `partial_evaluate_type4`
/// function (in this file) walks through the layers and calls `partial_evaluate` on every
/// constituent sumcheck. This maintains the invariant that after k rounds, every sumcheck
/// has been partially evaluated at the first k verifier challenges.
pub struct Type4SumcheckContext {
    pub layers: Vec<Type4LayerSumcheckContext>,
    pub output_layer: Type4OutputLayerSumcheckContext,
}

/// Type5 sumcheck verifies the inner product between the combined witness and
/// its self-conjugate: `<combined_witness, conjugated_combined_witness>`.
/// This constraint is used to check the norm of the witness, ensuring it stays
/// within acceptable bounds. The sumcheck computes the multilinear extension
/// of both vectors over the boolean hypercube and verifies that their inner
/// product equals the claimed norm value.
pub struct Type5SumcheckContext {
    pub conjugated_combined_witness: ElephantCell<LinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
}

// This is similar to Type3SumcheckContext, but for the type 3 projections.
// is checks if
// c^t ( I ⊗ projection_matrix ) · folded_witness =  c^t projection_image · fold_challenge
// This time
pub struct Type3_1ASumcheckContext {
    pub lhs_flatter_0_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub lhs_flatter_1_times_matrix_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub projection_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub output: ElephantCell<DiffSumcheck<RingElement>>,

    pub lhs_consistency_flatter_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub rhs_consistency_flatter_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub lhs_scalar_consistency_sumcheck: ElephantCell<LinearSumcheck<RingElement>>, // for 1 as to scale over all variables
    pub rhs_scalar_consistency_sumcheck: ElephantCell<LinearSumcheck<RingElement>>, // for e
    pub output_2: ElephantCell<DiffSumcheck<RingElement>>,
}

// see test_const_term_relation_to_prove
// pub struct Type3_1BSumcheckContext {
//     pub lhs_flatter_0_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
//     pub lhs_flatter_1_times_matrix_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
//     pub projection_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
//     pub output: ElephantCell<DiffSumcheck<RingElement>>,

// }

pub struct Type3_1ASumcheckContextWrapper {
    pub sumchecks: [Type3_1ASumcheckContext; NOF_BATCHES],
    // pub sumchecks_b: [Type3_1BSumcheckContext; NOF_BATCHES],
    pub projection_combiner_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub projection_combiner_constant_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub rhs_fold_challenge_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
}

fn partial_evaluate_type4(ctx: &mut Type4SumcheckContext, r: &RingElement) {
    for layer in ctx.layers.iter_mut() {
        layer.selector_sumcheck.borrow_mut().partial_evaluate(r);
        if let Some(child_sel) = &layer.child_selector_sumcheck {
            child_sel.borrow_mut().partial_evaluate(r);
        }
        if let Some(comb) = &layer.combiner_sumcheck {
            comb.borrow_mut().partial_evaluate(r);
        }
        if let Some(comb_const) = &layer.combiner_constant_sumcheck {
            comb_const.borrow_mut().partial_evaluate(r);
        }
        if let Some(commitment_sumcheck) = &layer.commitment_sumcheck {
            commitment_sumcheck.borrow_mut().partial_evaluate(r);
        }
        for ck in layer.ck_sumchecks.iter() {
            ck.borrow_mut().partial_evaluate(r);
        }
    }

    // Fold the output (leaf) layer
    ctx.output_layer
        .selector_sumcheck
        .borrow_mut()
        .partial_evaluate(r);
    for ck in ctx.output_layer.ck_sumchecks.iter() {
        ck.borrow_mut().partial_evaluate(r);
    }
}
