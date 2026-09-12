use crate::{
    common::{config::NOF_BATCHES, ring_arithmetic::RingElement},
    protocol::{
        intermediate_sumchecks::context::IntermediateSumcheckContext,
        sumcheck_utils::{
            combiner::Combiner,
            common::{HighOrderSumcheckData, SumcheckBaseData},
            diff::DiffSumcheck,
            elephant_cell::ElephantCell,
            linear::LinearSumcheck,
            product::ProductSumcheck,
            ring_to_field_combiner::RingToFieldCombiner,
            selector_eq::SelectorEq,
        },
        sumchecks::helpers::WeightedRecomposition,
    },
};

/// All sumchecks for constraint verification, grouped for consistent folding.
/// Each type verifies a different constraint (commitment correctness, opening
/// consistency, projection validity, recursive structure, witness norm).
/// Folding with a verifier challenge updates all constraints via `partial_evaluate_all`.
///
/// Note: `coarse_proj_sumcheck` and `fine_proj_sumchecks` are mutually exclusive - only one is used
pub struct SumcheckContext {
    pub combined_witness_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    /// Every selector the round uses, each registered once at construction and folded once per
    /// round. Selectors are shared across constraints, so they are folded from here rather than
    /// from the constraint contexts.
    pub selectors: Vec<ElephantCell<SelectorEq<RingElement>>>,
    /// The radix weights of the recompositions that carry them on a factor of their own, one
    /// per such recomposition.
    pub recomposition_weights: Vec<ElephantCell<LinearSumcheck<RingElement>>>,
    pub folding_challenges_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub commitment_fold_sumcheck: CommitmentFoldSumcheckContext,
    pub inner_eval_fold_sumchecks: Vec<InnerEvalFoldSumcheckContext>,
    pub outer_eval_claim_sumchecks: Vec<OuterEvalClaimSumcheckContext>,
    pub coarse_proj_sumcheck: Option<CoarseProjSumcheckContext>,
    pub com_verify_sumchecks: Vec<ComVerifySumcheckContext>,
    pub norm_check_sumcheck: NormCheckSumcheckContext,
    pub fine_proj_sumchecks: Option<FineProjSumcheckContextWrapper>, // it should never go together with coarse_proj_sumcheck, left as option for easier handling
    pub combiner: ElephantCell<Combiner<RingElement>>,
    pub field_combiner: ElephantCell<RingToFieldCombiner>,
    pub next: Option<Box<NextSumcheckContext>>,
}

pub enum NextSumcheckContext {
    Simple(SumcheckContext),
    Intermediate(IntermediateSumcheckContext),
}

impl SumcheckContext {
    pub fn partial_evaluate_all(&mut self, r: &RingElement) {
        self.combined_witness_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        for selector in self.selectors.iter() {
            selector.borrow_mut().partial_evaluate(r);
        }
        for weights in self.recomposition_weights.iter() {
            weights.borrow_mut().partial_evaluate(r);
        }
        self.folding_challenges_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.commitment_fold_sumcheck
            .combined_commitment_key_row
            .borrow_mut()
            .partial_evaluate(r);
        for inner_eval_fold_sc in self.inner_eval_fold_sumchecks.iter() {
            inner_eval_fold_sc
                .inner_evaluation_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }
        for outer_eval_claim_sc in self.outer_eval_claim_sumchecks.iter() {
            outer_eval_claim_sc
                .outer_evaluation_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }

        if let Some(coarse_proj_sc) = &mut self.coarse_proj_sumcheck {
            coarse_proj_sc
                .lhs_flatter_0_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            coarse_proj_sc
                .lhs_flatter_1_times_matrix_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            coarse_proj_sc
                .rhs_fold_challenge_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            coarse_proj_sc
                .rhs_projection_flatter_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }

        if let Some(fine_proj_sumchecks) = &mut self.fine_proj_sumchecks {
            fine_proj_sumchecks
                .rhs_fold_challenge_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            fine_proj_sumchecks
                .lhs_scalar_consistency_sumcheck
                .borrow_mut()
                .partial_evaluate(r);

            for fine_proj_sc in fine_proj_sumchecks.sumchecks.iter_mut() {
                fine_proj_sc
                    .lhs_flatter_0_sumcheck
                    .borrow_mut()
                    .partial_evaluate(r);
                fine_proj_sc
                    .lhs_flatter_1_times_matrix_sumcheck
                    .borrow_mut()
                    .partial_evaluate(r);
                fine_proj_sc
                    .lhs_consistency_flatter_sumcheck
                    .borrow_mut()
                    .partial_evaluate(r);
                fine_proj_sc
                    .rhs_consistency_flatter_sumcheck
                    .borrow_mut()
                    .partial_evaluate(r);
                fine_proj_sc
                    .rhs_scalar_consistency_sumcheck
                    .borrow_mut()
                    .partial_evaluate(r);
            }
        }

        for com_verify_sc in self.com_verify_sumchecks.iter_mut() {
            partial_evaluate_com_verify(com_verify_sc, r);
        }
        self.norm_check_sumcheck
            .conjugated_combined_witness
            .borrow_mut()
            .partial_evaluate(r);
    }
}

/// CommitmentFold: Basic commitment correctness constraint, over all `basic_commitment_rank` rows
/// at once.
///
/// Proves: `SUM_i w_i . (CK_i · folded_witness - commitment_i · fold_challenge) = 0`
///
/// Output DiffSumcheck computes:
///   LHS: block-weighted recomposed folded witness · (witness · combined CK row)
///   RHS: (SUM_i w_i . recomposed commitment_i) · (witness · fold_challenge)
pub struct CommitmentFoldSumcheckContext {
    /// `SUM_j w_row(j) . K_j` over the key rows one block is committed with.
    pub combined_commitment_key_row: ElephantCell<LinearSumcheck<RingElement>>,
    /// The folded witness recomposed per block, carrying the block weights.
    pub folded_witness_blocks: WeightedRecomposition,
    /// One recomposition per commitment row, each carrying that row's weight.
    pub basic_commitment_rows: Vec<WeightedRecomposition>,
    pub output: ElephantCell<DiffSumcheck<RingElement>>,
}

/// InnerEvalFold: Inner evaluation point consistency for openings.
///
/// Proves: `<inner_evaluation_points, folded_witness> = opening.rhs · fold_challenge`
///
/// Output DiffSumcheck:
///   LHS: folded_witness_selector · (recomposed_folded_witness · inner_eval_points)
///   RHS: opening_selector · (recomposed_opening_rhs · fold_challenge)
pub struct InnerEvalFoldSumcheckContext {
    pub inner_evaluation_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub output: ElephantCell<DiffSumcheck<RingElement>>,
}

/// OuterEvalClaim: Outer evaluation point consistency for openings (`T` in a paper)
///
/// Proves: `<outer_evaluation_points, opening.rhs> = claimed_evaluation` (public)
///
/// Output ProductSumcheck:
///   opening_selector · (recomposed_opening_rhs · outer_eval_points)
///
/// This is a product (not difference) since the result equals the public claimed_evaluation.
pub struct OuterEvalClaimSumcheckContext {
    pub outer_evaluation_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub output: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
}

/// CoarseProj: Projection image consistency constraint.
///
/// Proves: `<projection_coeffs, folded_witness> = <fold_tensor, projection_image>`
///
/// Output DiffSumcheck:
///   LHS: folded_witness_selector · (recomposed_folded_witness · projection_coeffs)
///   RHS: projection_selector · (recomposed_projection_image · fold_tensor)
///
/// projection_coeffs is derived from the projection matrix and a random flattening point.
/// fold_tensor = fold_challenge ⊗ projection_flattener ensures fold-then-project commutativity.
pub struct CoarseProjSumcheckContext {
    pub lhs_flatter_0_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub lhs_flatter_1_times_matrix_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub rhs_fold_challenge_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub rhs_projection_flatter_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub output: ElephantCell<DiffSumcheck<RingElement>>,
}

/// The key segments of one level, each tagged with the `(slices, slice)` that cuts it out of the
/// level's combined key row.
pub type KeySegments = Vec<(usize, usize, ElephantCell<LinearSumcheck<RingElement>>)>;

/// The block each placed piece of a level falls in, paired with the selector that carries the
/// block's weight.
pub type PieceSelectors = Vec<(usize, ElephantCell<SelectorEq<RingElement>>)>;

/// ComVerify layer: One layer in a recursive commitment tree, with its `rank` rows batched into
/// one constraint.
///
/// Proves: `SUM_i w_i . (CK_i · selected_witness_i - compose(child_commitment)_i) = 0`
///
/// Key fields:
/// - `blocks`, `block_len`, `blockwise_rank`: the shape the level's combined key row is built at
/// - `key_segments`: that row cut into the slices the placed pieces meet
/// - `piece_selectors`: the piece selectors, which carry the block weights
/// - `child`: the child's recomposition, carrying the row weights
pub struct ComVerifyLayerSumcheckContext {
    pub blocks: usize,
    pub block_len: usize,
    pub blockwise_rank: usize,
    pub piece_selectors: PieceSelectors,
    pub key_segments: KeySegments,
    pub child: WeightedRecomposition,
    pub output: ElephantCell<DiffSumcheck<RingElement>>,
}

/// ComVerify output layer: Leaf layer checking `SUM_i w_i . (CK_i · witness) = SUM_i w_i . rc_i`.
///
/// A ProductSumcheck tree (not a DiffSumcheck) since we check against a known public value.
pub struct ComVerifyOutputLayerSumcheckContext {
    pub blocks: usize,
    pub block_len: usize,
    pub blockwise_rank: usize,
    pub piece_selectors: PieceSelectors,
    pub key_segments: KeySegments,
    pub output: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
}

/// ComVerify: Complete recursive commitment verification structure.
///
/// Contains internal layers (parent-child consistency) and output layer (anchors to public commitment).
/// The protocol has three separate recursive trees: commitment, opening, and projection recursions.
pub struct ComVerifySumcheckContext {
    pub layers: Vec<ComVerifyLayerSumcheckContext>,
    pub output_layer: ComVerifyOutputLayerSumcheckContext,
}

/// NormCheck: Witness norm check via `<combined_witness, conjugated_combined_witness> = norm_claim`.
pub struct NormCheckSumcheckContext {
    pub conjugated_combined_witness: ElephantCell<LinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheck<RingElement>>,

    // we also give an opening to subvectors of the combined witness and its conjugate.
    pub output_2: ElephantCell<ProductSumcheck<RingElement>>,

    /// The same inner product scoped to the projection recursion's level-0 placements, present
    /// when the round's config asks for the exact projection-image norm.
    pub output_3: Option<ElephantCell<ProductSumcheck<RingElement>>>,
}

/// FineProj: fine (coefficient-level) projection validity (paper: Pi^proj-f).
///
/// Proves: `c^T (I ⊗ J) · folded_witness = c^T projection_image · fold_challenge`
/// over the coefficient embedding, via the trace-dual / constant-term trick.
///
/// Two outputs:
/// - `output`: main projection constraint
/// - `output_2`: consistency between the constant-term commitment and the
///   batched-projection commitment (paper: trace(r_i) = 0 checks)
pub struct FineProjSumcheckContext {
    pub lhs_flatter_0_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub lhs_flatter_1_times_matrix_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub output: ElephantCell<DiffSumcheck<RingElement>>,

    pub lhs_consistency_flatter_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub rhs_consistency_flatter_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub rhs_scalar_consistency_sumcheck: ElephantCell<LinearSumcheck<RingElement>>, // for e
    pub output_2: ElephantCell<DiffSumcheck<RingElement>>,
}

/// Wrapper for multiple FineProj sumchecks (one per batch) with shared combiners.
///
/// Contains `NOF_BATCHES` FineProj contexts plus shared sumchecks for recomposition
/// (combiner, constant) and constant term embeddings used across all batches.
pub struct FineProjSumcheckContextWrapper {
    pub sumchecks: [FineProjSumcheckContext; NOF_BATCHES],
    pub rhs_fold_challenge_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub lhs_scalar_consistency_sumcheck: ElephantCell<LinearSumcheck<RingElement>>, // for 1 as to scale over all variables
}

fn partial_evaluate_com_verify(ctx: &mut ComVerifySumcheckContext, r: &RingElement) {
    for layer in ctx.layers.iter_mut() {
        for (_, _, ck) in layer.key_segments.iter() {
            ck.borrow_mut().partial_evaluate(r);
        }
    }

    for (_, _, ck) in ctx.output_layer.key_segments.iter() {
        ck.borrow_mut().partial_evaluate(r);
    }
}
