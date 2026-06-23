use crate::{
    common::{config::NOF_BATCHES, ring_arithmetic::RingElement},
    protocol::{
        intermediate_sumchecks::context::IntermediateSumcheckContext,
        sumcheck_utils::{
            combiner::Combiner,
            common::SumcheckBaseData,
            elephant_cell::ElephantCell,
            factored_diff::FactoredDiffSumcheck,
            linear::LinearSumcheck,
            product::ProductSumcheck,
            ring_to_field_combiner::RingToFieldCombiner,
            selector_eq::SelectorEq,
        },
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
    pub folded_witness_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub folded_witness_combiner_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub folding_challenges_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub basic_commitment_combiner_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub commitment_key_rows_sumcheck: Vec<ElephantCell<LinearSumcheck<RingElement>>>,
    pub opening_combiner_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub commitment_fold_sumchecks: Vec<CommitmentFoldSumcheckContext>,
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
        self.folded_witness_selector_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.folded_witness_combiner_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.folding_challenges_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.basic_commitment_combiner_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        for ck_row_sc in self.commitment_key_rows_sumcheck.iter() {
            ck_row_sc.borrow_mut().partial_evaluate(r);
        }
        for commitment_fold_sc in self.commitment_fold_sumchecks.iter() {
            commitment_fold_sc
                .basic_commitment_row_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }
        self.opening_combiner_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        for inner_eval_fold_sc in self.inner_eval_fold_sumchecks.iter() {
            inner_eval_fold_sc
                .inner_evaluation_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            inner_eval_fold_sc
                .opening_selector_sumcheck
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
                .projection_combiner_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
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
            coarse_proj_sc
                .projection_selector_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }

        if let Some(fine_proj_sumchecks) = &mut self.fine_proj_sumchecks {
            fine_proj_sumchecks
                .projection_combiner_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            fine_proj_sumchecks
                .rhs_fold_challenge_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            fine_proj_sumchecks
                .lhs_scalar_consistency_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            fine_proj_sumchecks
                .projection_constant_terms_embedded_selector_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            fine_proj_sumchecks
                .projection_constant_terms_embedded_combiner_sumcheck
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
                    .projection_selector_sumcheck
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
        for norm_check_sc in self.norm_check_sumcheck.selectors.iter() {
            norm_check_sc.borrow_mut().partial_evaluate(r);
        }
    }
}

/// CommitmentFold: Basic commitment correctness constraint.
///
/// Proves: `CK · folded_witness = commitment · fold_challenge`
/// where folded_witness is recomposed from decomposed chunks.
///
/// Output DiffSumcheck computes:
///   LHS: selector · (recomposed_folded_witness · CK_row)
///   RHS: commitment_selector · (recomposed_commitment · fold_challenge)
pub struct CommitmentFoldSumcheckContext {
    pub basic_commitment_row_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub output: ElephantCell<FactoredDiffSumcheck<RingElement>>,
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
    pub opening_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub output: ElephantCell<FactoredDiffSumcheck<RingElement>>,
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
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
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
    pub projection_combiner_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub lhs_flatter_0_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub lhs_flatter_1_times_matrix_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub rhs_fold_challenge_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub rhs_projection_flatter_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub projection_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub output: ElephantCell<FactoredDiffSumcheck<RingElement>>,
}

/// ComVerify layer: One layer in a recursive commitment tree.
///
/// For each internal layer i, proves: `CK_i · selected_witness_i = compose(child_commitment_{i+1})`
///
/// Key fields:
/// - `selector_sumcheck`, `child_selector_sumcheck`: select layer and child data slices
/// - `ck_sumchecks`: commitment key rows (one per rank)
/// - `outputs`: DiffSumchecks proving the constraint for each CK row
pub struct ComVerifyLayerSumcheckContext {
    pub selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub child_selector_sumcheck: Option<Vec<ElephantCell<SelectorEq<RingElement>>>>,
    pub combiner_sumcheck: Option<ElephantCell<LinearSumcheck<RingElement>>>,
    pub data_selected_sumcheck: ElephantCell<ProductSumcheck<RingElement>>,
    pub commitment_sumcheck: Option<ElephantCell<LinearSumcheck<RingElement>>>,
    pub ck_sumchecks: Vec<ElephantCell<LinearSumcheck<RingElement>>>,
    pub outputs: Vec<ElephantCell<FactoredDiffSumcheck<RingElement>>>,
}

/// ComVerify output layer: Leaf layer checking `selector · (CK · witness) = public_commitment`.
///
/// Uses ProductSumchecks (not DiffSumchecks) since we check against a known public value.
pub struct ComVerifyOutputLayerSumcheckContext {
    pub selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub ck_sumchecks: Vec<ElephantCell<LinearSumcheck<RingElement>>>,
    pub outputs: Vec<ElephantCell<ProductSumcheck<RingElement>>>,
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
    pub selectors: Vec<ElephantCell<SelectorEq<RingElement>>>,
    pub output_2: ElephantCell<ProductSumcheck<RingElement>>,
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
    pub projection_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub output: ElephantCell<FactoredDiffSumcheck<RingElement>>,

    pub lhs_consistency_flatter_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub rhs_consistency_flatter_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub rhs_scalar_consistency_sumcheck: ElephantCell<LinearSumcheck<RingElement>>, // for e
    pub output_2: ElephantCell<FactoredDiffSumcheck<RingElement>>,
}

/// Wrapper for multiple FineProj sumchecks (one per batch) with shared combiners.
///
/// Contains `NOF_BATCHES` FineProj contexts plus shared sumchecks for recomposition
/// (combiner, constant) and constant term embeddings used across all batches.
pub struct FineProjSumcheckContextWrapper {
    pub sumchecks: [FineProjSumcheckContext; NOF_BATCHES],
    pub projection_combiner_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub projection_constant_terms_embedded_combiner_sumcheck:
        ElephantCell<LinearSumcheck<RingElement>>,
    pub rhs_fold_challenge_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub projection_constant_terms_embedded_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub lhs_scalar_consistency_sumcheck: ElephantCell<LinearSumcheck<RingElement>>, // for 1 as to scale over all variables
}

fn partial_evaluate_com_verify(ctx: &mut ComVerifySumcheckContext, r: &RingElement) {
    for layer in ctx.layers.iter_mut() {
        layer.selector_sumcheck.borrow_mut().partial_evaluate(r);
        if let Some(child_sel) = &layer.child_selector_sumcheck {
            for sel in child_sel.iter() {
                sel.borrow_mut().partial_evaluate(r);
            }
        }
        if let Some(comb) = &layer.combiner_sumcheck {
            comb.borrow_mut().partial_evaluate(r);
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
