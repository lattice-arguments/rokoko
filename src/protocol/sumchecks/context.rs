use crate::{
    common::{config::NOF_BATCHES, ring_arithmetic::RingElement},
    protocol::sumcheck_utils::{
        combiner::Combiner, common::SumcheckBaseData, diff::DiffSumcheck,
        elephant_cell::ElephantCell, linear::LinearSumcheck, product::ProductSumcheck,
        ring_to_field_combiner::RingToFieldCombiner, selector_eq::SelectorEq,
    },
};

/// All sumchecks for constraint verification, grouped for consistent folding.
/// Each type verifies a different constraint (commitment correctness, opening
/// consistency, projection validity, recursive structure, witness norm).
/// Folding with a verifier challenge updates all constraints via `partial_evaluate_all`.
///
/// Note: `type3sumcheck` and `type3_1_sumchecks` are mutually exclusive - only one is used
pub struct SumcheckContext {
    pub combined_witness_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub folded_witness_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub folded_witness_combiner_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub folding_challenges_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub basic_commitment_combiner_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub commitment_key_rows_sumcheck: Vec<ElephantCell<LinearSumcheck<RingElement>>>,
    pub opening_combiner_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub type0sumchecks: Vec<Type0SumcheckContext>,
    pub type1sumchecks: Vec<Type1SumcheckContext>,
    pub type2sumchecks: Vec<Type2SumcheckContext>,
    pub type3sumcheck: Option<Type3SumcheckContext>,
    pub type4sumchecks: Vec<Type4SumcheckContext>,
    pub type5sumcheck: Type5SumcheckContext,
    pub type3_1_sumchecks: Option<Type3_1SumcheckContextWrapper>, // it should never go together with type3sumcheck, left as option for easier handling
    pub combiner: ElephantCell<Combiner<RingElement>>,
    pub field_combiner: ElephantCell<RingToFieldCombiner>,
    pub next: Option<Box<SumcheckContext>>,
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
        for type0_sc in self.type0sumchecks.iter() {
            type0_sc
                .basic_commitment_row_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            type0_sc
                .folded_witness_block_selector_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }
        self.opening_combiner_sumcheck
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

        if let Some(type3_1_sumchecks) = &mut self.type3_1_sumchecks {
            type3_1_sumchecks
                .projection_combiner_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            type3_1_sumchecks
                .rhs_fold_challenge_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            type3_1_sumchecks
                .lhs_scalar_consistency_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            type3_1_sumchecks
                .projection_constant_terms_embedded_selector_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            type3_1_sumchecks
                .projection_constant_terms_embedded_combiner_sumcheck
                .borrow_mut()
                .partial_evaluate(r);

            for type3_1_sc in type3_1_sumchecks.sumchecks.iter_mut() {
                type3_1_sc
                    .lhs_flatter_0_sumcheck
                    .borrow_mut()
                    .partial_evaluate(r);
                type3_1_sc
                    .lhs_flatter_1_times_matrix_sumcheck
                    .borrow_mut()
                    .partial_evaluate(r);
                type3_1_sc
                    .projection_selector_sumcheck
                    .borrow_mut()
                    .partial_evaluate(r);
                type3_1_sc
                    .lhs_consistency_flatter_sumcheck
                    .borrow_mut()
                    .partial_evaluate(r);
                type3_1_sc
                    .rhs_consistency_flatter_sumcheck
                    .borrow_mut()
                    .partial_evaluate(r);
                type3_1_sc
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
        for type5_sc in self.type5sumcheck.selectors.iter() {
            type5_sc.borrow_mut().partial_evaluate(r);
        }
    }
}

/// Type0: Basic commitment correctness constraint.
///
/// Proves: `CK · folded_witness = commitment · fold_challenge`
/// where folded_witness is recomposed from decomposed chunks.
///
/// Output DiffSumcheck computes:
///   LHS: selector · (recomposed_folded_witness · CK_row)
///   RHS: commitment_selector · (recomposed_commitment · fold_challenge)
pub struct Type0SumcheckContext {
    pub folded_witness_block_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub basic_commitment_row_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub output: ElephantCell<DiffSumcheck<RingElement>>,
}

/// Type1: Inner evaluation point consistency for openings.
///
/// Proves: `<inner_evaluation_points, folded_witness> = opening.rhs · fold_challenge`
///
/// Output DiffSumcheck:
///   LHS: folded_witness_selector · (recomposed_folded_witness · inner_eval_points)
///   RHS: opening_selector · (recomposed_opening_rhs · fold_challenge)
pub struct Type1SumcheckContext {
    pub inner_evaluation_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub opening_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub output: ElephantCell<DiffSumcheck<RingElement>>,
}

/// Type2: Outer evaluation point consistency for openings (`T` in a paper)
///
/// Proves: `<outer_evaluation_points, opening.rhs> = claimed_evaluation` (public)
///
/// Output ProductSumcheck:
///   opening_selector · (recomposed_opening_rhs · outer_eval_points)
///
/// This is a product (not difference) since the result equals the public claimed_evaluation.
pub struct Type2SumcheckContext {
    pub outer_evaluation_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
}

/// Type3: Projection image consistency constraint.
///
/// Proves: `<projection_coeffs, folded_witness> = <fold_tensor, projection_image>`
///
/// Output DiffSumcheck:
///   LHS: folded_witness_selector · (recomposed_folded_witness · projection_coeffs)
///   RHS: projection_selector · (recomposed_projection_image · fold_tensor)
///
/// projection_coeffs is derived from the projection matrix and a random flattening point.
/// fold_tensor = fold_challenge ⊗ projection_flattener ensures fold-then-project commutativity.
pub struct Type3SumcheckContext {
    pub projection_combiner_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub lhs_flatter_0_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub lhs_flatter_1_times_matrix_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub rhs_fold_challenge_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub rhs_projection_flatter_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub projection_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub output: ElephantCell<DiffSumcheck<RingElement>>,
}

/// Type4 layer: One layer in a recursive commitment tree.
///
/// For each internal layer i, proves: `CK_i · selected_witness_i = compose(child_commitment_{i+1})`
///
/// Key fields:
/// - `selector_sumcheck`, `child_selector_sumcheck`: select layer and child data slices
/// - `ck_sumchecks`: commitment key rows (one per rank)
/// - `outputs`: DiffSumchecks proving the constraint for each CK row
/// while allowing for different constraint types (difference vs. product sumchecks), and
/// from the sharing of sub-computations across multiple CK rows to minimize prover work.
pub struct Type4LayerSumcheckContext {
    pub selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub child_selector_sumcheck: Option<Vec<ElephantCell<SelectorEq<RingElement>>>>,
    pub block_selector_sumchecks: Option<Vec<ElephantCell<SelectorEq<RingElement>>>>,
    pub combiner_sumcheck: Option<ElephantCell<LinearSumcheck<RingElement>>>,
    pub data_selected_sumcheck: ElephantCell<ProductSumcheck<RingElement>>,
    // pub rhs_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
    pub commitment_sumcheck: Option<ElephantCell<LinearSumcheck<RingElement>>>,
    pub ck_sumchecks: Vec<ElephantCell<LinearSumcheck<RingElement>>>,
    pub outputs: Vec<ElephantCell<DiffSumcheck<RingElement>>>,
}

/// Type4 output layer: Leaf layer checking `selector · (CK · witness) = public_commitment`.
///
/// Uses ProductSumchecks (not DiffSumchecks) since we check against a known public value.
pub struct Type4OutputLayerSumcheckContext {
    pub selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub ck_sumchecks: Vec<ElephantCell<LinearSumcheck<RingElement>>>,
    pub outputs: Vec<ElephantCell<ProductSumcheck<RingElement>>>,
}

/// Type4: Complete recursive commitment verification structure.
///
/// Contains internal layers (parent-child consistency) and output layer (anchors to public commitment).
/// The protocol has three separate recursive trees: commitment, opening, and projection recursions.
pub struct Type4SumcheckContext {
    pub layers: Vec<Type4LayerSumcheckContext>,
    pub output_layer: Type4OutputLayerSumcheckContext,
}

/// Type5: Witness norm check via `<combined_witness, conjugated_combined_witness> = norm_claim`.
pub struct Type5SumcheckContext {
    pub conjugated_combined_witness: ElephantCell<LinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheck<RingElement>>,

    // we also give an opening to subvectors of the combined witness and its conjugate.
    pub selectors: Vec<ElephantCell<SelectorEq<RingElement>>>,
    pub output_2: ElephantCell<ProductSumcheck<RingElement>>,
}

/// Type3_1: Projection validity constraint using Kronecker product structure.
///
/// Proves: `c^T (I ⊗ projection_matrix) · folded_witness = c^T projection_image · fold_challenge`
///
/// This is an alternative to Type3 that uses a Kronecker product structure (I ⊗ P) instead of
/// a block-diagonal structure. Used when the projection matrix has this specific form.
///
/// Contains two outputs:
/// - `output`: main projection constraint
/// - `output_2`: consistency check for constant terms
/// the norm is embedded in ct which are back embedded in ring.
pub struct Type3_1SumcheckContext {
    pub lhs_flatter_0_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub lhs_flatter_1_times_matrix_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub projection_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub output: ElephantCell<DiffSumcheck<RingElement>>,

    pub lhs_consistency_flatter_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub rhs_consistency_flatter_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub rhs_scalar_consistency_sumcheck: ElephantCell<LinearSumcheck<RingElement>>, // for e
    pub output_2: ElephantCell<DiffSumcheck<RingElement>>,
}

/// Wrapper for multiple Type3_1 sumchecks (one per batch) with shared combiners.
///
/// Contains `NOF_BATCHES` Type3_1 contexts plus shared sumchecks for recomposition
/// (combiner, constant) and constant term embeddings used across all batches.
pub struct Type3_1SumcheckContextWrapper {
    pub sumchecks: [Type3_1SumcheckContext; NOF_BATCHES],
    pub projection_combiner_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub projection_constant_terms_embedded_combiner_sumcheck:
        ElephantCell<LinearSumcheck<RingElement>>,
    pub rhs_fold_challenge_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub projection_constant_terms_embedded_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub lhs_scalar_consistency_sumcheck: ElephantCell<LinearSumcheck<RingElement>>, // for 1 as to scale over all variables
}

fn partial_evaluate_type4(ctx: &mut Type4SumcheckContext, r: &RingElement) {
    for layer in ctx.layers.iter_mut() {
        layer.selector_sumcheck.borrow_mut().partial_evaluate(r);
        if let Some(child_sel) = &layer.child_selector_sumcheck {
            for sel in child_sel.iter() {
                sel.borrow_mut().partial_evaluate(r);
            }
        }
        if let Some(block_sels) = &layer.block_selector_sumchecks {
            for sel in block_sels.iter() {
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
