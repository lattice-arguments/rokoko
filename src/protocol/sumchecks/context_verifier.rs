use crate::{
    common::{
        config::NOF_BATCHES,
        ring_arithmetic::{QuadraticExtension, RingElement},
    },
    protocol::{
        intermediate_sumchecks::context_verifier::IntermediateVerifierSumcheckContext,
        sumcheck_utils::{
            combiner::CombinerEvaluation,
            diff::DiffSumcheckEvaluation,
            elephant_cell::ElephantCell,
            linear::{
                BasicEvaluationLinearSumcheck, FakeEvaluationLinearSumcheck,
                RingToFieldWrapperEvaluation, StructuredRowEvaluationLinearSumcheck,
            },
            product::ProductSumcheckEvaluation,
            ring_to_field_combiner::RingToFieldCombinerEvaluation,
            selector_eq::SelectorEqEvaluation,
        },
    },
};

/// Verifier's sumcheck context - mirrors SumcheckContext but with evaluation-only types.
/// Uses ElephantCell for all evaluations to allow shared ownership, just like the prover.
pub struct VerifierSumcheckContext {
    // Base evaluations (leaf nodes that will be loaded with data)
    pub combined_witness_evaluation: ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    pub folded_witness_selector_evaluation: ElephantCell<SelectorEqEvaluation>,
    pub folded_witness_combiner_evaluation:
        ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
    pub folding_challenges_evaluation: ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
    pub basic_commitment_combiner_evaluation:
        ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
    pub commitment_key_rows_evaluation:
        Vec<ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>>,
    pub opening_combiner_evaluation: ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,

    // Type-specific contexts
    pub type0evaluations: Vec<Type0VerifierContext>,
    pub type1evaluations: Vec<Type1VerifierContext>,
    pub type2evaluations: Vec<Type2VerifierContext>,
    pub type3evaluation: Option<Type3VerifierContext>,
    pub type3_1_evaluations: Option<Type3_1VerifierContextWrapper>,
    pub type4evaluations: Vec<Type4VerifierContext>,
    pub type5evaluation: Type5VerifierContext,

    // Top-level combiners
    pub combiner_evaluation: ElephantCell<CombinerEvaluation<RingElement>>,
    pub field_combiner_evaluation: ElephantCell<RingToFieldCombinerEvaluation>,
    pub next: Option<Box<NextVerifierSumcheckContext>>,
}

pub enum NextVerifierSumcheckContext {
    Simple(VerifierSumcheckContext),
    Intermediate(IntermediateVerifierSumcheckContext),
}

impl VerifierSumcheckContext {
    pub fn evaluate_at_point(&mut self, point: &Vec<RingElement>) -> QuadraticExtension {
        self.field_combiner_evaluation
            .borrow_mut()
            .evaluate_at_ring_point(point)
            .clone()
    }
}

pub struct Type0VerifierContext {
    pub basic_commitment_row_evaluation: ElephantCell<SelectorEqEvaluation>,
    pub output: ElephantCell<DiffSumcheckEvaluation>,
}

pub struct Type1VerifierContext {
    pub inner_evaluation: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub opening_selector_evaluation: ElephantCell<SelectorEqEvaluation>,
    pub output: ElephantCell<DiffSumcheckEvaluation>,
}

pub struct Type2VerifierContext {
    pub outer_evaluation: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheckEvaluation>,
}

pub struct Type3VerifierContext {
    pub projection_combiner_evaluation: ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
    pub lhs_flatter_0_evaluation: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub lhs_flatter_1_times_matrix_evaluation_field:
        ElephantCell<BasicEvaluationLinearSumcheck<QuadraticExtension>>,
    pub lhs_flatter_1_times_matrix_evaluation: ElephantCell<RingToFieldWrapperEvaluation>,
    // RHS: Split into projection_flatter and fold_challenge
    pub rhs_projection_flatter_evaluation:
        ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub rhs_fold_challenge_evaluation: ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
    pub projection_selector_evaluation: ElephantCell<SelectorEqEvaluation>,
    pub output: ElephantCell<DiffSumcheckEvaluation>,
}

pub struct Type3_1VerifierContext {
    pub lhs_flatter_0_evaluation_field:
        ElephantCell<StructuredRowEvaluationLinearSumcheck<QuadraticExtension>>,
    pub lhs_flatter_0_evaluation: ElephantCell<RingToFieldWrapperEvaluation>,
    pub lhs_flatter_1_times_matrix_evaluation:
        ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
    pub projection_selector_evaluation: ElephantCell<SelectorEqEvaluation>,
    pub output: ElephantCell<DiffSumcheckEvaluation>,

    pub lhs_consistency_flatter_evaluation_field:
        ElephantCell<StructuredRowEvaluationLinearSumcheck<QuadraticExtension>>,
    pub rhs_consistency_flatter_evaluation_field:
        ElephantCell<StructuredRowEvaluationLinearSumcheck<QuadraticExtension>>,

    pub lhs_consistency_flatter_evaluation: ElephantCell<RingToFieldWrapperEvaluation>,
    pub rhs_consistency_flatter_evaluation: ElephantCell<RingToFieldWrapperEvaluation>,

    pub rhs_scalar_consistency_evaluation: ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,

    pub output_2: ElephantCell<DiffSumcheckEvaluation>,
}
pub struct Type3_1VerifierContextWrapper {
    pub sumchecks: [Type3_1VerifierContext; NOF_BATCHES],
    pub projection_combiner_evaluation: ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
    pub rhs_fold_challenge_evaluation: ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
    pub lhs_scalar_consistency_evaluation_field:
        ElephantCell<BasicEvaluationLinearSumcheck<QuadraticExtension>>,
    pub lhs_scalar_consistency_evaluation: ElephantCell<RingToFieldWrapperEvaluation>,
}

pub struct Type4VerifierContext {
    pub layers: Vec<Type4LayerVerifierContext>,
    pub output_layer: Type4OutputLayerVerifierContext,
}

pub struct Type4LayerVerifierContext {
    pub selector_evaluation: ElephantCell<SelectorEqEvaluation>,
    pub child_selector_evaluations: Vec<ElephantCell<SelectorEqEvaluation>>,
    pub block_selector_evaluations: Option<Vec<ElephantCell<SelectorEqEvaluation>>>,
    pub combiner_evaluation: ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
    pub ck_evaluations: Vec<ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>>,
    pub outputs: Vec<ElephantCell<DiffSumcheckEvaluation>>,
}

pub struct Type4OutputLayerVerifierContext {
    pub selector_evaluation: ElephantCell<SelectorEqEvaluation>,
    pub ck_evaluations: Vec<ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>>,
    pub outputs: Vec<ElephantCell<ProductSumcheckEvaluation>>,
}

pub struct Type5VerifierContext {
    pub conjugated_combined_witness_evaluation:
        ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheckEvaluation>,

    pub selectors: Vec<ElephantCell<SelectorEqEvaluation>>,
    pub output_2: ElephantCell<ProductSumcheckEvaluation>,
}
