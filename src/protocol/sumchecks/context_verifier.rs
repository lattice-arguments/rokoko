use crate::{
    common::{
        config::NOF_BATCHES,
        ring_arithmetic::{QuadraticExtension, RingElement},
    },
    protocol::{
        intermediate_sumchecks::context_verifier::IntermediateVerifierSumcheckContext,
        sumcheck_utils::{
            combiner::CombinerEvaluation,
            common::EvaluationSumcheckData,
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
        sumchecks::builder_verifier::WeightedRecompositionEvaluation,
    },
};

/// Verifier's sumcheck context - mirrors SumcheckContext but with evaluation-only types.
/// Uses ElephantCell for all evaluations to allow shared ownership, just like the prover.
pub struct VerifierSumcheckContext {
    // Base evaluations (leaf nodes that will be loaded with data)
    pub combined_witness_evaluation: ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    pub folding_challenges_evaluation: ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,

    // Type-specific contexts
    pub commitment_fold_evaluation: CommitmentFoldVerifierContext,
    pub inner_eval_fold_evaluations: Vec<InnerEvalFoldVerifierContext>,
    pub outer_eval_claim_evaluations: Vec<OuterEvalClaimVerifierContext>,
    pub coarse_proj_evaluation: Option<CoarseProjVerifierContext>,
    pub fine_proj_evaluations: Option<FineProjVerifierContextWrapper>,
    pub com_verify_evaluations: Vec<ComVerifyVerifierContext>,
    pub norm_check_evaluation: NormCheckVerifierContext,

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

/// Verifier dual of `CommitmentFoldSumcheckContext`. The combined key row lives inside the
/// output's tree; what the round loads are the weights its scales and recompositions carry.
pub struct CommitmentFoldVerifierContext {
    /// One scale per key row, holding `w_row(j)`.
    pub key_row_scales: Vec<ElephantCell<SelectorEqEvaluation>>,
    pub folded_witness_blocks: WeightedRecompositionEvaluation,
    pub basic_commitment_rows: Vec<WeightedRecompositionEvaluation>,
    pub output: ElephantCell<DiffSumcheckEvaluation>,
}

pub struct InnerEvalFoldVerifierContext {
    pub inner_evaluation: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub output: ElephantCell<DiffSumcheckEvaluation>,
}

pub struct OuterEvalClaimVerifierContext {
    pub outer_evaluation: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub output: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
}

pub struct CoarseProjVerifierContext {
    pub lhs_flatter_0_evaluation: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub lhs_flatter_1_times_matrix_evaluation_field:
        ElephantCell<BasicEvaluationLinearSumcheck<QuadraticExtension>>,
    pub lhs_flatter_1_times_matrix_evaluation: ElephantCell<RingToFieldWrapperEvaluation>,
    // RHS: Split into projection_flatter and fold_challenge
    pub rhs_projection_flatter_evaluation:
        ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub rhs_fold_challenge_evaluation: ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
    pub output: ElephantCell<DiffSumcheckEvaluation>,
}

pub struct FineProjVerifierContext {
    pub lhs_flatter_0_evaluation_field:
        ElephantCell<StructuredRowEvaluationLinearSumcheck<QuadraticExtension>>,
    pub lhs_flatter_0_evaluation: ElephantCell<RingToFieldWrapperEvaluation>,
    pub lhs_flatter_1_times_matrix_evaluation:
        ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
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
pub struct FineProjVerifierContextWrapper {
    pub sumchecks: [FineProjVerifierContext; NOF_BATCHES],
    pub rhs_fold_challenge_evaluation: ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
    pub lhs_scalar_consistency_evaluation_field:
        ElephantCell<BasicEvaluationLinearSumcheck<QuadraticExtension>>,
    pub lhs_scalar_consistency_evaluation: ElephantCell<RingToFieldWrapperEvaluation>,
}

pub struct ComVerifyVerifierContext {
    pub layers: Vec<ComVerifyLayerVerifierContext>,
    pub output_layer: ComVerifyOutputLayerVerifierContext,
}

/// The block each placed piece of a level falls in, paired with the selector that carries the
/// block's weight.
pub type PieceSelectorEvaluations = Vec<(usize, ElephantCell<SelectorEqEvaluation>)>;

pub struct ComVerifyLayerVerifierContext {
    pub blocks: usize,
    pub blockwise_rank: usize,
    pub piece_selectors: PieceSelectorEvaluations,
    pub key_row_scales: Vec<ElephantCell<SelectorEqEvaluation>>,
    pub child: WeightedRecompositionEvaluation,
    pub output: ElephantCell<DiffSumcheckEvaluation>,
}

pub struct ComVerifyOutputLayerVerifierContext {
    pub blocks: usize,
    pub blockwise_rank: usize,
    pub piece_selectors: PieceSelectorEvaluations,
    pub key_row_scales: Vec<ElephantCell<SelectorEqEvaluation>>,
    pub output: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
}

pub struct NormCheckVerifierContext {
    pub conjugated_combined_witness_evaluation:
        ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheckEvaluation>,

    pub selectors: Vec<ElephantCell<SelectorEqEvaluation>>,
    pub output_2: ElephantCell<ProductSumcheckEvaluation>,

    pub output_3: Option<ElephantCell<ProductSumcheckEvaluation>>,
}
