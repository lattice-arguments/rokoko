use crate::{
    common::{
        config::NOF_BATCHES,
        ring_arithmetic::{QuadraticExtension, RingElement},
    },
    protocol::sumcheck_utils::{
        combiner::CombinerEvaluation,
        elephant_cell::ElephantCell,
        linear::{
            BasicEvaluationLinearSumcheck, FakeEvaluationLinearSumcheck,
            StructuredRowEvaluationLinearSumcheck,
        },
        product::ProductSumcheckEvaluation,
        ring_to_field_combiner::RingToFieldCombinerEvaluation,
    },
};

pub struct CommitmentFoldIntermediateVerifierContext {
    pub output: ElephantCell<ProductSumcheckEvaluation>,
}

pub struct InnerEvalFoldIntermediateVerifierContext {
    pub inner_evaluation: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheckEvaluation>,
}

pub struct FineProjIntermediateVerifierContext {
    pub c_0_evaluation: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub j_batched_evaluation: ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheckEvaluation>,
}

pub struct NormCheckIntermediateVerifierContext {
    pub conjugated_witness_evaluation: ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheckEvaluation>,
}

pub struct IntermediateVerifierSumcheckContext {
    pub witness_evaluation: ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    pub conjugated_witness_evaluation: ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    pub witness_combiner_evaluation: ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
    pub commitment_key_rows_evaluation:
        Vec<ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>>,
    pub commitment_fold_evaluations: Vec<CommitmentFoldIntermediateVerifierContext>,
    pub inner_eval_fold_evaluations: Vec<InnerEvalFoldIntermediateVerifierContext>,
    pub fine_proj_evaluations: [FineProjIntermediateVerifierContext; NOF_BATCHES],
    pub norm_check_evaluation: NormCheckIntermediateVerifierContext,
    pub combiner_evaluation: ElephantCell<CombinerEvaluation<RingElement>>,
    pub field_combiner_evaluation: ElephantCell<RingToFieldCombinerEvaluation>,
    pub next: Option<Box<IntermediateVerifierSumcheckContext>>,
}

impl IntermediateVerifierSumcheckContext {
    pub fn evaluate_at_point(&mut self, point: &Vec<RingElement>) -> QuadraticExtension {
        self.field_combiner_evaluation
            .borrow_mut()
            .evaluate_at_ring_point(point)
            .clone()
    }
}
