use crate::{
    common::ring_arithmetic::{QuadraticExtension, RingElement},
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

pub struct Type0IntermediateVerifierContext {
    pub output: ElephantCell<ProductSumcheckEvaluation>,
}

pub struct Type1IntermediateVerifierContext {
    pub inner_evaluation: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheckEvaluation>,
}

pub struct Type5IntermediateVerifierContext {
    pub conjugated_witness_evaluation: ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheckEvaluation>,
}

pub struct IntermediateVerifierSumcheckContext {
    pub witness_evaluation: ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    pub conjugated_witness_evaluation: ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    pub witness_combiner_evaluation: ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
    pub commitment_key_rows_evaluation:
        Vec<ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>>,
    pub type0evaluations: Vec<Type0IntermediateVerifierContext>,
    pub type1evaluations: Vec<Type1IntermediateVerifierContext>,
    pub type5evaluation: Type5IntermediateVerifierContext,
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
