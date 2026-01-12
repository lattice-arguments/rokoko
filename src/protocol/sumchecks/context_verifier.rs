use crate::{
    common::ring_arithmetic::{QuadraticExtension, RingElement},
    protocol::sumcheck_utils::{
        diff::DiffSumcheckEvaluation,
        linear::{BasicEvaluationLinearSumcheck, FakeEvaluationLinearSumcheck, StructuredRowEvaluationLinearSumcheck},
        product::ProductSumcheckEvaluation,
        ring_to_field_combiner::RingToFieldCombinerEvaluation,
        selector_eq::SelectorEqEvaluation,
    },
};

/// Verifier's sumcheck context - mirrors SumcheckContext but with evaluation-only types.
/// The structure is IDENTICAL to the prover's context, but uses evaluation types instead.
/// All the sumcheck structure is built upfront - only leaf data nodes are loaded later.
pub struct VerifierSumcheckContext {
    // Base evaluations (leaf nodes that will be loaded with data)
    pub combined_witness_evaluation: FakeEvaluationLinearSumcheck<RingElement>,
    pub folded_witness_selector_evaluation: SelectorEqEvaluation,
    pub folded_witness_combiner_evaluation: BasicEvaluationLinearSumcheck<RingElement>,
    pub witness_combiner_constant_evaluation: BasicEvaluationLinearSumcheck<RingElement>,
    pub folding_challenges_evaluation: BasicEvaluationLinearSumcheck<RingElement>,
    pub basic_commitment_combiner_evaluation: BasicEvaluationLinearSumcheck<RingElement>,
    pub basic_commitment_combiner_constant_evaluation: BasicEvaluationLinearSumcheck<RingElement>,
    pub commitment_key_rows_evaluation: Vec<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub opening_combiner_evaluation: BasicEvaluationLinearSumcheck<RingElement>,
    pub opening_combiner_constant_evaluation: BasicEvaluationLinearSumcheck<RingElement>,
    pub projection_combiner_evaluation: BasicEvaluationLinearSumcheck<RingElement>,
    pub projection_combiner_constant_evaluation: BasicEvaluationLinearSumcheck<RingElement>,
    
    // Type-specific contexts
    pub type0evaluations: Vec<Type0VerifierContext>,
    pub type1evaluations: Vec<Type1VerifierContext>,
    pub type2evaluations: Vec<Type2VerifierContext>,
    pub type3evaluation: Type3VerifierContext,
    pub type4evaluations: [Type4VerifierContext; 3],
    pub type5evaluation: Type5VerifierContext,
    
    // Top-level combiner
    pub field_combiner_evaluation: RingToFieldCombinerEvaluation,
}

impl VerifierSumcheckContext {
    pub fn evaluate_at_point(&mut self, point: &Vec<QuadraticExtension>) -> QuadraticExtension {
        self.field_combiner_evaluation.evaluate(point).clone()
    }
}

pub struct Type0VerifierContext {
    pub basic_commitment_row_evaluation: SelectorEqEvaluation,
    pub output: DiffSumcheckEvaluation,
}

pub struct Type1VerifierContext {
    pub inner_evaluation: BasicEvaluationLinearSumcheck<RingElement>,
    pub opening_selector_evaluation: SelectorEqEvaluation,
    pub output: DiffSumcheckEvaluation,
}

pub struct Type2VerifierContext {
    pub outer_evaluation: BasicEvaluationLinearSumcheck<RingElement>,
    pub output: ProductSumcheckEvaluation,
}

pub struct Type3VerifierContext {
    pub lhs_evaluation: BasicEvaluationLinearSumcheck<RingElement>,
    pub rhs_evaluation: BasicEvaluationLinearSumcheck<RingElement>,
    pub projection_selector_evaluation: SelectorEqEvaluation,
    pub output: DiffSumcheckEvaluation,
}

pub struct Type4VerifierContext {
    pub layers: Vec<Type4LayerVerifierContext>,
    pub output_layer: Type4OutputLayerVerifierContext,
}

pub struct Type4LayerVerifierContext {
    pub selector_evaluation: SelectorEqEvaluation,
    pub child_selector_evaluation: SelectorEqEvaluation,
    pub combiner_evaluation: BasicEvaluationLinearSumcheck<RingElement>,
    pub combiner_constant_evaluation: BasicEvaluationLinearSumcheck<RingElement>,
    pub ck_evaluations: Vec<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub outputs: Vec<DiffSumcheckEvaluation>,
}

pub struct Type4OutputLayerVerifierContext {
    pub selector_evaluation: SelectorEqEvaluation,
    pub ck_evaluations: Vec<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub outputs: Vec<ProductSumcheckEvaluation>,
}

pub struct Type5VerifierContext {
    pub conjugated_combined_witness_evaluation: FakeEvaluationLinearSumcheck<RingElement>,
    pub output: ProductSumcheckEvaluation,
}
