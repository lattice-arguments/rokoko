use std::{cell::RefCell, rc::Rc};

use crate::{
    common::ring_arithmetic::{QuadraticExtension, RingElement},
    protocol::sumcheck_utils::{
        combiner::CombinerEvaluation,
        diff::DiffSumcheckEvaluation,
        linear::{
            BasicEvaluationLinearSumcheck, FakeEvaluationLinearSumcheck,
            StructuredRowEvaluationLinearSumcheck,
        },
        product::ProductSumcheckEvaluation,
        ring_to_field_combiner::RingToFieldCombinerEvaluation,
        selector_eq::SelectorEqEvaluation,
    },
};

/// Verifier's sumcheck context - mirrors SumcheckContext but with evaluation-only types.
/// Uses Rc<RefCell<>> for all evaluations to allow shared ownership, just like the prover.
pub struct VerifierSumcheckContext {
    // Base evaluations (leaf nodes that will be loaded with data)
    pub combined_witness_evaluation: Rc<RefCell<FakeEvaluationLinearSumcheck<RingElement>>>,
    pub folded_witness_selector_evaluation: Rc<RefCell<SelectorEqEvaluation>>,
    pub folded_witness_combiner_evaluation: Rc<RefCell<BasicEvaluationLinearSumcheck<RingElement>>>,
    pub witness_combiner_constant_evaluation:
        Rc<RefCell<BasicEvaluationLinearSumcheck<RingElement>>>,
    pub folding_challenges_evaluation: Rc<RefCell<BasicEvaluationLinearSumcheck<RingElement>>>,
    pub basic_commitment_combiner_evaluation:
        Rc<RefCell<BasicEvaluationLinearSumcheck<RingElement>>>,
    pub basic_commitment_combiner_constant_evaluation:
        Rc<RefCell<BasicEvaluationLinearSumcheck<RingElement>>>,
    pub commitment_key_rows_evaluation:
        Vec<Rc<RefCell<StructuredRowEvaluationLinearSumcheck<RingElement>>>>,
    pub opening_combiner_evaluation: Rc<RefCell<BasicEvaluationLinearSumcheck<RingElement>>>,
    pub opening_combiner_constant_evaluation:
        Rc<RefCell<BasicEvaluationLinearSumcheck<RingElement>>>,
    pub projection_combiner_evaluation: Rc<RefCell<BasicEvaluationLinearSumcheck<RingElement>>>,
    pub projection_combiner_constant_evaluation:
        Rc<RefCell<BasicEvaluationLinearSumcheck<RingElement>>>,

    // Type-specific contexts
    pub type0evaluations: Vec<Type0VerifierContext>,
    pub type1evaluations: Vec<Type1VerifierContext>,
    pub type2evaluations: Vec<Type2VerifierContext>,
    pub type3evaluation: Type3VerifierContext,
    pub type4evaluations: [Type4VerifierContext; 3],
    pub type5evaluation: Type5VerifierContext,

    // Top-level combiners
    pub combiner_evaluation: Rc<RefCell<CombinerEvaluation<RingElement>>>,
    pub field_combiner_evaluation: Rc<RefCell<RingToFieldCombinerEvaluation>>,
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
    pub basic_commitment_row_evaluation: Rc<RefCell<SelectorEqEvaluation>>,
    pub output: Rc<RefCell<DiffSumcheckEvaluation>>,
}

pub struct Type1VerifierContext {
    pub inner_evaluation: Rc<RefCell<StructuredRowEvaluationLinearSumcheck<RingElement>>>,
    pub opening_selector_evaluation: Rc<RefCell<SelectorEqEvaluation>>,
    pub output: Rc<RefCell<DiffSumcheckEvaluation>>,
}

pub struct Type2VerifierContext {
    pub outer_evaluation: Rc<RefCell<StructuredRowEvaluationLinearSumcheck<RingElement>>>,
    pub output: Rc<RefCell<ProductSumcheckEvaluation>>,
}

pub struct Type3VerifierContext {
    // TODO: make strucutred row evaluations
    pub lhs_flatter_0_evaluation: Rc<RefCell<StructuredRowEvaluationLinearSumcheck<RingElement>>>,
    pub lhs_flatter_1_times_matrix_evaluation:
        Rc<RefCell<BasicEvaluationLinearSumcheck<RingElement>>>,
    // RHS: Split into projection_flatter and fold_challenge
    pub rhs_projection_flatter_evaluation:
        Rc<RefCell<StructuredRowEvaluationLinearSumcheck<RingElement>>>,
    pub rhs_fold_challenge_evaluation: Rc<RefCell<BasicEvaluationLinearSumcheck<RingElement>>>,
    pub projection_selector_evaluation: Rc<RefCell<SelectorEqEvaluation>>,
    pub output: Rc<RefCell<DiffSumcheckEvaluation>>,
}

pub struct Type4VerifierContext {
    pub layers: Vec<Type4LayerVerifierContext>,
    pub output_layer: Type4OutputLayerVerifierContext,
}

pub struct Type4LayerVerifierContext {
    pub selector_evaluation: Rc<RefCell<SelectorEqEvaluation>>,
    pub child_selector_evaluation: Rc<RefCell<SelectorEqEvaluation>>,
    pub combiner_evaluation: Rc<RefCell<BasicEvaluationLinearSumcheck<RingElement>>>,
    pub combiner_constant_evaluation: Rc<RefCell<BasicEvaluationLinearSumcheck<RingElement>>>,
    pub ck_evaluations: Vec<Rc<RefCell<StructuredRowEvaluationLinearSumcheck<RingElement>>>>,
    pub outputs: Vec<Rc<RefCell<DiffSumcheckEvaluation>>>,
}

pub struct Type4OutputLayerVerifierContext {
    pub selector_evaluation: Rc<RefCell<SelectorEqEvaluation>>,
    pub ck_evaluations: Vec<Rc<RefCell<StructuredRowEvaluationLinearSumcheck<RingElement>>>>,
    pub outputs: Vec<Rc<RefCell<ProductSumcheckEvaluation>>>,
}

pub struct Type5VerifierContext {
    pub conjugated_combined_witness_evaluation:
        Rc<RefCell<FakeEvaluationLinearSumcheck<RingElement>>>,
    pub output: Rc<RefCell<ProductSumcheckEvaluation>>,
}
