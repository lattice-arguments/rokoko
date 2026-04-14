use crate::{
    common::{
        config::NOF_BATCHES,
        ring_arithmetic::{QuadraticExtension, RingElement},
    },
    protocol::sumcheck_utils::{
        combiner::CombinerEvaluation,
        diff::DiffSumcheckEvaluation,
        elephant_cell::ElephantCell,
        linear::{
            BasicEvaluationLinearSumcheck, FakeEvaluationLinearSumcheck,
            StructuredRowEvaluationLinearSumcheck,
        },
        product::ProductSumcheckEvaluation,
        ring_to_field_combiner::RingToFieldCombinerEvaluation,
        selector_eq::SelectorEqEvaluation,
    },
};

pub struct VerifierSumcheckContext {
    pub witness_evaluation: ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    pub witness_conjugated_evaluation: ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    pub main_witness_selector_evaluation: ElephantCell<SelectorEqEvaluation>,
    pub projection_selector_evaluation: Option<ElephantCell<SelectorEqEvaluation>>,
    pub type1evaluations: Vec<Type1VerifierSumcheckContext>,
    pub type3evaluation: Option<Type3VerifierSumcheckContext>,
    pub type31evaluations: Option<[Type31VerifierSumcheckContext; NOF_BATCHES]>,
    pub l2evaluation: Option<L2VerifierSumcheckContext>,
    pub linfevaluation: Option<LinfVerifierSumcheckContext>,
    pub vdfevaluation: Option<VDFVerifierSumcheckContext>,
    pub combiner_evaluation: ElephantCell<CombinerEvaluation<RingElement>>,
    pub field_combiner_evaluation: ElephantCell<RingToFieldCombinerEvaluation>,
    pub next: Option<Box<VerifierSumcheckContext>>,
}

pub struct L2VerifierSumcheckContext {
    pub output: ElephantCell<ProductSumcheckEvaluation>,
}

pub struct LinfVerifierSumcheckContext {
    pub all_one_constant_evaluation: ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheckEvaluation>,
    pub one_minus_wit_evaluation: ElephantCell<DiffSumcheckEvaluation>,
    pub one_minus_wit_selector_evaluation: ElephantCell<ProductSumcheckEvaluation>,
}

pub struct VDFVerifierSumcheckContext {
    pub vdf_step_powers_evaluation: ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    pub vdf_batched_row_evaluation: ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheckEvaluation>,
}

pub struct Type1VerifierSumcheckContext {
    pub inner_evaluation_sumcheck: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub outer_evaluation_sumcheck: ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheckEvaluation>,
}

pub struct Type3VerifierSumcheckContext {
    pub c2l_evaluation: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub c0l_evaluation: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    // TODO: this can be over fields, then then mapped to rings?. Actually, all of those can be over fields (I guess?).
    pub flattened_projection_matrix_evaluation:
        ElephantCell<BasicEvaluationLinearSumcheck<QuadraticExtension>>,
    pub c2r_evaluation: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub c0r_evaluation: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub c1r_evaluation: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub lhs: ElephantCell<ProductSumcheckEvaluation>,
    pub rhs: ElephantCell<ProductSumcheckEvaluation>,
    pub output: ElephantCell<DiffSumcheckEvaluation>,
}

// Type 3.1 verifier: evaluates <c_2 ⊗ c_0 ⊗ j_batched, witness> at the sumcheck point.
// c_2 and c_0 are succinct (StructuredRow), j_batched is explicit.
pub struct Type31VerifierSumcheckContext {
    pub c_2_evaluation: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub c_0_evaluation: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub j_batched_evaluation: ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheckEvaluation>,
}
