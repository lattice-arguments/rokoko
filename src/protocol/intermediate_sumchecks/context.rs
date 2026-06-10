use crate::{
    common::{config::NOF_BATCHES, ring_arithmetic::RingElement},
    protocol::sumcheck_utils::{
        combiner::Combiner, common::SumcheckBaseData, elephant_cell::ElephantCell,
        linear::LinearSumcheck, product::ProductSumcheck,
        ring_to_field_combiner::RingToFieldCombiner,
    },
};

pub struct CommitmentFoldIntermediateSumcheckContext {
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
}
pub struct InnerEvalFoldIntermediateSumcheckContext {
    pub inner_evaluation_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
}

pub struct NormCheckIntermediateSumcheckContext {
    pub conjugated_witness_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
}

pub struct FineProjIntermediateSumcheckContext {
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
    pub c_0_sumcheck: ElephantCell<LinearSumcheck<RingElement>>, // across blocks
    pub j_batched_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
}

pub struct IntermediateSumcheckContext {
    pub witness_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub witness_combiner_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub commitment_key_rows_sumcheck: Vec<ElephantCell<LinearSumcheck<RingElement>>>,
    pub commitment_fold_sumchecks: Vec<CommitmentFoldIntermediateSumcheckContext>,
    pub inner_eval_fold_sumchecks: Vec<InnerEvalFoldIntermediateSumcheckContext>,
    pub fine_proj_sumchecks: [FineProjIntermediateSumcheckContext; NOF_BATCHES],
    pub norm_check_sumcheck: NormCheckIntermediateSumcheckContext,
    pub combiner: ElephantCell<Combiner<RingElement>>,
    pub field_combiner: ElephantCell<RingToFieldCombiner>,
    pub next: Option<Box<IntermediateSumcheckContext>>,
}

impl IntermediateSumcheckContext {
    pub fn partial_evaluate_all(&mut self, r: &RingElement) {
        self.witness_sumcheck.borrow_mut().partial_evaluate(r);
        self.witness_combiner_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        for ck_row_sc in self.commitment_key_rows_sumcheck.iter() {
            ck_row_sc.borrow_mut().partial_evaluate(r);
        }
        for inner_eval_fold_sc in self.inner_eval_fold_sumchecks.iter() {
            inner_eval_fold_sc
                .inner_evaluation_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }
        for fine_proj_sc in self.fine_proj_sumchecks.iter() {
            fine_proj_sc.c_0_sumcheck.borrow_mut().partial_evaluate(r);
            fine_proj_sc
                .j_batched_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }
        self.norm_check_sumcheck
            .conjugated_witness_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
    }
}
