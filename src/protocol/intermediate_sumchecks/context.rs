use crate::{
    common::{config::NOF_BATCHES, ring_arithmetic::RingElement},
    protocol::sumcheck_utils::{
        combiner::Combiner, common::SumcheckBaseData, elephant_cell::ElephantCell,
        linear::LinearSumcheck, product::ProductSumcheck,
        ring_to_field_combiner::RingToFieldCombiner,
    },
};

pub struct Type0IntermediateSumcheckContext {
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
}
pub struct Type1IntermediateSumcheckContext {
    pub inner_evaluation_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
}

pub struct Type5IntermediateSumcheckContext {
    pub conjugated_witness_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
}


pub struct Type3_1IntermediateSumcheckContext {
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
    pub c_0_sumcheck: ElephantCell<LinearSumcheck<RingElement>>, // across blocks
    pub j_batched_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
}

pub struct IntermediateSumcheckContext {
    pub witness_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub witness_combiner_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub commitment_key_rows_sumcheck: Vec<ElephantCell<LinearSumcheck<RingElement>>>,
    pub type0sumchecks: Vec<Type0IntermediateSumcheckContext>,
    pub type1sumchecks: Vec<Type1IntermediateSumcheckContext>,
    pub type3_1sumcheck: [Type3_1IntermediateSumcheckContext; NOF_BATCHES],
    pub type5sumcheck: Type5IntermediateSumcheckContext,
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
        for type1_sc in self.type1sumchecks.iter() {
            type1_sc
                .inner_evaluation_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }
        for type3_1_sc in self.type3_1sumcheck.iter() {
            type3_1_sc.c_0_sumcheck.borrow_mut().partial_evaluate(r);
            type3_1_sc.j_batched_sumcheck.borrow_mut().partial_evaluate(r);
        }
        self.type5sumcheck
            .conjugated_witness_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
    }
}
