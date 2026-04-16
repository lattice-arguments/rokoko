use crate::{
    common::ring_arithmetic::RingElement,
    protocol::sumcheck_utils::{
        combiner::Combiner, common::SumcheckBaseData, elephant_cell::ElephantCell,
        linear::LinearSumcheck, product::ProductSumcheck,
        ring_to_field_combiner::RingToFieldCombiner,
    },
};

pub struct Type0IntermediateSumcheckContext {
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
}

pub struct Type5IntermediateSumcheckContext {
    pub conjugated_witness_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
}

pub struct IntermediateSumcheckContext {
    pub witness_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub commitment_key_rows_sumcheck: Vec<ElephantCell<LinearSumcheck<RingElement>>>,
    pub type0sumchecks: Vec<Type0IntermediateSumcheckContext>,
    pub type5sumcheck: Type5IntermediateSumcheckContext,
    pub combiner: ElephantCell<Combiner<RingElement>>,
    pub field_combiner: ElephantCell<RingToFieldCombiner>,
    pub next: Option<Box<IntermediateSumcheckContext>>,
}

impl IntermediateSumcheckContext {
    pub fn partial_evaluate_all(&mut self, r: &RingElement) {
        self.witness_sumcheck.borrow_mut().partial_evaluate(r);
        for ck_row_sc in self.commitment_key_rows_sumcheck.iter() {
            ck_row_sc.borrow_mut().partial_evaluate(r);
        }
        self.type5sumcheck
            .conjugated_witness_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
    }
}
