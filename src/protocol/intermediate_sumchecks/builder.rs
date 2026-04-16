use crate::{
    common::ring_arithmetic::RingElement,
    protocol::{
        config::{Config, IntermediateConfig},
        crs::CRS,
        sumcheck_utils::{
            combiner::Combiner, common::HighOrderSumcheckData, elephant_cell::ElephantCell,
            linear::LinearSumcheck, product::ProductSumcheck,
            ring_to_field_combiner::RingToFieldCombiner,
        },
        sumchecks::helpers::{ck_sumcheck, composition_sumcheck},
    },
};

use super::context::{IntermediateSumcheckContext, Type0IntermediateSumcheckContext};

pub fn init_intermediate_sumcheck(
    crs: &CRS,
    config: &IntermediateConfig,
) -> IntermediateSumcheckContext {
    let decomposed_witness_height = config.witness_height * config.witness_decomposition_chunks;
    let total_vars = decomposed_witness_height.ilog2() as usize;

    let witness_sumcheck = ElephantCell::new(LinearSumcheck::<RingElement>::new(
        decomposed_witness_height,
    ));

    let commitment_key_rows_sumcheck = (0..config.basic_commitment_rank)
        .map(|i| {
            ck_sumcheck(
                crs,
                total_vars,
                config.witness_height,
                i,
                config.witness_decomposition_chunks.ilog2() as usize,
            )
        })
        .collect::<Vec<ElephantCell<LinearSumcheck<RingElement>>>>();

    let witness_combiner_sumcheck = composition_sumcheck(
        config.witness_decomposition_base_log as u64,
        config.witness_decomposition_chunks,
        total_vars,
    );

    let recomposed_witness = ElephantCell::new(ProductSumcheck::new(
        witness_sumcheck.clone(),
        witness_combiner_sumcheck,
    ));

    let type0sumchecks = (0..config.basic_commitment_rank)
        .map(|i| Type0IntermediateSumcheckContext {
            output: ElephantCell::new(ProductSumcheck::new(
                recomposed_witness.clone(),
                commitment_key_rows_sumcheck[i].clone(),
            )),
        })
        .collect::<Vec<Type0IntermediateSumcheckContext>>();

    let mut all_outputs: Vec<ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>> =
        Vec::with_capacity(type0sumchecks.len());
    for type0 in &type0sumchecks {
        all_outputs.push(type0.output.clone());
    }

    let combiner = ElephantCell::new(Combiner::new(all_outputs));
    let field_combiner = ElephantCell::new(RingToFieldCombiner::new(combiner.clone()));

    IntermediateSumcheckContext {
        witness_sumcheck,
        commitment_key_rows_sumcheck,
        type0sumchecks,
        combiner,
        field_combiner,
        next: match config.next.as_deref() {
            Some(Config::Intermediate(next_config)) => {
                Some(Box::new(init_intermediate_sumcheck(crs, next_config)))
            }
            _ => None,
        },
    }
}
