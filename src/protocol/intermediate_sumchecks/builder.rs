use crate::common::arithmetic::ONE;
use crate::common::config::DEGREE;
use crate::protocol::config::{IntermediateConfig, Projection, SumcheckConfig};
use crate::protocol::intermediate_sumchecks::context::{
    IntermediateSumcheckContext, Type0IntermediateSumcheckContext,
};
use crate::protocol::sumcheck::SumcheckContext;
use crate::protocol::sumcheck_utils::sum::SumSumcheck;
use crate::protocol::sumchecks::context::Type3_1SumcheckContextWrapper;
use crate::protocol::sumchecks::helpers::{ck_sumcheck, composition_sumcheck};
use crate::{
    common::{config::NOF_BATCHES, ring_arithmetic::RingElement},
    protocol::{
        commitment::{self, Prefix},
        config::Config,
        crs::{self, CRS},
        sumcheck_utils::{
            combiner::Combiner, common::HighOrderSumcheckData, diff::DiffSumcheck,
            elephant_cell::ElephantCell, linear::LinearSumcheck, product::ProductSumcheck,
            ring_to_field_combiner::RingToFieldCombiner,
        },
        sumchecks::context::Type5SumcheckContext,
    },
};

pub fn init_intermediate_sumcheck(
    crs: &crs::CRS,
    config: &IntermediateConfig,
) -> IntermediateSumcheckContext {
    let decomposed_witness_height = config.witness_height * config.witness_decomposition_chunks;
    let total_vars = decomposed_witness_height.ilog2() as usize;

    let witness_sumcheck = ElephantCell::new(LinearSumcheck::<RingElement>::new(
        decomposed_witness_height,
    ));

    let conjugated_witness_sumcheck = ElephantCell::new(LinearSumcheck::<RingElement>::new(
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

    // let folding_challenges_sumcheck = ElephantCell::new(
    //     LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
    //         config.witness_width,
    //         total_vars - config.witness_width.ilog2() as usize,
    //         0,
    //     ),
    // );

    let folded_witness_combiner_sumcheck = composition_sumcheck(
        config.witness_decomposition_base_log as u64,
        config.witness_decomposition_chunks,
        config.witness_height.ilog2() as usize,
    );

    let recomposed_folded_witness = ElephantCell::new(ProductSumcheck::new(
        witness_sumcheck.clone(),
        folded_witness_combiner_sumcheck.clone(),
    ));

    // Type0 sumchecks
    // CK \cdot folded_witness = folded_commitment (public)
    let type0sumchecks = (0..config.basic_commitment_rank)
        .map(|i| {
            let ctxt = Type0IntermediateSumcheckContext {
                output: ElephantCell::new(ProductSumcheck::new(
                    recomposed_folded_witness.clone(),
                    commitment_key_rows_sumcheck[i].clone(),
                )),
            };
            ctxt
        })
        .collect::<Vec<Type0IntermediateSumcheckContext>>();

    let mut all_outputs: Vec<ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>> =
        vec![];
    for type0 in &type0sumchecks {
        all_outputs.push(type0.output.clone());
    }

    let combiner = ElephantCell::new(Combiner::new(all_outputs));

    let field_combiner = ElephantCell::new(RingToFieldCombiner::new(combiner.clone()));

    IntermediateSumcheckContext {
        witness_sumcheck,
        conjugated_witness_sumcheck,
        commitment_key_rows_sumcheck,
        type0sumchecks,
        combiner,
        field_combiner,
        next: None,
    }
}
