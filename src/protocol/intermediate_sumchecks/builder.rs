use crate::{
    common::{
        config::{DEGREE, NOF_BATCHES},
        ring_arithmetic::RingElement,
    },
    protocol::{
        config::{Config, IntermediateConfig},
        crs::CRS,
        intermediate_sumchecks::context::{
            Type1IntermediateSumcheckContext, Type3_1IntermediateSumcheckContext,
        },
        sumcheck_utils::{
            combiner::Combiner, common::HighOrderSumcheckData, elephant_cell::ElephantCell,
            linear::LinearSumcheck, product::ProductSumcheck,
            ring_to_field_combiner::RingToFieldCombiner,
        },
        sumchecks::helpers::{ck_sumcheck, composition_sumcheck},
    },
};

use super::context::{
    IntermediateSumcheckContext, Type0IntermediateSumcheckContext, Type5IntermediateSumcheckContext,
};

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
        witness_combiner_sumcheck.clone(),
    ));

    let type0sumchecks = (0..config.basic_commitment_rank)
        .map(|i| Type0IntermediateSumcheckContext {
            output: ElephantCell::new(ProductSumcheck::new(
                commitment_key_rows_sumcheck[i].clone(),
                recomposed_witness.clone(),
            )),
        })
        .collect::<Vec<Type0IntermediateSumcheckContext>>();

    let type1sumchecks = (0..config.nof_openings)
        .map(|_| {
            let inner_evaluation_sumcheck = ElephantCell::new(
                LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                    config.witness_height,
                    0,
                    config.witness_decomposition_chunks.ilog2() as usize,
                ),
            );

            Type1IntermediateSumcheckContext {
                inner_evaluation_sumcheck: inner_evaluation_sumcheck.clone(),
                output: ElephantCell::new(ProductSumcheck::new(
                    recomposed_witness.clone(),
                    inner_evaluation_sumcheck.clone(),
                )),
            }
        })
        .collect::<Vec<Type1IntermediateSumcheckContext>>();

    let height = config.projection_height;
    let inner_width = config.projection_ratio * height / DEGREE;
    let blocks = config.witness_height / inner_width;
    let type3_1sumcheck: [Type3_1IntermediateSumcheckContext; NOF_BATCHES] =
        std::array::from_fn(|_| {
            let c_0_sumcheck = ElephantCell::new(
                LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                    blocks,
                    total_vars
                        - blocks.ilog2() as usize
                        - inner_width.ilog2() as usize
                        - config.witness_decomposition_chunks.ilog2() as usize,
                    inner_width.ilog2() as usize
                        + config.witness_decomposition_chunks.ilog2() as usize,
                ),
            );
            let j_batched_sumcheck = ElephantCell::new(
                LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                    inner_width,
                    total_vars
                        - inner_width.ilog2() as usize
                        - config.witness_decomposition_chunks.ilog2() as usize,
                    config.witness_decomposition_chunks.ilog2() as usize,
                ),
            );

            let output = ElephantCell::new(ProductSumcheck::new(
                recomposed_witness.clone(),
                ElephantCell::new(ProductSumcheck::new(
                    c_0_sumcheck.clone(),
                    j_batched_sumcheck.clone(),
                )),
            ));
            Type3_1IntermediateSumcheckContext {
                output,
                c_0_sumcheck,
                j_batched_sumcheck,
            }
        });

    let conjugated_witness_sumcheck = ElephantCell::new(LinearSumcheck::<RingElement>::new(
        decomposed_witness_height,
    ));
    let type5sumcheck = Type5IntermediateSumcheckContext {
        conjugated_witness_sumcheck: conjugated_witness_sumcheck.clone(),
        output: ElephantCell::new(ProductSumcheck::new(
            witness_sumcheck.clone(),
            conjugated_witness_sumcheck,
        )),
    };

    let mut all_outputs: Vec<ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>> =
        Vec::new();

    for type0 in &type0sumchecks {
        all_outputs.push(type0.output.clone());
    }
    for type1 in &type1sumchecks {
        all_outputs.push(type1.output.clone());
    }
    for type3_1 in &type3_1sumcheck {
        all_outputs.push(type3_1.output.clone());
    }
    all_outputs.push(type5sumcheck.output.clone());

    let combiner = ElephantCell::new(Combiner::new(all_outputs));
    let field_combiner = ElephantCell::new(RingToFieldCombiner::new(combiner.clone()));

    IntermediateSumcheckContext {
        witness_sumcheck,
        witness_combiner_sumcheck,
        commitment_key_rows_sumcheck,
        type0sumchecks,
        type1sumchecks,
        type3_1sumcheck,
        type5sumcheck,
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
