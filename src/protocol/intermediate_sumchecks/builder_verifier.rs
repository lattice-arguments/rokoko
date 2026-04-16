use crate::{
    common::ring_arithmetic::RingElement,
    protocol::{
        config::{Config, IntermediateConfig},
        crs::CRS,
        intermediate_sumchecks::context_verifier::{
            IntermediateVerifierSumcheckContext, Type0IntermediateVerifierContext,
            Type1IntermediateVerifierContext, Type5IntermediateVerifierContext,
        },
        sumcheck_utils::{
            combiner::CombinerEvaluation, common::EvaluationSumcheckData,
            elephant_cell::ElephantCell,
            linear::{
                FakeEvaluationLinearSumcheck, StructuredRowEvaluationLinearSumcheck,
            },
            product::ProductSumcheckEvaluation,
            ring_to_field_combiner::RingToFieldCombinerEvaluation,
        },
        sumchecks::builder_verifier::{
            load_combiner_evaluation_data, structured_row_ck_evaluation,
        },
    },
};

type EvalData = dyn EvaluationSumcheckData<Element = RingElement>;

pub fn init_intermediate_verifier(
    crs: &CRS,
    config: &IntermediateConfig,
) -> IntermediateVerifierSumcheckContext {
    let decomposed_witness_height = config.witness_height * config.witness_decomposition_chunks;
    let total_vars = decomposed_witness_height.ilog2() as usize;

    let witness_evaluation = ElephantCell::new(FakeEvaluationLinearSumcheck::<RingElement>::new());
    let conjugated_witness_evaluation =
        ElephantCell::new(FakeEvaluationLinearSumcheck::<RingElement>::new());

    let witness_combiner_evaluation = load_combiner_evaluation_data(
        config.witness_decomposition_base_log as u64,
        config.witness_decomposition_chunks,
        total_vars,
    );

    let commitment_key_rows_evaluation = (0..config.basic_commitment_rank)
        .map(|i| {
            structured_row_ck_evaluation(
                crs,
                total_vars,
                config.witness_height,
                i,
                config.witness_decomposition_chunks.ilog2() as usize,
            )
        })
        .collect::<Vec<_>>();

    let recomposed_witness = ElephantCell::new(ProductSumcheckEvaluation::new(
        witness_evaluation.clone(),
        witness_combiner_evaluation.clone(),
    ));

    let inner_evaluation_structured = (0..config.nof_openings)
        .map(|_| {
            ElephantCell::new(
                StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
                    config.witness_height,
                    0,
                    config.witness_decomposition_chunks.ilog2() as usize,
                ),
            )
        })
        .collect::<Vec<_>>();

    let type0evaluations = (0..config.basic_commitment_rank)
        .map(|i| Type0IntermediateVerifierContext {
            output: ElephantCell::new(ProductSumcheckEvaluation::new(
                recomposed_witness.clone(),
                commitment_key_rows_evaluation[i].clone(),
            )),
        })
        .collect::<Vec<_>>();

    let type1evaluations = (0..config.nof_openings)
        .map(|i| Type1IntermediateVerifierContext {
            inner_evaluation: inner_evaluation_structured[i].clone(),
            output: ElephantCell::new(ProductSumcheckEvaluation::new(
                recomposed_witness.clone(),
                inner_evaluation_structured[i].clone(),
            )),
        })
        .collect::<Vec<_>>();

    let type5evaluation = Type5IntermediateVerifierContext {
        conjugated_witness_evaluation: conjugated_witness_evaluation.clone(),
        output: ElephantCell::new(ProductSumcheckEvaluation::new(
            witness_evaluation.clone(),
            conjugated_witness_evaluation.clone(),
        )),
    };

    let mut all_outputs: Vec<ElephantCell<EvalData>> = Vec::new();
    for type0 in &type0evaluations {
        all_outputs.push(type0.output.clone());
    }
    for type1 in &type1evaluations {
        all_outputs.push(type1.output.clone());
    }
    all_outputs.push(type5evaluation.output.clone());

    let combiner_evaluation = ElephantCell::new(CombinerEvaluation::new(all_outputs));
    let field_combiner_evaluation = ElephantCell::new(RingToFieldCombinerEvaluation::new(
        combiner_evaluation.clone(),
    ));

    IntermediateVerifierSumcheckContext {
        witness_evaluation,
        conjugated_witness_evaluation,
        witness_combiner_evaluation,
        commitment_key_rows_evaluation,
        type0evaluations,
        type1evaluations,
        type5evaluation,
        combiner_evaluation,
        field_combiner_evaluation,
        next: match config.next.as_deref() {
            Some(Config::Intermediate(next_config)) => {
                Some(Box::new(init_intermediate_verifier(crs, next_config)))
            }
            _ => None,
        },
    }
}
