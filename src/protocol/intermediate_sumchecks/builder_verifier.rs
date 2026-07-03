use crate::{
    common::{
        config::{DEGREE, NOF_BATCHES},
        ring_arithmetic::RingElement,
    },
    protocol::{
        config::{Config, IntermediateConfig},
        crs::CRS,
        intermediate_sumchecks::context_verifier::{
            CommitmentFoldIntermediateVerifierContext, FineProjIntermediateVerifierContext,
            InnerEvalFoldIntermediateVerifierContext, IntermediateVerifierSumcheckContext,
            NormCheckIntermediateVerifierContext,
        },
        sumcheck_utils::{
            combiner::CombinerEvaluation,
            common::EvaluationSumcheckData,
            elephant_cell::ElephantCell,
            linear::{FakeEvaluationLinearSumcheck, StructuredRowEvaluationLinearSumcheck},
            product::ProductSumcheckEvaluation,
            ring_to_field_combiner::RingToFieldCombinerEvaluation,
        },
        sumchecks::builder_verifier::{
            basic_evaluation_linear, load_combiner_evaluation_data, structured_row_ck_evaluation,
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

    let commitment_fold_evaluations = (0..config.basic_commitment_rank)
        .map(|i| CommitmentFoldIntermediateVerifierContext {
            output: ElephantCell::new(ProductSumcheckEvaluation::new(
                recomposed_witness.clone(),
                commitment_key_rows_evaluation[i].clone(),
            )),
        })
        .collect::<Vec<_>>();

    let inner_eval_fold_evaluations = (0..config.nof_openings)
        .map(|i| InnerEvalFoldIntermediateVerifierContext {
            inner_evaluation: inner_evaluation_structured[i].clone(),
            output: ElephantCell::new(ProductSumcheckEvaluation::new(
                recomposed_witness.clone(),
                inner_evaluation_structured[i].clone(),
            )),
        })
        .collect::<Vec<_>>();

    let height = config.projection_height;
    let inner_width = config.projection_ratio * height / DEGREE;
    let blocks = config.witness_height / inner_width;
    let fine_proj_evaluations: [FineProjIntermediateVerifierContext; NOF_BATCHES] =
        std::array::from_fn(|_| {
            let c_0_evaluation = ElephantCell::new(
                StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
                    blocks,
                    total_vars
                        - blocks.ilog2() as usize
                        - inner_width.ilog2() as usize
                        - config.witness_decomposition_chunks.ilog2() as usize,
                    inner_width.ilog2() as usize
                        + config.witness_decomposition_chunks.ilog2() as usize,
                ),
            );
            let j_batched_evaluation = basic_evaluation_linear(
                inner_width,
                total_vars
                    - inner_width.ilog2() as usize
                    - config.witness_decomposition_chunks.ilog2() as usize,
                config.witness_decomposition_chunks.ilog2() as usize,
            );

            let output = ElephantCell::new(ProductSumcheckEvaluation::new(
                recomposed_witness.clone(),
                ElephantCell::new(ProductSumcheckEvaluation::new(
                    c_0_evaluation.clone(),
                    j_batched_evaluation.clone(),
                )),
            ));

            FineProjIntermediateVerifierContext {
                c_0_evaluation,
                j_batched_evaluation,
                output,
            }
        });

    let norm_check_evaluation = NormCheckIntermediateVerifierContext {
        conjugated_witness_evaluation: conjugated_witness_evaluation.clone(),
        output: ElephantCell::new(ProductSumcheckEvaluation::new(
            witness_evaluation.clone(),
            conjugated_witness_evaluation.clone(),
        )),
    };

    let mut all_outputs: Vec<ElephantCell<EvalData>> = Vec::new();
    for commitment_fold in &commitment_fold_evaluations {
        all_outputs.push(commitment_fold.output.clone());
    }
    for inner_eval_fold in &inner_eval_fold_evaluations {
        all_outputs.push(inner_eval_fold.output.clone());
    }
    for fine_proj in &fine_proj_evaluations {
        all_outputs.push(fine_proj.output.clone());
    }
    all_outputs.push(norm_check_evaluation.output.clone());

    let combiner_evaluation = ElephantCell::new(CombinerEvaluation::new(all_outputs));
    let field_combiner_evaluation = ElephantCell::new(RingToFieldCombinerEvaluation::new(
        combiner_evaluation.clone(),
    ));

    IntermediateVerifierSumcheckContext {
        witness_evaluation,
        conjugated_witness_evaluation,
        witness_combiner_evaluation,
        commitment_key_rows_evaluation,
        commitment_fold_evaluations,
        inner_eval_fold_evaluations,
        fine_proj_evaluations,
        norm_check_evaluation,
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
