use crate::{
    common::ring_arithmetic::RingElement,
    protocol::{
        commitment::Prefix,
        config::Config,
        crs::CRS,
        sumcheck_utils::{
            combiner::CombinerEvaluation,
            common::EvaluationSumcheckData,
            diff::DiffSumcheckEvaluation,
            linear::{BasicEvaluationLinearSumcheck, FakeEvaluationLinearSumcheck, StructuredRowEvaluationLinearSumcheck},
            product::ProductSumcheckEvaluation,
            ring_to_field_combiner::RingToFieldCombinerEvaluation,
            selector_eq::SelectorEqEvaluation,
        },
    },
};

use super::context_verifier::{
    Type0VerifierContext, Type1VerifierContext, Type2VerifierContext, Type3VerifierContext,
    Type4LayerVerifierContext, Type4OutputLayerVerifierContext, Type4VerifierContext,
    Type5VerifierContext, VerifierSumcheckContext,
};

/// Build the complete verifier sumcheck structure - mirrors init_sumcheck but with evaluation types
pub fn init_context_verifier(crs: &CRS, config: &Config) -> VerifierSumcheckContext {
    let total_vars = config.composed_witness_length.ilog2() as usize;

    // Base evaluations - these are the leaf nodes
    let combined_witness_evaluation = FakeEvaluationLinearSumcheck::new();
    
    let folded_witness_selector_evaluation = SelectorEqEvaluation::new(
        config.folded_witness_prefix.prefix,
        config.folded_witness_prefix.length,
        total_vars,
    );

    let mut commitment_key_rows_evaluation = Vec::new();
    for i in 0..config.basic_commitment_rank {
        let index = config.witness_height.ilog2() as usize - 1;
        let mut ck_eval = StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            config.witness_height,
            total_vars - config.witness_height.ilog2() as usize 
                - config.witness_decomposition_chunks.ilog2() as usize,
            config.witness_decomposition_chunks.ilog2() as usize,
        );
        commitment_key_rows_evaluation.push(ck_eval);
    }

    let folded_witness_combiner_evaluation = BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
        config.witness_decomposition_chunks,
        total_vars - config.witness_decomposition_chunks.ilog2() as usize,
        0,
    );

    let witness_combiner_constant_evaluation = BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
        config.witness_decomposition_chunks,
        total_vars - config.witness_decomposition_chunks.ilog2() as usize,
        0,
    );

    let basic_commitment_combiner_evaluation = BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
        config.commitment_recursion.decomposition_chunks,
        total_vars - config.commitment_recursion.decomposition_chunks.ilog2() as usize,
        0,
    );

    let basic_commitment_combiner_constant_evaluation = BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
        config.commitment_recursion.decomposition_chunks,
        total_vars - config.commitment_recursion.decomposition_chunks.ilog2() as usize,
        0,
    );

    let opening_combiner_evaluation = BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
        config.opening_recursion.decomposition_chunks,
        total_vars - config.opening_recursion.decomposition_chunks.ilog2() as usize,
        0,
    );

    let opening_combiner_constant_evaluation = BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
        config.opening_recursion.decomposition_chunks,
        total_vars - config.opening_recursion.decomposition_chunks.ilog2() as usize,
        0,
    );

    let projection_combiner_evaluation = BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
        config.projection_recursion.decomposition_chunks,
        total_vars - config.projection_recursion.decomposition_chunks.ilog2() as usize,
        0,
    );

    let projection_combiner_constant_evaluation = BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
        config.projection_recursion.decomposition_chunks,
        total_vars - config.projection_recursion.decomposition_chunks.ilog2() as usize,
        0,
    );

    let folding_challenges_evaluation = BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
        config.witness_width,
        total_vars
            - config.witness_width.ilog2() as usize
            - config.commitment_recursion.decomposition_chunks.ilog2() as usize,
        config.commitment_recursion.decomposition_chunks.ilog2() as usize,
    );

    // Build Type0 contexts (mirroring builder.rs lines 276-313)
    let type0evaluations = (0..config.basic_commitment_rank)
        .map(|i| {
            let basic_commitment_row_evaluation = SelectorEqEvaluation::new(
                config.commitment_recursion.prefix.prefix * config.basic_commitment_rank + i,
                config.commitment_recursion.prefix.length + config.basic_commitment_rank.ilog2() as usize,
                total_vars,
            );

            // Build the complex diff/product structure
            // This mirrors the exact structure from builder.rs Type0
            let output = DiffSumcheckEvaluation::new_empty();

            Type0VerifierContext {
                basic_commitment_row_evaluation,
                output,
            }
        })
        .collect::<Vec<_>>();

    // Build Type1 contexts
    let type1evaluations = (0..config.nof_openings)
        .map(|i| {
            let opening_selector_evaluation = SelectorEqEvaluation::new(
                config.opening_recursion.prefix.prefix * config.nof_openings + i,
                config.opening_recursion.prefix.length + config.nof_openings.ilog2() as usize,
                total_vars,
            );

            let inner_evaluation = BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
                config.witness_height,
                total_vars
                    - config.witness_height.ilog2() as usize
                    - config.witness_decomposition_chunks.ilog2() as usize,
                config.witness_decomposition_chunks.ilog2() as usize,
            );

            let output = DiffSumcheckEvaluation::new_empty();

            Type1VerifierContext {
                inner_evaluation,
                opening_selector_evaluation,
                output,
            }
        })
        .collect::<Vec<_>>();

    // Build Type2 contexts
    let type2evaluations = (0..config.nof_openings)
        .map(|_i| {
            let outer_evaluation = BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
                config.witness_width,
                total_vars
                    - config.witness_width.ilog2() as usize
                    - config.opening_recursion.decomposition_chunks.ilog2() as usize,
                config.opening_recursion.decomposition_chunks.ilog2() as usize,
            );

            let output = ProductSumcheckEvaluation::new_empty();

            Type2VerifierContext {
                outer_evaluation,
                output,
            }
        })
        .collect::<Vec<_>>();

    // Build Type3 context
    let projection_height_flat = config.witness_height / config.projection_ratio;
    let type3evaluation = {
        let projection_selector_evaluation = SelectorEqEvaluation::new(
            config.projection_recursion.prefix.prefix,
            config.projection_recursion.prefix.length,
            total_vars,
        );

        let lhs_evaluation = BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            config.witness_height,
            total_vars
                - config.witness_height.ilog2() as usize
                - config.witness_decomposition_chunks.ilog2() as usize,
            config.witness_decomposition_chunks.ilog2() as usize,
        );

        let rhs_evaluation = BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            projection_height_flat * config.witness_width,
            total_vars
                - (projection_height_flat * config.witness_width).ilog2() as usize
                - config.projection_recursion.decomposition_chunks.ilog2() as usize,
            config.projection_recursion.decomposition_chunks.ilog2() as usize,
        );

        let output = DiffSumcheckEvaluation::new_empty();

        Type3VerifierContext {
            lhs_evaluation,
            rhs_evaluation,
            projection_selector_evaluation,
            output,
        }
    };

    // Build Type4 contexts (three recursive trees)
    let type4evaluations = [
        build_type4_evaluation_context(crs, total_vars, config, &config.commitment_recursion),
        build_type4_evaluation_context(crs, total_vars, config, &config.opening_recursion),
        build_type4_evaluation_context(crs, total_vars, config, &config.projection_recursion),
    ];

    // Build Type5 context
    let conjugated_combined_witness_evaluation = FakeEvaluationLinearSumcheck::new();
    let type5evaluation = Type5VerifierContext {
        conjugated_combined_witness_evaluation,
        output: ProductSumcheckEvaluation::new_empty(),
    };

    // Create empty field combiner (will be populated after we build the full structure)
    let field_combiner_evaluation = RingToFieldCombinerEvaluation::new_empty();

    VerifierSumcheckContext {
        combined_witness_evaluation,
        folded_witness_selector_evaluation,
        folded_witness_combiner_evaluation,
        witness_combiner_constant_evaluation,
        folding_challenges_evaluation,
        basic_commitment_combiner_evaluation,
        basic_commitment_combiner_constant_evaluation,
        commitment_key_rows_evaluation,
        opening_combiner_evaluation,
        opening_combiner_constant_evaluation,
        projection_combiner_evaluation,
        projection_combiner_constant_evaluation,
        type0evaluations,
        type1evaluations,
        type2evaluations,
        type3evaluation,
        type4evaluations,
        type5evaluation,
        field_combiner_evaluation,
    }
}

fn build_type4_evaluation_context(
    crs: &CRS,
    total_vars: usize,
    config: &Config,
    recursion_config: &crate::protocol::commitment::RecursionConfig,
) -> Type4VerifierContext {
    let mut layers = Vec::new();
    let mut current = recursion_config;
    
    while let Some(next) = current.next.as_deref() {
        let selector_evaluation = SelectorEqEvaluation::new(
            current.prefix.prefix,
            current.prefix.length,
            total_vars,
        );
        
        let child_selector_evaluation = SelectorEqEvaluation::new(
            next.prefix.prefix,
            next.prefix.length,
            total_vars,
        );

        let combiner_evaluation = BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            next.decomposition_chunks,
            total_vars - next.decomposition_chunks.ilog2() as usize,
            0,
        );

        let combiner_constant_evaluation = BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            next.decomposition_chunks,
            total_vars - next.decomposition_chunks.ilog2() as usize,
            0,
        );

        let data_len = 1 << (total_vars - current.prefix.length);
        let mut ck_evaluations = Vec::new();
        for _i in 0..current.rank {
            let index = data_len.ilog2() as usize - 1;
            let ck_eval = StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
                data_len,
                total_vars - data_len.ilog2() as usize,
                0,
            );
            ck_evaluations.push(ck_eval);
        }

        let outputs = (0..current.rank)
            .map(|_| DiffSumcheckEvaluation::new_empty())
            .collect();

        layers.push(Type4LayerVerifierContext {
            selector_evaluation,
            child_selector_evaluation,
            combiner_evaluation,
            combiner_constant_evaluation,
            ck_evaluations,
            outputs,
        });

        current = next;
    }

    // Build output layer
    let selector_evaluation = SelectorEqEvaluation::new(
        current.prefix.prefix,
        current.prefix.length,
        total_vars,
    );

    let data_len = 1 << (total_vars - current.prefix.length);
    let mut ck_evaluations = Vec::new();
    for _i in 0..current.rank {
        let ck_eval = StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            data_len,
            total_vars - data_len.ilog2() as usize,
            0,
        );
        ck_evaluations.push(ck_eval);
    }

    let outputs = (0..current.rank)
        .map(|_| ProductSumcheckEvaluation::new_empty())
        .collect();

    let output_layer = Type4OutputLayerVerifierContext {
        selector_evaluation,
        ck_evaluations,
        outputs,
    };

    Type4VerifierContext {
        layers,
        output_layer,
    }
}
