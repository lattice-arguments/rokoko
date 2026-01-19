use crate::{
    common::{
        arithmetic::{ONE, ONE_QUAD},
        config::{DEGREE, NOF_BATCHES},
        decomposition::get_decomposed_offset_scaled,
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
    },
    protocol::{
        commitment::{self, Prefix},
        config::{Config, Projection, SumcheckConfig},
        crs::CRS,
        sumcheck_utils::{
            combiner::CombinerEvaluation,
            common::EvaluationSumcheckData,
            diff::DiffSumcheckEvaluation,
            elephant_cell::ElephantCell,
            linear::{
                BasicEvaluationLinearSumcheck, FakeEvaluationLinearSumcheck,
                RingToFieldWrapperEvaluation, StructuredRowEvaluationLinearSumcheck,
            },
            product::ProductSumcheckEvaluation,
            ring_to_field_combiner::RingToFieldCombinerEvaluation,
            selector_eq::SelectorEqEvaluation,
            sum::SumSumcheckEvaluation,
        },
        sumchecks::context_verifier::{
            Type0VerifierContext, Type1VerifierContext, Type2VerifierContext, Type3VerifierContext,
            Type3_1VerifierContext, Type3_1VerifierContextWrapper, Type4LayerVerifierContext,
            Type4OutputLayerVerifierContext, Type4VerifierContext, Type5VerifierContext,
            VerifierSumcheckContext,
        },
    },
};

type EvalData = dyn EvaluationSumcheckData<Element = RingElement>;

fn selector_evaluation_from_prefix(
    prefix: &Prefix,
    total_vars: usize,
) -> ElephantCell<SelectorEqEvaluation> {
    ElephantCell::new(SelectorEqEvaluation::new(
        prefix.prefix,
        prefix.length,
        total_vars,
    ))
}

fn basic_evaluation_linear(
    count: usize,
    prefix_size: usize,
    suffix_size: usize,
) -> ElephantCell<BasicEvaluationLinearSumcheck<RingElement>> {
    ElephantCell::new(
        BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            count,
            prefix_size,
            suffix_size,
        ),
    )
}

fn load_combiner_evaluation_data(
    base_log: u64,
    chunks: usize,
    total_vars: usize,
) -> ElephantCell<BasicEvaluationLinearSumcheck<RingElement>> {
    let data = (0..chunks)
        .map(|i| {
            RingElement::constant(
                1u64 << (base_log as u64 * i as u64),
                Representation::IncompleteNTT,
            )
        })
        .collect::<Vec<_>>();

    let prefix_size = total_vars - (data.len().ilog2() as usize);
    let combiner_evaluation = basic_evaluation_linear(data.len(), prefix_size, 0);
    combiner_evaluation.borrow_mut().load_from(&data);
    combiner_evaluation
}

fn load_combiner_constant_evaluation(
    base_log: u64,
    chunks: usize,
    total_vars: usize,
) -> ElephantCell<BasicEvaluationLinearSumcheck<RingElement>> {
    let const_eval = basic_evaluation_linear(1, total_vars, 0);
    let value = RingElement::all(
        get_decomposed_offset_scaled(base_log as u64, chunks),
        Representation::IncompleteNTT,
    );
    const_eval.borrow_mut().load_from(&[value]);
    const_eval
}

fn structured_row_ck_evaluation(
    crs: &CRS,
    total_vars: usize,
    wit_dim: usize,
    i: usize,
    suffix: usize,
) -> ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>> {
    let prefix_size = total_vars - wit_dim.ilog2() as usize - suffix;
    let eval = ElephantCell::new(
        StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            wit_dim,
            prefix_size,
            suffix,
        ),
    );
    let structured_row = crs.structured_ck_for_wit_dim(wit_dim)[i].clone();
    eval.borrow_mut().load_from(structured_row);
    eval
}

fn build_type4_verifier_context(
    crs: &CRS,
    total_vars: usize,
    combined_witness_eval: ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    config: &commitment::RecursionConfig,
) -> Type4VerifierContext {
    let mut layers = Vec::new();
    let mut current = config;

    while let Some(next) = current.next.as_deref() {
        let selector_eval = selector_evaluation_from_prefix(&current.prefix, total_vars);
        let child_selectors_evals = (0..current.rank)
            .map(|i| {
                selector_evaluation_from_prefix(
                    &Prefix {
                        prefix: next.prefix.prefix * current.rank.next_power_of_two() + i,
                        length: next.prefix.length
                            + current.rank.next_power_of_two().ilog2() as usize,
                    },
                    total_vars,
                )
            })
            .collect::<Vec<_>>();

        selector_evaluation_from_prefix(&next.prefix, total_vars);

        let combiner_eval = load_combiner_evaluation_data(
            next.decomposition_base_log as u64,
            next.decomposition_chunks,
            total_vars,
        );
        let combiner_constant_eval = load_combiner_constant_evaluation(
            next.decomposition_base_log as u64,
            next.decomposition_chunks,
            total_vars,
        );

        let data_len = 1 << (total_vars - current.prefix.length);
        let ck_evals = (0..current.rank)
            .map(|i| structured_row_ck_evaluation(crs, total_vars, data_len, i, 0))
            .collect::<Vec<_>>();

        let data_selected_eval = ElephantCell::new(ProductSumcheckEvaluation::new(
            selector_eval.clone(),
            combined_witness_eval.clone(),
        ));

        let recomposed_combiner = ElephantCell::new(ProductSumcheckEvaluation::new(
            combined_witness_eval.clone(),
            combiner_eval.clone(),
        ));
        let recomposed_child_raw_eval = ElephantCell::new(DiffSumcheckEvaluation::new(
            recomposed_combiner.clone(),
            combiner_constant_eval.clone(),
        ));
        // let recomposed_child_evals  = ElephantCell::new(ProductSumcheckEvaluation::new(
        //     child_selector_eval.clone(),
        //     recomposed_child_raw_eval.clone(),
        // ));
        let recomposed_child_evals = (0..current.rank)
            .map(|i| {
                ElephantCell::new(ProductSumcheckEvaluation::new(
                    child_selectors_evals[i].clone(),
                    recomposed_child_raw_eval.clone(),
                ))
            })
            .collect::<Vec<_>>();

        let outputs = (0..current.rank)
            .map(|i| {
                let ck_with_data = ElephantCell::new(ProductSumcheckEvaluation::new(
                    ck_evals[i].clone(),
                    data_selected_eval.clone(),
                ));
                ElephantCell::new(DiffSumcheckEvaluation::new(
                    ck_with_data,
                    recomposed_child_evals[i].clone(),
                ))
            })
            .collect::<Vec<_>>();

        // ck_evals
        //     .iter()
        //     .map(|ck_eval| {
        //         let ck_with_data = ElephantCell::new(ProductSumcheckEvaluation::new(
        //             ck_eval.clone(),
        //             data_selected_eval.clone(),
        //         ));
        //         ElephantCell::new(DiffSumcheckEvaluation::new(
        //             ck_with_data,
        //             recomposed_child_evals[i].clone(),
        //         ))
        //     })
        //     .collect::<Vec<_>>();

        layers.push(Type4LayerVerifierContext {
            selector_evaluation: selector_eval,
            child_selector_evaluations: child_selectors_evals,
            combiner_evaluation: combiner_eval,
            combiner_constant_evaluation: combiner_constant_eval,
            ck_evaluations: ck_evals,
            outputs,
        });

        current = next;
    }

    let selector_eval = selector_evaluation_from_prefix(&current.prefix, total_vars);
    let data_len = 1 << (total_vars - current.prefix.length);
    let ck_evals = (0..current.rank)
        .map(|i| structured_row_ck_evaluation(crs, total_vars, data_len, i, 0))
        .collect::<Vec<_>>();

    let outputs = ck_evals
        .iter()
        .map(|ck_eval| {
            let witness_with_ck = ElephantCell::new(ProductSumcheckEvaluation::new(
                combined_witness_eval.clone(),
                ck_eval.clone(),
            ));
            ElephantCell::new(ProductSumcheckEvaluation::new(
                selector_eval.clone(),
                witness_with_ck,
            ))
        })
        .collect::<Vec<_>>();

    Type4VerifierContext {
        layers,
        output_layer: Type4OutputLayerVerifierContext {
            selector_evaluation: selector_eval,
            ck_evaluations: ck_evals,
            outputs,
        },
    }
}

pub fn init_verifier(crs: &CRS, config: &SumcheckConfig) -> VerifierSumcheckContext {
    let total_vars = config.composed_witness_length.ilog2() as usize;

    let combined_witness_evaluation =
        ElephantCell::new(FakeEvaluationLinearSumcheck::<RingElement>::new());

    let folded_witness_selector_evaluation =
        selector_evaluation_from_prefix(&config.folded_witness_prefix, total_vars);

    let folded_witness_combiner_evaluation = load_combiner_evaluation_data(
        config.witness_decomposition_base_log as u64,
        config.witness_decomposition_chunks,
        total_vars,
    );
    let witness_combiner_constant_evaluation = load_combiner_constant_evaluation(
        config.witness_decomposition_base_log as u64,
        config.witness_decomposition_chunks,
        total_vars,
    );

    let basic_commitment_combiner_evaluation = load_combiner_evaluation_data(
        config.commitment_recursion.decomposition_base_log as u64,
        config.commitment_recursion.decomposition_chunks,
        total_vars,
    );
    let basic_commitment_combiner_constant_evaluation = load_combiner_constant_evaluation(
        config.commitment_recursion.decomposition_base_log as u64,
        config.commitment_recursion.decomposition_chunks,
        total_vars,
    );

    let opening_combiner_evaluation = load_combiner_evaluation_data(
        config.opening_recursion.decomposition_base_log as u64,
        config.opening_recursion.decomposition_chunks,
        total_vars,
    );
    let opening_combiner_constant_evaluation = load_combiner_constant_evaluation(
        config.opening_recursion.decomposition_base_log as u64,
        config.opening_recursion.decomposition_chunks,
        total_vars,
    );

    let folding_challenges_evaluation = basic_evaluation_linear(
        config.witness_width,
        total_vars
            - config.witness_width.ilog2() as usize
            - config.commitment_recursion.decomposition_chunks.ilog2() as usize,
        config.commitment_recursion.decomposition_chunks.ilog2() as usize,
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

    let opening_selector_evaluations = (0..config.nof_openings)
        .map(|i| {
            selector_evaluation_from_prefix(
                &Prefix {
                    prefix: config.opening_recursion.prefix.prefix * config.nof_openings + i,
                    length: config.opening_recursion.prefix.length
                        + config.nof_openings.ilog2() as usize,
                },
                total_vars,
            )
        })
        .collect::<Vec<_>>();

    let inner_evaluation_structured = (0..config.nof_openings)
        .map(|_| {
            ElephantCell::new(
                StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
                    config.witness_height,
                    total_vars
                        - config.witness_height.ilog2() as usize
                        - config.witness_decomposition_chunks.ilog2() as usize,
                    config.witness_decomposition_chunks.ilog2() as usize,
                ),
            )
        })
        .collect::<Vec<_>>();

    let outer_evaluation_structured = (0..config.nof_openings)
        .map(|_| {
            ElephantCell::new(
                StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
                    config.witness_width,
                    total_vars
                        - config.witness_width.ilog2() as usize
                        - config.opening_recursion.decomposition_chunks.ilog2() as usize,
                    config.opening_recursion.decomposition_chunks.ilog2() as usize,
                ),
            )
        })
        .collect::<Vec<_>>();

    let recomposed_folded_witness = ElephantCell::new(DiffSumcheckEvaluation::new(
        ElephantCell::new(ProductSumcheckEvaluation::new(
            combined_witness_evaluation.clone(),
            folded_witness_combiner_evaluation.clone(),
        )),
        witness_combiner_constant_evaluation.clone(),
    ));
    let recomposed_opening = ElephantCell::new(DiffSumcheckEvaluation::new(
        ElephantCell::new(ProductSumcheckEvaluation::new(
            combined_witness_evaluation.clone(),
            opening_combiner_evaluation.clone(),
        )),
        opening_combiner_constant_evaluation.clone(),
    ));

    let basic_commitment_combiner_product = ElephantCell::new(ProductSumcheckEvaluation::new(
        combined_witness_evaluation.clone(),
        basic_commitment_combiner_evaluation.clone(),
    ));
    let basic_commitment_diff = ElephantCell::new(DiffSumcheckEvaluation::new(
        basic_commitment_combiner_product.clone(),
        basic_commitment_combiner_constant_evaluation.clone(),
    ));

    let folding_with_commitment_diff = ElephantCell::new(ProductSumcheckEvaluation::new(
        folding_challenges_evaluation.clone(),
        basic_commitment_diff.clone(),
    ));

    let type0evaluations = (0..config.basic_commitment_rank)
        .map(|i| {
            let row_selector = selector_evaluation_from_prefix(
                &Prefix {
                    prefix: config.commitment_recursion.prefix.prefix
                        * config.basic_commitment_rank.next_power_of_two()
                        + i,
                    length: config.commitment_recursion.prefix.length
                        + config.basic_commitment_rank.next_power_of_two().ilog2() as usize,
                },
                total_vars,
            );

            let ck_with_folded = ElephantCell::new(ProductSumcheckEvaluation::new(
                commitment_key_rows_evaluation[i].clone(),
                recomposed_folded_witness.clone(),
            ));
            let lhs = ElephantCell::new(ProductSumcheckEvaluation::new(
                folded_witness_selector_evaluation.clone(),
                ck_with_folded.clone(),
            ));

            let rhs = ElephantCell::new(ProductSumcheckEvaluation::new(
                row_selector.clone(),
                folding_with_commitment_diff.clone(),
            ));

            Type0VerifierContext {
                basic_commitment_row_evaluation: row_selector,
                output: ElephantCell::new(DiffSumcheckEvaluation::new(lhs, rhs)),
            }
        })
        .collect::<Vec<_>>();

    let type1evaluations = (0..config.nof_openings)
        .map(|i| {
            let inner_evaluation = inner_evaluation_structured[i].clone();
            let opening_selector = opening_selector_evaluations[i].clone();

            let lhs_inner = ElephantCell::new(ProductSumcheckEvaluation::new(
                recomposed_folded_witness.clone(),
                inner_evaluation.clone(),
            ));
            let lhs = ElephantCell::new(ProductSumcheckEvaluation::new(
                folded_witness_selector_evaluation.clone(),
                lhs_inner.clone(),
            ));

            let rhs_inner = ElephantCell::new(ProductSumcheckEvaluation::new(
                recomposed_opening.clone(),
                folding_challenges_evaluation.clone(),
            ));
            let rhs = ElephantCell::new(ProductSumcheckEvaluation::new(
                opening_selector.clone(),
                rhs_inner.clone(),
            ));

            Type1VerifierContext {
                inner_evaluation,
                opening_selector_evaluation: opening_selector,
                output: ElephantCell::new(DiffSumcheckEvaluation::new(lhs, rhs)),
            }
        })
        .collect::<Vec<_>>();

    let type2evaluations = (0..config.nof_openings)
        .map(|i| {
            let opening_selector = opening_selector_evaluations[i].clone();
            let outer_evaluation = outer_evaluation_structured[i].clone();

            let inner_product = ElephantCell::new(ProductSumcheckEvaluation::new(
                recomposed_opening.clone(),
                outer_evaluation.clone(),
            ));
            let output = ElephantCell::new(ProductSumcheckEvaluation::new(
                opening_selector.clone(),
                inner_product.clone(),
            ));

            Type2VerifierContext {
                outer_evaluation,
                output,
            }
        })
        .collect::<Vec<_>>();

    // Build Type3 with Product of split LHS and RHS coefficients
    let type3evaluation = {
        match &config.projection_recursion {
            Projection::Type0(projection_recursion) => {
                let projection_combiner_evaluation = load_combiner_evaluation_data(
                    projection_recursion.decomposition_base_log as u64,
                    projection_recursion.decomposition_chunks,
                    total_vars,
                );
                let projection_combiner_constant_evaluation = load_combiner_constant_evaluation(
                    projection_recursion.decomposition_base_log as u64,
                    projection_recursion.decomposition_chunks,
                    total_vars,
                );
                let projection_selector_evaluation =
                    selector_evaluation_from_prefix(&projection_recursion.prefix, total_vars);

                let projection_height_flat = config.witness_height / config.projection_ratio;

                // Split LHS projection coefficients evaluations
                let height = config.projection_height;
                let inner_width = config.projection_ratio * height;
                let blocks = config.witness_height / inner_width;

                let lhs_flatter_0_evaluation = ElephantCell::new(
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

                let lhs_flatter_1_times_matrix_evaluation_field = ElephantCell::new(
                BasicEvaluationLinearSumcheck::<QuadraticExtension>::new_with_prefixed_sufixed_data(
                    inner_width,
                    total_vars
                        - inner_width.ilog2() as usize
                        - config.witness_decomposition_chunks.ilog2() as usize,
                    config.witness_decomposition_chunks.ilog2() as usize,
                ),
            );

                let lhs_flatter_1_times_matrix_evaluation =
                    ElephantCell::new(RingToFieldWrapperEvaluation::new(
                        lhs_flatter_1_times_matrix_evaluation_field.clone(),
                    ));

                // we have flatter^T V  challenge
                // that since V is vectorised, we can write it as
                // <\vec(v), challenge  \otimes flatter> >
                let rhs_projection_flatter_evaluation = ElephantCell::new(
                    StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
                        projection_height_flat,
                        total_vars
                            - projection_height_flat.ilog2() as usize
                            - projection_recursion.decomposition_chunks.ilog2() as usize,
                        projection_recursion.decomposition_chunks.ilog2() as usize,
                    ),
                );

                let rhs_fold_challenge_evaluation = basic_evaluation_linear(
                    config.witness_width,
                    total_vars
                        - config.witness_width.ilog2() as usize
                        - projection_height_flat.ilog2() as usize
                        - projection_recursion.decomposition_chunks.ilog2() as usize,
                    projection_height_flat.ilog2() as usize
                        + projection_recursion.decomposition_chunks.ilog2() as usize,
                );

                let recomposed_projection = ElephantCell::new(DiffSumcheckEvaluation::new(
                    ElephantCell::new(ProductSumcheckEvaluation::new(
                        combined_witness_evaluation.clone(),
                        projection_combiner_evaluation.clone(),
                    )),
                    projection_combiner_constant_evaluation.clone(),
                ));

                let lhs_projection_coeff_product =
                    ElephantCell::new(ProductSumcheckEvaluation::new(
                        lhs_flatter_0_evaluation.clone(),
                        lhs_flatter_1_times_matrix_evaluation.clone(),
                    ));

                let rhs_fold_tensor_product = ElephantCell::new(ProductSumcheckEvaluation::new(
                    rhs_projection_flatter_evaluation.clone(),
                    rhs_fold_challenge_evaluation.clone(),
                ));

                let type3lhs_inner = ElephantCell::new(ProductSumcheckEvaluation::new(
                    recomposed_folded_witness.clone(),
                    lhs_projection_coeff_product,
                ));
                let type3lhs = ElephantCell::new(ProductSumcheckEvaluation::new(
                    folded_witness_selector_evaluation.clone(),
                    type3lhs_inner.clone(),
                ));

                let type3rhs_inner = ElephantCell::new(ProductSumcheckEvaluation::new(
                    recomposed_projection.clone(),
                    rhs_fold_tensor_product,
                ));
                let type3rhs = ElephantCell::new(ProductSumcheckEvaluation::new(
                    projection_selector_evaluation.clone(),
                    type3rhs_inner.clone(),
                ));

                Some(Type3VerifierContext {
                    projection_combiner_constant_evaluation,
                    projection_combiner_evaluation,
                    lhs_flatter_0_evaluation,
                    lhs_flatter_1_times_matrix_evaluation_field,
                    lhs_flatter_1_times_matrix_evaluation,
                    rhs_projection_flatter_evaluation,
                    rhs_fold_challenge_evaluation,
                    projection_selector_evaluation,
                    output: ElephantCell::new(DiffSumcheckEvaluation::new(type3lhs, type3rhs)),
                })
            }
            Projection::Type1(projection_recursion) => None,
        }
    };

    let type3_1_evaluations = match &config.projection_recursion {
        Projection::Type1(proj_config) => {
            let projection_combiner_evaluation = load_combiner_evaluation_data(
                proj_config
                    .recursion_batched_projection
                    .decomposition_base_log as u64,
                proj_config
                    .recursion_batched_projection
                    .decomposition_chunks,
                total_vars,
            );
            let projection_combiner_constant_evaluation = load_combiner_constant_evaluation(
                proj_config
                    .recursion_batched_projection
                    .decomposition_base_log as u64,
                proj_config
                    .recursion_batched_projection
                    .decomposition_chunks,
                total_vars,
            );
            let rhs_fold_challenge_evaluation = basic_evaluation_linear(
                config.witness_width,
                total_vars
                    - config.witness_width.ilog2() as usize
                    - proj_config
                        .recursion_batched_projection
                        .decomposition_chunks
                        .ilog2() as usize,
                proj_config
                    .recursion_batched_projection
                    .decomposition_chunks
                    .ilog2() as usize,
            );

            let projection_constant_terms_embedded_combiner_evaluation =
                load_combiner_evaluation_data(
                    proj_config.recursion_constant_term.decomposition_base_log as u64,
                    proj_config.recursion_constant_term.decomposition_chunks,
                    total_vars,
                );

            let projection_constant_terms_embedded_combiner_constant_evaluation =
                load_combiner_constant_evaluation(
                    proj_config.recursion_constant_term.decomposition_base_log as u64,
                    proj_config.recursion_constant_term.decomposition_chunks,
                    total_vars,
                );

            let recomposed_projection = ElephantCell::new(DiffSumcheckEvaluation::new(
                ElephantCell::new(ProductSumcheckEvaluation::new(
                    combined_witness_evaluation.clone(),
                    projection_combiner_evaluation.clone(),
                )),
                projection_combiner_constant_evaluation.clone(),
            ));

            let recomposed_projection_constant_terms_embedded =
                ElephantCell::new(DiffSumcheckEvaluation::new(
                    ElephantCell::new(ProductSumcheckEvaluation::new(
                        combined_witness_evaluation.clone(),
                        projection_constant_terms_embedded_combiner_evaluation.clone(),
                    )),
                    projection_constant_terms_embedded_combiner_constant_evaluation.clone(),
                ));

            let projection_constant_terms_embedded_selector_evaluation =
                selector_evaluation_from_prefix(
                    &proj_config.recursion_constant_term.prefix,
                    total_vars,
                );

            let mut lhs_scalar_consistency_evaluation_field = ElephantCell::new(
                BasicEvaluationLinearSumcheck::<QuadraticExtension>::new_with_prefixed_sufixed_data(
                    1, total_vars, 0,
                ),
            );

            lhs_scalar_consistency_evaluation_field
                .borrow_mut()
                .load_from(&[ONE_QUAD.clone()]);

            let lhs_scalar_consistency_evaluation = ElephantCell::new(
                RingToFieldWrapperEvaluation::new(lhs_scalar_consistency_evaluation_field.clone()),
            );

            let contexts: [Type3_1VerifierContext; NOF_BATCHES] = std::array::from_fn(|i| {
                // Split coefficients into block indices (elder vars) and within-block (LS vars)
                let height = config.projection_height;
                let inner_width = config.projection_ratio * height / DEGREE;
                let blocks = config.witness_height / inner_width;
                let lhs_flatter_0_evaluation_field = ElephantCell::new(
                    StructuredRowEvaluationLinearSumcheck::<QuadraticExtension>::new_with_prefixed_sufixed_data(
                        blocks,
                        total_vars
                            - blocks.ilog2() as usize
                            - inner_width.ilog2() as usize
                            - config.witness_decomposition_chunks.ilog2() as usize,
                        inner_width.ilog2() as usize
                            + config.witness_decomposition_chunks.ilog2() as usize,
                    ),
                );
                let lhs_flatter_1_times_matrix_evaluation = basic_evaluation_linear(
                    inner_width,
                    total_vars
                        - inner_width.ilog2() as usize
                        - config.witness_decomposition_chunks.ilog2() as usize,
                    config.witness_decomposition_chunks.ilog2() as usize,
                );
                let projection_selector_evaluation = selector_evaluation_from_prefix(
                    &Prefix {
                        prefix: proj_config.recursion_batched_projection.prefix.prefix
                            * NOF_BATCHES
                            + i,
                        length: proj_config.recursion_batched_projection.prefix.length
                            + NOF_BATCHES.ilog2() as usize,
                    },
                    total_vars,
                );

                let lhs_flatter_0_evaluation = ElephantCell::new(
                    RingToFieldWrapperEvaluation::new(lhs_flatter_0_evaluation_field.clone()),
                );

                let projection_coeff_product = ElephantCell::new(ProductSumcheckEvaluation::new(
                    lhs_flatter_0_evaluation.clone(),
                    lhs_flatter_1_times_matrix_evaluation.clone(),
                ));

                let lhs = ElephantCell::new(ProductSumcheckEvaluation::new(
                    folded_witness_selector_evaluation.clone(),
                    ElephantCell::new(ProductSumcheckEvaluation::new(
                        recomposed_folded_witness.clone(),
                        projection_coeff_product.clone(),
                    )),
                ));

                let rhs = ElephantCell::new(ProductSumcheckEvaluation::new(
                    projection_selector_evaluation.clone(),
                    ElephantCell::new(ProductSumcheckEvaluation::new(
                        recomposed_projection.clone(),
                        rhs_fold_challenge_evaluation.clone(),
                    )),
                ));

                let output = ElephantCell::new(DiffSumcheckEvaluation::new(lhs, rhs));

                let lhs_consistency_flatter_evaluation_field = ElephantCell::new(
                    StructuredRowEvaluationLinearSumcheck::<QuadraticExtension>::new_with_prefixed_sufixed_data(
                        config.witness_width,
                        total_vars
                            - config.witness_width.ilog2() as usize
                            - proj_config
                                .recursion_batched_projection
                                .decomposition_chunks
                                .ilog2() as usize,
                        proj_config
                            .recursion_batched_projection
                            .decomposition_chunks
                            .ilog2() as usize,
                    ),
                );

                let lhs_consistency_flatter_evaluation =
                    ElephantCell::new(RingToFieldWrapperEvaluation::new(
                        lhs_consistency_flatter_evaluation_field.clone(),
                    ));

                let lhs = ElephantCell::new(ProductSumcheckEvaluation::new(
                    lhs_scalar_consistency_evaluation.clone(),
                    ElephantCell::new(ProductSumcheckEvaluation::new(
                        projection_selector_evaluation.clone(),
                        ElephantCell::new(ProductSumcheckEvaluation::new(
                            lhs_consistency_flatter_evaluation.clone(),
                            recomposed_projection.clone(),
                        )),
                    )),
                ));

                // c_2 \otimes c_0 \otimes e_0
                let rhs_flatter_len =
                    config.witness_width * blocks * config.projection_height / DEGREE;

                let rhs_consistency_flatter_evaluation_field = ElephantCell::new(
                    StructuredRowEvaluationLinearSumcheck::<QuadraticExtension>::new_with_prefixed_sufixed_data(
                        rhs_flatter_len,
                        total_vars
                            - rhs_flatter_len.ilog2() as usize
                            - proj_config
                                .recursion_constant_term
                                .decomposition_chunks
                                .ilog2() as usize,
                        proj_config
                            .recursion_constant_term
                            .decomposition_chunks
                            .ilog2() as usize,
                    ),
                );

                let rhs_consistency_flatter_evaluation =
                    ElephantCell::new(RingToFieldWrapperEvaluation::new(
                        rhs_consistency_flatter_evaluation_field.clone(),
                    ));

                let mut rhs_scalar_consistency_evaluation = ElephantCell::new(
                    BasicEvaluationLinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                        1, total_vars, 0,
                    ),
                );

                let rhs = ElephantCell::new(ProductSumcheckEvaluation::new(
                    rhs_scalar_consistency_evaluation.clone(),
                    ElephantCell::new(ProductSumcheckEvaluation::new(
                        projection_constant_terms_embedded_selector_evaluation.clone(),
                        ElephantCell::new(ProductSumcheckEvaluation::new(
                            rhs_consistency_flatter_evaluation.clone(),
                            recomposed_projection_constant_terms_embedded.clone(),
                        )),
                    )),
                ));

                let output_consistency = ElephantCell::new(DiffSumcheckEvaluation::new(lhs, rhs));

                Type3_1VerifierContext {
                    lhs_flatter_0_evaluation_field,
                    lhs_flatter_0_evaluation,
                    lhs_flatter_1_times_matrix_evaluation,
                    projection_selector_evaluation,
                    output,
                    lhs_consistency_flatter_evaluation_field,
                    lhs_consistency_flatter_evaluation,
                    rhs_consistency_flatter_evaluation_field,
                    rhs_consistency_flatter_evaluation,
                    rhs_scalar_consistency_evaluation,
                    output_2: output_consistency,
                }
            });
            Some(Type3_1VerifierContextWrapper {
                sumchecks: contexts,
                projection_combiner_constant_evaluation,
                projection_combiner_evaluation,
                rhs_fold_challenge_evaluation,
                lhs_scalar_consistency_evaluation_field,
                lhs_scalar_consistency_evaluation,
            })
        }
        _ => None,
    };

    let mut type4evaluations = vec![
        build_type4_verifier_context(
            crs,
            total_vars,
            combined_witness_evaluation.clone(),
            &config.commitment_recursion,
        ),
        build_type4_verifier_context(
            crs,
            total_vars,
            combined_witness_evaluation.clone(),
            &config.opening_recursion,
        ),
    ];

    match &config.projection_recursion {
        Projection::Type0(proj_config) => {
            type4evaluations.push(build_type4_verifier_context(
                crs,
                total_vars,
                combined_witness_evaluation.clone(),
                &proj_config,
            ));
        }
        Projection::Type1(proj_config) => {
            type4evaluations.push(build_type4_verifier_context(
                crs,
                total_vars,
                combined_witness_evaluation.clone(),
                &proj_config.recursion_constant_term,
            ));

            type4evaluations.push(build_type4_verifier_context(
                crs,
                total_vars,
                combined_witness_evaluation.clone(),
                &proj_config.recursion_batched_projection,
            ));
        }
    }
    let conjugated_combined_witness_evaluation =
        ElephantCell::new(FakeEvaluationLinearSumcheck::<RingElement>::new());

    let mut most_inner_commitments_selectors = vec![];

    let most_inner_commitment_recursion = selector_evaluation_from_prefix(
        &config.commitment_recursion.most_inner_config().prefix,
        total_vars,
    );

    most_inner_commitments_selectors.push(most_inner_commitment_recursion);

    let most_inner_opening_recursion = selector_evaluation_from_prefix(
        &config.opening_recursion.most_inner_config().prefix,
        total_vars,
    );

    most_inner_commitments_selectors.push(most_inner_opening_recursion);

    match &config.projection_recursion {
        Projection::Type0(proj_config) => {
            let most_inner_projection_recursion = selector_evaluation_from_prefix(
                &proj_config.most_inner_config().prefix,
                total_vars,
            );
            most_inner_commitments_selectors.push(most_inner_projection_recursion);
        }
        Projection::Type1(proj_config) => {
            let most_inner_constant_term_recursion = selector_evaluation_from_prefix(
                &proj_config
                    .recursion_constant_term
                    .most_inner_config()
                    .prefix,
                total_vars,
            );
            most_inner_commitments_selectors.push(most_inner_constant_term_recursion);

            let most_inner_batched_projection_recursion = selector_evaluation_from_prefix(
                &proj_config
                    .recursion_batched_projection
                    .most_inner_config()
                    .prefix,
                total_vars,
            );
            most_inner_commitments_selectors.push(most_inner_batched_projection_recursion);
        }
    }

    let mut sum_of_selectors: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>> =
        most_inner_commitments_selectors[0].clone();

    for selector in most_inner_commitments_selectors.iter().skip(1) {
        sum_of_selectors = ElephantCell::new(SumSumcheckEvaluation::new(
            sum_of_selectors.clone(),
            selector.clone(),
        ));
    }

    let output = ElephantCell::new(ProductSumcheckEvaluation::new(
        combined_witness_evaluation.clone(),
        conjugated_combined_witness_evaluation.clone(),
    ));

    let output_2 = ElephantCell::new(ProductSumcheckEvaluation::new(
        sum_of_selectors.clone(),
        output.clone(),
    ));

    let type5evaluation = Type5VerifierContext {
        conjugated_combined_witness_evaluation: conjugated_combined_witness_evaluation.clone(),
        output,
        selectors: most_inner_commitments_selectors,
        output_2,
    };

    let mut all_outputs: Vec<ElephantCell<EvalData>> = vec![];
    for type0 in &type0evaluations {
        all_outputs.push(type0.output.clone());
    }
    for type1 in &type1evaluations {
        all_outputs.push(type1.output.clone());
    }
    for type2 in &type2evaluations {
        all_outputs.push(type2.output.clone());
    }
    if let Some(type3evaluation) = &type3evaluation {
        all_outputs.push(type3evaluation.output.clone());
    }
    if let Some(type3_1_evaluations) = &type3_1_evaluations {
        for type3_1 in &type3_1_evaluations.sumchecks {
            all_outputs.push(type3_1.output.clone());
            all_outputs.push(type3_1.output_2.clone());
        }
    }

    for type4 in &type4evaluations {
        for layer in &type4.layers {
            for output in &layer.outputs {
                all_outputs.push(output.clone());
            }
        }
        for output in &type4.output_layer.outputs {
            all_outputs.push(output.clone());
        }
    }
    all_outputs.push(type5evaluation.output.clone());
    all_outputs.push(type5evaluation.output_2.clone());

    let combiner_evaluation = ElephantCell::new(CombinerEvaluation::new(all_outputs));
    let field_combiner_evaluation = ElephantCell::new(RingToFieldCombinerEvaluation::new(
        combiner_evaluation.clone(),
    ));

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
        type0evaluations,
        type1evaluations,
        type2evaluations,
        type3evaluation,
        type3_1_evaluations,
        type4evaluations,
        type5evaluation,
        combiner_evaluation,
        field_combiner_evaluation,
        next: match &config.next {
            Some(next_config) => match next_config.as_ref() {
                Config::Sumcheck(next_sumcheck_config) => {
                    Some(Box::new(init_verifier(crs, next_sumcheck_config)))
                }
                _ => None,
            },
            None => None,
        },
    }
}
