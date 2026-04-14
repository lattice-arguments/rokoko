use crate::{
    common::{
        arithmetic::ALL_ONE_COEFFS,
        config::*,
        ring_arithmetic::{QuadraticExtension, RingElement},
    },
    protocol::{
        commitment::Prefix,
        config::RoundConfig,
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
        },
        sumchecks::context_verifier::{
            L2VerifierSumcheckContext, LinfVerifierSumcheckContext, Type1VerifierSumcheckContext,
            Type31VerifierSumcheckContext, Type3VerifierSumcheckContext,
            VDFVerifierSumcheckContext, VerifierSumcheckContext,
        },
    },
};
use std::array;

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

fn init_verifier_type_3_1_sumcheck(
    config: &RoundConfig,
    main_witness_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
) -> Type31VerifierSumcheckContext {
    let projection_ratio = match config {
        RoundConfig::IntermediateUnstructured {
            projection_ratio, ..
        }
        | RoundConfig::Last {
            projection_ratio, ..
        } => *projection_ratio,
        _ => panic!("type 3.1 verifier sumcheck should only be initialized for rounds with unstructured projection"),
    };
    let total_vars = config.extended_witness_length.ilog2() as usize;
    let single_col_height = config.extended_witness_length / config.main_witness_columns;
    let c0_len: usize = DEGREE * single_col_height / (PROJECTION_HEIGHT * projection_ratio);
    let c2_len: usize = config.main_witness_columns;

    let c_2_evaluation = ElephantCell::new(
        StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            c2_len,
            0,
            total_vars - c2_len.ilog2() as usize,
        ),
    );

    let c_0_evaluation = ElephantCell::new(
        StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            c0_len,
            c2_len.ilog2() as usize,
            total_vars - c2_len.ilog2() as usize - c0_len.ilog2() as usize,
        ),
    );

    let j_batched_evaluation = ElephantCell::new(
        BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            single_col_height / c0_len,
            c2_len.ilog2() as usize + c0_len.ilog2() as usize,
            0,
        ),
    );

    let output = ElephantCell::new(ProductSumcheckEvaluation::new(
        c_2_evaluation.clone(),
        ElephantCell::new(ProductSumcheckEvaluation::new(
            c_0_evaluation.clone(),
            ElephantCell::new(ProductSumcheckEvaluation::new(
                j_batched_evaluation.clone(),
                main_witness_evaluation.clone(),
            )),
        )),
    ));

    Type31VerifierSumcheckContext {
        c_2_evaluation,
        c_0_evaluation,
        j_batched_evaluation,
        output,
    }
}

fn init_verifier_type_1_sumcheck(
    config: &RoundConfig,
    main_witness_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
) -> Type1VerifierSumcheckContext {
    let single_col_height = (config.extended_witness_length >> config.main_witness_prefix.length)
        / config.main_witness_columns;
    let total_vars = config.extended_witness_length.ilog2() as usize;

    let inner_evaluation_sumcheck = ElephantCell::new(
        StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            single_col_height,
            total_vars - single_col_height.ilog2() as usize,
            0,
        ),
    );

    let outer_evaluation_sumcheck = ElephantCell::new(
        BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            config.main_witness_columns,
            total_vars
                - config.main_witness_columns.ilog2() as usize
                - single_col_height.ilog2() as usize,
            single_col_height.ilog2() as usize,
        ),
    );

    let output = ElephantCell::new(ProductSumcheckEvaluation::new(
        ElephantCell::new(ProductSumcheckEvaluation::new(
            inner_evaluation_sumcheck.clone(),
            outer_evaluation_sumcheck.clone(),
        )),
        main_witness_evaluation.clone(),
    ));

    Type1VerifierSumcheckContext {
        inner_evaluation_sumcheck,
        outer_evaluation_sumcheck,
        output,
    }
}

fn init_verifier_type_3_sumcheck(
    config: &RoundConfig,
    main_witness_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
    projection_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
) -> Type3VerifierSumcheckContext {
    match config {
        RoundConfig::Intermediate {
            projection_ratio, ..
        } => {
            let c2_len = config.main_witness_columns;
            let c1_len = PROJECTION_HEIGHT;
            let single_col_height =
                config.extended_witness_length / 2 / config.main_witness_columns;
            let c0_len: usize = single_col_height / (PROJECTION_HEIGHT * projection_ratio);
            let total_vars = config.extended_witness_length.ilog2() as usize;

            // LEFT: prefix, c2, c0, flattened_projection_matrix (c1^T J)
            let fltr_len = (projection_ratio * PROJECTION_HEIGHT).ilog2() as usize;

            let flattened_projection_matrix_evaluation = ElephantCell::new(
                BasicEvaluationLinearSumcheck::<QuadraticExtension>::new_with_prefixed_sufixed_data(
                    projection_ratio * PROJECTION_HEIGHT,
                    total_vars - fltr_len,
                    0,
                ),
            );
            let c0l_evaluation = ElephantCell::new(
                StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
                    c0_len,
                    total_vars - fltr_len - c0_len.ilog2() as usize,
                    fltr_len,
                ),
            );
            let c2l_evaluation = ElephantCell::new(
                StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
                    c2_len,
                    total_vars - fltr_len - c0_len.ilog2() as usize - c2_len.ilog2() as usize,
                    fltr_len + c0_len.ilog2() as usize,
                ),
            );

            // RIGHT: prefix, c2, c0, c1
            let c1r_evaluation = ElephantCell::new(
                StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
                    c1_len,
                    total_vars - c1_len.ilog2() as usize,
                    0,
                ),
            );
            let c0r_evaluation = ElephantCell::new(
                StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
                    c0_len,
                    total_vars - c1_len.ilog2() as usize - c0_len.ilog2() as usize,
                    c1_len.ilog2() as usize,
                ),
            );
            let c2r_evaluation = ElephantCell::new(
                StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
                    c2_len,
                    total_vars
                        - c1_len.ilog2() as usize
                        - c0_len.ilog2() as usize
                        - c2_len.ilog2() as usize,
                    c1_len.ilog2() as usize + c0_len.ilog2() as usize,
                ),
            );

            let lhs = ElephantCell::new(ProductSumcheckEvaluation::new(
                c2l_evaluation.clone(),
                ElephantCell::new(ProductSumcheckEvaluation::new(
                    c0l_evaluation.clone(),
                    ElephantCell::new(ProductSumcheckEvaluation::new(
                        ElephantCell::new(RingToFieldWrapperEvaluation::new(
                            flattened_projection_matrix_evaluation.clone(),
                        )),
                        main_witness_evaluation.clone(),
                    )),
                )),
            ));

            let rhs = ElephantCell::new(ProductSumcheckEvaluation::new(
                c2r_evaluation.clone(),
                ElephantCell::new(ProductSumcheckEvaluation::new(
                    c0r_evaluation.clone(),
                    ElephantCell::new(ProductSumcheckEvaluation::new(
                        c1r_evaluation.clone(),
                        projection_evaluation.clone(),
                    )),
                )),
            ));

            let output = ElephantCell::new(DiffSumcheckEvaluation::new(lhs.clone(), rhs.clone()));

            Type3VerifierSumcheckContext {
                c2l_evaluation,
                c0l_evaluation,
                flattened_projection_matrix_evaluation,
                c2r_evaluation,
                c0r_evaluation,
                c1r_evaluation,
                lhs,
                rhs,
                output,
            }
        }
        _ => panic!(
            "Type 3 sumcheck should only be initialized for intermediate rounds with projection"
        ),
    }
}

fn init_verifier_l2_sumcheck(
    witness_conjugated_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
    main_witness_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
) -> L2VerifierSumcheckContext {
    L2VerifierSumcheckContext {
        output: ElephantCell::new(ProductSumcheckEvaluation::new(
            witness_conjugated_evaluation,
            main_witness_evaluation,
        )),
    }
}

fn init_verifier_linf_sumcheck(
    witness_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
    main_witness_selector_evaluation: ElephantCell<
        dyn EvaluationSumcheckData<Element = RingElement>,
    >,
    witness_conjugated_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
) -> LinfVerifierSumcheckContext {
    let all_one_constant_evaluation =
        ElephantCell::new(FakeEvaluationLinearSumcheck::<RingElement>::new());
    all_one_constant_evaluation
        .borrow_mut()
        .set_result(ALL_ONE_COEFFS.clone());

    let one_minus_wit_evaluation = ElephantCell::new(DiffSumcheckEvaluation::new(
        all_one_constant_evaluation.clone(),
        witness_evaluation,
    ));

    let one_minus_wit_selector_evaluation = ElephantCell::new(ProductSumcheckEvaluation::new(
        main_witness_selector_evaluation,
        one_minus_wit_evaluation.clone(),
    ));

    let output = ElephantCell::new(ProductSumcheckEvaluation::new(
        witness_conjugated_evaluation.clone(),
        one_minus_wit_selector_evaluation.clone(),
    ));

    LinfVerifierSumcheckContext {
        one_minus_wit_evaluation,
        one_minus_wit_selector_evaluation,
        all_one_constant_evaluation,
        output,
    }
}

fn init_verifier_vdf_sumcheck(
    config: &RoundConfig,
    main_witness_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
) -> VDFVerifierSumcheckContext {
    let total_vars = config.extended_witness_length.ilog2() as usize;

    let vdf_step_powers_evaluation =
        ElephantCell::new(FakeEvaluationLinearSumcheck::<RingElement>::new());

    let vdf_batched_row_evaluation = ElephantCell::new(
        BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            VDF_MATRIX_WIDTH,
            total_vars - VDF_MATRIX_WIDTH.ilog2() as usize,
            0,
        ),
    );

    let output = ElephantCell::new(ProductSumcheckEvaluation::new(
        ElephantCell::new(ProductSumcheckEvaluation::new(
            vdf_step_powers_evaluation.clone(),
            vdf_batched_row_evaluation.clone(),
        )),
        main_witness_evaluation.clone(),
    ));

    VDFVerifierSumcheckContext {
        vdf_step_powers_evaluation,
        vdf_batched_row_evaluation,
        output,
    }
}

pub fn init_verifier_sumcheck(config: &RoundConfig) -> VerifierSumcheckContext {
    let total_vars = config.extended_witness_length.ilog2() as usize;

    let witness_evaluation = ElephantCell::new(FakeEvaluationLinearSumcheck::<RingElement>::new());
    let witness_conjugated_evaluation =
        ElephantCell::new(FakeEvaluationLinearSumcheck::<RingElement>::new());

    let main_witness_selector_evaluation =
        selector_evaluation_from_prefix(&config.main_witness_prefix, total_vars);
    let projection_selector_evaluation = match config {
        RoundConfig::Intermediate {
            projection_prefix, ..
        } => Some(selector_evaluation_from_prefix(
            projection_prefix,
            total_vars,
        )),
        _ => None,
    };

    let main_witness_evaluation: ElephantCell<ProductSumcheckEvaluation> =
        ElephantCell::new(ProductSumcheckEvaluation::new(
            witness_evaluation.clone(),
            main_witness_selector_evaluation.clone(),
        ));

    let projection_eval = match config {
        RoundConfig::Intermediate {
            projection_prefix, ..
        } => Some(ElephantCell::new(ProductSumcheckEvaluation::new(
            witness_evaluation.clone(),
            selector_evaluation_from_prefix(projection_prefix, total_vars),
        ))),
        _ => None,
    };

    let type1evaluations = (0..config.inner_evaluation_claims)
        .map(|_| init_verifier_type_1_sumcheck(config, main_witness_evaluation.clone()))
        .collect::<Vec<_>>();

    let type3evaluation = match config {
        RoundConfig::Intermediate { projection_ratio, .. } => Some(init_verifier_type_3_sumcheck(
            config,
            main_witness_evaluation.clone(),
            projection_eval.expect("Projection evaluation should be initialized for intermediate rounds with projection"),
        )),
        _ => None,
    };

    let l2evaluation = if config.l2 {
        Some(init_verifier_l2_sumcheck(
            witness_conjugated_evaluation.clone(),
            main_witness_evaluation.clone(),
        ))
    } else {
        None
    };

    let linfevaluation = if config.exact_binariness {
        Some(init_verifier_linf_sumcheck(
            witness_evaluation.clone(),
            main_witness_selector_evaluation.clone(),
            witness_conjugated_evaluation.clone(),
        ))
    } else {
        None
    };

    let vdfevaluation = if config.vdf {
        Some(init_verifier_vdf_sumcheck(
            config,
            main_witness_evaluation.clone(),
        ))
    } else {
        None
    };

    let type31evaluations = match config {
        RoundConfig::IntermediateUnstructured { .. } | RoundConfig::Last { .. } => {
            Some(array::from_fn(|_| {
                init_verifier_type_3_1_sumcheck(config, main_witness_evaluation.clone())
            }))
        }
        _ => None,
    };

    let mut all_outputs: Vec<ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>> =
        vec![];
    for type1 in &type1evaluations {
        all_outputs.push(type1.output.clone());
    }
    if let Some(type3) = &type3evaluation {
        all_outputs.push(type3.output.clone());
    }
    if let Some(l2) = &l2evaluation {
        all_outputs.push(l2.output.clone());
    }
    if let Some(linf) = &linfevaluation {
        all_outputs.push(linf.output.clone());
    }
    if let Some(vdf) = &vdfevaluation {
        all_outputs.push(vdf.output.clone());
    }
    if let Some(type31) = &type31evaluations {
        for sc in type31 {
            all_outputs.push(sc.output.clone());
        }
    }

    let combiner_evaluation = ElephantCell::new(CombinerEvaluation::new(all_outputs));
    let field_combiner_evaluation = ElephantCell::new(RingToFieldCombinerEvaluation::new(
        combiner_evaluation.clone(),
    ));

    VerifierSumcheckContext {
        witness_evaluation,
        witness_conjugated_evaluation,
        main_witness_selector_evaluation,
        projection_selector_evaluation,
        type1evaluations,
        type3evaluation,
        type31evaluations,
        l2evaluation,
        linfevaluation,
        vdfevaluation,
        combiner_evaluation,
        field_combiner_evaluation,
        next: match config {
            RoundConfig::Intermediate { next, .. } => Some(Box::new(init_verifier_sumcheck(next))),
            RoundConfig::IntermediateUnstructured { next, .. } => {
                Some(Box::new(init_verifier_sumcheck(next)))
            }
            RoundConfig::Last { .. } => None,
        },
    }
}
