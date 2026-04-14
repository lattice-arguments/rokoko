use std::array;

use crate::common::arithmetic::ALL_ONE_COEFFS;
use crate::protocol::crs::CRS;
use crate::protocol::sumcheck_utils::diff::DiffSumcheck;
use crate::protocol::sumcheck_utils::ring_to_field_combiner::RingToFieldCombiner;
use crate::protocol::sumchecks::helpers::sumcheck_from_prefix;
use crate::{
    common::config::*,
    common::ring_arithmetic::RingElement,
    protocol::{
        config::RoundConfig,
        sumcheck_utils::{
            combiner::Combiner, common::HighOrderSumcheckData, elephant_cell::ElephantCell,
            linear::LinearSumcheck, product::ProductSumcheck,
        },
        sumchecks::context::{
            L2ProverSumcheckContext, LinfSumcheckContext, ProverSumcheckContext,
            Type1ProverSumcheckContext, Type31ProverSumcheckContext, Type3ProverSumcheckContext,
            VDFProverSumcheckContext,
        },
    },
};

fn init_prover_type_1_sumcheck(
    config: &RoundConfig,
    main_witness_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
) -> Type1ProverSumcheckContext {
    let single_col_height = (config.extended_witness_length >> config.main_witness_prefix.length)
        / config.main_witness_columns;
    let total_vars = config.extended_witness_length.ilog2() as usize;
    let inner_evaluation_sumcheck =
        ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
            single_col_height,
            total_vars - single_col_height.ilog2() as usize,
            0,
        ));

    let outer_evaluation_sumcheck =
        ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
            config.main_witness_columns,
            total_vars
                - config.main_witness_columns.ilog2() as usize
                - single_col_height.ilog2() as usize,
            single_col_height.ilog2() as usize,
        ));
    // let outer_evaluation_sumcheck = ElephantCell::new(LinearSumcheck::new(config.main / 2));
    // we view MLE[w](evaluation_points_inner) as a sumcheck
    let output = ElephantCell::new(ProductSumcheck::new(
        ElephantCell::new(ProductSumcheck::new(
            inner_evaluation_sumcheck.clone(),
            outer_evaluation_sumcheck.clone(),
        )),
        main_witness_sumcheck.clone(),
    ));

    Type1ProverSumcheckContext {
        inner_evaluation_sumcheck,
        outer_evaluation_sumcheck,
        output,
    }
}

fn init_prover_vdf_sumcheck(
    config: &RoundConfig,
    main_witness_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
) -> VDFProverSumcheckContext {
    let total_vars = config.extended_witness_length.ilog2() as usize;
    let two_k = config.extended_witness_length / 2 / VDF_MATRIX_WIDTH; // 2K = total VDF steps across both columns

    // vdf_step_powers: varies over log2(2K) middle variables (one per VDF step)
    // prefix = 1 (main_witness_selector bit), suffix = 6 (element-within-block bits)
    let vdf_step_powers_sumcheck = ElephantCell::new(
        LinearSumcheck::new_with_prefixed_sufixed_data(two_k, 1, VDF_MATRIX_WIDTH.ilog2() as usize),
    );

    // vdf_batched_row: varies over 6 LSB variables (element within 64-element block)
    // prefix = total_vars - 6 (all higher bits)
    let vdf_batched_row_sumcheck =
        ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
            VDF_MATRIX_WIDTH,
            total_vars - VDF_MATRIX_WIDTH.ilog2() as usize,
            0,
        ));

    let output = ElephantCell::new(ProductSumcheck::new(
        ElephantCell::new(ProductSumcheck::new(
            vdf_step_powers_sumcheck.clone(),
            vdf_batched_row_sumcheck.clone(),
        )),
        main_witness_sumcheck.clone(),
    ));

    VDFProverSumcheckContext {
        vdf_step_powers_sumcheck,
        vdf_batched_row_sumcheck,
        output,
    }
}

fn init_prover_type_3_sumcheck(
    config: &RoundConfig,
    main_witness_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
    projection_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
) -> Type3ProverSumcheckContext {
    match config {
        RoundConfig::Intermediate {
            projection_ratio,
            projection_prefix,
            ..
        } => {
            let c2_len = config.main_witness_columns;
            let c1_len = PROJECTION_HEIGHT;
            // (c_2 \otimes c_0 \otimes c_1^T J) · witness = (c_2 \otimes c_0 \otimes c_1)^T projected_witness
            let single_col_height =
                config.extended_witness_length / 2 / config.main_witness_columns;

            let c0_len: usize = single_col_height / (PROJECTION_HEIGHT * projection_ratio);
            assert!(c0_len > 0, "c0_len must be greater than 0");

            let total_vars = config.extended_witness_length.ilog2() as usize;

            assert_eq!(c0_len * c1_len * c2_len, config.extended_witness_length / (2_usize.pow(projection_prefix.length as u32)), "c0_len * c1_len * c2_len must be equal to extended_witness_length, got c0_len: {}, c1_len: {}, c2_len: {}, extended_witness_length: {}", c0_len, c1_len, c2_len, config.extended_witness_length);

            // We have the following variables structure:
            // LEFT
            // prefix
            // c_2.ilog2() variables for c_2
            // c_0.ilog2() variables for c_0
            // (c_1^T J).ilog2() variables for (c_1^T J)

            // RIGHT
            // prefix
            // c_2.ilog2() variables for c_2
            // c_0.ilog2() variables for c_0
            // c_1.ilog2() variables for c_1

            // left
            let fltr_len = (projection_ratio * PROJECTION_HEIGHT).ilog2() as usize;

            let flattened_projection_matrix_sumcheck =
                ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
                    projection_ratio * PROJECTION_HEIGHT,
                    total_vars - fltr_len,
                    0,
                ));
            let c0l_sumcheck = ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
                c0_len,
                total_vars - fltr_len - c0_len.ilog2() as usize,
                fltr_len,
            ));

            let c2l_sumcheck = ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
                c2_len,
                total_vars - fltr_len - c0_len.ilog2() as usize - c2_len.ilog2() as usize,
                fltr_len + c0_len.ilog2() as usize,
            ));

            // right
            let c1r_sumcheck = ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
                c1_len,
                total_vars - c1_len.ilog2() as usize,
                0,
            ));

            let c0r_sumcheck = ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
                c0_len,
                total_vars - c1_len.ilog2() as usize - c0_len.ilog2() as usize,
                c1_len.ilog2() as usize,
            ));

            let c2r_sumcheck = ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
                c2_len,
                total_vars
                    - c1_len.ilog2() as usize
                    - c0_len.ilog2() as usize
                    - c2_len.ilog2() as usize,
                c1_len.ilog2() as usize + c0_len.ilog2() as usize,
            ));

            let lhs = ElephantCell::new(ProductSumcheck::new(
                c2l_sumcheck.clone(),
                ElephantCell::new(ProductSumcheck::new(
                    c0l_sumcheck.clone(),
                    ElephantCell::new(ProductSumcheck::new(
                        flattened_projection_matrix_sumcheck.clone(),
                        main_witness_sumcheck.clone(),
                    )),
                )),
            ));

            let rhs = ElephantCell::new(ProductSumcheck::new(
                c2r_sumcheck.clone(),
                ElephantCell::new(ProductSumcheck::new(
                    c0r_sumcheck.clone(),
                    ElephantCell::new(ProductSumcheck::new(
                        c1r_sumcheck.clone(),
                        projection_sumcheck.clone(),
                    )),
                )),
            ));

            let output = ElephantCell::new(DiffSumcheck::new(lhs.clone(), rhs.clone()));

            Type3ProverSumcheckContext {
                flattened_projection_matrix_sumcheck,
                c0l_sumcheck,
                c2l_sumcheck,
                c1r_sumcheck,
                c0r_sumcheck,
                c2r_sumcheck,
                lhs,
                rhs,
                output,
            }
        }
        _ => panic!("type 3 sumcheck should only be initialized for rounds with projection"),
    }
}

fn init_prover_type_3_1_sumcheck(
    config: &RoundConfig,
    main_witness_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
) -> Type31ProverSumcheckContext {
    let projection_ratio = match config {
        RoundConfig::IntermediateUnstructured {
            projection_ratio, ..
        }
        | RoundConfig::Last {
            projection_ratio, ..
        } => *projection_ratio,
        _ => panic!("type 3.1 sumcheck should only be initialized for rounds with projection"),
    };
    let total_vars = config.extended_witness_length.ilog2() as usize;
    let single_col_height = config.extended_witness_length / config.main_witness_columns; // no projection in sc for unstructured round
    let c0_len: usize = DEGREE * single_col_height / (PROJECTION_HEIGHT * projection_ratio); // typically, 1
    let c2_len: usize = config.main_witness_columns;

    let c_2_sumcheck = ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
        c2_len,
        0,
        total_vars - c2_len.ilog2() as usize,
    ));

    // c_0_sumcheck across blocks
    let c_0_sumcheck = ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
        c0_len,
        c2_len.ilog2() as usize,
        total_vars - c2_len.ilog2() as usize - c0_len.ilog2() as usize,
    ));

    // j_batched_sumcheck across elements within block
    let j_batched_sumcheck = ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
        single_col_height / c0_len,
        c2_len.ilog2() as usize + c0_len.ilog2() as usize,
        0,
    ));

    let output = ElephantCell::new(ProductSumcheck::new(
        c_2_sumcheck.clone(),
        ElephantCell::new(ProductSumcheck::new(
            c_0_sumcheck.clone(),
            ElephantCell::new(ProductSumcheck::new(
                j_batched_sumcheck.clone(),
                main_witness_sumcheck.clone(),
            )),
        )),
    ));

    Type31ProverSumcheckContext {
        c_2_sumcheck,
        c_0_sumcheck,
        j_batched_sumcheck,
        output,
    }
}

pub fn init_linf_sumcheck(
    witness_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
    main_witness_selector: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
    conjugated_witness_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
) -> LinfSumcheckContext {
    let all_one_constant_sumcheck =
        ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
            1,
            witness_sumcheck.borrow().variable_count(),
            0,
        ));

    all_one_constant_sumcheck
        .borrow_mut()
        .load_from(&vec![ALL_ONE_COEFFS.clone()]);

    let one_minus_wit_sumcheck = ElephantCell::new(DiffSumcheck::new(
        all_one_constant_sumcheck.clone(),
        witness_sumcheck.clone(),
    ));

    let one_minus_wit_selector_sumcheck = ElephantCell::new(ProductSumcheck::new(
        main_witness_selector.clone(),
        one_minus_wit_sumcheck.clone(),
    ));

    let output = ElephantCell::new(ProductSumcheck::new(
        conjugated_witness_sumcheck.clone(),
        one_minus_wit_selector_sumcheck.clone(),
    ));

    LinfSumcheckContext {
        all_one_constant_sumcheck,
        output,
    }
}

pub fn init_prover_sumcheck(crs: &CRS, config: &RoundConfig) -> ProverSumcheckContext {
    let witness_sumcheck = ElephantCell::new(LinearSumcheck::new(config.extended_witness_length));
    let witness_conjugated_sumcheck =
        ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
            config.extended_witness_length >> config.main_witness_prefix.length,
            config.main_witness_prefix.length as usize,
            0,
        ));

    let main_witness_selector_sumcheck = sumcheck_from_prefix(
        &config.main_witness_prefix,
        config.extended_witness_length.ilog2() as usize,
    );

    let main_witness_sumcheck: ElephantCell<ProductSumcheck<_>> =
        ElephantCell::new(ProductSumcheck::new(
            witness_sumcheck.clone(),
            main_witness_selector_sumcheck.clone(),
        ));

    let projection_selector_sumcheck = match config {
        RoundConfig::Intermediate {
            projection_prefix, ..
        } => Some(sumcheck_from_prefix(
            &projection_prefix,
            config.extended_witness_length.ilog2() as usize,
        )),
        _ => None,
    };

    let projection_sumcheck = match config {
        RoundConfig::Intermediate { .. } => Some(ElephantCell::new(ProductSumcheck::new(
            witness_sumcheck.clone(),
            projection_selector_sumcheck.as_ref().unwrap().clone(),
        ))),
        _ => None,
    };

    let type1sumcheck = (0..config.inner_evaluation_claims)
        .map(|_| init_prover_type_1_sumcheck(config, main_witness_sumcheck.clone()))
        .collect::<Vec<_>>();

    let type3sumcheck = match config {
        RoundConfig::Intermediate { .. } => Some(init_prover_type_3_sumcheck(
            config,
            main_witness_sumcheck.clone(),
            projection_sumcheck.clone().unwrap(),
        )),
        _ => None,
    };

    let type31sumchecks = match config {
        RoundConfig::IntermediateUnstructured { .. } | RoundConfig::Last { .. } => {
            Some(array::from_fn(|_| {
                init_prover_type_3_1_sumcheck(config, main_witness_sumcheck.clone())
            }))
        }
        _ => None,
    };

    let l2sumcheck = if config.l2 {
        Some(L2ProverSumcheckContext {
            output: ElephantCell::new(ProductSumcheck::new(
                witness_conjugated_sumcheck.clone(),
                main_witness_sumcheck.clone(),
            )),
        })
    } else {
        None
    };

    let linfsumcheck = if config.exact_binariness {
        Some(init_linf_sumcheck(
            witness_sumcheck.clone(),
            main_witness_selector_sumcheck.clone(),
            witness_conjugated_sumcheck.clone(),
        ))
    } else {
        None
    };

    let vdfsumcheck = if config.vdf {
        Some(init_prover_vdf_sumcheck(
            config,
            main_witness_sumcheck.clone(),
        ))
    } else {
        None
    };

    let mut all_outputs: Vec<ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>> =
        vec![];
    for type1 in &type1sumcheck {
        all_outputs.push(type1.output.clone());
    }

    if let Some(type3) = &type3sumcheck {
        all_outputs.push(type3.output.clone());
    }
    if let Some(l2) = &l2sumcheck {
        all_outputs.push(l2.output.clone());
    }
    if let Some(linf) = &linfsumcheck {
        all_outputs.push(linf.output.clone());
    }
    if let Some(vdf) = &vdfsumcheck {
        all_outputs.push(vdf.output.clone());
    }
    if let Some(type31) = &type31sumchecks {
        for sc in type31 {
            all_outputs.push(sc.output.clone());
        }
    }

    let combiner = ElephantCell::new(Combiner::new(all_outputs));
    let field_combiner = ElephantCell::new(RingToFieldCombiner::new(combiner.clone()));

    ProverSumcheckContext {
        witness_sumcheck,
        witness_conjugated_sumcheck,
        main_witness_selector_sumcheck,
        projection_selector_sumcheck,
        type1sumcheck,
        type3sumcheck,
        type31sumchecks,
        combiner,
        field_combiner,
        l2sumcheck,
        linfsumcheck,
        vdfsumcheck,
        next: match config {
            RoundConfig::Intermediate { next, .. } => {
                Some(Box::new(init_prover_sumcheck(crs, next)))
            }
            RoundConfig::IntermediateUnstructured { next, .. } => {
                Some(Box::new(init_prover_sumcheck(crs, next)))
            }
            RoundConfig::Last { .. } => None,
        },
    }
}
