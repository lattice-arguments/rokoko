use crate::common::arithmetic::ONE;
use crate::common::config::DEGREE;
use crate::protocol::config::{Projection, SumcheckConfig};
use crate::protocol::sumcheck_utils::sum::SumSumcheck;
use crate::protocol::sumchecks::context::Type3_1SumcheckContextWrapper;
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

use super::{
    context::{
        SumcheckContext, Type0SumcheckContext, Type1SumcheckContext, Type2SumcheckContext,
        Type3SumcheckContext, Type3_1SumcheckContext, Type4LayerSumcheckContext,
        Type4OutputLayerSumcheckContext, Type4SumcheckContext,
    },
    helpers::{ck_sumcheck, composition_sumcheck, sumcheck_from_prefix},
};

/// Builds sumcheck gadgets for recursive commitment verification.
///
/// For each internal layer i, proves: CK_i · witness_i = compose(child_commitment_{i+1})
/// where compose() reconstructs the parent from decomposed child chunks.
///
/// Each layer context contains:
/// - Selectors: identify witness slices for current and child layers
/// - Combiner: recomposes values from decomposed chunks (radix weights + signed-digit offset)
/// - CK sumchecks: one per rank, loaded with commitment key rows
/// - Output constraints: DiffSumcheck enforcing selector_i · (CK_i · witness_i) = selector_{i+1} · compose(witness_{i+1})
///
/// The leaf layer anchors to the public commitment value. RefCell enables shared mutation during folding.
fn build_type4_sumcheck_context(
    crs: &CRS,
    total_vars: usize,
    combined_witness_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
    config: &commitment::RecursionConfig,
) -> Type4SumcheckContext {
    let mut layers = Vec::new();
    let mut current = config;
    while let Some(next) = current.next.as_deref() {
        let selector_sumcheck = sumcheck_from_prefix(&current.prefix, total_vars);
        // let child_selector_sumcheck = sumcheck_from_prefix(&next.prefix, total_vars);

        let child_selector_sumchecks = (0..current.rank)
            .into_iter()
            .map(|i| {
                sumcheck_from_prefix(
                    &Prefix {
                        prefix: next.prefix.prefix * current.rank.next_power_of_two() + i,
                        length: next.prefix.length
                            + current.rank.next_power_of_two().ilog2() as usize,
                    },
                    total_vars,
                )
            })
            .collect::<Vec<_>>();

        let data_len = 1 << (total_vars - current.prefix.length);

        let data_selected_sumcheck = ElephantCell::new(ProductSumcheck::new(
            selector_sumcheck.clone(),
            combined_witness_sumcheck.clone(),
        ));

        let combiner_sumcheck = composition_sumcheck(
            next.decomposition_base_log as u64,
            next.decomposition_chunks,
            total_vars,
        );

        let recomposed_child_raw = ElephantCell::new(ProductSumcheck::new(
            combined_witness_sumcheck.clone(),
            combiner_sumcheck.clone(),
        ));

        // let recomposed_child_sumcheck = ElephantCell::new(ProductSumcheck::new(
        //     child_selector_sumcheck.clone(),
        //     recomposed_child_raw,
        // ));

        let recomposed_child_sumchecks = (0..current.rank)
            .into_iter()
            .map(|i| {
                ElephantCell::new(ProductSumcheck::new(
                    child_selector_sumchecks[i].clone(),
                    recomposed_child_raw.clone(),
                ))
            })
            .collect::<Vec<_>>();

        let blockwise_rank = current.rank / current.diag_blocks;
        let block_data_len = data_len / current.diag_blocks;

        let block_selector_sumchecks = if current.diag_blocks > 1 {
            Some(
                (0..current.diag_blocks)
                    .map(|b| {
                        sumcheck_from_prefix(
                            &Prefix {
                                prefix: current.prefix.prefix
                                    * current.diag_blocks.next_power_of_two()
                                    + b,
                                length: current.prefix.length
                                    + current.diag_blocks.next_power_of_two().ilog2() as usize,
                            },
                            total_vars,
                        )
                    })
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };

        let mut ck_sumchecks = Vec::with_capacity(blockwise_rank);
        for i in 0..blockwise_rank {
            ck_sumchecks.push(ck_sumcheck(crs, total_vars, block_data_len, i, 0));
        }

        let outputs = (0..current.rank)
            .map(|i| {
                let i_in_block = i % blockwise_rank;
                let block_i = i / blockwise_rank;

                let lhs = if let Some(ref block_sels) = block_selector_sumchecks {
                    let block_data_selected = ElephantCell::new(ProductSumcheck::new(
                        block_sels[block_i].clone(),
                        combined_witness_sumcheck.clone(),
                    ));
                    ElephantCell::new(ProductSumcheck::new(
                        ck_sumchecks[i_in_block].clone(),
                        block_data_selected,
                    ))
                } else {
                    ElephantCell::new(ProductSumcheck::new(
                        ck_sumchecks[i_in_block].clone(),
                        data_selected_sumcheck.clone(),
                    ))
                };

                let rhs = recomposed_child_sumchecks[i].clone();
                ElephantCell::new(DiffSumcheck::new(lhs, rhs))
            })
            .collect::<Vec<_>>();

        layers.push(Type4LayerSumcheckContext {
            selector_sumcheck,
            child_selector_sumcheck: Some(child_selector_sumchecks),
            block_selector_sumchecks,
            combiner_sumcheck: Some(combiner_sumcheck),
            data_selected_sumcheck,
            // rhs_sumcheck: recomposed_child_sumcheck,
            commitment_sumcheck: None,
            ck_sumchecks,
            outputs,
        });

        current = next;
    }
    assert_eq!(
        current.diag_blocks, 1,
        "Type4 Sumcheck: Your type0 recursion configuration is invalid. diag_blocks > 1 is currently not supported for type4 sumcheck output layers. Please set diag_blocks to 1 in your configuration."
    );
    // Build the output (leaf) layer
    // This is the base case that checks against the public commitment value
    let selector_sumcheck = sumcheck_from_prefix(&current.prefix, total_vars);

    let mut ck_sumchecks = Vec::with_capacity(current.rank);
    for i in 0..current.rank {
        ck_sumchecks.push(ck_sumcheck(
            crs,
            total_vars,
            1 << (total_vars - current.prefix.length),
            i,
            0,
        ));
    }

    let outputs = ck_sumchecks
        .iter()
        .map(|ck_row| {
            let output = ElephantCell::new(ProductSumcheck::new(
                selector_sumcheck.clone(),
                ElephantCell::new(ProductSumcheck::new(
                    combined_witness_sumcheck.clone(),
                    ck_row.clone(),
                )),
            ));
            output
        })
        .collect::<Vec<_>>();

    Type4SumcheckContext {
        layers,
        output_layer: Type4OutputLayerSumcheckContext {
            selector_sumcheck,
            ck_sumchecks,
            outputs,
        },
    }
}

/// Constructs all sumcheck gadgets for constraint verification:
///   - Type0: CK · folded_witness = commitment · fold_challenge
///   - Type1: inner_eval · folded_witness = opening.rhs · fold_challenge
///   - Type2: outer_eval · opening.rhs = claimed_evaluation
///   - Type3: projection_coeffs · folded_witness = fold_tensor · projection_image (block-diagonal)
///   - Type3_1: c^T (I ⊗ P) · folded_witness = c^T projection_image · fold_challenge (Kronecker) + consistency checks for batched projections
///   - Type4: recursive commitment well-formedness at each layer
///   - Type5: witness norm via <combined_witness, conjugate>
///             (also, we derive a specialised sumcheck for the most outer commitment layer)
///
/// Prefix padding enables composition without reindexing. Decomposition
/// offsets are preloaded to match commitment arithmetic.
pub fn init_sumcheck(crs: &crs::CRS, config: &SumcheckConfig) -> SumcheckContext {
    let total_vars = config.composed_witness_length.ilog2() as usize;

    let combined_witness_sumcheck = ElephantCell::new(LinearSumcheck::<RingElement>::new(
        config.composed_witness_length,
    ));

    let folded_witness_selector_sumcheck =
        sumcheck_from_prefix(&config.folded_witness_prefix, total_vars);

    let blockwise_rank = config.basic_commitment_rank / config.basic_commitment_diag_blocks;
    let commitment_key_rows_sumcheck = (0..blockwise_rank)
        .map(|i| {
            ck_sumcheck(
                crs,
                total_vars,
                config.witness_height / config.basic_commitment_diag_blocks,
                i,
                config.witness_decomposition_chunks.ilog2() as usize,
            )
        })
        .collect::<Vec<ElephantCell<LinearSumcheck<RingElement>>>>();

    let folded_witness_combiner_sumcheck = composition_sumcheck(
        config.witness_decomposition_base_log as u64,
        config.witness_decomposition_chunks,
        config.composed_witness_length.ilog2() as usize,
    );

    let basic_commitment_combiner_sumcheck = composition_sumcheck(
        config.commitment_recursion.decomposition_base_log as u64,
        config.commitment_recursion.decomposition_chunks,
        config.composed_witness_length.ilog2() as usize,
    );

    let opening_combiner_sumcheck = composition_sumcheck(
        config.opening_recursion.decomposition_base_log as u64,
        config.opening_recursion.decomposition_chunks,
        config.composed_witness_length.ilog2() as usize,
    );

    let folding_challenges_sumcheck = ElephantCell::new(
        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
            config.witness_width,
            config.composed_witness_length.ilog2() as usize
                - config.witness_width.ilog2() as usize
                - config.commitment_recursion.decomposition_chunks.ilog2() as usize,
            config.commitment_recursion.decomposition_chunks.ilog2() as usize,
        ),
    );

    // Type0 sumchecks
    // CK \cdot folded_witness - commitment \cdot fold_challenge = 0
    // We have CK = I_b \otimes CK_i, folded_witness - commitment \cdot fold_challenge = 0
    let type0sumchecks = (0..config.basic_commitment_rank)
        .map(|i| {
            let basic_commitment_row_sumcheck = sumcheck_from_prefix(
                &Prefix {
                    prefix: config.commitment_recursion.prefix.prefix
                        * config.basic_commitment_rank.next_power_of_two()
                        + i,
                    length: config.commitment_recursion.prefix.length
                        + config.basic_commitment_rank.next_power_of_two().ilog2() as usize,
                },
                total_vars,
            );

            let i_in_block = i % blockwise_rank; // e.g. for rank 8 with 2 diag blocks, the pattern of i_in_block is [0,1, 0,1, 0,1, 0,1]
            let block_i = i / blockwise_rank; // e.g. for rank 8 with 2 diag blocks, the pattern of block_i is [0,0, 1,1, 2,2, 3,3]

            // TODO: maybe better to split into two selectors, one to be reused across blocks?
            let folded_witness_block_selector_sumcheck = sumcheck_from_prefix(
                &Prefix {
                    prefix: config.folded_witness_prefix.prefix
                        * config.basic_commitment_diag_blocks.next_power_of_two()
                        + block_i,
                    length: config.folded_witness_prefix.length
                        + config
                            .basic_commitment_diag_blocks
                            .next_power_of_two()
                            .ilog2() as usize,
                },
                total_vars,
            );

            let ctxt = Type0SumcheckContext {
                folded_witness_block_selector_sumcheck: folded_witness_block_selector_sumcheck
                    .clone(),
                basic_commitment_row_sumcheck: basic_commitment_row_sumcheck.clone(),
                output: ElephantCell::new(DiffSumcheck::new(
                    ElephantCell::new(ProductSumcheck::new(
                        folded_witness_block_selector_sumcheck.clone(),
                        ElephantCell::new(ProductSumcheck::new(
                            ElephantCell::new(ProductSumcheck::new(
                                combined_witness_sumcheck.clone(),
                                folded_witness_combiner_sumcheck.clone(),
                            )),
                            commitment_key_rows_sumcheck[i_in_block].clone(),
                        )),
                    )),
                    ElephantCell::new(ProductSumcheck::new(
                        basic_commitment_row_sumcheck,
                        ElephantCell::new(ProductSumcheck::new(
                            ElephantCell::new(ProductSumcheck::new(
                                combined_witness_sumcheck.clone(),
                                basic_commitment_combiner_sumcheck.clone(),
                            )),
                            folding_challenges_sumcheck.clone(),
                        )),
                    )),
                )),
            };
            ctxt
        })
        .collect::<Vec<Type0SumcheckContext>>();

    // Type1 sumchecks
    // inner_evaluation_points \cdot folded_witness - opening.rhs \cdot fold_challenge = 0

    let recomposed_folded_witness = ElephantCell::new(ProductSumcheck::new(
        combined_witness_sumcheck.clone(),
        folded_witness_combiner_sumcheck.clone(),
    ));

    let recomposed_opening = ElephantCell::new(ProductSumcheck::new(
        combined_witness_sumcheck.clone(),
        opening_combiner_sumcheck.clone(),
    ));

    let type1sumchecks = (0..config.nof_openings)
        .map(|i| {
            let opening_selector_sumcheck = sumcheck_from_prefix(
                &Prefix {
                    prefix: config.opening_recursion.prefix.prefix * config.nof_openings + i,
                    length: config.opening_recursion.prefix.length
                        + config.nof_openings.ilog2() as usize,
                },
                total_vars,
            );

            let inner_evaluation_sumcheck = ElephantCell::new(
                LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                    config.witness_height,
                    total_vars
                        - config.witness_height.ilog2() as usize
                        - config.witness_decomposition_chunks.ilog2() as usize,
                    config.witness_decomposition_chunks.ilog2() as usize,
                ),
            );

            let lhs = ElephantCell::new(ProductSumcheck::new(
                folded_witness_selector_sumcheck.clone(),
                ElephantCell::new(ProductSumcheck::new(
                    recomposed_folded_witness.clone(),
                    inner_evaluation_sumcheck.clone(),
                )),
            ));

            let rhs = ElephantCell::new(ProductSumcheck::new(
                opening_selector_sumcheck.clone(),
                ElephantCell::new(ProductSumcheck::new(
                    recomposed_opening.clone(),
                    folding_challenges_sumcheck.clone(),
                )),
            ));

            let output = ElephantCell::new(DiffSumcheck::new(lhs, rhs));

            Type1SumcheckContext {
                inner_evaluation_sumcheck,
                opening_selector_sumcheck,
                output,
            }
        })
        .collect::<Vec<Type1SumcheckContext>>();

    // Type2 sumchecks
    // <opening.rhs[i], outer_evaluation_points> = evaluations[i] (public)
    let type2sumchecks = type1sumchecks
        .iter()
        .map(|type1_sc| {
            let outer_evaluation_sumcheck = ElephantCell::new(
                LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                    config.witness_width,
                    total_vars
                        - config.witness_width.ilog2() as usize
                        - config.opening_recursion.decomposition_chunks.ilog2() as usize,
                    config.opening_recursion.decomposition_chunks.ilog2() as usize,
                ),
            );

            let output = ElephantCell::new(ProductSumcheck::new(
                type1_sc.opening_selector_sumcheck.clone(),
                ElephantCell::new(ProductSumcheck::new(
                    recomposed_opening.clone(),
                    outer_evaluation_sumcheck.clone(),
                )),
            ));

            Type2SumcheckContext {
                outer_evaluation_sumcheck,
                output,
            }
        })
        .collect::<Vec<Type2SumcheckContext>>();

    // type3 sumchecks
    // projection_matrix_flatter \cdot (I \otimes projection_matrix) \cdot folded_witness - projection_matrix_flatter \cdot projection_image \cdot fold_challenge = 0
    // Here, we treat projection_matrix_flatter \cdot (I \otimes projection_matrix) as a single multilinear polynomial
    // Also, we treat projection_matrix_flatter \tensor fold_challenge as a single multilinear polynomial

    // It corresponds to:
    // \sum_z Diff(Prod(projection_matrix_flatter \cdot (I \otimes projection_matrix), folded_witness), Prod(projection_matrix_flatter \tensor fold_challenge, projection_image))
    // change to:
    // \sum_z Diff(Prod(projection_matrix_flatter_0, Prod(projection_matrix_flatter_1 \cdot (I \otimes projection_matrix), folded_witness)), Prod(Prod(projection_matrix_flatter, Prod(fold_challenge, projection_image))

    let projection_height_flat = config.witness_height / config.projection_ratio;
    let type3sumcheck = match &config.projection_recursion {
        Projection::Type0(projection_recursion) => {
            let projection_selector_sumcheck =
                sumcheck_from_prefix(&projection_recursion.prefix, total_vars);

            let projection_combiner_sumcheck = composition_sumcheck(
                projection_recursion.decomposition_base_log as u64,
                projection_recursion.decomposition_chunks,
                config.composed_witness_length.ilog2() as usize,
            );

            let recomposed_projection = ElephantCell::new(ProductSumcheck::new(
                combined_witness_sumcheck.clone(),
                projection_combiner_sumcheck.clone(),
            ));

            // Split projection coefficients into two parts:
            // 1. projection_flatter_0: elder variables (block indices)
            // 2. projection_flatter_1 · matrix: LS variables (within-block)
            let height = config.projection_height;
            let inner_width = config.projection_ratio * height;
            let blocks = config.witness_height / inner_width;

            if blocks == 0 {
                panic!("Type3 Sumcheck: Your type0 projection configuration is invalid. The number of blocks computed as witness_height / (projection_ratio * projection_height) is zero. Please check your configuration.");
            }

            // Elder variables: projection_flatter_0 (length = blocks)
            let lhs_flatter_0_sumcheck = ElephantCell::new(
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

            // LS variables: projection_flatter_1 · matrix (length = inner_width)
            let lhs_flatter_1_times_matrix_sumcheck = ElephantCell::new(
                LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                    inner_width,
                    total_vars
                        - inner_width.ilog2() as usize
                        - config.witness_decomposition_chunks.ilog2() as usize,
                    config.witness_decomposition_chunks.ilog2() as usize,
                ),
            );

            // Combined projection coefficients via Product
            let projection_coeff_product = ElephantCell::new(ProductSumcheck::new(
                lhs_flatter_0_sumcheck.clone(),
                lhs_flatter_1_times_matrix_sumcheck.clone(),
            ));

            // Split RHS into Product of two LinearSumchecks:
            let rhs_fold_challenge_sumcheck = ElephantCell::new(
                LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                    config.witness_width,
                    total_vars
                        - config.witness_width.ilog2() as usize
                        - projection_height_flat.ilog2() as usize
                        - projection_recursion.decomposition_chunks.ilog2() as usize,
                    projection_height_flat.ilog2() as usize
                        + projection_recursion.decomposition_chunks.ilog2() as usize,
                ),
            );

            let rhs_projection_flatter_sumcheck = ElephantCell::new(
                LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                    projection_height_flat,
                    total_vars
                        - projection_height_flat.ilog2() as usize
                        - projection_recursion.decomposition_chunks.ilog2() as usize,
                    projection_recursion.decomposition_chunks.ilog2() as usize,
                ),
            );

            let rhs_fold_tensor_product = ElephantCell::new(ProductSumcheck::new(
                rhs_fold_challenge_sumcheck.clone(),
                rhs_projection_flatter_sumcheck.clone(),
            ));

            let lhs = ElephantCell::new(ProductSumcheck::new(
                folded_witness_selector_sumcheck.clone(),
                ElephantCell::new(ProductSumcheck::new(
                    recomposed_folded_witness.clone(),
                    projection_coeff_product,
                )),
            ));
            let rhs = ElephantCell::new(ProductSumcheck::new(
                projection_selector_sumcheck.clone(),
                ElephantCell::new(ProductSumcheck::new(
                    recomposed_projection.clone(),
                    rhs_fold_tensor_product,
                )),
            ));
            let output = ElephantCell::new(DiffSumcheck::new(lhs, rhs));

            Some(Type3SumcheckContext {
                projection_combiner_sumcheck,
                lhs_flatter_0_sumcheck,
                lhs_flatter_1_times_matrix_sumcheck,
                rhs_fold_challenge_sumcheck,
                rhs_projection_flatter_sumcheck,
                projection_selector_sumcheck,
                output,
            })
        }
        _ => None,
    };

    // let type_3_1_sumchecks = match &config.projection_recursion {
    // Type3_1_A sumchecks for batched projections
    // Similar to type3 but for each batch: c_0'^T (I ⊗ j_batched) · folded_witness = projection_image_i · fold_challenge
    // c_0 and c_1 are u64 challenges that need to be lifted to RingElement
    // j_batched is already a Vec<RingElement>
    let type3_1_sumchecks = match &config.projection_recursion {
        Projection::Type1(projection_recursion) => {
            // Projection combiner for decomposition (same for all batches)
            let projection_combiner_sumcheck = {
                composition_sumcheck(
                    projection_recursion
                        .recursion_batched_projection
                        .decomposition_base_log as u64,
                    projection_recursion
                        .recursion_batched_projection
                        .decomposition_chunks,
                    config.composed_witness_length.ilog2() as usize,
                )
            };

            let projection_constant_terms_embedded_combiner_sumcheck = composition_sumcheck(
                projection_recursion
                    .recursion_constant_term
                    .decomposition_base_log as u64,
                projection_recursion
                    .recursion_constant_term
                    .decomposition_chunks,
                config.composed_witness_length.ilog2() as usize,
            );
            let recomposed_projection = ElephantCell::new(ProductSumcheck::new(
                combined_witness_sumcheck.clone(),
                projection_combiner_sumcheck.clone(),
            ));

            let recomposed_projection_constant_terms_embedded =
                ElephantCell::new(ProductSumcheck::new(
                    combined_witness_sumcheck.clone(),
                    projection_constant_terms_embedded_combiner_sumcheck.clone(),
                ));
            let projection_constant_terms_embedded_selector_sumcheck = sumcheck_from_prefix(
                &projection_recursion.recursion_constant_term.prefix,
                total_vars,
            );

            // RHS: fold_challenge (same for all batches)
            let rhs_fold_challenge_sumcheck = ElephantCell::new(
                LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                    config.witness_width,
                    total_vars
                        - config.witness_width.ilog2() as usize
                        - projection_recursion
                            .recursion_batched_projection
                            .decomposition_chunks
                            .ilog2() as usize,
                    projection_recursion
                        .recursion_batched_projection
                        .decomposition_chunks
                        .ilog2() as usize,
                ),
            );

            let lhs_scalar_consistency_sumcheck = ElephantCell::new(
                LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(1, total_vars, 0),
            );

            lhs_scalar_consistency_sumcheck
                .borrow_mut()
                .load_from(&[ONE.clone()]);
            // Build one context per batch
            // Each batch has its own projection result stored at a different prefix location
            let contexts: [Type3_1SumcheckContext; NOF_BATCHES] = std::array::from_fn(|i| {
                // Create selectors and combiners similar to type3
                // Note: We'll load the actual challenge data (c_0, c_1, j_batched) in the loader

                // Split coefficients into block indices (elder vars) and within-block (LS vars)
                let height = config.projection_height;
                let inner_width = config.projection_ratio * height / DEGREE;
                let blocks = config.witness_height / inner_width;

                // Elder variables: c_0 coefficients (block indices)
                let lhs_flatter_0_sumcheck = ElephantCell::new(
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

                // LS variables: c_1 · j_batched (within-block coefficients)
                let lhs_flatter_1_times_matrix_sumcheck = ElephantCell::new(
                    LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                        inner_width,
                        total_vars
                            - inner_width.ilog2() as usize
                            - config.witness_decomposition_chunks.ilog2() as usize,
                        config.witness_decomposition_chunks.ilog2() as usize,
                    ),
                );

                // Selector for batch i's projection
                // Each batch occupies a distinct prefix within the recursion_batched_projection tree
                let projection_selector_sumcheck = sumcheck_from_prefix(
                    &Prefix {
                        prefix: projection_recursion
                            .recursion_batched_projection
                            .prefix
                            .prefix
                            * NOF_BATCHES
                            + i,
                        length: projection_recursion
                            .recursion_batched_projection
                            .prefix
                            .length
                            + NOF_BATCHES.ilog2() as usize,
                    },
                    total_vars,
                );

                // Build the constraint tree
                let projection_coeff_product = ElephantCell::new(ProductSumcheck::new(
                    lhs_flatter_0_sumcheck.clone(),
                    lhs_flatter_1_times_matrix_sumcheck.clone(),
                ));

                let lhs = ElephantCell::new(ProductSumcheck::new(
                    folded_witness_selector_sumcheck.clone(),
                    ElephantCell::new(ProductSumcheck::new(
                        recomposed_folded_witness.clone(),
                        projection_coeff_product,
                    )),
                ));

                let rhs = ElephantCell::new(ProductSumcheck::new(
                    projection_selector_sumcheck.clone(),
                    ElephantCell::new(ProductSumcheck::new(
                        recomposed_projection.clone(),
                        rhs_fold_challenge_sumcheck.clone(),
                    )),
                ));

                let output = ElephantCell::new(DiffSumcheck::new(lhs, rhs));

                let lhs_consistency_flatter_sumcheck = ElephantCell::new(
                    LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                        config.witness_width,
                        total_vars
                            - config.witness_width.ilog2() as usize
                            - projection_recursion
                                .recursion_batched_projection
                                .decomposition_chunks
                                .ilog2() as usize,
                        projection_recursion
                            .recursion_batched_projection
                            .decomposition_chunks
                            .ilog2() as usize,
                    ),
                );

                let lhs = ElephantCell::new(ProductSumcheck::new(
                    lhs_scalar_consistency_sumcheck.clone(),
                    ElephantCell::new(ProductSumcheck::new(
                        projection_selector_sumcheck.clone(),
                        ElephantCell::new(ProductSumcheck::new(
                            lhs_consistency_flatter_sumcheck.clone(),
                            recomposed_projection.clone(),
                        )),
                    )),
                ));

                // c_2 \otimes c_0 \otimes e_0
                let rhs_flatter_len =
                    config.witness_width * blocks * config.projection_height / DEGREE;

                let rhs_consistency_flatter_sumcheck = ElephantCell::new(
                    LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                        rhs_flatter_len,
                        total_vars
                            - rhs_flatter_len.ilog2() as usize
                            - projection_recursion
                                .recursion_constant_term
                                .decomposition_chunks
                                .ilog2() as usize,
                        projection_recursion
                            .recursion_constant_term
                            .decomposition_chunks
                            .ilog2() as usize,
                    ),
                );

                let rhs_scalar_consistency_sumcheck = ElephantCell::new(
                    LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(1, total_vars, 0),
                );

                let rhs = ElephantCell::new(ProductSumcheck::new(
                    rhs_scalar_consistency_sumcheck.clone(),
                    ElephantCell::new(ProductSumcheck::new(
                        projection_constant_terms_embedded_selector_sumcheck.clone(),
                        ElephantCell::new(ProductSumcheck::new(
                            rhs_consistency_flatter_sumcheck.clone(),
                            recomposed_projection_constant_terms_embedded.clone(),
                        )),
                    )),
                ));

                let output_consistency = ElephantCell::new(DiffSumcheck::new(lhs, rhs));

                Type3_1SumcheckContext {
                    lhs_flatter_0_sumcheck,
                    lhs_flatter_1_times_matrix_sumcheck,
                    projection_selector_sumcheck,
                    output,
                    lhs_consistency_flatter_sumcheck,
                    rhs_scalar_consistency_sumcheck,
                    rhs_consistency_flatter_sumcheck,
                    output_2: output_consistency,
                }
            });

            Some(Type3_1SumcheckContextWrapper {
                sumchecks: contexts,
                projection_combiner_sumcheck,
                projection_constant_terms_embedded_combiner_sumcheck,
                rhs_fold_challenge_sumcheck,
                lhs_scalar_consistency_sumcheck,
                projection_constant_terms_embedded_selector_sumcheck,
            })
        }
        _ => None,
    };

    let conjugated_combined_witness_sumcheck = ElephantCell::new(
        LinearSumcheck::<RingElement>::new(config.composed_witness_length),
    );

    let mut most_inner_commitments_selectors = Vec::new();

    let most_inner_commitment_recursion = sumcheck_from_prefix(
        &config.commitment_recursion.most_inner_config().prefix,
        total_vars,
    );

    most_inner_commitments_selectors.push(most_inner_commitment_recursion);

    let most_inner_opening_recursion = sumcheck_from_prefix(
        &config.opening_recursion.most_inner_config().prefix,
        total_vars,
    );

    most_inner_commitments_selectors.push(most_inner_opening_recursion);

    // if let Some(config.p
    match config.projection_recursion {
        Projection::Type0(ref proj_config) => {
            let most_inner_projection_recursion =
                sumcheck_from_prefix(&proj_config.most_inner_config().prefix, total_vars);
            most_inner_commitments_selectors.push(most_inner_projection_recursion);
        }
        Projection::Type1(ref proj_config) => {
            let most_inner_constant_term_recursion = sumcheck_from_prefix(
                &proj_config
                    .recursion_constant_term
                    .most_inner_config()
                    .prefix,
                total_vars,
            );
            most_inner_commitments_selectors.push(most_inner_constant_term_recursion);
            let most_inner_batched_projection_recursion = sumcheck_from_prefix(
                &proj_config
                    .recursion_batched_projection
                    .most_inner_config()
                    .prefix,
                total_vars,
            );
            most_inner_commitments_selectors.push(most_inner_batched_projection_recursion);
        }
        Projection::Skip => {
            // No type4 sumcheck for projection
        }
    }

    let mut sum_of_selectors: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>> =
        most_inner_commitments_selectors[0].clone();

    for selector in most_inner_commitments_selectors.iter().skip(1) {
        sum_of_selectors =
            ElephantCell::new(SumSumcheck::new(sum_of_selectors.clone(), selector.clone()));
    }

    let output = ElephantCell::new(ProductSumcheck::new(
        combined_witness_sumcheck.clone(),
        conjugated_combined_witness_sumcheck.clone(),
    ));

    let output_2 = ElephantCell::new(ProductSumcheck::new(
        sum_of_selectors.clone(),
        output.clone(),
    ));

    let type5sumcheck = Type5SumcheckContext {
        conjugated_combined_witness: conjugated_combined_witness_sumcheck.clone(),
        output,
        selectors: most_inner_commitments_selectors,
        output_2,
    };

    // Type4 sumchecks: Three separate recursive commitment trees
    // 1. Commitment recursion: verifies the basic witness commitments are well-formed
    // 2. Opening recursion: verifies the opening proofs are correctly committed
    // 3. Projection recursion: verifies the projection images are correctly committed
    // Each tree has its own depth, rank, and decomposition parameters defined in config.

    let mut type4sumchecks = vec![
        build_type4_sumcheck_context(
            crs,
            total_vars,
            combined_witness_sumcheck.clone(),
            &config.commitment_recursion,
        ),
        build_type4_sumcheck_context(
            crs,
            total_vars,
            combined_witness_sumcheck.clone(),
            &config.opening_recursion,
        ),
    ];

    match &config.projection_recursion {
        Projection::Type0(recursion_config) => {
            type4sumchecks.push(build_type4_sumcheck_context(
                crs,
                total_vars,
                combined_witness_sumcheck.clone(),
                recursion_config,
            ));
        }
        Projection::Type1(recursion_config) => {
            type4sumchecks.push(build_type4_sumcheck_context(
                crs,
                total_vars,
                combined_witness_sumcheck.clone(),
                &recursion_config.recursion_constant_term,
            ));
            type4sumchecks.push(build_type4_sumcheck_context(
                crs,
                total_vars,
                combined_witness_sumcheck.clone(),
                &recursion_config.recursion_batched_projection,
            ));
        }
        Projection::Skip => {
            // No type4 sumcheck for projection
        }
    }

    let mut all_outputs: Vec<ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>> =
        vec![];
    for type0 in &type0sumchecks {
        all_outputs.push(type0.output.clone());
    }
    for type1 in &type1sumchecks {
        all_outputs.push(type1.output.clone());
    }
    for type2 in &type2sumchecks {
        all_outputs.push(type2.output.clone());
    }

    if let Some(type3sumcheck) = &type3sumcheck {
        all_outputs.push(type3sumcheck.output.clone());
    } else if let Some(type3_1_contexts) = &type3_1_sumchecks {
        for type3_1_ctx in type3_1_contexts.sumchecks.iter() {
            all_outputs.push(type3_1_ctx.output.clone());
            all_outputs.push(type3_1_ctx.output_2.clone());
        }
    }

    for type4 in &type4sumchecks {
        for layer in &type4.layers {
            for output in &layer.outputs {
                all_outputs.push(output.clone());
            }
        }
        for output in &type4.output_layer.outputs {
            all_outputs.push(output.clone());
        }
    }

    all_outputs.push(type5sumcheck.output.clone());
    all_outputs.push(type5sumcheck.output_2.clone());

    let combiner = ElephantCell::new(Combiner::new(all_outputs));

    let field_combiner = ElephantCell::new(RingToFieldCombiner::new(combiner.clone()));

    SumcheckContext {
        combined_witness_sumcheck: combined_witness_sumcheck.clone(),
        folded_witness_selector_sumcheck,
        folded_witness_combiner_sumcheck,
        folding_challenges_sumcheck,
        basic_commitment_combiner_sumcheck,
        commitment_key_rows_sumcheck,
        opening_combiner_sumcheck,
        type0sumchecks,
        type1sumchecks,
        type2sumchecks,
        type3sumcheck,
        type4sumchecks,
        type5sumcheck,
        type3_1_sumchecks,
        combiner,
        field_combiner,
        next: match &config.next {
            Some(next_config) => match next_config.as_ref() {
                Config::Sumcheck(next_simple_config) => {
                    Some(Box::new(init_sumcheck(crs, next_simple_config)))
                }
                Config::Simple(_) => None,
            },
            None => None,
        },
    }
}
