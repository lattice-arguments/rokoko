use crate::common::arithmetic::ONE;
use crate::common::config::DEGREE;
use crate::protocol::config::{Projection, SumcheckConfig};
use crate::protocol::intermediate_sumchecks::builder::init_intermediate_sumcheck;
use crate::protocol::sumcheck_utils::sum::SumSumcheck;
use crate::protocol::sumchecks::context::{FineProjSumcheckContextWrapper, NextSumcheckContext};
use crate::{
    common::{config::NOF_BATCHES, ring_arithmetic::RingElement},
    protocol::{
        commitment::{self},
        config::Config,
        crs::{self, CRS},
        sumcheck_utils::{
            combiner::Combiner, common::HighOrderSumcheckData, diff::DiffSumcheck,
            elephant_cell::ElephantCell, linear::LinearSumcheck, product::ProductSumcheck,
            ring_to_field_combiner::RingToFieldCombiner, selector_eq::SelectorEq,
        },
        sumchecks::context::NormCheckSumcheckContext,
    },
};

use super::{
    context::{
        CoarseProjSumcheckContext, ComVerifyLayerSumcheckContext,
        ComVerifyOutputLayerSumcheckContext, ComVerifySumcheckContext,
        CommitmentFoldSumcheckContext, FineProjSumcheckContext, InnerEvalFoldSumcheckContext,
        OuterEvalClaimSumcheckContext, SumcheckContext,
    },
    helpers::{
        ck_segment_sumcheck, ck_sumcheck, plane_selectors, raw_plane_selectors, sum_of,
        sumcheck_from_prefix, weighted_sum,
    },
};

type Data = ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>;
type Selectors = Vec<ElephantCell<SelectorEq<RingElement>>>;

/// Builds sumcheck gadgets for recursive commitment verification.
///
/// For each internal layer i, proves: CK_i . witness_i = compose(child_commitment_{i+1})
/// where compose() recomposes the parent out of the child's digit planes.
///
/// A level's decomposed input is placed one component per row, each cut into `chunks` digit
/// planes and each plane into dyadic blocks, so both sides of the constraint are sums:
///
///     lhs_i = SUM_{row, plane}  ck_i[slice(row, plane)] . (selector_{row, plane} . W)
///     rhs_i = SUM_{plane}       2^{base_log . plane} . (selector_{plane, element i} . W)
///
/// The leaf layer anchors to the public commitment value, and is the lhs alone.
/// The output count stays `rank` per layer: the sums happen inside one output.
fn placed_plane_products(
    total_vars: usize,
    config: &commitment::RecursionConfig,
    witness: &Data,
    selectors: &mut Selectors,
) -> Vec<Vec<ElephantCell<ProductSumcheck<RingElement>>>> {
    config
        .placements
        .iter()
        .map(|placement| {
            let planes =
                raw_plane_selectors(placement, config.decomposition_chunks, total_vars);
            selectors.extend(planes.iter().cloned());
            planes
                .iter()
                .map(|selector| {
                    ElephantCell::new(ProductSumcheck::new(selector.clone(), witness.clone()))
                })
                .collect()
        })
        .collect()
}

/// Commitment key row `i` met by the level's raw digits, one product per placed (row, plane).
fn ck_over_planes(
    crs: &CRS,
    total_vars: usize,
    config: &commitment::RecursionConfig,
    i: usize,
    data_selected: &[Vec<ElephantCell<ProductSumcheck<RingElement>>>],
    ck_sumchecks: &mut Vec<ElephantCell<LinearSumcheck<RingElement>>>,
) -> Data {
    let committed_len = config.committed_len();
    let slices = config.slices();
    let mut terms: Vec<Data> = Vec::new();

    for row in 0..config.placements.len() {
        for plane in 0..config.decomposition_chunks {
            let ck = ck_segment_sumcheck(
                crs,
                total_vars,
                committed_len,
                i,
                slices,
                config.slice_index(row, plane),
            );
            ck_sumchecks.push(ck.clone());
            terms.push(ElephantCell::new(ProductSumcheck::new(
                ck,
                data_selected[row][plane].clone(),
            )) as Data);
        }
    }

    sum_of(terms)
}

fn build_com_verify_sumcheck_context(
    crs: &CRS,
    total_vars: usize,
    combined_witness_sumcheck: Data,
    config: &commitment::RecursionConfig,
    selectors: &mut Selectors,
) -> ComVerifySumcheckContext {
    let mut layers = Vec::new();
    let mut current = config;
    while let Some(next) = current.next.as_deref() {
        debug_assert!(
            current
                .placements
                .iter()
                .all(|p| p.size == current.placements[0].size),
            "row components of one level are equally sized"
        );

        let data_selected = placed_plane_products(
            total_vars,
            current,
            &combined_witness_sumcheck,
            selectors,
        );

        let mut ck_sumchecks = Vec::with_capacity(
            current.rank * current.placements.len() * current.decomposition_chunks,
        );

        let outputs = (0..current.rank)
            .map(|i| {
                let lhs = ck_over_planes(
                    crs,
                    total_vars,
                    current,
                    i,
                    &data_selected,
                    &mut ck_sumchecks,
                );

                // The child level's input is this level's commitment; element i of it is
                // recomposed out of the child's digit planes.
                let child_planes = plane_selectors(
                    next.placement(),
                    next.decomposition_chunks,
                    next.decomposition_base_log,
                    total_vars,
                    current.rank,
                    i,
                );
                selectors.extend(child_planes.iter().cloned());
                let rhs = weighted_sum(&child_planes, combined_witness_sumcheck.clone());

                ElephantCell::new(DiffSumcheck::new(lhs, rhs))
            })
            .collect::<Vec<_>>();

        layers.push(ComVerifyLayerSumcheckContext {
            ck_sumchecks,
            outputs,
        });

        current = next;
    }

    // Build the output (leaf) layer
    // This is the base case that checks against the public commitment value
    let data_selected =
        placed_plane_products(total_vars, current, &combined_witness_sumcheck, selectors);
    let mut ck_sumchecks = Vec::with_capacity(
        current.rank * current.placements.len() * current.decomposition_chunks,
    );
    let outputs = (0..current.rank)
        .map(|i| ck_over_planes(crs, total_vars, current, i, &data_selected, &mut ck_sumchecks))
        .collect::<Vec<_>>();

    ComVerifySumcheckContext {
        layers,
        output_layer: ComVerifyOutputLayerSumcheckContext {
            ck_sumchecks,
            outputs,
        },
    }
}

/// Constructs all sumcheck gadgets for constraint verification:
///   - CommitmentFold: CK · folded_witness = commitment · fold_challenge
///   - InnerEvalFold: inner_eval · folded_witness = opening.rhs · fold_challenge
///   - OuterEvalClaim: outer_eval · opening.rhs = claimed_evaluation
///   - CoarseProj: projection_coeffs · folded_witness = fold_tensor · projection_image (block-diagonal)
///   - FineProj: c^T (I ⊗ P) · folded_witness = c^T projection_image · fold_challenge (Kronecker) + consistency checks for batched projections
///   - ComVerify: recursive commitment well-formedness at each layer
///   - NormCheck: witness norm via <combined_witness, conjugate>
///             (also, we derive a specialised sumcheck for the most outer commitment layer)
///
/// Prefix padding enables composition without reindexing. Decomposition
/// offsets are preloaded to match commitment arithmetic.
pub fn init_sumcheck(crs: &crs::CRS, config: &SumcheckConfig) -> SumcheckContext {
    let total_vars = config.composed_witness_length.ilog2() as usize;

    // Every selector built below is registered here and folded once per round; a selector that
    // is shared by several constraints is created once and registered once.
    let mut selectors: Selectors = Vec::new();

    let combined_witness_sumcheck = ElephantCell::new(LinearSumcheck::<RingElement>::new(
        config.composed_witness_length,
    ));
    let witness = combined_witness_sumcheck.clone() as Data;

    // The folded witness is placed digit-major, so recomposing it is a weighted sum over its
    // digit planes rather than a factor carrying the radix weights on the low variables.
    let folded_witness_planes = plane_selectors(
        &config.folded_witness_placement,
        config.witness_decomposition_chunks,
        config.witness_decomposition_base_log,
        total_vars,
        1,
        0,
    );
    selectors.extend(folded_witness_planes.iter().cloned());

    let commitment_key_rows_sumcheck = (0..config.basic_commitment_rank)
        .map(|i| ck_sumcheck(crs, total_vars, config.witness_height, i, 0))
        .collect::<Vec<ElephantCell<LinearSumcheck<RingElement>>>>();

    let folding_challenges_sumcheck = ElephantCell::new(
        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
            config.witness_width,
            total_vars - config.witness_width.ilog2() as usize,
            0,
        ),
    );

    let witness_with_folding_challenges = ElephantCell::new(ProductSumcheck::new(
        witness.clone(),
        folding_challenges_sumcheck.clone(),
    )) as Data;

    // CommitmentFold sumchecks
    // CK \cdot folded_witness - commitment \cdot fold_challenge = 0
    let commitment_fold_sumchecks = (0..config.basic_commitment_rank)
        .map(|i| {
            let basic_commitment_row_planes = plane_selectors(
                &config.commitment_recursion.placements[i],
                config.commitment_recursion.decomposition_chunks,
                config.commitment_recursion.decomposition_base_log,
                total_vars,
                1,
                0,
            );
            selectors.extend(basic_commitment_row_planes.iter().cloned());

            let lhs = weighted_sum(
                &folded_witness_planes,
                ElephantCell::new(ProductSumcheck::new(
                    witness.clone(),
                    commitment_key_rows_sumcheck[i].clone(),
                )) as Data,
            );

            let rhs = weighted_sum(
                &basic_commitment_row_planes,
                witness_with_folding_challenges.clone(),
            );

            CommitmentFoldSumcheckContext {
                output: ElephantCell::new(DiffSumcheck::new(lhs, rhs)),
            }
        })
        .collect::<Vec<CommitmentFoldSumcheckContext>>();

    // InnerEvalFold sumchecks
    // inner_evaluation_points \cdot folded_witness - opening.rhs \cdot fold_challenge = 0
    // One opening per row of the opening commitment, each with its own placement.
    let opening_planes = (0..config.nof_openings)
        .map(|i| {
            let planes = plane_selectors(
                &config.opening_recursion.placements[i],
                config.opening_recursion.decomposition_chunks,
                config.opening_recursion.decomposition_base_log,
                total_vars,
                1,
                0,
            );
            selectors.extend(planes.iter().cloned());
            planes
        })
        .collect::<Vec<_>>();

    let inner_eval_fold_sumchecks = (0..config.nof_openings)
        .map(|i| {
            let inner_evaluation_sumcheck = ElephantCell::new(
                LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                    config.witness_height,
                    total_vars - config.witness_height.ilog2() as usize,
                    0,
                ),
            );

            let lhs = weighted_sum(
                &folded_witness_planes,
                ElephantCell::new(ProductSumcheck::new(
                    witness.clone(),
                    inner_evaluation_sumcheck.clone(),
                )) as Data,
            );

            let rhs = weighted_sum(&opening_planes[i], witness_with_folding_challenges.clone());

            InnerEvalFoldSumcheckContext {
                inner_evaluation_sumcheck,
                output: ElephantCell::new(DiffSumcheck::new(lhs, rhs)),
            }
        })
        .collect::<Vec<InnerEvalFoldSumcheckContext>>();

    // OuterEvalClaim sumchecks
    // <opening.rhs[i], outer_evaluation_points> = evaluations[i] (public)
    let outer_eval_claim_sumchecks = (0..config.nof_openings)
        .map(|i| {
            let outer_evaluation_sumcheck = ElephantCell::new(
                LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                    config.witness_width,
                    total_vars - config.witness_width.ilog2() as usize,
                    0,
                ),
            );

            let output = weighted_sum(
                &opening_planes[i],
                ElephantCell::new(ProductSumcheck::new(
                    witness.clone(),
                    outer_evaluation_sumcheck.clone(),
                )) as Data,
            );

            OuterEvalClaimSumcheckContext {
                outer_evaluation_sumcheck,
                output,
            }
        })
        .collect::<Vec<OuterEvalClaimSumcheckContext>>();

    // coarse_proj sumchecks
    // projection_matrix_flatter \cdot (I \otimes projection_matrix) \cdot folded_witness - projection_matrix_flatter \cdot projection_image \cdot fold_challenge = 0
    // Here, we treat projection_matrix_flatter \cdot (I \otimes projection_matrix) as a single multilinear polynomial
    // Also, we treat projection_matrix_flatter \tensor fold_challenge as a single multilinear polynomial

    // It corresponds to:
    // \sum_z Diff(Prod(projection_matrix_flatter \cdot (I \otimes projection_matrix), folded_witness), Prod(projection_matrix_flatter \tensor fold_challenge, projection_image))
    // change to:
    // \sum_z Diff(Prod(projection_matrix_flatter_0, Prod(projection_matrix_flatter_1 \cdot (I \otimes projection_matrix), folded_witness)), Prod(Prod(projection_matrix_flatter, Prod(fold_challenge, projection_image))

    let projection_height_flat = config.witness_height / config.projection_ratio;
    let coarse_proj_sumcheck = match &config.projection_recursion {
        Projection::Coarse(projection_recursion) => {
            let projection_planes = plane_selectors(
                projection_recursion.placement(),
                projection_recursion.decomposition_chunks,
                projection_recursion.decomposition_base_log,
                total_vars,
                1,
                0,
            );
            selectors.extend(projection_planes.iter().cloned());

            // Split projection coefficients into two parts:
            // 1. projection_flatter_0: elder variables (block indices)
            // 2. projection_flatter_1 . matrix: LS variables (within-block)
            let height = config.projection_height;
            let inner_width = config.projection_ratio * height;
            let blocks = config.witness_height / inner_width;

            if blocks == 0 {
                panic!("Coarse-projection sumcheck: invalid configuration. The number of blocks computed as witness_height / (projection_ratio * projection_height) is zero. Please check your configuration.");
            }

            // Elder variables: projection_flatter_0 (length = blocks)
            let lhs_flatter_0_sumcheck = ElephantCell::new(
                LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                    blocks,
                    total_vars - blocks.ilog2() as usize - inner_width.ilog2() as usize,
                    inner_width.ilog2() as usize,
                ),
            );

            // LS variables: projection_flatter_1 . matrix (length = inner_width)
            let lhs_flatter_1_times_matrix_sumcheck = ElephantCell::new(
                LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                    inner_width,
                    total_vars - inner_width.ilog2() as usize,
                    0,
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
                        - projection_height_flat.ilog2() as usize,
                    projection_height_flat.ilog2() as usize,
                ),
            );

            let rhs_projection_flatter_sumcheck = ElephantCell::new(
                LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                    projection_height_flat,
                    total_vars - projection_height_flat.ilog2() as usize,
                    0,
                ),
            );

            let rhs_fold_tensor_product = ElephantCell::new(ProductSumcheck::new(
                rhs_fold_challenge_sumcheck.clone(),
                rhs_projection_flatter_sumcheck.clone(),
            ));

            let lhs = weighted_sum(
                &folded_witness_planes,
                ElephantCell::new(ProductSumcheck::new(witness.clone(), projection_coeff_product))
                    as Data,
            );
            let rhs = weighted_sum(
                &projection_planes,
                ElephantCell::new(ProductSumcheck::new(witness.clone(), rhs_fold_tensor_product))
                    as Data,
            );
            let output = ElephantCell::new(DiffSumcheck::new(lhs, rhs));

            Some(CoarseProjSumcheckContext {
                lhs_flatter_0_sumcheck,
                lhs_flatter_1_times_matrix_sumcheck,
                rhs_fold_challenge_sumcheck,
                rhs_projection_flatter_sumcheck,
                output,
            })
        }
        _ => None,
    };

    // let fine_proj_sumchecks = match &config.projection_recursion {
    // FineProj-consistency sumchecks for batched projections
    // Similar to coarse_proj but for each batch: c_0'^T (I ⊗ j_batched) · folded_witness = projection_image_i · fold_challenge
    // c_0 and c_1 are u64 challenges that need to be lifted to RingElement
    // j_batched is already a Vec<RingElement>
    let fine_proj_sumchecks = match &config.projection_recursion {
        Projection::Fine(projection_recursion) => {
            let projection_constant_terms_embedded_planes = plane_selectors(
                projection_recursion.recursion_constant_term.placement(),
                projection_recursion
                    .recursion_constant_term
                    .decomposition_chunks,
                projection_recursion
                    .recursion_constant_term
                    .decomposition_base_log,
                total_vars,
                1,
                0,
            );
            selectors.extend(projection_constant_terms_embedded_planes.iter().cloned());

            // RHS: fold_challenge (same for all batches)
            let rhs_fold_challenge_sumcheck = ElephantCell::new(
                LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                    config.witness_width,
                    total_vars - config.witness_width.ilog2() as usize,
                    0,
                ),
            );

            let witness_with_rhs_fold_challenge = ElephantCell::new(ProductSumcheck::new(
                witness.clone(),
                rhs_fold_challenge_sumcheck.clone(),
            )) as Data;

            let lhs_scalar_consistency_sumcheck = ElephantCell::new(
                LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(1, total_vars, 0),
            );

            lhs_scalar_consistency_sumcheck
                .borrow_mut()
                .load_from(&[ONE.clone()]);

            // Each batch is one piece of every digit plane of the batched-projection component.
            let batched_planes: [Selectors; NOF_BATCHES] = std::array::from_fn(|i| {
                let planes = plane_selectors(
                    projection_recursion.recursion_batched_projection.placement(),
                    projection_recursion
                        .recursion_batched_projection
                        .decomposition_chunks,
                    projection_recursion
                        .recursion_batched_projection
                        .decomposition_base_log,
                    total_vars,
                    NOF_BATCHES,
                    i,
                );
                selectors.extend(planes.iter().cloned());
                planes
            });

            // Build one context per batch
            let contexts: [FineProjSumcheckContext; NOF_BATCHES] = std::array::from_fn(|i| {
                // Split coefficients into block indices (elder vars) and within-block (LS vars)
                let height = config.projection_height;
                let inner_width = config.projection_ratio * height / DEGREE;
                let blocks = config.witness_height / inner_width;

                // Elder variables: c_0 coefficients (block indices)
                let lhs_flatter_0_sumcheck = ElephantCell::new(
                    LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                        blocks,
                        total_vars - blocks.ilog2() as usize - inner_width.ilog2() as usize,
                        inner_width.ilog2() as usize,
                    ),
                );

                // LS variables: c_1 . j_batched (within-block coefficients)
                let lhs_flatter_1_times_matrix_sumcheck = ElephantCell::new(
                    LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                        inner_width,
                        total_vars - inner_width.ilog2() as usize,
                        0,
                    ),
                );

                // Build the constraint tree
                let projection_coeff_product = ElephantCell::new(ProductSumcheck::new(
                    lhs_flatter_0_sumcheck.clone(),
                    lhs_flatter_1_times_matrix_sumcheck.clone(),
                ));

                let lhs = weighted_sum(
                    &folded_witness_planes,
                    ElephantCell::new(ProductSumcheck::new(
                        witness.clone(),
                        projection_coeff_product,
                    )) as Data,
                );

                let rhs = weighted_sum(&batched_planes[i], witness_with_rhs_fold_challenge.clone());

                let output = ElephantCell::new(DiffSumcheck::new(lhs, rhs));

                let lhs_consistency_flatter_sumcheck = ElephantCell::new(
                    LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                        config.witness_width,
                        total_vars - config.witness_width.ilog2() as usize,
                        0,
                    ),
                );

                let lhs = ElephantCell::new(ProductSumcheck::new(
                    lhs_scalar_consistency_sumcheck.clone(),
                    weighted_sum(
                        &batched_planes[i],
                        ElephantCell::new(ProductSumcheck::new(
                            lhs_consistency_flatter_sumcheck.clone(),
                            witness.clone(),
                        )) as Data,
                    ),
                ));

                // c_2 \otimes c_0 \otimes e_0
                let rhs_flatter_len =
                    config.witness_width * blocks * config.projection_height / DEGREE;

                let rhs_consistency_flatter_sumcheck = ElephantCell::new(
                    LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                        rhs_flatter_len,
                        total_vars - rhs_flatter_len.ilog2() as usize,
                        0,
                    ),
                );

                let rhs_scalar_consistency_sumcheck = ElephantCell::new(
                    LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(1, total_vars, 0),
                );

                let rhs = ElephantCell::new(ProductSumcheck::new(
                    rhs_scalar_consistency_sumcheck.clone(),
                    weighted_sum(
                        &projection_constant_terms_embedded_planes,
                        ElephantCell::new(ProductSumcheck::new(
                            rhs_consistency_flatter_sumcheck.clone(),
                            witness.clone(),
                        )) as Data,
                    ),
                ));

                let output_consistency = ElephantCell::new(DiffSumcheck::new(lhs, rhs));

                FineProjSumcheckContext {
                    lhs_flatter_0_sumcheck,
                    lhs_flatter_1_times_matrix_sumcheck,
                    output,
                    lhs_consistency_flatter_sumcheck,
                    rhs_scalar_consistency_sumcheck,
                    rhs_consistency_flatter_sumcheck,
                    output_2: output_consistency,
                }
            });

            Some(FineProjSumcheckContextWrapper {
                sumchecks: contexts,
                rhs_fold_challenge_sumcheck,
                lhs_scalar_consistency_sumcheck,
            })
        }
        _ => None,
    };

    let conjugated_combined_witness_sumcheck = ElephantCell::new(
        LinearSumcheck::<RingElement>::new(config.composed_witness_length),
    );

    // The norm claim covers the most inner commitment of every recursion tree: all blocks of
    // every row it places.
    let mut most_inner_commitments_selectors: Selectors = Vec::new();
    let push_most_inner = |recursion: &commitment::RecursionConfig,
                               out: &mut Selectors| {
        for placement in &recursion.most_inner_config().placements {
            for block in &placement.blocks {
                out.push(sumcheck_from_prefix(block, total_vars));
            }
        }
    };

    push_most_inner(&config.commitment_recursion, &mut most_inner_commitments_selectors);
    push_most_inner(&config.opening_recursion, &mut most_inner_commitments_selectors);

    match config.projection_recursion {
        Projection::Coarse(ref proj_config) => {
            push_most_inner(proj_config, &mut most_inner_commitments_selectors);
        }
        Projection::Fine(ref proj_config) => {
            push_most_inner(
                &proj_config.recursion_constant_term,
                &mut most_inner_commitments_selectors,
            );
            push_most_inner(
                &proj_config.recursion_batched_projection,
                &mut most_inner_commitments_selectors,
            );
        }
        Projection::Skip => {
            // No com_verify sumcheck for projection
        }
    }

    selectors.extend(most_inner_commitments_selectors.iter().cloned());

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

    // SUM_j 2^{2 . base_log . j} . <d_j, conj d_j> over the projection recursion's own digit
    // planes. The recomposed image is SUM_j 2^{base_log . j} d_j, so Cauchy-Schwarz turns this
    // claim into a bound on the image itself, up to sqrt(chunks).
    let output_3 = config.projection_norm_scope().map(|recursion| {
        let projection_planes = plane_selectors(
            recursion.placement(),
            recursion.decomposition_chunks,
            2 * recursion.decomposition_base_log,
            total_vars,
            1,
            0,
        );
        selectors.extend(projection_planes.iter().cloned());

        let mut sum_of_projection_planes: Data = projection_planes[0].clone();
        for selector in projection_planes.iter().skip(1) {
            sum_of_projection_planes = ElephantCell::new(SumSumcheck::new(
                sum_of_projection_planes.clone(),
                selector.clone(),
            ));
        }

        ElephantCell::new(ProductSumcheck::new(
            sum_of_projection_planes,
            output.clone(),
        ))
    });

    let norm_check_sumcheck = NormCheckSumcheckContext {
        conjugated_combined_witness: conjugated_combined_witness_sumcheck.clone(),
        output,
        output_2,
        output_3,
    };

    // ComVerify sumchecks: Three separate recursive commitment trees
    // 1. Commitment recursion: verifies the basic witness commitments are well-formed
    // 2. Opening recursion: verifies the opening proofs are correctly committed
    // 3. Projection recursion: verifies the projection images are correctly committed
    // Each tree has its own depth, rank, and decomposition parameters defined in config.

    let mut com_verify_sumchecks = vec![
        build_com_verify_sumcheck_context(
            crs,
            total_vars,
            witness.clone(),
            &config.commitment_recursion,
            &mut selectors,
        ),
        build_com_verify_sumcheck_context(
            crs,
            total_vars,
            witness.clone(),
            &config.opening_recursion,
            &mut selectors,
        ),
    ];

    match &config.projection_recursion {
        Projection::Coarse(recursion_config) => {
            com_verify_sumchecks.push(build_com_verify_sumcheck_context(
                crs,
                total_vars,
                witness.clone(),
                recursion_config,
                &mut selectors,
            ));
        }
        Projection::Fine(recursion_config) => {
            com_verify_sumchecks.push(build_com_verify_sumcheck_context(
                crs,
                total_vars,
                witness.clone(),
                &recursion_config.recursion_constant_term,
                &mut selectors,
            ));
            com_verify_sumchecks.push(build_com_verify_sumcheck_context(
                crs,
                total_vars,
                witness.clone(),
                &recursion_config.recursion_batched_projection,
                &mut selectors,
            ));
        }
        Projection::Skip => {
            // No com_verify sumcheck for projection
        }
    }

    let mut all_outputs: Vec<ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>> =
        vec![];
    for commitment_fold in &commitment_fold_sumchecks {
        all_outputs.push(commitment_fold.output.clone());
    }
    for inner_eval_fold in &inner_eval_fold_sumchecks {
        all_outputs.push(inner_eval_fold.output.clone());
    }
    for outer_eval_claim in &outer_eval_claim_sumchecks {
        all_outputs.push(outer_eval_claim.output.clone());
    }

    if let Some(coarse_proj_sumcheck) = &coarse_proj_sumcheck {
        all_outputs.push(coarse_proj_sumcheck.output.clone());
    } else if let Some(fine_proj_contexts) = &fine_proj_sumchecks {
        for fine_proj_ctx in fine_proj_contexts.sumchecks.iter() {
            all_outputs.push(fine_proj_ctx.output.clone());
            all_outputs.push(fine_proj_ctx.output_2.clone());
        }
    }

    for com_verify in &com_verify_sumchecks {
        for layer in &com_verify.layers {
            for output in &layer.outputs {
                all_outputs.push(output.clone());
            }
        }
        for output in &com_verify.output_layer.outputs {
            all_outputs.push(output.clone());
        }
    }

    all_outputs.push(norm_check_sumcheck.output.clone());
    all_outputs.push(norm_check_sumcheck.output_2.clone());
    if let Some(output_3) = &norm_check_sumcheck.output_3 {
        all_outputs.push(output_3.clone());
    }

    let combiner = ElephantCell::new(Combiner::new(all_outputs));

    let field_combiner = ElephantCell::new(RingToFieldCombiner::new(combiner.clone()));

    SumcheckContext {
        combined_witness_sumcheck: combined_witness_sumcheck.clone(),
        selectors,
        folding_challenges_sumcheck,
        commitment_key_rows_sumcheck,
        commitment_fold_sumchecks,
        inner_eval_fold_sumchecks,
        outer_eval_claim_sumchecks,
        coarse_proj_sumcheck,
        com_verify_sumchecks,
        norm_check_sumcheck,
        fine_proj_sumchecks,
        combiner,
        field_combiner,
        next: match &config.next {
            Some(next_config) => match next_config.as_ref() {
                Config::Sumcheck(next_simple_config) => Some(Box::new(
                    NextSumcheckContext::Simple(init_sumcheck(crs, next_simple_config)),
                )),
                Config::Simple(_) => None,
                Config::Intermediate(next_intermediate_config) => {
                    Some(Box::new(NextSumcheckContext::Intermediate(
                        init_intermediate_sumcheck(crs, next_intermediate_config),
                    )))
                }
            },
            None => None,
        },
    }
}
