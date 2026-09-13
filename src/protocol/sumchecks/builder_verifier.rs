use std::collections::HashMap;

#[cfg(not(feature = "standard"))]
use crate::common::structured_row::StructuredRow;
#[cfg(feature = "standard")]
use crate::protocol::sumchecks::helpers::weighted_recomposition_layout;

use crate::{
    common::{
        arithmetic::ONE_QUAD,
        config::{DEGREE, NOF_BATCHES},
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
    },
    protocol::{
        commitment::{self, Placement, Prefix},
        config::{Config, Projection, SumcheckConfig},
        crs::VerifierCRS,
        intermediate_sumchecks::builder_verifier::init_intermediate_verifier,
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
            CoarseProjVerifierContext, ComVerifyLayerVerifierContext,
            ComVerifyOutputLayerVerifierContext, ComVerifyVerifierContext,
            CommitmentFoldVerifierContext, FineProjVerifierContext, FineProjVerifierContextWrapper,
            InnerEvalFoldVerifierContext, NextVerifierSumcheckContext, NormCheckVerifierContext,
            OuterEvalClaimVerifierContext, VerifierSumcheckContext,
        },
        sumchecks::helpers::{block_recomposition_weights, row_committed_pieces, BlockWeights},
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

fn sum_of_evaluations(terms: Vec<ElephantCell<EvalData>>) -> ElephantCell<EvalData> {
    terms
        .into_iter()
        .reduce(|acc, term| ElephantCell::new(SumSumcheckEvaluation::new(acc, term)))
        .expect("a component has at least one placed block")
}

/// Verifier dual of `Recomposition`.
struct RecompositionEvaluation {
    factor: ElephantCell<EvalData>,
}

impl RecompositionEvaluation {
    /// Dual of `Recomposition::factor`.
    fn factor(&self) -> ElephantCell<EvalData> {
        self.factor.clone()
    }

    /// Dual of `Recomposition::times`.
    fn times(&self, payload: ElephantCell<EvalData>) -> ElephantCell<EvalData> {
        ElephantCell::new(ProductSumcheckEvaluation::new(self.factor.clone(), payload))
            as ElephantCell<EvalData>
    }
}

/// Dual of `FoldedLeaves::recomposition`. The verifier folds nothing, so the leaves need no
/// registry; the two sides get the same blocks and weights from
/// `block_recomposition_weights`.
fn recomposition_evaluation(
    placement: &Placement,
    chunks: usize,
    base_log: usize,
    total_vars: usize,
    parts: usize,
    part: usize,
) -> RecompositionEvaluation {
    let terms = block_recomposition_weights(placement, chunks, base_log, total_vars, parts, part)
        .into_iter()
        .map(|block_weights| {
            let BlockWeights {
                prefix,
                weights,
                suffix,
            } = block_weights;
            let block = selector_evaluation_from_prefix(&prefix, total_vars);
            let weights_evaluation = basic_evaluation_linear(weights.len(), prefix.length, suffix);
            weights_evaluation.borrow_mut().load_from(&weights);

            ElephantCell::new(ProductSumcheckEvaluation::new(block, weights_evaluation))
                as ElephantCell<EvalData>
        })
        .collect();

    RecompositionEvaluation {
        factor: sum_of_evaluations(terms),
    }
}

pub fn basic_evaluation_linear(
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

#[cfg(feature = "standard")]
/// Verifier dual of `WeightedRecomposition`: the same geometry, with the leaves loaded once the
/// round has sampled the weights they carry.
pub struct WeightedRecompositionEvaluation {
    factor: ElephantCell<EvalData>,
    leaves: Vec<(
        ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
        Vec<(RingElement, usize)>,
    )>,
}

#[cfg(feature = "standard")]
impl WeightedRecompositionEvaluation {
    fn times(&self, payload: ElephantCell<EvalData>) -> ElephantCell<EvalData> {
        ElephantCell::new(ProductSumcheckEvaluation::new(self.factor.clone(), payload))
            as ElephantCell<EvalData>
    }

    pub fn load(&self, part_weights: &[RingElement]) {
        let mut values: Vec<RingElement> = Vec::new();
        for (leaf, entries) in &self.leaves {
            values.clear();
            for (radix, part) in entries {
                let mut value = RingElement::zero(Representation::IncompleteNTT);
                value *= (radix, &part_weights[*part]);
                values.push(value);
            }
            leaf.borrow_mut().load_from(&values);
        }
    }
}

#[cfg(feature = "standard")]
/// Dual of `FoldedLeaves::weighted_recomposition`.
fn weighted_recomposition_evaluation(
    placement: &Placement,
    chunks: usize,
    base_log: usize,
    total_vars: usize,
    parts: usize,
) -> WeightedRecompositionEvaluation {
    let mut leaves = Vec::new();
    let terms = weighted_recomposition_layout(placement, chunks, base_log, total_vars, parts)
        .into_iter()
        .map(|(prefix, suffix, entries)| {
            let run = selector_evaluation_from_prefix(&prefix, total_vars);
            let weights = basic_evaluation_linear(entries.len(), prefix.length, suffix);

            leaves.push((weights.clone(), entries));

            ElephantCell::new(ProductSumcheckEvaluation::new(run, weights))
                as ElephantCell<EvalData>
        })
        .collect();

    WeightedRecompositionEvaluation {
        factor: sum_of_evaluations(terms),
        leaves,
    }
}

#[cfg(feature = "standard")]
/// A node that evaluates to a constant wherever the sumcheck point falls: a selector on no
/// variables, whose scale the round loads. It is how a row weight enters a product without
/// raising the constraint's degree.
fn weight_evaluation(total_vars: usize) -> ElephantCell<SelectorEqEvaluation> {
    ElephantCell::new(SelectorEqEvaluation::new(0, 0, total_vars))
}

#[cfg(feature = "standard")]
/// The combined key row `SUM_j w_row(j) . K_j` as the verifier meets it: the dense rows evaluated
/// at the point and weighted afterwards, which costs what the rows alone cost rather than a
/// freshly combined dense row per round.
fn combined_ck_evaluation(
    rows: Vec<ElephantCell<EvalData>>,
    scales: &[ElephantCell<SelectorEqEvaluation>],
) -> ElephantCell<EvalData> {
    sum_of_evaluations(
        rows.into_iter()
            .zip(scales)
            .map(|(row, scale)| {
                ElephantCell::new(ProductSumcheckEvaluation::new(scale.clone(), row))
                    as ElephantCell<EvalData>
            })
            .collect(),
    )
}

pub fn load_combiner_evaluation_data(
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

#[cfg(not(feature = "standard"))]
pub fn structured_row_ck_evaluation(
    crs: &VerifierCRS,
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

#[cfg(feature = "standard")]
/// The `i`-th commitment key row as the verifier meets it: a dense vector, so its multilinear
/// extension costs the length of the row. Block-diagonal commitment is what keeps that short.
pub fn ck_row_evaluation(
    crs: &VerifierCRS,
    total_vars: usize,
    wit_dim: usize,
    i: usize,
    suffix: usize,
) -> ElephantCell<BasicEvaluationLinearSumcheck<RingElement>> {
    let prefix_size = total_vars - wit_dim.ilog2() as usize - suffix;
    let eval = basic_evaluation_linear(wit_dim, prefix_size, suffix);
    eval.borrow_mut()
        .load_from(&crs.ck_for_wit_dim(wit_dim)[i].preprocessed_row);
    eval
}

#[cfg(not(feature = "standard"))]
/// The `segment`-th of `segments` equal dyadic slices of the `i`-th commitment key row, the
/// verifier's counterpart to `ck_segment_sumcheck`. Layers are MS-first, so the leading
/// `log2(segments)` of them pick the segment: what they contribute at this index is a constant,
/// and the slice is still a tensor row over the layers below. `segments == 1` reproduces
/// `structured_row_ck_evaluation` exactly.
pub fn structured_row_ck_segment_evaluation(
    crs: &VerifierCRS,
    total_vars: usize,
    wit_dim: usize,
    i: usize,
    segments: usize,
    segment: usize,
) -> ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>> {
    let segment_len = wit_dim / segments;
    let fixed = segments.ilog2() as usize;
    let row = crs.structured_ck_for_wit_dim(wit_dim)[i].clone();

    let scale = StructuredRow {
        tensor_layers: row.tensor_layers[..fixed].to_vec(),
    }
    .at(segment);
    let sliced = StructuredRow {
        tensor_layers: row.tensor_layers[fixed..].to_vec(),
    };

    let eval = ElephantCell::new(
        StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            segment_len,
            total_vars - segment_len.ilog2() as usize,
            0,
        ),
    );
    eval.borrow_mut().load_scaled_from(sliced, scale);
    eval
}

#[cfg(feature = "standard")]
/// The `segment`-th of `segments` equal dyadic slices of that row: the part that meets one
/// separately placed row block of the commitment's input.
pub fn ck_segment_evaluation(
    crs: &VerifierCRS,
    total_vars: usize,
    wit_dim: usize,
    i: usize,
    segments: usize,
    segment: usize,
) -> ElephantCell<BasicEvaluationLinearSumcheck<RingElement>> {
    let len = wit_dim / segments;
    let eval = basic_evaluation_linear(len, total_vars - len.ilog2() as usize, 0);
    eval.borrow_mut().load_from(
        &crs.ck_for_wit_dim(wit_dim)[i].preprocessed_row[segment * len..(segment + 1) * len],
    );
    eval
}

/// Verifier dual of `InputPiece`.
struct InputPieceEvaluation {
    /// Which block of the level's input the piece falls in.
    #[cfg(feature = "standard")]
    block: usize,
    ck_slices: usize,
    ck_slice: usize,
    data: ElephantCell<EvalData>,
    /// The piece's own selector, which carries the weight of the block the piece falls in.
    #[cfg(feature = "standard")]
    selector: ElephantCell<SelectorEqEvaluation>,
}

#[cfg(not(feature = "standard"))]
fn selected_input_piece_evaluations(
    total_vars: usize,
    config: &commitment::RecursionConfig,
    witness: &ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
) -> Vec<Vec<InputPieceEvaluation>> {
    (0..config.placements.len())
        .map(|row| {
            row_committed_pieces(config, row)
                .into_iter()
                .map(|(prefix, ck_slices, ck_slice)| InputPieceEvaluation {
                    ck_slices,
                    ck_slice,
                    data: ElephantCell::new(ProductSumcheckEvaluation::new(
                        selector_evaluation_from_prefix(&prefix, total_vars),
                        witness.clone(),
                    )) as ElephantCell<EvalData>,
                })
                .collect()
        })
        .collect()
}

#[cfg(feature = "standard")]
/// Dual of `piece_block_selectors`.
fn piece_block_selectors(
    data_selected: &[Vec<InputPieceEvaluation>],
) -> Vec<(usize, ElephantCell<SelectorEqEvaluation>)> {
    data_selected
        .iter()
        .flatten()
        .map(|piece| (piece.block, piece.selector.clone()))
        .collect()
}

#[cfg(feature = "standard")]
fn selected_input_piece_evaluations(
    total_vars: usize,
    config: &commitment::RecursionConfig,
    witness: &ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
) -> Vec<Vec<InputPieceEvaluation>> {
    let blocks = config.diag_blocks;

    (0..config.placements.len())
        .map(|row| {
            let mut pieces = Vec::new();
            for (prefix, ck_slices, ck_slice) in row_committed_pieces(config, row) {
                let split = blocks.div_ceil(ck_slices);
                let per_block = ck_slices * split / blocks;
                for part in 0..split {
                    let prefix = Prefix {
                        prefix: prefix.prefix * split + part,
                        length: prefix.length + split.ilog2() as usize,
                    };
                    let slice = ck_slice * split + part;
                    let selector = selector_evaluation_from_prefix(&prefix, total_vars);
                    pieces.push(InputPieceEvaluation {
                        block: slice / per_block,
                        ck_slices: per_block,
                        ck_slice: slice % per_block,
                        data: ElephantCell::new(ProductSumcheckEvaluation::new(
                            selector.clone(),
                            witness.clone(),
                        )) as ElephantCell<EvalData>,
                        selector,
                    });
                }
            }
            pieces
        })
        .collect()
}

#[cfg(not(feature = "standard"))]
fn ck_over_pieces_evaluation(
    crs: &VerifierCRS,
    total_vars: usize,
    config: &commitment::RecursionConfig,
    i: usize,
    data_selected: &[Vec<InputPieceEvaluation>],
    ck_evals: &mut Vec<ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>>,
) -> ElephantCell<EvalData> {
    let committed_len = config.committed_len();
    let mut terms: Vec<ElephantCell<EvalData>> = Vec::new();

    for pieces in data_selected {
        for piece in pieces {
            let ck = structured_row_ck_segment_evaluation(
                crs,
                total_vars,
                committed_len,
                i,
                piece.ck_slices,
                piece.ck_slice,
            );
            ck_evals.push(ck.clone());
            terms.push(
                ElephantCell::new(ProductSumcheckEvaluation::new(ck, piece.data.clone()))
                    as ElephantCell<EvalData>,
            );
        }
    }

    sum_of_evaluations(terms)
}

#[cfg(feature = "standard")]
/// The combined key-row slices a level's pieces share, addressed by `(slices, slice)`.
type CkSegments = HashMap<(usize, usize), ElephantCell<EvalData>>;

#[cfg(feature = "standard")]
/// The level's key is `I_blocks (x) K`, so the slice of the combined key row that a piece meets
/// is the same vector whichever block the piece falls in: the `blockwise_rank` dense rows behind
/// it are evaluated once and reused by every piece on that slice. What separates the blocks is
/// the piece's own selector, whose scale carries the block's weight.
fn combined_ck_over_pieces_evaluation(
    crs: &VerifierCRS,
    total_vars: usize,
    config: &commitment::RecursionConfig,
    data_selected: &[Vec<InputPieceEvaluation>],
    key_row_scales: &[ElephantCell<SelectorEqEvaluation>],
    segments: &mut CkSegments,
) -> ElephantCell<EvalData> {
    let block_len = config.block_len();
    let blockwise_rank = config.blockwise_rank();
    let mut terms: Vec<ElephantCell<EvalData>> = Vec::new();

    for pieces in data_selected {
        for piece in pieces {
            let ck = segments
                .entry((piece.ck_slices, piece.ck_slice))
                .or_insert_with(|| {
                    let rows = (0..blockwise_rank)
                        .map(|i| {
                            ck_segment_evaluation(
                                crs,
                                total_vars,
                                block_len,
                                i,
                                piece.ck_slices,
                                piece.ck_slice,
                            ) as ElephantCell<EvalData>
                        })
                        .collect();
                    combined_ck_evaluation(rows, key_row_scales)
                })
                .clone();
            terms.push(
                ElephantCell::new(ProductSumcheckEvaluation::new(ck, piece.data.clone()))
                    as ElephantCell<EvalData>,
            );
        }
    }

    sum_of_evaluations(terms)
}

#[cfg(not(feature = "standard"))]
fn build_com_verify_verifier_context(
    crs: &VerifierCRS,
    total_vars: usize,
    combined_witness_eval: ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    config: &commitment::RecursionConfig,
) -> ComVerifyVerifierContext {
    let mut layers = Vec::new();
    let mut current = config;

    while let Some(next) = current.next.as_deref() {
        let data_selected =
            selected_input_piece_evaluations(total_vars, current, &combined_witness_eval);

        let mut ck_evals = Vec::new();
        let outputs = (0..current.rank)
            .map(|i| {
                let ck_with_data = ck_over_pieces_evaluation(
                    crs,
                    total_vars,
                    current,
                    i,
                    &data_selected,
                    &mut ck_evals,
                );

                let child = recomposition_evaluation(
                    next.placement(),
                    next.decomposition_chunks,
                    next.decomposition_base_log,
                    total_vars,
                    current.rank,
                    i,
                );
                let recomposed_child = child.times(combined_witness_eval.clone());

                ElephantCell::new(DiffSumcheckEvaluation::new(ck_with_data, recomposed_child))
            })
            .collect::<Vec<_>>();

        layers.push(ComVerifyLayerVerifierContext {
            ck_evaluations: ck_evals,
            outputs,
        });

        current = next;
    }

    let data_selected =
        selected_input_piece_evaluations(total_vars, current, &combined_witness_eval);
    let mut ck_evals = Vec::new();
    let outputs = (0..current.rank)
        .map(|i| {
            ck_over_pieces_evaluation(crs, total_vars, current, i, &data_selected, &mut ck_evals)
        })
        .collect::<Vec<_>>();

    ComVerifyVerifierContext {
        layers,
        output_layer: ComVerifyOutputLayerVerifierContext {
            ck_evaluations: ck_evals,
            outputs,
        },
    }
}

#[cfg(feature = "standard")]
fn build_com_verify_verifier_context(
    crs: &VerifierCRS,
    total_vars: usize,
    combined_witness_eval: ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    config: &commitment::RecursionConfig,
) -> ComVerifyVerifierContext {
    let mut layers = Vec::new();
    let mut current = config;

    while let Some(next) = current.next.as_deref() {
        let data_selected =
            selected_input_piece_evaluations(total_vars, current, &combined_witness_eval);

        let key_row_scales = (0..current.blockwise_rank())
            .map(|_| weight_evaluation(total_vars))
            .collect::<Vec<_>>();
        let mut segments = CkSegments::new();
        let ck_with_data = combined_ck_over_pieces_evaluation(
            crs,
            total_vars,
            current,
            &data_selected,
            &key_row_scales,
            &mut segments,
        );

        let child = weighted_recomposition_evaluation(
            next.placement(),
            next.decomposition_chunks,
            next.decomposition_base_log,
            total_vars,
            current.rank,
        );
        let recomposed_child = child.times(combined_witness_eval.clone());

        layers.push(ComVerifyLayerVerifierContext {
            blocks: current.diag_blocks,
            blockwise_rank: current.blockwise_rank(),
            piece_selectors: piece_block_selectors(&data_selected),
            key_row_scales,
            child,
            output: ElephantCell::new(DiffSumcheckEvaluation::new(ck_with_data, recomposed_child)),
        });

        current = next;
    }

    let data_selected =
        selected_input_piece_evaluations(total_vars, current, &combined_witness_eval);
    let key_row_scales = (0..current.blockwise_rank())
        .map(|_| weight_evaluation(total_vars))
        .collect::<Vec<_>>();
    let mut segments = CkSegments::new();
    let output = combined_ck_over_pieces_evaluation(
        crs,
        total_vars,
        current,
        &data_selected,
        &key_row_scales,
        &mut segments,
    );

    ComVerifyVerifierContext {
        layers,
        output_layer: ComVerifyOutputLayerVerifierContext {
            blocks: current.diag_blocks,
            blockwise_rank: current.blockwise_rank(),
            piece_selectors: piece_block_selectors(&data_selected),
            key_row_scales,
            output,
        },
    }
}

pub fn init_verifier(crs: &VerifierCRS, config: &SumcheckConfig) -> VerifierSumcheckContext {
    let total_vars = config.composed_witness_length.ilog2() as usize;

    let combined_witness_evaluation =
        ElephantCell::new(FakeEvaluationLinearSumcheck::<RingElement>::new());
    let witness = combined_witness_evaluation.clone() as ElephantCell<EvalData>;

    let folded_witness = recomposition_evaluation(
        &config.folded_witness_placement,
        config.witness_decomposition_chunks,
        config.witness_decomposition_base_log,
        total_vars,
        1,
        0,
    );

    let folding_challenges_evaluation = basic_evaluation_linear(
        config.witness_width,
        total_vars - config.witness_width.ilog2() as usize,
        0,
    );

    let witness_with_folding_challenges = ElephantCell::new(ProductSumcheckEvaluation::new(
        witness.clone(),
        folding_challenges_evaluation.clone(),
    )) as ElephantCell<EvalData>;

    #[cfg(not(feature = "standard"))]
    let commitment_key_rows_evaluation = (0..config.basic_commitment_rank)
        .map(|i| structured_row_ck_evaluation(crs, total_vars, config.witness_height, i, 0))
        .collect::<Vec<_>>();

    // Batched, the key rows are combined at the point rather than evaluated one by one, and the
    // folded witness carries the block weights.
    #[cfg(feature = "standard")]
    let (key_row_scales, combined_commitment_key_row, folded_witness_blocks) = {
        let blocks = config.basic_commitment_diag_blocks;
        let blockwise_rank = config.basic_commitment_rank / blocks;
        let key_row_scales = (0..blockwise_rank)
            .map(|_| weight_evaluation(total_vars))
            .collect::<Vec<_>>();
        let combined = combined_ck_evaluation(
            (0..blockwise_rank)
                .map(|i| {
                    ck_row_evaluation(crs, total_vars, config.witness_height / blocks, i, 0)
                        as ElephantCell<EvalData>
                })
                .collect(),
            &key_row_scales,
        );
        let folded = weighted_recomposition_evaluation(
            &config.folded_witness_placement,
            config.witness_decomposition_chunks,
            config.witness_decomposition_base_log,
            total_vars,
            blocks,
        );
        (key_row_scales, combined, folded)
    };

    #[cfg(not(feature = "standard"))]
    let commitment_fold_evaluations = (0..config.basic_commitment_rank)
        .map(|i| {
            let row = recomposition_evaluation(
                &config.commitment_recursion.placements[i],
                config.commitment_recursion.decomposition_chunks,
                config.commitment_recursion.decomposition_base_log,
                total_vars,
                1,
                0,
            );

            let lhs = folded_witness.times(ElephantCell::new(ProductSumcheckEvaluation::new(
                witness.clone(),
                commitment_key_rows_evaluation[i].clone(),
            )) as ElephantCell<EvalData>);

            let rhs = row.times(witness_with_folding_challenges.clone());

            CommitmentFoldVerifierContext {
                output: ElephantCell::new(DiffSumcheckEvaluation::new(lhs, rhs)),
            }
        })
        .collect::<Vec<_>>();

    #[cfg(feature = "standard")]
    let commitment_fold_evaluation = {
        let basic_commitment_rows = (0..config.basic_commitment_rank)
            .map(|i| {
                weighted_recomposition_evaluation(
                    &config.commitment_recursion.placements[i],
                    config.commitment_recursion.decomposition_chunks,
                    config.commitment_recursion.decomposition_base_log,
                    total_vars,
                    1,
                )
            })
            .collect::<Vec<_>>();

        let lhs = folded_witness_blocks.times(ElephantCell::new(ProductSumcheckEvaluation::new(
            witness.clone(),
            combined_commitment_key_row,
        )) as ElephantCell<EvalData>);

        let rhs = sum_of_evaluations(
            basic_commitment_rows
                .iter()
                .map(|row| row.times(witness_with_folding_challenges.clone()))
                .collect(),
        );

        CommitmentFoldVerifierContext {
            key_row_scales,
            folded_witness_blocks,
            basic_commitment_rows,
            output: ElephantCell::new(DiffSumcheckEvaluation::new(lhs, rhs)),
        }
    };

    let openings = (0..config.nof_openings)
        .map(|i| {
            recomposition_evaluation(
                &config.opening_recursion.placements[i],
                config.opening_recursion.decomposition_chunks,
                config.opening_recursion.decomposition_base_log,
                total_vars,
                1,
                0,
            )
        })
        .collect::<Vec<_>>();

    let inner_eval_fold_evaluations = (0..config.nof_openings)
        .map(|i| {
            let inner_evaluation = ElephantCell::new(
                StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
                    config.witness_height,
                    total_vars - config.witness_height.ilog2() as usize,
                    0,
                ),
            );

            let lhs = folded_witness.times(ElephantCell::new(ProductSumcheckEvaluation::new(
                witness.clone(),
                inner_evaluation.clone(),
            )) as ElephantCell<EvalData>);

            let rhs = openings[i].times(witness_with_folding_challenges.clone());

            InnerEvalFoldVerifierContext {
                inner_evaluation,
                output: ElephantCell::new(DiffSumcheckEvaluation::new(lhs, rhs)),
            }
        })
        .collect::<Vec<_>>();

    let outer_eval_claim_evaluations = (0..config.nof_openings)
        .map(|i| {
            let outer_evaluation = ElephantCell::new(
                StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
                    config.witness_width,
                    total_vars - config.witness_width.ilog2() as usize,
                    0,
                ),
            );

            let output = openings[i].times(ElephantCell::new(ProductSumcheckEvaluation::new(
                witness.clone(),
                outer_evaluation.clone(),
            )) as ElephantCell<EvalData>);

            OuterEvalClaimVerifierContext {
                outer_evaluation,
                output,
            }
        })
        .collect::<Vec<_>>();

    // Build CoarseProj with Product of split LHS and RHS coefficients
    let coarse_proj_evaluation = {
        match &config.projection_recursion {
            Projection::Coarse(projection_recursion) => {
                let projection = recomposition_evaluation(
                    projection_recursion.placement(),
                    projection_recursion.decomposition_chunks,
                    projection_recursion.decomposition_base_log,
                    total_vars,
                    1,
                    0,
                );

                let projection_height_flat = config.witness_height / config.projection_ratio;

                // Split LHS projection coefficients evaluations
                let height = config.projection_height;
                let inner_width = config.projection_ratio * height;
                let blocks = config.witness_height / inner_width;

                let lhs_flatter_0_evaluation = ElephantCell::new(
                    StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
                        blocks,
                        total_vars - blocks.ilog2() as usize - inner_width.ilog2() as usize,
                        inner_width.ilog2() as usize,
                    ),
                );

                let lhs_flatter_1_times_matrix_evaluation_field = ElephantCell::new(
                BasicEvaluationLinearSumcheck::<QuadraticExtension>::new_with_prefixed_sufixed_data(
                    inner_width,
                    total_vars - inner_width.ilog2() as usize,
                    0,
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
                        total_vars - projection_height_flat.ilog2() as usize,
                        0,
                    ),
                );

                let rhs_fold_challenge_evaluation = basic_evaluation_linear(
                    config.witness_width,
                    total_vars
                        - config.witness_width.ilog2() as usize
                        - projection_height_flat.ilog2() as usize,
                    projection_height_flat.ilog2() as usize,
                );

                let lhs_projection_coeff_product =
                    ElephantCell::new(ProductSumcheckEvaluation::new(
                        lhs_flatter_0_evaluation.clone(),
                        lhs_flatter_1_times_matrix_evaluation.clone(),
                    ));

                let rhs_fold_tensor_product = ElephantCell::new(ProductSumcheckEvaluation::new(
                    rhs_projection_flatter_evaluation.clone(),
                    rhs_fold_challenge_evaluation.clone(),
                ));

                let coarse_proj_lhs = folded_witness.times(ElephantCell::new(
                    ProductSumcheckEvaluation::new(witness.clone(), lhs_projection_coeff_product),
                )
                    as ElephantCell<EvalData>);

                let coarse_proj_rhs = projection.times(ElephantCell::new(
                    ProductSumcheckEvaluation::new(witness.clone(), rhs_fold_tensor_product),
                ) as ElephantCell<EvalData>);

                Some(CoarseProjVerifierContext {
                    lhs_flatter_0_evaluation,
                    lhs_flatter_1_times_matrix_evaluation_field,
                    lhs_flatter_1_times_matrix_evaluation,
                    rhs_projection_flatter_evaluation,
                    rhs_fold_challenge_evaluation,
                    output: ElephantCell::new(DiffSumcheckEvaluation::new(
                        coarse_proj_lhs,
                        coarse_proj_rhs,
                    )),
                })
            }
            Projection::Fine(_projection_recursion) => None,
            Projection::Skip => None,
        }
    };

    let fine_proj_evaluations = match &config.projection_recursion {
        Projection::Fine(proj_config) => {
            let rhs_fold_challenge_evaluation = basic_evaluation_linear(
                config.witness_width,
                total_vars - config.witness_width.ilog2() as usize,
                0,
            );

            let witness_with_rhs_fold_challenge = ElephantCell::new(ProductSumcheckEvaluation::new(
                witness.clone(),
                rhs_fold_challenge_evaluation.clone(),
            )) as ElephantCell<EvalData>;

            let projection_constant_terms_embedded = recomposition_evaluation(
                proj_config.recursion_constant_term.placement(),
                proj_config.recursion_constant_term.decomposition_chunks,
                proj_config.recursion_constant_term.decomposition_base_log,
                total_vars,
                1,
                0,
            );

            let lhs_scalar_consistency_evaluation_field = ElephantCell::new(
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

            let batched: [RecompositionEvaluation; NOF_BATCHES] = std::array::from_fn(|i| {
                recomposition_evaluation(
                    proj_config.recursion_batched_projection.placement(),
                    proj_config
                        .recursion_batched_projection
                        .decomposition_chunks,
                    proj_config
                        .recursion_batched_projection
                        .decomposition_base_log,
                    total_vars,
                    NOF_BATCHES,
                    i,
                )
            });

            let contexts: [FineProjVerifierContext; NOF_BATCHES] = std::array::from_fn(|i| {
                // Split coefficients into block indices (elder vars) and within-block (LS vars)
                let height = config.projection_height;
                let inner_width = config.projection_ratio * height / DEGREE;
                let blocks = config.witness_height / inner_width;
                let lhs_flatter_0_evaluation_field = ElephantCell::new(
                    StructuredRowEvaluationLinearSumcheck::<QuadraticExtension>::new_with_prefixed_sufixed_data(
                        blocks,
                        total_vars - blocks.ilog2() as usize - inner_width.ilog2() as usize,
                        inner_width.ilog2() as usize,
                    ),
                );
                let lhs_flatter_1_times_matrix_evaluation = basic_evaluation_linear(
                    inner_width,
                    total_vars - inner_width.ilog2() as usize,
                    0,
                );

                let lhs_flatter_0_evaluation = ElephantCell::new(
                    RingToFieldWrapperEvaluation::new(lhs_flatter_0_evaluation_field.clone()),
                );

                let projection_coeff_product = ElephantCell::new(ProductSumcheckEvaluation::new(
                    lhs_flatter_0_evaluation.clone(),
                    lhs_flatter_1_times_matrix_evaluation.clone(),
                ));

                let lhs = folded_witness.times(ElephantCell::new(ProductSumcheckEvaluation::new(
                    witness.clone(),
                    projection_coeff_product.clone(),
                )) as ElephantCell<EvalData>);

                let rhs = batched[i].times(witness_with_rhs_fold_challenge.clone());

                let output = ElephantCell::new(DiffSumcheckEvaluation::new(lhs, rhs));

                let lhs_consistency_flatter_evaluation_field = ElephantCell::new(
                    StructuredRowEvaluationLinearSumcheck::<QuadraticExtension>::new_with_prefixed_sufixed_data(
                        config.witness_width,
                        total_vars - config.witness_width.ilog2() as usize,
                        0,
                    ),
                );

                let lhs_consistency_flatter_evaluation =
                    ElephantCell::new(RingToFieldWrapperEvaluation::new(
                        lhs_consistency_flatter_evaluation_field.clone(),
                    ));

                let lhs = ElephantCell::new(ProductSumcheckEvaluation::new(
                    lhs_scalar_consistency_evaluation.clone(),
                    batched[i].times(ElephantCell::new(ProductSumcheckEvaluation::new(
                        lhs_consistency_flatter_evaluation.clone(),
                        witness.clone(),
                    )) as ElephantCell<EvalData>),
                ));

                // c_2 \otimes c_0 \otimes e_0
                let rhs_flatter_len =
                    config.witness_width * blocks * config.projection_height / DEGREE;

                let rhs_consistency_flatter_evaluation_field = ElephantCell::new(
                    StructuredRowEvaluationLinearSumcheck::<QuadraticExtension>::new_with_prefixed_sufixed_data(
                        rhs_flatter_len,
                        total_vars - rhs_flatter_len.ilog2() as usize,
                        0,
                    ),
                );

                let rhs_consistency_flatter_evaluation =
                    ElephantCell::new(RingToFieldWrapperEvaluation::new(
                        rhs_consistency_flatter_evaluation_field.clone(),
                    ));

                let rhs_scalar_consistency_evaluation = ElephantCell::new(
                    BasicEvaluationLinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                        1, total_vars, 0,
                    ),
                );

                let rhs = ElephantCell::new(ProductSumcheckEvaluation::new(
                    rhs_scalar_consistency_evaluation.clone(),
                    projection_constant_terms_embedded.times(ElephantCell::new(
                        ProductSumcheckEvaluation::new(
                            rhs_consistency_flatter_evaluation.clone(),
                            witness.clone(),
                        ),
                    )
                        as ElephantCell<EvalData>),
                ));

                let output_consistency = ElephantCell::new(DiffSumcheckEvaluation::new(lhs, rhs));

                FineProjVerifierContext {
                    lhs_flatter_0_evaluation_field,
                    lhs_flatter_0_evaluation,
                    lhs_flatter_1_times_matrix_evaluation,
                    output,
                    lhs_consistency_flatter_evaluation_field,
                    lhs_consistency_flatter_evaluation,
                    rhs_consistency_flatter_evaluation_field,
                    rhs_consistency_flatter_evaluation,
                    rhs_scalar_consistency_evaluation,
                    output_2: output_consistency,
                }
            });
            Some(FineProjVerifierContextWrapper {
                sumchecks: contexts,
                rhs_fold_challenge_evaluation,
                lhs_scalar_consistency_evaluation_field,
                lhs_scalar_consistency_evaluation,
            })
        }
        _ => None,
    };

    let mut com_verify_evaluations = vec![
        build_com_verify_verifier_context(
            crs,
            total_vars,
            combined_witness_evaluation.clone(),
            &config.commitment_recursion,
        ),
        build_com_verify_verifier_context(
            crs,
            total_vars,
            combined_witness_evaluation.clone(),
            &config.opening_recursion,
        ),
    ];

    match &config.projection_recursion {
        Projection::Coarse(proj_config) => {
            com_verify_evaluations.push(build_com_verify_verifier_context(
                crs,
                total_vars,
                combined_witness_evaluation.clone(),
                &proj_config,
            ));
        }
        Projection::Fine(proj_config) => {
            com_verify_evaluations.push(build_com_verify_verifier_context(
                crs,
                total_vars,
                combined_witness_evaluation.clone(),
                &proj_config.recursion_constant_term,
            ));

            com_verify_evaluations.push(build_com_verify_verifier_context(
                crs,
                total_vars,
                combined_witness_evaluation.clone(),
                &proj_config.recursion_batched_projection,
            ));
        }
        Projection::Skip => {
            // Do nothing
        }
    }
    let conjugated_combined_witness_evaluation =
        ElephantCell::new(FakeEvaluationLinearSumcheck::<RingElement>::new());

    let mut most_inner_commitments_selectors: Vec<ElephantCell<SelectorEqEvaluation>> = vec![];
    let push_most_inner =
        |recursion: &commitment::RecursionConfig,
         out: &mut Vec<ElephantCell<SelectorEqEvaluation>>| {
            for placement in &recursion.most_inner_config().placements {
                for block in &placement.blocks {
                    out.push(selector_evaluation_from_prefix(block, total_vars));
                }
            }
        };

    push_most_inner(
        &config.commitment_recursion,
        &mut most_inner_commitments_selectors,
    );
    push_most_inner(
        &config.opening_recursion,
        &mut most_inner_commitments_selectors,
    );

    match &config.projection_recursion {
        Projection::Coarse(proj_config) => {
            push_most_inner(proj_config, &mut most_inner_commitments_selectors);
        }
        Projection::Fine(proj_config) => {
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
            // Do nothing
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

    let output_3 = config.projection_norm_scope().map(|recursion| {
        let squared_projection = recomposition_evaluation(
            recursion.placement(),
            recursion.decomposition_chunks,
            2 * recursion.decomposition_base_log,
            total_vars,
            1,
            0,
        );

        ElephantCell::new(ProductSumcheckEvaluation::new(
            squared_projection.factor(),
            output.clone(),
        ))
    });

    let norm_check_evaluation = NormCheckVerifierContext {
        conjugated_combined_witness_evaluation: conjugated_combined_witness_evaluation.clone(),
        output,
        selectors: most_inner_commitments_selectors,
        output_2,
        output_3,
    };

    let mut all_outputs: Vec<ElephantCell<EvalData>> = vec![];
    #[cfg(not(feature = "standard"))]
    for commitment_fold in &commitment_fold_evaluations {
        all_outputs.push(commitment_fold.output.clone());
    }
    #[cfg(feature = "standard")]
    all_outputs.push(commitment_fold_evaluation.output.clone());
    for inner_eval_fold in &inner_eval_fold_evaluations {
        all_outputs.push(inner_eval_fold.output.clone());
    }
    for outer_eval_claim in &outer_eval_claim_evaluations {
        all_outputs.push(outer_eval_claim.output.clone());
    }
    if let Some(coarse_proj_evaluation) = &coarse_proj_evaluation {
        all_outputs.push(coarse_proj_evaluation.output.clone());
    }
    if let Some(fine_proj_evaluations) = &fine_proj_evaluations {
        for fine_proj in &fine_proj_evaluations.sumchecks {
            all_outputs.push(fine_proj.output.clone());
            all_outputs.push(fine_proj.output_2.clone());
        }
    }

    for com_verify in &com_verify_evaluations {
        #[cfg(not(feature = "standard"))]
        {
            for layer in &com_verify.layers {
                for output in &layer.outputs {
                    all_outputs.push(output.clone());
                }
            }
            for output in &com_verify.output_layer.outputs {
                all_outputs.push(output.clone());
            }
        }
        #[cfg(feature = "standard")]
        {
            for layer in &com_verify.layers {
                all_outputs.push(layer.output.clone());
            }
            all_outputs.push(com_verify.output_layer.output.clone());
        }
    }
    all_outputs.push(norm_check_evaluation.output.clone());
    all_outputs.push(norm_check_evaluation.output_2.clone());
    if let Some(output_3) = &norm_check_evaluation.output_3 {
        all_outputs.push(output_3.clone());
    }

    let combiner_evaluation = ElephantCell::new(CombinerEvaluation::new(all_outputs));
    let field_combiner_evaluation = ElephantCell::new(RingToFieldCombinerEvaluation::new(
        combiner_evaluation.clone(),
    ));

    VerifierSumcheckContext {
        combined_witness_evaluation,
        folding_challenges_evaluation,
        #[cfg(not(feature = "standard"))]
        commitment_key_rows_evaluation,
        #[cfg(not(feature = "standard"))]
        commitment_fold_evaluations,
        #[cfg(feature = "standard")]
        commitment_fold_evaluation,
        inner_eval_fold_evaluations,
        outer_eval_claim_evaluations,
        coarse_proj_evaluation,
        fine_proj_evaluations,
        com_verify_evaluations,
        norm_check_evaluation,
        combiner_evaluation,
        field_combiner_evaluation,
        next: match &config.next {
            Some(next_config) => match next_config.as_ref() {
                Config::Sumcheck(next_sumcheck_config) => Some(Box::new(
                    NextVerifierSumcheckContext::Simple(init_verifier(crs, next_sumcheck_config)),
                )),
                Config::Intermediate(next_intermediate_config) => {
                    Some(Box::new(NextVerifierSumcheckContext::Intermediate(
                        init_intermediate_verifier(crs, next_intermediate_config),
                    )))
                }
                _ => None,
            },
            None => None,
        },
    }
}
