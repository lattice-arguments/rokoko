use crate::{
    common::{
        ring_arithmetic::{Representation, RingElement},
        structured_row::PreprocessedRow,
    },
    protocol::{
        commitment::{Placement, Prefix},
        crs::CRS,
        sumcheck_utils::{
            common::HighOrderSumcheckData, elephant_cell::ElephantCell, linear::LinearSumcheck,
            product::ProductSumcheck, selector_eq::SelectorEq,
        },
    },
};

/// Everything the batched round does not change, it takes from the plain one.
pub(crate) use super::helpers_plain::{
    ck_sumcheck, composition_sumcheck, plane_weight, projection_flatter_1_times_matrix,
    row_committed_pieces, split_projection_flatter, sum_of, sumcheck_from_prefix,
    tensor_product_u64,
};

type Data = ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>;

/// One run of a placed component as a recomposition term: where the run sits, the radix weights
/// it carries, and the variables below them. `prefix.length`, `weights.len()` and `suffix` are
/// the geometry the linear sumcheck over the weights is laid out on, and they add up to
/// `total_vars`.
pub(crate) struct BlockWeights {
    pub prefix: Prefix,
    pub weights: Vec<RingElement>,
    pub suffix: usize,
}

/// The runs of a placed component that carry weight. `parts`/`part` address one of `parts` equal
/// pieces of every plane -- an opening, a projection batch, one commitment element -- and
/// `(1, 0)` takes the whole plane.
///
/// A block holds a power-of-two run of the component's slices at a slice-aligned offset, and
/// slice `plane * parts + part` sits on the block's own low bits. A single part is therefore the
/// whole block and its weights are the block's planes; every other part is one slice per plane,
/// scattered `parts` apart, and each is addressed by its own prefix. Addressing them costs a term
/// per plane rather than a length-`chunks * parts` vector per part, which is what keeps a level
/// linear rather than quadratic in `parts`.
pub(crate) fn block_recomposition_weights(
    placement: &Placement,
    chunks: usize,
    base_log: usize,
    total_vars: usize,
    parts: usize,
    part: usize,
) -> Vec<BlockWeights> {
    let slice_len = placement.size / (chunks * parts);

    placement
        .blocks_with_offsets()
        .into_iter()
        .flat_map(|(offset, size, prefix)| {
            let slices = size / slice_len;
            let planes = slices / parts;
            let first_plane = offset / slice_len / parts;
            debug_assert_eq!(slices % parts, 0, "a block holds whole planes");

            if parts == 1 {
                let weights = (0..planes)
                    .map(|plane| plane_weight(base_log, first_plane + plane))
                    .collect::<Vec<_>>();
                let suffix = total_vars - prefix.length - planes.ilog2() as usize;
                return vec![BlockWeights {
                    prefix,
                    weights,
                    suffix,
                }];
            }

            (0..planes)
                .map(|plane| {
                    let prefix = Prefix {
                        prefix: prefix.prefix * slices + plane * parts + part,
                        length: prefix.length + slices.ilog2() as usize,
                    };
                    BlockWeights {
                        suffix: total_vars - prefix.length,
                        prefix,
                        weights: vec![plane_weight(base_log, first_plane + plane)],
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The factor `SUM_j 2^{base_log . j} . selector_j` a placed component is recomposed by: one
/// run selector times the radix weights that run carries, summed over the runs.
#[derive(Clone)]
pub(crate) struct Recomposition {
    factor: Data,
}

impl Recomposition {
    /// The recomposition on its own, as a single sumcheck node.
    pub(crate) fn factor(&self) -> Data {
        self.factor.clone()
    }

    /// The recomposed component times `payload`, the shape it enters a constraint in.
    pub(crate) fn times(&self, payload: Data) -> Data {
        ElephantCell::new(ProductSumcheck::new(self.factor.clone(), payload)) as Data
    }
}

/// The entries of one recomposition leaf: the radix weight each carries and which part of the
/// component it belongs to. A single part puts the planes in one leaf; several parts put one
/// plane's parts in one leaf.
type WeightedEntries = Vec<(RingElement, usize)>;

/// A recomposition whose parts are weighted rather than one of them selected:
/// `SUM_part weights[part] . recomposition(.., parts, part)`. The geometry is fixed at setup and
/// the weights are loaded once the round has sampled them, so one constraint stands for what was
/// a family of `parts` of them.
pub struct WeightedRecomposition {
    factor: Data,
    leaves: Vec<(ElephantCell<LinearSumcheck<RingElement>>, WeightedEntries)>,
}

impl WeightedRecomposition {
    pub fn times(&self, payload: Data) -> Data {
        ElephantCell::new(ProductSumcheck::new(self.factor.clone(), payload)) as Data
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

/// Where the weighted recomposition puts its leaves: one per block when the component has a
/// single part, one per digit plane otherwise, since a plane's parts are a dyadic run on the
/// block's low bits while the planes themselves are not.
pub(crate) fn weighted_recomposition_layout(
    placement: &Placement,
    chunks: usize,
    base_log: usize,
    total_vars: usize,
    parts: usize,
) -> Vec<(Prefix, usize, WeightedEntries)> {
    let slice_len = placement.size / (chunks * parts);

    placement
        .blocks_with_offsets()
        .into_iter()
        .flat_map(|(offset, size, prefix)| {
            let slices = size / slice_len;
            let planes = slices / parts;
            let first_plane = offset / slice_len / parts;
            debug_assert_eq!(slices % parts, 0, "a block holds whole planes");

            if parts == 1 {
                let entries = (0..planes)
                    .map(|plane| (plane_weight(base_log, first_plane + plane), 0))
                    .collect();
                let suffix = total_vars - prefix.length - planes.ilog2() as usize;
                return vec![(prefix, suffix, entries)];
            }

            (0..planes)
                .map(|plane| {
                    let prefix = Prefix {
                        prefix: prefix.prefix * planes + plane,
                        length: prefix.length + planes.ilog2() as usize,
                    };
                    let radix = plane_weight(base_log, first_plane + plane);
                    let entries = (0..parts).map(|part| (radix.clone(), part)).collect();
                    let suffix = total_vars - prefix.length - parts.ilog2() as usize;
                    (prefix, suffix, entries)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// How many tensor layers a round samples for its row-batching challenge. One set covers every
/// commitment-row family of the round, and a family takes as many of the trailing layers as its
/// row count needs.
pub(crate) const ROW_BATCH_LAYERS: usize = 16;

/// The weights one family of commitment rows is batched under. Row `block * blockwise_rank + row`
/// carries `blocks[block] . rows[row]`, which `all` holds in row order. The two groups sit on
/// layer groups of their own, so neither count has to be a power of two.
pub(crate) struct RowWeights {
    pub blocks: Vec<RingElement>,
    pub rows: Vec<RingElement>,
    pub all: Vec<RingElement>,
}

/// Derives a family's weights from the round's layers: the key-row group is the trailing
/// `log2(blockwise_rank)` layers and the block group the `log2(blocks)` before it, each rounded
/// up to a whole layer and cut back to the count it addresses.
pub(crate) fn row_batch_weights(
    layers: &[RingElement],
    blocks: usize,
    blockwise_rank: usize,
) -> RowWeights {
    let row_bits = blockwise_rank.next_power_of_two().ilog2() as usize;
    let block_bits = blocks.next_power_of_two().ilog2() as usize;
    assert!(
        row_bits + block_bits <= layers.len(),
        "{blocks} x {blockwise_rank} rows ask for more layers than the round samples"
    );
    let split = layers.len() - row_bits;

    let mut block_weights =
        PreprocessedRow::from_layers(&layers[split - block_bits..split]).preprocessed_row;
    block_weights.truncate(blocks);
    let mut row_weights = PreprocessedRow::from_layers(&layers[split..]).preprocessed_row;
    row_weights.truncate(blockwise_rank);

    let mut all = Vec::with_capacity(blocks * blockwise_rank);
    for block in &block_weights {
        for row in &row_weights {
            let mut weight = RingElement::zero(Representation::IncompleteNTT);
            weight *= (block, row);
            all.push(weight);
        }
    }

    RowWeights {
        blocks: block_weights,
        rows: row_weights,
        all,
    }
}

/// The one dense key row the batched constraint meets: `SUM_j w_row(j) . K_j` over the
/// `blockwise_rank` key rows that meet every `wit_dim`-long block of a commitment's input. The
/// segment of the combined row is the combination of the rows' segments, so a level materialises
/// the row once and hands out slices of it.
pub(crate) fn combined_ck_row(
    crs: &CRS,
    wit_dim: usize,
    row_weights: &[RingElement],
) -> Vec<RingElement> {
    let mut combined = vec![RingElement::zero(Representation::IncompleteNTT); wit_dim];
    let mut term = RingElement::zero(Representation::IncompleteNTT);

    for (key_row, weight) in crs.ck_for_wit_dim(wit_dim).iter().zip(row_weights) {
        for (out, key) in combined.iter_mut().zip(key_row.preprocessed_row.iter()) {
            term *= (key, weight);
            *out += &term;
        }
    }

    combined
}

/// The registry of leaves a round folds once per sumcheck round: every selector it uses and the
/// radix weights of the recompositions that carry them on their own factor. The same block
/// selector is registered once per part a component is recomposed in; a `SelectorEq` folds in
/// O(1), so the repeats cost the round a pointer each.
pub(crate) struct FoldedLeaves {
    pub selectors: Vec<ElephantCell<SelectorEq<RingElement>>>,
    pub weights: Vec<ElephantCell<LinearSumcheck<RingElement>>>,
}

impl FoldedLeaves {
    pub(crate) fn new() -> Self {
        FoldedLeaves {
            selectors: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Builds the weighted recomposition of a placed component and registers its leaves.
    pub(crate) fn weighted_recomposition(
        &mut self,
        placement: &Placement,
        chunks: usize,
        base_log: usize,
        total_vars: usize,
        parts: usize,
    ) -> WeightedRecomposition {
        let mut leaves = Vec::new();
        let terms = weighted_recomposition_layout(placement, chunks, base_log, total_vars, parts)
            .into_iter()
            .map(|(prefix, suffix, entries)| {
                let run = sumcheck_from_prefix(&prefix, total_vars);
                let weights = ElephantCell::new(
                    LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                        entries.len(),
                        prefix.length,
                        suffix,
                    ),
                );

                let factor = ElephantCell::new(ProductSumcheck::new(
                    run.clone() as Data,
                    weights.clone() as Data,
                )) as Data;

                self.selectors.push(run);
                self.weights.push(weights.clone());
                leaves.push((weights, entries));

                factor
            })
            .collect();

        WeightedRecomposition {
            factor: sum_of(terms),
            leaves,
        }
    }

    /// Builds the recomposition of a placed component and registers its leaves.
    pub(crate) fn recomposition(
        &mut self,
        placement: &Placement,
        chunks: usize,
        base_log: usize,
        total_vars: usize,
        parts: usize,
        part: usize,
    ) -> Recomposition {
        let terms =
            block_recomposition_weights(placement, chunks, base_log, total_vars, parts, part)
                .into_iter()
                .map(|block_weights| {
                    let BlockWeights {
                        prefix,
                        weights,
                        suffix,
                    } = block_weights;
                    let run = sumcheck_from_prefix(&prefix, total_vars);
                    let weights_sumcheck = ElephantCell::new(
                        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                            weights.len(),
                            prefix.length,
                            suffix,
                        ),
                    );
                    weights_sumcheck.borrow_mut().load_from(&weights);

                    let factor = ElephantCell::new(ProductSumcheck::new(
                        run.clone() as Data,
                        weights_sumcheck.clone() as Data,
                    )) as Data;

                    self.selectors.push(run);
                    self.weights.push(weights_sumcheck);

                    factor
                })
                .collect();

        Recomposition {
            factor: sum_of(terms),
        }
    }
}
