use crate::{
    common::{
        arithmetic::{pow_mod, HALF_WAY_MOD_Q},
        config::{MOD_Q, NOF_BATCHES, SLOT_DEGREE},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{FieldExtension, Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
        sumcheck_element::SumcheckElement,
    },
    hexl::bindings::{eltwise_reduce_mod, multiply_mod},
    protocol::{
        commitment::{Placement, Prefix, RecursionConfig},
        crs::CRS,
        sumcheck_utils::{
            common::HighOrderSumcheckData, elephant_cell::ElephantCell, linear::LinearSumcheck,
            product::ProductSumcheck, selector_eq::SelectorEq, sum::SumSumcheck,
        },
    },
};

type Data = ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>;

/// Builds the sumcheck carrying radix weights (1, base, base^2, ...) used to recompose a
/// base-`2^{base_log}` decomposition laid out element-major, with the digit index on the low
/// variables. The layout allocator places digit-major instead (see `block_recomposition_weights`);
/// this is for the intermediate round, whose witness is one flat element-major hypercube.
pub(crate) fn composition_sumcheck(
    base_log: u64,
    chunks: usize,
    total_vars: usize,
) -> ElephantCell<LinearSumcheck<RingElement>> {
    let composition_basis = (0..chunks)
        .map(|i| {
            RingElement::constant(pow_mod(2, base_log * i as u64), Representation::IncompleteNTT)
        })
        .collect::<Vec<RingElement>>();
    let combiner_sumcheck = ElephantCell::new(
        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
            composition_basis.len(),
            total_vars - composition_basis.len().ilog2() as usize,
            0,
        ),
    );

    combiner_sumcheck.borrow_mut().load_from(&composition_basis);

    combiner_sumcheck
}

/// The radix weight `2^{base_log * plane}` of a digit plane, reduced mod q: an unreduced shift
/// wraps once `base_log * plane >= log2 q`.
pub(crate) fn plane_weight(base_log: usize, plane: usize) -> RingElement {
    RingElement::constant(
        pow_mod(2, (base_log * plane) as u64),
        Representation::IncompleteNTT,
    )
}

/// The pieces of a level's row that a commitment key row meets separately, one per dyadic block
/// of the row's placement: the block's prefix in the next round's witness paired with the dyadic
/// slice of the key row covering it, as `(prefix, slices, slice)`.
///
/// A block holds a whole power-of-two run of planes at a plane-aligned offset, and digit-major
/// lays those planes out contiguously and in the order the commitment ran over them, so the run
/// is one aligned slice on both sides.
pub(crate) fn row_committed_pieces(
    config: &RecursionConfig,
    row: usize,
) -> Vec<(Prefix, usize, usize)> {
    let row_len = config.row_len();
    let padded_chunks = config.padded_chunks();

    config.placements[row]
        .blocks_with_offsets()
        .into_iter()
        .map(|(offset, size, prefix)| {
            let planes = size / row_len;
            (
                prefix,
                config.segments() * padded_chunks / planes,
                (row * padded_chunks + offset / row_len) / planes,
            )
        })
        .collect()
}

pub(crate) fn sum_of(
    terms: Vec<ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>>,
) -> ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>> {
    terms
        .into_iter()
        .reduce(|acc, term| ElephantCell::new(SumSumcheck::new(acc, term)))
        .expect("a component has at least one placed block")
}

/// One dyadic block of a placed component as a recomposition term: where the block sits, the
/// radix weights it carries, and the variables below them. `prefix.length`, `weights.len()` and
/// `suffix` are the geometry the linear sumcheck over the weights is laid out on, and they add up
/// to `total_vars`.
pub(crate) struct BlockWeights {
    pub prefix: Prefix,
    pub weights: Vec<RingElement>,
    pub suffix: usize,
}

/// Where batch `batch` of the batched projection lives, as `(placement, parts, part)` for
/// `block_recomposition_weights`: its own row of a level with one row per batch, or else the
/// `batch`-th piece of every digit plane of the level's single row.
pub(crate) fn batch_piece(recursion: &RecursionConfig, batch: usize) -> (&Placement, usize, usize) {
    match recursion.placements.len() {
        1 => (recursion.placement(), NOF_BATCHES, batch),
        _ => (&recursion.placements[batch], 1, 0),
    }
}

/// The dyadic blocks of a placed component, each with the radix weights it carries. A block holds
/// a power-of-two run of the component's slices at a slice-aligned offset, so the weights it
/// carries are a function of the block's own low bits alone. `parts`/`part` address one of
/// `parts` equal pieces of every plane -- an opening, a projection batch, one commitment element
/// -- and `(1, 0)` takes the whole plane; a slice outside the `part`-th piece weighs zero.
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
        .map(|(offset, size, prefix)| {
            let weights: Vec<RingElement> = (offset / slice_len..(offset + size) / slice_len)
                .map(|slice| {
                    if slice % parts == part {
                        plane_weight(base_log, slice / parts)
                    } else {
                        RingElement::zero(Representation::IncompleteNTT)
                    }
                })
                .collect();
            let suffix = total_vars - prefix.length - weights.len().ilog2() as usize;
            BlockWeights {
                prefix,
                weights,
                suffix,
            }
        })
        .collect()
}

/// The factor `SUM_j 2^{base_log . j} . selector_j` a placed component is recomposed by: one
/// block selector times the radix weights of the planes that block holds, summed over the
/// blocks. The two factors of a block live on disjoint variables -- the block address above, the
/// plane index below -- so the recomposition costs one product per block rather than one per
/// digit plane.
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
                    let block = sumcheck_from_prefix(&prefix, total_vars);
                    let weights_sumcheck = ElephantCell::new(
                        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                            weights.len(),
                            prefix.length,
                            suffix,
                        ),
                    );
                    weights_sumcheck.borrow_mut().load_from(&weights);

                    let factor = ElephantCell::new(ProductSumcheck::new(
                        block.clone() as Data,
                        weights_sumcheck.clone() as Data,
                    )) as Data;

                    self.selectors.push(block);
                    self.weights.push(weights_sumcheck);

                    factor
                })
                .collect();

        Recomposition {
            factor: sum_of(terms),
        }
    }
}

/// Creates a selector (SelectorEq) that evaluates to 1 where the first `prefix.length`
/// bits match `prefix.prefix`, and 0 elsewhere. Used to enforce constraints only on
/// specific witness slices. Prefix padding ensures alignment with the global hypercube.
pub(crate) fn sumcheck_from_prefix(
    prefix: &Prefix,
    total_vars: usize,
) -> ElephantCell<SelectorEq<RingElement>> {
    ElephantCell::new(SelectorEq::<RingElement>::new(
        prefix.prefix,
        prefix.length,
        total_vars,
    ))
}

fn ck_row_sumcheck(
    row: &[RingElement],
    total_vars: usize,
    sufix: usize,
) -> ElephantCell<LinearSumcheck<RingElement>> {
    let sumcheck = ElephantCell::new(
        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
            row.len(),
            total_vars - row.len().ilog2() as usize - sufix,
            sufix,
        ),
    );

    sumcheck.borrow_mut().load_from(row);

    sumcheck
}

/// Loads the i-th row of the commitment key into a linear sumcheck with appropriate padding:
/// - `wit_dim`: dimension for this CK row (varies for recursive layers)
/// - `sufix`: trailing variables for decomposition chunks
/// - prefix padding aligns with the global hypercube
///
/// Uses preprocessed CRS data to avoid recomputing tensor structures.
pub(crate) fn ck_sumcheck(
    crs: &CRS,
    total_vars: usize,
    wit_dim: usize,
    i: usize,
    sufix: usize,
) -> ElephantCell<LinearSumcheck<RingElement>> {
    ck_row_sumcheck(
        &crs.ck_for_wit_dim(wit_dim)[i].preprocessed_row,
        total_vars,
        sufix,
    )
}

/// The `segment`-th of `segments` equal dyadic slices of the `i`-th commitment key row: the part
/// of the row that meets one separately placed row block of the commitment's input.
/// `segments == 1` reproduces `ck_sumcheck`.
pub(crate) fn ck_segment_sumcheck(
    crs: &CRS,
    total_vars: usize,
    wit_dim: usize,
    i: usize,
    segments: usize,
    segment: usize,
) -> ElephantCell<LinearSumcheck<RingElement>> {
    let len = wit_dim / segments;
    ck_row_sumcheck(
        &crs.ck_for_wit_dim(wit_dim)[i].preprocessed_row[segment * len..(segment + 1) * len],
        total_vars,
        0,
    )
}

pub fn tensor_product_u64(a: &Vec<u64>, b: &Vec<u64>) -> Vec<u64> {
    let mut result: Vec<u64> = vec![0u64; a.len() * b.len()];
    let mut idx = 0;
    for a_elem in a.iter() {
        for b_elem in b.iter() {
            unsafe { result[idx] = multiply_mod(*a_elem, *b_elem, MOD_Q) }
            // result[idx] = a_elem.wrapping_mul(*b_elem);
            idx += 1;
        }
    }
    result
}

/// Splits projection_flatter into two components for the elder/LS variable separation.
///
/// This function decomposes a projection flattening vector into:
/// - projection_flatter_0: operates on "elder variables" (block indices)
/// - projection_flatter_1: operates on "LS variables" (within-block indices)
///
/// The split follows the tensor structure: given a StructuredRow with tensor_layers,
/// we partition the layers at the boundary between block-level and within-block indexing.
/// Specifically, if we have `blocks = witness_height / inner_width`, then the first
/// `blocks.ilog2()` layers correspond to block selection (elder), and the remaining
/// `height.ilog2()` layers handle within-block positions (LS).
///
/// This decomposition enables us to structure the projection coefficient sumcheck as a
/// product of two independent linear sumchecks, which can improve verifier efficiency
/// when the two components have different sparsity patterns or when we want to fold
/// them separately.
pub(crate) fn split_projection_flatter(
    projection_flatter: &StructuredRow,
    projection_height: usize,
) -> (StructuredRow, StructuredRow) {
    let height = projection_height;
    let height_log = height.ilog2() as usize;
    let tensor_layers = &projection_flatter.tensor_layers;

    debug_assert!(tensor_layers.len() >= height_log);
    let block_layers = tensor_layers.len() - height_log;

    let projection_flatter_0 = StructuredRow {
        tensor_layers: tensor_layers[..block_layers].to_vec(),
    };
    let projection_flatter_1 = StructuredRow {
        tensor_layers: tensor_layers[block_layers..].to_vec(),
    };

    (projection_flatter_0, projection_flatter_1)
}

/// Computes the product of projection_flatter_1 with the projection matrix.
///
/// This function computes the linear combination:
///   projection_flatter_1 · (I ⊗ projection_matrix)
///
/// where projection_flatter_1 operates on the "within-block" indices (LS variables)
/// and the projection_matrix defines the projection structure. The result is a vector
/// of length `inner_width = projection_ratio * height` that captures how the projection
/// matrix rows are weighted by projection_flatter_1.
///
/// **Computational Strategy:**
/// For each row in the projection matrix, we:
/// 1. Check if projection_flatter_1[row] is non-zero (skip if zero for efficiency)
/// 2. For each non-zero entry in that row, accumulate the weighted contribution
/// 3. Handle the sign of the projection matrix entry (positive or negative)
///
/// The result is then used in the LS-variable linear sumcheck component, which gets
/// multiplied with the elder-variable component to form the complete projection
/// coefficient sumcheck.
pub fn projection_flatter_1_times_matrix(
    projection_matrix: &ProjectionMatrix,
    projection_flatter_1: &PreprocessedRow,
) -> Vec<FieldExtension> {
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        return projection_flatter_1_times_matrix_ref(projection_matrix, projection_flatter_1);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        let height = projection_matrix.projection_height;
        let inner_width = projection_matrix.projection_ratio * height;
        let chunks = inner_width / 8;

        // Per row: the slot-zero weight, then its negation.
        let mut weights = vec![0u64; 2 * SLOT_DEGREE * height];
        for (row, w) in weights.chunks_exact_mut(2 * SLOT_DEGREE).enumerate() {
            let weight = projection_flatter_1.preprocessed_row[row].slot_zero();
            for k in 0..SLOT_DEGREE {
                w[k] = weight.coeffs[k];
                w[SLOT_DEGREE + k] = weight.coeffs[k].wrapping_neg();
            }
        }

        let mut result_field = vec![FieldExtension::zero(); inner_width];
        let mut chunk = 0;
        while chunk + COARSE_GROUP <= chunks {
            unsafe {
                columns_times_matrix::<COARSE_GROUP>(
                    projection_matrix,
                    &weights,
                    chunk,
                    &mut result_field,
                )
            };
            chunk += COARSE_GROUP;
        }
        while chunk < chunks {
            unsafe {
                columns_times_matrix::<1>(projection_matrix, &weights, chunk, &mut result_field)
            };
            chunk += 1;
        }
        for i in 8 * chunks..inner_width {
            result_field[i].coeffs.fill(*HALF_WAY_MOD_Q);
            for row in 0..height {
                let (is_positive, is_non_zero) = projection_matrix[(row, i)];
                if !is_non_zero {
                    continue;
                }
                let w = &weights[2 * SLOT_DEGREE * row..];
                for k in 0..SLOT_DEGREE {
                    let sign = if is_positive { 0 } else { SLOT_DEGREE };
                    result_field[i].coeffs[k] = result_field[i].coeffs[k].wrapping_add(w[sign + k]);
                }
            }
        }

        unsafe {
            // this is a bit ugly but we want to avoid calling eltwise_reduce_mod separately
            let flat = result_field.as_mut_ptr() as *mut u64;
            eltwise_reduce_mod(flat, flat, (SLOT_DEGREE * inner_width) as u64, MOD_Q);
        }

        result_field
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const COARSE_GROUP: usize = 16 / SLOT_DEGREE;

/// Columns `8 chunk .. 8 (chunk + G)`, unreduced, accumulated in registers over all rows.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn columns_times_matrix<const G: usize>(
    projection_matrix: &ProjectionMatrix,
    weights: &[u64],
    chunk: usize,
    result_field: &mut [FieldExtension],
) {
    use std::arch::x86_64::*;

    let pos = projection_matrix.pos_masks.data.as_ptr();
    let inc = projection_matrix.non_zero_masks.data.as_ptr();
    let mask_width = projection_matrix.width;
    let mut acc = [_mm512_set1_epi64(*HALF_WAY_MOD_Q as i64); 16];
    for row in 0..projection_matrix.projection_height {
        let w = weights.as_ptr().add(2 * SLOT_DEGREE * row);
        let positive: [__m512i; SLOT_DEGREE] =
            std::array::from_fn(|k| _mm512_set1_epi64(*w.add(k) as i64));
        let negative: [__m512i; SLOT_DEGREE] =
            std::array::from_fn(|k| _mm512_set1_epi64(*w.add(SLOT_DEGREE + k) as i64));
        for g in 0..G {
            let at = row * mask_width + chunk + g;
            let (k_pos, k_inc) = (*pos.add(at), *inc.add(at));
            for k in 0..SLOT_DEGREE {
                let signed = _mm512_mask_blend_epi64(k_pos, negative[k], positive[k]);
                let a = &mut acc[SLOT_DEGREE * g + k];
                *a = _mm512_mask_add_epi64(*a, k_inc, *a, signed);
            }
        }
    }
    for g in 0..G {
        let mut lanes = [[0u64; 8]; SLOT_DEGREE];
        for k in 0..SLOT_DEGREE {
            _mm512_storeu_epi64(lanes[k].as_mut_ptr() as *mut i64, acc[SLOT_DEGREE * g + k]);
        }
        for l in 0..8 {
            result_field[8 * (chunk + g) + l].coeffs = std::array::from_fn(|k| lanes[k][l]);
        }
    }
}

pub fn projection_flatter_1_times_matrix_ref(
    projection_matrix: &ProjectionMatrix,
    projection_flatter_1: &PreprocessedRow,
) -> Vec<FieldExtension> {
    let height = projection_matrix.projection_height;
    let projection_ratio = projection_matrix.projection_ratio;
    let inner_width = projection_ratio * height;

    let mut result_field = vec![FieldExtension::zero(); inner_width];
    for i in 0..inner_width {
        result_field[i].coeffs.fill(*HALF_WAY_MOD_Q);
    }

    for inner_row in 0..height {
        let weight = &projection_flatter_1.preprocessed_row[inner_row];
        let weight_field = weight.slot_zero();

        for i in 0..inner_width {
            let (is_positive, is_non_zero) = projection_matrix[(inner_row, i)];
            if !is_non_zero {
                continue;
            }
            for k in 0..SLOT_DEGREE {
                if is_positive {
                    result_field[i].coeffs[k] += weight_field.coeffs[k];
                } else {
                    result_field[i].coeffs[k] -= weight_field.coeffs[k];
                }
            }
        }
    }

    unsafe {
        // this is a bit ugly but we want to avoid calling eltwise_reduce_mod separately
        let flat = result_field.as_mut_ptr() as *mut u64;
        eltwise_reduce_mod(flat, flat, (SLOT_DEGREE * inner_width) as u64, MOD_Q);
    }

    result_field
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{hash::HashWrapper, init_common};

    #[test]
    fn projection_flatter_1_times_matrix_matches_the_reference() {
        init_common();
        for (ratio, height) in [(2, 16), (4, 64), (8, 3), (5, 8), (32, 256)] {
            let mut projection_matrix = ProjectionMatrix::new(ratio, height);
            projection_matrix.sample(&mut HashWrapper::new());
            for top in [false, true] {
                let flatter_1 = PreprocessedRow {
                    preprocessed_row: (0..height)
                        .map(|_| match top {
                            false => RingElement::random(Representation::IncompleteNTT),
                            true => RingElement::all(MOD_Q - 1, Representation::IncompleteNTT),
                        })
                        .collect(),
                };
                assert_eq!(
                    projection_flatter_1_times_matrix(&projection_matrix, &flatter_1),
                    projection_flatter_1_times_matrix_ref(&projection_matrix, &flatter_1),
                    "ratio {ratio}, height {height}, top {top}"
                );
            }
        }
    }
}
