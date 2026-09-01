use crate::{
    common::{
        arithmetic::{pow_mod, HALF_WAY_MOD_Q},
        config::{HALF_DEGREE, MOD_Q},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
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
/// variables. The layout allocator places digit-major instead (see `plane_selectors`); this is
/// for the intermediate round, whose witness is one flat element-major hypercube.
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
/// wraps once `base_log * plane >= 50`.
pub(crate) fn plane_weight(base_log: usize, plane: usize) -> RingElement {
    RingElement::constant(
        pow_mod(2, (base_log * plane) as u64),
        Representation::IncompleteNTT,
    )
}

/// One selector per digit plane of a placed component, each scaled by its radix weight, so that
/// `SUM_j scaled_j * W` recomposes the component out of its planes. `parts`/`part` address one of
/// `parts` equal pieces of every plane -- an opening, a projection batch, one commitment element
/// -- and `(1, 0)` takes the whole plane.
pub(crate) fn plane_selectors(
    placement: &Placement,
    chunks: usize,
    base_log: usize,
    total_vars: usize,
    parts: usize,
    part: usize,
) -> Vec<ElephantCell<SelectorEq<RingElement>>> {
    (0..chunks)
        .map(|plane| {
            let prefix = placement.slice(plane * parts + part, chunks * parts);
            ElephantCell::new(SelectorEq::<RingElement>::new_scaled(
                prefix.prefix,
                prefix.length,
                total_vars,
                plane_weight(base_log, plane),
            ))
        })
        .collect()
}

/// The pieces of a level's row that a commitment key row meets separately: the row's prefix in
/// the next round's witness paired with the dyadic slice of the key row covering it, as
/// `(prefix, slices, slice)`.
///
/// One piece per digit plane in general. A single-block placement with a power-of-two plane
/// count needs only one: digit-major lays the planes out contiguously and in order inside the
/// block, in the same order the commitment ran over them, so the whole row is one aligned run
/// on both sides.
pub(crate) fn row_committed_pieces(
    config: &RecursionConfig,
    row: usize,
) -> Vec<(Prefix, usize, usize)> {
    let placement = &config.placements[row];

    if placement.blocks.len() == 1 && config.decomposition_chunks.is_power_of_two() {
        return vec![(placement.blocks[0], config.segments(), row)];
    }

    (0..config.decomposition_chunks)
        .map(|plane| {
            (
                placement.slice(plane, config.decomposition_chunks),
                config.slices(),
                config.slice_index(row, plane),
            )
        })
        .collect()
}

/// `SUM_j selector_j * payload`, the shape every recomposed component enters a constraint in.
pub(crate) fn weighted_sum(
    selectors: &[ElephantCell<SelectorEq<RingElement>>],
    payload: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
) -> ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>> {
    sum_of(
        selectors
            .iter()
            .map(|selector| {
                ElephantCell::new(ProductSumcheck::new(selector.clone(), payload.clone()))
                    as ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>
            })
            .collect(),
    )
}

pub(crate) fn sum_of(
    terms: Vec<ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>>,
) -> ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>> {
    terms
        .into_iter()
        .reduce(|acc, term| ElephantCell::new(SumSumcheck::new(acc, term)))
        .expect("a component has at least one placed plane")
}

/// The block prefix and radix weight vector of the single-block form of a recomposition, when
/// the component admits it: one dyadic block and a power-of-two digit count put the plane index
/// in an aligned bit field of the within-block offset, so the whole weighting is a function of
/// those bits alone. The vector addresses the `chunks * parts` slices of the block and is zero
/// outside the `part`-th piece of each plane.
pub(crate) fn block_recomposition_weights(
    placement: &Placement,
    chunks: usize,
    base_log: usize,
    parts: usize,
    part: usize,
) -> Option<(Prefix, Vec<RingElement>)> {
    if placement.blocks.len() != 1 || !chunks.is_power_of_two() {
        return None;
    }

    let mut weights = vec![RingElement::zero(Representation::IncompleteNTT); chunks * parts];
    for plane in 0..chunks {
        weights[plane * parts + part] = plane_weight(base_log, plane);
    }

    Some((placement.blocks[0], weights))
}

/// The factor `SUM_j 2^{base_log . j} . selector_j` a placed component is recomposed by.
pub(crate) enum Recomposition {
    /// One scaled selector per digit plane, distributed over whatever the recomposition meets.
    Planes(Vec<ElephantCell<SelectorEq<RingElement>>>),
    /// The block selector times the radix weights, which live on disjoint variables: the block
    /// address above, the plane index below. One product in place of `chunks` of them.
    Block { factor: Data },
}

impl Recomposition {
    /// The recomposition on its own, as a single sumcheck node.
    pub(crate) fn factor(&self) -> Data {
        match self {
            Recomposition::Planes(planes) => {
                sum_of(planes.iter().map(|plane| plane.clone() as Data).collect())
            }
            Recomposition::Block { factor } => factor.clone(),
        }
    }

    /// The recomposed component times `payload`, the shape it enters a constraint in.
    pub(crate) fn times(&self, payload: Data) -> Data {
        match self {
            Recomposition::Planes(planes) => weighted_sum(planes, payload),
            Recomposition::Block { factor } => {
                ElephantCell::new(ProductSumcheck::new(factor.clone(), payload)) as Data
            }
        }
    }
}

/// The leaves a round folds once per sumcheck round: every selector it uses and the radix
/// weights of the recompositions that carry them on their own factor. A leaf shared by several
/// constraints is created once and registered once.
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

    pub(crate) fn push_selector(&mut self, selector: ElephantCell<SelectorEq<RingElement>>) {
        self.selectors.push(selector);
    }

    pub(crate) fn extend_selectors(
        &mut self,
        selectors: impl IntoIterator<Item = ElephantCell<SelectorEq<RingElement>>>,
    ) {
        self.selectors.extend(selectors);
    }

    /// Builds the recomposition of a placed component and registers its leaves. `parts`/`part`
    /// address one of `parts` equal pieces of every plane -- an opening, a projection batch, one
    /// commitment element -- and `(1, 0)` takes the whole plane.
    pub(crate) fn recomposition(
        &mut self,
        placement: &Placement,
        chunks: usize,
        base_log: usize,
        total_vars: usize,
        parts: usize,
        part: usize,
    ) -> Recomposition {
        match block_recomposition_weights(placement, chunks, base_log, parts, part) {
            Some((prefix, weights)) => {
                let block = sumcheck_from_prefix(&prefix, total_vars);
                let weights_sumcheck = ElephantCell::new(
                    LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                        weights.len(),
                        prefix.length,
                        total_vars - prefix.length - weights.len().ilog2() as usize,
                    ),
                );
                weights_sumcheck.borrow_mut().load_from(&weights);

                let factor = ElephantCell::new(ProductSumcheck::new(
                    block.clone() as Data,
                    weights_sumcheck.clone() as Data,
                )) as Data;

                self.selectors.push(block);
                self.weights.push(weights_sumcheck);

                Recomposition::Block { factor }
            }
            None => {
                let planes = plane_selectors(placement, chunks, base_log, total_vars, parts, part);
                self.selectors.extend(planes.iter().cloned());
                Recomposition::Planes(planes)
            }
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
) -> Vec<QuadraticExtension> {
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        return projection_flatter_1_times_matrix_ref(projection_matrix, projection_flatter_1);
    }
    let height = projection_matrix.projection_height;
    let projection_ratio = projection_matrix.projection_ratio;
    let inner_width = projection_ratio * height;

    let mut result_field = vec![QuadraticExtension::zero(); inner_width];
    for i in 0..inner_width {
        result_field[i].coeffs.fill(*HALF_WAY_MOD_Q);
    }

    for inner_row in 0..height {
        let weight = &projection_flatter_1.preprocessed_row[inner_row];
        let weight_field = QuadraticExtension {
            coeffs: [weight.v[0], weight.v[HALF_DEGREE]],
        };

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            use std::arch::x86_64::*;

            unsafe {
                // Interleave weight values: [weight.coeffs[0], weight.coeffs[1], weight.coeffs[0], weight.coeffs[1], ...]
                let weight_vec = _mm512_set_epi64(
                    weight_field.coeffs[1] as i64,
                    weight_field.coeffs[0] as i64,
                    weight_field.coeffs[1] as i64,
                    weight_field.coeffs[0] as i64,
                    weight_field.coeffs[1] as i64,
                    weight_field.coeffs[0] as i64,
                    weight_field.coeffs[1] as i64,
                    weight_field.coeffs[0] as i64,
                );

                // Process 8 QuadraticExtension elements at a time
                // Each QuadraticExtension has layout: [coeffs[0], coeffs[1]]
                // So 8 elements = 16 consecutive u64s in memory (interleaved)
                for i in (0..inner_width).step_by(8) {
                    if i + 8 > inner_width {
                        break; // Handle remainder with scalar code
                    }

                    let (k_pos, k_inc) = projection_matrix.get_row_masks_u8(inner_row, i);

                    // Duplicate each bit in the mask for interleaved access
                    // k_pos has 8 bits for 8 elements, we need 16 bits for 16 u64s (interleaved coeffs)
                    // Bit pattern: abcdefgh -> aabbccddeeffgghh
                    // Use BMI2 PDEP instruction to efficiently duplicate bits
                    let k_pos_16 =
                        (_pdep_u32(k_pos as u32, 0x5555) | _pdep_u32(k_pos as u32, 0xAAAA)) as u16;
                    let k_inc_16 =
                        (_pdep_u32(k_inc as u32, 0x5555) | _pdep_u32(k_inc as u32, 0xAAAA)) as u16;

                    // Get base pointer to the coeffs array (16 consecutive u64s)
                    let base_ptr = result_field[i].coeffs.as_mut_ptr();

                    // Load first 8 u64s (coeffs[0] and coeffs[1] for first 4 elements)
                    let current_low = _mm512_loadu_epi64(base_ptr as *const i64);
                    // Load next 8 u64s (coeffs[0] and coeffs[1] for next 4 elements)
                    let current_high = _mm512_loadu_epi64(base_ptr.add(8) as *const i64);

                    // Compute masks for add and subtract operations
                    let k_add_low = (k_inc_16 & k_pos_16) as u8;
                    let k_sub_low = (k_inc_16 & !k_pos_16) as u8;
                    let k_add_high = ((k_inc_16 & k_pos_16) >> 8) as u8;
                    let k_sub_high = ((k_inc_16 & !k_pos_16) >> 8) as u8;

                    // Apply masked operations for low part
                    let result_low =
                        _mm512_mask_add_epi64(current_low, k_add_low, current_low, weight_vec);
                    let result_low =
                        _mm512_mask_sub_epi64(result_low, k_sub_low, result_low, weight_vec);

                    // Apply masked operations for high part
                    let result_high =
                        _mm512_mask_add_epi64(current_high, k_add_high, current_high, weight_vec);
                    let result_high =
                        _mm512_mask_sub_epi64(result_high, k_sub_high, result_high, weight_vec);

                    // Store results back
                    _mm512_storeu_epi64(base_ptr as *mut i64, result_low);
                    _mm512_storeu_epi64(base_ptr.add(8) as *mut i64, result_high);
                }

                // Handle remainder with scalar code
                for i in (inner_width / 8 * 8)..inner_width {
                    let (is_positive, is_non_zero) = projection_matrix[(inner_row, i)];
                    if !is_non_zero {
                        continue;
                    }
                    if is_positive {
                        result_field[i].coeffs[0] += weight_field.coeffs[0];
                        result_field[i].coeffs[1] += weight_field.coeffs[1];
                    } else {
                        result_field[i].coeffs[0] -= weight_field.coeffs[0];
                        result_field[i].coeffs[1] -= weight_field.coeffs[1];
                    }
                }
            }
        }
    }

    unsafe {
        // this is a bit ugly but we want to avoid calling eltwise_reduce_mod separately
        eltwise_reduce_mod(
            result_field[0].coeffs.as_mut_ptr(),
            result_field[0].coeffs.as_ptr(),
            2 * inner_width as u64,
            MOD_Q,
        );
    }

    result_field
}

pub fn projection_flatter_1_times_matrix_ref(
    projection_matrix: &ProjectionMatrix,
    projection_flatter_1: &PreprocessedRow,
) -> Vec<QuadraticExtension> {
    let height = projection_matrix.projection_height;
    let projection_ratio = projection_matrix.projection_ratio;
    let inner_width = projection_ratio * height;

    let mut result_field = vec![QuadraticExtension::zero(); inner_width];
    for i in 0..inner_width {
        result_field[i].coeffs.fill(*HALF_WAY_MOD_Q);
    }

    for inner_row in 0..height {
        let weight = &projection_flatter_1.preprocessed_row[inner_row];
        let weight_field = QuadraticExtension {
            coeffs: [weight.v[0], weight.v[HALF_DEGREE]],
        };

        for i in 0..inner_width {
            let (is_positive, is_non_zero) = projection_matrix[(inner_row, i)];
            if !is_non_zero {
                continue;
            }
            if is_positive {
                result_field[i].coeffs[0] += weight_field.coeffs[0];
                result_field[i].coeffs[1] += weight_field.coeffs[1];
            } else {
                result_field[i].coeffs[0] -= weight_field.coeffs[0];
                result_field[i].coeffs[1] -= weight_field.coeffs[1];
            }
        }
    }

    unsafe {
        // this is a bit ugly but we want to avoid calling eltwise_reduce_mod separately
        eltwise_reduce_mod(
            result_field[0].coeffs.as_mut_ptr(),
            result_field[0].coeffs.as_ptr(),
            2 * inner_width as u64,
            MOD_Q,
        );
    }

    result_field
}
