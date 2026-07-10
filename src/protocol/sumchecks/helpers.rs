use num::range;

use crate::{
    common::{
        arithmetic::HALF_WAY_MOD_Q,
        config::{HALF_DEGREE, MOD_Q},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
        sumcheck_element::SumcheckElement,
    },
    hexl::bindings::{eltwise_reduce_mod, multiply_mod},
    protocol::{
        commitment::Prefix,
        crs::CRS,
        sumcheck_utils::{
            elephant_cell::ElephantCell, linear::LinearSumcheck, selector_eq::SelectorEq,
        },
    },
};

/// Builds the sumcheck carrying radix weights (1, base, base^2, ...) used to
/// recompose a base-`2^{base_log}` decomposition; prefix padding enables
/// composition without re-indexing the hypercube.
pub(crate) fn composition_sumcheck(
    base_log: u64,
    chunks: usize,
    total_vars: usize,
) -> ElephantCell<LinearSumcheck<RingElement>> {
    let conmposition_basis = range(0, chunks)
        .map(|i| {
            // Basis element corresponding to 2^{base_log * i}
            RingElement::constant(
                1u64 << (base_log as u64 * i as u64),
                Representation::IncompleteNTT,
            )
        })
        .collect::<Vec<RingElement>>();
    let combiner_sumcheck = ElephantCell::new(
        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
            conmposition_basis.len(),
            total_vars - conmposition_basis.len().ilog2() as usize,
            0,
        ),
    );

    combiner_sumcheck
        .borrow_mut()
        .load_from(&conmposition_basis);

    combiner_sumcheck
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
    let ck = crs.ck_for_wit_dim(wit_dim);

    let sumcheck = ElephantCell::new(
        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
            wit_dim,
            total_vars - wit_dim.ilog2() as usize - sufix,
            sufix,
        ),
    );

    sumcheck.borrow_mut().load_from(&ck[i].preprocessed_row);

    sumcheck
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
