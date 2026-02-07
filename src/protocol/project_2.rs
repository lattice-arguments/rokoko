use std::any::Any;

use crate::{
    common::{
        arithmetic::{precompute_structured_values_fast, HALF_WAY_MOD_Q_RING_CF},
        config::{DEGREE, MOD_Q, NOF_BATCHES},
        hash::HashWrapper,
        matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{Representation, RingElement},
    },
    hexl::bindings::{add_mod, eltwise_reduce_mod, multiply_mod},
    protocol::config::{ConfigBase, SimpleConfig},
};

#[cfg(test)]
use crate::{
    common::matrix::new_vec_zero_preallocated, protocol::sumchecks::helpers::tensor_product_u64,
};

/// Computes J_batched = c'_1^T * J_embedded
///
/// J_embedded applies dual embedding: each coefficient j ∈ {-1,0,1} becomes a polynomial
/// where the constant term is j and non-constant terms are -j (to maintain inner product).
///
/// # Arguments
/// * `projection_matrix` - The projection matrix
/// * `c_1_values` - Precomputed c'_1 values (length = PROJECTION_HEIGHT)
///
/// # Returns
/// Vector of inner_width_ring ring elements representing the batched projection matrix
pub fn compute_j_batched(
    projection_matrix: &ProjectionMatrix,
    c_1_values: &[u64],
) -> Vec<RingElement> {
    use crate::common::matrix::new_vec_zero_preallocated;

    let inner_width_ring =
        projection_matrix.projection_ratio * (projection_matrix.projection_height / DEGREE);
    let mut j_batched = new_vec_zero_preallocated(inner_width_ring);

    for el in j_batched.iter_mut() {
        el.set_from(&HALF_WAY_MOD_Q_RING_CF);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        use std::arch::x86_64::*;

        let degree_blocks = DEGREE / 8;
        for i in 0..inner_width_ring {
            let base_index = i * DEGREE;
            let row_ptr = j_batched[i].v.as_mut_ptr();
            // For each coefficient position in the ring element
            for k in 0..projection_matrix.projection_height {
                let coeff = c_1_values[k];
                let coeff_vec = unsafe { _mm512_set1_epi64(coeff as i64) };

                // Process 8 coefficients at a time using AVX-512
                let mut j_ = 0;
                while j_ + 1 < degree_blocks {
                    let col_index_base0 = base_index + j_ * 8;
                    let col_index_base1 = col_index_base0 + 8;

                    let (k_pos0, k_inc0) = projection_matrix.get_row_masks_u8(k, col_index_base0);
                    let (k_pos1, k_inc1) = projection_matrix.get_row_masks_u8(k, col_index_base1);

                    unsafe {
                        let base_ptr0 = row_ptr.add(j_ * 8);
                        let base_ptr1 = row_ptr.add((j_ + 1) * 8);

                        let current0 = _mm512_load_epi64(base_ptr0 as *const i64);
                        let current1 = _mm512_load_epi64(base_ptr1 as *const i64);

                        let k_add0 = k_inc0 & k_pos0;
                        let k_sub0 = k_inc0 & !k_pos0;
                        let k_add1 = k_inc1 & k_pos1;
                        let k_sub1 = k_inc1 & !k_pos1;

                        let result0 = _mm512_mask_add_epi64(current0, k_add0, current0, coeff_vec);
                        let result0 = _mm512_mask_sub_epi64(result0, k_sub0, result0, coeff_vec);
                        let result1 = _mm512_mask_add_epi64(current1, k_add1, current1, coeff_vec);
                        let result1 = _mm512_mask_sub_epi64(result1, k_sub1, result1, coeff_vec);

                        _mm512_store_epi64(base_ptr0 as *mut i64, result0);
                        _mm512_store_epi64(base_ptr1 as *mut i64, result1);
                    }
                    j_ += 2;
                }

                if j_ < degree_blocks {
                    let col_index_base = base_index + j_ * 8;
                    let (k_pos, k_inc) = projection_matrix.get_row_masks_u8(k, col_index_base);

                    unsafe {
                        let base_ptr = row_ptr.add(j_ * 8);
                        let current = _mm512_load_epi64(base_ptr as *const i64);

                        let k_add = k_inc & k_pos;
                        let k_sub = k_inc & !k_pos;

                        let result = _mm512_mask_add_epi64(current, k_add, current, coeff_vec);
                        let result = _mm512_mask_sub_epi64(result, k_sub, result, coeff_vec);

                        _mm512_store_epi64(base_ptr as *mut i64, result);
                    }
                }
            }
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        println!("Using scalar code for compute_j_batched");
        for i in 0..inner_width_ring {
            let row = &mut j_batched[i].v;
            let base_index = i * DEGREE;
            for k in 0..projection_matrix.projection_height {
                let coeff = c_1_values[k];
                let mut j = 0;

                while j + 3 < DEGREE {
                    let col_index0 = base_index + j;
                    let col_index1 = col_index0 + 1;
                    let col_index2 = col_index0 + 2;
                    let col_index3 = col_index0 + 3;

                    let (is_positive0, is_non_zero0) = &projection_matrix[(k, col_index0)];
                    let (is_positive1, is_non_zero1) = &projection_matrix[(k, col_index1)];
                    let (is_positive2, is_non_zero2) = &projection_matrix[(k, col_index2)];
                    let (is_positive3, is_non_zero3) = &projection_matrix[(k, col_index3)];

                    if *is_non_zero0 {
                        if *is_positive0 {
                            row[j] += coeff;
                        } else {
                            row[j] -= coeff;
                        }
                    }
                    if *is_non_zero1 {
                        if *is_positive1 {
                            row[j + 1] += coeff;
                        } else {
                            row[j + 1] -= coeff;
                        }
                    }
                    if *is_non_zero2 {
                        if *is_positive2 {
                            row[j + 2] += coeff;
                        } else {
                            row[j + 2] -= coeff;
                        }
                    }
                    if *is_non_zero3 {
                        if *is_positive3 {
                            row[j + 3] += coeff;
                        } else {
                            row[j + 3] -= coeff;
                        }
                    }

                    j += 4;
                }

                while j < DEGREE {
                    let col_index = base_index + j;
                    let (is_positive, is_non_zero) = &projection_matrix[(k, col_index)];
                    if *is_non_zero {
                        if *is_positive {
                            row[j] += coeff;
                        } else {
                            row[j] -= coeff;
                        }
                    }
                    j += 1;
                }
            }
        }
    }

    // Convert j_batched to NTT for efficient multiplication
    for bp in j_batched.iter_mut() {
        unsafe {
            eltwise_reduce_mod(bp.v.as_mut_ptr(), bp.v.as_mut_ptr(), DEGREE as u64, MOD_Q);
        }
        bp.to_representation(Representation::IncompleteNTT);
        bp.conjugate_in_place();
    }

    j_batched
}

// Compute the coefficient-wise projection: V = (I_d ⊗ J) * coeff(W)
//
// This function projects the witness W through a structured projection matrix.
//
// Mathematical structure:
// - W ∈ R_q^{m × r} is the witness matrix (input in NTT representation)
// - J ∈ {-1,0,1}^{n_rp × m_rp} is the projection matrix (with n_rp = PROJECTION_HEIGHT)
// - coeff(W) ∈ Z_q^{m·DEGREE × r} is the witness converted to coefficient representation
// - V = (I_d ⊗ J) * coeff(W) ∈ Z_q^{d·n_rp × r} is the projected result
//   where d = m / projection_ratio and I_d is a d×d identity matrix
//
// The tensor product (I_d ⊗ J) means we apply J independently to each of d blocks
// of the coefficient-represented witness. Each block has m_rp·DEGREE coefficients.
//
// Output representation:
// V is in Z_q (coefficient space), but we represent it as ring elements for efficiency:
// V' = embed_coefficients(V) ∈ R_q^{d·n_rp / DEGREE × r}
// This packs DEGREE consecutive coefficients of V into each ring element.
pub fn project_coefficients(
    witness: &VerticallyAlignedMatrix<RingElement>,
    projection_matrix: &ProjectionMatrix,
) -> VerticallyAlignedMatrix<RingElement> {
    let mut witness_coeff =
        VerticallyAlignedMatrix::new_zero_preallocated(witness.height, witness.width);

    witness_coeff.data.clone_from_slice(&witness.data);

    for i in 0..witness_coeff.data.len() {
        witness_coeff.data[i].from_incomplete_ntt_to_even_odd_coefficients();
        witness_coeff.data[i].from_even_odd_coefficients_to_coefficients(); // TODO: even-odd is enough
    }

    #[cfg(feature = "debug-hardness")]
    {
        println!("Projecting coefficients with projection matrix:");
        let norm = l2_norm_coeffs(&witness_coeff.data);
        println!("L2 norm of witness coefficients: {}", norm);
    }

    // Allocate the output matrix for the projected result
    // The projection reduces witness.height by projection_ratio
    // Result is in coefficient representation (will be packed into ring elements)
    let mut image_ct = VerticallyAlignedMatrix::new_zero_preallocated(
        witness.height / projection_matrix.projection_ratio,
        witness.width,
    );

    for el in image_ct.data.iter_mut() {
        el.set_from(&HALF_WAY_MOD_Q_RING_CF);
    }

    // Verify dimensions: each ring element in image corresponds to projection_ratio
    // ring elements in the witness (after applying the projection matrix J)
    debug_assert_eq!(image_ct.width, witness.width);
    debug_assert_eq!(
        image_ct.height * projection_matrix.projection_ratio,
        witness.height
    );

    // Process each column independently (no interaction between columns)
    for col in 0..witness.used_cols {
        // Process the projection in chunks
        // Each chunk processes (PROJECTION_HEIGHT / DEGREE) ring elements of the output
        // which corresponds to PROJECTION_HEIGHT coefficients in the result
        for rows_chunk in 0..image_ct.height / (projection_matrix.projection_height / DEGREE) {
            // Extract the corresponding slice of witness coefficients for this chunk
            // This is the input to one application of the projection matrix J
            let subwitness = witness_coeff.col_slice(
                col,
                rows_chunk
                    * projection_matrix.projection_ratio
                    * (projection_matrix.projection_height / DEGREE),
                (rows_chunk + 1)
                    * projection_matrix.projection_ratio
                    * (projection_matrix.projection_height / DEGREE),
            );

            // Get mutable slice of the output for this chunk
            let projection_subimage = image_ct.col_slice_mut(
                col,
                rows_chunk * (projection_matrix.projection_height / DEGREE),
                (rows_chunk + 1) * (projection_matrix.projection_height / DEGREE),
            );

            // Apply projection matrix J to this chunk
            // J has PROJECTION_HEIGHT rows (output coefficients)
            for inner_row in 0..projection_matrix.projection_height {
                // Map this output coefficient to its position in a ring element
                let current_projection_row = inner_row / DEGREE; // Which ring element
                let current_projection_coeff_index = inner_row % DEGREE; // Which coeff in that element

                let target = &mut projection_subimage[current_projection_row].v
                    [current_projection_coeff_index];
                // Compute the inner product: projection_subimage[inner_row] = J[inner_row, :] · subwitness
                // J has (projection_ratio * PROJECTION_HEIGHT) columns

                #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
                {
                    use std::arch::x86_64::*;

                    let total_cols = projection_matrix.projection_width;
                    debug_assert!(total_cols % 8 == 0);
                    let chunks_per_row = projection_matrix.chunks_per_row;
                    let blocks_per_ring = DEGREE / 8;


                    unsafe {
                        let row_base = inner_row * chunks_per_row;
                        let kpos_row = projection_matrix.k_pos_plan.as_ptr().add(row_base);
                        let kinc_row = projection_matrix.k_inc_plan.as_ptr().add(row_base);

                        let mut acc0 = _mm512_setzero_si512();
                        let mut acc1 = _mm512_setzero_si512();

                        let mut chunk_idx = 0usize;

                        while chunk_idx + 1 < chunks_per_row {
                            let k_pos0 = *kpos_row.add(chunk_idx);
                            let k_inc0 = *kinc_row.add(chunk_idx);

                            let add0: __mmask8 = (k_inc0 &  k_pos0) as __mmask8;
                            let sub0: __mmask8 = (k_inc0 & !k_pos0) as __mmask8;

                            let ring0 = chunk_idx / blocks_per_ring;
                            let off0  = (chunk_idx - ring0 * blocks_per_ring) * 8;

                            let coeff0 = _mm512_load_epi64(
                                subwitness.get_unchecked(ring0).v.as_ptr().add(off0) as *const i64
                            );

                            acc0 = _mm512_mask_add_epi64(acc0, add0, acc0, coeff0);
                            acc0 = _mm512_mask_sub_epi64(acc0, sub0, acc0, coeff0);

                            let k_pos1 = *kpos_row.add(chunk_idx + 1);
                            let k_inc1 = *kinc_row.add(chunk_idx + 1);

                            let add1: __mmask8 = (k_inc1 &  k_pos1) as __mmask8;
                            let sub1: __mmask8 = (k_inc1 & !k_pos1) as __mmask8;

                            let ring1 = (chunk_idx + 1) / blocks_per_ring;
                            let off1  = ((chunk_idx + 1) - ring1 * blocks_per_ring) * 8;

                            let coeff1 = _mm512_load_epi64(
                                subwitness.get_unchecked(ring1).v.as_ptr().add(off1) as *const i64
                            );

                            acc1 = _mm512_mask_add_epi64(acc1, add1, acc1, coeff1);
                            acc1 = _mm512_mask_sub_epi64(acc1, sub1, acc1, coeff1);

                            chunk_idx += 2;
                        }

                        if chunk_idx < chunks_per_row {
                            let k_pos = *kpos_row.add(chunk_idx);
                            let k_inc = *kinc_row.add(chunk_idx);

                            let add: __mmask8 = (k_inc &  k_pos) as __mmask8;
                            let sub: __mmask8 = (k_inc & !k_pos) as __mmask8;

                            let ring = chunk_idx / blocks_per_ring;
                            let off  = (chunk_idx - ring * blocks_per_ring) * 8;

                            let coeff = _mm512_load_epi64(
                                subwitness.get_unchecked(ring).v.as_ptr().add(off) as *const i64
                            );

                            acc0 = _mm512_mask_add_epi64(acc0, add, acc0, coeff);
                            acc0 = _mm512_mask_sub_epi64(acc0, sub, acc0, coeff);
                        }

                        let acc = _mm512_add_epi64(acc0, acc1);
                        let sum = _mm512_reduce_add_epi64(acc) as u64;
                        *target = target.wrapping_add(sum);
                    }
                }

                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
                {
                    let total_cols =
                        projection_matrix.projection_ratio * projection_matrix.projection_height;
                    for i in 0..total_cols {
                        let (is_positive, is_non_zero) = &projection_matrix[(inner_row, i)];
                        if !*is_non_zero {
                            continue;
                        }

                        // Add or subtract the witness coefficient depending on J's sign
                        if *is_positive {
                            *target += subwitness[i / DEGREE].v[i % DEGREE];
                        } else {
                            *target -= subwitness[i / DEGREE].v[i % DEGREE];
                        }
                    }
                }
            }
        }
    }
    for el in image_ct.data.iter_mut() {
        unsafe {
            eltwise_reduce_mod(el.v.as_mut_ptr(), el.v.as_mut_ptr(), DEGREE as u64, MOD_Q);
        }
    }

    #[cfg(feature = "debug-hardness")]
    {
        let norm = l2_norm_coeffs(&image_ct.data);
        println!("L2 norm of projected witness coefficients: {}", norm);
    }
    image_ct
}

// Compute batched projection: c'^t V where V = (I_d ⊗ embed_dual(J)) * (W)
//
// Instead of computing the full projection V and then batching, we perform the batching
// during the projection computation using a tensor product decomposition.
//
// Mathematical structure:
// - W ∈ R_q^{m × r} is the witness matrix
// - J ∈ {-1,0,1}^{n_rp × m_rp} is the projection matrix (with n_rp = PROJECTION_HEIGHT)
// - V = (I_d ⊗ embed_dual(J)) * W ∈ R^{d·n_rp × r} is the projected result in coefficient form
// - c' ∈ Z^{d·n_rp·r} is the batching challenge, decomposed as c' = c'_0 ⊗ c'_1 where:
//   * c'_0 ∈ Z^d (batches over coefficient blocks in the image)
//   * c'_1 ∈ Z^{n_rp} (batches over PROJECTION_HEIGHT coefficients)
//   * d = (witness.height / projection_ratio) * DEGREE / PROJECTION_HEIGHT
//
// Algorithm:
// 1. J_batched = c'_1^T * J_embedded (batch projection matrix rows with dual embedding)
// 2. result = c'_0^T ⊗ J_batched · W (tensor and vector-matrix product)
//
// Each c'_i is structured as a tensor product: c'_i = ⊗ (1 - l_j, l_j)
// This allows us to sample only log(|c'_i|) random values (the layers l_j)
// and compute any c'_i[k] on-the-fly using the binary representation of k.

pub struct BatchedProjectionChallenges {
    pub c_0_values: Vec<u64>,
    pub c_1_values: Vec<u64>,
    pub c_1_layers: Vec<u64>,
    pub c_2_values: Vec<u64>, // for columns, not used here but for consistency
    pub j_batched: Vec<RingElement>, // this is technically not needed, but since it's computed anyway, we return it for reuse
}

pub struct BatchedProjectionChallengesSuccinct {
    pub c_0_layers: Vec<u64>,
    pub c_1_layers: Vec<u64>,
    pub c_2_layers: Vec<u64>, // for columns, not used here but for consistency
    pub j_batched: Vec<RingElement>, // this is technically not needed, but since it's computed anyway, we return it for reuse
}

pub fn sample_layers(
    projection_matrix: &ProjectionMatrix,
    witness_width: usize,
    witness_height: usize,
    hash_wrapper: &mut HashWrapper,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    // Sample structured challenge layers
    let d = (witness_height / projection_matrix.projection_ratio) * DEGREE
        / projection_matrix.projection_height;

    let c_0_layers: Vec<u64> = (0..d.ilog2())
        .map(|_| hash_wrapper.sample_u64_mod_q())
        .collect();

    let c_1_layers: Vec<u64> = (0..projection_matrix.projection_height.ilog2())
        // .map(|_| hash_wrapper.sample_u64_mod_q())
        .map(|_| 1)
        .collect();

    let c_2_layers: Vec<u64> = (0..witness_width.ilog2())
        .map(|_| hash_wrapper.sample_u64_mod_q())
        .collect();

    (c_0_layers, c_1_layers, c_2_layers)
}

fn batch_projection_into(
    result: &mut [RingElement],
    witness: &VerticallyAlignedMatrix<RingElement>,
    projection_matrix: &ProjectionMatrix,
    hash_wrapper: &mut HashWrapper,
    is_simple_config: bool,
) -> BatchedProjectionChallenges {
    // Sample structured challenge layers
    // c'_0 is over d (number of blocks in image_ct when viewed coefficient-wise)
    // c'_1 is over n_rp (projection height coefficients)

    // d = image_ct.height * DEGREE / PROJECTION_HEIGHT
    // let c_0_layers: Vec<u64> = (0..d.ilog2())
    //     .map(|_| hash_wrapper.sample_u64_mod_q())
    //     .collect();

    // let c_1_layers: Vec<u64> = (0..PROJECTION_HEIGHT.ilog2())
    //     .map(|_| hash_wrapper.sample_u64_mod_q())
    //     .collect();
    let (c_0_layers, c_1_layers, c_2_layers) = sample_layers(
        projection_matrix,
        if is_simple_config { 1 } else { witness.width },
        witness.height,
        hash_wrapper,
    );
    // Precompute all structured row values for c_0 and c_1
    // For k layers, we compute all 2^k values in O(2^k) time

    let c_0_values = precompute_structured_values_fast(&c_0_layers);
    let c_1_values = precompute_structured_values_fast(&c_1_layers);
    let c_2_values = precompute_structured_values_fast(&c_2_layers);

    // ===== Step 1: Batch projection matrix with dual embedding =====
    let j_batched = compute_j_batched(projection_matrix, &c_1_values);
    let inner_width_ring =
        projection_matrix.projection_ratio * (projection_matrix.projection_height / DEGREE);

    // ===== Step 2: Apply c'_0 batching and compute final inner product =====
    // Process each column independently (no batching across columns)
    // For each column, split witness into num_chunks chunks where each chunk
    // corresponds to one application of the projection matrix.
    // We compute for each column: result[col] = Σ_chunk c'_0[chunk] * <J_batched, W[chunk, col]>
    let num_chunks = witness.height / inner_width_ring;

    for col in 0..witness.used_cols {
        let mut col_result = RingElement::zero(Representation::IncompleteNTT);

        for chunk in 0..num_chunks {
            let c_0_coeff = c_0_values[chunk];
            let mut chunk_result = RingElement::zero(Representation::IncompleteNTT);

            // Inner product of j_batched with the corresponding chunk of witness column
            for i in 0..inner_width_ring {
                let mut temp = RingElement::zero(Representation::IncompleteNTT);
                temp *= (&witness[(chunk * inner_width_ring + i, col)], &j_batched[i]);
                chunk_result += &temp;
            }

            // Multiply by c_0 coefficient and accumulate
            for deg in 0..DEGREE {
                unsafe {
                    // TODO: vectorize
                    let temp = multiply_mod(chunk_result.v[deg], c_0_coeff, MOD_Q);
                    col_result.v[deg] = add_mod(col_result.v[deg], temp, MOD_Q);
                }
            }
        }

        result[col] = col_result;
    }

    //////

    BatchedProjectionChallenges {
        c_0_values,
        c_1_values,
        c_1_layers,
        c_2_values,
        j_batched,
    }
}

pub fn batch_projection_n_times(
    witness: &VerticallyAlignedMatrix<RingElement>,
    projection_matrix: &ProjectionMatrix,
    hash_wrapper: &mut HashWrapper,
    n: usize,
    is_simple_config: bool,
) -> (
    HorizontallyAlignedMatrix<RingElement>,
    [BatchedProjectionChallenges; NOF_BATCHES],
) {
    debug_assert_eq!(n, NOF_BATCHES, "Only n=NOF_BATCHES is expected");
    let mut result = HorizontallyAlignedMatrix::new_zero_preallocated(n, witness.width);
    let challenges = [
        batch_projection_into(
            &mut result.row_slice_mut(0),
            witness,
            projection_matrix,
            hash_wrapper,
            is_simple_config,
        ),
        batch_projection_into(
            &mut result.row_slice_mut(1),
            witness,
            projection_matrix,
            hash_wrapper,
            is_simple_config,
        ),
    ];

    // let expanded_c_0 = challenges[0]
    //     .c_0_values
    //     .iter()
    //     .map(|&x| RingElement::constant(x, Representation::IncompleteNTT))
    //     .collect::<Vec<RingElement>>();

    // let j_folded_expanded = tensor_product(&expanded_c_0, &challenges[0].j_batched);

    // // let ip = inner_product(
    // //     &j_folded_expanded,
    // //     witness.col(0)
    // // );

    // let mut ip = RingElement::zero(Representation::IncompleteNTT);
    // for i in 0..j_folded_expanded.len() {
    //     let mut temp = RingElement::zero(Representation::IncompleteNTT);
    //     temp *= (&witness[(i, 0)], &j_folded_expanded[i]);
    //     ip += &temp;
    // }

    // debug_assert_eq!(
    //     ip,
    //     result[(1, 0)],
    //     "Tensor product folding should match batched projection"
    // );
    // let mut challenges = Vec::with_capacity(n);
    // for i in 0..n {
    //     challenges.push(batch_projection_into(
    //         &mut result.row_slice_mut(i),
    //         witness,
    //         projection_matrix,
    //         hash_wrapper,
    //     ));
    // }
    (result, challenges)
}

pub fn verifier_sample_projection_challenges(
    projection_matrix: &ProjectionMatrix,
    // config: &Config,
    config: &dyn ConfigBase,
    hash_wrapper: &mut HashWrapper,
) -> BatchedProjectionChallengesSuccinct {
    let is_simple_config = (config as &dyn Any).is::<SimpleConfig>(); // we don't batch over columns in simple config

    let (c_0_layers, c_1_layers, c_2_layers) = sample_layers(
        projection_matrix,
        if is_simple_config {
            1
        } else {
            config.witness_width()
        },
        config.witness_height(),
        hash_wrapper,
    );

    let c_1_values = precompute_structured_values_fast(&c_1_layers);
    let j_batched = compute_j_batched(projection_matrix, &c_1_values);

    BatchedProjectionChallengesSuccinct {
        c_0_layers,
        c_1_layers,
        c_2_layers,
        j_batched,
    }
}

#[test]
fn test_batch_projection() {
    // Test that batch_projection correctly computes c'^t * vec(V)
    // where V = project_coefficients(witness), checked separately for each column
    //
    // This verifies the correctness of the batched projection algorithm by:
    // 1. Computing V explicitly via project_coefficients
    // 2. Computing the same result efficiently via batch_projection
    // 3. Sampling the same random challenges for both
    // 4. For each column: manually computing c'^t * vec(V[:, col])
    // 5. Comparing all results to ensure consistency

    let witness = VerticallyAlignedMatrix {
        data: vec![RingElement::random(Representation::IncompleteNTT); 16],
        width: 2,
        height: 8,
        used_cols: 2,
    };

    let mut projection_matrix = ProjectionMatrix::new(2, 256);
    let mut hash_wrapper = HashWrapper::new();
    projection_matrix.sample(&mut hash_wrapper);

    // Compute the full projection V = (I_d ⊗ J) * coeff(W) explicitly
    let image_ct = project_coefficients(&witness, &projection_matrix);

    let (c_0_layers, c_1_layers, _c_2_layers) = sample_layers(
        &projection_matrix,
        witness.width,
        witness.height,
        &mut hash_wrapper,
    );

    let c_0_values = precompute_structured_values_fast(&c_0_layers);
    let c_1_values = precompute_structured_values_fast(&c_1_layers);

    let num_chunks_in_image = image_ct.height / (projection_matrix.projection_height / DEGREE);

    // Compute expected results for each column separately
    let mut expected_cts = vec![0u64; witness.width];

    for col in 0..image_ct.width {
        let mut expected_ct = 0u64;

        for chunk_idx in 0..num_chunks_in_image {
            let c_0_coeff = c_0_values[chunk_idx];
            // Each chunk contains projection_matrix.projection_height coefficients
            for coeff_idx in 0..projection_matrix.projection_height {
                let c_1_coeff = c_1_values[coeff_idx];
                // Map flat coefficient index to (row, degree) in the ring element matrix
                let row_in_chunk = coeff_idx / DEGREE; // Which ring element in this chunk
                let deg = coeff_idx % DEGREE; // Which coefficient in that ring element
                let row = chunk_idx * (projection_matrix.projection_height / DEGREE) + row_in_chunk;
                unsafe {
                    // The challenge at this position is c'_0[chunk] * c'_1[coeff]
                    let c_combined = multiply_mod(c_0_coeff, c_1_coeff, MOD_Q);

                    let temp = multiply_mod(image_ct[(row, col)].v[deg], c_combined, MOD_Q);
                    expected_ct = add_mod(expected_ct, temp, MOD_Q);
                }
            }
        }

        expected_cts[col] = expected_ct;
    }

    // Now compute using the optimized batch_projection which does both operations in one pass
    // Create a fresh hash_wrapper to sample the same challenges
    let mut hash_wrapper2 = HashWrapper::new();
    projection_matrix.sample(&mut hash_wrapper2); // Consume same randomness to sync state

    let mut result = new_vec_zero_preallocated(witness.width);
    batch_projection_into(
        &mut result,
        &witness,
        &projection_matrix,
        &mut hash_wrapper2,
        false,
    );

    // Check each column separately
    for col in 0..witness.width {
        let mut col_result = result[col].clone();
        col_result.to_representation(Representation::Coefficients);

        debug_assert_eq!(
            col_result.v[0], expected_cts[col],
            "batch_projection column {} should produce the same result as computing project_coefficients followed by batching",
            col
        );
    }
}

#[test]
fn test_const_term_relation_to_prove() {
    let witness = VerticallyAlignedMatrix {
        data: vec![RingElement::random(Representation::IncompleteNTT); 8 * 64],
        width: 8,
        height: 64,
        used_cols: 8,
    };
    let mut projection_matrix = ProjectionMatrix::new(4, 256);

    let mut hash_wrapper = HashWrapper::new();
    projection_matrix.sample(&mut hash_wrapper);

    // Compute the full projection V = (I_d ⊗ J) * coeff(W) explicitly
    let mut image_ct = project_coefficients(&witness, &projection_matrix);
    for el in image_ct.data.iter_mut() {
        el.to_representation(Representation::IncompleteNTT);
    }
    debug_assert_eq!(image_ct.height, 16);

    debug_assert_eq!(image_ct.width, 8);

    let mut batched_projected_witness =
        HorizontallyAlignedMatrix::new_zero_preallocated(1, witness.width);

    let challenges = batch_projection_into(
        &mut batched_projected_witness.row_slice_mut(0),
        &witness,
        &projection_matrix,
        &mut hash_wrapper,
        false,
    );

    // Now, we want to check if
    // let B = batched_projected_witness
    // let V = image_ct
    // let V' = coefficients of V
    // assume that challenges are "expanded" into vectors c_0 c_1 c_2
    // We need to check if
    // constant_term( c_0^T B c_2) = (c_0^T \otimes c_1^T ) V' c_2
    // Let c_1 de split into (e_0 \otimes e_1) where e_1 has length DEGREE and e_0 has length PROJECTION_HEIGHT / DEGREE
    // Let e = embed_dual(J) c_1
    // Then, we need to check if
    // constant_term( B c_2) = constant_term((c_0^T \otimes e_0^T) e V c_2)
    // due to memory alignment:
    // constant_term(<B, c_2>) = e <c_2 \otimes c_0 \otimes e_0, V>
    let c_2_values = challenges.c_2_values;
    let c_1_values = challenges.c_1_values;
    let c_0_values = challenges.c_0_values;
    let c1_layers = challenges.c_1_layers;
    let (e_0_values, e_1_values) = {
        let mut e_0_layers = Vec::new();
        let mut e_1_layers = Vec::new();
        for (i, &layer) in c1_layers.iter().enumerate() {
            if i < c1_layers.len() - DEGREE.ilog2() as usize {
                e_0_layers.push(layer);
            } else {
                e_1_layers.push(layer);
            }
        }
        (
            precompute_structured_values_fast(&e_0_layers),
            precompute_structured_values_fast(&e_1_layers),
        )
    };

    let _tensor_product = tensor_product_u64(&e_0_values, &e_1_values);
    debug_assert_eq!(_tensor_product, c_1_values);
    let lhs_multipier_ring = c_2_values
        .iter()
        .map(|&x| RingElement::constant(x, Representation::IncompleteNTT))
        .collect::<Vec<RingElement>>();

    let rhs_multipier_ring = {
        // c_2 \otimes c_0 \otimes e_0
        // first over u64
        let values_0 = tensor_product_u64(&c_2_values, &c_0_values);
        let values_1 = tensor_product_u64(&values_0, &e_0_values);
        let vals_over_ring = values_1
            .iter()
            .map(|&x| RingElement::constant(x, Representation::IncompleteNTT))
            .collect::<Vec<RingElement>>();
        vals_over_ring
    };

    let e = {
        let mut e = RingElement::zero(Representation::Coefficients);
        for (i, &val) in e_1_values.iter().enumerate() {
            e.v[i as usize] = val;
        }
        e.from_coefficients_to_even_odd_coefficients();
        e.from_even_odd_coefficients_to_incomplete_ntt_representation();
        e.conjugate_in_place();
        e
    };

    let mut lhs = {
        let mut acc = RingElement::zero(Representation::IncompleteNTT);
        for col in 0..batched_projected_witness.data.len() {
            let mut temp = RingElement::zero(Representation::IncompleteNTT);
            temp *= (
                &batched_projected_witness.data[col],
                &lhs_multipier_ring[col],
            );
            acc += &temp;
        }
        acc
    };

    let mut rhs = {
        let mut acc = RingElement::zero(Representation::IncompleteNTT);
        for i in 0..image_ct.data.len() {
            let mut temp = RingElement::zero(Representation::IncompleteNTT);
            temp *= (&image_ct.data[i], &rhs_multipier_ring[i]);
            acc += &temp;
        }
        acc *= &e;
        acc
    };
    lhs.to_representation(Representation::Coefficients);
    rhs.to_representation(Representation::Coefficients);
    debug_assert_eq!(
        lhs.v[0], rhs.v[0],
        "Constant terms of LHS and RHS should match"
    );
}
