use num::traits::ops::mul_add;

use crate::{
    common::{
        arithmetic::{
            inner_product, inner_product_into, precompute_structured_values,
            precompute_structured_values_fast, HALF_WAY_MOD_Q_RING_CF,
        },
        config::{self, DEGREE, MOD_Q, NOF_BATCHES},
        hash::HashWrapper,
        matrix::{new_vec_zero_preallocated, HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        pool::preallocate_ring_element_vecs,
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{Representation, RingElement},
        structured_row::StructuredRow,
    },
    hexl::bindings::{add_mod, eltwise_reduce_mod, multiply_mod, sub_mod},
    protocol::{
        config::Config, open::evaluation_point_to_structured_row,
        sumchecks::helpers::tensor_product,
    },
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
    use crate::{
        common::matrix::new_vec_zero_preallocated,
        hexl::bindings::{add_mod, sub_mod},
    };

    let inner_width_ring =
        projection_matrix.projection_ratio * (projection_matrix.projection_height / DEGREE);
    let mut j_batched = new_vec_zero_preallocated(inner_width_ring);

    for el in j_batched.iter_mut() {
        el.set_from(&HALF_WAY_MOD_Q_RING_CF);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                compute_j_batched_avx512(&mut j_batched, projection_matrix, c_1_values, inner_width_ring);
            }
        } else {
            compute_j_batched_scalar(&mut j_batched, projection_matrix, c_1_values, inner_width_ring);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        compute_j_batched_scalar(&mut j_batched, projection_matrix, c_1_values, inner_width_ring);
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn compute_j_batched_avx512(
    j_batched: &mut [RingElement],
    projection_matrix: &ProjectionMatrix,
    c_1_values: &[u64],
    inner_width_ring: usize,
) {
    use core::arch::x86_64::*;

    #[inline(always)]
    unsafe fn masked_signed_add_epi64(acc: __m512i, a: __m512i, k_inc: u8, k_pos: u8) -> __m512i {
        // signed = (k_pos ? +a : -a)
        let neg = _mm512_sub_epi64(_mm512_setzero_si512(), a);
        let signed = _mm512_mask_mov_epi64(neg, k_pos, a);

        // Add signed values to accumulator, but only for included lanes
        _mm512_mask_add_epi64(acc, k_inc, acc, signed)
    }

    for i in 0..inner_width_ring {
        let base_ptr = j_batched[i].v.as_mut_ptr();
        
        // Process each chunk of 8 coefficients
        for j_chunk in 0..DEGREE / 8 {
            let j_base = j_chunk * 8;
            
            // Load current 8 values
            let mut acc = _mm512_loadu_si512(base_ptr.add(j_base) as *const _);
            
            // Accumulate contributions from all projection rows
            for k in 0..projection_matrix.projection_height {
                let coeff = c_1_values[k];
                
                // Build masks for this row across 8 columns
                let mut k_inc: u8 = 0; // which lanes to include (is_non_zero)
                let mut k_pos: u8 = 0; // which lanes are positive (is_positive)
                
                for lane in 0..8 {
                    let col_index = i * DEGREE + j_base + lane;
                    let (is_positive, is_non_zero) = &projection_matrix[(k, col_index)];
                    
                    if *is_non_zero {
                        k_inc |= 1 << lane;
                        if *is_positive {
                            k_pos |= 1 << lane;
                        }
                    }
                }
                
                // Skip if no contribution from this row
                if k_inc == 0 {
                    continue;
                }
                
                // Broadcast coefficient to all lanes
                let coeff_vec = _mm512_set1_epi64(coeff as i64);
                
                // Add or subtract coefficient based on masks
                acc = masked_signed_add_epi64(acc, coeff_vec, k_inc, k_pos);
            }
            
            // Store result back
            _mm512_storeu_si512(base_ptr.add(j_base) as *mut _, acc);
        }
    }
}

#[inline(always)]
fn compute_j_batched_scalar(
    j_batched: &mut [RingElement],
    projection_matrix: &ProjectionMatrix,
    c_1_values: &[u64],
    inner_width_ring: usize,
) {
    for i in 0..inner_width_ring {
        for k in 0..projection_matrix.projection_height {
            for j_ in 0..DEGREE / 8 {
                for bit in 0..8 {
                    let j = j_ * 8 + bit;
                    let col_index = i * DEGREE + j;
                    let coeff = c_1_values[k];
                    unsafe {
                        let (is_positive, is_non_zero) = &projection_matrix[(k, col_index)];
                        if !*is_non_zero {
                            continue;
                        }
                        if *is_positive {
                            j_batched[i].v[j] += coeff;
                        } else {
                            j_batched[i].v[j] -= coeff;
                        }
                    }
                }
            }
        }
    }
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
    // Convert witness from NTT representation to coefficient representation
    // Each ring element contains DEGREE coefficients that we'll project separately
    let mut witness_coeff =
        VerticallyAlignedMatrix::new_zero_preallocated(witness.height, witness.width);

    witness_coeff.data.clone_from_slice(&witness.data);

    for i in 0..witness_coeff.data.len() {
        witness_coeff.data[i].from_incomplete_ntt_to_even_odd_coefficients();
        witness_coeff.data[i].from_even_odd_coefficients_to_coefficients(); // TODO: even-odd is enough
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
    assert_eq!(image_ct.width, witness.width);
    assert_eq!(
        image_ct.height * projection_matrix.projection_ratio,
        witness.height
    );

    // Process each column independently (no interaction between columns)
    for col in 0..witness.width {
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

                // Compute the inner product: projection_subimage[inner_row] = J[inner_row, :] · subwitness
                // J has (projection_ratio * PROJECTION_HEIGHT) columns
                for i in 0..projection_matrix.projection_ratio * projection_matrix.projection_height
                {
                    let (is_positive, is_non_zero) = &projection_matrix[(inner_row, i)];
                    if !*is_non_zero {
                        continue;
                    }

                    // Add or subtract the witness coefficient depending on J's sign
                    if *is_positive {
                        // TODO: use vectorisation
                        projection_subimage[current_projection_row].v
                            [current_projection_coeff_index] +=
                            subwitness[i / DEGREE].v[i % DEGREE];
                    } else {
                        // TODO: use vectorisation
                        projection_subimage[current_projection_row].v
                            [current_projection_coeff_index] -=
                            subwitness[i / DEGREE].v[i % DEGREE];
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
        .map(|_| hash_wrapper.sample_u64_mod_q())
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
) -> BatchedProjectionChallenges {
    // Sample structured challenge layers
    // c'_0 is over d (number of blocks in image_ct when viewed coefficient-wise)
    // c'_1 is over n_rp (projection height coefficients)

    // d = image_ct.height * DEGREE / PROJECTION_HEIGHT
    let d = (witness.height / projection_matrix.projection_ratio) * DEGREE
        / projection_matrix.projection_height;
    // let c_0_layers: Vec<u64> = (0..d.ilog2())
    //     .map(|_| hash_wrapper.sample_u64_mod_q())
    //     .collect();

    // let c_1_layers: Vec<u64> = (0..PROJECTION_HEIGHT.ilog2())
    //     .map(|_| hash_wrapper.sample_u64_mod_q())
    //     .collect();
    // TODO: fix randomness consumption
    let (c_0_layers, c_1_layers, c_2_layers) = sample_layers(
        projection_matrix,
        witness.width,
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

    for col in 0..witness.width {
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
                    let temp = multiply_mod(chunk_result.v[deg], c_0_coeff, MOD_Q);
                    col_result.v[deg] = add_mod(col_result.v[deg], temp, MOD_Q);
                }
            }
        }

        result[col] = col_result;
    }
    BatchedProjectionChallenges {
        c_0_values,
        c_1_values,
        c_2_values,
        j_batched,
    }
}

pub fn batch_projection_n_times(
    witness: &VerticallyAlignedMatrix<RingElement>,
    projection_matrix: &ProjectionMatrix,
    hash_wrapper: &mut HashWrapper,
    n: usize,
) -> (
    HorizontallyAlignedMatrix<RingElement>,
    [BatchedProjectionChallenges; NOF_BATCHES],
) {
    assert_eq!(n, NOF_BATCHES, "Only n=NOF_BATCHES is expected");
    let mut result = HorizontallyAlignedMatrix::new_zero_preallocated(n, witness.width);
    let challenges = [
        batch_projection_into(
            &mut result.row_slice_mut(0),
            witness,
            projection_matrix,
            hash_wrapper,
        ),
        batch_projection_into(
            &mut result.row_slice_mut(1),
            witness,
            projection_matrix,
            hash_wrapper,
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

    // assert_eq!(
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
    config: &Config,
    hash_wrapper: &mut HashWrapper,
) -> BatchedProjectionChallengesSuccinct {
    let (c_0_layers, c_1_layers, c_2_layers) = sample_layers(
        projection_matrix,
        config.witness_width,
        config.witness_height,
        hash_wrapper,
    );

    let c_1_values = precompute_structured_values_fast(&c_1_layers)
        .iter()
        .map(|&x| RingElement::constant(x, Representation::IncompleteNTT))
        .collect::<Vec<RingElement>>();

    let j_batched = compute_j_batched(
        projection_matrix,
        &c_1_values.iter().map(|el| el.v[0]).collect::<Vec<u64>>(),
    );

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
    };

    let mut projection_matrix = ProjectionMatrix::new(2, 256);
    let mut hash_wrapper = HashWrapper::new();
    projection_matrix.sample(&mut hash_wrapper);

    // Compute the full projection V = (I_d ⊗ J) * coeff(W) explicitly
    let image_ct = project_coefficients(&witness, &projection_matrix);

    // Sample the same structured challenges as batch_projection
    // No column batching: we process each column independently

    // c'_0: batches over d coefficient blocks in the image
    // d = (number of ring elements in image) * (coefficients per ring element) / (block size)
    let d = image_ct.height * DEGREE / projection_matrix.projection_height;
    // let c_0_layers: Vec<u64> = (0..d.ilog2())
    //     .map(|_| hash_wrapper.sample_u64_mod_q())
    //     .collect();

    // // c'_1: batches over PROJECTION_HEIGHT coefficients within each block
    // let c_1_layers: Vec<u64> = (0..PROJECTION_HEIGHT.ilog2())
    //     .map(|_| hash_wrapper.sample_u64_mod_q())
    //     .collect();

    // TODO: fix randomness consumption
    // let c_0_layers: Vec<u64> = (0..d.ilog2()).map(|_| 2u64).collect();

    // let c_1_layers: Vec<u64> = (0..projection_matrix.projection_height.ilog2())
    //     .map(|_| 2u64)
    //     .collect();

    let (c_0_layers, c_1_layers, _c_2_layers) = sample_layers(
        &projection_matrix,
        witness.width,
        witness.height,
        &mut hash_wrapper,
    );
    // Precompute all structured row values
    // let precompute_structured_values = |layers: &[u64]| -> Vec<u64> {
    //     let size = 1 << layers.len();
    //     let mut values = vec![1u64; size];

    //     for (layer_idx, &layer) in layers.iter().rev().enumerate() {
    //         let layer_complement = unsafe { sub_mod(1, layer, MOD_Q) };

    //         for i in 0..size {
    //             if (i >> layer_idx) & 1 == 1 {
    //                 unsafe {
    //                     values[i] = multiply_mod(values[i], layer, MOD_Q);
    //                 }
    //             } else {
    //                 unsafe {
    //                     values[i] = multiply_mod(values[i], layer_complement, MOD_Q);
    //                 }
    //             }
    //         }
    //     }

    //     values
    // };

    let c_0_values = precompute_structured_values_fast(&c_0_layers);
    let c_1_values = precompute_structured_values_fast(&c_1_layers);

    let inner_width_ring =
        projection_matrix.projection_ratio * (projection_matrix.projection_height / DEGREE);
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
    );

    // Check each column separately
    for col in 0..witness.width {
        let mut col_result = result[col].clone();
        col_result.to_representation(Representation::Coefficients);

        assert_eq!(
            col_result.v[0], expected_cts[col],
            "batch_projection column {} should produce the same result as computing project_coefficients followed by batching",
            col
        );
    }
}
