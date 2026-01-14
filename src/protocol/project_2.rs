use num::traits::ops::mul_add;

use crate::{
    common::{
        arithmetic::{inner_product, inner_product_into},
        config::{DEGREE, MOD_Q, PROJECTION_HEIGHT},
        hash::HashWrapper,
        matrix::{new_vec_zero_preallocated, VerticallyAlignedMatrix},
        pool::preallocate_ring_element_vecs,
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{Representation, RingElement},
        structured_row::StructuredRow,
    },
    hexl::bindings::{add_mod, multiply_mod, sub_mod},
    protocol::open::evaluation_point_to_structured_row,
};


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
fn project_coefficients(
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
        el.representation = Representation::Coefficients; // TODO: let's just preallocate properly
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
        for rows_chunk in 0..image_ct.height / (PROJECTION_HEIGHT / DEGREE) {
            // Extract the corresponding slice of witness coefficients for this chunk
            // This is the input to one application of the projection matrix J
            let subwitness = witness_coeff.col_slice(
                col,
                rows_chunk * projection_matrix.projection_ratio * (PROJECTION_HEIGHT / DEGREE),
                (rows_chunk + 1)
                    * projection_matrix.projection_ratio
                    * (PROJECTION_HEIGHT / DEGREE),
            );
            
            // Get mutable slice of the output for this chunk
            let projection_subimage = image_ct.col_slice_mut(
                col,
                rows_chunk * (PROJECTION_HEIGHT / DEGREE),
                (rows_chunk + 1) * (PROJECTION_HEIGHT / DEGREE),
            );
            
            // Apply projection matrix J to this chunk
            // J has PROJECTION_HEIGHT rows (output coefficients)
            for inner_row in 0..PROJECTION_HEIGHT {
                // Map this output coefficient to its position in a ring element
                let current_projection_row = inner_row / DEGREE;        // Which ring element
                let current_projection_coeff_index = inner_row % DEGREE; // Which coeff in that element
                
                // Compute the inner product: projection_subimage[inner_row] = J[inner_row, :] · subwitness
                // J has (projection_ratio * PROJECTION_HEIGHT) columns
                for i in 0..projection_matrix.projection_ratio * PROJECTION_HEIGHT {
                    let (is_positive, is_non_zero) = &projection_matrix[(inner_row, i)];
                    if !*is_non_zero {
                        continue;
                    }
                    
                    // Add or subtract the witness coefficient depending on J's sign
                    if *is_positive {
                        unsafe {
                            // TODO: set it first to u64::MAX / 2 and perform addition/sub without mod. Make mod at the end once
                            projection_subimage[current_projection_row].v
                                [current_projection_coeff_index] = add_mod(
                                projection_subimage[current_projection_row].v
                                    [current_projection_coeff_index],
                                subwitness[i / DEGREE].v[i % DEGREE],
                                MOD_Q,
                            );
                        }
                    } else {
                        unsafe {
                            projection_subimage[current_projection_row].v
                                [current_projection_coeff_index] = sub_mod(
                                projection_subimage[current_projection_row].v
                                    [current_projection_coeff_index],
                                subwitness[i / DEGREE].v[i % DEGREE],
                                MOD_Q,
                            );
                        }
                    }
                }
            }
        }
    }
    image_ct
}

// Compute batched projection: c'^t * vec(V) where V = (I_d ⊗ J) * coeff(W)
//
// Instead of computing the full projection V and then batching, we perform the batching
// during the projection computation using a tensor product decomposition.
//
// Mathematical structure:
// - W ∈ R_q^{m × r} is the witness matrix
// - J ∈ {-1,0,1}^{n_rp × m_rp} is the projection matrix (with n_rp = PROJECTION_HEIGHT)
// - V = (I_d ⊗ J) * coeff(W) ∈ Z^{d·n_rp × r} is the projected result in coefficient form
// - c' ∈ Z^{d·n_rp·r} is the batching challenge, decomposed as c' = c'_0 ⊗ c'_1 ⊗ c'_2 where:
//   * c'_0 ∈ Z^r (batches over witness columns)
//   * c'_1 ∈ Z^d (batches over coefficient blocks in the image)
//   * c'_2 ∈ Z^{n_rp} (batches over PROJECTION_HEIGHT coefficients)
//   * d = (witness.height / projection_ratio) * DEGREE / PROJECTION_HEIGHT
//
// Algorithm:
// 1. W_batched = W * c'_0 (batch witness columns)
// 2. J_batched = c'_2^T * J_embedded (batch projection matrix rows with dual embedding)
// 3. result = c'_1^T ⊗ J_batched · W_batched (tensor and inner product)
//
// Each c'_i is structured as a tensor product: c'_i = ⊗ (1 - l_j, l_j)
// This allows us to sample only log(|c'_i|) random values (the layers l_j)
// and compute any c'_i[k] on-the-fly using the binary representation of k.
fn batch_projection(
    witness: &VerticallyAlignedMatrix<RingElement>,
    projection_matrix: &ProjectionMatrix,
    hash_wrapper: &mut HashWrapper,
) -> RingElement {

    // Sample structured challenge layers
    // c'_0 is over the witness columns (r)
    // c'_1 is over d (number of blocks in image_ct when viewed coefficient-wise)
    // c'_2 is over n_rp (projection height coefficients)
    
    let c_0_layers: Vec<u64> = (0..witness.width.ilog2())
        .map(|_| hash_wrapper.sample_u64_mod_q())
        .collect();
    
    // d = image_ct.height * DEGREE / PROJECTION_HEIGHT
    let d = (witness.height / projection_matrix.projection_ratio) * DEGREE / PROJECTION_HEIGHT;
    let c_1_layers: Vec<u64> = (0..d.ilog2())
        .map(|_| hash_wrapper.sample_u64_mod_q())
        .collect();
    
    let c_2_layers: Vec<u64> = (0..PROJECTION_HEIGHT.ilog2())
        .map(|_| hash_wrapper.sample_u64_mod_q())
        .collect();

    // Helper function to compute structured row value from layers
    let compute_structured_value = |layers: &[u64], index: usize| -> u64 {
        let mut result = 1u64;
        for (bit_pos, &layer) in layers.iter().enumerate() {
            if (index >> bit_pos) & 1 == 1 {
                unsafe {
                    result = multiply_mod(result, layer, MOD_Q);
                }
            } else {
                unsafe {
                    result = multiply_mod(result, sub_mod(1, layer, MOD_Q), MOD_Q);
                }
            }
        }
        result
    };

    // ===== Step 1: Batch witness columns =====
    // Compute W_batched = W * c'_0
    // This reduces the witness from shape (witness.height, witness.width) to (witness.height, 1)
    // by taking a linear combination of columns weighted by c'_0
    let mut w_batched = VerticallyAlignedMatrix::new_zero_preallocated(witness.height, 1);
    
    // for el in w_batched.data.iter_mut() {
    //     el.representation = Representation::Coefficients;
    // }
    
    for row in 0..witness.height {
        for col in 0..witness.width {
            let coeff = compute_structured_value(&c_0_layers, col);
            for deg in 0..DEGREE {
                unsafe {
                    let temp = multiply_mod(
                        witness[(row, col)].v[deg],
                        coeff,
                        MOD_Q,
                    );
                    w_batched[(row, 0)].v[deg] = add_mod(
                        w_batched[(row, 0)].v[deg],
                        temp,
                        MOD_Q,
                    );
                }
            }
        }
    }

    // ===== Step 2: Batch projection matrix with dual embedding =====
    // Compute J_batched = c'_2^T * J_embedded
    // J_embedded applies dual embedding: each coefficient j ∈ {-1,0,1} becomes a polynomial
    // where the constant term is j and non-constant terms are -j (to maintain inner product)
    // J_batched will be a vector of inner_width_ring ring elements
    let inner_width_ring = projection_matrix.projection_ratio * (PROJECTION_HEIGHT / DEGREE);
    let mut j_batched = new_vec_zero_preallocated(inner_width_ring);
    
    for el in j_batched.iter_mut() {
        el.representation = Representation::Coefficients;
    }
    
    // Iterate over each ring element in J_batched
    for i in 0..inner_width_ring {
        // For each coefficient position in the ring element
        for j in 0..DEGREE {
            let col_index = i * DEGREE + j;  // Flatten to coefficient space
            // Accumulate weighted sum over PROJECTION_HEIGHT rows using c'_2
            for k in 0..PROJECTION_HEIGHT {
                let coeff = compute_structured_value(&c_2_layers, k);
                unsafe {
                    let (is_positive, is_non_zero) = &projection_matrix[(k, col_index)];
                    if !*is_non_zero {
                        continue;
                    }
                    // Dual embedding: constant term (j=0) keeps sign, non-constant terms flip sign
                    // This ensures <embed_dual(a), embed(b)> = a * b for scalars a, b
                    let deg_idx = if j == 0 { 0 } else { DEGREE - j };
                    if *is_positive {
                        if j == 0 {
                            j_batched[i].v[0] = add_mod(j_batched[i].v[0], coeff, MOD_Q);
                        } else {
                            j_batched[i].v[deg_idx] = sub_mod(j_batched[i].v[deg_idx], coeff, MOD_Q);
                        }
                    } else {
                        if j == 0 {
                            j_batched[i].v[0] = sub_mod(j_batched[i].v[0], coeff, MOD_Q);
                        } else {
                            j_batched[i].v[deg_idx] = add_mod(j_batched[i].v[deg_idx], coeff, MOD_Q);
                        }
                    }
                }
            }
        }
    }

    // Convert j_batched to NTT for efficient multiplication
    for bp in j_batched.iter_mut() {
        bp.to_representation(Representation::IncompleteNTT);
    }
 
    // ===== Step 3: Apply c'_1 batching and compute final inner product =====
    // The witness (after batching by c'_0) can be split into num_chunks chunks,
    // where each chunk corresponds to one application of the projection matrix.
    // We compute: result = Σ_chunk c'_1[chunk] * <J_batched, W_batched[chunk]>
    let num_chunks = witness.height / inner_width_ring;
    let mut result = RingElement::zero(Representation::IncompleteNTT);
    
    for chunk in 0..num_chunks {
        let c_1_coeff = compute_structured_value(&c_1_layers, chunk);
        let mut chunk_result = RingElement::zero(Representation::IncompleteNTT);
        
        // Inner product of j_batched with the corresponding chunk of w_batched
        for i in 0..inner_width_ring {
            let mut temp = RingElement::zero(Representation::IncompleteNTT);
            temp *= (&w_batched[(chunk * inner_width_ring + i, 0)], &j_batched[i]);
            chunk_result += &temp;
        }
        
        // Multiply by c_1 coefficient and accumulate
        for deg in 0..DEGREE {
            unsafe {
                let temp = multiply_mod(chunk_result.v[deg], c_1_coeff, MOD_Q);
                result.v[deg] = add_mod(result.v[deg], temp, MOD_Q);
            }
        }
    }

    // result.to_representation(Representation::Coefficients);
    result
}

  
  

#[test]
fn test_batch_projection() {
    // Test that batch_projection correctly computes c'^t * vec(V)
    // where V = project_coefficients(witness)
    //
    // This verifies the correctness of the batched projection algorithm by:
    // 1. Computing V explicitly via project_coefficients
    // 2. Sampling the same random challenges used in batch_projection
    // 3. Manually computing c'^t * vec(V) with the correct flattening structure
    // 4. Comparing the constant term with batch_projection's result
    
    let mut witness = VerticallyAlignedMatrix {
        data: vec![RingElement::random(Representation::IncompleteNTT); 16],
        width: 2,
        height: 8,
    };

    let mut projection_matrix = ProjectionMatrix::new(2);
    let mut hash_wrapper = HashWrapper::new();
    projection_matrix.sample(&mut hash_wrapper);

    // Compute the full projection V = (I_d ⊗ J) * coeff(W)
    let image_ct = project_coefficients(&witness, &projection_matrix);

    // Sample the same structured challenges as batch_projection
    // Each challenge is structured as a tensor product, so we only sample the "layers"
    // c'_0: batches over r = witness.width columns
    let c_0_layers: Vec<u64> = (0..witness.width.ilog2())
        .map(|_| hash_wrapper.sample_u64_mod_q())
        .collect();
    
    // c'_1: batches over d coefficient blocks in the image
    // d = (number of ring elements in image) * (coefficients per ring element) / (block size)
    let d = image_ct.height * DEGREE / PROJECTION_HEIGHT;
    let c_1_layers: Vec<u64> = (0..d.ilog2())
        .map(|_| hash_wrapper.sample_u64_mod_q())
        .collect();
    
    // c'_2: batches over PROJECTION_HEIGHT coefficients within each block
    let c_2_layers: Vec<u64> = (0..PROJECTION_HEIGHT.ilog2())
        .map(|_| hash_wrapper.sample_u64_mod_q())
        .collect();

    // Helper function to compute structured row value from layers
    let compute_structured_value = |layers: &[u64], index: usize| -> u64 {
        let mut result = 1u64;
        for (bit_pos, &layer) in layers.iter().enumerate() {
            if (index >> bit_pos) & 1 == 1 {
                unsafe {
                    result = multiply_mod(result, layer, MOD_Q);
                }
            } else {
                unsafe {
                    result = multiply_mod(result, sub_mod(1, layer, MOD_Q), MOD_Q);
                }
            }
        }
        result
    };

    // Manually compute c'^t * vec(V) with the correct flattening structure
    // Flattening order: (column, chunk, coefficient_in_chunk)
    // where each chunk has PROJECTION_HEIGHT coefficients spread across ring elements
    let mut expected_ct = 0u64;
    let inner_width_ring = projection_matrix.projection_ratio * (PROJECTION_HEIGHT / DEGREE);
    let num_chunks_in_image = image_ct.height / (PROJECTION_HEIGHT / DEGREE);
    
    // Iterate in the same order as the tensor product decomposition
    for col in 0..image_ct.width {
        let c_0_coeff = compute_structured_value(&c_0_layers, col);
        for chunk_idx in 0..num_chunks_in_image {
            let c_1_coeff = compute_structured_value(&c_1_layers, chunk_idx);
            // Each chunk contains PROJECTION_HEIGHT coefficients
            for coeff_idx in 0..PROJECTION_HEIGHT {
                let c_2_coeff = compute_structured_value(&c_2_layers, coeff_idx);
                // Map flat coefficient index to (row, degree) in the ring element matrix
                let row_in_chunk = coeff_idx / DEGREE;  // Which ring element in this chunk
                let deg = coeff_idx % DEGREE;           // Which coefficient in that ring element
                let row = chunk_idx * (PROJECTION_HEIGHT / DEGREE) + row_in_chunk;
                unsafe {
                    // The challenge at this position is c'_0[col] * c'_1[chunk] * c'_2[coeff]
                    let c_combined = multiply_mod(c_0_coeff, c_1_coeff, MOD_Q);
                    let c_combined = multiply_mod(c_combined, c_2_coeff, MOD_Q);
                    
                    let temp = multiply_mod(
                        image_ct[(row, col)].v[deg],
                        c_combined,
                        MOD_Q,
                    );
                    expected_ct = add_mod(expected_ct, temp, MOD_Q);
                }
            }
        }
    }

    // Create a fresh hash_wrapper to sample the same challenges in batch_projection
    let mut hash_wrapper2 = HashWrapper::new();
    projection_matrix.sample(&mut hash_wrapper2); // Consume same randomness to sync state
    
    let mut result = batch_projection(&witness, &projection_matrix, &mut hash_wrapper2);
    result.to_representation(Representation::Coefficients);
    
    // The result is a full ring element, but we only check the constant term
    // since that's what the batching computes (the higher degree terms depend on NTT operations)
    assert_eq!(result.v[0], expected_ct, "Constant term of batch_projection should equal c'^t * vec(V)");
}
