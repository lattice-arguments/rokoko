use crate::common::config::*;
use crate::common::norms::l2_norm_coeffs;
use crate::protocol::config::{RoundConfig, SalsaaProof};
use crate::protocol::project::BatchingChallenges;
use crate::protocol::project_2::verifier_sample_projection_challenges;
use crate::protocol::sumchecks::context_verifier::VerifierSumcheckContext;
use crate::{
    common::{
        arithmetic::{field_to_ring_element_into, precompute_structured_values_fast, ONE},
        config::{DEGREE, HALF_DEGREE, MOD_Q, NOF_BATCHES},
        decomposition::compose_from_decomposed,
        hash::HashWrapper,
        matrix::{new_vec_zero_preallocated, HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
        sumcheck_element::SumcheckElement,
    },
    hexl::bindings::{add_mod, eltwise_mult_mod, multiply_mod},
    protocol::{
        commitment::{commit_basic, BasicCommitment},
        crs::CRS,
        open::evaluation_point_to_structured_row,
        project_2::BatchedProjectionChallengesSuccinct,
        vdf::{compute_ip_vdf_claim, VDFCrs},
    },
};
use std::array;

/// Computes the batched claim from individual sumcheck claims.
/// Type1 sumchecks are product sumchecks with claims = <evaluation_outer, column_claims>,
/// Type3 is a diff sumcheck with claim = 0.
fn batch_claims(
    config: &RoundConfig,
    claims: &HorizontallyAlignedMatrix<RingElement>,
    evaluation_points_outer: &[RingElement],
    ip_l2_claim: Option<&RingElement>,
    ip_linf_claim: Option<&RingElement>,
    ip_vdf_claim: Option<&RingElement>,
    type31_claims: &[RingElement],
    combination: &[RingElement],
) -> RingElement {
    let mut batched_claim = RingElement::zero(Representation::IncompleteNTT);
    let mut idx = 0;

    // Type1 sumchecks: claim = <evaluation_outer, column_claims[i]>
    for i in 0..config.inner_evaluation_claims {
        let mut type1_claim = RingElement::zero(Representation::IncompleteNTT);
        for (c, r) in claims.row(i).iter().zip(evaluation_points_outer.iter()) {
            type1_claim += &(c * r);
        }
        let mut weighted = type1_claim;
        weighted *= &combination[idx];
        batched_claim += &weighted;
        idx += 1;
    }

    if let RoundConfig::Intermediate { .. } = config {
        // zero claim, nothing to add
        idx += 1;
    }

    // L2: product sumcheck over conjugated witness and selected witness.
    if config.l2 {
        let mut weighted = ip_l2_claim
            .expect("Missing l2 claim in proof while l2 constraint is enabled")
            .clone();
        weighted *= &combination[idx];
        batched_claim += &weighted;
        idx += 1;
    }

    // Linf: exact-binariness sumcheck claim.
    if config.exact_binariness {
        let mut weighted = ip_linf_claim
            .expect("Missing linf claim in proof while exact_binariness is enabled")
            .clone();
        weighted *= &combination[idx];
        batched_claim += &weighted;
        idx += 1;
    }

    // VDF: product sumcheck claim = -y_0 + c^{2K} · y_t
    if config.vdf {
        let mut weighted = ip_vdf_claim
            .expect("Missing vdf claim in proof while vdf is enabled")
            .clone();
        weighted *= &combination[idx];
        batched_claim += &weighted;
        idx += 1;
    }

    // Type31: projection sumcheck claims for IntermediateUnstructured
    for type31_claim in type31_claims {
        let mut weighted = type31_claim.clone();
        weighted *= &combination[idx];
        batched_claim += &weighted;
        idx += 1;
    }

    assert_eq!(
        idx,
        combination.len(),
        "batch_claims: index mismatch with combination length"
    );
    batched_claim
}

pub fn verifier_round(
    config: &RoundConfig,
    crs: &CRS,
    verifier_context: &mut VerifierSumcheckContext,
    commitment: &BasicCommitment,
    proof: &SalsaaProof,
    evaluation_points_inner: &[StructuredRow],
    claims: &HorizontallyAlignedMatrix<RingElement>,
    hash_wrapper: &mut HashWrapper,
    vdf_crs_param: Option<&VDFCrs>,
    vdf_outputs: Option<(
        &[RingElement; VDF_MATRIX_HEIGHT],
        &[RingElement; VDF_MATRIX_HEIGHT],
    )>, // (y_0, y_t) - only for first round
    round_index: usize,
) {
    let round_start = std::time::Instant::now();
    let projection_matrix = match config {
        RoundConfig::Intermediate { .. } => {
            let mut pm = ProjectionMatrix::new(config.main_witness_columns, PROJECTION_HEIGHT);
            pm.sample(hash_wrapper);
            Some(pm)
        }
        _ => None,
    };

    let batching_challenges = match config {
        RoundConfig::Intermediate { .. } => Some(BatchingChallenges::sample(config, hash_wrapper)),
        _ => None,
    };

    let projection_challenges_unstructured = match config {
        RoundConfig::IntermediateUnstructured {
            projection_ratio, ..
        }
        | RoundConfig::Last {
            projection_ratio, ..
        } => {
            let mut projection_matrix = ProjectionMatrix::new(*projection_ratio, PROJECTION_HEIGHT);
            projection_matrix.sample(hash_wrapper);
            let challenges: [BatchedProjectionChallengesSuccinct; NOF_BATCHES] =
                array::from_fn(|_| {
                    verifier_sample_projection_challenges(&projection_matrix, config, hash_wrapper)
                });
            Some(challenges)
        }
        _ => None,
    };

    // Projection image CT consistency check (for rounds with unstructured projection)
    {
        let (projection_image_ct, projection_image_batched) = match proof {
            SalsaaProof::IntermediateUnstructured {
                projection_image_ct,
                projection_image_batched,
                ..
            }
            | SalsaaProof::Last {
                projection_image_ct,
                projection_image_batched,
                ..
            } => (Some(projection_image_ct), Some(projection_image_batched)),
            _ => (None, None),
        };

        if let (Some(projection_image_ct), Some(projection_image_batched)) =
            (projection_image_ct, projection_image_batched)
        {
            let l2_norm_proj = l2_norm_coeffs(&projection_image_ct.data);
            println!("L2 norm of projection image: {}", l2_norm_proj);

            let challenges = projection_challenges_unstructured
                .as_ref()
                .expect("Missing projection challenges for CT consistency check");
            let mut temp = RingElement::zero(Representation::IncompleteNTT);
            for i in 0..NOF_BATCHES {
                let c_0_values = precompute_structured_values_fast(&challenges[i].c_0_layers);
                let c_1_values = precompute_structured_values_fast(&challenges[i].c_1_layers);
                debug_assert_eq!(
                    c_1_values.len() % DEGREE,
                    0,
                    "c_1_values must be divisible by DEGREE"
                );
                let rows_per_chunk = c_1_values.len() / DEGREE;
                debug_assert!(rows_per_chunk > 0, "rows_per_chunk must be non-zero");
                debug_assert_eq!(
                    projection_image_ct.height % rows_per_chunk,
                    0,
                    "projection_image_ct height must be divisible by rows_per_chunk"
                );
                let num_chunks = projection_image_ct.height / rows_per_chunk;
                debug_assert_eq!(c_0_values.len(), num_chunks, "c_0_values length mismatch");

                for k in 0..projection_image_batched.width {
                    let mut expected_ct = 0u64;
                    for chunk_idx in 0..num_chunks {
                        let mut chunk_ct = 0u64;
                        let chunk_row_base = chunk_idx * rows_per_chunk;
                        for row_in_chunk in 0..rows_per_chunk {
                            let row = chunk_row_base + row_in_chunk;
                            unsafe {
                                eltwise_mult_mod(
                                    temp.v.as_mut_ptr(),
                                    c_1_values.as_ptr().add(DEGREE * row_in_chunk),
                                    projection_image_ct[(row, k)].v.as_ptr(),
                                    DEGREE as u64,
                                    MOD_Q,
                                );
                            }
                            for l in 0..DEGREE {
                                chunk_ct += temp.v[l];
                            }
                        }
                        chunk_ct %= MOD_Q;
                        unsafe {
                            expected_ct = add_mod(
                                expected_ct,
                                multiply_mod(chunk_ct, c_0_values[chunk_idx], MOD_Q),
                                MOD_Q,
                            );
                        }
                    }
                    let ct = projection_image_batched[(i, k)].constant_term_from_incomplete_ntt();
                    assert_eq!(
                        ct, expected_ct,
                        "Projection constant term consistency check failed at batch {}, column {}",
                        i, k
                    );
                }
            }
            println!("Projection image consistency ct check passed");
        }
    }

    let vdf_challenge = if config.vdf {
        let mut challenge = RingElement::zero(Representation::IncompleteNTT);
        hash_wrapper.sample_ring_element_ntt_slots_into(&mut challenge);
        Some(challenge)
    } else {
        None
    };

    if config.l2 {
        let claim: &RingElement = proof
            .ip_l2_claim
            .as_ref()
            .expect("Missing l2 claim in proof while l2 constraint is enabled");
        let ct = claim.constant_term_from_incomplete_ntt();
        println!("asserted norm is sqrt({})", ct);
    }

    if config.exact_binariness {
        let claim: &RingElement = proof
            .ip_linf_claim
            .as_ref()
            .expect("Missing linf claim in proof while exact_binariness is enabled");
        let ct = claim.constant_term_from_incomplete_ntt();
        if ct != 0 {
            println!(
                "Binariness verification failed: constant term is not zero, got {}",
                ct
            );
        } else {
            println!("Binariness verification passed: constant term is zero");
        }
    }

    let mut evaluation_points_outer = new_vec_zero_preallocated(config.main_witness_columns);
    hash_wrapper.sample_ring_element_vec_into(&mut evaluation_points_outer);

    // Sample random batching coefficients (same Fiat-Shamir as prover)
    let num_sumchecks = verifier_context
        .combiner_evaluation
        .borrow()
        .sumchecks_count();
    let mut combination = new_vec_zero_preallocated(num_sumchecks);
    hash_wrapper.sample_ring_element_vec_into(&mut combination);

    let mut combination_to_field = RingElement::zero(Representation::IncompleteNTT);
    hash_wrapper.sample_ring_element_into(&mut combination_to_field);
    combination_to_field.from_incomplete_ntt_to_homogenized_field_extensions();
    let qe = combination_to_field.split_into_quadratic_extensions();

    // Compute type31 claims for rounds with unstructured projection
    let type31_claims: Vec<RingElement> = match proof {
        SalsaaProof::IntermediateUnstructured {
            projection_image_batched,
            ..
        }
        | SalsaaProof::Last {
            projection_image_batched,
            ..
        } => {
            let challenges = projection_challenges_unstructured
                .as_ref()
                .expect("Missing projection challenges for type31 claims");
            let mut claims_vec = Vec::with_capacity(NOF_BATCHES);
            for batch_idx in 0..NOF_BATCHES {
                let c_2_values =
                    precompute_structured_values_fast(&challenges[batch_idx].c_2_layers);
                let mut claim = RingElement::zero(Representation::IncompleteNTT);
                let mut temp = RingElement::zero(Representation::IncompleteNTT);
                for k in 0..projection_image_batched.width {
                    temp *= (
                        &projection_image_batched[(batch_idx, k)],
                        &RingElement::constant(c_2_values[k], Representation::IncompleteNTT),
                    );
                    claim += &temp;
                }
                claims_vec.push(claim);
            }
            claims_vec
        }
        _ => vec![],
    };

    // Compute expected batched claim over field
    let batched_claim = batch_claims(
        config,
        claims,
        &evaluation_points_outer,
        proof.ip_l2_claim.as_ref(),
        proof.ip_linf_claim.as_ref(),
        compute_ip_vdf_claim(
            config,
            vdf_challenge.as_ref(),
            vdf_outputs.map(|(y_0, y_t)| (y_0, y_t, vdf_crs_param.unwrap())),
        )
        .as_ref(),
        &type31_claims,
        &combination,
    );

    let mut batched_claim_over_field = {
        let batched_claim_field = {
            let mut temp = batched_claim.clone();
            temp.from_incomplete_ntt_to_homogenized_field_extensions();
            temp
        };
        let mut temp = batched_claim_field.split_into_quadratic_extensions();
        let mut result = QuadraticExtension::zero();
        for i in 0..HALF_DEGREE {
            temp[i] *= &qe[i];
            result += &temp[i];
        }
        result
    };

    // Verify each sumcheck round: poly(0) + poly(1) == running_claim
    let mut num_vars = proof.sumcheck_transcript.len();
    let mut evaluation_points_field: Vec<QuadraticExtension> = Vec::new();
    let mut evaluation_points_ring: Vec<RingElement> = Vec::new();

    let mut round_idx = 0;
    while num_vars > 0 {
        num_vars -= 1;
        let poly_over_field = &proof.sumcheck_transcript[round_idx];

        hash_wrapper.update_with_quadratic_extension_slice(&poly_over_field.coefficients);

        assert_eq!(
            poly_over_field.at_zero() + poly_over_field.at_one(),
            batched_claim_over_field,
            "Sumcheck round {}: poly(0) + poly(1) != running claim",
            round_idx,
        );

        let mut f = QuadraticExtension::zero();
        hash_wrapper.sample_field_element_into(&mut f);

        batched_claim_over_field = poly_over_field.at(&f);

        evaluation_points_field.push(f);

        let mut r = RingElement::zero(Representation::IncompleteNTT);
        field_to_ring_element_into(&mut r, &f);
        r.from_homogenized_field_extensions_to_incomplete_ntt();
        evaluation_points_ring.push(r);

        round_idx += 1;
    }

    // verify evaluation claims (TODO: change to recompute them from the proof data)

    // Replay Fiat-Shamir: sample folding challenges (same as prover does post-sumcheck)
    let mut folding_challenges = new_vec_zero_preallocated(config.main_witness_columns);
    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut folding_challenges);

    let outer_points_len =
        config.main_witness_columns.ilog2() as usize + config.main_witness_prefix.length;
    let evaluation_points_ring_tree = evaluation_points_ring
        .iter()
        .rev()
        .cloned()
        .collect::<Vec<_>>();
    let layer = &evaluation_points_ring_tree[outer_points_len];
    let conj_layer = layer.conjugate();

    // Compute the folded claim: sum_i folding_challenges[i] * claims[(0, i)]
    let mut folded_claim = RingElement::zero(Representation::IncompleteNTT);
    for i in 0..config.main_witness_columns {
        let mut term = folding_challenges[i].clone();
        term *= &proof.claims[(0, i)];
        folded_claim += &term;
    }

    let mut folded_conj_claim = RingElement::zero(Representation::IncompleteNTT);
    for i in 0..config.main_witness_columns {
        let mut term = folding_challenges[i].clone();
        term *= &proof.claims[(1, i)];
        folded_conj_claim += &term;
    }

    match (config, proof) {
        (
            RoundConfig::Intermediate {
                decomposition_base_log,
                next,
                ..
            },
            SalsaaProof::Intermediate {
                new_claims,
                decomposed_split_commitment,
                claim_over_projection,
                projection_commitment,
                next: next_proof,
                ..
            },
        ) => {
            let recomposed_claims = HorizontallyAlignedMatrix {
                height: 2,
                width: 4,
                data: compose_from_decomposed(&new_claims.data, *decomposition_base_log, 2),
            };

            assert_eq!(
                folded_claim,
                &(&(&*ONE - layer) * &recomposed_claims[(0, 0)])
                    + &(layer * &recomposed_claims[(0, 1)]),
                "Recomposed claim for the witness does not match the original claim"
            );

            assert_eq!(
                folded_conj_claim,
                &(&(&*ONE - &conj_layer) * &recomposed_claims[(1, 0)])
                    + &(&conj_layer * &recomposed_claims[(1, 1)]),
                "Recomposed conjugate claim for the witness does not match the original claim"
            );

            // Check claims over the projection
            assert_eq!(
                claim_over_projection[0],
                &(&(&*ONE - layer) * &recomposed_claims[(0, 2)])
                    + &(layer * &recomposed_claims[(0, 3)]),
                "Recomposed claim for the projection does not match the original claim"
            );

            assert_eq!(
                claim_over_projection[1],
                &(&(&*ONE - &conj_layer) * &recomposed_claims[(1, 2)])
                    + &(&conj_layer * &recomposed_claims[(1, 3)]),
                "Recomposed conjugate claim for the projection does not match the original claim"
            );

            let recomposed_commitments = HorizontallyAlignedMatrix {
                height: RANK,
                width: 4,
                data: compose_from_decomposed(
                    &decomposed_split_commitment.data,
                    *decomposition_base_log,
                    2,
                ),
            };

            let mut temp = RingElement::zero(Representation::IncompleteNTT);
            for r in 0..RANK {
                let layer = crs.structured_ck_for_wit_dim(
                    config.extended_witness_length / 2 / config.main_witness_columns,
                )[r]
                    .tensor_layers
                    .first()
                    .unwrap();

                let mut folded_commitment_r = RingElement::zero(Representation::IncompleteNTT);
                for i in 0..config.main_witness_columns {
                    temp *= (&folding_challenges[i], &commitment[(r, i)]);
                    folded_commitment_r += &temp;
                }

                assert_eq!(
                    folded_commitment_r,
                    &(&(&*ONE - layer) * &recomposed_commitments[(r, 0)])
                        + &(layer * &recomposed_commitments[(r, 1)]),
                    "Recomposed commitment for the witness does not match the folded commitment"
                );

                assert_eq!(
                    projection_commitment[(r, 0)],
                    &(&(&*ONE - layer) * &recomposed_commitments[(r, 2)])
                        + &(layer * &recomposed_commitments[(r, 3)]),
                    "Recomposed commitment for the projection does not match"
                );
            }

            verifier_context.load_data(
                config,
                proof,
                &evaluation_points_ring_tree,
                evaluation_points_inner,
                &evaluation_points_outer,
                &batching_challenges,
                &projection_matrix,
                &None,
                &combination,
                qe,
                vdf_challenge.as_ref(),
                vdf_crs_param,
            );

            let verifier_eval = *verifier_context
                .field_combiner_evaluation
                .borrow_mut()
                .evaluate_at_ring_point(&evaluation_points_ring);

            assert_eq!(
                verifier_eval, batched_claim_over_field,
                "Verifier final check failed: tree evaluation does not match sumcheck claim"
            );

            // Recurse into the next round
            let new_evaluation_points_inner = evaluation_points_ring_tree
                .iter()
                .skip(outer_points_len + 1)
                .cloned()
                .collect::<Vec<_>>();

            let new_evaluation_points_inner_conjugated = new_evaluation_points_inner
                .iter()
                .map(RingElement::conjugate)
                .collect::<Vec<_>>();

            let next_level_eval_points = vec![
                evaluation_point_to_structured_row(&new_evaluation_points_inner),
                evaluation_point_to_structured_row(&new_evaluation_points_inner_conjugated),
            ];

            println!(
                "Verifier round {} took {:?}",
                round_index,
                round_start.elapsed()
            );

            verifier_round(
                next,
                crs,
                verifier_context.next.as_mut().unwrap(),
                decomposed_split_commitment,
                next_proof,
                &next_level_eval_points,
                new_claims,
                hash_wrapper,
                None, // VDF only in first round
                None, // no VDF outputs in recursive rounds
                round_index + 1,
            );
        }

        (
            RoundConfig::IntermediateUnstructured {
                decomposition_base_log,
                next,
                ..
            },
            SalsaaProof::IntermediateUnstructured {
                new_claims,
                decomposed_split_commitment,
                next: next_proof,
                ..
            },
        ) => {
            // Recompose claims: width=2 (no projection columns)
            let recomposed_claims = HorizontallyAlignedMatrix {
                height: 2,
                width: 2,
                data: compose_from_decomposed(new_claims, *decomposition_base_log, 2),
            };

            assert_eq!(
                folded_claim,
                &(&(&*ONE - layer) * &recomposed_claims[(0, 0)])
                    + &(layer * &recomposed_claims[(0, 1)]),
                "IntermediateUnstructured: recomposed claim does not match the folded claim"
            );

            assert_eq!(
                folded_conj_claim,
                &(&(&*ONE - &conj_layer) * &recomposed_claims[(1, 0)])
                    + &(&conj_layer * &recomposed_claims[(1, 1)]),
                "IntermediateUnstructured: recomposed conjugate claim does not match"
            );

            // Recompose commitments: width=2 (no projection)
            let recomposed_commitments = HorizontallyAlignedMatrix {
                height: RANK,
                width: 2,
                data: compose_from_decomposed(
                    &decomposed_split_commitment.data,
                    *decomposition_base_log,
                    2,
                ),
            };

            let mut temp = RingElement::zero(Representation::IncompleteNTT);
            for r in 0..RANK {
                let layer = crs.structured_ck_for_wit_dim(
                    (config.extended_witness_length >> config.main_witness_prefix.length)
                        / config.main_witness_columns,
                )[r]
                    .tensor_layers
                    .first()
                    .unwrap();

                let mut folded_commitment_r = RingElement::zero(Representation::IncompleteNTT);
                for i in 0..config.main_witness_columns {
                    temp *= (&folding_challenges[i], &commitment[(r, i)]);
                    folded_commitment_r += &temp;
                }

                assert_eq!(
                    folded_commitment_r,
                    &(&(&*ONE - layer) * &recomposed_commitments[(r, 0)])
                        + &(layer * &recomposed_commitments[(r, 1)]),
                    "IntermediateUnstructured: recomposed commitment does not match"
                );
            }

            verifier_context.load_data(
                config,
                proof,
                &evaluation_points_ring_tree,
                evaluation_points_inner,
                &evaluation_points_outer,
                &batching_challenges,
                &projection_matrix,
                &projection_challenges_unstructured,
                &combination,
                qe,
                vdf_challenge.as_ref(),
                vdf_crs_param,
            );

            let verifier_eval = *verifier_context
                .field_combiner_evaluation
                .borrow_mut()
                .evaluate_at_ring_point(&evaluation_points_ring);

            assert_eq!(
                verifier_eval, batched_claim_over_field,
                "IntermediateUnstructured: tree evaluation does not match sumcheck claim"
            );

            // Recurse into the next round
            let new_evaluation_points_inner = evaluation_points_ring_tree
                .iter()
                .skip(outer_points_len + 1)
                .cloned()
                .collect::<Vec<_>>();

            let new_evaluation_points_inner_conjugated = new_evaluation_points_inner
                .iter()
                .map(RingElement::conjugate)
                .collect::<Vec<_>>();

            let next_level_eval_points = vec![
                evaluation_point_to_structured_row(&new_evaluation_points_inner),
                evaluation_point_to_structured_row(&new_evaluation_points_inner_conjugated),
            ];

            let recomposed_new_claims = HorizontallyAlignedMatrix {
                height: 2,
                width: next.main_witness_columns,
                data: new_claims.clone(),
            };

            println!(
                "Verifier round {} (unstructured) took {:?}",
                round_index,
                round_start.elapsed()
            );

            verifier_round(
                next,
                crs,
                verifier_context.next.as_mut().unwrap(),
                decomposed_split_commitment,
                next_proof,
                &next_level_eval_points,
                &recomposed_new_claims,
                hash_wrapper,
                None,
                None,
                round_index + 1,
            );
        }

        (RoundConfig::Last { .. }, SalsaaProof::Last { folded_witness, .. }) => {
            // Last round: verify claims directly from the witness data

            // Reconstruct the folded witness as a VerticallyAlignedMatrix (1 column)
            let folded_witness_matrix = VerticallyAlignedMatrix {
                height: folded_witness.len(),
                width: 1,
                data: folded_witness.clone(),
                used_cols: 1,
            };

            // Reconstruct projected witness (1 column, same height as folded)
            // let projected_witness_matrix = VerticallyAlignedMatrix {
            //     height: projected_witness.len(),
            //     width: 1,
            //     data: projected_witness.clone(),
            //     used_cols: 1,
            // };

            // Use the current round's sumcheck evaluation points, including the "layer" variable
            // (no +1 skip since there's no split at the last round).
            // The prover computes claims using evaluation_points[outer_points_len..] from THIS round's sumcheck.
            let current_inner_points: Vec<_> = evaluation_points_ring_tree
                .iter()
                .skip(outer_points_len)
                .cloned()
                .collect();

            let eval_points_inner_expanded = PreprocessedRow::from_structured_row(
                &evaluation_point_to_structured_row(&current_inner_points),
            );

            let current_inner_points_conjugated: Vec<_> = current_inner_points
                .iter()
                .map(RingElement::conjugate)
                .collect();

            let eval_points_inner_conj_expanded = PreprocessedRow::from_structured_row(
                &evaluation_point_to_structured_row(&current_inner_points_conjugated),
            );

            // Compute expected claim over folded witness: <eval_points, folded_witness>
            let mut temp = RingElement::zero(Representation::IncompleteNTT);
            let mut expected_folded_claim = RingElement::zero(Representation::IncompleteNTT);
            for (w, r) in folded_witness
                .iter()
                .zip(eval_points_inner_expanded.preprocessed_row.iter())
            {
                temp *= (w, r);
                expected_folded_claim += &temp;
            }

            assert_eq!(
                folded_claim, expected_folded_claim,
                "Last round: folded claim does not match evaluation of the folded witness"
            );

            // Compute expected conjugate claim over folded witness
            let mut expected_folded_conj_claim = RingElement::zero(Representation::IncompleteNTT);
            for (w, r) in folded_witness
                .iter()
                .zip(eval_points_inner_conj_expanded.preprocessed_row.iter())
            {
                temp *= (w, r);
                expected_folded_conj_claim += &temp;
            }

            assert_eq!(
                folded_conj_claim, expected_folded_conj_claim,
                "Last round: folded conjugate claim does not match evaluation of the folded witness"
            );

            // // Compute expected claim over projected witness
            // let mut expected_projection_claim = RingElement::zero(Representation::IncompleteNTT);
            // for (w, r) in projected_witness.iter().zip(eval_points_inner_expanded.preprocessed_row.iter()) {
            //     temp *= (w, r);
            //     expected_projection_claim += &temp;
            // }

            // assert_eq!(
            //     proof.claim_over_projection[0], expected_projection_claim,
            //     "Last round: projection claim does not match evaluation of the projected witness"
            // );

            // // Compute expected conjugate claim over projected witness
            // let mut expected_projection_conj_claim = RingElement::zero(Representation::IncompleteNTT);
            // for (w, r) in projected_witness.iter().zip(eval_points_inner_conj_expanded.preprocessed_row.iter()) {
            //     temp *= (w, r);
            //     expected_projection_conj_claim += &temp;
            // }

            // assert_eq!(
            //     proof.claim_over_projection[1], expected_projection_conj_claim,
            //     "Last round: conjugate projection claim does not match evaluation of the projected witness"
            // );

            let comm_time = std::time::Instant::now();

            // Verify commitment: commit(folded_witness) should match folded commitments
            let folded_witness_commitment = commit_basic(crs, &folded_witness_matrix, RANK);
            // let projected_witness_commitment = commit_basic(crs, &projected_witness_matrix, RANK);

            let elapsed = comm_time.elapsed();
            println!(
                "Verifier commitment recomputation took {} µs",
                elapsed.as_micros()
            );

            for r in 0..RANK {
                let mut folded_commitment_r = RingElement::zero(Representation::IncompleteNTT);
                for i in 0..config.main_witness_columns {
                    temp *= (&folding_challenges[i], &commitment[(r, i)]);
                    folded_commitment_r += &temp;
                }

                assert_eq!(
                    folded_commitment_r,
                    folded_witness_commitment[(r, 0)],
                    "Last round: folded witness commitment does not match"
                );

                // assert_eq!(
                //     proof.projection_commitment[(r, 0)], projected_witness_commitment[(r, 0)],
                //     "Last round: projected witness commitment does not match"
                // );
            }

            verifier_context.load_data(
                config,
                proof,
                &evaluation_points_ring_tree,
                evaluation_points_inner,
                &evaluation_points_outer,
                &batching_challenges,
                &projection_matrix,
                &projection_challenges_unstructured,
                &combination,
                qe,
                vdf_challenge.as_ref(),
                vdf_crs_param,
            );

            let verifier_eval = *verifier_context
                .field_combiner_evaluation
                .borrow_mut()
                .evaluate_at_ring_point(&evaluation_points_ring);

            assert_eq!(
                verifier_eval, batched_claim_over_field,
                "Verifier final check failed: tree evaluation does not match sumcheck claim"
            );

            println!(
                "Verifier round {} (last) took {:?}",
                round_index,
                round_start.elapsed()
            );
            // No recursion at the last round
        }

        _ => panic!("Config and proof variant mismatch"),
    }
}
