use std::array;

use crate::{
    common::{
        arithmetic::precompute_structured_values_fast,
        config::{DEGREE, MOD_Q, NOF_BATCHES},
        hash::HashWrapper,
        matrix::{new_vec_zero_preallocated, HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        norms::l2_norm_coeffs,
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    hexl::bindings::eltwise_mult_mod,
    protocol::{
        commitment::{commit_basic, BasicCommitment},
        config::{
            Config, NextRoundCommitment, RoundProof, SimpleConfig, SimpleRoundProof,
            SumcheckConfig, SumcheckRoundProof,
        },
        crs::CRS,
        open::{
            evaluation_point_to_structured_row, evaluation_point_to_structured_row_conjugate,
            open_at,
        },
        project_2::{verifier_sample_projection_challenges, BatchedProjectionChallengesSuccinct},
        sumchecks::{
            context_verifier::VerifierSumcheckContext, runner_verifier::sumcheck_verifier,
        },
    },
};

pub fn verifier_round(
    crs: &CRS,
    config: &SumcheckConfig,
    rc_commitment: &[RingElement],
    round_proof: &SumcheckRoundProof,
    evaluation_points_inner: &[StructuredRow],
    evaluation_points_outer: &[StructuredRow],
    claims: &[RingElement],
    sumcheck_context_verifier: &mut VerifierSumcheckContext,
    hash_wrapper_verifier: Option<HashWrapper>,
) {
    let start = std::time::Instant::now();
    let mut hash_wrapper_verifier = hash_wrapper_verifier.unwrap_or_else(HashWrapper::new);

    let evaluation_points = sumcheck_verifier(
        &config,
        sumcheck_context_verifier,
        &rc_commitment,
        &round_proof,
        &evaluation_points_inner,
        &evaluation_points_outer,
        &claims,
        &mut hash_wrapper_verifier,
    );

    let elapsed = start.elapsed().as_nanos();
    println!("Verifier: {} ns", elapsed);

    match &round_proof.next {
        Some(next_round_proof) => {
            let next_round_commitment =
                round_proof
                    .next_round_commitment
                    .as_ref()
                    .unwrap_or_else(|| {
                        panic!(
                        "Next round commitment must be present when next round proof is present."
                    )
                    });

            match next_round_proof.as_ref() {
                RoundProof::Sumcheck(next_sumcheck_round_proof) => {
                    let next_sumcheck_config = match &config.next {
                        Some(next_config) => match next_config.as_ref() {
                            Config::Sumcheck(next_sumcheck_config) => next_sumcheck_config,
                            _ => panic!("Expected sumcheck config for next round."),
                        },
                        None => panic!("Next sumcheck config must be present."),
                    };

                    let (new_evaluation_points_outer, new_evaluation_points_inner) =
                        evaluation_points
                            .split_at(next_sumcheck_config.witness_width.ilog2() as usize);

                    let next_round_commiments_recursive = match &next_round_commitment {
                        NextRoundCommitment::Recursive(rc) => rc,
                        _ => panic!("Expected recursive commitment for next round."),
                    };

                    let inner_rows = [
                        evaluation_point_to_structured_row(new_evaluation_points_inner),
                        evaluation_point_to_structured_row_conjugate(new_evaluation_points_inner),
                    ];
                    let outer_rows = [
                        evaluation_point_to_structured_row(new_evaluation_points_outer),
                        evaluation_point_to_structured_row_conjugate(new_evaluation_points_outer),
                    ];
                    let new_claims = [
                        round_proof.claim_over_witness.clone(),
                        round_proof.claim_over_witness_conjugate.conjugate(),
                    ];

                    verifier_round(
                        crs,
                        &next_sumcheck_config,
                        next_round_commiments_recursive.as_slice(),
                        next_sumcheck_round_proof,
                        &inner_rows,
                        &outer_rows,
                        &new_claims,
                        sumcheck_context_verifier.next.as_mut().unwrap(),
                        Some(hash_wrapper_verifier),
                    );
                }

                RoundProof::Simple(next_simple_round_proof) => {
                    let next_simple_config = match &config.next {
                        Some(next_config) => match next_config.as_ref() {
                            Config::Simple(next_simple_config) => next_simple_config,
                            _ => panic!("Expected simple config for next round."),
                        },
                        None => panic!("Next simple config must be present."),
                    };

                    let (new_evaluation_points_outer, new_evaluation_points_inner) =
                        evaluation_points
                            .split_at(next_simple_config.witness_width.ilog2() as usize);

                    let commitment = match &next_round_commitment {
                        NextRoundCommitment::Simple(basic_commitment) => basic_commitment,
                        _ => panic!("Expected simple commitment for next round."),
                    };

                    let inner_rows = [
                        evaluation_point_to_structured_row(new_evaluation_points_inner),
                        evaluation_point_to_structured_row_conjugate(new_evaluation_points_inner),
                    ];
                    let outer_rows = [
                        evaluation_point_to_structured_row(new_evaluation_points_outer),
                        evaluation_point_to_structured_row_conjugate(new_evaluation_points_outer),
                    ];
                    let new_claims = [
                        round_proof.claim_over_witness.clone(),
                        round_proof.claim_over_witness_conjugate.conjugate(),
                    ];

                    verifier_round_simple(
                        crs,
                        next_simple_config,
                        commitment,
                        next_simple_round_proof,
                        &inner_rows,
                        &outer_rows,
                        &new_claims,
                        Some(hash_wrapper_verifier),
                    );
                }
            }
        }
        None => {}
    }
}

pub fn verifier_round_simple(
    crs: &CRS,
    config: &SimpleConfig,
    commitment: &BasicCommitment,
    round_proof: &SimpleRoundProof,
    evaluation_points_inner: &[StructuredRow],
    evaluation_points_outer: &[StructuredRow],
    claims: &[RingElement],
    hash_wrapper: Option<HashWrapper>,
) {
    let start = std::time::Instant::now();
    let mut hash_wrapper = hash_wrapper.unwrap_or_else(HashWrapper::new);
    hash_wrapper.update_with_ring_element_slice(&commitment.data);
    hash_wrapper.update_with_ring_element_slice(&round_proof.opening_rhs.data);

    let mut projection_matrix =
        ProjectionMatrix::new(config.projection_ratio, config.projection_height);

    projection_matrix.sample(&mut hash_wrapper);

    hash_wrapper.update_with_ring_element_slice(&round_proof.projection_image_ct.data);

    let challenges: [BatchedProjectionChallengesSuccinct; NOF_BATCHES] = array::from_fn(|_| {
        verifier_sample_projection_challenges(&projection_matrix, config, &mut hash_wrapper)
    });

    debug_assert_eq!(
        challenges[0].c_0_layers.len(),
        0,
        "In simple verifier, projection challenges c_0_layers length must be zero. At least, this is unimplemented otherwise."
    );

    hash_wrapper.update_with_ring_element_slice(&round_proof.batched_projection_image.data);

    let mut folding_challenges =
        vec![RingElement::zero(Representation::IncompleteNTT); config.witness_width];

    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut folding_challenges);

    let commitment_of_folded_witness = commit_basic(
        &crs,
        &round_proof.folded_witness,
        config.basic_commitment_rank,
    );

    let mut folded_commitment =
        HorizontallyAlignedMatrix::new_zero_preallocated(config.basic_commitment_rank, 1);

    let mut temp = RingElement::zero(Representation::IncompleteNTT);

    for i in 0..config.basic_commitment_rank {
        for col in 0..commitment.width {
            temp *= (&commitment[(i, col)], &folding_challenges[col]);
            folded_commitment[(i, 0)] += &temp;
        }
    }

    for i in 0..config.basic_commitment_rank {
        assert_eq!(
            commitment_of_folded_witness[(i, 0)],
            folded_commitment[(i, 0)]
        );
    }

    let opening_to_folded_witness = open_at(
        &round_proof.folded_witness,
        evaluation_points_inner,
        evaluation_points_outer,
    );

    let mut folded_opening =
        HorizontallyAlignedMatrix::new_zero_preallocated(round_proof.opening_rhs.height, 1);

    for i in 0..round_proof.opening_rhs.height {
        for col in 0..commitment.width {
            temp *= (&round_proof.opening_rhs[(i, col)], &folding_challenges[col]);
            folded_opening[(i, 0)] += &temp;
        }
    }

    assert_eq!(opening_to_folded_witness.rhs, folded_opening);

    let mut batched_projection_of_folded_witness = VerticallyAlignedMatrix::new_zero_preallocated(
        round_proof.batched_projection_image.height,
        1,
    );

    for i in 0..round_proof.batched_projection_image.height {
        let j_batched = &challenges[i].j_batched;
        for j in 0..j_batched.len() {
            temp *= (&round_proof.folded_witness[(j, 0)], &j_batched[j]);
            batched_projection_of_folded_witness[(i, 0)] += &temp;
        }
    }

    let mut folded_batched_projection_image = VerticallyAlignedMatrix::new_zero_preallocated(
        round_proof.batched_projection_image.height,
        1,
    );

    for i in 0..round_proof.batched_projection_image.height {
        for j in 0..commitment.width {
            temp *= (
                &round_proof.batched_projection_image[(i, j)],
                &folding_challenges[j],
            );
            folded_batched_projection_image[(i, 0)] += &temp;
        }
    }

    assert_eq!(
        batched_projection_of_folded_witness,
        folded_batched_projection_image
    );

    // check constant terms
    for i in 0..NOF_BATCHES {
        let c_1_layers = &challenges[i].c_1_layers;
        let c_1_values = precompute_structured_values_fast(&c_1_layers);

        debug_assert_eq!(
            c_1_values.len(),
            round_proof.projection_image_ct.height * DEGREE,
            "c_1_values length mismatch."
        );

        for k in 0..config.witness_width {
            let mut expected_ct = 0;
            for j in 0..c_1_values.len() / DEGREE {
                unsafe {
                    eltwise_mult_mod(
                        temp.v.as_mut_ptr(),
                        c_1_values.as_ptr().add(DEGREE * j),
                        round_proof.projection_image_ct[(j, k)].v.as_ptr(),
                        DEGREE as u64,
                        MOD_Q,
                    );
                }
                for l in 0..DEGREE {
                    // TODO: vectorize
                    expected_ct += temp.v[l];
                }
            }
            expected_ct %= MOD_Q;

            let ct =
                round_proof.batched_projection_image[(i, k)].constant_term_from_incomplete_ntt();
            assert_eq!(ct, expected_ct);
        }
    }

    let mut evaluation = new_vec_zero_preallocated(round_proof.opening_rhs.height);

    for i in 0..round_proof.opening_rhs.height {
        let preprocessed_row = PreprocessedRow::from_structured_row(&evaluation_points_outer[i]);
        for col in 0..round_proof.opening_rhs.width {
            temp *= (
                &round_proof.opening_rhs[(i, col)],
                &preprocessed_row.preprocessed_row[col],
            );
            evaluation[i] += &temp;
        }
    }

    let mut witness_even_odd = new_vec_zero_preallocated(round_proof.folded_witness.height);
    witness_even_odd.clone_from_slice(&round_proof.folded_witness.data);

    for w in witness_even_odd.iter_mut() {
        w.from_incomplete_ntt_to_even_odd_coefficients();
    }

    let l2_norm_witness = l2_norm_coeffs(&witness_even_odd);
    let l2_norm_proj = l2_norm_coeffs(&round_proof.projection_image_ct.data);

    println!(
        "L2 norm of folded witness in simple verifier: {}",
        l2_norm_witness
    );
    println!(
        "L2 norm of projection image in simple verifier: {}",
        l2_norm_proj
    );

    let elapsed = start.elapsed().as_nanos();
    println!("Simple verifier: {} ns", elapsed);

    assert_eq!(claims, &evaluation);
}
