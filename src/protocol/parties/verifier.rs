use crate::{
    common::{
        arithmetic::precompute_structured_values_fast,
        config::{DEGREE, MOD_Q, NOF_BATCHES},
        hash::HashWrapper,
        matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        norms::{assert_norm_bounded, l2_norm_coeffs},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{FieldExtension, Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    hexl::bindings::{add_mod, eltwise_mult_mod, multiply_mod},
    protocol::{
        boundary::{BoundaryCapture, VerifierBoundary},
        commitment::{commit_basic_internal, BasicCommitment},
        config::{
            config_base_from_config, Config, IntermediateConfig, IntermediateRoundProof,
            NextRoundCommitment, Projection, RoundProof, SimpleConfig, SimpleRoundProof,
            SumcheckConfig, SumcheckRoundProof,
        },
        crs::VerifierCRS,
        intermediate_sumchecks::{
            context_verifier::IntermediateVerifierSumcheckContext,
            runner_verifier::intermediate_sumcheck_verifier,
        },
        open::{
            evaluation_point_to_structured_row, evaluation_point_to_structured_row_conjugate,
            open_at,
        },
        project_fine::{
            verifier_sample_projection_challenges_collectively, BatchedProjectionChallengesSuccinct,
        },
        sumcheck_utils::polynomial::Polynomial,
        sumchecks::{
            context_verifier::{NextVerifierSumcheckContext, VerifierSumcheckContext},
            runner_verifier::sumcheck_verifier,
        },
    },
};

#[tracing::instrument(skip_all, name = "verifier_round")]
pub fn verifier_round(
    crs: &VerifierCRS,
    config: &SumcheckConfig,
    rc_commitment: &[RingElement],
    round_proof: &SumcheckRoundProof,
    evaluation_points_inner: &[StructuredRow],
    evaluation_points_outer: &[StructuredRow],
    claims: &[RingElement],
    sumcheck_context_verifier: &mut VerifierSumcheckContext,
    hash_wrapper_verifier: Option<HashWrapper>,
    boundary: Option<BoundaryCapture<'_, VerifierBoundary>>,
) {
    check_sumcheck_round_shape(
        config,
        rc_commitment,
        round_proof,
        evaluation_points_inner,
        evaluation_points_outer,
        claims,
    );

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

    let at_cut = boundary.as_ref().is_some_and(BoundaryCapture::is_at_cut);

    if at_cut {
        let next_config = match config.next.as_deref() {
            Some(Config::Sumcheck(next_sumcheck_config)) => next_sumcheck_config.clone(),
            _ => panic!("round boundary cut requires the next round to be a Sumcheck round"),
        };
        let next_round_commitment = round_proof
            .next_round_commitment
            .as_ref()
            .expect("cut round must still carry the next-round commitment");
        let commitment_root = match next_round_commitment {
            NextRoundCommitment::Recursive(rc) => rc.clone(),
            NextRoundCommitment::Simple(_) => {
                panic!("round boundary cut requires a recursive next-round commitment")
            }
        };
        let capture = boundary.expect("round boundary cut requires a boundary slot");
        *capture.slot = Some(VerifierBoundary {
            config: next_config,
            commitment_root,
            claims: [
                round_proof.claim_over_witness.clone(),
                round_proof.claim_over_witness_conjugate.conjugate(),
            ],
            evaluation_points,
            transcript: hash_wrapper_verifier,
        });
        return;
    }

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
                        match sumcheck_context_verifier.next.as_deref_mut() {
                            Some(NextVerifierSumcheckContext::Simple(ctx)) => ctx,
                            _ => panic!("Expected Simple context for next round."),
                        },
                        Some(hash_wrapper_verifier),
                        boundary.and_then(BoundaryCapture::advance),
                    );
                }

                RoundProof::Simple(next_simple_round_proof) => {
                    assert!(
                        boundary.is_none(),
                        "round boundary cut requires the next round to be a Sumcheck round"
                    );
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
                RoundProof::Intermediate(next_intermediate_round_proof) => {
                    assert!(
                        boundary.is_none(),
                        "round boundary cut requires the next round to be a Sumcheck round"
                    );
                    let next_intermediate_config = match &config.next {
                        Some(next_config) => match next_config.as_ref() {
                            Config::Intermediate(next_intermediate_config) => {
                                next_intermediate_config
                            }
                            _ => panic!("Expected intermediate config for next round."),
                        },
                        None => panic!("Next intermediate config must be present."),
                    };

                    let (new_evaluation_points_outer, new_evaluation_points_inner) =
                        evaluation_points
                            .split_at(next_intermediate_config.witness_width.ilog2() as usize);

                    let commitment = match &next_round_commitment {
                        NextRoundCommitment::Simple(basic_commitment) => basic_commitment,
                        _ => panic!(
                            "Expected NextRoundCommitment::Simple for intermediate next round."
                        ),
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

                    verifier_round_intermediate(
                        crs,
                        next_intermediate_config,
                        commitment,
                        next_intermediate_round_proof,
                        &inner_rows,
                        &outer_rows,
                        &new_claims,
                        match sumcheck_context_verifier.next.as_deref_mut() {
                            Some(NextVerifierSumcheckContext::Intermediate(ctx)) => ctx,
                            _ => panic!("Expected Intermediate context for next round."),
                        },
                        Some(hash_wrapper_verifier),
                    );
                }
            }
        }
        None => {
            assert!(
                boundary.is_none(),
                "round boundary cut requested past the end of the round chain"
            );
            assert!(
                config.next.is_none(),
                "proof ends before the configured round chain"
            );
        }
    }
}

pub(crate) fn fold_matrix_claims(
    matrix: &HorizontallyAlignedMatrix<RingElement>,
    folding_challenges: &[RingElement],
) -> Vec<RingElement> {
    assert_eq!(
        folding_challenges.len(),
        matrix.width,
        "folding_challenges length must equal matrix width"
    );
    let mut folded_claims = vec![RingElement::zero(Representation::IncompleteNTT); matrix.height];
    let mut temp = RingElement::zero(Representation::IncompleteNTT);

    for row in 0..matrix.height {
        for col in 0..matrix.width {
            temp *= (&matrix[(row, col)], &folding_challenges[col]);
            folded_claims[row] += &temp;
        }
    }

    folded_claims
}

#[tracing::instrument(skip_all, name = "verifier_round_intermediate")]
pub fn verifier_round_intermediate(
    crs: &VerifierCRS,
    config: &IntermediateConfig,
    commitment: &BasicCommitment,
    round_proof: &IntermediateRoundProof,
    evaluation_points_inner: &[StructuredRow],
    evaluation_points_outer: &[StructuredRow],
    claims: &[RingElement],
    sumcheck_context_verifier: &mut IntermediateVerifierSumcheckContext,
    hash_wrapper: Option<HashWrapper>,
) {
    check_intermediate_round_shape(config, commitment, round_proof, claims.len());

    let mut hash_wrapper = hash_wrapper.unwrap_or_else(HashWrapper::new);
    hash_wrapper.update_with_ring_element_slice(&commitment.data);

    hash_wrapper.update_with_ring_element_slice(&round_proof.opening_rhs.data);

    let mut temp = RingElement::zero(Representation::IncompleteNTT);

    let mut last_col_opening_rhs =
        vec![RingElement::zero(Representation::IncompleteNTT); round_proof.opening_rhs.height];
    // instead of checking if claims are consistent with opening_rhs,
    // we assume they are and recompute the last column of opening_rhs to save on communication
    {
        let _s = tracing::info_span!("verifier_intermediate::recompute_last_col").entered();
        for i in 0..round_proof.opening_rhs.height {
            let preprocessed_row =
                PreprocessedRow::from_structured_row(&evaluation_points_outer[i]);

            last_col_opening_rhs[i].set_from(&claims[i]);
            for col in 0..round_proof.opening_rhs.width - 1 {
                temp *= (
                    &round_proof.opening_rhs[(i, col)],
                    &preprocessed_row.preprocessed_row[col],
                );
                last_col_opening_rhs[i] -= &temp;
            }
            temp.set_from(&preprocessed_row.preprocessed_row[round_proof.opening_rhs.width - 1]);
            temp.from_incomplete_ntt_to_homogenized_field_extensions();
            let mut inv_remaining_challenge = temp.inverse();
            inv_remaining_challenge.from_homogenized_field_extensions_to_incomplete_ntt();
            last_col_opening_rhs[i] *= &inv_remaining_challenge;
            temp.representation = Representation::IncompleteNTT;
        }
    }

    let mut projection_matrix =
        ProjectionMatrix::new(config.projection_ratio, config.projection_height);

    let challenges: [BatchedProjectionChallengesSuccinct; NOF_BATCHES] = {
        let _s = tracing::info_span!("verifier_intermediate::sample_projection").entered();
        projection_matrix.sample(&mut hash_wrapper);
        hash_wrapper.update_with_ring_element_slice(&round_proof.projection_image_ct.data);
        verifier_sample_projection_challenges_collectively(
            &projection_matrix,
            config,
            &mut hash_wrapper,
        )
    };

    let rows_per_chunk = config.projection_height / DEGREE;

    // constant term consistency
    {
        let _s = tracing::info_span!("verifier_intermediate::ct_consistency").entered();
        for i in 0..NOF_BATCHES {
            let c_0_values = precompute_structured_values_fast(&challenges[i].c_0_layers);
            let c_1_values = precompute_structured_values_fast(&challenges[i].c_1_layers);

            debug_assert_eq!(
                c_1_values.len(),
                config.projection_height,
                "c_1_values length mismatch."
            );

            for col in 0..config.witness_width {
                let mut expected_ct = 0u64;

                for row in 0..round_proof.projection_image_ct.height {
                    let chunk_idx = row / rows_per_chunk;
                    let c_0_coeff = c_0_values[chunk_idx];
                    let c_1_offset = (row % rows_per_chunk) * DEGREE;

                    unsafe {
                        eltwise_mult_mod(
                            temp.v.as_mut_ptr(),
                            c_1_values.as_ptr().add(c_1_offset),
                            round_proof.projection_image_ct[(row, col)].v.as_ptr(),
                            DEGREE as u64,
                            MOD_Q,
                        );
                    }

                    let mut row_sum = 0u64;
                    for l in 0..DEGREE {
                        unsafe {
                            row_sum = add_mod(row_sum, temp.v[l], MOD_Q);
                        }
                    }

                    unsafe {
                        let weighted = multiply_mod(row_sum, c_0_coeff, MOD_Q);
                        expected_ct = add_mod(expected_ct, weighted, MOD_Q);
                    }
                }

                let ct = round_proof.batched_projection_image[(i, col)]
                    .constant_term_from_incomplete_ntt();
                assert_eq!(ct, expected_ct);
            }
        }
    }

    hash_wrapper.update_with_ring_element_slice(&round_proof.batched_projection_image.data);

    let mut folding_challenges =
        vec![RingElement::zero(Representation::IncompleteNTT); config.witness_width];
    {
        let _s = tracing::info_span!("verifier_intermediate::sample_folding_challenges").entered();
        hash_wrapper.sample_low_op_norm_ring_vec_into(&mut folding_challenges);
    }

    let next_round_commitment =
        match round_proof
            .next_round_commitment
            .as_ref()
            .unwrap_or_else(|| {
                panic!("Next round commitment must be present for intermediate round proof.")
            }) {
            NextRoundCommitment::Simple(basic_commitment) => basic_commitment,
            _ => panic!("Expected simple commitment for intermediate round."),
        };
    hash_wrapper.update_with_ring_element_slice(&next_round_commitment.data);

    let mut folded_commitment =
        vec![RingElement::zero(Representation::IncompleteNTT); config.basic_commitment_rank];
    for row in 0..config.basic_commitment_rank {
        for col in 0..commitment.width {
            temp *= (&commitment[(row, col)], &folding_challenges[col]);
            folded_commitment[row] += &temp;
        }
    }

    let mut folded_opening_claims =
        vec![RingElement::zero(Representation::IncompleteNTT); round_proof.opening_rhs.height];
    for row in 0..round_proof.opening_rhs.height {
        for col in 0..round_proof.opening_rhs.width - 1 {
            temp *= (
                &round_proof.opening_rhs[(row, col)],
                &folding_challenges[col],
            );
            folded_opening_claims[row] += &temp;
        }
        temp *= (
            &last_col_opening_rhs[row],
            &folding_challenges[round_proof.opening_rhs.width - 1],
        );
        folded_opening_claims[row] += &temp;
    }

    let folded_batched_projection_claims =
        fold_matrix_claims(&round_proof.batched_projection_image, &folding_challenges);

    let l2_norm_proj = l2_norm_coeffs(&round_proof.projection_image_ct.data);
    assert_norm_bounded(
        "projection image in intermediate verifier",
        l2_norm_proj,
        config.projection_norm_bound,
    );

    let intermediate_evaluation_points = intermediate_sumcheck_verifier(
        config,
        sumcheck_context_verifier,
        &round_proof,
        &folded_commitment,
        &folded_opening_claims,
        &folded_batched_projection_claims,
        evaluation_points_inner,
        &challenges,
        &mut hash_wrapper,
    );

    let next_round_proof = round_proof.next.as_ref().unwrap_or_else(|| {
        panic!("Next round proof must be present for intermediate round proof.")
    });

    let next_round_config = config.next.as_ref().unwrap_or_else(|| {
        panic!("Next round config must be present for intermediate round proof.")
    });

    let next_witness_width = match next_round_config.as_ref() {
        Config::Simple(simple_config) => simple_config.witness_width,
        Config::Intermediate(intermediate_config) => intermediate_config.witness_width,
        Config::Sumcheck(_) => {
            unreachable!("Intermediate round must be followed by simple or intermediate round.")
        }
    };
    let (new_evaluation_points_outer, new_evaluation_points_inner) =
        intermediate_evaluation_points.split_at(next_witness_width.ilog2() as usize);
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

    match (next_round_proof.as_ref(), next_round_config.as_ref()) {
        (RoundProof::Simple(simple_round_proof), Config::Simple(simple_config)) => {
            verifier_round_simple(
                crs,
                simple_config,
                next_round_commitment,
                simple_round_proof,
                &inner_rows,
                &outer_rows,
                &new_claims,
                Some(hash_wrapper),
            );
        }
        (
            RoundProof::Intermediate(intermediate_round_proof),
            Config::Intermediate(intermediate_config),
        ) => {
            verifier_round_intermediate(
                crs,
                intermediate_config,
                next_round_commitment,
                intermediate_round_proof,
                &inner_rows,
                &outer_rows,
                &new_claims,
                sumcheck_context_verifier.next.as_deref_mut().unwrap(),
                Some(hash_wrapper),
            );
        }
        _ => panic!("Next round proof and config type mismatch."),
    }
}

#[tracing::instrument(skip_all, name = "verifier_round_simple")]
pub fn verifier_round_simple(
    crs: &VerifierCRS,
    config: &SimpleConfig,
    commitment: &BasicCommitment,
    round_proof: &SimpleRoundProof,
    evaluation_points_inner: &[StructuredRow],
    evaluation_points_outer: &[StructuredRow],
    claims: &[RingElement],
    hash_wrapper: Option<HashWrapper>,
) {
    check_simple_round_shape(config, commitment, round_proof, claims.len());

    let mut hash_wrapper = hash_wrapper.unwrap_or_else(HashWrapper::new);
    hash_wrapper.update_with_ring_element_slice(&commitment.data);
    hash_wrapper.update_with_ring_element_slice(&round_proof.opening_rhs.data);

    let mut projection_matrix =
        ProjectionMatrix::new(config.projection_ratio, config.projection_height);

    projection_matrix.sample(&mut hash_wrapper);

    hash_wrapper.update_with_ring_element_slice(&round_proof.projection_image_ct.data);

    let challenges: [BatchedProjectionChallengesSuccinct; NOF_BATCHES] =
        verifier_sample_projection_challenges_collectively(
            &projection_matrix,
            config,
            &mut hash_wrapper,
        );

    debug_assert_eq!(
        challenges[0].c_0_layers.len(),
        0,
        "In simple verifier, projection challenges c_0_layers length must be zero. At least, this is unimplemented otherwise."
    );

    hash_wrapper.update_with_ring_element_slice(&round_proof.batched_projection_image.data);

    let mut folding_challenges =
        vec![RingElement::zero(Representation::IncompleteNTT); config.witness_width];

    hash_wrapper.sample_low_op_norm_ring_vec_into(&mut folding_challenges);

    debug_assert_eq!(crs.simple_ck.len(), config.basic_commitment_rank);
    let commitment_of_folded_witness = commit_basic_internal(
        &crs.simple_ck,
        &round_proof.folded_witness,
        config.basic_commitment_rank,
    );

    let mut folded_commitment = HorizontallyAlignedMatrix {
        data: vec![
            RingElement::zero(Representation::IncompleteNTT);
            config.basic_commitment_rank * 1
        ],
        width: 1,
        height: config.basic_commitment_rank,
    };

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
            folded_commitment[(i, 0)],
            "Folded commitment at row {} does not match expected value.",
            i
        );
    }

    let opening_to_folded_witness = open_at(
        &round_proof.folded_witness,
        evaluation_points_inner,
        evaluation_points_outer,
        false,
    );

    let mut folded_opening = HorizontallyAlignedMatrix {
        data: vec![
            RingElement::zero(Representation::IncompleteNTT);
            round_proof.opening_rhs.height * 1
        ],
        width: 1,
        height: round_proof.opening_rhs.height,
    };

    let mut last_col_opening_rhs =
        vec![RingElement::zero(Representation::IncompleteNTT); round_proof.opening_rhs.height];
    // instead of checking if claims are consistent with opening_rhs,
    // we assume they are and recompute the last column of opening_rhs to save on communication
    for i in 0..round_proof.opening_rhs.height {
        let preprocessed_row = PreprocessedRow::from_structured_row(&evaluation_points_outer[i]);

        last_col_opening_rhs[i].set_from(&claims[i]);
        for col in 0..round_proof.opening_rhs.width - 1 {
            temp *= (
                &round_proof.opening_rhs[(i, col)],
                &preprocessed_row.preprocessed_row[col],
            );
            last_col_opening_rhs[i] -= &temp;
        }
        temp.set_from(&preprocessed_row.preprocessed_row[round_proof.opening_rhs.width - 1]);
        temp.from_incomplete_ntt_to_homogenized_field_extensions();
        let mut inv_remaining_challenge = temp.inverse();
        inv_remaining_challenge.from_homogenized_field_extensions_to_incomplete_ntt();
        last_col_opening_rhs[i] *= &inv_remaining_challenge;
        temp.representation = Representation::IncompleteNTT;
    }

    for i in 0..round_proof.opening_rhs.height {
        for col in 0..commitment.width - 1 {
            temp *= (&round_proof.opening_rhs[(i, col)], &folding_challenges[col]);
            folded_opening[(i, 0)] += &temp;
        }
        temp *= (
            &last_col_opening_rhs[i],
            &folding_challenges[commitment.width - 1],
        );
        folded_opening[(i, 0)] += &temp;
    }

    assert_eq!(opening_to_folded_witness.rhs, folded_opening);

    let mut batched_projection_of_folded_witness = VerticallyAlignedMatrix {
        data: vec![
            RingElement::zero(Representation::IncompleteNTT);
            round_proof.batched_projection_image.height * 1
        ],
        width: 1,
        height: round_proof.batched_projection_image.height,
        used_cols: 1,
    };

    for i in 0..round_proof.batched_projection_image.height {
        let j_batched = &challenges[i].j_batched;
        for j in 0..j_batched.len() {
            temp *= (&round_proof.folded_witness[(j, 0)], &j_batched[j]);
            batched_projection_of_folded_witness[(i, 0)] += &temp;
        }
    }

    let mut folded_batched_projection_image = VerticallyAlignedMatrix {
        data: vec![
            RingElement::zero(Representation::IncompleteNTT);
            round_proof.batched_projection_image.height * 1
        ],
        width: 1,
        height: round_proof.batched_projection_image.height,
        used_cols: 1,
    };

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

    let mut witness_strided =
        vec![RingElement::zero(Representation::IncompleteNTT); round_proof.folded_witness.height];
    witness_strided.clone_from_slice(&round_proof.folded_witness.data);

    for w in witness_strided.iter_mut() {
        w.from_incomplete_ntt_to_strided_coefficients();
    }

    let l2_norm_witness = l2_norm_coeffs(&witness_strided);
    let l2_norm_proj = l2_norm_coeffs(&round_proof.projection_image_ct.data);

    assert_norm_bounded(
        "folded witness in simple verifier",
        l2_norm_witness,
        config.witness_norm_bound,
    );
    assert_norm_bounded(
        "projection image in simple verifier",
        l2_norm_proj,
        config.projection_norm_bound,
    );
}

/// Panics unless a size a round proof carries is the one the round's config determines. The
/// transcript absorbs only the entries of a matrix or a slice, not its shape, so a shape is
/// trusted nowhere: every loop and index the verifier drives off a proof goes through one of the
/// passes below first.
fn expect_shape<T: PartialEq + std::fmt::Debug>(what: &str, got: T, expected: T) {
    assert!(
        got == expected,
        "proof shape: {what} is {got:?}, the round's config determines {expected:?}"
    );
}

fn vertical_shape<T>(matrix: &VerticallyAlignedMatrix<T>) -> (usize, usize, usize, usize) {
    (
        matrix.height,
        matrix.width,
        matrix.used_cols,
        matrix.data.len(),
    )
}

fn horizontal_shape<T>(matrix: &HorizontallyAlignedMatrix<T>) -> (usize, usize, usize) {
    (matrix.height, matrix.width, matrix.data.len())
}

fn is_zero(element: &RingElement) -> bool {
    element.v.iter().all(|&c| c == 0)
}

/// One round polynomial per variable, each at most cubic and with the tail the transcript absorbs
/// past its coefficients zero.
fn expect_polys(what: &str, polys: &[Polynomial<FieldExtension>], variables: usize) {
    expect_shape(what, polys.len(), variables);
    for (round, poly) in polys.iter().enumerate() {
        assert!(
            poly.num_coefficients <= poly.coefficients.len()
                && poly.coefficients[poly.num_coefficients..]
                    .iter()
                    .all(|c| c.coeffs.iter().all(|&x| x == 0)),
            "proof shape: {what} round {round} is not a cubic with a zero tail"
        );
    }
}

/// A basic commitment to a `width`-column witness: `rank` key rows padded with zero rows to a
/// power of two.
fn expect_basic_commitment(what: &str, commitment: &BasicCommitment, rank: usize, width: usize) {
    let height = rank.next_power_of_two();
    expect_shape(
        &format!("{what} (height, width, entries)"),
        horizontal_shape(commitment),
        (height, width, height * width),
    );
    assert!(
        commitment.data[rank * width..].iter().all(is_zero),
        "proof shape: {what} has non-zero padding rows"
    );
}

/// One opening row per claim; the last column is recomputed from the claims, so it travels as
/// zero.
fn expect_opening_rhs(
    what: &str,
    opening_rhs: &HorizontallyAlignedMatrix<RingElement>,
    claims: usize,
    width: usize,
) {
    expect_shape(
        &format!("{what} opening_rhs (height, width, entries)"),
        horizontal_shape(opening_rhs),
        (claims, width, claims * width),
    );
    assert!(
        (0..claims).all(|row| is_zero(&opening_rhs[(row, width - 1)])),
        "proof shape: {what} opening_rhs carries a non-zero last column"
    );
}

/// The projection images of a round with a fine projection: the constant terms, one row per
/// `projection_ratio` witness rows, and `NOF_BATCHES` batched rows.
fn expect_projection_images(
    what: &str,
    projection_image_ct: &VerticallyAlignedMatrix<RingElement>,
    batched_projection_image: &HorizontallyAlignedMatrix<RingElement>,
    image_height: usize,
    width: usize,
) {
    expect_shape(
        &format!("{what} projection_image_ct (height, width, used_cols, entries)"),
        vertical_shape(projection_image_ct),
        (image_height, width, width, image_height * width),
    );
    expect_shape(
        &format!("{what} batched_projection_image (height, width, entries)"),
        horizontal_shape(batched_projection_image),
        (NOF_BATCHES, width, NOF_BATCHES * width),
    );
}

/// The commitment a round sends for the next one: a recursive commitment's root when the next
/// round is a sumcheck round, a basic commitment otherwise, and nothing after the last round.
fn expect_next_round_commitment(next: Option<&Config>, commitment: &Option<NextRoundCommitment>) {
    match (next, commitment) {
        (None, None) => {}
        (Some(Config::Sumcheck(next)), Some(NextRoundCommitment::Recursive(root))) => expect_shape(
            "next-round commitment root length",
            root.len(),
            next.commitment_recursion.most_inner_config().rank,
        ),
        (
            Some(next @ (Config::Intermediate(_) | Config::Simple(_))),
            Some(NextRoundCommitment::Simple(basic)),
        ) => {
            let next = config_base_from_config(next);
            expect_basic_commitment(
                "next-round commitment",
                basic,
                next.basic_commitment_rank(),
                next.witness_width(),
            );
        }
        (next, commitment) => panic!(
            "proof shape: the next-round commitment is {}, the round's config determines {}",
            match commitment {
                None => "absent",
                Some(NextRoundCommitment::Recursive(_)) => "a recursive commitment root",
                Some(NextRoundCommitment::Simple(_)) => "a basic commitment",
            },
            match next {
                None => "none",
                Some(Config::Sumcheck(_)) => "a recursive commitment root",
                Some(_) => "a basic commitment",
            }
        ),
    }
}

/// The structural pass over a sumcheck round, run before any of the round is read. It covers the
/// statement too, since below the root that is the previous round's output.
fn check_sumcheck_round_shape(
    config: &SumcheckConfig,
    rc_commitment: &[RingElement],
    proof: &SumcheckRoundProof,
    evaluation_points_inner: &[StructuredRow],
    evaluation_points_outer: &[StructuredRow],
    claims: &[RingElement],
) {
    let layers = |rows: &[StructuredRow]| -> Vec<usize> {
        rows.iter().map(|row| row.tensor_layers.len()).collect()
    };
    expect_shape("claim count", claims.len(), config.nof_openings);
    expect_shape(
        "inner evaluation point layers",
        layers(evaluation_points_inner),
        vec![config.witness_height.ilog2() as usize; config.nof_openings],
    );
    expect_shape(
        "outer evaluation point layers",
        layers(evaluation_points_outer),
        vec![config.witness_width.ilog2() as usize; config.nof_openings],
    );
    expect_shape(
        "commitment root length",
        rc_commitment.len(),
        config.commitment_recursion.most_inner_config().rank,
    );

    expect_polys(
        "sumcheck round polys",
        &proof.polys,
        config.composed_witness_length.ilog2() as usize,
    );
    expect_shape(
        "projection_norm_claim presence",
        proof.projection_norm_claim.is_some(),
        config.projection_norm_scope().is_some(),
    );
    expect_shape(
        "rc_opening_inner length",
        proof.rc_opening_inner.len(),
        config.opening_recursion.most_inner_config().rank,
    );
    let (coarse, fine) = match &config.projection_recursion {
        Projection::Coarse(recursion) => (Some(recursion), None),
        Projection::Fine(recursion) => (None, Some(recursion)),
        Projection::Skip => (None, None),
    };
    expect_shape(
        "rc_coarse_projection_inner length",
        proof.rc_coarse_projection_inner.as_ref().map(Vec::len),
        coarse.map(|recursion| recursion.most_inner_config().rank),
    );
    expect_shape(
        "rc_fine_projection_inner lengths",
        proof
            .rc_fine_projection_inner
            .as_ref()
            .map(|(constant_term, batched)| (constant_term.len(), batched.len())),
        fine.map(|recursion| {
            (
                recursion.recursion_constant_term.most_inner_config().rank,
                recursion
                    .recursion_batched_projection
                    .most_inner_config()
                    .rank,
            )
        }),
    );
    expect_shape(
        "constant_term_claims length",
        proof.constant_term_claims.as_ref().map(Vec::len),
        fine.map(|_| NOF_BATCHES),
    );
    expect_next_round_commitment(config.next.as_deref(), &proof.next_round_commitment);
}

/// The structural pass over an intermediate round, `commitment` being the one the previous round
/// sent for it.
fn check_intermediate_round_shape(
    config: &IntermediateConfig,
    commitment: &BasicCommitment,
    proof: &IntermediateRoundProof,
    claims: usize,
) {
    expect_basic_commitment(
        "intermediate round commitment",
        commitment,
        config.basic_commitment_rank,
        config.witness_width,
    );
    expect_opening_rhs(
        "intermediate round",
        &proof.opening_rhs,
        claims,
        config.witness_width,
    );
    expect_polys(
        "intermediate round polys",
        &proof.polys,
        (config.witness_height * config.witness_decomposition_chunks).ilog2() as usize,
    );
    expect_next_round_commitment(config.next.as_deref(), &proof.next_round_commitment);
    expect_projection_images(
        "intermediate round",
        &proof.projection_image_ct,
        &proof.batched_projection_image,
        config.witness_height / config.projection_ratio,
        config.witness_width,
    );
}

/// The structural pass over the last round, `commitment` being the one the previous round sent
/// for it.
fn check_simple_round_shape(
    config: &SimpleConfig,
    commitment: &BasicCommitment,
    proof: &SimpleRoundProof,
    claims: usize,
) {
    expect_basic_commitment(
        "simple round commitment",
        commitment,
        config.basic_commitment_rank,
        config.witness_width,
    );
    expect_shape(
        "simple round folded_witness (height, width, used_cols, entries)",
        vertical_shape(&proof.folded_witness),
        (config.witness_height, 1, 1, config.witness_height),
    );
    expect_projection_images(
        "simple round",
        &proof.projection_image_ct,
        &proof.batched_projection_image,
        config.witness_height / config.projection_ratio,
        config.witness_width,
    );
    expect_opening_rhs(
        "simple round",
        &proof.opening_rhs,
        claims,
        config.witness_width,
    );
}
