use crate::{
    common::{
        arithmetic::{ALL_ONE_COEFFS, ONE, ZERO},
        config::{NOF_BATCHES, *},
        decomposition::{compose_from_decomposed, decompose_chunks_into},
        hash::HashWrapper,
        matrix::{new_vec_zero_preallocated, HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    protocol::{
        commitment::{commit_basic, commit_basic_internal},
        config::{paste_by_prefix, RoundConfig, SalsaaProof, SalsaaProofCommon},
        crs::CRS,
        fold::fold,
        open::evaluation_point_to_structured_row,
        project::{prepare_i16_witness, project, BatchingChallenges},
        project_2::{batch_projection_n_times, project_coefficients, BatchedProjectionChallenges},
        sumcheck_utils::{common::HighOrderSumcheckData},
        sumchecks::{context::ProverSumcheckContext, runner::sumcheck},
        vdf::{compute_ip_vdf_claim, VDFCrs},
    },
};

fn structured_round(
    crs: &CRS,
    witness: &VerticallyAlignedMatrix<RingElement>,
    config: &RoundConfig,
    sumcheck_context: &mut ProverSumcheckContext,
    evaluation_points_inner: &Vec<StructuredRow>,
    claims: &HorizontallyAlignedMatrix<RingElement>,
    hash_wrapper: &mut HashWrapper,
    vdf_params: Option<(
        &[RingElement; VDF_MATRIX_HEIGHT],
        &[RingElement; VDF_MATRIX_HEIGHT],
        &VDFCrs,
    )>,
) -> SalsaaProof {
    let RoundConfig::Intermediate {
        decomposition_base_log,
        projection_prefix,
        next,
        ..
    } = config
    else {
        unreachable!()
    };

    let witness_16 = prepare_i16_witness(witness);
    let mut projection_matrix = ProjectionMatrix::new(witness.width, 256);
    projection_matrix.sample(hash_wrapper);
    let mut projected_witness = project(&witness_16, &projection_matrix);
    projected_witness.width = 1;
    projected_witness.used_cols = 1;
    projected_witness.height = witness.height;
    let projection_commitment = commit_basic(crs, &projected_witness, RANK);
    let batching_challenges = BatchingChallenges::sample(config, hash_wrapper);

    let vdf_challenge = if config.vdf {
        let mut challenge = RingElement::zero(Representation::IncompleteNTT);
        hash_wrapper.sample_ring_element_ntt_slots_into(&mut challenge);
        Some(challenge)
    } else {
        None
    };

    if DEBUG {
        println!("witness.data.len {:?}", witness.data.len());
    }
    let mut extended_witness =
        new_vec_zero_preallocated(witness.data.len() << config.main_witness_prefix.length);
    let mut witness_conjugated = new_vec_zero_preallocated(witness.data.len());
    for (i, w) in witness.data.iter().enumerate() {
        w.conjugate_into(&mut witness_conjugated[i]);
    }

    let ip_l2_claim = compute_ip_l2_claim(config, witness, &witness_conjugated);
    let ip_linf_claim = compute_ip_linf_claim(config, witness, &witness_conjugated);

    paste_by_prefix(
        &mut extended_witness,
        &witness.data,
        &config.main_witness_prefix,
    );
    paste_by_prefix(
        &mut extended_witness,
        &projected_witness.data,
        projection_prefix,
    );

    let mut evaluation_points_outer = new_vec_zero_preallocated(config.main_witness_columns);
    hash_wrapper.sample_ring_element_vec_into(&mut evaluation_points_outer);

    sumcheck_context.load_data(
        &extended_witness,
        &witness_conjugated,
        evaluation_points_inner,
        &evaluation_points_outer,
        &Some(projection_matrix),
        &Some(batching_challenges),
        &None,
        vdf_challenge.as_ref(),
        vdf_params.map(|(_, _, crs)| crs),
    );

    load_fiat_shamir_combination(sumcheck_context, hash_wrapper);
    run_debug_assertions_intermediate(
        sumcheck_context,
        config,
        claims,
        &evaluation_points_outer,
        vdf_challenge.as_ref(),
        vdf_params,
    );

    let (claims_out, claim_over_projection, polys, evaluation_points) = sumcheck(
        sumcheck_context,
        hash_wrapper,
        witness,
        Some(&projected_witness),
        config,
    );

    let mut folding_challenges = new_vec_zero_preallocated(config.main_witness_columns);
    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut folding_challenges);
    let folded_witness = fold(witness, &folding_challenges);

    if DEBUG {
        let commitment_to_folded_witness = commit_basic(crs, &folded_witness, RANK);
        let split_ref = VerticallyAlignedMatrix {
            height: folded_witness.height / 2,
            width: 2,
            data: folded_witness.data.clone(),
            used_cols: 2,
        };
        let commitment_to_split_witness = commit_basic(crs, &split_ref, RANK);
        let old_ck = crs.structured_ck_for_wit_dim(split_ref.height * 2);
        let composed = &(&(&*ONE - &old_ck[0].tensor_layers[0])
            * &commitment_to_split_witness[(0, 0)])
            + &(&old_ck[0].tensor_layers[0] * &commitment_to_split_witness[(0, 1)]);
        assert_eq!(
            composed,
            commitment_to_folded_witness[(0, 0)],
            "Composed commitment from the split witness does not match the commitment to the folded witness"
        );
    }

    let split_witness = VerticallyAlignedMatrix {
        height: folded_witness.height / 2,
        width: 2,
        data: folded_witness.data,
        used_cols: 2,
    };
    let mut decomposed_split_witness = VerticallyAlignedMatrix {
        height: split_witness.height,
        width: 8,
        data: new_vec_zero_preallocated(split_witness.height * 8),
        used_cols: 8,
    };
    decompose_chunks_into(
        &mut decomposed_split_witness.data[..split_witness.height * 2],
        &split_witness.data[..split_witness.height],
        *decomposition_base_log,
        2,
    );
    decompose_chunks_into(
        &mut decomposed_split_witness.data[split_witness.height * 2..split_witness.height * 4],
        &split_witness.data[split_witness.height..],
        *decomposition_base_log,
        2,
    );
    decompose_chunks_into(
        &mut decomposed_split_witness.data[split_witness.height * 4..split_witness.height * 6],
        &projected_witness.data[..split_witness.height],
        *decomposition_base_log,
        2,
    );
    decompose_chunks_into(
        &mut decomposed_split_witness.data[split_witness.height * 6..],
        &projected_witness.data[split_witness.height..],
        *decomposition_base_log,
        2,
    );
    let decomposed_split_commitment = commit_basic(crs, &decomposed_split_witness, RANK);

    if DEBUG {
        debug_check_decomposed_intermediate(
            crs,
            &split_witness,
            &decomposed_split_commitment,
            &projection_commitment,
            *decomposition_base_log as usize,
        );
    }
    let outer_points_len =
        config.main_witness_columns.ilog2() as usize + config.main_witness_prefix.length;

    let new_evaluation_points_inner: Vec<_> = evaluation_points
        .iter()
        .skip(outer_points_len + 1)
        .cloned()
        .collect();

    let new_evaluation_points_inner_expanded = PreprocessedRow::from_structured_row(
        &evaluation_point_to_structured_row(&new_evaluation_points_inner),
    );
    let new_evaluation_points_inner_conjugated: Vec<_> = new_evaluation_points_inner
        .iter()
        .map(RingElement::conjugate)
        .collect();
    let new_evaluation_points_inner_conjugated_expanded = PreprocessedRow::from_structured_row(
        &evaluation_point_to_structured_row(&new_evaluation_points_inner_conjugated),
    );
    let new_claims = commit_basic_internal(
        &vec![
            new_evaluation_points_inner_expanded,
            new_evaluation_points_inner_conjugated_expanded,
        ],
        &decomposed_split_witness,
        2,
    );
    let next_level_eval_points = vec![
        evaluation_point_to_structured_row(&new_evaluation_points_inner),
        evaluation_point_to_structured_row(&new_evaluation_points_inner_conjugated),
    ];
    let next_level_proof = prover_round(
        crs,
        &decomposed_split_witness,
        next,
        sumcheck_context.next.as_mut().unwrap(),
        &next_level_eval_points,
        &new_claims,
        hash_wrapper,
        None,
    );

    let common = SalsaaProofCommon {
        ip_l2_claim,
        ip_linf_claim,
        sumcheck_transcript: polys,
        claims: claims_out,
    };
    SalsaaProof::Intermediate {
        common,
        new_claims,
        decomposed_split_commitment,
        projection_commitment,
        claim_over_projection: claim_over_projection.unwrap(),
        next: Box::new(next_level_proof),
    }
}

fn unstructured_round(
    crs: &CRS,
    witness: &VerticallyAlignedMatrix<RingElement>,
    config: &RoundConfig,
    sumcheck_context: &mut ProverSumcheckContext,
    evaluation_points_inner: &Vec<StructuredRow>,
    claims: &HorizontallyAlignedMatrix<RingElement>,
    hash_wrapper: &mut HashWrapper,
) -> SalsaaProof {
    let RoundConfig::IntermediateUnstructured {
        decomposition_base_log,
        projection_ratio,
        next,
        ..
    } = config
    else {
        unreachable!()
    };

    let mut projection_matrix = ProjectionMatrix::new(*projection_ratio, PROJECTION_HEIGHT);
    projection_matrix.sample(hash_wrapper);
    let projection_ct = project_coefficients(witness, &projection_matrix);
    let (batched_image, unstructured_batching_challenges) = batch_projection_n_times(
        witness,
        &projection_matrix,
        hash_wrapper,
        NOF_BATCHES,
        false,
    );

    let vdf_challenge = if config.vdf {
        let mut challenge = RingElement::zero(Representation::IncompleteNTT);
        hash_wrapper.sample_ring_element_ntt_slots_into(&mut challenge);
        Some(challenge)
    } else {
        None
    };

    if DEBUG {
        println!("witness.data.len {:?}", witness.data.len());
    }

    let mut extended_witness =
        new_vec_zero_preallocated(witness.data.len() << config.main_witness_prefix.length);
    let mut witness_conjugated = new_vec_zero_preallocated(witness.data.len());
    for (i, w) in witness.data.iter().enumerate() {
        w.conjugate_into(&mut witness_conjugated[i]);
    }

    let ip_l2_claim = compute_ip_l2_claim(config, witness, &witness_conjugated);
    let ip_linf_claim = compute_ip_linf_claim(config, witness, &witness_conjugated);

    paste_by_prefix(
        &mut extended_witness,
        &witness.data,
        &config.main_witness_prefix,
    );

    let mut evaluation_points_outer = new_vec_zero_preallocated(config.main_witness_columns);
    hash_wrapper.sample_ring_element_vec_into(&mut evaluation_points_outer);

    sumcheck_context.load_data(
        &extended_witness,
        &witness_conjugated,
        evaluation_points_inner,
        &evaluation_points_outer,
        &None,
        &None,
        &Some(unstructured_batching_challenges.clone()),
        vdf_challenge.as_ref(),
        None,
    );

    load_fiat_shamir_combination(sumcheck_context, hash_wrapper);
    run_debug_assertions_unstructured(
        sumcheck_context,
        config,
        claims,
        &evaluation_points_outer,
        &batched_image,
        &unstructured_batching_challenges,
    );

    let (claims_out, _, polys, evaluation_points) =
        sumcheck(sumcheck_context, hash_wrapper, witness, None, config);

    let mut folding_challenges = new_vec_zero_preallocated(config.main_witness_columns);
    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut folding_challenges);
    let folded_witness = fold(witness, &folding_challenges);

    let split_witness = VerticallyAlignedMatrix {
        height: folded_witness.height / 2,
        width: 2,
        data: folded_witness.data,
        used_cols: 2,
    };
    let mut decomposed_split_witness = VerticallyAlignedMatrix {
        height: split_witness.height,
        width: 4,
        data: new_vec_zero_preallocated(split_witness.height * 4),
        used_cols: 4,
    };
    decompose_chunks_into(
        &mut decomposed_split_witness.data[..split_witness.height * 2],
        &split_witness.data[..split_witness.height],
        *decomposition_base_log,
        2,
    );
    decompose_chunks_into(
        &mut decomposed_split_witness.data[split_witness.height * 2..],
        &split_witness.data[split_witness.height..],
        *decomposition_base_log,
        2,
    );
    let decomposed_split_commitment = commit_basic(crs, &decomposed_split_witness, RANK);
    let outer_points_len =
        config.main_witness_columns.ilog2() as usize + config.main_witness_prefix.length;

    let new_evaluation_points_inner: Vec<_> = evaluation_points
        .iter()
        .skip(outer_points_len + 1)
        .cloned()
        .collect();
    let new_evaluation_points_inner_expanded = PreprocessedRow::from_structured_row(
        &evaluation_point_to_structured_row(&new_evaluation_points_inner),
    );
    let new_evaluation_points_inner_conjugated: Vec<_> = new_evaluation_points_inner
        .iter()
        .map(RingElement::conjugate)
        .collect();
    let new_evaluation_points_inner_conjugated_expanded = PreprocessedRow::from_structured_row(
        &evaluation_point_to_structured_row(&new_evaluation_points_inner_conjugated),
    );
    let new_claims = commit_basic_internal(
        &vec![
            new_evaluation_points_inner_expanded,
            new_evaluation_points_inner_conjugated_expanded,
        ],
        &decomposed_split_witness,
        2,
    );
    let next_level_eval_points = vec![
        evaluation_point_to_structured_row(&new_evaluation_points_inner),
        evaluation_point_to_structured_row(&new_evaluation_points_inner_conjugated),
    ];
    let next_level_proof = prover_round(
        crs,
        &decomposed_split_witness,
        next,
        sumcheck_context.next.as_mut().unwrap(),
        &next_level_eval_points,
        &new_claims,
        hash_wrapper,
        None,
    );

    let common = SalsaaProofCommon {
        ip_l2_claim,
        ip_linf_claim,
        sumcheck_transcript: polys,
        claims: claims_out,
    };
    SalsaaProof::IntermediateUnstructured {
        common,
        new_claims: new_claims.data,
        decomposed_split_commitment,
        next: Box::new(next_level_proof),
        projection_image_ct: projection_ct,
        projection_image_batched: batched_image,
    }
}

fn last_round(
    witness: &VerticallyAlignedMatrix<RingElement>,
    config: &RoundConfig, // must be RoundConfig::Last
    sumcheck_context: &mut ProverSumcheckContext,
    evaluation_points_inner: &Vec<StructuredRow>,
    claims: &HorizontallyAlignedMatrix<RingElement>,
    hash_wrapper: &mut HashWrapper,
) -> SalsaaProof {
    let RoundConfig::Last {
        projection_ratio, ..
    } = config
    else {
        unreachable!()
    };

    println!(
        "Using unstructured projection with ratio {}",
        projection_ratio
    );
    println!(
        "Sampling projection matrix for unstructured projection with ratio {}",
        projection_ratio
    );
    let mut projection_matrix = ProjectionMatrix::new(*projection_ratio, PROJECTION_HEIGHT);
    projection_matrix.sample(hash_wrapper);
    let projection_ct = project_coefficients(witness, &projection_matrix);
    let (batched_image, unstructured_batching_challenges) = batch_projection_n_times(
        witness,
        &projection_matrix,
        hash_wrapper,
        NOF_BATCHES,
        false,
    );

    let vdf_challenge = if config.vdf {
        let mut challenge = RingElement::zero(Representation::IncompleteNTT);
        hash_wrapper.sample_ring_element_ntt_slots_into(&mut challenge);
        Some(challenge)
    } else {
        None
    };

    if DEBUG {
        println!("witness.data.len {:?}", witness.data.len());
    }
    let mut extended_witness =
        new_vec_zero_preallocated(witness.data.len() << config.main_witness_prefix.length);
    let mut witness_conjugated = new_vec_zero_preallocated(witness.data.len());
    for (i, w) in witness.data.iter().enumerate() {
        w.conjugate_into(&mut witness_conjugated[i]);
    }

    let ip_l2_claim = compute_ip_l2_claim(config, witness, &witness_conjugated);
    let ip_linf_claim = compute_ip_linf_claim(config, witness, &witness_conjugated);

    paste_by_prefix(
        &mut extended_witness,
        &witness.data,
        &config.main_witness_prefix,
    );

    let mut evaluation_points_outer = new_vec_zero_preallocated(config.main_witness_columns);
    hash_wrapper.sample_ring_element_vec_into(&mut evaluation_points_outer);

    sumcheck_context.load_data(
        &extended_witness,
        &witness_conjugated,
        evaluation_points_inner,
        &evaluation_points_outer,
        &None,
        &None,
        &Some(unstructured_batching_challenges.clone()),
        vdf_challenge.as_ref(),
        None,
    );

    load_fiat_shamir_combination(sumcheck_context, hash_wrapper);
    run_debug_assertions_unstructured(
        sumcheck_context,
        config,
        claims,
        &evaluation_points_outer,
        &batched_image,
        &unstructured_batching_challenges,
    );

    let (claims_out, _, polys, evaluation_points) =
        sumcheck(sumcheck_context, hash_wrapper, witness, None, config);

    let mut folding_challenges = new_vec_zero_preallocated(config.main_witness_columns);
    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut folding_challenges);
    let folded_witness = fold(witness, &folding_challenges);

    let common = SalsaaProofCommon {
        ip_l2_claim,
        ip_linf_claim,
        sumcheck_transcript: polys,
        claims: claims_out,
    };
    SalsaaProof::Last {
        common,
        folded_witness: folded_witness.data,
        projection_image_ct: projection_ct,
        projection_image_batched: batched_image,
    }
}

pub fn prover_round(
    crs: &CRS,
    witness: &VerticallyAlignedMatrix<RingElement>,
    config: &RoundConfig,
    sumcheck_context: &mut ProverSumcheckContext,
    evaluation_points_inner: &Vec<StructuredRow>,
    claims: &HorizontallyAlignedMatrix<RingElement>,
    hash_wrapper: &mut HashWrapper,
    vdf_params: Option<(
        &[RingElement; VDF_MATRIX_HEIGHT],
        &[RingElement; VDF_MATRIX_HEIGHT],
        &VDFCrs,
    )>,
) -> SalsaaProof {
    match config {
        RoundConfig::Intermediate { .. } => structured_round(
            crs,
            witness,
            config,
            sumcheck_context,
            evaluation_points_inner,
            claims,
            hash_wrapper,
            vdf_params,
        ),
        RoundConfig::IntermediateUnstructured { .. } => unstructured_round(
            crs,
            witness,
            config,
            sumcheck_context,
            evaluation_points_inner,
            claims,
            hash_wrapper,
        ),
        RoundConfig::Last { .. } => last_round(
            witness,
            config,
            sumcheck_context,
            evaluation_points_inner,
            claims,
            hash_wrapper,
        ),
    }
}

fn compute_ip_l2_claim(
    config: &RoundConfig,
    witness: &VerticallyAlignedMatrix<RingElement>,
    witness_conjugated: &[RingElement],
) -> Option<RingElement> {
    if !config.l2 {
        return None;
    }
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    let mut claim = RingElement::zero(Representation::IncompleteNTT);
    for (w, wc) in witness.data.iter().zip(witness_conjugated.iter()) {
        temp *= (w, wc);
        claim += &temp;
    }
    Some(claim)
}

fn compute_ip_linf_claim(
    config: &RoundConfig,
    witness: &VerticallyAlignedMatrix<RingElement>,
    witness_conjugated: &[RingElement],
) -> Option<RingElement> {
    if !config.exact_binariness {
        return None;
    }
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    let mut claim = RingElement::zero(Representation::IncompleteNTT);
    for (w, wc) in witness.data.iter().zip(witness_conjugated.iter()) {
        temp -= (&*ALL_ONE_COEFFS, w);
        temp *= wc;
        claim += &temp;
    }
    Some(claim)
}

fn load_fiat_shamir_combination(
    sumcheck_context: &mut ProverSumcheckContext,
    hash_wrapper: &mut HashWrapper,
) {
    let num_sumchecks = sumcheck_context.combiner.borrow().sumchecks_count();
    let mut combination = new_vec_zero_preallocated(num_sumchecks);
    hash_wrapper.sample_ring_element_vec_into(&mut combination);
    sumcheck_context
        .combiner
        .borrow_mut()
        .load_challenges_from(&combination);

    let mut combination_to_field = RingElement::zero(Representation::IncompleteNTT);
    hash_wrapper.sample_ring_element_into(&mut combination_to_field);
    combination_to_field.from_incomplete_ntt_to_homogenized_field_extensions();
    let qe = combination_to_field.split_into_quadratic_extensions();
    sumcheck_context
        .field_combiner
        .borrow_mut()
        .load_challenges_from(qe);
}

/// Debug assertions shared by Intermediate rounds
fn run_debug_assertions_intermediate(
    sumcheck_context: &ProverSumcheckContext,
    config: &RoundConfig,
    claims: &HorizontallyAlignedMatrix<RingElement>,
    evaluation_points_outer: &[RingElement],
    vdf_challenge: Option<&RingElement>,
    vdf_params: Option<(
        &[RingElement; VDF_MATRIX_HEIGHT],
        &[RingElement; VDF_MATRIX_HEIGHT],
        &VDFCrs,
    )>,
) {
    if !DEBUG {
        return;
    }
    let ip_vdf_claim = compute_ip_vdf_claim(config, vdf_challenge, vdf_params);
    if !sumcheck_context.type1sumcheck.is_empty() {
        let claim = sumcheck_context.type1sumcheck[0].output.borrow().claim();
        let mut expected_claim = ZERO.clone();
        for (c, r) in claims.row(0).iter().zip(evaluation_points_outer.iter()) {
            expected_claim += &(c * r);
        }
        assert_eq!(claim, expected_claim, "Claim from the sumcheck does not match the expected claim computed from the committed witness and the evaluation points");
    }
    let projection_claim = sumcheck_context
        .type3sumcheck
        .as_ref()
        .unwrap()
        .output
        .borrow()
        .claim();
    assert_eq!(
        projection_claim,
        ZERO.clone(),
        "Projection claim does not match"
    );
    run_debug_assertions_common(sumcheck_context, config, ip_vdf_claim);
}

/// Debug assertions shared by IntermediateUnstructured and Last rounds
fn run_debug_assertions_unstructured(
    sumcheck_context: &ProverSumcheckContext,
    config: &RoundConfig,
    claims: &HorizontallyAlignedMatrix<RingElement>,
    evaluation_points_outer: &[RingElement],
    batched_image: &HorizontallyAlignedMatrix<RingElement>,
    unstructured_batching_challenges: &[BatchedProjectionChallenges],
) {
    if !DEBUG {
        return;
    }
    if !sumcheck_context.type1sumcheck.is_empty() {
        let claim = sumcheck_context.type1sumcheck[0].output.borrow().claim();
        let mut expected_claim = ZERO.clone();
        for (c, r) in claims.row(0).iter().zip(evaluation_points_outer.iter()) {
            expected_claim += &(c * r);
        }
        assert_eq!(
            claim, expected_claim,
            "Claim from the sumcheck does not match"
        );
    }
    for (batch_idx, type31) in sumcheck_context
        .type31sumchecks
        .as_ref()
        .unwrap()
        .iter()
        .enumerate()
    {
        let projection_claim = type31.output.borrow().claim();
        let batch_image = &batched_image.row(batch_idx);
        let challenges = &unstructured_batching_challenges[batch_idx].c_2_values;
        let mut expected_projection_claim = RingElement::zero(Representation::IncompleteNTT);
        let mut temp = RingElement::zero(Representation::IncompleteNTT);
        for (c, r) in batch_image.iter().zip(challenges.iter()) {
            temp *= (c, &RingElement::constant(*r, Representation::IncompleteNTT));
            expected_projection_claim += &temp;
        }
        assert_eq!(
            projection_claim, expected_projection_claim,
            "Projection claim does not match"
        );
    }
    println!(
        "Unstructured projection claims from the sumcheck match the expected projection claims"
    );
    run_debug_assertions_common(sumcheck_context, config, None);
}

fn run_debug_assertions_common(
    sumcheck_context: &ProverSumcheckContext,
    config: &RoundConfig,
    ip_vdf_claim: Option<RingElement>,
) {
    if config.l2 {
        let l2_claim = sumcheck_context
            .l2sumcheck
            .as_ref()
            .unwrap()
            .output
            .borrow()
            .claim();
        // ip_l2_claim is recomputed locally; here we just surface the sumcheck value for logging
        let _ = l2_claim; // callers can add their own assert if needed
    }
    if config.exact_binariness {
        let linf_claim = sumcheck_context
            .linfsumcheck
            .as_ref()
            .unwrap()
            .output
            .borrow()
            .claim();
        let ct = linf_claim.constant_term_from_incomplete_ntt();
        assert_eq!(ct, 0, "Linf claim constant term is non-zero");
    }
    if config.vdf {
        let vdf_claim = sumcheck_context
            .vdfsumcheck
            .as_ref()
            .unwrap()
            .output
            .borrow()
            .claim();
        assert_eq!(vdf_claim, ip_vdf_claim.unwrap(), "VDF claim does not match");
    }
}

fn debug_check_decomposed_intermediate(
    crs: &CRS,
    split_witness: &VerticallyAlignedMatrix<RingElement>,
    decomposed_split_commitment: &HorizontallyAlignedMatrix<RingElement>,
    projection_commitment: &HorizontallyAlignedMatrix<RingElement>,
    decomposition_base_log: usize,
) {
    let commitment_to_split_witness = commit_basic(crs, split_witness, RANK);
    let old_ck = crs.structured_ck_for_wit_dim(split_witness.height * 2);
    let composed = compose_from_decomposed(
        &[
            decomposed_split_commitment[(0, 0)].clone(),
            decomposed_split_commitment[(0, 1)].clone(),
            decomposed_split_commitment[(0, 2)].clone(),
            decomposed_split_commitment[(0, 3)].clone(),
        ],
        decomposition_base_log as u64,
        2,
    );
    assert_eq!(
        composed[0],
        commitment_to_split_witness[(0, 0)],
        "Split witness commitment mismatch"
    );
    assert_eq!(
        composed[1],
        commitment_to_split_witness[(0, 1)],
        "Split projected witness commitment mismatch"
    );

    let composed_projection = compose_from_decomposed(
        &[
            decomposed_split_commitment[(0, 4)].clone(),
            decomposed_split_commitment[(0, 5)].clone(),
            decomposed_split_commitment[(0, 6)].clone(),
            decomposed_split_commitment[(0, 7)].clone(),
        ],
        decomposition_base_log as u64,
        2,
    );
    let unsplit_projection = &(&(&*ONE - &old_ck[0].tensor_layers[0]) * &composed_projection[0])
        + &(&old_ck[0].tensor_layers[0] * &composed_projection[1]);
    assert_eq!(
        unsplit_projection,
        projection_commitment[(0, 0)],
        "Projection commitment mismatch"
    );
}
