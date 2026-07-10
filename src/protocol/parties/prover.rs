use crate::{
    common::{
        config::NOF_BATCHES,
        decomposition::decompose,
        hash::HashWrapper,
        matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    protocol::{
        commitment::{
            commit_basic, recursive_commit, BasicCommitment, CommitmentWithAux,
            RecursiveCommitmentWithAux,
        },
        config::{
            config_base_from_config, paste_by_prefix, paste_recursive_commitment, Config,
            IntermediateConfig, IntermediateRoundProof, NextRoundCommitment, Projection,
            RoundProof, SimpleConfig, SimpleRoundProof, SumcheckConfig, SumcheckRoundProof,
        },
        crs::CRS,
        fold::fold,
        intermediate_sumchecks::{
            context::IntermediateSumcheckContext, runner::run_intermediate_sumcheck,
        },
        open::{
            evaluation_point_to_structured_row, evaluation_point_to_structured_row_conjugate,
            open_at,
        },
        project_coarse::{prepare_i16_witness, project},
        project_fine::{batch_projection_n_times, project_coefficients},
        sumcheck::{sumcheck, SumcheckContext},
        sumchecks::context::NextSumcheckContext,
    },
};

/// Outer evaluation claims t_j = <T[j,:], r_j> (paper: l_j^T W r_j = t_j).
pub fn outer_eval_claims(
    rhs: &HorizontallyAlignedMatrix<RingElement>,
    evaluation_points_outer: &Vec<StructuredRow>,
) -> Vec<RingElement> {
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    let mut result = vec![RingElement::zero(Representation::IncompleteNTT); rhs.height];
    for i in 0..rhs.height {
        let preprocessed_row_outer =
            PreprocessedRow::from_structured_row(&evaluation_points_outer[i]);
        for col in 0..rhs.width {
            temp *= (
                &rhs[(i, col)],
                &preprocessed_row_outer.preprocessed_row[col],
            );
            result[i] += &temp;
        }
    }
    result
}

/// Evaluation point and its conjugate, as the next round's evaluation rows
/// (paper, Pi^lin output: l_0 = tensor(c), l_1 = tensor(conj(c))).
fn evaluation_point_pair(points: &[RingElement]) -> Vec<StructuredRow> {
    vec![
        evaluation_point_to_structured_row(points),
        evaluation_point_to_structured_row_conjugate(points),
    ]
}

enum PendingNextCommitment {
    Recursive(RecursiveCommitmentWithAux),
    Basic(BasicCommitment),
}

pub fn prover_round(
    crs: &CRS,
    config: &SumcheckConfig,
    commitment_with_aux: &CommitmentWithAux,
    witness: &VerticallyAlignedMatrix<RingElement>,
    evaluation_points_inner: &Vec<StructuredRow>,
    evaluation_points_outer: &Vec<StructuredRow>,
    sumcheck_context: &mut SumcheckContext,
    with_claims: bool,
    hash_wrapper: Option<HashWrapper>,
) -> (SumcheckRoundProof, Option<Vec<RingElement>>) {
    let mut hash_wrapper = hash_wrapper.unwrap_or_else(HashWrapper::new);
    let rc_commitment = &commitment_with_aux.rc_commitment_with_aux;

    let start = std::time::Instant::now();
    hash_wrapper.update_with_ring_element_slice(&rc_commitment.most_inner_commitment());

    let t0 = std::time::Instant::now();
    let opening = open_at(
        &witness,
        &evaluation_points_inner,
        &evaluation_points_outer,
        false,
    );

    let claims = if with_claims {
        Some(outer_eval_claims(&opening.rhs, evaluation_points_outer))
    } else {
        None
    };
    println!("  open_at: {} ms", t0.elapsed().as_millis());
    let t1 = std::time::Instant::now();

    let rc_opening = recursive_commit(crs, &config.opening_recursion, &opening.rhs.data);
    println!("  rc_opening: {} ms", t1.elapsed().as_millis());

    hash_wrapper.update_with_ring_element_slice(&rc_opening.most_inner_commitment());

    let mut projection_matrix =
        ProjectionMatrix::new(config.projection_ratio, config.projection_height);

    projection_matrix.sample(&mut hash_wrapper);

    #[cfg(feature = "debug-decomp")]
    let mut dbg_coarse_image: Option<VerticallyAlignedMatrix<RingElement>> = None;
    let rc_coarse_projection = match &config.projection_recursion {
        Projection::Coarse(proj_config) => {
            let t2 = std::time::Instant::now();
            let witness_i16 = match &commitment_with_aux.witness_i16 {
                Some(witness_i16) => witness_i16,
                None => &prepare_i16_witness(witness),
            };
            let projection_image = project(witness_i16, &projection_matrix);
            #[cfg(feature = "debug-decomp")]
            {
                dbg_coarse_image = Some(projection_image.clone());
            }
            println!("  project: {} ms", t2.elapsed().as_millis());

            let t3 = std::time::Instant::now();
            let rc_coarse_projection = recursive_commit(&crs, &proj_config, &projection_image.data);
            println!("  rc_projection: {} ms", t3.elapsed().as_millis());

            hash_wrapper
                .update_with_ring_element_slice(&rc_coarse_projection.most_inner_commitment());
            Some(rc_coarse_projection)
        }
        _ => None,
    };

    let rc_fine_projection = match &config.projection_recursion {
        Projection::Fine(proj_config) => {
            let t2 = std::time::Instant::now();
            let projection_image_ct = project_coefficients(&witness, &projection_matrix);
            println!("  project_cf: {} ms", t2.elapsed().as_millis());
            let t3 = std::time::Instant::now();
            let rc_projection_ct = recursive_commit(
                &crs,
                &proj_config.recursion_constant_term,
                &projection_image_ct.data,
            );
            println!("  rc_projection_ct: {} ms", t3.elapsed().as_millis());

            hash_wrapper.update_with_ring_element_slice(&rc_projection_ct.most_inner_commitment());

            let t4 = std::time::Instant::now();
            let (projection_batched, fine_proj_batching_challenges) = batch_projection_n_times(
                &witness,
                &projection_matrix,
                &mut hash_wrapper,
                proj_config.nof_batches,
                false,
            );
            println!(
                "  batch_projection_n_times: {} ms",
                t4.elapsed().as_millis()
            );

            let t5 = std::time::Instant::now();
            let rc_projection_batched = recursive_commit(
                &crs,
                &proj_config.recursion_batched_projection,
                &projection_batched.data,
            );
            println!("  rc_projection_batched: {} ms", t5.elapsed().as_millis());
            hash_wrapper
                .update_with_ring_element_slice(&rc_projection_batched.most_inner_commitment());

            Some((
                rc_projection_ct,
                rc_projection_batched,
                fine_proj_batching_challenges,
            ))
        }
        _ => None,
    };

    if let Projection::Skip = &config.projection_recursion {
        println!("  Skipping projection recursion as per configuration. Likely the first round\n");
    }
    let mut fold_challenge = vec![RingElement::zero(Representation::IncompleteNTT); witness.width];

    hash_wrapper.sample_low_op_norm_ring_vec_into(&mut fold_challenge);

    let t4 = std::time::Instant::now();
    let folded_witness = fold(&witness, &fold_challenge);
    println!("  fold: {} ms", t4.elapsed().as_millis());

    #[cfg(feature = "debug-decomp")]
    if let Some(image) = &dbg_coarse_image {
        use crate::common::norms;

        let folded_image = fold(image, &vec![fold_challenge.clone(); 1].concat());
        let check = crate::protocol::project_coarse::project_ring(&folded_witness, &projection_matrix);
        let mismatch = check
            .data
            .iter()
            .zip(folded_image.data.iter())
            .filter(|(a, b)| a.v != b.v)
            .count();
        let input_norm = norms::inf_norm(&witness.data);
        println!("  [debug] input norm: {}", input_norm);
        assert_eq!(
            mismatch,
            0,
            "  [debug] coarse projection consistency: {} / {} mismatching rows",
            mismatch,
            check.data.len()
        );
        
    }

    let mut next_round_data = vec![RingElement::zero(Representation::IncompleteNTT); config.composed_witness_length];

    let t5 = std::time::Instant::now();
    let folded_witness_decomposed = decompose(
        &folded_witness.data,
        config.witness_decomposition_base_log as u64,
        config.witness_decomposition_chunks,
    );
    println!("  decompose: {} ms", t5.elapsed().as_millis());

    paste_by_prefix(
        &mut next_round_data,
        &folded_witness_decomposed,
        &config.folded_witness_prefix,
    );

    match &config.projection_recursion {
        Projection::Coarse(projection_config) => {
            paste_recursive_commitment(
                &mut next_round_data,
                &rc_coarse_projection.as_ref().unwrap(),
                &projection_config,
            );
        }
        Projection::Fine(projection_config) => {
            paste_recursive_commitment(
                &mut next_round_data,
                &rc_fine_projection.as_ref().unwrap().0,
                &projection_config.recursion_constant_term,
            );
            paste_recursive_commitment(
                &mut next_round_data,
                &rc_fine_projection.as_ref().unwrap().1,
                &projection_config.recursion_batched_projection,
            );
        }
        Projection::Skip => {
            // Do nothing
        }
    }

    paste_recursive_commitment(&mut next_round_data, &rc_opening, &config.opening_recursion);

    paste_recursive_commitment(
        &mut next_round_data,
        &rc_commitment,
        &config.commitment_recursion,
    );

    let t6 = std::time::Instant::now();

    let next_config_base = config.next.as_ref().map(|c| config_base_from_config(c));

    // Paper: U = reshape_{r'}(w-hat); the packed vector is the next round's witness.
    let next_round_witness = VerticallyAlignedMatrix {
        height: next_config_base.map_or(config.composed_witness_length, |c| c.witness_height()),
        width: next_config_base.map_or(1, |c| c.witness_width()),
        used_cols: next_config_base.map_or(1, |c| {
            (c.witness_width() as f64 * config.next_level_usage_ratio).ceil() as usize
        }),
        data: next_round_data,
    };

    #[cfg(feature = "debug-hardness")]
    crate::protocol::parties::debug_hardness::check_sumcheck_round(
        config,
        &next_round_witness.data,
        rc_commitment,
        &rc_opening,
        rc_coarse_projection.as_ref(),
        rc_fine_projection.as_ref().map(|(ct, b, _)| (ct, b)),
        next_config_base.map_or(1, |c| c.witness_width()),
    );

    // Commit to the next-round witness before the sumcheck, so the commitment
    // is bound by the transcript before any sumcheck challenge is sampled.
    let pending_commitment = config.next.as_deref().map(|next_config| {
        let base = config_base_from_config(next_config);
        assert_eq!(
            next_round_witness.data.len(),
            base.witness_height() * base.witness_width(),
            "composed length doesn't match the next round's witness dimensions"
        );
        let basic_commitment =
            commit_basic(&crs, &next_round_witness, base.basic_commitment_rank());
        match next_config {
            Config::Sumcheck(next_sumcheck_config) => {
                let rc = recursive_commit(
                    &crs,
                    &next_sumcheck_config.commitment_recursion,
                    &basic_commitment.data,
                );
                hash_wrapper.update_with_ring_element_slice(&rc.most_inner_commitment());
                println!(
                    "Next round commitment created of length {}.",
                    rc.committed_data.len()
                );
                PendingNextCommitment::Recursive(rc)
            }
            _ => {
                hash_wrapper.update_with_ring_element_slice(&basic_commitment.data);
                PendingNextCommitment::Basic(basic_commitment)
            }
        }
    });

    let sumcheck_output = sumcheck(
        &config,
        &next_round_witness.data,
        &projection_matrix,
        &fold_challenge,
        &rc_fine_projection
            .as_ref()
            .map(|(_, _, challenges)| challenges),
        &opening,
        sumcheck_context,
        &mut hash_wrapper,
    );

    println!("  sumcheck: {} ms", t6.elapsed().as_millis());

    let (
        claim_over_witness,
        claim_over_witness_conjugate,
        norm_claim,
        most_inner_norm_claim,
        sumcheck_transcript,
        evaluation_points,
        constant_term_claims,
    ) = sumcheck_output;

    // Recurse: the sumcheck evaluation point splits into (outer, inner) =
    // (c_0, c_1), which become the next round's evaluation points.
    let (next_proof, next_round_commitment) = match (config.next.as_deref(), pending_commitment) {
        (None, _) => (None, None),
        (
            Some(Config::Sumcheck(next_sumcheck_config)),
            Some(PendingNextCommitment::Recursive(rc)),
        ) => {
            let (points_outer, points_inner) =
                evaluation_points.split_at(next_sumcheck_config.witness_width.ilog2() as usize);
            let most_inner_commitment = rc.most_inner_commitment().clone();
            let next_commitment_with_aux = CommitmentWithAux::from_rc_commitment_with_aux(rc);
            let next_context = match sumcheck_context.next.as_deref_mut() {
                Some(NextSumcheckContext::Simple(next_ctx)) => next_ctx,
                _ => panic!("Expected NextSumcheckContext::Simple in sumcheck_context.next"),
            };
            let proof = prover_round(
                &crs,
                next_sumcheck_config,
                &next_commitment_with_aux,
                &next_round_witness,
                &evaluation_point_pair(points_inner),
                &evaluation_point_pair(points_outer),
                next_context,
                false,
                Some(hash_wrapper),
            )
            .0;
            (
                Some(RoundProof::Sumcheck(proof)),
                Some(NextRoundCommitment::Recursive(most_inner_commitment)),
            )
        }
        (
            Some(Config::Simple(next_simple_config)),
            Some(PendingNextCommitment::Basic(basic_commitment)),
        ) => {
            let (points_outer, points_inner) =
                evaluation_points.split_at(next_simple_config.witness_width.ilog2() as usize);
            let proof = prover_round_simple(
                next_simple_config,
                &basic_commitment,
                &next_round_witness,
                &evaluation_point_pair(points_inner),
                &evaluation_point_pair(points_outer),
                Some(hash_wrapper),
            );
            (
                Some(RoundProof::Simple(proof)),
                Some(NextRoundCommitment::Simple(basic_commitment)),
            )
        }
        (
            Some(Config::Intermediate(next_intermediate_config)),
            Some(PendingNextCommitment::Basic(basic_commitment)),
        ) => {
            let (points_outer, points_inner) = evaluation_points
                .split_at(next_intermediate_config.witness_width.ilog2() as usize);
            let next_context = match sumcheck_context.next.as_deref_mut() {
                Some(NextSumcheckContext::Intermediate(next_ctx)) => next_ctx,
                _ => panic!("Expected NextSumcheckContext::Intermediate in sumcheck_context.next"),
            };
            let proof = prover_round_intermediate(
                crs,
                next_intermediate_config,
                &basic_commitment,
                &next_round_witness,
                &evaluation_point_pair(points_inner),
                &evaluation_point_pair(points_outer),
                next_context,
                Some(hash_wrapper),
            );
            (
                Some(RoundProof::Intermediate(proof)),
                Some(NextRoundCommitment::Simple(basic_commitment)),
            )
        }
        _ => unreachable!("next config and pending commitment kinds must match"),
    };

    let rp = SumcheckRoundProof {
        polys: sumcheck_transcript,
        claim_over_witness,
        claim_over_witness_conjugate,
        norm_claim,
        most_inner_norm_claim,
        next_round_commitment,
        rc_opening_inner: rc_opening.most_inner_commitment().clone(),
        rc_coarse_projection_inner: rc_coarse_projection
            .as_ref()
            .map(|rc| rc.most_inner_commitment().clone()),
        rc_fine_projection_inner: rc_fine_projection.as_ref().map(|(rc_ct, rc_batched, _)| {
            (
                rc_ct.most_inner_commitment().clone(),
                rc_batched.most_inner_commitment().clone(),
            )
        }),
        constant_term_claims,
        next: next_proof.map(Box::new),
    };


    let elapsed = start.elapsed().as_nanos();
    println!("Prover: {} ns", elapsed);
    (rp, claims)
}

pub fn prover_round_intermediate(
    crs: &CRS,
    config: &IntermediateConfig,
    commitment: &BasicCommitment,
    witness: &VerticallyAlignedMatrix<RingElement>,
    evaluation_points_inner: &Vec<StructuredRow>,
    evaluation_points_outer: &Vec<StructuredRow>,
    sumcheck_context: &mut IntermediateSumcheckContext,
    hash_wrapper: Option<HashWrapper>,
) -> IntermediateRoundProof {
    println!("Prover intermediate round started.");
    let mut hash_wrapper = hash_wrapper.unwrap_or_else(HashWrapper::new);
    hash_wrapper.update_with_ring_element_slice(&commitment.data);

    let opening = open_at(
        &witness,
        &evaluation_points_inner,
        &evaluation_points_outer,
        true,
    );
    println!(
        "evaluation_points_inner length: {}, evaluation_points_outer length: {}",
        evaluation_points_inner.len(),
        evaluation_points_outer.len()
    );
    println!(
        "int opening height: {}, width: {}",
        opening.rhs.height, opening.rhs.width
    );

    hash_wrapper.update_with_ring_element_slice(&opening.rhs.data);

    let mut projection_matrix =
        ProjectionMatrix::new(config.projection_ratio, config.projection_height);
    projection_matrix.sample(&mut hash_wrapper);
    let projection_image_ct = project_coefficients(witness, &projection_matrix);
    hash_wrapper.update_with_ring_element_slice(&projection_image_ct.data);
    assert_eq!(
        config.projection_nof_batches, NOF_BATCHES,
        "projection_nof_batches must equal NOF_BATCHES"
    );
    let (batched_projection_image, fine_proj_batching_challenges) = batch_projection_n_times(
        witness,
        &projection_matrix,
        &mut hash_wrapper,
        config.projection_nof_batches,
        false,
    );

    hash_wrapper.update_with_ring_element_slice(&batched_projection_image.data);

    let mut fold_challenge =
        vec![RingElement::zero(Representation::IncompleteNTT); witness.width];
    hash_wrapper.sample_low_op_norm_ring_vec_into(&mut fold_challenge);

    let folded_witness = fold(&witness, &fold_challenge);

    let folded_witness_decomposed = decompose(
        &folded_witness.data,
        config.witness_decomposition_base_log as u64,
        config.witness_decomposition_chunks,
    );

    let next_config_base = config.next.as_ref().map(|c| config_base_from_config(c));

    println!("Creating next round commitment.");

    let next_round_config = next_config_base
        .as_ref()
        .expect("Intermediate round must have a next round config");
    let height = next_round_config.witness_height();
    let width = next_round_config.witness_width();
    let used_cols = next_round_config.witness_width();
    assert!(
        used_cols <= width,
        "next_round_witness used_cols ({}) exceeds width ({})",
        used_cols,
        width
    );
    assert_eq!(
        folded_witness_decomposed.len(),
        height * width,
        "next_round_witness data length ({}) does not match expected {}x{} ({})",
        folded_witness_decomposed.len(),
        height,
        width,
        height * width
    );
    let next_round_witness = VerticallyAlignedMatrix {
        height,
        width,
        used_cols,
        data: folded_witness_decomposed,
    };

    let next_round_commitment: HorizontallyAlignedMatrix<RingElement> = commit_basic(
        &crs,
        &next_round_witness,
        next_config_base
            .map(|c| c.basic_commitment_rank())
            .unwrap(),
    );
    hash_wrapper.update_with_ring_element_slice(&next_round_commitment.data);

    println!(
        "Next round commitment created of length {}.",
        next_round_commitment.data.len()
    );

    let (intermediate_sumcheck_proof, evaluation_points) = run_intermediate_sumcheck(
        config,
        &next_round_witness.data,
        evaluation_points_inner,
        &fine_proj_batching_challenges,
        sumcheck_context,
        &mut hash_wrapper,
    );

    #[cfg(feature = "debug-hardness")]
    crate::protocol::parties::debug_hardness::check_intermediate_round(
        config,
        &next_round_witness.data,
        &folded_witness.data,
        &projection_image_ct.data,
    );

    let next_level_proof = match &config.next {
        None => {
            unreachable!("Intermediate round proof should always have a next round config, as it is not the last round.")
        }
        Some(next_config) => match &next_config.as_ref() {
            Config::Sumcheck(_) => {
                unreachable!("Next round after intermediate round should be simple or intermediate, not sumcheck.")
            }
            Config::Intermediate(next_intermediate_config) => {
                let (new_evaluation_points_outer, new_evaluation_points_inner) = evaluation_points
                    .split_at(next_intermediate_config.witness_width.ilog2() as usize);
                let proof = prover_round_intermediate(
                    crs,
                    next_intermediate_config,
                    &next_round_commitment,
                    &next_round_witness,
                    &vec![
                        evaluation_point_to_structured_row(new_evaluation_points_inner),
                        evaluation_point_to_structured_row_conjugate(new_evaluation_points_inner),
                    ],
                    &vec![
                        evaluation_point_to_structured_row(new_evaluation_points_outer),
                        evaluation_point_to_structured_row_conjugate(new_evaluation_points_outer),
                    ],
                    sumcheck_context.next.as_deref_mut().unwrap(),
                    Some(hash_wrapper),
                );
                RoundProof::Intermediate(proof)
            }
            Config::Simple(next_simple_config) => {
                let (new_evaluation_points_outer, new_evaluation_points_inner) =
                    evaluation_points.split_at(next_simple_config.witness_width.ilog2() as usize);
                let proof = prover_round_simple(
                    next_simple_config,
                    &next_round_commitment,
                    &next_round_witness,
                    &vec![
                        evaluation_point_to_structured_row(new_evaluation_points_inner),
                        evaluation_point_to_structured_row_conjugate(new_evaluation_points_inner),
                    ],
                    &vec![
                        evaluation_point_to_structured_row(new_evaluation_points_outer),
                        evaluation_point_to_structured_row_conjugate(new_evaluation_points_outer),
                    ],
                    Some(hash_wrapper),
                );
                RoundProof::Simple(proof)
            }
        },
    };

    IntermediateRoundProof {
        opening_rhs: opening.rhs,
        polys: intermediate_sumcheck_proof.polys,
        claim_over_witness: intermediate_sumcheck_proof.claim_over_witness,
        claim_over_witness_conjugate: intermediate_sumcheck_proof.claim_over_witness_conjugate,
        norm_claim: intermediate_sumcheck_proof.norm_claim,
        next_round_commitment: Some(NextRoundCommitment::Simple(next_round_commitment)),
        projection_image_ct,
        batched_projection_image,
        next: Some(Box::new(next_level_proof)),
    }
}

// this is only for the last round
pub fn prover_round_simple(
    config: &SimpleConfig,
    commitment: &BasicCommitment,
    witness: &VerticallyAlignedMatrix<RingElement>,
    evaluation_points_inner: &Vec<StructuredRow>,
    evaluation_points_outer: &Vec<StructuredRow>,
    hash_wrapper: Option<HashWrapper>,
) -> SimpleRoundProof {
    println!("Prover simple round started.");
    let mut hash_wrapper = hash_wrapper.unwrap_or_else(HashWrapper::new);

    hash_wrapper.update_with_ring_element_slice(&commitment.data);

    let opening = open_at(
        &witness,
        &evaluation_points_inner,
        &evaluation_points_outer,
        true,
    );
    println!(
        "evaluation_points_inner length: {}, evaluation_points_outer length: {}",
        evaluation_points_inner.len(),
        evaluation_points_outer.len()
    );
    println!(
        "opening height: {}, width: {}",
        opening.rhs.height, opening.rhs.width
    );

    hash_wrapper.update_with_ring_element_slice(&opening.rhs.data);

    let mut projection_matrix =
        ProjectionMatrix::new(config.projection_ratio, config.projection_height);

    projection_matrix.sample(&mut hash_wrapper);

    let projection_image_ct = project_coefficients(&witness, &projection_matrix);

    hash_wrapper.update_with_ring_element_slice(&projection_image_ct.data);
    // let projection_image = project(&witness, &projection_matrix);

    let (batched_projection_image, _) = batch_projection_n_times(
        // we don't need the challenges here
        &witness,
        &projection_matrix,
        &mut hash_wrapper,
        config.projection_nof_batches,
        true,
    );

    // let projection_image = project(&witness, &projection_matrix);

    hash_wrapper.update_with_ring_element_slice(&batched_projection_image.data);

    let mut fold_challenge = vec![RingElement::zero(Representation::IncompleteNTT); witness.width];

    hash_wrapper.sample_low_op_norm_ring_vec_into(&mut fold_challenge);

    let folded_witness = fold(&witness, &fold_challenge);

    #[cfg(feature = "debug-hardness")]
    crate::protocol::parties::debug_hardness::check_simple_round(
        config,
        &folded_witness.data,
        &projection_image_ct.data,
    );

    SimpleRoundProof {
        folded_witness,
        projection_image_ct,
        batched_projection_image,
        opening_rhs: opening.rhs,
    }
}
