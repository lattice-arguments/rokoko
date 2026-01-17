use std::{process::exit, sync::LazyLock};

use num::range;

use crate::{
    common::{
        arithmetic::inner_product,
        config::{self, MOD_Q},
        decomposition::{compose_from_decomposed, decompose},
        hash::HashWrapper,
        matrix::{new_vec_zero_preallocated, HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        norms,
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{Representation, RingElement},
        sampling::sample_random_short_vector,
        structured_row::{self, PreprocessedRow, StructuredRow},
    },
    protocol::{
        commitment::{
            commit_basic, commit_basic_internal, recursive_commit, RecursiveCommitmentWithAux,
        },
        config::{paste_by_prefix, paste_recursive_commitment, Config, Projection, CONFIG},
        crs::{CK, CRS},
        fold::fold,
        open::{evaluation_point_to_structured_row, open_at},
        prefix::check_prefixing_correctness,
        project::project,
        project_2::{batch_projection_n_times, project_coefficients},
        proof::Proof,
        sumcheck::{self, init_sumcheck, sumcheck, SumcheckContext},
        sumcheck_utils::{
            common::{EvaluationSumcheckData, HighOrderSumcheckData, SumcheckBaseData},
            linear::{LinearSumcheck, StructuredRowEvaluationLinearSumcheck},
        },
        sumchecks::{
            builder_verifier::init_verifier,
            context_verifier::VerifierSumcheckContext,
            helpers::projection_coefficients,
            runner_verifier::{sumcheck_verifier, RoundProof},
        }, // sumcheck::sumcheck,
    },
};

pub fn claims(
    rhs: &HorizontallyAlignedMatrix<RingElement>,
    evaluation_points_outer: &Vec<StructuredRow>,
) -> Vec<RingElement> {
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    let mut result = new_vec_zero_preallocated(rhs.height);
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

pub fn prover_round(
    crs: &CRS,
    config: &Config,
    rc_commitment: &RecursiveCommitmentWithAux,
    witness: &VerticallyAlignedMatrix<RingElement>,
    evaluation_points_inner: &Vec<StructuredRow>,
    evaluation_points_outer: &Vec<StructuredRow>,
    sumcheck_context: &mut SumcheckContext,
    with_claims: bool,
) -> (RoundProof, Option<Vec<RingElement>>) {
    let mut hash_wrapper = HashWrapper::new();

    let start = std::time::Instant::now();
    hash_wrapper.update_with_ring_element_slice(&rc_commitment.most_inner_commitment());

    let t0 = std::time::Instant::now();
    let opening = open_at(&witness, &evaluation_points_inner, &evaluation_points_outer);

    let claims = if with_claims {
        let cc = claims(&opening.rhs, evaluation_points_outer);
        // assert_eq!(cc[0], opening);
        Some(cc)
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

    let rc_projection_image = match &config.projection_recursion {
        Projection::Type0(proj_config) => {
            let t2 = std::time::Instant::now();
            let projection_image = project(&witness, &projection_matrix);
            println!("  project: {} ms", t2.elapsed().as_millis());

            let t3 = std::time::Instant::now();
            let rc_projection_image = recursive_commit(&crs, &proj_config, &projection_image.data);
            println!("  rc_projection: {} ms", t3.elapsed().as_millis());

            hash_wrapper
                .update_with_ring_element_slice(&rc_projection_image.most_inner_commitment());
            Some(rc_projection_image)
        }
        _ => None,
    };

    let rcs_projection_1 = match &config.projection_recursion {
        Projection::Type1(proj_config) => {
            // TODO implement Type1 projection recursion
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
            let (projection_batched, challenges_batching_projection_1) = batch_projection_n_times(
                &witness,
                &projection_matrix,
                &mut hash_wrapper,
                proj_config.nof_batches,
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
                challenges_batching_projection_1,
            ))
        }
        _ => None,
    };
    let mut fold_challenge = vec![RingElement::zero(Representation::IncompleteNTT); witness.width];

    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut fold_challenge);

    let t4 = std::time::Instant::now();
    let folded_witness = fold(&witness, &fold_challenge);
    println!("  fold: {} ms", t4.elapsed().as_millis());

    let mut next_round_data = new_vec_zero_preallocated(config.composed_witness_length);

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
        Projection::Type0(projection_config) => {
            paste_recursive_commitment(
                &mut next_round_data,
                &rc_projection_image.as_ref().unwrap(),
                &projection_config,
            );
        }
        Projection::Type1(projection_config) => {
            paste_recursive_commitment(
                &mut next_round_data,
                &rcs_projection_1.as_ref().unwrap().0,
                &projection_config.recursion_constant_term,
            );
            paste_recursive_commitment(
                &mut next_round_data,
                &rcs_projection_1.as_ref().unwrap().1,
                &projection_config.recursion_batched_projection,
            );
        }
    }

    println!("Pasting opening_recursion commitments.");
    paste_recursive_commitment(&mut next_round_data, &rc_opening, &config.opening_recursion);
    println!("Pasting commitment_recursion commitments.");

    paste_recursive_commitment(
        &mut next_round_data,
        &rc_commitment,
        &config.commitment_recursion,
    );
    println!("Pasting done.");

    let ell_inf_norm = norms::inf_norm(&next_round_data);
    let ell_2_norm = norms::l2_norm(&next_round_data);

    println!(
        "Next round data norms: L_inf = {}, bit_len = {}, L_2 = {}, MOD_Q = {}",
        ell_inf_norm,
        ell_inf_norm.ilog2(),
        ell_2_norm,
        MOD_Q
    );

    assert!(
        ell_2_norm * ell_2_norm < (MOD_Q as f64 / 2f64),
        "norm too large, aborting"
    );

    let t6 = std::time::Instant::now();

    let next_round_witness = VerticallyAlignedMatrix {
        height: if let Some(next_config) = &config.next {
            next_config.witness_height
        } else {
            config.composed_witness_length // do nothing further
        },
        width: if let Some(next_config) = &config.next {
            next_config.witness_width
        } else {
            1 // do nothing further
        },
        data: next_round_data,
    };

    let next_round_rc_commitment_with_aux = if let Some(next_config) = &config.next {
        assert_eq!(
            next_round_witness.data.len(),
            next_config.witness_height * next_config.witness_width
        );
        let basic_commitment =
            commit_basic(&crs, &next_round_witness, next_config.basic_commitment_rank);

        println!(
            "Next round basic commitment created of width {} and height {}.",
            basic_commitment.width, basic_commitment.height
        );

        let rc_commitment = recursive_commit(
            &crs,
            &next_config.commitment_recursion,
            &basic_commitment.data,
        );

        println!(
            "Next round commitment created of length {}.",
            rc_commitment.committed_data.len()
        );

        Some(rc_commitment)
    } else {
        None
    };

    let next_round_commitment = next_round_rc_commitment_with_aux
        .as_ref()
        .map(|rc_commitment_with_aux| rc_commitment_with_aux.most_inner_commitment().clone());

    let (
        claim_over_witness,
        claim_over_witness_conjugate,
        norm_claim,
        sumcheck_transcript,
        evaluation_points,
        constant_term_claims,
    ) = sumcheck(
        &config,
        &next_round_witness.data,
        &projection_matrix,
        &fold_challenge,
        &rcs_projection_1
            .as_ref()
            .map(|(_, _, challenges)| challenges),
        &opening,
        sumcheck_context,
        &mut hash_wrapper,
    );
    println!("  sumcheck: {} ms", t6.elapsed().as_millis());

    assert!(
        ell_2_norm * ell_2_norm < (MOD_Q as f64 / 2f64),
        "norm too large, aborting"
    );

    let next_level_proof = match next_round_rc_commitment_with_aux {
        None => None,
        Some(rc_commitment_with_aux) => {
            let (new_evaluation_points_outer, new_evaluation_points_inner) = evaluation_points
                .split_at(config.next.as_ref().unwrap().witness_width.ilog2() as usize);
            Some(
                prover_round(
                    &crs,
                    config.next.as_ref().unwrap(),
                    &rc_commitment_with_aux,
                    &next_round_witness,
                    &vec![
                        evaluation_point_to_structured_row(&new_evaluation_points_inner.to_vec()),
                        evaluation_point_to_structured_row(
                            &new_evaluation_points_inner
                                .iter()
                                .map(|f| {
                                    let mut f = f.clone();
                                    f.conjugate_in_place();
                                    f
                                })
                                .collect::<Vec<_>>(),
                        ),
                    ],
                    &vec![
                        evaluation_point_to_structured_row(&new_evaluation_points_outer.to_vec()),
                        evaluation_point_to_structured_row(
                            &new_evaluation_points_outer
                                .iter()
                                .map(|f| {
                                    let mut f = f.clone();
                                    f.conjugate_in_place();
                                    f
                                })
                                .collect::<Vec<_>>(),
                        ),
                    ],
                    sumcheck_context.next.as_mut().unwrap(),
                    false,
                )
                .0,
            )
        }
    };

    let rp = RoundProof {
        polys: sumcheck_transcript,
        claim_over_witness: claim_over_witness,
        claim_over_witness_conjugate: claim_over_witness_conjugate,
        norm_claim: norm_claim,
        next_round_commitment,
        rc_opening_inner: rc_opening.most_inner_commitment().clone(),
        rc_projection_inner: rc_projection_image
            .as_ref()
            .map(|rc| rc.most_inner_commitment().clone()),
        rcs_projection_1_inner: rcs_projection_1.as_ref().map(|(rc_ct, rc_batched, _)| {
            (
                rc_ct.most_inner_commitment().clone(),
                rc_batched.most_inner_commitment().clone(),
            )
        }),
        constant_term_claims,
        next: next_level_proof.map(Box::new),
    };

    let elapsed = start.elapsed().as_nanos();
    println!("Prover: {} ns", elapsed);
    (rp, claims)
}
