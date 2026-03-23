use crate::{
    common::{
        decomposition::decompose,
        hash::HashWrapper,
        matrix::{new_vec_zero_preallocated, HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    protocol::{
        commitment::{commit_basic, recursive_commit, BasicCommitment, CommitmentWithAux},
        config::{
            paste_by_prefix, paste_recursive_commitment, Config, ConfigBase, NextRoundCommitment,
            Projection, RoundProof, SimpleConfig, SimpleRoundProof, SumcheckConfig,
            SumcheckRoundProof,
        },
        crs::CRS,
        fold::fold,
        open::{
            evaluation_point_to_structured_row, evaluation_point_to_structured_row_conjugate,
            open_at,
        },
        project::{prepare_i16_witness, project},
        project_2::{batch_projection_n_times, project_coefficients},
        sumcheck::{sumcheck, SumcheckContext},
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

fn config_base_from_config(config: &Config) -> &dyn ConfigBase {
    match config {
        Config::Sumcheck(sumcheck_config) => sumcheck_config,
        Config::Simple(simple_config) => simple_config,
    }
}

pub static mut ROUND_ID: usize = 0;

#[allow(dead_code)]
fn get_and_increment_round_id() -> usize {
    unsafe {
        let current_id = ROUND_ID;
        ROUND_ID += 1;
        current_id
    }
}

#[allow(dead_code)]
static DEBUG_HARDNESS_FROM_ROUND: usize = 0;

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
    let opening = open_at(&witness, &evaluation_points_inner, &evaluation_points_outer);

    let claims = if with_claims {
        Some(claims(&opening.rhs, evaluation_points_outer))
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
            let witness_i16 = match &commitment_with_aux.witness_i16 {
                Some(witness_i16) => witness_i16,
                None => &prepare_i16_witness(witness),
            };
            let projection_image = project(witness_i16, &projection_matrix);
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
                challenges_batching_projection_1,
            ))
        }
        _ => None,
    };

    match &config.projection_recursion {
        Projection::Skip => {
            println!(
                "  Skipping projection recursion as per configuration. Likely the first round\n"
            );
        }
        _ => {}
    }
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

    #[cfg(feature = "debug-hardness")]
    if get_and_increment_round_id() >= DEBUG_HARDNESS_FROM_ROUND {
        use crate::common::{
            config::{DEGREE, MOD_Q},
            estimator::{estimate_rsis_security, RSISParameters},
            norms,
        };
        use crate::protocol::commitment::{RecursionConfig, RecursiveCommitmentWithAux};

        println!("=== Debug Hardness Check ===");
        // let witness_norm = norms::l2_norm(&witness.data);
        // println!("Original witness L_2 norm: {}", witness_norm);

        let recommited_ell_inf_norm = norms::inf_norm(&next_round_data);
        let recommited_ell_2_norm = norms::l2_norm(&next_round_data);
        let recommited_ell_2_norm_sq = recommited_ell_2_norm.powf(2.0);

        let most_inner_commitment_data_ell_2 = {
            let commitment_data = &rc_commitment
                .most_inner_commitment_with_aux()
                .committed_data;
            let norm_commitment_data_ell_2_sq = norms::l2_norm(&commitment_data).powf(2.0) as u64;

            let opening_data = &rc_opening.most_inner_commitment_with_aux().committed_data;
            let norm_opening_data_ell_2_sq = norms::l2_norm(&opening_data).powf(2.0) as u64;

            let norm_projection_data_ell_2_sq = match &config.projection_recursion {
                Projection::Type0(_) => {
                    let rc_proj = rc_projection_image.as_ref().unwrap();
                    let proj_data = &rc_proj.most_inner_commitment_with_aux().committed_data;
                    norms::l2_norm(&proj_data).powf(2.0) as u64
                }
                Projection::Type1(_) => {
                    let (rc_ct, rc_batched, _) = rcs_projection_1.as_ref().unwrap();
                    let proj_ct_data = &rc_ct.most_inner_commitment_with_aux().committed_data;
                    let proj_batched_data =
                        &rc_batched.most_inner_commitment_with_aux().committed_data;
                    norms::l2_norm(&proj_ct_data).powf(2.0) as u64
                        + norms::l2_norm(&proj_batched_data).powf(2.0) as u64
                }
                Projection::Skip => 0,
            };
            ((norm_commitment_data_ell_2_sq
                + norm_opening_data_ell_2_sq
                + norm_projection_data_ell_2_sq) as f64)
                .sqrt()
        };
        println!(
            "Most inner commitment data L_2 norm: {}",
            most_inner_commitment_data_ell_2
        );

        // we subtract the most inner commitment data norm from the recommited norm to get the rest, including the witness
        let recommited_ell_2_norm_rest =
            (recommited_ell_2_norm.powf(2.0) - most_inner_commitment_data_ell_2.powf(2.0)).sqrt();

        fn debug_hardness_recursive_commitment(
            rc: &RecursiveCommitmentWithAux,
            config: &RecursionConfig,
            name: &str,
            extracted_norm: f64,
            extracted_norm_most_inner: f64,
            depth: usize,
        ) {
            let ell_inf_norm = norms::inf_norm(&rc.committed_data);
            let ell_2_norm = norms::l2_norm(&rc.committed_data);

            let current_extracted_norm = match config.next {
                Some(_) => extracted_norm,
                None => extracted_norm_most_inner,
            };

            let hardness = estimate_rsis_security(&RSISParameters {
                m: rc.committed_data.len() as u64,
                n: config.rank as u64,
                length_bound: current_extracted_norm.ceil() as u64,
            });
            let indent = "  ".repeat(depth);
            println!(
                "{}Recursive Commitment '{}' norms: L_2 = {}, bit_len = {}, MOD_Q = {} => estimated security for extraction: {:?}",
                indent,
                name,
                ell_2_norm,
                ell_inf_norm.ilog2(),
                MOD_Q,
                hardness,
            );

            if let (Some(next_rc), Some(next_config)) = (&rc.next, &config.next) {
                debug_hardness_recursive_commitment(
                    next_rc,
                    next_config,
                    name,
                    extracted_norm,
                    extracted_norm_most_inner,
                    depth + 1,
                );
            }
        }

        debug_hardness_recursive_commitment(
            rc_commitment,
            &config.commitment_recursion,
            "Commitment",
            recommited_ell_2_norm_rest,
            most_inner_commitment_data_ell_2,
            0,
        );

        debug_hardness_recursive_commitment(
            &rc_opening,
            &config.opening_recursion,
            "Opening",
            recommited_ell_2_norm_rest,
            most_inner_commitment_data_ell_2,
            0,
        );

        if let (Some(rc_projection_image), Projection::Type0(projection_config)) =
            (&rc_projection_image, &config.projection_recursion)
        {
            debug_hardness_recursive_commitment(
                rc_projection_image,
                projection_config,
                "Projection Image",
                recommited_ell_2_norm_rest,
                most_inner_commitment_data_ell_2,
                0,
            );
        }

        if let (Some(rcs_projection_1), Projection::Type1(projection_config)) =
            (&rcs_projection_1, &config.projection_recursion)
        {
            debug_hardness_recursive_commitment(
                &rcs_projection_1.0,
                &projection_config.recursion_constant_term,
                "Projection 1 Constant Term",
                recommited_ell_2_norm_rest,
                most_inner_commitment_data_ell_2,
                0,
            );
            debug_hardness_recursive_commitment(
                &rcs_projection_1.1,
                &projection_config.recursion_batched_projection,
                "Projection 1 Batched",
                recommited_ell_2_norm_rest,
                most_inner_commitment_data_ell_2,
                0,
            );
        }
        println!(
            "Next round data norms: L_inf = {}, bit_len = {}, L_2 = {}, MOD_Q = {}",
            recommited_ell_inf_norm,
            recommited_ell_inf_norm.ilog2(),
            recommited_ell_2_norm,
            MOD_Q
        );

        let next_level_width = if let Some(next_config) = &config.next {
            config_base_from_config(next_config).witness_width()
        } else {
            1
        };

        let recomposed_witness_bound = recommited_ell_2_norm_rest
            // * (next_level_width as f64).sqrt() // norm conversion as we get argument about columsn separately in extraction, TODO: I think it is not necessary
            * (config
                .witness_decomposition_base_log
                .pow((config.witness_decomposition_chunks - 1) as u32)) as f64;

        let extracted_witness_bound = recomposed_witness_bound * DEGREE as f64 * 8.0; // factor 4 for difference in numerator and denominator in extraction and 2 for ISIS to SIS

        let recomposed_projection_bound = match &config.projection_recursion {
            Projection::Type0(proj_config) => {
                recommited_ell_2_norm // we don't use rest here, as projection might live in the most inner commitment
                    * (proj_config
                        .decomposition_base_log
                        .pow((proj_config.decomposition_chunks - 1) as u32))
                        as f64
            }
            Projection::Type1(proj_config) => {
                recommited_ell_2_norm
                    * (proj_config
                        .recursion_constant_term
                        .decomposition_base_log
                        .pow((proj_config.recursion_constant_term.decomposition_chunks - 1) as u32))
                        as f64
            }
            Projection::Skip => 0.0, // not used
        };

        let argued_witness_bound = recomposed_projection_bound / 5.477f64; // sqrt(30) as in the paper

        let worse_bound = if extracted_witness_bound > argued_witness_bound {
            println!(
                "Using extracted witness bound {} for security estimation.",
                extracted_witness_bound
            );
            extracted_witness_bound
        } else {
            println!(
                "Using projection-argued witness bound {} for security estimation.",
                argued_witness_bound
            );
            argued_witness_bound
        };

        match &config.projection_recursion {
            Projection::Skip => {
                // do nothing as we cannot use inner product norm extraction anyway
            }
            _ => {
                assert!(
                    next_level_width as f64 * argued_witness_bound * argued_witness_bound
                        < (MOD_Q as f64 / 2f64),
                    "Witness bound too large for inner-product norm extraction!"
                );
            }
        }

        let basic_commitment_security = estimate_rsis_security(&RSISParameters {
            m: config.witness_height as u64,
            n: config.basic_commitment_rank as u64,
            length_bound: worse_bound.ceil() as u64,
        });
        println!(
            "Basic commitment estimated security for extraction: {:?} with rank {}",
            basic_commitment_security, config.basic_commitment_rank
        );
    }

    let t6 = std::time::Instant::now();

    let base_next_roung_config = config.next.as_ref().map(|c| config_base_from_config(c));

    let next_round_witness = VerticallyAlignedMatrix {
        height: if let Some(next_config) = &base_next_roung_config {
            next_config.witness_height()
        } else {
            config.composed_witness_length // do nothing further
        },
        width: if let Some(next_config) = &base_next_roung_config {
            next_config.witness_width()
        } else {
            1 // do nothing further
        },
        used_cols: if let Some(next_config) = &base_next_roung_config {
            (next_config.witness_width() as f64 * config.next_level_usage_ratio).ceil() as usize
        } else {
            1 // do nothing further
        },
        data: next_round_data,
    };

    let next_level_data = match &config.next {
        None => {
            let sumcheck_output = sumcheck(
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
            (None, sumcheck_output, None) // this should never happen, but we let it be for test purposes
        }
        Some(next_config) => {
            match &next_config.as_ref() {
                Config::Sumcheck(next_sumcheck_config) => {
                    // let next_round_rc_commitment_with_aux =
                    // if let Some(next_config) = &base_next_roung_config {
                    debug_assert_eq!(
                        next_round_witness.data.len(),
                        next_sumcheck_config.witness_height * next_sumcheck_config.witness_width
                    );
                    let basic_commitment = commit_basic(
                        &crs,
                        &next_round_witness,
                        next_sumcheck_config.basic_commitment_rank,
                    );

                    let next_round_rc_commitment_with_aux = recursive_commit(
                        &crs,
                        &next_sumcheck_config.commitment_recursion,
                        &basic_commitment.data,
                    );
                    hash_wrapper.update_with_ring_element_slice(
                        &next_round_rc_commitment_with_aux.most_inner_commitment(),
                    );

                    println!(
                        "Next round commitment created of length {}.",
                        next_round_rc_commitment_with_aux.committed_data.len()
                    );

                    let next_round_rc_commitment = next_round_rc_commitment_with_aux
                        .most_inner_commitment()
                        .clone();

                    let next_round_commitment_with_aux =
                        CommitmentWithAux::from_rc_commitment_with_aux(
                            next_round_rc_commitment_with_aux,
                        );

                    let sumcheck_output = sumcheck(
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
                    println!("Next round prover done.");

                    let evaluation_points = &sumcheck_output.5;

                    let (new_evaluation_points_outer, new_evaluation_points_inner) =
                        evaluation_points
                            .split_at(next_sumcheck_config.witness_width.ilog2() as usize);
                    (
                        Some(RoundProof::Sumcheck(
                            prover_round(
                                &crs,
                                next_sumcheck_config,
                                &next_round_commitment_with_aux,
                                &next_round_witness,
                                &vec![
                                    evaluation_point_to_structured_row(
                                        &new_evaluation_points_inner.to_vec(),
                                    ),
                                    evaluation_point_to_structured_row_conjugate(
                                        &new_evaluation_points_inner.to_vec(),
                                    ),
                                ],
                                &vec![
                                    evaluation_point_to_structured_row(
                                        &new_evaluation_points_outer.to_vec(),
                                    ),
                                    evaluation_point_to_structured_row_conjugate(
                                        &new_evaluation_points_outer.to_vec(),
                                    ),
                                ],
                                sumcheck_context.next.as_mut().unwrap(),
                                false,
                                Some(hash_wrapper),
                            )
                            .0,
                        )),
                        sumcheck_output,
                        Some(NextRoundCommitment::Recursive(next_round_rc_commitment)),
                    )
                }
                Config::Simple(next_simple_config) => {
                    debug_assert_eq!(
                        next_round_witness.data.len(),
                        next_simple_config.witness_height * next_simple_config.witness_width
                    );
                    let basic_commitment = commit_basic(
                        &crs,
                        &next_round_witness,
                        next_simple_config.basic_commitment_rank,
                    );

                    hash_wrapper.update_with_ring_element_slice(&basic_commitment.data);

                    let sumcheck_output = sumcheck(
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

                    let evaluation_points = &sumcheck_output.5;

                    let (new_evaluation_points_outer, new_evaluation_points_inner) =
                        evaluation_points
                            .split_at(next_simple_config.witness_width.ilog2() as usize);

                    (
                        Some(RoundProof::Simple(prover_round_simple(
                            next_simple_config,
                            &basic_commitment,
                            &next_round_witness,
                            &vec![
                                evaluation_point_to_structured_row(
                                    &new_evaluation_points_inner.to_vec(),
                                ),
                                evaluation_point_to_structured_row_conjugate(
                                    &new_evaluation_points_inner.to_vec(),
                                ),
                            ],
                            &vec![
                                evaluation_point_to_structured_row(
                                    &new_evaluation_points_outer.to_vec(),
                                ),
                                evaluation_point_to_structured_row_conjugate(
                                    &new_evaluation_points_outer.to_vec(),
                                ),
                            ],
                            Some(hash_wrapper),
                        ))),
                        sumcheck_output,
                        Some(NextRoundCommitment::Simple(basic_commitment)),
                    )
                }
            }
        }
    };

    println!("  sumcheck: {} ms", t6.elapsed().as_millis());

    let (
        claim_over_witness,
        claim_over_witness_conjugate,
        norm_claim,
        most_inner_norm_claim,
        sumcheck_transcript,
        _evaluation_points,
        constant_term_claims,
    ) = next_level_data.1;

    let rp = SumcheckRoundProof {
        polys: sumcheck_transcript,
        claim_over_witness: claim_over_witness,
        claim_over_witness_conjugate: claim_over_witness_conjugate,
        norm_claim: norm_claim,
        most_inner_norm_claim,
        next_round_commitment: next_level_data.2,
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
        next: next_level_data.0.map(Box::new),
    };

    let elapsed = start.elapsed().as_nanos();
    println!("Prover: {} ns", elapsed);
    (rp, claims)
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
    let mut hash_wrapper = hash_wrapper.unwrap_or_else(HashWrapper::new);

    hash_wrapper.update_with_ring_element_slice(&commitment.data);

    let opening = open_at(&witness, &evaluation_points_inner, &evaluation_points_outer);

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

    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut fold_challenge);

    let folded_witness = fold(&witness, &fold_challenge);

    #[cfg(feature = "debug-hardness")]
    {
        use crate::common::{
            config::{DEGREE, MOD_Q},
            estimator::{estimate_rsis_security, RSISParameters},
            norms,
        };
        let folded_witness_l2_norm = norms::l2_norm(&folded_witness.data);

        println!("Folded witness norm: {}", folded_witness_l2_norm);

        let projection_l2_norm = norms::l2_norm_coeffs(&projection_image_ct.data);

        let extracted_witness_bound = folded_witness_l2_norm * DEGREE as f64 * 8.0; // factor 4 for difference in numerator and denominator in extraction and 2 for ISIS to SIS

        let argued_witness_bound = projection_l2_norm / 5.477f64; // sqrt(30) as in the paper
        let worse_bound = if extracted_witness_bound > argued_witness_bound {
            println!(
                "Using extracted witness bound {} for security estimation.",
                extracted_witness_bound
            );
            extracted_witness_bound
        } else {
            println!(
                "Using projection-argued witness bound {} for security estimation.",
                argued_witness_bound
            );
            argued_witness_bound
        };

        let basic_commitment_security = estimate_rsis_security(&RSISParameters {
            m: config.witness_height as u64,
            n: config.basic_commitment_rank as u64,
            length_bound: worse_bound.ceil() as u64,
        });
        println!(
            "Basic commitment estimated security for extraction: {:?} with rank {}",
            basic_commitment_security, config.basic_commitment_rank
        );
    }

    SimpleRoundProof {
        folded_witness,
        projection_image_ct,
        batched_projection_image,
        opening_rhs: opening.rhs,
    }
}
