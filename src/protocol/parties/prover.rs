use crate::common::config::*;
use crate::protocol::config::{RoundConfig, SalsaaProof, SalsaaProofCommon};
use crate::protocol::parties::executor::compute_ip_vdf_claim;
use crate::protocol::project::BatchingChallenges;
use crate::protocol::project_2::{batch_projection_n_times, project_coefficients};
use crate::protocol::sumcheck_utils::common::HighOrderSumcheckData;
use crate::protocol::sumcheck_utils::polynomial::Polynomial;
use crate::protocol::sumchecks::context::ProverSumcheckContext;
use crate::{
    common::{
        arithmetic::{field_to_ring_element_into, ALL_ONE_COEFFS, ONE, ZERO},
        config::NOF_BATCHES,
        decomposition::{compose_from_decomposed, decompose_chunks_into},
        hash::HashWrapper,
        matrix::{new_vec_zero_preallocated, HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
        sumcheck_element::SumcheckElement,
    },
    protocol::{
        commitment::{commit_basic, commit_basic_internal, BasicCommitment, Prefix},
        config::paste_by_prefix,
        crs::{self, CRS},
        fold::fold,
        open::{claim, evaluation_point_to_structured_row},
        project::{self, prepare_i16_witness, project},
    },
};

const VDF_MATRIX_HEIGHT: usize = 4;
pub struct vdf_crs {
    pub A: HorizontallyAlignedMatrix<RingElement>,
}

pub fn prover_round(
    crs: &CRS,
    witness: &VerticallyAlignedMatrix<RingElement>,
    config: &RoundConfig,
    sumcheck_context: &mut ProverSumcheckContext,
    evaluation_points_inner: &Vec<StructuredRow>,
    claims: &HorizontallyAlignedMatrix<RingElement>,
    // evaluation_points_outer: &Vec<StructuredRow>,
    hash_wrapper: &mut HashWrapper,
    vdf_params: Option<(
        &[RingElement; VDF_MATRIX_HEIGHT],
        &[RingElement; VDF_MATRIX_HEIGHT],
        &vdf_crs,
    )>, // (y_0, y_t, crs) - only for first round
) -> SalsaaProof {
    let (projection_matrix, projection_commitment, projected_witness, batching_challenges) =
        match config {
            RoundConfig::Intermediate { .. } => {
                let witness_16 = prepare_i16_witness(witness);

                let mut projection_matrix = ProjectionMatrix::new(witness.width, 256);

                projection_matrix.sample(hash_wrapper);

                let mut projected_witness = project(&witness_16, &projection_matrix);

                // The projection procent r columns into r columns with less rows, so we rearrange the projected witness taking advantage of the vertical alignment
                projected_witness.width = 1;
                projected_witness.used_cols = 1;
                projected_witness.height = witness.height;

                let projection_commitment = commit_basic(crs, &projected_witness, RANK);

                let batching_challenges = BatchingChallenges::sample(config, hash_wrapper);

                (
                    Some(projection_matrix),
                    Some(projection_commitment),
                    Some(projected_witness),
                    Some(batching_challenges),
                )
            }
            _ => (None, None, None, None),
        };

    let (
        unstructured_projection_matrix,
        batched_image,
        unstructured_batching_challenges,
        projection_ct,
    ) = match config {
        RoundConfig::IntermediateUnstructured {
            projection_ratio, ..
        }
        | RoundConfig::Last {
            projection_ratio, ..
        } => {
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
            let projection = project_coefficients(witness, &projection_matrix);
            let (batched_image, challenges) = batch_projection_n_times(
                witness,
                &projection_matrix,
                hash_wrapper,
                NOF_BATCHES,
                false,
            );
            (
                Some(projection_matrix),
                Some(batched_image),
                Some(challenges),
                Some(projection),
            )
        }
        _ => (None, None, None, None),
    };

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

    let ip_l2_claim = if config.l2 {
        let mut temp = RingElement::zero(Representation::IncompleteNTT);
        let mut claim = RingElement::zero(Representation::IncompleteNTT);
        for (w, wc) in witness.data.iter().zip(witness_conjugated.iter()) {
            temp *= (w, wc);
            claim += &temp;
        }
        Some(claim)
    } else {
        None
    };

    let ip_linf_claim = if config.exact_binariness {
        let mut temp = RingElement::zero(Representation::IncompleteNTT);
        let mut claim = RingElement::zero(Representation::IncompleteNTT);
        for (w, wc) in witness.data.iter().zip(witness_conjugated.iter()) {
            temp -= (&*ALL_ONE_COEFFS, w);
            temp *= wc;
            claim += &temp;
        }
        Some(claim)
    } else {
        None
    };

    paste_by_prefix(
        &mut extended_witness,
        &witness.data,
        &config.main_witness_prefix,
    );

    match config {
        RoundConfig::Intermediate {
            projection_prefix, ..
        } => {
            paste_by_prefix(
                &mut extended_witness,
                &projected_witness.as_ref().unwrap().data,
                projection_prefix,
            );
        }
        _ => {}
    }

    let mut evaluation_points_outer = new_vec_zero_preallocated(config.main_witness_columns);
    hash_wrapper.sample_ring_element_vec_into(&mut evaluation_points_outer);

    sumcheck_context.load_data(
        &extended_witness,
        &witness_conjugated,
        evaluation_points_inner,
        &evaluation_points_outer,
        &projection_matrix,
        &batching_challenges,
        &unstructured_batching_challenges,
        vdf_challenge.as_ref(),
        vdf_params.map(|(_, _, crs)| crs),
    );

    // Sample random batching coefficients from Fiat-Shamir
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

    if DEBUG {
        let ip_vdf_claim = compute_ip_vdf_claim(config, vdf_challenge.as_ref(), vdf_params);

        if sumcheck_context.type1sumcheck.len() > 0 {
            let claim = sumcheck_context.type1sumcheck[0].output.borrow().claim();

            let mut expected_claim = ZERO.clone();
            for (c, r) in claims.row(0).iter().zip(evaluation_points_outer.iter()) {
                expected_claim += &(c * r);
            }
            assert_eq!(claim, expected_claim, "Claim from the sumcheck does not match the expected claim computed from the committed witness and the evaluation points");
        }

        match config {
            RoundConfig::Intermediate { .. } => {
                let projection_claim = sumcheck_context
                    .type3sumcheck
                    .as_ref()
                    .unwrap()
                    .output
                    .borrow()
                    .claim();
                let expected_projection_claim = ZERO.clone();
                assert_eq!(
                projection_claim, expected_projection_claim,
                "Projection claim from the sumcheck does not match the expected projection claim"
            );
            }
            _ => {}
        };

        if config.l2 {
            let l2_claim = sumcheck_context
                .l2sumcheck
                .as_ref()
                .unwrap()
                .output
                .borrow()
                .claim();
            assert_eq!(
                l2_claim, ip_l2_claim.clone().unwrap(),
                "L2 claim from the projection sumcheck does not match the expected l2 claim computed from the witness"
            );
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
            assert_eq!(ct, 0, "Linf claim from the projection sumcheck is not zero, which means that the witness is not exactly binary as expected");

            assert_eq!(
                linf_claim, ip_linf_claim.clone().unwrap(),
                "Linf claim from the projection sumcheck does not match the expected linf claim computed from the witness"
            );
        }

        if config.vdf {
            let vdf_claim = sumcheck_context
                .vdfsumcheck
                .as_ref()
                .unwrap()
                .output
                .borrow()
                .claim();

            assert_eq!(
                vdf_claim,
                ip_vdf_claim.clone().unwrap(),
                "VDF claim from the sumcheck does not match the expected VDF claim"
            );
        }

        match config {
            RoundConfig::IntermediateUnstructured { .. } => {
                for (batch_idx, type31) in sumcheck_context
                    .type31sumchecks
                    .as_ref()
                    .unwrap()
                    .iter()
                    .enumerate()
                {
                    let projection_claim = type31.output.borrow().claim();
                    let batch_image = &batched_image.as_ref().unwrap().row(batch_idx);
                    let challenges =
                        &unstructured_batching_challenges.as_ref().unwrap()[batch_idx].c_2_values;
                    let mut expected_projection_claim =
                        RingElement::zero(Representation::IncompleteNTT);
                    let mut temp = RingElement::zero(Representation::IncompleteNTT);
                    for (c, r) in batch_image.iter().zip(challenges.iter()) {
                        temp *= (c, &RingElement::constant(*r, Representation::IncompleteNTT));
                        expected_projection_claim += &temp;
                    }
                    assert_eq!(
                        projection_claim, expected_projection_claim,
                        "Projection claim from the sumcheck does not match the expected projection claim"
                    );
                }
                println!("Unstructured projection claims from the sumcheck match the expected projection claims");
            }
            _ => {}
        }
    }

    let mut num_vars = sumcheck_context.combiner.borrow().variable_count();

    let mut time_poly = 0u128;
    let mut time_eval = 0u128;
    let mut evaluation_points = Vec::new();
    let mut polys = Vec::new();

    while num_vars > 0 {
        num_vars -= 1;

        let t1 = std::time::Instant::now();
        let mut poly_over_field = Polynomial::<QuadraticExtension>::new(0);

        sumcheck_context
            .field_combiner
            .borrow_mut()
            .univariate_polynomial_into(&mut poly_over_field);
        time_poly += t1.elapsed().as_millis();

        hash_wrapper.update_with_quadratic_extension_slice(&poly_over_field.coefficients);

        let mut r = RingElement::zero(Representation::IncompleteNTT);
        let mut f = QuadraticExtension::zero();

        hash_wrapper.sample_field_element_into(&mut f);

        field_to_ring_element_into(&mut r, &f);
        r.from_homogenized_field_extensions_to_incomplete_ntt();

        evaluation_points.push(r.clone());

        let t2 = std::time::Instant::now();
        sumcheck_context.partial_evaluate_all(&r);
        time_eval += t2.elapsed().as_millis();

        polys.push(poly_over_field);
    }

    // Sumcheck rounds produce LS-first challenges; the rest of this prover flow
    // still expects the legacy outer-first (MS-style) view for slicing/splitting.
    evaluation_points.reverse();

    if DEBUG {
        println!(
            "Polynomial time: {:?} ms, Evaluation time: {:?} ms",
            time_poly, time_eval
        );
    }

    let outer_points_len =
        config.main_witness_columns.ilog2() as usize + config.main_witness_prefix.length;
    let evaluation_points_inner = evaluation_points
        .iter()
        .skip(outer_points_len)
        .cloned()
        .collect::<Vec<_>>();
    let mut preprocessed_evaluation_points_inner = PreprocessedRow::from_structured_row(
        &evaluation_point_to_structured_row(&evaluation_points_inner),
    );

    let mut temp = RingElement::zero(Representation::IncompleteNTT);

    let mut claims =
        HorizontallyAlignedMatrix::new_zero_preallocated(2, config.main_witness_columns);

    let mut claim_over_projection = match config {
        RoundConfig::Intermediate { .. } => Some(new_vec_zero_preallocated(2)),
        _ => None,
    };

    for i in 0..config.main_witness_columns {
        for (w, r) in witness
            .col(i)
            .iter()
            .zip(preprocessed_evaluation_points_inner.preprocessed_row.iter())
        {
            temp *= (w, r);
            claims[(0, i)] += &temp;
        }
    }

    match config {
        RoundConfig::Intermediate { .. } => {
            for (c, r) in projected_witness
                .as_ref()
                .unwrap()
                .data
                .iter()
                .zip(preprocessed_evaluation_points_inner.preprocessed_row.iter())
            {
                temp *= (c, r);
                claim_over_projection.as_mut().unwrap()[0] += &temp;
            }
        }
        _ => {}
    }

    // now let's conjugate eval point in place and repeat the logic to get the claims for the conjugated witness, which will be used in the l2 and linf sumchecks
    for r in preprocessed_evaluation_points_inner
        .preprocessed_row
        .iter_mut()
    {
        r.conjugate_in_place();
    }

    for i in 0..witness.width {
        for (w, r) in witness
            .col(i)
            .iter()
            .zip(preprocessed_evaluation_points_inner.preprocessed_row.iter())
        {
            temp *= (w, r);
            claims[(1, i)] += &temp;
        }
    }

    match config {
        RoundConfig::Intermediate { .. } => {
            for (c, r) in projected_witness
                .as_ref()
                .unwrap()
                .data
                .iter()
                .zip(preprocessed_evaluation_points_inner.preprocessed_row.iter())
            {
                temp *= (c, r);
                claim_over_projection.as_mut().unwrap()[1] += &temp;
            }
        }
        _ => {}
    }

    // for i in 0..config.main_witness_columns {
    //     claims[(1, i)].conjugate_in_place(); // we had evals over conjugated witness, now we have conjugated evals over a regular witness
    // }

    // // we have conjugated claims for completeness (TODO: do we really need them?)
    // for (c, r) in claim_over_projection.iter().zip(preprocessed_evaluation_points_inner.preprocessed_row.iter_mut()) {
    //     r.conjugate_in_place();
    //     temp *= (c, r);
    //     claim_over_projection[1] += &temp;
    // }

    let mut folding_challenges = new_vec_zero_preallocated(config.main_witness_columns);
    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut folding_challenges);

    let folded_witness = fold(&witness, &folding_challenges);

    let common = SalsaaProofCommon {
        // projection_commitment,
        ip_l2_claim,
        ip_linf_claim,
        sumcheck_transcript: polys,
        claims,
        // claim_over_projection,
    };

    match config {
        RoundConfig::Intermediate {
            decomposition_base_log,
            next,
            ..
        } => {
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
                assert_eq!(composed, commitment_to_folded_witness[(0, 0)], "Composed commitment from the split witness does not match the commitment to the folded witness");
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
                &mut decomposed_split_witness.data
                    [split_witness.height * 2..split_witness.height * 4],
                &split_witness.data[split_witness.height..],
                *decomposition_base_log,
                2,
            );

            decompose_chunks_into(
                &mut decomposed_split_witness.data
                    [split_witness.height * 4..split_witness.height * 6],
                &projected_witness.as_ref().unwrap().data[..split_witness.height],
                *decomposition_base_log,
                2,
            );

            decompose_chunks_into(
                &mut decomposed_split_witness.data[split_witness.height * 6..],
                &projected_witness.as_ref().unwrap().data[split_witness.height..],
                *decomposition_base_log,
                2,
            );

            let decomposed_split_commitment = commit_basic(crs, &decomposed_split_witness, RANK);

            if DEBUG {
                let commitment_to_split_witness = commit_basic(crs, &split_witness, RANK);
                let old_ck = crs.structured_ck_for_wit_dim(split_witness.height * 2);

                let composed = compose_from_decomposed(
                    &vec![
                        decomposed_split_commitment[(0, 0)].clone(),
                        decomposed_split_commitment[(0, 1)].clone(),
                        decomposed_split_commitment[(0, 2)].clone(),
                        decomposed_split_commitment[(0, 3)].clone(),
                    ],
                    *decomposition_base_log,
                    2,
                );

                assert_eq!(composed[0], commitment_to_split_witness[(0, 0)], "Composed commitment from the decomposed split witness does not match the commitment to the split witness");

                assert_eq!(composed[1], commitment_to_split_witness[(0, 1)], "Composed commitment from the decomposed split projected witness does not match the commitment to the projected witness");

                let composed_projection = compose_from_decomposed(
                    &vec![
                        decomposed_split_commitment[(0, 4)].clone(),
                        decomposed_split_commitment[(0, 5)].clone(),
                        decomposed_split_commitment[(0, 6)].clone(),
                        decomposed_split_commitment[(0, 7)].clone(),
                    ],
                    *decomposition_base_log,
                    2,
                );

                let unsplit_projection = &(&(&*ONE - &old_ck[0].tensor_layers[0])
                    * &composed_projection[0])
                    + &(&old_ck[0].tensor_layers[0] * &composed_projection[1]);

                assert_eq!(unsplit_projection, projection_commitment.as_ref().unwrap()[(0, 0)], "Composed commitment from the decomposed split projected witness does not match the commitment to the projected witness");
            }

            let new_evaluation_points_inner = evaluation_points
                .iter()
                .skip(outer_points_len + 1)
                .cloned()
                .collect::<Vec<_>>();

            let new_evaluation_points_inner_expanded = PreprocessedRow::from_structured_row(
                &evaluation_point_to_structured_row(&new_evaluation_points_inner),
            );

            let new_evaluation_points_inner_conjugated = new_evaluation_points_inner
                .iter()
                .map(RingElement::conjugate)
                .collect::<Vec<_>>();

            let new_evaluation_points_inner_conjugated_expanded =
                PreprocessedRow::from_structured_row(&evaluation_point_to_structured_row(
                    &new_evaluation_points_inner_conjugated,
                ));

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
                None, // VDF only in first round
            );

            SalsaaProof::Intermediate {
                common,
                new_claims,
                decomposed_split_commitment,
                projection_commitment: projection_commitment.unwrap(),
                claim_over_projection: claim_over_projection.unwrap(),
                next: Box::new(next_level_proof),
            }
        }

        RoundConfig::IntermediateUnstructured {
            decomposition_base_log,
            next,
            ..
        } => {
            // Same as Intermediate but without projection columns:
            // fold → split → decompose → 4 columns (2 split × 2 decomp chunks)
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

            let new_evaluation_points_inner = evaluation_points
                .iter()
                .skip(outer_points_len + 1)
                .cloned()
                .collect::<Vec<_>>();

            let new_evaluation_points_inner_expanded = PreprocessedRow::from_structured_row(
                &evaluation_point_to_structured_row(&new_evaluation_points_inner),
            );

            let new_evaluation_points_inner_conjugated = new_evaluation_points_inner
                .iter()
                .map(RingElement::conjugate)
                .collect::<Vec<_>>();

            let new_evaluation_points_inner_conjugated_expanded =
                PreprocessedRow::from_structured_row(&evaluation_point_to_structured_row(
                    &new_evaluation_points_inner_conjugated,
                ));

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

            SalsaaProof::IntermediateUnstructured {
                common,
                new_claims: new_claims.data,
                decomposed_split_commitment,
                next: Box::new(next_level_proof),
                projection_image_ct: projection_ct.unwrap(),
                projection_image_batched: batched_image.unwrap(),
            }
        }

        RoundConfig::Last { .. } => {
            // Last round: send the folded witness and projected witness directly, no decomposition
            SalsaaProof::Last {
                common,
                folded_witness: folded_witness.data,
                projection_image_ct: projection_ct.unwrap(),
                projection_image_batched: batched_image.unwrap(),
            }
        }
    }
}
