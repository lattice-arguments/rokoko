use crate::{
    common::{
        arithmetic::field_to_ring_element,
        config::{HALF_DEGREE, NOF_BATCHES},
        hash::HashWrapper,
        norms::assert_norm_bounded,
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        structured_row::StructuredRow,
        sumcheck_element::SumcheckElement,
    },
    protocol::{
        config::{NextRoundCommitment, Projection, SumcheckConfig, SumcheckRoundProof},
        open::evaluation_point_to_structured_row,
        project_fine::{
            verifier_sample_projection_challenges_collectively, BatchedProjectionChallengesSuccinct,
        },
        sumcheck_utils::common::EvaluationSumcheckData,
        sumchecks::{
            context_verifier::VerifierSumcheckContext, loader_verifier::load_verifier_sumcheck_data,
        },
    },
};

fn batch_claims(
    config: &SumcheckConfig,
    claims: &[RingElement],
    rc_commitment_inner: &[RingElement],
    rc_opening_inner: &[RingElement],
    rc_coarse_projection_inner: Option<&[RingElement]>,
    rc_fine_projection_inner: Option<(&[RingElement], &[RingElement])>,
    rcs_projection_1_constant_term_claims: Option<&[RingElement]>,
    norm_claim: &RingElement,
    most_inner_norm_claim: &RingElement,
    combination: &[RingElement],
) -> RingElement {
    let mut batched_claim = RingElement::zero(Representation::IncompleteNTT);
    let mut idx = 0;

    // CommitmentFold: zero claims (difference sumchecks)
    idx += config.basic_commitment_rank;

    // InnerEvalFold: zero claims (difference sumchecks)
    idx += config.nof_openings;

    // OuterEvalClaim: claims for evaluations
    for claim in claims.iter() {
        let mut weighted = claim.clone();
        weighted *= &combination[idx];
        batched_claim += &weighted;
        idx += 1;
    }

    if rc_coarse_projection_inner.is_some() {
        // CoarseProj: zero claim (difference sumcheck)
        idx += 1;
    }

    if rc_fine_projection_inner.is_some() {
        // FineProj-consistency: zero claim (difference sumcheck) + consistence between ct comm and bp comm
        for i in 0..NOF_BATCHES {
            idx += 1;
            let mut weighted = rcs_projection_1_constant_term_claims.as_ref().unwrap()[i].clone();
            weighted *= &combination[idx];
            batched_claim += &weighted;
            idx += 1;
        }
    }

    // ComVerify: Three recursion trees (commitment, opening, projection)
    // Each tree has: (layers with rank each) + (output layer with rank)
    for (recursion_idx, rc_inner) in [
        Some(rc_commitment_inner),
        Some(rc_opening_inner),
        rc_coarse_projection_inner,
        rc_fine_projection_inner.map(|(rc_ct, _)| rc_ct),
        rc_fine_projection_inner.map(|(_, rc_bp)| rc_bp),
    ]
    .iter()
    .enumerate()
    {
        if rc_inner.is_none() {
            continue;
        }
        let rc_inner = rc_inner.as_ref().unwrap();
        let recursion_config = match recursion_idx {
            0 => &config.commitment_recursion,
            1 => &config.opening_recursion,
            2 => match &config.projection_recursion {
                Projection::Coarse(proj_config) => proj_config,
                // Projection::Fine(proj_config) => &proj_config.recursion_constant_term,
                _ => unreachable!(),
            },
            3 => match &config.projection_recursion {
                Projection::Fine(proj_config) => &proj_config.recursion_constant_term,
                _ => unreachable!(),
            },
            4 => match &config.projection_recursion {
                Projection::Fine(proj_config) => &proj_config.recursion_batched_projection,
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };

        // Internal layers (zero claims)
        let mut current = recursion_config;
        while let Some(next) = current.next.as_deref() {
            idx += current.rank; // Each layer has rank outputs, all zero claims
            current = next;
        }

        // Output layer: rc_inner claims
        for rc_value in rc_inner.iter() {
            let mut weighted = rc_value.clone();
            weighted *= &combination[idx];
            batched_claim += &weighted;
            idx += 1;
        }
    }

    // NormCheck: norm claim
    let mut weighted_norm = norm_claim.clone();
    weighted_norm *= &combination[idx];
    batched_claim += &weighted_norm;
    idx += 1;

    let mut weighted_norm = most_inner_norm_claim.clone();
    weighted_norm *= &combination[idx];
    batched_claim += &weighted_norm;

    batched_claim
}

pub fn sumcheck_verifier(
    config: &SumcheckConfig,
    verifier_sumcheck_context: &mut VerifierSumcheckContext,
    rc_commitment: &[RingElement],
    round_proof: &SumcheckRoundProof,
    evaluation_points_inner: &[StructuredRow],
    evaluation_points_outer: &[StructuredRow],
    claims: &[RingElement],
    hash_wrapper: &mut HashWrapper,
) -> Vec<RingElement> {
    hash_wrapper.update_with_ring_element_slice(rc_commitment);
    hash_wrapper.update_with_ring_element_slice(&round_proof.rc_opening_inner);
    let mut projection_matrix =
        ProjectionMatrix::new(config.projection_ratio, config.projection_height);

    projection_matrix.sample(hash_wrapper);
    if let Some(rc_coarse_projection_inner) = &round_proof.rc_coarse_projection_inner {
        hash_wrapper.update_with_ring_element_slice(rc_coarse_projection_inner);
    }
    let challenges_3_1 = if let Some((rcs_projection_1_ct, rcs_projection_1_batched)) =
        &round_proof.rc_fine_projection_inner
    {
        hash_wrapper.update_with_ring_element_slice(rcs_projection_1_ct);
        let challenges_3_1: [BatchedProjectionChallengesSuccinct; NOF_BATCHES] =
            verifier_sample_projection_challenges_collectively(
                &projection_matrix,
                config,
                hash_wrapper,
            );
        hash_wrapper.update_with_ring_element_slice(rcs_projection_1_batched);
        Some(challenges_3_1)
    } else {
        None
    };

    let mut folding_challenges =
        vec![RingElement::zero(Representation::IncompleteNTT); config.witness_width];
    hash_wrapper.sample_low_op_norm_ring_vec_into(&mut folding_challenges);

    let projection_height_flat = config.witness_height / config.projection_ratio;

    if let Some(next_round_commitment) = round_proof.next_round_commitment.as_ref() {
        match &next_round_commitment {
            NextRoundCommitment::Recursive(recursive) => {
                hash_wrapper.update_with_ring_element_slice(&recursive);
            }
            NextRoundCommitment::Simple(basic_commitment) => {
                hash_wrapper.update_with_ring_element_slice(&basic_commitment.data);
            }
        }
    }

    let projection_matrix_flatter_structured = match config.projection_recursion {
        Projection::Coarse(_) => {
            let mut projection_matrix_flatter_base =
                vec![RingElement::zero(Representation::IncompleteNTT); projection_height_flat.ilog2() as usize];
            hash_wrapper
                .sample_ring_element_ntt_slots_same_vec_into(&mut projection_matrix_flatter_base);

            Some(evaluation_point_to_structured_row(
                &projection_matrix_flatter_base,
            ))
        }
        Projection::Fine(_) => None,
        Projection::Skip => None,
    };

    hash_wrapper.update_with_ring_element(&round_proof.norm_claim);
    hash_wrapper.update_with_ring_element(&round_proof.most_inner_norm_claim);
    if let Some(constant_term_claims) = &round_proof.constant_term_claims {
        hash_wrapper.update_with_ring_element_slice(constant_term_claims);
    }

    // Sample random batching coefficients from Fiat-Shamir
    let num_sumchecks = verifier_sumcheck_context
        .combiner_evaluation
        .borrow()
        .sumchecks_count();
    let mut combination = vec![RingElement::zero(Representation::IncompleteNTT); num_sumchecks];
    hash_wrapper.sample_ring_element_vec_into(&mut combination);

    let mut combination_to_field = RingElement::zero(Representation::IncompleteNTT);
    hash_wrapper.sample_ring_element_into(&mut combination_to_field);
    combination_to_field.from_incomplete_ntt_to_homogenized_field_extensions();
    let qe = combination_to_field.split_into_quadratic_extensions();

    // Batched claim must match the combiner's output order; see batch_claims.

    let batched_claim = batch_claims(
        config,
        claims,
        rc_commitment,
        &round_proof.rc_opening_inner,
        round_proof.rc_coarse_projection_inner.as_deref(),
        round_proof
            .rc_fine_projection_inner
            .as_ref()
            .map(|(a, b)| (a.as_slice(), b.as_slice())),
        round_proof.constant_term_claims.as_deref(),
        &round_proof.norm_claim,
        &round_proof.most_inner_norm_claim,
        &combination,
    );



    if let Some(constant_term_claims) = &round_proof.constant_term_claims {
        for ct_claim in constant_term_claims.iter() {
            let ct = ct_claim.constant_term_from_incomplete_ntt();
            assert_eq!(ct, 0);
        }
    }

    let norm_ct = round_proof.norm_claim.constant_term_from_incomplete_ntt();
    assert_norm_bounded(
        "norm claim via inner-product",
        (norm_ct as f64).sqrt(),
        config.norm_bound,
    );

    let most_inner_norm_ct = round_proof
        .most_inner_norm_claim
        .constant_term_from_incomplete_ntt();
    assert_norm_bounded(
        "most inner norm claim via inner-product",
        (most_inner_norm_ct as f64).sqrt(),
        config.most_inner_norm_bound,
    );

    let mut batched_claim_over_field = {
        let batched_claim = {
            let mut temp = batched_claim.clone();
            temp.from_incomplete_ntt_to_homogenized_field_extensions();
            temp
        };
        let mut temp = batched_claim.split_into_quadratic_extensions();
        let mut result = QuadraticExtension { coeffs: [0, 0] };
        for i in 0..HALF_DEGREE {
            temp[i] *= &qe[i];
            result += &temp[i];
        }
        result
    };

    let mut num_vars = round_proof.polys.len();

    let mut evaluation_points: Vec<QuadraticExtension> = vec![];
    while num_vars > 0 {
        num_vars -= 1;

        let poly_over_field = round_proof
            .polys
            .get(round_proof.polys.len() - num_vars - 1)
            .unwrap();

        hash_wrapper.update_with_quadratic_extension_slice(&poly_over_field.coefficients);

        assert_eq!(
            poly_over_field.at_zero() + poly_over_field.at_one(),
            batched_claim_over_field,
            "round-poly claim mismatch at witness_height={} num_vars_left={}",
            config.witness_height,
            num_vars
        );

        let mut f = QuadraticExtension::zero();

        hash_wrapper.sample_field_element_into(&mut f);

        batched_claim_over_field = poly_over_field.at(&f);
        evaluation_points.push(f);
    }

    load_verifier_sumcheck_data(
        verifier_sumcheck_context,
        &folding_challenges,
        &round_proof.claim_over_witness,
        &round_proof.claim_over_witness_conjugate,
        evaluation_points_inner,
        evaluation_points_outer,
        &projection_matrix,
        &projection_matrix_flatter_structured,
        &challenges_3_1, // for 1 projection type only
        &combination,
        &qe,
    );

    assert_eq!(
        &batched_claim_over_field,
        verifier_sumcheck_context
            .field_combiner_evaluation
            .borrow_mut()
            .evaluate(&evaluation_points)
    );

    hash_wrapper.update_with_ring_element(&round_proof.claim_over_witness);
    hash_wrapper.update_with_ring_element(&round_proof.claim_over_witness_conjugate);

    let eps = evaluation_points
        .iter()
        .rev()
        .map(|f| {
            let mut r = field_to_ring_element(f);
            r.from_homogenized_field_extensions_to_incomplete_ntt();
            r
        })
        .collect();

    eps
}
