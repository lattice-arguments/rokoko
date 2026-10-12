use crate::{
    common::{
        arithmetic::{field_to_ring_element, ONE},
        config::{MOD_Q, NOF_BATCHES},
        hash::HashWrapper,
        norms::assert_norm_bounded,
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{FieldExtension, Representation, RingElement},
        structured_row::StructuredRow,
        sumcheck_element::SumcheckElement,
    },
    hexl::bindings::multiply_mod,
    protocol::{
        config::{NextRoundCommitment, Projection, SumcheckConfig, SumcheckRoundProof},
        open::evaluation_point_to_structured_row,
        project_fine::{
            image_constant_term, verifier_sample_projection_challenges_collectively,
            verifier_terminal_projection_challenges, BatchedProjectionChallengesSuccinct,
        },
        sumcheck_utils::common::EvaluationSumcheckData,
        sumchecks::{
            batching::{
                field_terms, sample_batching, slot_form, slot_weights, Batching, FieldClaim,
                OuterClaim, TraceTerms,
            },
            builder::terminal_config,
            context_verifier::VerifierSumcheckContext,
            loader_verifier::load_verifier_sumcheck_data,
        },
    },
};

/// The ring-claimed terms' share of the batched claim, in the combiner's output order.
fn batch_claims(
    config: &SumcheckConfig,
    claims: &[OuterClaim],
    rc_commitment_inner: &[RingElement],
    rc_opening_inner: &[RingElement],
    rc_coarse_projection_inner: Option<&[RingElement]>,
    rc_fine_projection_inner: Option<(&[RingElement], Option<&[RingElement]>)>,
    trace_terms: &TraceTerms,
    combination: &[RingElement],
) -> RingElement {
    let mut batched_claim = RingElement::zero(Representation::IncompleteNTT);
    let mut idx = 0;

    // CommitmentFold: zero claims (difference sumchecks)
    idx += config.basic_commitment_rank;

    // InnerEvalFold: zero claims (difference sumchecks)
    idx += config.nof_openings;

    // OuterEvalClaim: claims for evaluations, those held as field claims batched apart
    for claim in claims.iter() {
        if let OuterClaim::Ring(claim) = claim {
            let mut weighted = claim.clone();
            weighted *= &combination[idx];
            batched_claim += &weighted;
        }
        idx += 1;
    }

    if rc_coarse_projection_inner.is_some() {
        // CoarseProj: zero claim (difference sumcheck)
        idx += 1;
    }

    if rc_fine_projection_inner.is_some() {
        // FineProj: zero claim (difference sumcheck), then the trace-claimed constant terms
        for i in 0..NOF_BATCHES {
            idx += 1;
            debug_assert_eq!(Some(idx), trace_terms.constant_terms.map(|terms| terms[i]));
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
        rc_fine_projection_inner.and_then(|(_, rc_bp)| rc_bp),
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
                Projection::Fine(proj_config) => {
                    proj_config.recursion_batched_projection.as_ref().unwrap()
                }
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

    // NormCheck: the trace-claimed norm terms close the combiner
    debug_assert_eq!(
        idx,
        trace_terms.norm,
        "batched claim walked {idx} of {} combiner outputs",
        combination.len()
    );

    batched_claim
}

pub fn sumcheck_verifier(
    config: &SumcheckConfig,
    verifier_sumcheck_context: &mut VerifierSumcheckContext,
    rc_commitment: &[RingElement],
    round_proof: &SumcheckRoundProof,
    evaluation_points_inner: &[StructuredRow],
    evaluation_points_outer: &[StructuredRow],
    claims: &[OuterClaim],
    defer_witness_claim: bool,
    hash_wrapper: &mut HashWrapper,
) -> (Vec<RingElement>, Option<FieldClaim>) {
    assert_eq!(
        claims.len(),
        config.nof_openings,
        "outer claim count does not match the round's openings"
    );
    assert_eq!(
        round_proof.claim_over_witness.is_none(),
        defer_witness_claim,
        "witness evaluation presence does not match the next round"
    );
    assert_eq!(
        round_proof.rc_fine_projection_inner.is_some(),
        matches!(config.projection_recursion, Projection::Fine(_)),
        "fine projection leaves do not match the round's config"
    );
    let batched_in_opening = config.batched_projection_in_opening();
    hash_wrapper.update_with_ring_element_slice(rc_commitment);
    if !batched_in_opening {
        hash_wrapper.update_with_ring_element_slice(&round_proof.rc_opening_inner);
    }
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
                config.witness_width,
                config.witness_height,
                hash_wrapper,
            );
        match rcs_projection_1_batched {
            Some(rcs_projection_1_batched) => {
                assert!(!batched_in_opening, "a batched leaf the config does not commit");
                hash_wrapper.update_with_ring_element_slice(rcs_projection_1_batched)
            }
            None => {
                assert!(batched_in_opening, "the batched leaf is missing");
                hash_wrapper.update_with_ring_element_slice(&round_proof.rc_opening_inner)
            }
        }
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

    let terminal_projection = match (
        terminal_config(config),
        round_proof.terminal_projection_image.as_ref(),
    ) {
        (Some(next), Some(image)) => {
            assert_eq!(
                (image.height, image.width),
                (config.composed_witness_length / next.projection_ratio, 1),
                "terminal projection image shape"
            );
            Some(verifier_terminal_projection_challenges(
                next,
                config.composed_witness_length,
                image,
                hash_wrapper,
            ))
        }
        (None, None) => None,
        _ => panic!("terminal projection image presence does not match the round's config"),
    };
    let terminal_constant_terms: Option<[u64; NOF_BATCHES]> =
        terminal_projection.as_ref().map(|projection| {
            let image = &round_proof.terminal_projection_image.as_ref().unwrap().data;
            std::array::from_fn(|i| {
                let challenges = &projection.challenges[i];
                let constant_term =
                    image_constant_term(image, &challenges.c_0_layers, &challenges.c_1_layers);
                unsafe { multiply_mod(constant_term, projection.weights[i], MOD_Q) }
            })
        });

    let projection_matrix_flatter_structured = match config.projection_recursion {
        Projection::Coarse(_) => {
            let mut projection_matrix_flatter_base =
                vec![
                    RingElement::zero(Representation::IncompleteNTT);
                    projection_height_flat.ilog2() as usize
                ];
            hash_wrapper
                .sample_ring_element_ntt_slots_same_vec_into(&mut projection_matrix_flatter_base);

            Some(evaluation_point_to_structured_row(
                &projection_matrix_flatter_base,
            ))
        }
        Projection::Fine(_) => None,
        Projection::Skip => None,
    };

    let projection_norm_claim = match (
        config.projection_norm_scope(),
        round_proof.projection_norm_claim.as_ref(),
    ) {
        (Some(_), Some(claim)) => Some(claim),
        (None, None) => None,
        (scope, claim) => panic!(
            "projection norm claim presence {} does not match the round's config {}",
            claim.is_some(),
            scope.is_some()
        ),
    };

    let binariness_claim = round_proof.binariness_claim.as_ref();
    assert_eq!(
        binariness_claim.is_some(),
        !config.binary_leaves().is_empty(),
        "binariness claim presence does not match the round's binary leaves"
    );

    hash_wrapper.update_with_field_extension_element(&round_proof.norm_claim);
    hash_wrapper.update_with_field_extension_element(&round_proof.most_inner_norm_claim);
    if let Some(projection_norm_claim) = projection_norm_claim {
        hash_wrapper.update_with_field_extension_element(projection_norm_claim);
    }
    if let Some(binariness_claim) = binariness_claim {
        hash_wrapper.update_with_field_extension_element(binariness_claim);
    }
    if let Some(constant_term_claims) = &round_proof.constant_term_claims {
        hash_wrapper.update_with_field_extension_slice(constant_term_claims);
    }

    // The field-claimed terms' values, in the order of `field_terms`.
    let outer_claims = claims
        .iter()
        .enumerate()
        .filter_map(|(opening, claim)| match claim {
            OuterClaim::Ring(_) => None,
            OuterClaim::Field(claim) => Some((opening, claim)),
        });
    let mut field_claims: Vec<FieldExtension> =
        outer_claims.clone().map(|(_, claim)| claim.value).collect();
    let with_constant_term = |claim: &FieldExtension, constant_term: u64| {
        assert_eq!(claim.coeffs[0], constant_term, "a trace claim with the wrong part 0");
        *claim
    };
    field_claims.extend(match (&round_proof.constant_term_claims, &challenges_3_1) {
        (Some(claims), Some(_)) => {
            assert_eq!(claims.len(), NOF_BATCHES, "constant-term claim count");
            claims
                .iter()
                .enumerate()
                .map(|(i, claim)| {
                    with_constant_term(claim, terminal_constant_terms.map_or(0, |terms| terms[i]))
                })
                .collect()
        }
        (None, None) => vec![],
        _ => panic!("constant-term claim presence does not match the round's config"),
    });
    field_claims.extend([round_proof.norm_claim, round_proof.most_inner_norm_claim]);
    field_claims.extend(projection_norm_claim.copied());
    field_claims.extend(binariness_claim.map(|claim| with_constant_term(claim, 0)));

    let num_sumchecks = verifier_sumcheck_context
        .combiner_evaluation
        .borrow()
        .sumchecks_count();
    let trace_terms = TraceTerms::new(config, num_sumchecks);
    let field_terms = field_terms(
        config,
        num_sumchecks,
        outer_claims.map(|(opening, claim)| (opening, &claim.weights)),
    );
    assert_eq!(field_terms.len(), field_claims.len());
    let Batching {
        combination,
        fe,
        rho,
    } = sample_batching(hash_wrapper, num_sumchecks, &field_terms);

    let batched_claim = batch_claims(
        config,
        claims,
        rc_commitment,
        &round_proof.rc_opening_inner,
        round_proof.rc_coarse_projection_inner.as_deref(),
        round_proof
            .rc_fine_projection_inner
            .as_ref()
            .map(|(a, b)| (a.as_slice(), b.as_deref())),
        &trace_terms,
        &combination,
    );

    for (label, claim, bound) in [
        (
            "norm claim via inner-product",
            Some(&round_proof.norm_claim),
            config.norm_bound,
        ),
        (
            "most inner norm claim via inner-product",
            Some(&round_proof.most_inner_norm_claim),
            config.most_inner_norm_bound,
        ),
        (
            "projection image norm claim via inner-product",
            projection_norm_claim,
            config.projection_norm_bound,
        ),
    ] {
        if let Some(claim) = claim {
            assert_norm_bounded(label, (claim.coeffs[0] as f64).sqrt(), bound);
        }
    }

    let mut batched_claim_over_field = slot_form(&fe, &batched_claim);
    for (rho, claim) in rho.iter().zip(field_claims.iter()) {
        batched_claim_over_field += &(*rho * *claim);
    }

    let mut num_vars = round_proof.polys.len();

    let mut evaluation_points: Vec<FieldExtension> = vec![];
    while num_vars > 0 {
        num_vars -= 1;

        let poly_over_field = round_proof
            .polys
            .get(round_proof.polys.len() - num_vars - 1)
            .unwrap();

        hash_wrapper.update_with_field_extension_slice(&poly_over_field.coefficients);

        assert_eq!(
            poly_over_field.at_zero() + poly_over_field.at_one(),
            batched_claim_over_field,
            "round-poly claim mismatch at witness_height={} num_vars_left={}",
            config.witness_height,
            num_vars
        );

        let mut f = FieldExtension::zero();

        hash_wrapper.sample_field_element_into(&mut f);

        batched_claim_over_field = poly_over_field.at(&f);
        evaluation_points.push(f);
    }

    load_verifier_sumcheck_data(
        verifier_sumcheck_context,
        &folding_challenges,
        round_proof.claim_over_witness.as_ref().unwrap_or(&ONE),
        &round_proof.claim_over_witness_conjugate,
        evaluation_points_inner,
        evaluation_points_outer,
        &projection_matrix,
        &projection_matrix_flatter_structured,
        &challenges_3_1, // for 1 projection type only
        terminal_projection.as_ref(),
        &combination,
        &fe,
    );

    // Without z_0 the final check is the next round's first outer claim
    // `Σ_s λ_s · slot_s(z_0) = v`, `λ_s = fe_s · slot_s(A)` for the combination `A` at z_0 = 1.
    let next_outer_claim = match &round_proof.claim_over_witness {
        Some(claim_over_witness) => {
            assert_eq!(
                &batched_claim_over_field,
                verifier_sumcheck_context
                    .field_combiner_evaluation
                    .borrow_mut()
                    .evaluate(&evaluation_points)
            );
            hash_wrapper.update_with_ring_element(claim_over_witness);
            None
        }
        None => {
            let slope = verifier_sumcheck_context
                .field_combiner_evaluation
                .borrow_mut()
                .evaluate_ring(&evaluation_points);
            Some(FieldClaim {
                weights: slot_weights(&fe, &slope),
                value: batched_claim_over_field,
            })
        }
    };
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

    (eps, next_outer_claim)
}
