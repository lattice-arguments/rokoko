use crate::{
    common::{
        arithmetic::field_to_ring_element,
        config::HALF_DEGREE,
        hash::HashWrapper,
        matrix::new_vec_zero_preallocated,
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        structured_row::StructuredRow,
        sumcheck_element::SumcheckElement,
    },
    protocol::{
        config::{IntermediateConfig, IntermediateRoundProof},
        sumcheck_utils::common::EvaluationSumcheckData,
    },
};

use super::{
    context_verifier::IntermediateVerifierSumcheckContext,
    loader_verifier::load_intermediate_verifier_sumcheck_data,
};

pub fn batch_claims_linear(
    claims: &[RingElement],
    combination: &[RingElement],
    start_idx: usize,
) -> (RingElement, usize) {
    assert!(
        start_idx + claims.len() <= combination.len(),
        "Not enough combination challenges for linear claim batching"
    );

    let mut batched = RingElement::zero(Representation::IncompleteNTT);
    let mut idx = start_idx;
    for claim in claims {
        let mut weighted = claim.clone();
        weighted *= &combination[idx];
        batched += &weighted;
        idx += 1;
    }
    (batched, idx)
}

pub fn intermediate_sumcheck_verifier(
    config: &IntermediateConfig,
    verifier_sumcheck_context: &mut IntermediateVerifierSumcheckContext,
    proof: &IntermediateRoundProof,
    folded_commitment: &[RingElement],
    folded_opening_claims: &[RingElement],
    evaluation_points_inner: &[StructuredRow],
    hash_wrapper: &mut HashWrapper,
) -> Vec<RingElement> {
    assert_eq!(
        folded_commitment.len(),
        config.basic_commitment_rank,
        "Folded commitment length mismatch for intermediate batcher"
    );

    let num_sumchecks = verifier_sumcheck_context
        .combiner_evaluation
        .borrow()
        .sumchecks_count();
    let mut combination = new_vec_zero_preallocated(num_sumchecks);
    // hash_wrapper.sample_ring_element_into(&mut combination[num_sumchecks - 1]);
    hash_wrapper.sample_ring_element_vec_into(&mut combination);
    // assert_eq!(
    //     num_sumchecks,
    //     config.basic_commitment_rank + 1,
    //     "Intermediate verifier expected folded commitment plus one type5 claim"
    // );

    let mut combination_to_field = RingElement::zero(Representation::IncompleteNTT);
    hash_wrapper.sample_ring_element_into(&mut combination_to_field);
    combination_to_field.from_incomplete_ntt_to_homogenized_field_extensions();
    let qe: [QuadraticExtension; HALF_DEGREE] =
        combination_to_field.split_into_quadratic_extensions();

    let (mut batched_claim, idx) = batch_claims_linear(folded_commitment, &combination, 0);
    let (batched_type1_claims, idx) =
        batch_claims_linear(folded_opening_claims, &combination, idx);
    batched_claim += &batched_type1_claims;
    let mut weighted_norm = proof.norm_claim.clone();
    weighted_norm *= &combination[idx];
    batched_claim += &weighted_norm;

    let mut batched_claim_over_field = {
        let mut temp = batched_claim.clone();
        temp.from_incomplete_ntt_to_homogenized_field_extensions();
        let mut split = temp.split_into_quadratic_extensions();
        let mut result = QuadraticExtension::zero();

        for i in 0..HALF_DEGREE {
            split[i] *= &qe[i];
            result += &split[i];
        }
        result
    };

    let mut evaluation_points_field: Vec<QuadraticExtension> =
        Vec::with_capacity(proof.polys.len());

    for poly_over_field in proof.polys.iter() {
        hash_wrapper.update_with_quadratic_extension_slice(&poly_over_field.coefficients);

        assert_eq!(
            poly_over_field.at_zero() + poly_over_field.at_one(),
            batched_claim_over_field
        );

        let mut challenge = QuadraticExtension::zero();
        hash_wrapper.sample_field_element_into(&mut challenge);

        batched_claim_over_field = poly_over_field.at(&challenge);
        evaluation_points_field.push(challenge);
    }

    load_intermediate_verifier_sumcheck_data(
        verifier_sumcheck_context,
        &proof.claim_over_witness,
        &proof.claim_over_witness_conjugate,
        evaluation_points_inner,
        &combination,
        &qe,
    );

    assert_eq!(
        &batched_claim_over_field,
        verifier_sumcheck_context
            .field_combiner_evaluation
            .borrow_mut()
            .evaluate(&evaluation_points_field)
    );

    evaluation_points_field
        .iter()
        .rev()
        .map(|f| {
            let mut r = field_to_ring_element(f);
            r.from_homogenized_field_extensions_to_incomplete_ntt();
            r
        })
        .collect()
}
