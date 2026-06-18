use crate::{
    common::{
        arithmetic::field_to_ring_element_into,
        config::{HALF_DEGREE, NOF_BATCHES},
        hash::HashWrapper,
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        structured_row::StructuredRow,
        sumcheck_element::SumcheckElement,
    },
    protocol::{
        config::IntermediateConfig,
        project_fine::BatchedProjectionChallenges,
        sumcheck_utils::{
            common::{HighOrderSumcheckData, SumcheckBaseData},
            polynomial::Polynomial,
        },
    },
};

use super::{context::IntermediateSumcheckContext, loader::load_intermediate_sumcheck_data};

#[derive(Clone)]
pub struct IntermediateSumcheckProof {
    pub claim_over_witness: RingElement,
    pub claim_over_witness_conjugate: RingElement,
    pub norm_claim: RingElement,
    pub polys: Vec<Polynomial<QuadraticExtension>>,
}

pub fn run_intermediate_sumcheck(
    config: &IntermediateConfig,
    combined_witness: &[RingElement],
    evaluation_points_inner: &[StructuredRow],
    fine_proj_batching_challenges: &[BatchedProjectionChallenges; NOF_BATCHES],
    sumcheck_context: &mut IntermediateSumcheckContext,
    hash_wrapper: &mut HashWrapper,
) -> (IntermediateSumcheckProof, Vec<RingElement>) {
    let mut conjugated_combined_witness = vec![RingElement::zero(Representation::IncompleteNTT); combined_witness.len()];
    combined_witness
        .iter()
        .zip(conjugated_combined_witness.iter_mut())
        .for_each(|(orig, conj)| {
            orig.conjugate_into(conj);
        });
    let mut norm_claim = RingElement::zero(Representation::IncompleteNTT);
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    for (w, wc) in combined_witness
        .iter()
        .zip(conjugated_combined_witness.iter())
    {
        temp *= (w, wc);
        norm_claim += &temp;
    }

    hash_wrapper.update_with_ring_element(&norm_claim);

    let num_sumchecks = sumcheck_context.combiner.borrow().sumchecks_count();
    println!("num_sumchecks: {}", num_sumchecks);

    let mut combination = vec![RingElement::zero(Representation::IncompleteNTT); num_sumchecks];
    hash_wrapper.sample_ring_element_vec_into(&mut combination);

    let mut combination_to_field = RingElement::zero(Representation::IncompleteNTT);
    hash_wrapper.sample_ring_element_into(&mut combination_to_field);
    combination_to_field.from_incomplete_ntt_to_homogenized_field_extensions();
    let qe: [QuadraticExtension; HALF_DEGREE] =
        combination_to_field.split_into_quadratic_extensions();

    load_intermediate_sumcheck_data(
        sumcheck_context,
        config,
        combined_witness,
        &conjugated_combined_witness,
        evaluation_points_inner,
        &combination,
        fine_proj_batching_challenges,
        &qe,
    );

    let norm_check_claim = sumcheck_context.norm_check_sumcheck.output.borrow_mut().claim();
    assert_eq!(
        norm_check_claim, norm_claim,
        "NormCheck intermediate claim mismatch: expected <w, conj(w)>"
    );

    let mut num_vars = sumcheck_context.combiner.borrow().variable_count();
    let mut polys: Vec<Polynomial<QuadraticExtension>> = Vec::with_capacity(num_vars);
    let mut evaluation_points: Vec<RingElement> = Vec::with_capacity(num_vars);

    while num_vars > 0 {
        num_vars -= 1;

        let mut poly_over_field = Polynomial::<QuadraticExtension>::new(0);
        sumcheck_context
            .field_combiner
            .borrow_mut()
            .univariate_polynomial_into(false, &mut poly_over_field);

        hash_wrapper.update_with_quadratic_extension_slice(&poly_over_field.coefficients);

        let mut challenge_field = QuadraticExtension::zero();
        hash_wrapper.sample_field_element_into(&mut challenge_field);

        let mut challenge_ring = RingElement::zero(Representation::IncompleteNTT);
        field_to_ring_element_into(&mut challenge_ring, &challenge_field);
        challenge_ring.from_homogenized_field_extensions_to_incomplete_ntt();

        sumcheck_context.partial_evaluate_all(&challenge_ring);

        evaluation_points.push(challenge_ring);
        polys.push(poly_over_field);
    }
    #[cfg(feature = "profile-sumcheck")]
    crate::protocol::sumcheck_utils::profile::print_and_reset("intermediate-sumcheck");

    let claim_over_witness = sumcheck_context
        .witness_sumcheck
        .borrow()
        .final_evaluations()
        .clone();
    let claim_over_witness_conjugate = sumcheck_context
        .norm_check_sumcheck
        .conjugated_witness_sumcheck
        .borrow()
        .final_evaluations()
        .clone();

    hash_wrapper.update_with_ring_element(&claim_over_witness);
    hash_wrapper.update_with_ring_element(&claim_over_witness_conjugate);

    evaluation_points.reverse();

    (
        IntermediateSumcheckProof {
            claim_over_witness,
            claim_over_witness_conjugate,
            norm_claim,
            polys,
        },
        evaluation_points,
    )
}
