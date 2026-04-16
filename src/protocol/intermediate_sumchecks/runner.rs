use crate::{
    common::{
        arithmetic::field_to_ring_element_into,
        config::HALF_DEGREE,
        hash::HashWrapper,
        matrix::new_vec_zero_preallocated,
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        sumcheck_element::SumcheckElement,
    },
    protocol::{
        config::IntermediateConfig,
        sumcheck_utils::{
            common::{HighOrderSumcheckData, SumcheckBaseData},
            polynomial::Polynomial,
        },
    },
};

use super::{context::IntermediateSumcheckContext, loader::load_intermediate_sumcheck_data};

#[derive(Clone)]
pub struct IntermediateType0SumcheckProof {
    pub claim_over_witness: RingElement,
    pub type0_claims: Vec<RingElement>,
    pub polys: Vec<Polynomial<QuadraticExtension>>,
}

pub fn run_intermediate_sumcheck(
    config: &IntermediateConfig,
    combined_witness: &[RingElement],
    sumcheck_context: &mut IntermediateSumcheckContext,
    hash_wrapper: &mut HashWrapper,
) -> (IntermediateType0SumcheckProof, Vec<RingElement>) {
    let num_sumchecks = sumcheck_context.combiner.borrow().sumchecks_count();
    let mut combination = new_vec_zero_preallocated(num_sumchecks);
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
        &combination,
        &qe,
    );

    let type0_claims = sumcheck_context
        .type0sumchecks
        .iter()
        .map(|type0_sc| type0_sc.output.borrow_mut().claim())
        .collect::<Vec<_>>();

    let mut num_vars = sumcheck_context.combiner.borrow().variable_count();
    let mut polys: Vec<Polynomial<QuadraticExtension>> = Vec::with_capacity(num_vars);
    let mut evaluation_points: Vec<RingElement> = Vec::with_capacity(num_vars);

    while num_vars > 0 {
        num_vars -= 1;

        let mut poly_over_field = Polynomial::<QuadraticExtension>::new(0);
        sumcheck_context
            .field_combiner
            .borrow_mut()
            .univariate_polynomial_into(&mut poly_over_field);

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

    let claim_over_witness = sumcheck_context
        .witness_sumcheck
        .borrow()
        .final_evaluations()
        .clone();

    evaluation_points.reverse();

    (
        IntermediateType0SumcheckProof {
            claim_over_witness,
            type0_claims,
            polys,
        },
        evaluation_points,
    )
}
