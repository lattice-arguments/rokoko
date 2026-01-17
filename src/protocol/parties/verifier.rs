use crate::{
    common::{
        config::{self, MOD_Q},
        decomposition::{compose_from_decomposed, decompose},
        hash::HashWrapper,
        matrix::{new_vec_zero_preallocated, HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        norms,
        ring_arithmetic::{Representation, RingElement},
        sampling::sample_random_short_vector,
        structured_row::{self, PreprocessedRow, StructuredRow},
    },
    protocol::{
        commitment::{commit_basic, recursive_commit, RecursiveCommitmentWithAux},
        config::{paste_by_prefix, paste_recursive_commitment, Config, Projection, CONFIG},
        crs::{CK, CRS},
        fold::fold,
        open::{
            claim, evaluation_point_to_structured_row,
            evaluation_point_to_structured_row_conjugate, open_at, Opening,
        },
        prefix::check_prefixing_correctness,
        project::project,
        project_2::{batch_projection_n_times, project_coefficients},
        proof::Proof,
        sumcheck::{self, init_sumcheck, sumcheck, SumcheckContext},
        sumchecks::{
            builder_verifier::init_verifier,
            context_verifier::VerifierSumcheckContext,
            runner_verifier::{sumcheck_verifier, RoundProof},
        }, // sumcheck::sumcheck,
    },
};

pub fn verifier_round(
    crs: &CRS,
    config: &Config,
    rc_commitment: &Vec<RingElement>,
    round_proof: &RoundProof,
    evaluation_points_inner: &Vec<StructuredRow>,
    evaluation_points_outer: &Vec<StructuredRow>,
    claims: &Vec<RingElement>,
    sumcheck_context_verifier: &mut VerifierSumcheckContext,
) {
    let start = std::time::Instant::now();
    let mut hash_wrapper_verifier = HashWrapper::new();

    let evaluation_points = sumcheck_verifier(
        &config,
        sumcheck_context_verifier,
        &rc_commitment,
        &round_proof,
        &evaluation_points_inner,
        &evaluation_points_outer,
        &claims,
        &mut hash_wrapper_verifier,
    );
    let elapsed = start.elapsed().as_nanos();
    println!("Verifier: {} ns", elapsed);
    match &round_proof.next {
        Some(next_round_proof) => {
            let next_round_commitment =
                round_proof
                    .next_round_commitment
                    .as_ref()
                    .unwrap_or_else(|| {
                        panic!(
                        "Next round commitment must be present when next round proof is present."
                    )
                    });

            let (new_evaluation_points_outer, new_evaluation_points_inner) = evaluation_points
                .split_at(config.next.as_ref().unwrap().witness_width.ilog2() as usize);

            verifier_round(
                crs,
                &config.next.as_ref().unwrap(),
                &next_round_commitment,
                next_round_proof,
                &vec![
                    evaluation_point_to_structured_row(&new_evaluation_points_inner.to_vec()),
                    evaluation_point_to_structured_row_conjugate(
                        &new_evaluation_points_inner.to_vec(),
                    ),
                ],
                &vec![
                    evaluation_point_to_structured_row(&new_evaluation_points_outer.to_vec()),
                    evaluation_point_to_structured_row_conjugate(
                        &new_evaluation_points_outer.to_vec(),
                    ),
                ],
                &vec![
                    round_proof.claim_over_witness.clone(),
                    round_proof.claim_over_witness_conjugate.conjugate(),
                ],
                sumcheck_context_verifier.next.as_mut().unwrap(),
            );
        }
        None => {}
    }
}
