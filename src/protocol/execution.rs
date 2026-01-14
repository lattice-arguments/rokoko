use std::{process::exit, sync::LazyLock};

use num::range;

use crate::{
    common::{
        arithmetic::inner_product,
        config::MOD_Q,
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
        commitment::{commit_basic, recursive_commit, RecursiveCommitment},
        config::{paste_by_prefix, paste_recursive_commitment, Projection, CONFIG},
        crs::{CK, CRS},
        fold::fold,
        open::{claim, evaluation_point_to_structured_row, open_at, Opening},
        prefix::check_prefixing_correctness,
        project::project,
        proof::Proof,
        sumcheck::{self, init_sumcheck, sumcheck, SumcheckContext},
        sumcheck_utils::{
            common::{EvaluationSumcheckData, HighOrderSumcheckData, SumcheckBaseData},
            linear::{LinearSumcheck, StructuredRowEvaluationLinearSumcheck},
        },
        sumchecks::{
            builder_verifier::init_verifier,
            context_verifier::VerifierSumcheckContext,
            runner::{sumcheck_verifier, RoundProof},
        }, // sumcheck::sumcheck,
    },
};

pub fn prover_round(
    crs: &CRS,
    rc_commitment: &RecursiveCommitment,
    witness: &VerticallyAlignedMatrix<RingElement>,
    evaluation_points_inner: &Vec<StructuredRow>,
    evaluation_points_outer: &Vec<StructuredRow>,
    claims: &Vec<RingElement>,
    sumcheck_context: &mut SumcheckContext,
    verifier_sumcheck_context: &mut VerifierSumcheckContext,
) {
    let mut hash_wrapper = HashWrapper::new();

    let start = std::time::Instant::now();
    hash_wrapper.update_with_ring_element_slice(&rc_commitment.most_inner_commitment());

    let t0 = std::time::Instant::now();
    let opening = open_at(&witness, &evaluation_points_inner, &evaluation_points_outer);
    println!("  open_at: {} ms", t0.elapsed().as_millis());

    let t1 = std::time::Instant::now();
    let rc_opening = recursive_commit(crs, &CONFIG.opening_recursion, &opening.rhs.data);
    println!("  rc_opening: {} ms", t1.elapsed().as_millis());

    hash_wrapper.update_with_ring_element_slice(&rc_opening.most_inner_commitment());

    let mut projection_matrix = ProjectionMatrix::new(CONFIG.projection_ratio);

    projection_matrix.sample(&mut hash_wrapper);

    let rc_projection_image = match &CONFIG.projection_recursion {
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

    // TODO implement Type1 projection recursion

    let mut fold_challenge = vec![RingElement::zero(Representation::IncompleteNTT); witness.width];

    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut fold_challenge);

    let t4 = std::time::Instant::now();
    let folded_witness = fold(&witness, &fold_challenge);
    println!("  fold: {} ms", t4.elapsed().as_millis());

    let mut next_round_data = new_vec_zero_preallocated(CONFIG.composed_witness_length);

    let t5 = std::time::Instant::now();
    let folded_witness_decomposed = decompose(
        &folded_witness.data,
        CONFIG.witness_decomposition_base_log as u64,
        CONFIG.witness_decomposition_chunks,
    );
    println!("  decompose: {} ms", t5.elapsed().as_millis());

    // TODO: can we avoid those copies?
    paste_by_prefix(
        &mut next_round_data,
        &folded_witness_decomposed,
        &CONFIG.folded_witness_prefix,
    );

    match &CONFIG.projection_recursion {
        Projection::Type0(projection_config) => {
            paste_recursive_commitment(
                &mut next_round_data,
                &rc_projection_image.as_ref().unwrap(),
                &projection_config,
            );
        }
        Projection::Type1(projection_config) => {
            //  paste_recursive_commitment(
            //     &mut next_round_data,
            //     &rc_projection_image,
            //     &projection_config.recursion_constant_term,
            // );
            // paste_recursive_commitment(
            //     &mut next_round_data,
            //     &rc_projection_image,
            //     &projection_config.recursion_batched_projection,
            // );
        }
    }

    paste_recursive_commitment(&mut next_round_data, &rc_opening, &CONFIG.opening_recursion);

    paste_recursive_commitment(
        &mut next_round_data,
        &rc_commitment,
        &CONFIG.commitment_recursion,
    );

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
    let (claim_over_witness, claim_over_witness_conjugate, norm_claim, sumcheck_transcript) =
        sumcheck(
            &CONFIG,
            &next_round_data,
            &projection_matrix,
            &fold_challenge,
            &opening,
            sumcheck_context,
            &mut hash_wrapper,
        );
    println!("  sumcheck: {} ms", t6.elapsed().as_millis());

    assert!(
        ell_2_norm * ell_2_norm < (MOD_Q as f64 / 2f64),
        "norm too large, aborting"
    );

    // TODO: recurse

    // let rc_proj_inner = rc_projection_image.as_ref().map(|rc| rc.most_inner_commitment());

    let rp = RoundProof {
        polys: &sumcheck_transcript,
        claim_over_witness: &claim_over_witness,
        claim_over_witness_conjugate: &claim_over_witness_conjugate,
        norm_claim: &norm_claim,
        rc_commitment_inner: rc_commitment.most_inner_commitment(),
        rc_opening_inner: rc_opening.most_inner_commitment(),
        rc_projection_inner: rc_projection_image
            .as_ref()
            .map(|rc| rc.most_inner_commitment()),
    };

    let elapsed = start.elapsed().as_nanos();
    println!("Prover: {} ns", elapsed);

    let mut hash_wrapper_verifier = HashWrapper::new();

    let start = std::time::Instant::now();
    sumcheck_verifier(
        &CONFIG,
        verifier_sumcheck_context,
        &rp,
        &evaluation_points_inner,
        &evaluation_points_outer,
        &claims,
        &mut hash_wrapper_verifier,
    );

    let elapsed = start.elapsed().as_nanos();
    println!("Verifier: {} ns", elapsed);

    // RoundOutput {
    //     folded_witness,
    //     projection_image,
    //     opening,
    // }
}

pub fn execute() {
    check_prefixing_correctness(&CONFIG);
    let crs = CRS::gen_crs(CONFIG.witness_height * 4, CONFIG.basic_commitment_rank);

    let mut sumcheck_context = init_sumcheck(&crs, &CONFIG);
    let mut sumcheck_context_verifier = init_verifier(&crs, &CONFIG);

    let witness = VerticallyAlignedMatrix {
        height: CONFIG.witness_height,
        width: CONFIG.witness_width,
        data: sample_random_short_vector(
            CONFIG.witness_height * CONFIG.witness_width,
            2,
            Representation::IncompleteNTT,
        ),
    };

    let basic_commitment = commit_basic(&crs, &witness, CONFIG.basic_commitment_rank);

    let rc_commitment =
        recursive_commit(&crs, &CONFIG.commitment_recursion, &basic_commitment.data);

    let evaluation_points_inner = vec![evaluation_point_to_structured_row(
        &range(0, witness.height.ilog2() as usize)
            .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 2))
            .collect::<Vec<RingElement>>(),
    )];

    let evaluation_points_outer = vec![evaluation_point_to_structured_row(
        &range(0, witness.width.ilog2() as usize)
            .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 2))
            .collect::<Vec<RingElement>>(),
    )];

    let claims = vec![claim(
        &witness,
        &evaluation_points_inner[0],
        &evaluation_points_outer[0],
    )];
    let round_output = prover_round(
        &crs,
        &rc_commitment,
        &witness,
        &evaluation_points_inner,
        &evaluation_points_outer,
        &claims,
        &mut sumcheck_context,
        &mut sumcheck_context_verifier,
    );
}
