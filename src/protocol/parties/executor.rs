use crate::{
    common::{matrix::VerticallyAlignedMatrix, ring_arithmetic::RingElement},
    protocol::{
        config::{to_kb, Config, SizeableProof, CONFIG},
        crs::CRS,
        evaluation_point_sampler::{sample_initial_evaluation_points, InitialEvaluationPoints},
        open::claim,
        params::{decompose_witness, witness_sampler, WITNESS_CONFIG},
        parties::{commiter::commit, prover::prover_round, verifier::verifier_round},
        sumcheck::init_sumcheck,
        sumchecks::builder_verifier::init_verifier,
    },
};

pub fn execute() {
    // check_prefixing_correctness(&CONFIG);
    let config = match &*CONFIG {
        Config::Sumcheck(config) => config,
        _ => panic!("Expected sumcheck config at the top level."),
    };

    let witness_config = &*WITNESS_CONFIG;

    println!("Sampling evaluation points...");
    let evaluation_points = sample_initial_evaluation_points(
        witness_config.height,
        witness_config.width,
        witness_config.decomposition_base_log,
        witness_config.decomposition_chunks,
    );

    println!("Generating CRS...");

    let crs_start = std::time::Instant::now();
    let crs = CRS::gen_crs(
        config.composed_witness_length,
        config.basic_commitment_rank + 2,
    );
    let crs_duration = crs_start.elapsed().as_nanos();
    println!("TOTAL CRS gen time: {:?} ns", crs_duration);

    let mut sumcheck_context = init_sumcheck(&crs, &config);
    let mut sumcheck_context_verifier = init_verifier(&crs, &config);
    println!("Sumcheck contexts initialized.");

    let witness = witness_sampler();

    println!("===== COMMITTING WITNESS =====");
    let start = std::time::Instant::now();

    let witness_decomposed = decompose_witness(&witness);
    print!("Witness decomposed. ");

    let (commitment_with_aux, rc_commitment) = commit(&crs, &config, &witness_decomposed);

    let commit_duration = start.elapsed().as_nanos();
    println!("TOTAL Commit time: {:?} ns", commit_duration);

    println!("===== COMMITTING WITNESS DONE =====");

    let start = std::time::Instant::now();

    println!("==== PROVER STARTING ===");

    let (proof, claims) = prover_round(
        &crs,
        &config,
        &commitment_with_aux,
        &witness_decomposed,
        &evaluation_points.inner,
        &evaluation_points.outer,
        &mut sumcheck_context,
        true,
        None,
    );
    let claims = claims.expect("Prover round must return claims when with_claims is true.");
    println!("==== PROVER DONE ===");
    check_prover_claims_match_witness(&witness, &evaluation_points, &claims);

    let prover_duration = start.elapsed().as_nanos();
    println!("TOTAL Prover time: {:?} ns", prover_duration);

    print!("==== PROOF SIZE ====\n");
    let proof_size_bits = proof.size_in_bits();
    println!("Total proof size: {} KB", to_kb(proof_size_bits));
    println!("====================\n");

    let start = std::time::Instant::now();
    println!("==== VERIFIER STARTING ===");
    verifier_round(
        &crs,
        &config,
        &rc_commitment,
        &proof,
        &evaluation_points.inner,
        &evaluation_points.outer,
        &claims,
        &mut sumcheck_context_verifier,
        None,
    );
    println!("==== VERIFIER DONE ===");
    let verifier_duration = start.elapsed().as_nanos();
    println!("TOTAL Verifier time: {:?} ns", verifier_duration);
}

fn check_prover_claims_match_witness(
    witness: &VerticallyAlignedMatrix<RingElement>,
    evaluation_points: &InitialEvaluationPoints,
    prover_claims: &[RingElement],
) {
    assert_eq!(
        prover_claims.len(),
        evaluation_points.witness_inner.len(),
        "Prover returned a different number of claims than sampled witness points."
    );

    for (i, ((inner, outer), prover_claim)) in evaluation_points
        .witness_inner
        .iter()
        .zip(evaluation_points.outer.iter())
        .zip(prover_claims.iter())
        .enumerate()
    {
        let mut expected_claim = claim(witness, inner, outer);
        expected_claim *= &evaluation_points.witness_claim_scale;
        assert_eq!(
            &expected_claim, prover_claim,
            "Prover claim {i} does not match the direct witness claim."
        );
    }
    println!("Prover claims match direct witness claims.");
}
