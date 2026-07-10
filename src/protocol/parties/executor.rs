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

/// SNARK mode: prove user-supplied sumcheck claims about a committed witness,
/// then run the PCS chain on the resulting evaluation claims.
pub fn execute_snark() {
    use crate::common::{
        hash::HashWrapper,
        ring_arithmetic::{Representation, RingElement},
        sampling::sample_random_short_vector,
    };
    use crate::protocol::params::P_EN_TWO_EVALS;
    use crate::protocol::snark::{eq, prove_claims, verify_claims, witness_in, Claim, Region};

    let config = match &*P_EN_TWO_EVALS {
        Config::Sumcheck(config) => config,
        _ => panic!("Expected sumcheck config at the top level."),
    };

    println!("Generating CRS...");
    let crs = CRS::gen_crs(
        config.composed_witness_length,
        config.basic_commitment_rank + 2,
    );

    let mut sumcheck_context = init_sumcheck(&crs, &config);
    let mut sumcheck_context_verifier = init_verifier(&crs, &config);

    let witness = VerticallyAlignedMatrix {
        height: config.witness_height,
        width: config.witness_width,
        used_cols: config.witness_width,
        data: sample_random_short_vector(
            config.witness_height * config.witness_width,
            2u64.pow(7),
            crate::common::ring_arithmetic::Representation::IncompleteNTT,
        ),
    };

    println!("===== COMMITTING WITNESS =====");
    let start = std::time::Instant::now();
    let (commitment_with_aux, rc_commitment) = commit(&crs, &config, &witness);
    println!("TOTAL Commit time: {:?} ns", start.elapsed().as_nanos());

    let total_vars = (config.witness_height * config.witness_width).ilog2() as usize;
    let n = config.witness_height * config.witness_width;
    let everything = Region::whole(n);

    let structured_point: Vec<RingElement> = (0..total_vars)
        .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 1 << 10))
        .collect();
    let t1 = {
        use crate::common::structured_row::PreprocessedRow;
        let expanded = PreprocessedRow::from_layers(&structured_point).preprocessed_row;
        let mut acc = RingElement::zero(Representation::IncompleteNTT);
        let mut temp = RingElement::zero(Representation::IncompleteNTT);
        for (a, w) in expanded.iter().zip(witness.data.iter()) {
            temp *= (a, w);
            acc += &temp;
        }
        acc
    };
    let claim_linear = Claim::sums_to(eq(structured_point) * witness_in(everything), t1);

    let segment = Region::new(n / 4, n / 4, n);
    let mut t2 = RingElement::zero(Representation::IncompleteNTT);
    {
        let mut temp = RingElement::zero(Representation::IncompleteNTT);
        for w in &witness.data[segment.range()] {
            temp *= (w, w);
            t2 += &temp;
        }
    }
    let claim_square = Claim::sums_to(witness_in(segment) * witness_in(segment), t2);

    // P_EN_TWO_EVALS is compiled for two openings, so the statement must use
    // the conjugate; the norm claim is the natural way.
    let mut t3 = RingElement::zero(Representation::IncompleteNTT);
    {
        let mut temp = RingElement::zero(Representation::IncompleteNTT);
        for w in &witness.data {
            temp *= (w, &w.conjugate());
            t3 += &temp;
        }
    }
    let claim_norm = Claim::sums_to(
        witness_in(everything) * witness_in(everything).conjugate(),
        t3,
    );

    let claims = vec![claim_linear, claim_square, claim_norm];

    println!("==== SNARK PROVER STARTING ===");
    let start = std::time::Instant::now();

    let mut hash_wrapper = HashWrapper::new();
    hash_wrapper.update_with_ring_element_slice(
        &commitment_with_aux
            .rc_commitment_with_aux
            .most_inner_commitment(),
    );

    let (initial_proof, chain_inputs) = prove_claims(&witness, &claims, &mut hash_wrapper);


    println!(
        "Initial claims sumcheck done: {} ms",
        start.elapsed().as_millis()
    );

    let (proof, _) = prover_round(
        &crs,
        &config,
        &commitment_with_aux,
        &witness,
        &chain_inputs.evaluation_points_inner,
        &chain_inputs.evaluation_points_outer,
        &mut sumcheck_context,
        false,
        Some(hash_wrapper),
    );
    println!("==== SNARK PROVER DONE ===");
    println!("TOTAL Prover time: {:?} ns", start.elapsed().as_nanos());

    let proof_size_bits = proof.size_in_bits();
    println!("Total proof size: {} KB", to_kb(proof_size_bits));

    println!("==== SNARK VERIFIER STARTING ===");
    let start = std::time::Instant::now();

    let mut hash_wrapper_verifier = HashWrapper::new();
    hash_wrapper_verifier.update_with_ring_element_slice(&rc_commitment);

    let chain_inputs_verifier = verify_claims(
        (config.witness_height, config.witness_width),
        &claims,
        &initial_proof,
        &mut hash_wrapper_verifier,
    );

    verifier_round(
        &crs,
        &config,
        &rc_commitment,
        &proof,
        &chain_inputs_verifier.evaluation_points_inner,
        &chain_inputs_verifier.evaluation_points_outer,
        &chain_inputs_verifier.claims,
        &mut sumcheck_context_verifier,
        Some(hash_wrapper_verifier),
    );
    println!("==== SNARK VERIFIER DONE ===");
    println!("TOTAL Verifier time: {:?} ns", start.elapsed().as_nanos());
}
