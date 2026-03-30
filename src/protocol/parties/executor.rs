use num::range;

use crate::{
    common::ring_arithmetic::{Representation, RingElement},
    protocol::{
        config::{to_kb, Config, SizeableProof, CONFIG},
        crs::CRS,
        open::evaluation_point_to_structured_row,
        params::{decompose_witness, witness_sampler},
        parties::{commiter::commit, prover::prover_round, verifier::verifier_round},
        sumcheck::init_sumcheck,
        sumchecks::builder_verifier::init_verifier,
    },
};

pub fn execute() {
    // check_prefixing_correctness(&CONFIG);
    println!("Generating CRS...");

    let config = match &*CONFIG {
        Config::Sumcheck(config) => config,
        _ => panic!("Expected sumcheck config at the top level."),
    };


    let crs = CRS::gen_crs(
        config.composed_witness_length,
        config.basic_commitment_rank / config.basic_commitment_diag_blocks + 2,
    );

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

    let evaluation_points_inner = vec![evaluation_point_to_structured_row(
        &range(0, witness_decomposed.height.ilog2() as usize)
            .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 2))
            .collect::<Vec<RingElement>>(),
    )];

    let evaluation_points_outer = vec![evaluation_point_to_structured_row(
        &range(0, witness_decomposed.width.ilog2() as usize)
            .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 2))
            .collect::<Vec<RingElement>>(),
    )];

    let start = std::time::Instant::now();

    println!("==== PROVER STARTING ===");

    let (proof, claims) = prover_round(
        &crs,
        &config,
        &commitment_with_aux,
        &witness_decomposed,
        &evaluation_points_inner,
        &evaluation_points_outer,
        &mut sumcheck_context,
        true,
        None,
    );
    println!("==== PROVER DONE ===");
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
        &evaluation_points_inner,
        &evaluation_points_outer,
        &claims.unwrap(),
        &mut sumcheck_context_verifier,
        None,
    );
    println!("==== VERIFIER DONE ===");
    let verifier_duration = start.elapsed().as_nanos();
    println!("TOTAL Verifier time: {:?} ns", verifier_duration);
}
