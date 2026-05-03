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

    let _commit_span = tracing::info_span!("commit").entered();
    let witness_decomposed = decompose_witness(&witness);
    let (commitment_with_aux, rc_commitment) = commit(&crs, &config, &witness_decomposed);
    drop(_commit_span);

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

    let _prover_span = tracing::info_span!("prover").entered();
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
    drop(_prover_span);

    print!("==== PROOF SIZE ====\n");
    let proof_size_bits = proof.size_in_bits();
    println!("Total proof size: {} KB", to_kb(proof_size_bits));
    println!("====================\n");

    let _verifier_span = tracing::info_span!("verifier").entered();
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
    drop(_verifier_span);
}
