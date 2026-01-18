use num::range;

use crate::{
    common::{
        matrix::VerticallyAlignedMatrix,
        ring_arithmetic::{Representation, RingElement},
        sampling::sample_random_short_vector,
    },
    protocol::{
        config::{to_kb, Config, SizeableProof, CONFIG},
        crs::CRS,
        open::{claim, evaluation_point_to_structured_row},
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

    let crs = CRS::gen_crs(config.composed_witness_length, config.basic_commitment_rank);

    let mut sumcheck_context = init_sumcheck(&crs, &config);
    let mut sumcheck_context_verifier = init_verifier(&crs, &config);
    println!("Sumcheck contexts initialized.");

    let witness = VerticallyAlignedMatrix {
        height: config.witness_height,
        width: config.witness_width,
        data: sample_random_short_vector(
            config.witness_height * config.witness_width,
            2,
            Representation::IncompleteNTT,
        ),
    };

    let (rc_commitment_with_aux, rc_commitment) = commit(&crs, &config, &witness);

    println!("Witness generated.");

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

    let (proof, claims) = prover_round(
        &crs,
        &config,
        &rc_commitment_with_aux,
        &witness,
        &evaluation_points_inner,
        &evaluation_points_outer,
        &mut sumcheck_context,
        true,
    );

    print!("==== PROOF SIZE ====\n");
    let proof_size_bits = proof.size_in_bits();
    println!("Total proof size: {} KB", to_kb(proof_size_bits));
    println!("====================\n");

    verifier_round(
        &crs,
        &config,
        &rc_commitment,
        &proof,
        &evaluation_points_inner,
        &evaluation_points_outer,
        &claims.unwrap(),
        &mut sumcheck_context_verifier,
    );
}
