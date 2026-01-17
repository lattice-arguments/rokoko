use num::range;

use crate::{
    common::{
        matrix::VerticallyAlignedMatrix,
        ring_arithmetic::{Representation, RingElement},
        sampling::sample_random_short_vector,
    },
    protocol::{
        config::CONFIG,
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
    let crs = CRS::gen_crs(CONFIG.composed_witness_length, CONFIG.basic_commitment_rank);

    let mut sumcheck_context = init_sumcheck(&crs, &CONFIG);
    let mut sumcheck_context_verifier = init_verifier(&crs, &CONFIG);
    println!("Sumcheck contexts initialized.");

    let witness = VerticallyAlignedMatrix {
        height: CONFIG.witness_height,
        width: CONFIG.witness_width,
        data: sample_random_short_vector(
            CONFIG.witness_height * CONFIG.witness_width,
            2,
            Representation::IncompleteNTT,
        ),
    };

    let (rc_commitment_with_aux, rc_commitment) = commit(&crs, &CONFIG, &witness);

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

    let claims_ = vec![claim(
        &witness,
        &evaluation_points_inner[0],
        &evaluation_points_outer[0],
    )];

    let (proof, claims) = prover_round(
        &crs,
        &CONFIG,
        &rc_commitment_with_aux,
        &witness,
        &evaluation_points_inner,
        &evaluation_points_outer,
        &mut sumcheck_context,
        true,
    );

    assert_eq!(claims_[0], claims.as_ref().unwrap()[0]);

    verifier_round(
        &crs,
        &CONFIG,
        &rc_commitment,
        &proof,
        &evaluation_points_inner,
        &evaluation_points_outer,
        &claims.unwrap(),
        &mut sumcheck_context_verifier,
    );
}
