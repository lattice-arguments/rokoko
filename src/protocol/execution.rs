use blake3::Hash;
use num::range;

use crate::{
    common::{
        hash::HashWrapper,
        matrix::{VerticallyAlignedMatrix, ZeroNew},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{Representation, RingElement},
        sampling::sample_random_short_vector,
    },
    protocol::{
        commitment::{self, commit, Commitment},
        crs::{self, CRS},
        fold::fold,
        open::{open_at, Opening},
        project::project,
        verifier,
    },
};

pub struct RoundOutput {
    folded_witness: VerticallyAlignedMatrix<RingElement>,
    projection_image: VerticallyAlignedMatrix<RingElement>,
    opening: Opening,
}

pub fn prover_simple_round(
    commitment: &Commitment,
    witness: &VerticallyAlignedMatrix<RingElement>,
) -> RoundOutput {
    let mut hash_wrapper = HashWrapper::new();

    hash_wrapper.update_with_ring_element_slice(&commitment.commitment.data);

    let evaluation_points_inner = vec![range(0, witness.height.ilog2() as usize)
        .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 2))
        .collect::<Vec<RingElement>>()];

    let evaluation_points_outer = vec![range(0, witness.width.ilog2() as usize)
        .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 2))
        .collect::<Vec<RingElement>>()];

    let opening = open_at(&witness, &evaluation_points_inner, &evaluation_points_outer);

    hash_wrapper.update_with_ring_element_slice(&opening.rhs.data);

    let mut projection_matrix = ProjectionMatrix::new(8);

    projection_matrix.sample(&mut hash_wrapper);

    let projection_image = project(&witness, &projection_matrix);

    hash_wrapper.update_with_ring_element_slice(&projection_image.data);

    let mut fold_challenge = vec![RingElement::zero(Representation::IncompleteNTT); witness.width];

    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut fold_challenge);

    let folded_witness = fold(&witness, &fold_challenge);

    RoundOutput {
        folded_witness,
        projection_image,
        opening,
    }
}

pub fn verifier_simple_round(crs: &CRS, commitment: &Commitment, round_output: &RoundOutput) {
    // We check if:
    // folded commitment == commit(folded witness)
    // projection_image is correct projection of witness
    // opening is valid opening of witness at evaluation points

    let folded_commitment = commit(crs, &round_output.folded_witness);
}

pub fn execute() {
    let crs = CRS::gen_crs(256, 2);

    let witness = VerticallyAlignedMatrix {
        height: 256,
        width: 16,
        data: sample_random_short_vector(256 * 16, 1, Representation::IncompleteNTT),
    };

    let commitment = commit(&crs, &witness);

    prover_simple_round(&commitment, &witness);
}
