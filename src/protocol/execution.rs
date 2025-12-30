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
        commitment::{commit, init_commitment},
        crs::CRS,
        fold::fold,
        open::open_at,
        project::project,
    },
};

pub fn execute() {
    let crs = CRS::gen_crs(256, 2);
    let mut hash_wrapper = HashWrapper::new();

    let witness = VerticallyAlignedMatrix {
        height: 256,
        width: 16,
        data: sample_random_short_vector(256 * 16, 1, Representation::IncompleteNTT),
    };

    let mut folded_witness = VerticallyAlignedMatrix::new_zero(
        witness.height,
        witness.width,
        &RingElement::zero(Representation::IncompleteNTT),
    );

    let mut commitment = init_commitment(crs.ck.len(), witness.width);

    commit(&mut commitment, &crs, &witness);

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

    let mut projection_image = VerticallyAlignedMatrix::new_zero(
        witness.height / projection_matrix.projection_ratio,
        witness.width,
        &RingElement::zero(Representation::IncompleteNTT),
    );

    project(&mut projection_image, &witness, &projection_matrix);

    hash_wrapper.update_with_ring_element_slice(&projection_image.data);

    let mut fold_challenge = vec![RingElement::zero(Representation::IncompleteNTT); witness.width];

    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut fold_challenge);

    fold(&mut folded_witness, &witness, &fold_challenge);
}
