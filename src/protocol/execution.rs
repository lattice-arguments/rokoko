use blake3::Hash;
use num::range;

use crate::{
    common::{
        hash::HashWrapper,
        matrix::VerticallyAlignedMatrix,
        ring_arithmetic::{Representation, RingElement},
        sampling::sample_random_short_vector,
    },
    protocol::{
        commitment::{self, commit},
        crs::CRS,
        open::open_at,
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

    let commitment = commit(&crs, &witness);

    let evaluation_points_inner = vec![range(0, witness.height.ilog2() as usize)
        .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 2))
        .collect::<Vec<RingElement>>()];

    let evaluation_points_outer = vec![range(0, witness.width.ilog2() as usize)
        .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 2))
        .collect::<Vec<RingElement>>()];

    let opening = open_at(&witness, &evaluation_points_inner, &evaluation_points_outer);

    hash_wrapper.update_with_ring_element_slice(&opening.rhs.data);
}
