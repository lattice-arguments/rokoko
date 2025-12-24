use crate::{
    common::{
        ring_arithmetic::Representation, sampling::sample_random_short_vector,
        vertically_aligned_matrix::VerticallyAlignedMatrix,
    },
    protocol::{
        commitment::{self, commit},
        crs::CRS,
    },
};

pub fn execute() {
    let crs = CRS::gen_crs(256, 2);

    let witness = VerticallyAlignedMatrix {
        height: 256,
        width: 16,
        data: sample_random_short_vector(256 * 16, 1, Representation::IncompleteNTT),
    };

    let commitment = commit(&crs, &witness);
}
