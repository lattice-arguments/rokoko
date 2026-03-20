use crate::{common::{matrix::VerticallyAlignedMatrix, ring_arithmetic::RingElement}, protocol::crs::CRS};

pub fn commit(
    witness: VerticallyAlignedMatrix<RingElement>,
    crs: &CRS,
) -> VerticallyAlignedMatrix<RingElement> {
    panic!("Not implemented yet");
}