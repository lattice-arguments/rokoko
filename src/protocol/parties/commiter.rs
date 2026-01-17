use crate::{
    common::{config, matrix::VerticallyAlignedMatrix, ring_arithmetic::RingElement},
    protocol::{
        commitment::{commit_basic, recursive_commit, RecursiveCommitmentWithAux},
        config::Config,
        crs::CRS,
    },
};

pub fn commit(
    crs: &CRS,
    config: &Config,
    witness: &VerticallyAlignedMatrix<RingElement>,
) -> (RecursiveCommitmentWithAux, Vec<RingElement>) {
    let basic_commitment = commit_basic(&crs, &witness, config.basic_commitment_rank);

    let rc_commitment_with_aux =
        recursive_commit(&crs, &config.commitment_recursion, &basic_commitment.data);

    let rc_commitment = rc_commitment_with_aux.most_inner_commitment().clone();

    (rc_commitment_with_aux, rc_commitment)
}
