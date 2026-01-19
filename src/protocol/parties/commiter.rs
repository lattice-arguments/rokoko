use crate::{
    common::{matrix::VerticallyAlignedMatrix, ring_arithmetic::RingElement},
    protocol::{
        commitment::{commit_basic, recursive_commit, BasicCommitmentAux, RecursiveCommitmentWithAux},
        config::{ConfigBase, SumcheckConfig},
        crs::CRS, project::Signed16RingElement,
    },
};

pub fn commit(
    crs: &CRS,
    config: &SumcheckConfig,
    witness: &VerticallyAlignedMatrix<RingElement>,
) -> (RecursiveCommitmentWithAux, Vec<RingElement>, VerticallyAlignedMatrix<Signed16RingElement>) {
    let basic_commitment = commit_basic(&crs, &witness, config.basic_commitment_rank);

    let rc_commitment_with_aux =
        recursive_commit(&crs, &config.commitment_recursion, &basic_commitment.commitment.data);

    let rc_commitment = rc_commitment_with_aux.most_inner_commitment().clone();

    (rc_commitment_with_aux, rc_commitment, basic_commitment.witness_i16.unwrap())
}
