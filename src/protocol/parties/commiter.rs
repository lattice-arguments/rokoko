use crate::{
    common::{matrix::VerticallyAlignedMatrix, ring_arithmetic::RingElement},
    protocol::{
        commitment::{commit_basic, recursive_commit, CommitmentWithAux},
        config::SumcheckConfig,
        crs::CRS,
    },
};

pub fn commit(
    crs: &CRS,
    config: &SumcheckConfig,
    witness: &VerticallyAlignedMatrix<RingElement>,
) -> (CommitmentWithAux, Vec<RingElement>) {
    let basic_commitment = commit_basic(&crs, &witness, config.basic_commitment_rank);

    let rc_commitment_with_aux =
        recursive_commit(&crs, &config.commitment_recursion, &basic_commitment.data);

    let rc_commitment = rc_commitment_with_aux.most_inner_commitment().clone();

    let commitment_with_aux = CommitmentWithAux {
        rc_commitment_with_aux,
        witness_i16: None,
    };

    (commitment_with_aux, rc_commitment)
}
