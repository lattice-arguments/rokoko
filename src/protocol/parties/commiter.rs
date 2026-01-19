use crate::{
    common::{matrix::VerticallyAlignedMatrix, ring_arithmetic::RingElement},
    protocol::{
        commitment::{
            commit_basic, recursive_commit, CommitmentWithAux, RecursiveCommitmentWithAux,
        },
        config::{ConfigBase, SumcheckConfig},
        crs::CRS,
        project::{prepare_i16_witness, Signed16RingElement},
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
    let witness_i16 = prepare_i16_witness(witness);

    let commitment_with_aux = CommitmentWithAux {
        rc_commitment_with_aux,
        witness_i16: Some(witness_i16),
    };

    (commitment_with_aux, rc_commitment)
}
