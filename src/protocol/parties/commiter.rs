use crate::{
    common::{matrix::VerticallyAlignedMatrix, ring_arithmetic::RingElement},
    protocol::{commitment::CommitmentWithAux, config::SumcheckConfig, crs::CRS},
};

#[cfg(not(feature = "parallel-commitment"))]
use crate::protocol::commitment::{commit_basic, recursive_commit};

#[cfg(feature = "parallel-commitment")]
use crate::protocol::commitment::{
    commit_basic_parallel as commit_basic, recursive_commit_parallel as recursive_commit,
};

pub fn commit(
    crs: &CRS,
    config: &SumcheckConfig,
    witness: &VerticallyAlignedMatrix<RingElement>,
) -> (CommitmentWithAux, Vec<RingElement>) {
    let basic_commitment = {
        let _s = tracing::info_span!("commit::basic").entered();
        commit_basic(&crs, &witness, config.basic_commitment_rank)
    };

    let rc_commitment_with_aux = {
        let _s = tracing::info_span!("commit::recursive").entered();
        recursive_commit(&crs, &config.commitment_recursion, &basic_commitment.data)
    };

    let rc_commitment = rc_commitment_with_aux.most_inner_commitment().clone();

    let commitment_with_aux = CommitmentWithAux {
        rc_commitment_with_aux,
        witness_i16: None,
    };

    (commitment_with_aux, rc_commitment)
}
