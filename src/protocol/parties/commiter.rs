use crate::{
    common::{matrix::VerticallyAlignedMatrix, ring_arithmetic::RingElement},
    protocol::{commitment::CommitmentWithAux, config::SumcheckConfig, crs::CRS},
};

#[cfg(all(not(feature = "parallel"), not(feature = "crt-commitment")))]
use crate::protocol::commitment::commit_basic;
#[cfg(not(feature = "parallel"))]
use crate::protocol::commitment::recursive_commit;

#[cfg(all(feature = "parallel", not(feature = "crt-commitment")))]
use crate::protocol::commitment::commit_basic_parallel as commit_basic;
#[cfg(feature = "parallel")]
use crate::protocol::commitment::recursive_commit_parallel as recursive_commit;

/// The root commitment over the CRT basis; the recursion stays on the ring path, whose shapes
/// are too small to pay for a preprocessed key.
#[cfg(feature = "crt-commitment")]
use crate::protocol::commitment_crt::commit_basic as commit_basic_root;
#[cfg(not(feature = "crt-commitment"))]
use commit_basic as commit_basic_root;

pub fn commit(
    crs: &CRS,
    config: &SumcheckConfig,
    witness: &VerticallyAlignedMatrix<RingElement>,
) -> (CommitmentWithAux, Vec<RingElement>) {
    let basic_commitment = {
        let _s = tracing::info_span!("commit::basic").entered();
        commit_basic_root(&crs, &witness, config.basic_commitment_rank)
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
