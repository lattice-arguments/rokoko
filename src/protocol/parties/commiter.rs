use crate::{
    common::{matrix::VerticallyAlignedMatrix, ring_arithmetic::RingElement},
    protocol::{commitment::CommitmentWithAux, config::SumcheckConfig, crs::CRS},
};

#[cfg(not(feature = "parallel"))]
use crate::protocol::commitment::{commit_basic, recursive_commit};

#[cfg(feature = "parallel")]
use crate::protocol::commitment::{
    commit_basic_parallel as commit_basic, recursive_commit_parallel as recursive_commit,
};

#[cfg(feature = "crt-commitment")]
use crate::protocol::project_coarse::Signed16RingElement;

/// The digits the CRT basis wants, when the caller already has them from the decomposition.
#[cfg(feature = "crt-commitment")]
pub type Digits<'a> = Option<&'a VerticallyAlignedMatrix<Signed16RingElement>>;
#[cfg(not(feature = "crt-commitment"))]
pub type Digits<'a> = Option<&'a ()>;

pub fn commit(
    crs: &CRS,
    config: &SumcheckConfig,
    witness: &VerticallyAlignedMatrix<RingElement>,
    digits: Digits,
) -> (CommitmentWithAux, Vec<RingElement>) {
    let basic_commitment = {
        let _s = tracing::info_span!("commit::basic").entered();
        basic(crs, config, witness, digits)
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

/// The root commitment over the CRT basis when the CRS carries a key of this shape, and over the
/// ring otherwise; the recursion always stays on the ring path, whose shapes are too small to pay
/// for a preprocessed key.
#[cfg(feature = "crt-commitment")]
fn basic(
    crs: &CRS,
    config: &SumcheckConfig,
    witness: &VerticallyAlignedMatrix<RingElement>,
    digits: Digits,
) -> crate::protocol::commitment::BasicCommitment {
    use crate::protocol::commitment_crt::{commit_basic_crt, commit_basic_crt_streaming};

    let rank = config.basic_commitment_rank;
    let fitting = crs
        .crt_root
        .as_ref()
        .filter(|(_, key)| key.rows == rank && key.n == witness.height);
    match (fitting, digits) {
        (Some((plan, key)), Some(digits)) => commit_basic_crt(key, digits, plan, rank),
        (Some((plan, key)), None) => commit_basic_crt_streaming(key, witness, plan, rank),
        (None, _) => commit_basic(crs, witness, rank),
    }
}

#[cfg(not(feature = "crt-commitment"))]
fn basic(
    crs: &CRS,
    config: &SumcheckConfig,
    witness: &VerticallyAlignedMatrix<RingElement>,
    _digits: Digits,
) -> crate::protocol::commitment::BasicCommitment {
    commit_basic(crs, witness, config.basic_commitment_rank)
}
