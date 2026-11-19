//! Round-boundary capture: `ProverBoundary` and `VerifierBoundary` hold each party's state at
//! the start of a sumcheck round (config, witness or commitment root, claims, evaluation
//! points, transcript), and `BoundaryCapture` tells `prover_round` and `verifier_round` after
//! which round to stop the chain and fill them in.

use std::num::NonZeroUsize;

use crate::common::hash::HashWrapper;
use crate::common::matrix::VerticallyAlignedMatrix;
use crate::ring::ring_arithmetic::RingElement;
use crate::commitment::RecursiveCommitmentWithAux;
use crate::rok::config::SumcheckConfig;

#[derive(Debug)]
pub struct ProverBoundary {
    pub config: SumcheckConfig,
    pub witness: VerticallyAlignedMatrix<RingElement>,
    pub commitment: RecursiveCommitmentWithAux,
    pub claims: [RingElement; 2],
    pub evaluation_points: Vec<RingElement>,
    pub transcript: HashWrapper,
}

#[derive(Debug)]
pub struct VerifierBoundary {
    pub config: SumcheckConfig,
    pub commitment_root: Vec<RingElement>,
    pub claims: [RingElement; 2],
    pub evaluation_points: Vec<RingElement>,
    pub transcript: HashWrapper,
}

pub struct BoundaryCapture<'a, B> {
    pub cut: NonZeroUsize,
    pub slot: &'a mut Option<B>,
}

impl<'a, B> BoundaryCapture<'a, B> {
    pub fn is_at_cut(&self) -> bool {
        self.cut.get() == 1
    }

    pub fn advance(self) -> Option<Self> {
        NonZeroUsize::new(self.cut.get() - 1).map(|cut| BoundaryCapture { cut, slot: self.slot })
    }
}
