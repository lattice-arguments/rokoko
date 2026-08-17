use std::num::NonZeroUsize;

use crate::common::hash::HashWrapper;
use crate::common::matrix::VerticallyAlignedMatrix;
use crate::common::ring_arithmetic::RingElement;
use crate::protocol::commitment::RecursiveCommitmentWithAux;
use crate::protocol::config::SumcheckConfig;

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
