use crate::{
    common::ring_arithmetic::{QuadraticExtension, RingElement},
    protocol::sumcheck_utils::polynomial::Polynomial,
};

pub struct Proof {
    pub rc_commitment: Vec<RingElement>,
    pub rc_opening: Vec<RingElement>,
    pub rc_projection_image: Vec<RingElement>,
    pub norm_claim: RingElement,
    pub sumcheck_transcript: Vec<Polynomial<QuadraticExtension>>,
    pub claim_over_witness: RingElement,
    pub claim_over_witness_conjugate: RingElement,
}
