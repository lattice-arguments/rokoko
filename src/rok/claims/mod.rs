//! The SNARK mode: claims about a committed witness, proved down to the evaluation claims the
//! commitment chain starts from.

mod language;
mod lowering;

pub use crate::common::hash::HashWrapper as Transcript;
pub use language::*;
pub use lowering::InitialSumcheckProof as ClaimsProof;
pub use lowering::SnarkClaim as Claim;
pub use lowering::{embed_fe, eq_layers_fe, fe_one_minus, tensor_at};
pub use lowering::{prove_claims, verify_claims, ChainInputs, ClaimExpr, WitnessShape};
