//! Claims for a Ξ^sum start. `language` is the claim language, and `lowering` reduces such claims
//! to the Ξ^lin claims the chain starts from (`prove_claims`, `verify_claims`); both are
//! re-exported here, with the transcript as `Transcript`. Guide in `docs/claims.md`, demo in
//! `examples/claims.rs`.

mod language;
mod lowering;

pub use crate::common::hash::HashWrapper as Transcript;
pub use lowering::InitialSumcheckProof as ClaimsProof;
pub use lowering::SnarkClaim as Claim;
pub use lowering::{prove_claims, verify_claims, ChainInputs, ClaimExpr, WitnessShape};
pub use lowering::{embed_fe, eq_layers_fe, fe_one_minus, tensor_at};
pub use language::*;
