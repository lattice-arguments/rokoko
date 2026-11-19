//! The reductions of knowledge. A run proves a chain of rounds, each reducing a Ξ^lin statement
//! (evaluation claims about a committed witness) to a smaller one. `config` defines the chain,
//! its proof types, the `Instantiation` a run takes and the CRS the chain needs. A sumcheck
//! round opens the witness (`open`), projects it (`proj`: Π^proj-c, Π^proj-f, or not at all
//! under `Projection::Skip`) and folds it (`fold`), then proves the resulting constraints with
//! the sumchecks in `constraints`. `claims` holds the claim language and the reduction of a
//! Ξ^sum start to the Ξ^lin claims the chain begins with, `evaluation_point_sampler` samples a
//! Ξ^lin start's points, and `wire` encodes proofs. The parties sit above this layer:
//! `crate::committer`, `crate::prover` and `crate::verifier`.

pub mod claims;
pub mod config;
pub mod constraints;
pub mod evaluation_point_sampler;
pub mod fold;
pub mod open;
pub mod proj;
pub mod wire;
