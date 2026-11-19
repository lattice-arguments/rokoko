//! RoKoko: a lattice-based succinct argument system, run as a PCS (a Ξ^lin start) or as a SNARK
//! (`rok::claims`, a Ξ^sum start). The binary, `src/bin/rokoko`, runs either end to end and
//! picks the SNARK with feature `snark`.
//!
//! The library is layered, lowest first. Each layer imports only from the layers below it and
//! from `tracing` (logging and profiling, beside them); the test in `tests/layers.rs` enforces
//! it.
//! - `ring`: ring arithmetic, constants and the `FiniteRing` trait; the only caller of the
//!   `incomplete-rexl` kernels.
//! - `common`: operations on ring vectors (decomposition, norms, matrices, structured rows,
//!   sampling) and the Fiat-Shamir transcript with its short challenges.
//! - `sumcheck`: the generic sumcheck engine and its gadgets.
//! - `commitment`: the Ajtai commitment scheme and its common reference string.
//! - `rok`: the reductions of knowledge: openings, projections, folding, the round
//!   constraints, the claim language, the chain config and the proof encoding.
//! - `tools`: diagnostics and test hooks: the security estimator, per-round hardness tracking,
//!   and round-boundary capture.
//! - The parties, on top: `committer` commits the witness, and `prover` and `verifier` walk the
//!   round chain.
//!
//! The binary's `executor.rs` runs a whole chain end to end (`execute`, `execute_snark`); start
//! reading there. The test module `tests.rs` runs single rounds end to end.

#![feature(coerce_unsized, unsize)]
#![allow(static_mut_refs)] // keep to suppress warning
pub mod commitment;
pub mod committer;
pub mod common;
pub mod prover;
pub mod ring;
pub mod rok;
pub mod sumcheck;
pub mod tools;
pub mod tracing;
pub mod verifier;

#[cfg(test)]
mod tests;
