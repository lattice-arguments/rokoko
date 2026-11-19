//! Diagnostics and test hooks beside the protocol: `estimator`, the SIS/RSIS bit-security
//! estimator; `debug_hardness` (feature `debug-hardness`), which the prover calls to track each
//! round's norms and their RSIS hardness; and `boundary`, which lets a test stop the prover and
//! verifier after a given round and capture their state.

pub mod boundary;
#[cfg(feature = "debug-hardness")]
pub mod debug_hardness;
pub mod estimator;
