//! The chain configuration. `chain` defines the round chain (`Config`: Sumcheck, Intermediate or
//! Simple rounds), the per-round configs and the proof types, re-exported here; `generator`
//! derives a chain with its witness placements, and `params` holds the `Instantiation` a run
//! takes.

mod chain;
pub mod crs;
pub mod generator;
pub mod params;

pub use chain::*;
