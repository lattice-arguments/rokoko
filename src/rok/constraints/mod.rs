//! The constraints each round of the chain proves, built as sumcheck claims from the round's
//! config and run with the engine in `crate::sumcheck`. `sumcheck_round` serves a Sumcheck
//! round and `intermediate_round` an Intermediate round; both follow builder -> context ->
//! loader -> runner, each with a `_verifier` twin.

pub mod intermediate_round;
pub mod sumcheck_round;
