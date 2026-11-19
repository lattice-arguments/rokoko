//! Sumchecks of an Intermediate round: one batched sumcheck over the decomposed folded witness
//! `w̃ = G^{-1}(W c)`, proving CommitmentFold, InnerEvalFold, FineProj and NormCheck claims
//! against a basic commitment, with no recursive commitment and no ComVerify.
//! The files follow the builder -> context -> loader -> runner pattern of `rok::constraints::sumcheck_round`,
//! each with a `_verifier` twin, and reuse its helpers. Read `context.rs` first, then
//! `builder.rs`.

pub mod builder;
pub mod builder_verifier;
pub mod context;
pub mod context_verifier;
pub mod loader;
pub mod loader_verifier;
pub mod runner;
pub mod runner_verifier;
