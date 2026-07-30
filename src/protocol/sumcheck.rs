//! Sumcheck Protocol Implementation
//!
//! This module has been refactored into a more maintainable structure under `sumchecks/`.
//! The sumcheck protocol is complex and involves many types of constraints (CommitmentFold-NormCheck),
//! each with its own sumcheck gadgets and verification logic.
//!
//! **Constraint Types Overview:**
//!
//! - **CommitmentFold**: Basic commitment correctness - verifies `CK · folded_witness = commitment · fold_challenge`
//! - **InnerEvalFold**: Inner evaluation consistency - verifies opening RHS matches witness evaluation
//! - **OuterEvalClaim**: Outer evaluation consistency - verifies opening produces the claimed scalar result
//! - **CoarseProj**: Projection validity (block-diagonal) - verifies projection image is correctly computed from witness
//! - **FineProj**: Projection validity (Kronecker) - verifies `c^T (I ⊗ P) · witness = c^T projection_image · fold_challenge`
//! - **ComVerify**: Recursive commitment well-formedness - verifies the entire recursive commitment
//!   tree structure (internal layer parent-child consistency + leaf layer anchoring to public values)
//! - **NormCheck**: Witness norm check - verifies `<combined_witness, conjugated_witness> = norm_claim`
//!
//! **Module Organization:**
//!
//! - `sumchecks::context`: Type definitions for all sumcheck contexts (CommitmentFold-ComVerify).
//!   Each type represents a different semantic constraint in the protocol (commitment
//!   correctness, opening consistency, projection validity, recursive commitment
//!   well-formedness).
//!
//! - `sumchecks::builder`: Construction of sumcheck contexts. The `init_sumcheck` function
//!   wires together all the constraint gadgets, loads CRS data, and prepares the
//!   decomposition/recomposition machinery.
//!
//! - `sumchecks::helpers`: Utility functions for common operations (tensor products,
//!   projection coefficient computation, CK row loading, prefix selectors, composition
//!   sumchecks). These helpers encapsulate repeated patterns and make the builder code
//!   more readable.
//!
//! - `sumchecks::runner`: The main `sumcheck` function that executes a full protocol
//!   round. This is currently written as a test/simulation (with assertions at each
//!   step), but can be adapted for interactive prover/verifier operation.
//!
//! - `sumchecks::mod`: Module glue and exports.
//!
//! **Public API:**
//!
//! For backward compatibility, this file re-exports the main types and functions that
//! external code depends on. If you're looking to understand or modify the sumcheck
//! logic, start with `sumchecks/builder.rs` (for setup) and `sumchecks/runner.rs`
//! (for execution).

// Re-export the public API
pub use crate::protocol::sumchecks::{
    builder::init_sumcheck,
    context::{
        CoarseProjSumcheckContext, ComVerifyLayerSumcheckContext,
        ComVerifyOutputLayerSumcheckContext, ComVerifySumcheckContext,
        CommitmentFoldSumcheckContext, InnerEvalFoldSumcheckContext, OuterEvalClaimSumcheckContext,
        SumcheckContext,
    },
    runner::sumcheck,
};
