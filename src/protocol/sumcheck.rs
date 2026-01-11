//! Sumcheck Protocol Implementation
//!
//! This module has been refactored into a more maintainable structure under `sumchecks/`.
//! The sumcheck protocol is complex and involves many types of constraints (Type0-Type4),
//! each with its own sumcheck gadgets and verification logic.
//!
//! **Constraint Types Overview:**
//!
//! - **Type0**: Basic commitment correctness - verifies `CK · folded_witness = commitment · fold_challenge`
//! - **Type1**: Inner evaluation consistency - verifies opening RHS matches witness evaluation
//! - **Type2**: Outer evaluation consistency - verifies opening produces the claimed scalar result
//! - **Type3**: Projection validity - verifies projection image is correctly computed from witness
//! - **Type4**: Recursive commitment well-formedness - verifies the entire recursive commitment
//!   tree structure (internal layer parent-child consistency + leaf layer anchoring to public values)
//!
//! **Module Organization:**
//!
//! - `sumchecks::context`: Type definitions for all sumcheck contexts (Type0-Type4).
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
        SumcheckContext, Type0SumcheckContext, Type1SumcheckContext, Type2SumcheckContext,
        Type3SumcheckContext, Type4LayerSumcheckContext, Type4OutputLayerSumcheckContext,
        Type4SumcheckContext,
    },
    runner::sumcheck,
};
