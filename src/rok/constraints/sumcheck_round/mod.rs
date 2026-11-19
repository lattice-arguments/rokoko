//! Sumcheck gadgets of a round: one per paper constraint (CommitmentFold-NormCheck).
//! The sumcheck engine they are built on is `crate::sumcheck`.
//!
//! A Sumcheck round proves all its constraints with one batched sumcheck over the packed
//! witness `ŵ` (`combined_witness`). The folded witness, the commitment, opening and projection
//! values and the recursive-commitment levels are placed in `ŵ` as digit planes, which the
//! gadgets recompose with prefix selectors and radix weights.
//!
//! **Constraint types:**
//!
//! - **CommitmentFold**: `CK · folded_witness = commitment · fold_challenge`, one per key row
//! - **InnerEvalFold**: `<l_j, folded_witness> = opening.rhs[j] · fold_challenge`, per opening
//! - **OuterEvalClaim**: `<opening.rhs[j], r_j> = t_j`, the public claim, per opening
//! - **CoarseProj** (Π^proj-c): `<flattener · (I ⊗ J), folded_witness> =
//!   <fold_challenge ⊗ flattener, projection_image>`
//! - **FineProj** (Π^proj-f): per batch, `c^T (I ⊗ J) · folded_witness = c^T projection_image ·
//!   fold_challenge` over coefficients, plus consistency of the batched-projection commitment
//!   with the constant-term commitment
//! - **ComVerify**: per recursion tree (commitment, opening, projection; two for
//!   `Projection::Fine`, none under `Projection::Skip`), each internal level's key rows times
//!   its input equal its commitment
//!   recomposed from the child level, and the leaf level equals the public value
//!   (`rc_commitment`, `rc_*_inner`)
//! - **NormCheck**: `<ŵ, conj(ŵ)> = norm_claim`, the same inner product restricted to the
//!   innermost commitment of every tree, and, under `exact_projection_norm`, one weighted over
//!   the projection tree's level-0 digit planes
//!
//! **How the files fit together:** each side runs builder -> context -> loader -> runner.
//! `init_sumcheck` (`builder.rs`) wires the gadgets into a `SumcheckContext` (`context.rs`)
//! once per round; `sumcheck` (`runner.rs`) fills it through `loader.rs` and folds it one
//! variable per sumcheck round. The `_verifier` files are evaluation-only twins:
//! `init_verifier` builds the same tree in the same output order, and `sumcheck_verifier`
//! replays the round's transcript, checks the round polynomials and evaluates the tree at the
//! final point.
//! `helpers.rs` serves both sides and the Intermediate round (`rok::constraints::intermediate_round`).
//! Read `context.rs` first, then `builder.rs`.

pub mod builder;
pub mod builder_verifier;
pub mod context;
pub mod context_verifier;
pub mod helpers;
pub mod loader;
pub mod loader_verifier;
pub mod runner;
pub mod runner_verifier;
