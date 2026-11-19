//! The projection RoKs `Π^proj-c` (`coarse`, on whole ring elements) and `Π^proj-f` (`fine`, on
//! the witness coefficients, with batching), the ternary projection matrix `J` they share
//! (`matrix`), and the `i16` kernels that apply it (`kernels`).

pub mod coarse;
pub mod fine;
pub mod kernels;
pub mod matrix;
