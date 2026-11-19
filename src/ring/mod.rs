//! Ring arithmetic over `R_q = Z_q[X]/(X^N + 1)`, the lowest layer of the library.
//! Start with `ring_arithmetic`, which defines `RingElement`, `FieldExtension` and their
//! arithmetic. It builds on `config`, the constants `build.rs` generates from the ring spec,
//! and on `kernels`, which wraps the `incomplete-rexl` kernels and is the only place the
//! library calls them. `helpers` adds shared constants and `Z_q` helpers on top, and
//! `finite_ring` is the `FiniteRing` trait both element types implement.

pub mod config;
pub mod finite_ring;
pub mod helpers;
pub mod kernels;
pub mod ring_arithmetic;
