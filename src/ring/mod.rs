//! Ring arithmetic: `RingElement`, `FieldExtension` and the ring constants generated
//! from the ring spec. `kernels` wraps the `incomplete-rexl` kernels and is the only
//! place the library calls them.

pub mod config;
pub mod finite_ring;
pub mod helpers;
pub mod kernels;
pub mod ring_arithmetic;
