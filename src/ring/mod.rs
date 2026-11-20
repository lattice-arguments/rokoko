//! Ring arithmetic: `RingElement`, `FieldExtension` and the ring constants generated
//! from the ring spec. `kernels` provides raw-pointer wrappers for the `incomplete-rexl`
//! kernels and re-exports `SlotRing`, which `ring_arithmetic` also calls directly.

pub mod config;
pub mod finite_ring;
pub mod helpers;
pub mod kernels;
pub mod ring_arithmetic;
