//! Arithmetic over `R_q` and its slot field; `build.rs` generates `config` from `ring-params/`.

pub mod config;
pub mod finite_ring;
pub mod helpers;
pub mod kernels;
pub mod ring_arithmetic;
