#![feature(coerce_unsized, unsize)]
#![allow(static_mut_refs)] // keep to suppress warning
pub mod common;
pub mod hexl;
pub mod protocol;
#[cfg(feature = "incomplete-rexl")]
pub mod tfhe;
