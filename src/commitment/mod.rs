//! The Ajtai commitment scheme. `ajtai` commits ring vectors, one basic commitment per witness
//! column or recursively level by level, and lays out where each committed component sits in the
//! next round's witness; its items are re-exported here. `crs` holds the commitment keys.

mod ajtai;
pub mod crs;

pub use ajtai::*;
