//! Tracing-based profiling infrastructure for rokoko.
//!
//! Call [`setup_tracing`] once at binary startup with the desired output
//! formats. Hold the returned [`TracingGuards`] alive for the duration of the
//! program; dropping them flushes pending trace data.
//!
//! Modeled on `jolt-profiling`. Output layers are added incrementally:
//! [`TracingFormat::Default`] now selects an indented hierarchical
//! [`ConsoleLayer`]; Chrome JSON and snapshot layers land in checkpoint 3.

mod console;
mod setup;

pub use console::ConsoleLayer;
pub use setup::{setup_tracing, TracingFormat, TracingGuards};
