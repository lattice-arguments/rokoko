//! Tracing-based profiling infrastructure for rokoko.
//!
//! Call [`setup_tracing`] once at binary startup with the desired output
//! formats. Hold the returned [`TracingGuards`] alive for the duration of the
//! program; dropping them flushes pending trace data.
//!
//! Modeled on `jolt-profiling`. Output layers (`ConsoleLayer`, Chrome JSON,
//! `SnapshotLayer`) are added in subsequent checkpoints; this initial version
//! installs only a `RUST_LOG`-driven log layer so that `tracing` events from
//! library code surface during profile-enabled runs.

mod setup;

pub use setup::{setup_tracing, TracingFormat, TracingGuards};
