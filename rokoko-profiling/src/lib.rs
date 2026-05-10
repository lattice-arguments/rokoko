//! `tracing`-based profiling for rokoko. Call [`setup_tracing`] once at
//! startup, hold the returned [`TracingGuards`] for the program's lifetime.
//! See `bench_results/PROFILING.md` for the user-facing workflow.

mod console;
mod log;
mod setup;
mod snapshot;

pub use console::{ConsoleLayer, ConsoleSummaryGuard};
pub use log::LogLayer;
pub use setup::{setup_tracing, TracingFormat, TracingGuards};
pub use snapshot::{SnapshotGuard, SnapshotLayer};
