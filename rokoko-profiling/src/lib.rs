//! `tracing`-based profiling for rokoko. Call [`setup_tracing`] once at
//! startup and hold the returned [`TracingGuards`] for the program's lifetime.

mod console;
mod log;
mod setup;
mod snapshot;

pub use console::{ConsoleLayer, ConsoleSummaryGuard};
pub use log::LogLayer;
pub use setup::{setup_tracing, TracingFormat, TracingGuards};
pub use snapshot::{SnapshotGuard, SnapshotLayer};
