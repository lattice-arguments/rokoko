//! Tracing-based profiling infrastructure for rokoko.
//!
//! Call [`setup_tracing`] once at binary startup with the desired output
//! formats. Hold the returned [`TracingGuards`] alive for the duration of the
//! program; dropping them flushes pending trace data and writes the snapshot
//! JSON.
//!
//! Modeled on `jolt-profiling`. Three output formats:
//! - [`TracingFormat::Default`]: indented hierarchical [`ConsoleLayer`].
//! - [`TracingFormat::Chrome`]: Chrome/Perfetto JSON timeline at
//!   `bench_results/traces/{name}.json` (loads in
//!   <https://ui.perfetto.dev/>).
//! - [`TracingFormat::Snapshot`]: aggregated span totals at
//!   `bench_results/snapshots/{name}.json` for diff-friendly PR evidence.

mod console;
mod log;
mod setup;
mod snapshot;

pub use console::{ConsoleLayer, ConsoleSummaryGuard};
pub use log::LogLayer;
pub use setup::{setup_tracing, TracingFormat, TracingGuards};
pub use snapshot::{SnapshotGuard, SnapshotLayer};
