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
//!   `target/profiles/{name}.json` (loads in <https://ui.perfetto.dev/>).
//! - [`TracingFormat::Snapshot`]: aggregated span totals at
//!   `bench/snapshots/{name}.json` for diff-friendly PR evidence.

mod console;
mod setup;
mod snapshot;

pub use console::ConsoleLayer;
pub use setup::{setup_tracing, TracingFormat, TracingGuards};
pub use snapshot::{SnapshotGuard, SnapshotLayer};
