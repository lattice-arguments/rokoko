//! Logging and profiling on `tracing-subscriber`. `setup` installs the global subscriber for a
//! run and returns `TracingGuards`; dropping them prints or writes the reports. Every run gets
//! `LogLayer`, which prints event messages to stdout. Feature `events` adds `ConsoleLayer`, a
//! timing summary printed at exit; feature `profile` adds `SnapshotLayer`, which writes
//! `snapshot.json` and the Chrome-trace `trace.json` to `run_dir()`. All layers share the
//! `RUST_LOG` level filter `RustLog`.

#[cfg(feature = "events")]
mod console;
mod filter;
#[cfg(feature = "profile")]
mod jsonw;
mod log;
mod setup;
#[cfg(feature = "profile")]
#[cfg(feature = "profile")]
mod snapshot;
#[cfg(feature = "profile")]

#[cfg(feature = "profile")]
pub use setup::{print_artifact_paths, run_dir};
pub use setup::{setup, TracingGuards};
