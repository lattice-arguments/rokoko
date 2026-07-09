mod setup;
#[cfg(feature = "events")]
mod console;
mod log;
#[cfg(feature = "profile")]
mod snapshot;

pub use setup::{print_artifact_paths, setup, trace_name, timestamp_for_filename, TracingGuards};
