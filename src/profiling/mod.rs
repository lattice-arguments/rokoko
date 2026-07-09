mod console;
mod log;
mod setup;
#[cfg(feature = "profile")]
mod snapshot;

pub use setup::{print_artifact_paths, setup, timestamp_for_filename, TracingGuards};
