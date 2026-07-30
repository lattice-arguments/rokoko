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
