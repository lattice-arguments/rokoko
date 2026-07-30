#[cfg(feature = "events")]
mod console;
mod filter;
mod log;
mod setup;
#[cfg(feature = "profile")]
mod jsonw;
#[cfg(feature = "profile")]
mod snapshot;
#[cfg(feature = "profile")]
mod timefmt;

#[cfg(feature = "profile")]
pub use setup::{print_artifact_paths, run_dir};
pub use setup::{setup, TracingGuards};
