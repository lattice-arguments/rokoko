use std::any::Any;

use tracing_subscriber::{prelude::*, EnvFilter};

/// Output format for the tracing subscriber stack.
///
/// Variants beyond the always-on log layer are wired in subsequent checkpoints;
/// this initial version recognises them but produces no extra output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TracingFormat {
    /// Indented hierarchical console output (wired in checkpoint 2).
    Default,
    /// Chrome/Perfetto JSON trace file (wired in checkpoint 3).
    Chrome,
    /// Aggregated span totals serialized to JSON (wired in checkpoint 3).
    Snapshot,
}

/// Opaque container for tracing flush guards. Must be held alive for the
/// duration of profiling — dropping it flushes pending trace data.
#[must_use = "guards must be held alive for the duration of profiling"]
pub struct TracingGuards(#[allow(dead_code)] Vec<Box<dyn Any>>);

/// Initialize the global tracing subscriber.
///
/// Always installs a `RUST_LOG`-driven log layer (default filter: `info`).
/// Output layers selected by `formats` are added in subsequent checkpoints.
///
/// # Panics
/// Panics if called more than once — the global subscriber can only be set once.
pub fn setup_tracing(_formats: &[TracingFormat], _trace_name: &str) -> TracingGuards {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let log_layer = tracing_subscriber::fmt::layer()
        .compact()
        .with_target(false)
        .with_filter(filter);

    tracing_subscriber::registry().with(log_layer).init();

    TracingGuards(Vec::new())
}
