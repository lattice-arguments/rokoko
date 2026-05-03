use std::any::Any;

use tracing_subscriber::{prelude::*, registry::Registry, EnvFilter, Layer};

use crate::console::ConsoleLayer;

/// Output format for the tracing subscriber stack.
///
/// `Chrome` and `Snapshot` are recognised but currently no-ops; they are wired
/// in checkpoint 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TracingFormat {
    /// Indented hierarchical console output via [`ConsoleLayer`].
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
/// Always installs a `RUST_LOG`-driven log layer (default filter: `info`) plus
/// any output layers requested via `formats`.
///
/// # Panics
/// Panics if called more than once — the global subscriber can only be set once.
pub fn setup_tracing(formats: &[TracingFormat], _trace_name: &str) -> TracingGuards {
    let filter =
        || EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let mut layers: Vec<Box<dyn Layer<Registry> + Send + Sync>> = Vec::new();
    let guards: Vec<Box<dyn Any>> = Vec::new();

    layers.push(
        tracing_subscriber::fmt::layer()
            .compact()
            .with_target(false)
            .with_filter(filter())
            .boxed(),
    );

    if formats.contains(&TracingFormat::Default) {
        layers.push(ConsoleLayer::new().with_filter(filter()).boxed());
    }

    tracing_subscriber::registry().with(layers).init();

    TracingGuards(guards)
}
