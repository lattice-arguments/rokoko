use std::any::Any;

use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::{prelude::*, registry::Registry, EnvFilter, Layer};

use crate::console::ConsoleLayer;
use crate::log::LogLayer;
use crate::snapshot::SnapshotLayer;

/// Output format for the tracing subscriber stack.
///
/// Multiple variants can be selected at once; their outputs are independent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TracingFormat {
    /// Indented hierarchical console output via [`crate::ConsoleLayer`].
    Default,
    /// Chrome/Perfetto JSON trace at `target/profiles/{trace_name}.json`.
    Chrome,
    /// Aggregated span totals at `bench/snapshots/{trace_name}.json`.
    Snapshot,
}

/// Opaque container for tracing flush guards. Must be held alive for the
/// duration of profiling — dropping it flushes pending trace data and writes
/// the snapshot JSON.
#[must_use = "guards must be held alive for the duration of profiling"]
pub struct TracingGuards(#[allow(dead_code)] Vec<Box<dyn Any>>);

/// Initialize the global tracing subscriber.
///
/// Always installs a `RUST_LOG`-driven log layer (default filter: `info`) plus
/// any output layers requested via `formats`. The same `RUST_LOG` filter is
/// applied to every layer, so a focused selector (e.g.
/// `RUST_LOG=rokoko::sumcheck=info`) prunes all three artifacts consistently.
///
/// `features` is a comma-separated string describing the active rokoko Cargo
/// features (e.g. `"p-26,incomplete-rexl,unsafe-sumcheck"`). It is recorded in
/// the snapshot metadata so future diffs can warn on feature mismatch.
///
/// # Panics
/// Panics if called more than once — the global subscriber can only be set once.
pub fn setup_tracing(
    formats: &[TracingFormat],
    trace_name: &str,
    features: &str,
) -> TracingGuards {
    let filter =
        || EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let mut layers: Vec<Box<dyn Layer<Registry> + Send + Sync>> = Vec::new();
    let mut guards: Vec<Box<dyn Any>> = Vec::new();

    layers.push(LogLayer::new().with_filter(filter()).boxed());

    if formats.contains(&TracingFormat::Default) {
        let (console_layer, console_guard) = ConsoleLayer::new();
        layers.push(console_layer.with_filter(filter()).boxed());
        guards.push(Box::new(console_guard));
    }

    if formats.contains(&TracingFormat::Chrome) {
        let chrome_path = format!("target/profiles/{trace_name}.json");
        let _ = std::fs::create_dir_all("target/profiles");
        let (chrome_layer, chrome_guard) = ChromeLayerBuilder::new()
            .file(&chrome_path)
            .include_args(true)
            .build();
        layers.push(chrome_layer.with_filter(filter()).boxed());
        guards.push(Box::new(chrome_guard));
    }

    if formats.contains(&TracingFormat::Snapshot) {
        let (snapshot_layer, snapshot_guard) = SnapshotLayer::new(trace_name, features);
        layers.push(snapshot_layer.with_filter(filter()).boxed());
        guards.push(Box::new(snapshot_guard));
    }

    tracing_subscriber::registry().with(layers).init();

    TracingGuards(guards)
}
