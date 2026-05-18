use std::any::Any;

use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::{prelude::*, registry::Registry, EnvFilter, Layer};

use crate::console::ConsoleLayer;
use crate::log::LogLayer;
use crate::snapshot::SnapshotLayer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TracingFormat {
    Default,
    Chrome,
    Snapshot,
}

#[must_use = "guards must be held alive for the duration of profiling"]
pub struct TracingGuards(#[allow(dead_code)] Vec<Box<dyn Any>>);

/// Panics if called more than once — the global subscriber can only be set once.
pub fn setup_tracing(
    formats: &[TracingFormat],
    trace_name: &str,
    features: &str,
) -> TracingGuards {
    let filter =
        || EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    // `ROKOKO_PROFILE_FOCUS=commit,sumcheck` restricts the console summary and
    // snapshot to those subtrees; Chrome JSON stays unfiltered (Perfetto scopes visually).
    let focus: Vec<String> = std::env::var("ROKOKO_PROFILE_FOCUS")
        .ok()
        .map(|s| {
            s.split(',')
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .map(String::from)
                .collect()
        })
        .unwrap_or_default();

    let mut layers: Vec<Box<dyn Layer<Registry> + Send + Sync>> = Vec::new();
    let mut guards: Vec<Box<dyn Any>> = Vec::new();

    layers.push(LogLayer.with_filter(filter()).boxed());

    if formats.contains(&TracingFormat::Default) {
        let (console_layer, console_guard) = ConsoleLayer::new(focus.clone());
        layers.push(console_layer.with_filter(filter()).boxed());
        guards.push(Box::new(console_guard));
    }

    if formats.contains(&TracingFormat::Chrome) {
        let chrome_path = format!("bench_results/traces/{trace_name}.json");
        let _ = std::fs::create_dir_all("bench_results/traces");
        let (chrome_layer, chrome_guard) = ChromeLayerBuilder::new()
            .file(&chrome_path)
            .include_args(true)
            .build();
        layers.push(chrome_layer.with_filter(filter()).boxed());
        guards.push(Box::new(chrome_guard));
    }

    if formats.contains(&TracingFormat::Snapshot) {
        let (snapshot_layer, snapshot_guard) =
            SnapshotLayer::new(trace_name, features, focus);
        layers.push(snapshot_layer.with_filter(filter()).boxed());
        guards.push(Box::new(snapshot_guard));
    }

    tracing_subscriber::registry().with(layers).init();

    TracingGuards(guards)
}
