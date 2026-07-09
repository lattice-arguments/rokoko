use std::any::Any;
use std::process::Command;

use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::{prelude::*, registry::Registry, EnvFilter, Layer};

#[cfg(feature = "profile")]
use tracing_chrome::ChromeLayerBuilder;

use super::console::ConsoleLayer;
use super::log::LogLayer;
#[cfg(feature = "profile")]
use super::snapshot::SnapshotLayer;

#[must_use = "guards must be held alive for the duration of profiling"]
pub struct TracingGuards(#[allow(dead_code)] Vec<Box<dyn Any>>);

/// Install a tracing subscriber stack from two orthogonal flags:
///
/// - `events`: console summary (`ConsoleLayer`).
/// - `profile`: file artifacts (`ChromeLayer` JSON + `SnapshotLayer` JSON).
///
/// Note that `ConsoleLayer` aggregates by (parent, child) edge (where time went); while
/// `SnapshotLayer` aggregates by span name (total time anywhere).
///
/// Level filtering is `info` by default; the env `RUST_LOG` is set to control the logging level
///
/// Panics if called more than once — the global subscriber can only be set once.
pub fn setup(
    events: bool,
    profile: bool,
    trace_name: &str,
    features: &str,
) -> TracingGuards {
    let filter =
        || EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let parse_csv = |var: &str| -> Vec<String> {
        std::env::var(var)
            .ok()
            .map(|s| {
                s.split(',')
                    .map(str::trim)
                    .filter(|t| !t.is_empty())
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default()
    };

    let focus = parse_csv("PROFILE_FOCUS");

    let max_level = <EnvFilter as Layer<Registry>>::max_level_hint(&filter())
        .unwrap_or(LevelFilter::INFO);
    let linear = max_level >= LevelFilter::DEBUG;

    let mut layers: Vec<Box<dyn Layer<Registry> + Send + Sync>> = Vec::new();
    let mut guards: Vec<Box<dyn Any>> = Vec::new();

    layers.push(LogLayer.with_filter(filter()).boxed());

    if events {
        let (console_layer, console_guard) = ConsoleLayer::new(focus.clone(), linear);
        layers.push(console_layer.with_filter(filter()).boxed());
        guards.push(Box::new(console_guard));
    }

    #[cfg(feature = "profile")]
    if profile {
        let run_dir = format!("profiles/{trace_name}");
        let _ = std::fs::create_dir_all(&run_dir);
        let chrome_path = format!("{run_dir}/trace.json");
        let (chrome_layer, chrome_guard) = ChromeLayerBuilder::new()
            .file(&chrome_path)
            .include_args(true)
            .build();
        layers.push(chrome_layer.with_filter(filter()).boxed());
        guards.push(Box::new(chrome_guard));

        let (snapshot_layer, snapshot_guard) =
            SnapshotLayer::new(trace_name, features, focus);
        layers.push(snapshot_layer.with_filter(filter()).boxed());
        guards.push(Box::new(snapshot_guard));
    }

    // `profile`-only file artifacts are compiled out without that feature.
    #[cfg(not(feature = "profile"))]
    let _ = (profile, trace_name, features);

    if !layers.is_empty() {
        tracing_subscriber::registry().with(layers).init();
    }

    TracingGuards(guards)
}

/// UTC `YYYYMMDD-HHMMSS` — filesystem-safe, lex-sortable.
pub fn timestamp_for_filename() -> String {
    Command::new("date")
        .args(["-u", "+%Y%m%d-%H%M%S"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

pub fn print_artifact_paths(trace_base: &str) {
    println!(
        "\n\
        Profile written to profiles/{trace_base}/\n\
        \n  \
        trace.json     (Chrome trace — view in Firefox Profiler / Perfetto)\n  \
        snapshot.json  (per-span totals + run metadata, for multi-run analysis)\n\
        \n\
        To view the trace, drag profiles/{trace_base}/trace.json into either:\n  \
        https://profiler.firefox.com/\n  \
        https://ui.perfetto.dev/"
    );
}
