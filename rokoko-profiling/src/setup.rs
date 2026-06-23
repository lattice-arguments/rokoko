use std::any::Any;
use std::process::Command;

use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::{prelude::*, registry::Registry, EnvFilter, Layer};

use crate::console::ConsoleLayer;
use crate::log::LogLayer;
use crate::snapshot::SnapshotLayer;

#[must_use = "guards must be held alive for the duration of profiling"]
pub struct TracingGuards(#[allow(dead_code)] Vec<Box<dyn Any>>);

/// Install a tracing subscriber stack from three orthogonal flags:
///
/// - `events`: console summary (`ConsoleLayer`). Empty `ROKOKO_LINEAR_PHASES`
///   defaults to `verifier` so the per-round verifier breakdown is on by default.
/// - `profile`: file artifacts (`ChromeLayer` JSON + `SnapshotLayer` JSON).
/// - `debug`: lower the level filter from `info` to `trace`, surfacing the
///   `tracing::trace!` memory-layout dumps. No-op without `events` or `profile`.
///
/// Panics if called more than once — the global subscriber can only be set once.
pub fn setup(
    events: bool,
    profile: bool,
    debug: bool,
    trace_name: &str,
    features: &str,
) -> TracingGuards {
    let level = if debug { "trace" } else { "info" };
    let filter =
        || EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));

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

    let focus = parse_csv("ROKOKO_PROFILE_FOCUS");

    let mut layers: Vec<Box<dyn Layer<Registry> + Send + Sync>> = Vec::new();
    let mut guards: Vec<Box<dyn Any>> = Vec::new();

    if events || profile {
        layers.push(LogLayer.with_filter(filter()).boxed());
    }

    if events {
        let mut linear_phases = parse_csv("ROKOKO_LINEAR_PHASES");
        if linear_phases.is_empty() {
            linear_phases.push("verifier".to_string());
        }
        let (console_layer, console_guard) = ConsoleLayer::new(focus.clone(), linear_phases);
        layers.push(console_layer.with_filter(filter()).boxed());
        guards.push(Box::new(console_guard));
    }

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
