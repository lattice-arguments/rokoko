use std::any::Any;

use tracing_subscriber::{prelude::*, registry::Registry, Layer};

use super::filter::RustLog;

#[cfg(feature = "profile")]
use std::sync::OnceLock;

#[cfg(feature = "events")]
use super::console::ConsoleLayer;
use super::log::LogLayer;
#[cfg(feature = "profile")]
use super::snapshot::SnapshotLayer;

#[must_use = "guards must be held alive for the duration of profiling"]
pub struct TracingGuards(#[allow(dead_code)] Vec<Box<dyn Any>>);

/// Install the global tracing subscriber. Can be only set once.
/// Two different layers are optionally selected based on the feature flags:
/// - `events`: console summary (`ConsoleLayer`).
/// - `profile`: file artifacts (`SnapshotLayer` writes both `snapshot.json` and the
///   Chrome-trace `trace.json`).
///
/// Note that `ConsoleLayer` aggregates by (parent, child) edge (where time went); while
/// `SnapshotLayer` aggregates by span name (total time anywhere).
///
/// Level filtering is `info` by default; the env `RUST_LOG` is set to control the logging level
///
/// Panics if called more than once — the global subscriber can only be set once.
pub fn setup() -> TracingGuards {
    let filter = RustLog::from_default_env();

    let mut layers: Vec<Box<dyn Layer<Registry> + Send + Sync>> = Vec::new();
    #[cfg_attr(
        not(any(feature = "events", feature = "profile")),
        allow(unused_mut)
    )]
    let mut guards: Vec<Box<dyn Any>> = Vec::new();

    layers.push(LogLayer.with_filter(filter).boxed());

    #[cfg(feature = "events")]
    {
        use tracing_subscriber::filter::LevelFilter;
        let linear = filter.max_level() >= LevelFilter::DEBUG;
        let (console_layer, console_guard) = ConsoleLayer::new(linear);
        layers.push(console_layer.with_filter(filter).boxed());
        guards.push(Box::new(console_guard));
    }

    #[cfg(feature = "profile")]
    {
        let features = super::snapshot::active_features();
        let dir = run_dir();
        std::fs::create_dir_all(dir).expect("create profile run dir");
        let (snapshot_layer, snapshot_guard) = SnapshotLayer::new(dir, &features);
        layers.push(snapshot_layer.with_filter(filter).boxed());
        guards.push(Box::new(snapshot_guard));
    }

    tracing_subscriber::registry().with(layers).init();

    TracingGuards(guards)
}

/// Directory holding run's artifacts: `profiles/<params>_<timestamp>/`.
#[cfg(feature = "profile")]
pub fn run_dir() -> &'static str {
    static RUN_DIR: OnceLock<String> = OnceLock::new();
    RUN_DIR.get_or_init(|| format!("profiles/{}_{}", trace_name(), timestamp_for_filename()))
}

/// UTC `YYYYMMDD-HHMMSS`
#[cfg(feature = "profile")]
fn timestamp_for_filename() -> String {
    let f = time::macros::format_description!("[year][month][day]-[hour][minute][second]");
    time::OffsetDateTime::now_utc()
        .format(f)
        .expect("format filename timestamp")
}

#[cfg(feature = "profile")]
pub fn print_artifact_paths(run_dir: &str) {
    println!(
        "\n\
        Profile written to {run_dir}/\n\
        \n  \
        trace.json     (Chrome trace — view in Firefox Profiler / Perfetto)\n  \
        snapshot.json  (per-span totals + run metadata, for multi-run analysis)\n\
        \n\
        To view the trace, drag {run_dir}/trace.json into either:\n  \
        https://profiler.firefox.com/\n  \
        https://ui.perfetto.dev/"
    );
}

/// Parameter set of the build, used to name the artifact directory.
#[cfg(feature = "profile")]
fn trace_name() -> &'static str {
    use crate::protocol::params::{compiled_size, SizeConfig};
    match compiled_size() {
        SizeConfig::Small => "p26",
        SizeConfig::Medium => "p28",
        SizeConfig::Large => "p30",
    }
}
