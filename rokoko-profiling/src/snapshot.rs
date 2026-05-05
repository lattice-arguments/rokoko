//! Snapshot layer: aggregate span totals into a small diff-friendly JSON.
//!
//! Hooks `on_new_span` to record start time and `on_close` to accumulate
//! `(total_ns, calls)` keyed by span name. Aggregation is **flat** — span
//! instances at different recursion depths are summed into one entry. The
//! Chrome JSON preserves per-instance detail; this layer trades that for
//! diff-ability and PR-friendliness.
//!
//! On guard drop, serializes to `bench_results/snapshots/{trace_name}.json`
//! with metadata (git SHA, ISO 8601 date, active rokoko features, machine
//! string).

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::Serialize;
use tracing::span::{Attributes, Id};
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

#[derive(Default, Serialize)]
struct SpanAggregate {
    total_ns: u128,
    calls: u64,
}

#[derive(Serialize)]
struct SnapshotMetadata {
    git_sha: String,
    date: String,
    features: String,
    machine: String,
}

#[derive(Serialize)]
struct Snapshot<'a> {
    metadata: &'a SnapshotMetadata,
    spans: &'a HashMap<String, SpanAggregate>,
}

type Aggregates = Arc<Mutex<HashMap<String, SpanAggregate>>>;

pub struct SnapshotLayer {
    aggregates: Aggregates,
    /// Subtree-focus filter. If non-empty, only spans whose name (or any
    /// ancestor's name) matches at least one token are aggregated.
    focus: Vec<String>,
}

/// Holds the snapshot output path and metadata. Drop writes the JSON.
pub struct SnapshotGuard {
    aggregates: Aggregates,
    path: PathBuf,
    metadata: SnapshotMetadata,
}

impl SnapshotLayer {
    pub fn new(
        trace_name: &str,
        features: &str,
        focus: Vec<String>,
    ) -> (Self, SnapshotGuard) {
        let aggregates: Aggregates = Arc::new(Mutex::new(HashMap::new()));
        let path = PathBuf::from(format!("bench_results/snapshots/{trace_name}.json"));
        let metadata = SnapshotMetadata {
            git_sha: git_sha(),
            date: now_iso8601(),
            features: features.to_string(),
            machine: machine_string(),
        };
        let layer = SnapshotLayer {
            aggregates: Arc::clone(&aggregates),
            focus,
        };
        let guard = SnapshotGuard {
            aggregates,
            path,
            metadata,
        };
        (layer, guard)
    }
}

struct Timing {
    start: Instant,
}

impl<S> Layer<S> for SnapshotLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, _attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let span = ctx.span(id).expect("span exists at on_new_span");
        span.extensions_mut().insert(Timing {
            start: Instant::now(),
        });
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let span = ctx.span(&id).expect("span exists at on_close");
        let ext = span.extensions();
        let Some(timing) = ext.get::<Timing>() else {
            return;
        };
        if !crate::console::is_in_focus(&span, &self.focus) {
            return;
        }
        let elapsed_ns = timing.start.elapsed().as_nanos();
        let name = span.name().to_string();
        let mut agg = self.aggregates.lock().expect("aggregates lock poisoned");
        let entry = agg.entry(name).or_default();
        entry.total_ns += elapsed_ns;
        entry.calls += 1;
    }
}

impl Drop for SnapshotGuard {
    fn drop(&mut self) {
        if let Some(parent) = self.path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let agg = self.aggregates.lock().expect("aggregates lock poisoned");
        let snapshot = Snapshot {
            metadata: &self.metadata,
            spans: &agg,
        };
        match serde_json::to_string_pretty(&snapshot) {
            Ok(json) => {
                if let Err(e) = fs::write(&self.path, json) {
                    eprintln!(
                        "rokoko-profiling: snapshot write failed at {}: {e}",
                        self.path.display()
                    );
                }
            }
            Err(e) => eprintln!("rokoko-profiling: snapshot serialize failed: {e}"),
        }
    }
}

fn git_sha() -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn now_iso8601() -> String {
    Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn machine_string() -> String {
    let uname = Command::new("uname")
        .args(["-srm"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let cores = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "?".to_string());
    format!("{uname} / {cores} cores")
}
