use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
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
    focus: Vec<String>,
}

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
        let path = PathBuf::from(format!("profiles/{trace_name}/snapshot.json"));
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
    env!("GIT_SHA").to_string()
}

fn now_iso8601() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

fn machine_string() -> String {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "?".to_string());
    let kernel = sysinfo::System::kernel_version()
        .unwrap_or_else(|| "unknown".to_string());
    let os = sysinfo::System::name().unwrap_or_else(|| std::env::consts::OS.to_string());
    format!("{os} {kernel} {} / {cores} cores", std::env::consts::ARCH)
}
