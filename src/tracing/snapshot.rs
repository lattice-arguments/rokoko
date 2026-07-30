// Span aggregation + artifact writing:
//   - snapshot.json  per-span totals + run metadata
//   - trace.json     Chrome Trace Event timeline (flat array of "ph":"X" events)

use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

use serde_json::{json, Map, Value};

/// Process-wide zero point for Chrome trace timestamps.
fn base() -> Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    *START.get_or_init(Instant::now)
}

static NEXT_TID: AtomicU64 = AtomicU64::new(1);
thread_local! {
    static TID: Cell<u64> = const { Cell::new(0) };
}

fn tid() -> u64 {
    TID.with(|t| {
        if t.get() == 0 {
            t.set(NEXT_TID.fetch_add(1, Ordering::Relaxed));
        }
        t.get()
    })
}

#[derive(Default)]
struct SpanAggregate {
    total_ns: u128,
    calls: u64,
}

struct SnapshotMetadata {
    git_sha: String,
    date: String,
    features: String,
    machine: String,
}

/// One completed span, in Chrome Trace Event terms.
struct ChromeEvent {
    name: &'static str,
    ts_us: u128,
    dur_us: u128,
    tid: u64,
    args: Map<String, Value>,
}

struct Shared {
    aggregates: HashMap<&'static str, SpanAggregate>,
    events: Vec<ChromeEvent>,
}

type State = Arc<Mutex<Shared>>;

pub struct SnapshotLayer {
    state: State,
}

pub struct SnapshotGuard {
    state: State,
    dir: PathBuf,
    metadata: SnapshotMetadata,
}

impl SnapshotLayer {
    pub fn new(run_dir: &str, features: &str) -> (Self, SnapshotGuard) {
        base(); // pin the zero point before any span opens
        let state: State = Arc::new(Mutex::new(Shared {
            aggregates: HashMap::new(),
            events: Vec::new(),
        }));
        let metadata = SnapshotMetadata {
            git_sha: git_sha(),
            date: {
                let now = time::OffsetDateTime::now_utc().replace_nanosecond(0).unwrap();
                now.format(&time::format_description::well_known::Rfc3339)
                    .expect("format metadata date")
            },
            features: features.to_string(),
            machine: machine_string(),
        };
        let layer = SnapshotLayer {
            state: Arc::clone(&state),
        };
        let guard = SnapshotGuard {
            state,
            dir: PathBuf::from(run_dir),
            metadata,
        };
        (layer, guard)
    }
}

struct Timing {
    start: Instant,
    ts_us: u128,
    tid: u64,
    args: Map<String, Value>,
}

impl<S> Layer<S> for SnapshotLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let span = ctx.span(id).expect("span exists at on_new_span");
        let start = Instant::now();
        let mut args = ArgVisitor::default();
        attrs.record(&mut args);
        span.extensions_mut().insert(Timing {
            start,
            ts_us: start.saturating_duration_since(base()).as_micros(),
            tid: tid(),
            args: args.finish(),
        });
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let span = ctx.span(&id).expect("span exists at on_close");
        let timing = { span.extensions_mut().remove::<Timing>() };
        let Some(timing) = timing else {
            return;
        };
        let elapsed_ns = timing.start.elapsed().as_nanos();
        let name = span.name();
        let mut st = self.state.lock().expect("state lock poisoned");
        let entry = st.aggregates.entry(name).or_default();
        entry.total_ns += elapsed_ns;
        entry.calls += 1;
        st.events.push(ChromeEvent {
            name,
            ts_us: timing.ts_us,
            dur_us: elapsed_ns / 1_000,
            tid: timing.tid,
            args: timing.args,
        });
    }
}

impl SnapshotGuard {
    fn snapshot_json(&self, st: &Shared) -> String {
        let m = &self.metadata;
        let spans: Map<String, Value> = st
            .aggregates
            .iter()
            .map(|(&name, a)| {
                (
                    name.to_string(),
                    json!({ "total_ns": a.total_ns as u64, "calls": a.calls }),
                )
            })
            .collect();
        let doc = json!({
            "metadata": {
                "git_sha": m.git_sha,
                "date": m.date,
                "features": m.features,
                "machine": m.machine,
            },
            "spans": spans,
        });
        let mut out = serde_json::to_string_pretty(&doc).expect("serialize snapshot.json");
        out.push('\n');
        out
    }

    fn trace_json(&self, st: &Shared) -> String {
        let events: Vec<Value> = st
            .events
            .iter()
            .map(|e| {
                let mut ev = json!({
                    "name": e.name,
                    "cat": "span",
                    "ph": "X",
                    "pid": 1,
                    "tid": e.tid,
                    "ts": e.ts_us as u64,
                    "dur": e.dur_us as u64,
                });
                if !e.args.is_empty() {
                    ev["args"] = Value::Object(e.args.clone());
                }
                ev
            })
            .collect();
        let mut out = serde_json::to_string(&Value::Array(events)).expect("serialize trace.json");
        out.push('\n');
        out
    }
}

impl Drop for SnapshotGuard {
    fn drop(&mut self) {
        let _ = fs::create_dir_all(&self.dir);
        let st = self.state.lock().expect("state lock poisoned");
        for (name, body) in [
            ("snapshot.json", self.snapshot_json(&st)),
            ("trace.json", self.trace_json(&st)),
        ] {
            let path = self.dir.join(name);
            if let Err(e) = fs::write(&path, body) {
                eprintln!("profiling: write failed at {}: {e}", path.display());
            }
        }
    }
}

/// Collects span fields into a JSON object, mirroring `tracing-chrome`'s `include_args`.
#[derive(Default)]
struct ArgVisitor {
    out: Map<String, Value>,
}

impl ArgVisitor {
    fn finish(self) -> Map<String, Value> {
        self.out
    }
}

impl Visit for ArgVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.out.insert(field.name().to_string(), Value::from(value));
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.out
            .insert(field.name().to_string(), Value::from(format!("{value:?}")));
    }
}

pub fn active_features() -> String {
    [
        cfg!(feature = "p-26").then_some("p-26"),
        cfg!(feature = "p-28").then_some("p-28"),
        cfg!(feature = "p-30").then_some("p-30"),
        cfg!(feature = "incomplete-rexl").then_some("incomplete-rexl"),
        cfg!(feature = "unsafe-sumcheck").then_some("unsafe-sumcheck"),
        cfg!(feature = "debug-hardness").then_some("debug-hardness"),
        cfg!(feature = "debug-decomp").then_some("debug-decomp"),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join(",")
}

fn git_sha() -> String {
    option_env!("GIT_SHA").unwrap_or("unknown").to_string()
}

fn machine_string() -> String {
    let cores = std::thread::available_parallelism()
        .map_or_else(|_| "?".to_string(), |n| n.get().to_string());
    let kernel = sysinfo::System::kernel_version().unwrap_or_else(|| "unknown".to_string());
    let os = sysinfo::System::name().unwrap_or_else(|| std::env::consts::OS.to_string());
    format!("{os} {kernel} {} / {cores} cores", std::env::consts::ARCH)
}
