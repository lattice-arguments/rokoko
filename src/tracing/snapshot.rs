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

use super::jsonw;

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
    args: String,
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
    args: String,
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
        let mut out = String::with_capacity(512);
        out.push_str("{\n  \"metadata\": {\n");
        jsonw::field_str(&mut out, "    ", "git_sha", &m.git_sha, false);
        jsonw::field_str(&mut out, "    ", "date", &m.date, false);
        jsonw::field_str(&mut out, "    ", "features", &m.features, false);
        jsonw::field_str(&mut out, "    ", "machine", &m.machine, true);
        out.push_str("  },\n  \"spans\": {\n");
        let mut names: Vec<&'static str> = st.aggregates.keys().copied().collect();
        names.sort_unstable();
        for (i, &name) in names.iter().enumerate() {
            let a = &st.aggregates[name];
            out.push_str("    ");
            jsonw::str_into(&mut out, name);
            out.push_str(&format!(
                ": {{\n      \"total_ns\": {},\n      \"calls\": {}\n    }}",
                a.total_ns, a.calls
            ));
            if i + 1 < names.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("  }\n}\n");
        out
    }

    fn trace_json(&self, st: &Shared) -> String {
        let mut out = String::with_capacity(256 + st.events.len() * 128);
        out.push_str("[\n");
        for (i, e) in st.events.iter().enumerate() {
            out.push_str("  {\"name\": ");
            jsonw::str_into(&mut out, e.name);
            out.push_str(&format!(
                ", \"cat\": \"span\", \"ph\": \"X\", \"pid\": 1, \"tid\": {}, \"ts\": {}, \"dur\": {}",
                e.tid, e.ts_us, e.dur_us
            ));
            if !e.args.is_empty() {
                out.push_str(", \"args\": ");
                out.push_str(&e.args);
            }
            out.push('}');
            if i + 1 < st.events.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("]\n");
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
    out: String,
}

impl ArgVisitor {
    fn push(&mut self, key: &str, value: &str) {
        if self.out.is_empty() {
            self.out.push('{');
        } else {
            self.out.push_str(", ");
        }
        jsonw::str_into(&mut self.out, key);
        self.out.push_str(": ");
        jsonw::str_into(&mut self.out, value);
    }

    fn finish(mut self) -> String {
        if !self.out.is_empty() {
            self.out.push('}');
        }
        self.out
    }
}

impl Visit for ArgVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.push(field.name(), value);
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.push(field.name(), &format!("{value:?}"));
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
