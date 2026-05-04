//! Console output layer.
//!
//! Two outputs:
//!
//! 1. **Live stream**: as the program runs, span-close events with depth ≤
//!    [`MAX_LIVE_DEPTH`] are emitted to stdout in compact form so progress is
//!    visible without flooding the terminal with the full span tree.
//!
//! 2. **End-of-run summary**: when the layer's guard drops, a hierarchical
//!    tree of every span observed during the run is printed, derived from
//!    parent/child edges tracked at `on_close` time. Cycles introduced by
//!    recursive spans (e.g. `prover_round → next_witness_and_recurse →
//!    prover_round`) are detected and elided rather than expanded.
//!
//! The full per-instance detail still lives in the Chrome JSON; the snapshot
//! JSON still has the flat by-name aggregates. This layer's job is to make
//! the human-readable view useful at a glance.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::io::{self, Write as _};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

/// Live-stream output is suppressed for spans below this depth. Roots are at
/// depth 0; their children at 1; grandchildren at 2. Beyond that, lines are
/// dropped from the terminal but still tracked for the summary.
const MAX_LIVE_DEPTH: usize = 2;

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
struct EdgeKey {
    /// `None` indicates a root-level span (no surrounding span).
    parent: Option<String>,
    child: String,
}

#[derive(Default, Clone)]
struct EdgeAggregate {
    total_ns: u128,
    calls: u64,
}

type EdgeMap = HashMap<EdgeKey, EdgeAggregate>;

pub struct ConsoleLayer {
    edges: Arc<Mutex<EdgeMap>>,
}

/// RAII guard that prints the end-of-run summary when dropped. Must be held
/// alive for the duration of profiling — `setup_tracing` packs it into
/// [`crate::TracingGuards`].
pub struct ConsoleSummaryGuard {
    edges: Arc<Mutex<EdgeMap>>,
}

impl ConsoleLayer {
    pub fn new() -> (Self, ConsoleSummaryGuard) {
        let edges = Arc::new(Mutex::new(HashMap::new()));
        (
            ConsoleLayer {
                edges: Arc::clone(&edges),
            },
            ConsoleSummaryGuard { edges },
        )
    }
}

struct Timing {
    start: Instant,
    attrs: String,
}

impl<S> Layer<S> for ConsoleLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let span = ctx.span(id).expect("span exists at on_new_span");
        let mut buf = String::new();
        attrs.record(&mut AttrVisitor(&mut buf));
        span.extensions_mut().insert(Timing {
            start: Instant::now(),
            attrs: buf,
        });
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let span = ctx.span(&id).expect("span exists at on_close");
        let ext = span.extensions();
        let Some(timing) = ext.get::<Timing>() else {
            return;
        };
        let elapsed = timing.start.elapsed();
        let depth = span.scope().skip(1).count();
        let parent_name = span.parent().map(|p| p.name().to_string());
        let child_name = span.name().to_string();

        // Track edge for the end-of-run summary, regardless of live depth.
        {
            let mut edges = self.edges.lock().expect("edges lock poisoned");
            let entry = edges
                .entry(EdgeKey {
                    parent: parent_name.clone(),
                    child: child_name.clone(),
                })
                .or_default();
            entry.total_ns += elapsed.as_nanos();
            entry.calls += 1;
        }

        // Live stream: only emit spans at the top of the tree.
        if depth > MAX_LIVE_DEPTH {
            return;
        }

        let indent = "  ".repeat(depth);
        let mut line = String::with_capacity(64);
        let _ = write!(
            line,
            "{indent}{name:<width$}  {elapsed}{attrs}",
            indent = indent,
            name = child_name,
            width = 40usize.saturating_sub(2 * depth),
            elapsed = format_duration(elapsed),
            attrs = timing.attrs,
        );
        let mut out = io::stdout().lock();
        let _ = writeln!(out, "{line}");
    }
}

impl Drop for ConsoleSummaryGuard {
    fn drop(&mut self) {
        let edges = self.edges.lock().expect("edges lock poisoned").clone();
        if edges.is_empty() {
            return;
        }
        let mut out = io::stdout().lock();
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "─── profile summary ───────────────────────────────────────────"
        );

        // Roots: edges with no parent. Sort by total_ns descending for stable,
        // useful ordering (biggest phase first).
        let mut roots: Vec<(String, u128)> = edges
            .iter()
            .filter(|(k, _)| k.parent.is_none())
            .map(|(k, v)| (k.child.clone(), v.total_ns))
            .collect();
        roots.sort_by(|a, b| b.1.cmp(&a.1));
        roots.dedup_by(|a, b| a.0 == b.0);

        for (root, _) in &roots {
            let mut visited = HashSet::new();
            print_subtree(&mut out, &edges, root, None, None, 0, &mut visited);
        }

        let _ = writeln!(
            out,
            "───────────────────────────────────────────────────────────────"
        );
    }
}

/// Walks the call graph from `name` down through every edge whose parent is
/// `name`. All times and call counts come from the **edge** `(parent, name)` —
/// not from a global by-name aggregate. A span called from multiple parents
/// (e.g. `commit::basic_internal`, called from both `commit::basic` during
/// witness commit and from `prover_round::next_witness_and_recurse` during
/// prover rounds) thus shows the correct slice of its time under each parent,
/// and percentages relative to parent never exceed 100%.
fn print_subtree(
    out: &mut impl io::Write,
    edges: &EdgeMap,
    name: &str,
    parent: Option<&str>,
    parent_total_ns: Option<u128>,
    depth: usize,
    visited: &mut HashSet<String>,
) {
    let indent = "  ".repeat(depth);

    // Cycle: this name is already on the path from root to here. Emit a stub
    // and stop; the full subtree is documented above.
    if visited.contains(name) {
        let _ = writeln!(out, "{indent}…{name} (recursive — see above)");
        return;
    }

    let edge_key = EdgeKey {
        parent: parent.map(String::from),
        child: name.to_string(),
    };
    let agg = edges.get(&edge_key).cloned().unwrap_or_default();
    let time = format_duration(Duration::from_nanos(agg.total_ns as u64));
    let calls_suffix = if agg.calls > 1 {
        format!(" × {}", agg.calls)
    } else {
        String::new()
    };
    let pct_suffix = match parent_total_ns {
        Some(parent_ns) if parent_ns > 0 && agg.total_ns <= parent_ns => {
            let pct = (agg.total_ns as f64 / parent_ns as f64) * 100.0;
            format!("   {pct:>4.1}%")
        }
        _ => String::new(),
    };

    let _ = writeln!(
        out,
        "{indent}{name:<width$}  {time:>8}{calls}{pct}",
        indent = indent,
        name = name,
        width = 48usize.saturating_sub(2 * depth),
        time = time,
        calls = calls_suffix,
        pct = pct_suffix,
    );

    visited.insert(name.to_string());

    // Children: distinct child names from all (parent=name, *) edges, sorted
    // by total_ns descending so heavy hitters surface first.
    let mut children: Vec<(String, u128)> = edges
        .iter()
        .filter(|(k, _)| k.parent.as_deref() == Some(name))
        .map(|(k, v)| (k.child.clone(), v.total_ns))
        .collect();
    children.sort_by(|a, b| b.1.cmp(&a.1));
    children.dedup_by(|a, b| a.0 == b.0);

    for (child, _) in children {
        print_subtree(
            out,
            edges,
            &child,
            Some(name),
            Some(agg.total_ns),
            depth + 1,
            visited,
        );
    }

    visited.remove(name);
}

struct AttrVisitor<'a>(&'a mut String);

impl Visit for AttrVisitor<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let _ = write!(self.0, " {}={:?}", field.name(), value);
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        let _ = write!(self.0, " {}={}", field.name(), value);
    }
    fn record_i64(&mut self, field: &Field, value: i64) {
        let _ = write!(self.0, " {}={}", field.name(), value);
    }
    fn record_u64(&mut self, field: &Field, value: u64) {
        let _ = write!(self.0, " {}={}", field.name(), value);
    }
    fn record_bool(&mut self, field: &Field, value: bool) {
        let _ = write!(self.0, " {}={}", field.name(), value);
    }
}

fn format_duration(d: Duration) -> String {
    let ns = d.as_nanos();
    if ns >= 1_000_000_000 {
        format!("{:.2} s", d.as_secs_f64())
    } else if ns >= 1_000_000 {
        format!("{} ms", ns / 1_000_000)
    } else if ns >= 1_000 {
        format!("{} μs", ns / 1_000)
    } else {
        format!("{ns} ns")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_duration_picks_units() {
        assert_eq!(format_duration(Duration::from_nanos(500)), "500 ns");
        assert_eq!(format_duration(Duration::from_nanos(2_500)), "2 μs");
        assert_eq!(format_duration(Duration::from_millis(7)), "7 ms");
        assert_eq!(format_duration(Duration::from_secs_f64(1.234)), "1.23 s");
    }
}
