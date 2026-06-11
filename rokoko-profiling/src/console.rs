use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::io::{self, Write as _};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::{LookupSpan, SpanRef};

/// Matching rule per token: `n == tok` or `n` starts with `"{tok}::"`. Empty
/// focus = no filter.
pub(crate) fn is_in_focus<S>(span: &SpanRef<'_, S>, focus: &[String]) -> bool
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    if focus.is_empty() {
        return true;
    }
    for ancestor in span.scope() {
        let n = ancestor.name();
        for tok in focus {
            if n == tok || n.starts_with(&format!("{tok}::")) {
                return true;
            }
        }
    }
    false
}

const MAX_LIVE_DEPTH: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Tree,
    Linear,
}

impl Mode {
    fn from_env() -> Self {
        match std::env::var("ROKOKO_TRACE_MODE")
            .ok()
            .as_deref()
            .map(str::trim)
        {
            Some("linear") => Mode::Linear,
            _ => Mode::Tree,
        }
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
struct EdgeKey {
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
    focus: Vec<String>,
    mode: Mode,
}

pub struct ConsoleSummaryGuard {
    edges: Arc<Mutex<EdgeMap>>,
    mode: Mode,
}

impl ConsoleLayer {
    pub fn new(focus: Vec<String>) -> (Self, ConsoleSummaryGuard) {
        let edges = Arc::new(Mutex::new(HashMap::new()));
        let mode = Mode::from_env();
        (
            ConsoleLayer {
                edges: Arc::clone(&edges),
                focus,
                mode,
            },
            ConsoleSummaryGuard { edges, mode },
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

        if !is_in_focus(&span, &self.focus) {
            return;
        }

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

        if self.mode == Mode::Tree && depth > MAX_LIVE_DEPTH {
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

const NAME_END: usize = 48;
const TIME_WIDTH: usize = 10;
const CALLS_WIDTH: usize = 6;
const PCT_WIDTH: usize = 7;
const HEADER_WIDTH: usize = NAME_END + 2 + TIME_WIDTH;

impl Drop for ConsoleSummaryGuard {
    fn drop(&mut self) {
        if self.mode == Mode::Linear {
            return;
        }
        let edges = self.edges.lock().expect("edges lock poisoned").clone();
        if edges.is_empty() {
            return;
        }
        let mut out = io::stdout().lock();
        let _ = writeln!(out);

        let mut roots: Vec<(String, u128)> = edges
            .iter()
            .filter(|(k, _)| k.parent.is_none())
            .map(|(k, v)| (k.child.clone(), v.total_ns))
            .collect();
        roots.sort_by(|a, b| b.1.cmp(&a.1));
        roots.dedup_by(|a, b| a.0 == b.0);

        for (i, (root, total_ns)) in roots.iter().enumerate() {
            if i > 0 {
                let _ = writeln!(out);
            }
            let phase = root.to_uppercase();
            let time = format_duration(Duration::from_nanos(*total_ns as u64));
            let prefix = format!("=== {phase} ");
            let suffix = format!(" {time}");
            let filler_count =
                HEADER_WIDTH.saturating_sub(prefix.len() + suffix.len());
            let _ = writeln!(out, "{prefix}{}{suffix}", "=".repeat(filler_count));

            let mut visited = HashSet::new();
            visited.insert(root.clone());
            let mut shown_edges: HashSet<(Option<String>, String)> = HashSet::new();

            let mut children: Vec<(String, u128)> = edges
                .iter()
                .filter(|(k, _)| k.parent.as_deref() == Some(root.as_str()))
                .map(|(k, v)| (k.child.clone(), v.total_ns))
                .collect();
            children.sort_by(|a, b| b.1.cmp(&a.1));
            children.dedup_by(|a, b| a.0 == b.0);

            for (child, _) in children {
                print_subtree(
                    &mut out,
                    &edges,
                    &child,
                    Some(root),
                    Some(*total_ns),
                    1,
                    &mut visited,
                    &mut shown_edges,
                );
            }
        }

        let _ = writeln!(out);
    }
}

/// All times and call counts are read from the `(parent, child)` edge — not
/// from a by-name aggregate — so a span called from multiple parents shows the
/// correct slice under each parent and percent-of-parent never exceeds 100%.
fn print_subtree(
    out: &mut impl io::Write,
    edges: &EdgeMap,
    name: &str,
    parent: Option<&str>,
    parent_total_ns: Option<u128>,
    depth: usize,
    visited: &mut HashSet<String>,
    shown_edges: &mut HashSet<(Option<String>, String)>,
) {
    let indent = "  ".repeat(depth);
    let display = strip_common_prefix(name, parent);

    // Re-expanding the same edge elsewhere in the tree would print identical
    // numbers a second time and mislead the reader, so stub on repeat.
    let edge_id = (parent.map(String::from), name.to_string());
    if shown_edges.contains(&edge_id) {
        let _ = writeln!(out, "{indent}…{display}");
        return;
    }

    if visited.contains(name) {
        let _ = writeln!(out, "{indent}…{display}");
        return;
    }

    shown_edges.insert(edge_id);

    let edge_key = EdgeKey {
        parent: parent.map(String::from),
        child: name.to_string(),
    };
    let agg = edges.get(&edge_key).cloned().unwrap_or_default();
    let time = format_duration(Duration::from_nanos(agg.total_ns as u64));
    let calls_str = if agg.calls > 1 {
        format!("× {}", agg.calls)
    } else {
        String::new()
    };
    let pct_str = match parent_total_ns {
        Some(parent_ns) if parent_ns > 0 && agg.total_ns <= parent_ns => {
            let pct = (agg.total_ns as f64 / parent_ns as f64) * 100.0;
            format!("{pct:.1}%")
        }
        _ => String::new(),
    };

    let name_field_width = NAME_END.saturating_sub(2 * depth);
    let _ = writeln!(
        out,
        "{indent}{name:<name_w$}  {time:>time_w$}  {calls:>calls_w$}  {pct:>pct_w$}",
        indent = indent,
        name = display,
        name_w = name_field_width,
        time = time,
        time_w = TIME_WIDTH,
        calls = calls_str,
        calls_w = CALLS_WIDTH,
        pct = pct_str,
        pct_w = PCT_WIDTH,
    );

    visited.insert(name.to_string());

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
            shown_edges,
        );
    }

    visited.remove(name);
}

/// Strip the longest `::`-bounded prefix shared between `name` and `parent`.
fn strip_common_prefix<'a>(name: &'a str, parent: Option<&str>) -> &'a str {
    let Some(parent) = parent else {
        return name;
    };
    let mut bytes_to_skip = 0usize;
    let mut name_iter = name.split("::");
    let mut parent_iter = parent.split("::");
    loop {
        let (Some(n_seg), Some(p_seg)) = (name_iter.next(), parent_iter.next()) else {
            break;
        };
        if n_seg != p_seg {
            break;
        }
        bytes_to_skip += n_seg.len() + 2;
    }
    if bytes_to_skip == 0 || bytes_to_skip > name.len() {
        name
    } else {
        &name[bytes_to_skip..]
    }
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

    #[test]
    fn strip_common_prefix_strips_segments() {
        assert_eq!(strip_common_prefix("commit", None), "commit");

        assert_eq!(strip_common_prefix("commit::basic", Some("commit")), "basic");
        assert_eq!(
            strip_common_prefix("commit::decompose_witness", Some("commit")),
            "decompose_witness"
        );

        assert_eq!(
            strip_common_prefix("commit::basic_internal", Some("commit::basic")),
            "basic_internal"
        );

        assert_eq!(
            strip_common_prefix("sumcheck::round::poly", Some("sumcheck::round")),
            "poly"
        );

        // cross-phase: no shared first segment → keep full name
        assert_eq!(
            strip_common_prefix(
                "commit::basic_internal",
                Some("prover_round::next_witness_and_recurse"),
            ),
            "commit::basic_internal"
        );

        // shared name prefix without `::` boundary → not a real segment match
        assert_eq!(
            strip_common_prefix("prover_round", Some("prover")),
            "prover_round"
        );
    }
}
