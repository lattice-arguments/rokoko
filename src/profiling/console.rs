use std::collections::{HashMap, HashSet};
use std::io::{self, Write as _};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tracing::span::{Attributes, Id};
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::{LookupSpan, SpanRef};

/// `n == tok`, or `n` starts with `"{tok}::"`.
pub(crate) fn matches_token(n: &str, tok: &str) -> bool {
    n == tok || n.strip_prefix(tok).is_some_and(|rest| rest.starts_with("::"))
}

/// Empty `focus` means no filter (everything matches).
pub(crate) fn is_in_focus<S>(span: &SpanRef<'_, S>, focus: &[String]) -> bool
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    focus.is_empty() || outermost_matching_ancestor(span, focus).is_some()
}

fn round_base(name: &str) -> Option<&str> {
    let idx = name.rfind("_round")?;
    let end = idx + "_round".len();
    match name.as_bytes().get(end) {
        None | Some(b'_') => Some(&name[..end]),
        _ => None,
    }
}

fn same_round_chain(a: &str, b: &str) -> bool {
    a == b || matches!((round_base(a), round_base(b)), (Some(x), Some(y)) if x == y)
}

fn linear_depth<S>(span: &SpanRef<'_, S>) -> usize
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    let mut depth = 0;
    let mut child: Option<String> = None;
    for ancestor in span.scope() {
        let name = ancestor.name();
        if child.as_deref().is_some_and(|c| !same_round_chain(c, name)) {
            depth += 1;
        }
        child = Some(name.to_string());
    }
    depth
}

fn outermost_matching_ancestor<S>(span: &SpanRef<'_, S>, tokens: &[String]) -> Option<String>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    let mut root = None;
    for ancestor in span.scope() {
        let n = ancestor.name();
        if tokens.iter().any(|tok| matches_token(n, tok)) {
            root = Some(n.to_string());
        }
    }
    root
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

#[derive(Default)]
struct PhaseData {
    total: Duration,
    lines: Vec<(Instant, String)>,
    rounds: Vec<(Instant, String, Duration)>,
}

type LinearBuffers = HashMap<String, PhaseData>;

pub struct ConsoleLayer {
    edges: Arc<Mutex<EdgeMap>>,
    root_order: Arc<Mutex<Vec<String>>>,
    linear_buffers: Arc<Mutex<LinearBuffers>>,
    focus: Vec<String>,
    linear: bool,
}

pub struct ConsoleSummaryGuard {
    edges: Arc<Mutex<EdgeMap>>,
    root_order: Arc<Mutex<Vec<String>>>,
    linear_buffers: Arc<Mutex<LinearBuffers>>,
}

impl ConsoleLayer {
    pub fn new(focus: Vec<String>, linear: bool) -> (Self, ConsoleSummaryGuard) {
        let edges = Arc::new(Mutex::new(HashMap::new()));
        let root_order = Arc::new(Mutex::new(Vec::new()));
        let linear_buffers = Arc::new(Mutex::new(HashMap::new()));
        (
            ConsoleLayer {
                edges: Arc::clone(&edges),
                root_order: Arc::clone(&root_order),
                linear_buffers: Arc::clone(&linear_buffers),
                focus,
                linear,
            },
            ConsoleSummaryGuard {
                edges,
                root_order,
                linear_buffers,
            },
        )
    }
}

struct Timing {
    start: Instant,
    recursive_child_ns: u128,
}

impl<S> Layer<S> for ConsoleLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, _attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let span = ctx.span(id).expect("span exists at on_new_span");
        span.extensions_mut().insert(Timing {
            start: Instant::now(),
            recursive_child_ns: 0,
        });
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let span = ctx.span(&id).expect("span exists at on_close");
        let (start, elapsed, recursive_child_ns) = {
            let ext = span.extensions();
            let Some(timing) = ext.get::<Timing>() else {
                return;
            };
            (timing.start, timing.start.elapsed(), timing.recursive_child_ns)
        };
        let parent = span.parent();
        let parent_name = parent.as_ref().map(|p| p.name().to_string());
        let child_name = span.name().to_string();

        if let Some(parent) = &parent {
            if same_round_chain(parent.name(), span.name()) {
                if let Some(pt) = parent.extensions_mut().get_mut::<Timing>() {
                    pt.recursive_child_ns += elapsed.as_nanos();
                }
            }
        }

        if !is_in_focus(&span, &self.focus) {
            return;
        }

        if self.linear {
            let per_round =
                Duration::from_nanos(elapsed.as_nanos().saturating_sub(recursive_child_ns) as u64);
            let root_name = span
                .scope()
                .last()
                .map(|s| s.name().to_string())
                .unwrap_or_else(|| child_name.clone());
            let is_root = parent_name.is_none();

            {
                let mut buffers = self
                    .linear_buffers
                    .lock()
                    .expect("linear_buffers lock poisoned");
                let phase = buffers.entry(root_name).or_default();
                if is_root {
                    phase.total = per_round;
                } else {
                    let depth = linear_depth(&span);
                    let indent = "  ".repeat(depth);
                    let width = NAME_END.saturating_sub(2 * depth);
                    let line = format!(
                        "{indent}{name:<width$}  {elapsed:>elapsed_w$}",
                        indent = indent,
                        name = child_name,
                        width = width,
                        elapsed = format_duration(per_round),
                        elapsed_w = TIME_WIDTH,
                    );
                    phase.lines.push((start, line));
                }
                if round_base(&child_name).is_some() {
                    phase.rounds.push((start, child_name.clone(), per_round));
                }
            }

            if is_root {
                let mut order = self.root_order.lock().expect("root_order lock poisoned");
                if !order.iter().any(|n| n == &child_name) {
                    order.push(child_name.clone());
                }
            }
            return;
        }

        let mut edges = self.edges.lock().expect("edges lock poisoned");
        let entry = edges
            .entry(EdgeKey {
                parent: parent_name.clone(),
                child: child_name.clone(),
            })
            .or_default();
        entry.total_ns += elapsed.as_nanos();
        entry.calls += 1;
        if parent_name.is_none() {
            let mut order = self.root_order.lock().expect("root_order lock poisoned");
            if !order.iter().any(|n| n == &child_name) {
                order.push(child_name.clone());
            }
        }
    }
}

const NAME_END: usize = 48;
const TIME_WIDTH: usize = 10;
const CALLS_WIDTH: usize = 6;
const PCT_WIDTH: usize = 7;
const HEADER_WIDTH: usize = NAME_END + 2 + TIME_WIDTH;

impl Drop for ConsoleSummaryGuard {
    fn drop(&mut self) {
        let root_order = self.root_order.lock().expect("root_order lock poisoned").clone();
        if root_order.is_empty() {
            return;
        }
        let edges = self.edges.lock().expect("edges lock poisoned").clone();
        let mut linear_buffers = self
            .linear_buffers
            .lock()
            .expect("linear_buffers lock poisoned");

        let mut out = io::stdout().lock();
        let _ = writeln!(out);

        let roots: Vec<(String, u128)> = root_order
            .into_iter()
            .map(|name| {
                let total = edges
                    .get(&EdgeKey {
                        parent: None,
                        child: name.clone(),
                    })
                    .map(|agg| agg.total_ns)
                    .unwrap_or(0);
                (name, total)
            })
            .collect();

        for (i, (root, total_ns)) in roots.iter().enumerate() {
            if i > 0 {
                let _ = writeln!(out);
            }

            if let Some(mut phase) = linear_buffers.remove(root) {
                write_phase_header(&mut out, root, phase.total);

                phase.lines.sort_by_key(|(start, _)| *start);
                for (_, line) in &phase.lines {
                    let _ = writeln!(out, "{line}");
                }

                if !phase.rounds.is_empty() {
                    phase.rounds.sort_by_key(|(start, _, _)| *start);
                    let _ = writeln!(out, "  rounds:");
                    let mut rounds_total = Duration::ZERO;
                    for (i, (_, name, dur)) in phase.rounds.iter().enumerate() {
                        rounds_total += *dur;
                        let label = format!("round {}  {name}", i + 1);
                        let _ = writeln!(
                            out,
                            "    {label:<label_w$}  {time:>time_w$}",
                            label = label,
                            label_w = NAME_END.saturating_sub(4),
                            time = format_duration(*dur),
                            time_w = TIME_WIDTH,
                        );
                    }
                    let base = phase
                        .rounds
                        .first()
                        .and_then(|(_, name, _)| round_base(name))
                        .unwrap_or("round");
                    let _ = writeln!(
                        out,
                        "    {label:<label_w$}  {time:>time_w$}",
                        label = format!("total {base}"),
                        label_w = NAME_END.saturating_sub(4),
                        time = format_duration(rounds_total),
                        time_w = TIME_WIDTH,
                    );
                }
                continue;
            }

            write_phase_header(&mut out, root, Duration::from_nanos(*total_ns as u64));

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

fn write_phase_header(out: &mut impl io::Write, root: &str, total: Duration) {
    let phase = root.to_uppercase();
    let time = format_duration(total);
    let prefix = format!("=== {phase} ");
    let suffix = format!(" {time}");
    let filler_count = HEADER_WIDTH.saturating_sub(prefix.len() + suffix.len());
    let _ = writeln!(out, "{prefix}{}{suffix}", "=".repeat(filler_count));
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
