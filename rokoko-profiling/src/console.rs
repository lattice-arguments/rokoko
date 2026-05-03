//! Bespoke console layer: indented hierarchical span timing on stdout.
//!
//! Each span emits one line on close. Indentation = nesting depth (two spaces
//! per ancestor). Span attributes (e.g. `depth=0`, `round=1`) are appended
//! after the name. Elapsed time is rendered with units (s, ms, μs, ns) chosen
//! by magnitude so the output reads cleanly across the full dynamic range of
//! the protocol — microsecond gadgets through tens-of-seconds top-level phases.

use std::fmt::Write as _;
use std::io::{self, Write as _};
use std::time::{Duration, Instant};

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

pub struct ConsoleLayer;

impl ConsoleLayer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ConsoleLayer {
    fn default() -> Self {
        Self::new()
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
        // depth = ancestor count (root span sits at depth 0)
        let depth = span.scope().skip(1).count();
        let indent = "  ".repeat(depth);
        let mut line = String::with_capacity(64);
        let _ = write!(
            line,
            "{indent}{name}{attrs}  {elapsed}",
            name = span.name(),
            attrs = timing.attrs,
            elapsed = format_duration(elapsed),
        );
        let mut out = io::stdout().lock();
        let _ = writeln!(out, "{line}");
    }
}

/// Walks the structured key/value pairs attached to a span and serializes
/// them inline (` depth=0 round=5`) for console rendering. Named to avoid
/// collision with the cryptographic notion of "field" used throughout rokoko.
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
