//! Bare log-event layer.
//!
//! Renders one line per `tracing::info!` / `debug!` / `trace!` event, prefixed
//! with the level. No span-context prefix and no timestamp — the alternative is
//! `fmt::layer` which (for the Compact format we'd want) prepends the full
//! span ancestor chain to every event. With a recursive `prover_round` span,
//! that prefix runs to hundreds of characters, drowning the actual message.
//!
//! Span timing/structure already lives in the [`crate::ConsoleLayer`] summary;
//! this layer's job is just to render protocol diagnostic events as readable
//! lines when the user opts into them via `RUST_LOG=debug` (or `=trace`).

use std::fmt::Write as _;
use std::io::{self, Write as _};

use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, Layer};

pub struct LogLayer;

impl LogLayer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LogLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Subscriber> Layer<S> for LogLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut msg = MessageVisitor(String::new());
        event.record(&mut msg);
        let level = event.metadata().level();
        let mut out = io::stdout().lock();
        let _ = writeln!(out, "{level} {}", msg.0);
    }
}

struct MessageVisitor(String);

impl Visit for MessageVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.0.push_str(value);
        }
    }
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(&mut self.0, "{value:?}");
        }
    }
}
