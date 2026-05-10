//! Renders just the `message` field of each event, no prefix. Span context
//! is shown by [`crate::ConsoleLayer`]; `fmt::layer`'s ancestor chain would
//! be unreadable under the recursive `prover_round` span.

use std::fmt::Write as _;
use std::io::{self, Write as _};

use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, Layer};

pub struct LogLayer;

impl<S: Subscriber> Layer<S> for LogLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut msg = MessageVisitor(String::new());
        event.record(&mut msg);
        let mut out = io::stdout().lock();
        let _ = writeln!(out, "{}", msg.0);
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
