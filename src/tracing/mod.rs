#[cfg(feature = "events")]
mod console;
mod log;
mod setup;
#[cfg(feature = "profile")]
mod snapshot;

pub use setup::{print_artifact_paths, setup, timestamp_for_filename, trace_name, TracingGuards};

#[cfg(any(feature = "profile", feature = "events"))]
use tracing::Subscriber;
#[cfg(any(feature = "profile", feature = "events"))]
use tracing_subscriber::registry::{LookupSpan, SpanRef};

#[cfg(any(feature = "profile", feature = "events"))]
pub(crate) fn matches_token(n: &str, tok: &str) -> bool {
    n == tok
        || n.strip_prefix(tok)
            .is_some_and(|rest| rest.starts_with("::"))
}

#[cfg(any(feature = "profile", feature = "events"))]
pub(crate) fn is_in_focus<S>(span: &SpanRef<'_, S>, focus: &[String]) -> bool
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    focus.is_empty()
        || span
            .scope()
            .any(|ancestor| focus.iter().any(|tok| matches_token(ancestor.name(), tok)))
}
