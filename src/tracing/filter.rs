// `RUST_LOG` parsing, in place of `tracing-subscriber`'s `env-filter`.
//
// `RUST_LOG` here is just a level, case-insensitively:
//   trace  debug  info  warn  error  off
//
// `warning` and `err` are accepted as aliases. Note that other standard RUST_LOG notation — including the
// per-target syntax `env-filter` supports (`rokoko=debug`, `rokoko::protocol=trace`), and numeric
// values — is not recognised, and an unset or unrecognised `RUST_LOG` falls back to `info`.

use std::env;

use tracing::level_filters::LevelFilter;
use tracing::subscriber::Interest;
use tracing::{Level, Metadata};
use tracing_subscriber::layer::{Context, Filter};

const DEFAULT_LEVEL: LevelFilter = LevelFilter::INFO;

#[derive(Clone, Copy, Debug)]
pub struct RustLog {
    level: LevelFilter,
}

impl RustLog {
    pub fn from_default_env() -> Self {
        let level = env::var("RUST_LOG")
            .ok()
            .and_then(|spec| Self::parse_level(&spec))
            .unwrap_or(DEFAULT_LEVEL);
        Self { level }
    }

    fn parse_level(spec: &str) -> Option<LevelFilter> {
        match spec.trim().to_ascii_lowercase().as_str() {
            "trace" => Some(LevelFilter::TRACE),
            "debug" => Some(LevelFilter::DEBUG),
            "info" => Some(LevelFilter::INFO),
            "warn" | "warning" => Some(LevelFilter::WARN),
            "error" | "err" => Some(LevelFilter::ERROR),
            "off" => Some(LevelFilter::OFF),
            _ => None,
        }
    }

    #[cfg(any(feature = "events", test))]
    pub fn max_level(&self) -> LevelFilter {
        self.level
    }

    fn enables(&self, level: Level) -> bool {
        LevelFilter::from_level(level) <= self.level
    }

    fn is_enabled(&self, meta: &Metadata<'_>) -> bool {
        self.enables(*meta.level())
    }
}

impl<S> Filter<S> for RustLog {
    fn enabled(&self, meta: &Metadata<'_>, _cx: &Context<'_, S>) -> bool {
        self.is_enabled(meta)
    }

    fn callsite_enabled(&self, meta: &'static Metadata<'static>) -> Interest {
        if self.is_enabled(meta) {
            Interest::always()
        } else {
            Interest::never()
        }
    }

    fn max_level_hint(&self) -> Option<LevelFilter> {
        Some(self.level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(level: LevelFilter) -> RustLog {
        RustLog { level }
    }

    #[test]
    fn every_level_name_parses() {
        assert_eq!(RustLog::parse_level("trace"), Some(LevelFilter::TRACE));
        assert_eq!(RustLog::parse_level("debug"), Some(LevelFilter::DEBUG));
        assert_eq!(RustLog::parse_level("info"), Some(LevelFilter::INFO));
        assert_eq!(RustLog::parse_level("warn"), Some(LevelFilter::WARN));
        assert_eq!(RustLog::parse_level("error"), Some(LevelFilter::ERROR));
        assert_eq!(RustLog::parse_level("off"), Some(LevelFilter::OFF));
    }

    #[test]
    fn aliases_parse() {
        assert_eq!(RustLog::parse_level("warning"), Some(LevelFilter::WARN));
        assert_eq!(RustLog::parse_level("err"), Some(LevelFilter::ERROR));
    }

    #[test]
    fn parsing_is_case_insensitive() {
        assert_eq!(RustLog::parse_level("INFO"), Some(LevelFilter::INFO));
        assert_eq!(RustLog::parse_level("Debug"), Some(LevelFilter::DEBUG));
        assert_eq!(RustLog::parse_level("tRaCe"), Some(LevelFilter::TRACE));
    }

    #[test]
    fn surrounding_whitespace_is_ignored() {
        assert_eq!(RustLog::parse_level("  debug\n"), Some(LevelFilter::DEBUG));
    }

    #[test]
    fn unrecognised_specs_are_rejected() {
        assert_eq!(RustLog::parse_level(""), None);
        assert_eq!(RustLog::parse_level("nonsense"), None);
    }

    #[test]
    fn max_level_reports_the_configured_level() {
        assert_eq!(at(LevelFilter::DEBUG).max_level(), LevelFilter::DEBUG);
        assert_eq!(at(LevelFilter::OFF).max_level(), LevelFilter::OFF);
    }

    #[test]
    fn a_level_enables_itself_and_everything_more_severe() {
        let filter = at(LevelFilter::INFO);
        assert!(filter.enables(Level::ERROR));
        assert!(filter.enables(Level::WARN));
        assert!(filter.enables(Level::INFO));
        assert!(!filter.enables(Level::DEBUG));
        assert!(!filter.enables(Level::TRACE));
    }

    #[test]
    fn trace_enables_everything() {
        let filter = at(LevelFilter::TRACE);
        assert!(filter.enables(Level::ERROR));
        assert!(filter.enables(Level::TRACE));
    }

    #[test]
    fn off_enables_nothing() {
        let filter = at(LevelFilter::OFF);
        assert!(!filter.enables(Level::ERROR));
        assert!(!filter.enables(Level::TRACE));
    }
}
