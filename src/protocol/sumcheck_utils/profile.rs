//! Per-gadget timing inside [`super::combiner::Combiner::univariate_polynomial_into`].
//!
//! Gated on feature `profile-sumcheck`. When disabled, [`timer`] returns a
//! zero-sized guard and [`print_and_reset`] is a no-op, so the instrumentation
//! compiles to nothing at call sites.
//!
//! # Usage
//!
//! Gadgets identify themselves via
//! [`super::common::HighOrderSumcheckData::gadget_kind`]. The Combiner wraps
//! each child call in a [`Guard`] that records its elapsed time on drop; the
//! sumcheck runner calls [`print_and_reset`] at the end of a round to emit the
//! breakdown and clear the accumulator.
//!
//! # Container vs. leaf
//!
//! Some gadgets (`Combiner`, `RingToFieldCombiner`) are themselves containers
//! that delegate to children; their recorded time overlaps their children's.
//! The printout separates containers from leaves so the overlap is visible
//! rather than double-counted in the leaf total.

/// Identifies the concrete gadget type that implements
/// [`super::common::HighOrderSumcheckData`]. Used as a hash key in the profile
/// accumulator — strongly-typed so adding a new gadget requires an explicit
/// variant rather than an ad-hoc string.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GadgetKind {
    Linear,
    Product,
    Diff,
    Sum,
    Selector,
    Combiner,
    RingToField,
    Unknown,
}

impl GadgetKind {
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            GadgetKind::Linear => "LinearSumcheck",
            GadgetKind::Product => "ProductSumcheck",
            GadgetKind::Diff => "DiffSumcheck",
            GadgetKind::Sum => "SumSumcheck",
            GadgetKind::Selector => "SelectorEq",
            GadgetKind::Combiner => "Combiner",
            GadgetKind::RingToField => "RingToFieldCombiner",
            GadgetKind::Unknown => "unknown",
        }
    }

    /// Container gadgets delegate to children whose time is attributed
    /// separately; their own recorded time overlaps the leaves' total.
    #[inline]
    pub fn is_container(self) -> bool {
        matches!(
            self,
            GadgetKind::Combiner | GadgetKind::RingToField | GadgetKind::Diff
        )
    }
}

#[cfg(feature = "profile-sumcheck")]
mod enabled {
    use super::GadgetKind;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    thread_local! {
        static PROFILE: RefCell<HashMap<GadgetKind, (Duration, usize)>> =
            RefCell::new(HashMap::new());
    }

    /// RAII timer that records its elapsed lifetime against `kind` when
    /// dropped. Zero-sized when the `profile-sumcheck` feature is disabled.
    #[must_use = "the timer records elapsed time when dropped; bind it to a name"]
    pub struct Guard {
        start: Instant,
        kind: GadgetKind,
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            let elapsed = self.start.elapsed();
            PROFILE.with(|p| {
                let mut p = p.borrow_mut();
                let entry = p.entry(self.kind).or_insert((Duration::ZERO, 0));
                entry.0 += elapsed;
                entry.1 += 1;
            });
        }
    }

    #[inline]
    pub fn timer(kind: GadgetKind) -> Guard {
        Guard {
            start: Instant::now(),
            kind,
        }
    }

    pub fn print_and_reset(label: &str) {
        let entries: Vec<(GadgetKind, Duration, usize)> = PROFILE.with(|p| {
            p.borrow_mut()
                .drain()
                .map(|(k, (d, c))| (k, d, c))
                .collect()
        });
        if entries.is_empty() {
            return;
        }

        let (mut leaves, mut containers): (Vec<_>, Vec<_>) =
            entries.into_iter().partition(|(k, _, _)| !k.is_container());
        leaves.sort_by(|a, b| b.1.cmp(&a.1));
        containers.sort_by(|a, b| b.1.cmp(&a.1));

        let leaf_summary: Vec<String> = leaves
            .iter()
            .map(|(kind, dur, count)| format!("{} {} ms [{}]", kind.name(), dur.as_millis(), count))
            .collect();
        println!("    [{}] gadget poly: {}", label, leaf_summary.join(", "));

        if !containers.is_empty() {
            let container_summary: Vec<String> = containers
                .iter()
                .map(|(kind, dur, count)| {
                    format!("{} {} ms [{}]", kind.name(), dur.as_millis(), count)
                })
                .collect();
            println!(
                "    [{}] containers (overlap leaves): {}",
                label,
                container_summary.join(", ")
            );
        }
    }
}

#[cfg(not(feature = "profile-sumcheck"))]
mod disabled {
    use super::GadgetKind;

    /// Zero-sized guard; the profile machinery compiles away when the
    /// `profile-sumcheck` feature is off.
    #[must_use]
    pub struct Guard;

    impl Drop for Guard {
        #[inline(always)]
        fn drop(&mut self) {}
    }

    #[inline(always)]
    pub fn timer(_kind: GadgetKind) -> Guard {
        Guard
    }

    #[inline(always)]
    pub fn print_and_reset(_label: &str) {}
}

#[cfg(feature = "profile-sumcheck")]
pub use enabled::{print_and_reset, timer, Guard};

#[cfg(not(feature = "profile-sumcheck"))]
pub use disabled::{print_and_reset, timer, Guard};
