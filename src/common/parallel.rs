//! Parallelism abstractions.
//!
//! When the `parallel` feature is enabled, these macros/helpers expand to
//! rayon parallel iterators. When it is disabled, they expand to the
//! corresponding serial std iterators, so the binary behaves exactly as
//! the single-threaded version.
//!
//! Usage:
//!   par_iter!(slice)                  -> rayon `par_iter` or std `iter`
//!   par_iter_mut!(slice)              -> rayon `par_iter_mut` or std `iter_mut`
//!   par_chunks!(slice, n)             -> rayon `par_chunks(n)` or std `chunks(n)`
//!   par_chunks_mut!(slice, n)         -> rayon `par_chunks_mut(n)` or std `chunks_mut(n)`
//!   par_range!(a..b)                  -> rayon `(a..b).into_par_iter()` or std `a..b`
//!
//! The resulting types differ between the two modes, but both support
//! `.enumerate()`, `.for_each(|x| ...)`, `.map(...)`, `.zip(...)` and friends,
//! so code consuming these macros compiles unchanged in both configurations.
//!
//! There is also a per-thread polynomial scratch pool used by the
//! parallel sumcheck path (`with_scratch_poly`). It replaces the
//! per-node `RefCell<Polynomial<_>>` scratches that would otherwise
//! race when two rayon workers enter a shared sub-tree simultaneously.

#[cfg(feature = "parallel")]
use std::cell::RefCell;

#[cfg(feature = "parallel")]
use crate::{
    common::ring_arithmetic::RingElement,
    protocol::sumcheck_utils::polynomial::Polynomial,
};

#[cfg(feature = "parallel")]
thread_local! {
    // Stack of reusable scratch polynomials, one stack per worker thread.
    // Children pop+reinit a buffer on entry and push it back on exit, so the
    // stack's depth grows with the recursion depth of the sumcheck tree
    // (4–5 levels in practice). No cross-thread sharing, so no locking.
    static SCRATCH_POLY_STACK: RefCell<Vec<Polynomial<RingElement>>> = RefCell::new(Vec::new());
}

/// Borrow a `Polynomial<RingElement>` from the per-thread pool, run `f` with
/// it, and return it to the pool afterwards. The buffer is not pre-zeroed;
/// callers should set it from scratch (e.g. via `univariate_polynomial_at_point_into`).
#[cfg(feature = "parallel")]
#[inline]
pub fn with_scratch_poly<F, R>(min_cap: usize, f: F) -> R
where
    F: FnOnce(&mut Polynomial<RingElement>) -> R,
{
    let mut p = SCRATCH_POLY_STACK.with(|stack| {
        stack
            .borrow_mut()
            .pop()
            .unwrap_or_else(|| Polynomial::new(min_cap))
    });
    let r = f(&mut p);
    SCRATCH_POLY_STACK.with(|stack| {
        stack.borrow_mut().push(p);
    });
    r
}

#[cfg(feature = "parallel")]
pub use rayon::prelude::*;

/// Choose a chunk size for parallel loops so that we get roughly
/// `rayon::current_num_threads() * oversubscription` chunks. In serial mode
/// we produce exactly one chunk, which recovers the original loop exactly.
///
/// `oversubscription` controls rayon's work-stealing granularity: having
/// more chunks than threads helps balance uneven work, but too many chunks
/// cost scheduling overhead. 4× is a reasonable default.
#[inline]
pub fn chunk_size_for_par(total: usize, oversubscription: usize) -> usize {
    #[cfg(feature = "parallel")]
    {
        let threads = rayon::current_num_threads().max(1);
        let desired_chunks = threads.saturating_mul(oversubscription).max(1);
        core::cmp::max(1, total.div_ceil(desired_chunks))
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = oversubscription;
        core::cmp::max(1, total)
    }
}

/// Expands to `iter.par_iter()` when parallel, `iter.iter()` otherwise.
#[macro_export]
macro_rules! par_iter {
    ($e:expr) => {{
        #[cfg(feature = "parallel")]
        {
            use $crate::common::parallel::IntoParallelRefIterator;
            ($e).par_iter()
        }
        #[cfg(not(feature = "parallel"))]
        {
            ($e).iter()
        }
    }};
}

/// Expands to `iter.par_iter_mut()` when parallel, `iter.iter_mut()` otherwise.
#[macro_export]
macro_rules! par_iter_mut {
    ($e:expr) => {{
        #[cfg(feature = "parallel")]
        {
            use $crate::common::parallel::IntoParallelRefMutIterator;
            ($e).par_iter_mut()
        }
        #[cfg(not(feature = "parallel"))]
        {
            ($e).iter_mut()
        }
    }};
}

/// Expands to `iter.par_chunks(n)` when parallel, `iter.chunks(n)` otherwise.
#[macro_export]
macro_rules! par_chunks {
    ($e:expr, $n:expr) => {{
        #[cfg(feature = "parallel")]
        {
            use $crate::common::parallel::ParallelSlice;
            ($e).par_chunks($n)
        }
        #[cfg(not(feature = "parallel"))]
        {
            ($e).chunks($n)
        }
    }};
}

/// Expands to `iter.par_chunks_mut(n)` when parallel, `iter.chunks_mut(n)` otherwise.
#[macro_export]
macro_rules! par_chunks_mut {
    ($e:expr, $n:expr) => {{
        #[cfg(feature = "parallel")]
        {
            use $crate::common::parallel::ParallelSliceMut;
            ($e).par_chunks_mut($n)
        }
        #[cfg(not(feature = "parallel"))]
        {
            ($e).chunks_mut($n)
        }
    }};
}

/// Expands to `(range).into_par_iter()` when parallel, or the plain range otherwise.
/// Useful for parallelizing a `for i in 0..n` loop.
#[macro_export]
macro_rules! par_range {
    ($range:expr) => {{
        #[cfg(feature = "parallel")]
        {
            use $crate::common::parallel::IntoParallelIterator;
            ($range).into_par_iter()
        }
        #[cfg(not(feature = "parallel"))]
        {
            ($range)
        }
    }};
}
