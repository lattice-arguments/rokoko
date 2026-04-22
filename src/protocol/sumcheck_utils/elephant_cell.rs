use std::marker::Unsize;
use std::ops::CoerceUnsized;
/// ElephantCell: A cell type that can operate in safe or unsafe mode via compile-time feature flag.
///
/// - Safe mode (default): Uses Rc<RefCell<T>> with runtime borrow checking
/// - Unsafe mode (feature="unsafe-sumcheck"): Uses Rc<UnsafeCell<T>> with zero-cost access
/// - Unsafe + parallel: Uses Arc<UnsafeCell<T>> + unsafe Send/Sync so the sumcheck
///   tree can cross rayon thread boundaries. Safety invariant: within one
///   sumcheck sweep, any given ElephantCell is touched from at most one
///   thread at a time. Parallelism happens only across disjoint sub-trees
///   (e.g. lhs vs rhs of a DiffSumcheck).
///
/// Safety invariant for unsafe mode: During polynomial generation (read operations),
/// no mutations occur. Mutations only happen during partial_evaluate, which is never
/// concurrent with polynomial generation.
#[cfg(not(feature = "parallel"))]
use std::rc::Rc;

#[cfg(feature = "parallel")]
use std::sync::Arc as Rc;

#[cfg(not(feature = "unsafe-sumcheck"))]
use std::cell::{Ref, RefCell, RefMut};

#[cfg(feature = "unsafe-sumcheck")]
use std::cell::UnsafeCell;

// Safe mode: Rc<RefCell<T>>
#[cfg(not(feature = "unsafe-sumcheck"))]
pub struct ElephantCell<T: ?Sized> {
    inner: Rc<RefCell<T>>,
}

#[cfg(not(feature = "unsafe-sumcheck"))]
impl<T: ?Sized> ElephantCell<T> {
    pub fn borrow(&self) -> Ref<'_, T> {
        self.inner.borrow()
    }

    pub fn borrow_mut(&self) -> RefMut<'_, T> {
        self.inner.borrow_mut()
    }

    #[inline(always)]
    pub fn get_ref(&self) -> Ref<'_, T> {
        self.inner.borrow()
    }
}

#[cfg(not(feature = "unsafe-sumcheck"))]
impl<T> ElephantCell<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: Rc::new(RefCell::new(value)),
        }
    }
}

#[cfg(not(feature = "unsafe-sumcheck"))]
impl<T: ?Sized> Clone for ElephantCell<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
        }
    }
}

#[cfg(not(feature = "unsafe-sumcheck"))]
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<ElephantCell<U>> for ElephantCell<T> {}

// Unsafe mode: Rc<UnsafeCell<T>>
#[cfg(feature = "unsafe-sumcheck")]
pub struct ElephantCell<T: ?Sized> {
    inner: Rc<UnsafeCell<T>>,
}

#[cfg(feature = "unsafe-sumcheck")]
impl<T: ?Sized> ElephantCell<T> {
    #[inline(always)]
    pub fn borrow_mut(&self) -> &mut T {
        unsafe { &mut *self.inner.get() }
    }

    #[inline(always)]
    pub fn get_ref(&self) -> &T {
        unsafe { &*self.inner.get() }
    }

    #[inline(always)]
    pub fn borrow(&self) -> &T {
        self.get_ref()
    }
}

#[cfg(feature = "unsafe-sumcheck")]
impl<T> ElephantCell<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: Rc::new(UnsafeCell::new(value)),
        }
    }
}

#[cfg(feature = "unsafe-sumcheck")]
impl<T: ?Sized> Clone for ElephantCell<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
        }
    }
}

#[cfg(feature = "unsafe-sumcheck")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<ElephantCell<U>> for ElephantCell<T> {}

// Under `parallel + unsafe-sumcheck`, the sumcheck tree must cross thread
// boundaries for rayon parallelism. UnsafeCell is not Sync by default; we
// uphold the same "no concurrent mutation" invariant as the single-threaded
// path, just scoped to "at most one thread mutates a given cell at a time"
// instead of "at most one caller mutates at a time". Leaves that would race
// on shared scratch are either read-only or are duplicated per sub-tree by
// the builder.
#[cfg(all(feature = "unsafe-sumcheck", feature = "parallel"))]
unsafe impl<T: ?Sized + Send> Send for ElephantCell<T> {}
// We only require `T: Send` (not `Sync`): the inner `UnsafeCell<T>` guarantees
// no aliasing when the callers uphold the invariant that no two threads
// mutate the same cell concurrently. The sumcheck tree's own scratch cells
// are per-node, and parallel call sites are structured so distinct threads
// only traverse disjoint sub-trees, so the invariant holds.
#[cfg(all(feature = "unsafe-sumcheck", feature = "parallel"))]
unsafe impl<T: ?Sized + Send> Sync for ElephantCell<T> {}
