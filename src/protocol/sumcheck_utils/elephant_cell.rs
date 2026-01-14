use std::marker::Unsize;
use std::ops::CoerceUnsized;
/// ElephantCell: A cell type that can operate in safe or unsafe mode via compile-time feature flag.
///
/// - Safe mode (default): Uses Rc<RefCell<T>> with runtime borrow checking
/// - Unsafe mode (feature="unsafe-sumcheck"): Uses Rc<UnsafeCell<T>> with zero-cost access
///
/// Safety invariant for unsafe mode: During polynomial generation (read operations),
/// no mutations occur. Mutations only happen during partial_evaluate, which is never
/// concurrent with polynomial generation.
use std::rc::Rc;

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
    pub fn borrow(&self) -> Ref<T> {
        self.inner.borrow()
    }

    pub fn borrow_mut(&self) -> RefMut<T> {
        self.inner.borrow_mut()
    }

    #[inline(always)]
    pub fn get_ref(&self) -> Ref<T> {
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
