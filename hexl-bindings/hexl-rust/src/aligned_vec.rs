use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

pub struct AlignedVecU64 {
    ptr: NonNull<u64>,
    len: usize,
}

unsafe impl Send for AlignedVecU64 {}
unsafe impl Sync for AlignedVecU64 {}

impl AlignedVecU64 {
    pub fn new(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
            };
        }
        let layout =
            Layout::from_size_align(len * std::mem::size_of::<u64>(), 64).expect("layout");
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw as *mut u64).expect("alloc failed");
        Self { ptr, len }
    }

    pub fn from_slice(values: &[u64]) -> Self {
        let out = Self::new(values.len());
        if out.len != 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(values.as_ptr(), out.ptr.as_ptr(), values.len());
            }
        }
        out
    }

    pub fn from_vec(values: Vec<u64>) -> Self {
        Self::from_slice(&values)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_ptr(&self) -> *const u64 {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u64 {
        self.ptr.as_ptr()
    }

    pub fn as_slice(&self) -> &[u64] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u64] {
        if self.len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Default for AlignedVecU64 {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Clone for AlignedVecU64 {
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice())
    }
}

impl Deref for AlignedVecU64 {
    type Target = [u64];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for AlignedVecU64 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl Drop for AlignedVecU64 {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        let layout =
            Layout::from_size_align(self.len * std::mem::size_of::<u64>(), 64).expect("layout");
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}
