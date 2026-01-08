use std::{
    collections::HashMap,
    ops::{Index, IndexMut},
    sync::{LazyLock, Mutex},
};

use crate::common::ring_arithmetic::{Representation, RingElement};

pub trait ZeroNew<T> {
    fn new_zero(height: usize, width: usize, zero: &T) -> Self;
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerticallyAlignedMatrix<T> {
    pub data: Vec<T>,
    pub width: usize,  // number of cols
    pub height: usize, // number of rows
}

static ZERO_REP_INCOMPLETE_NTT: LazyLock<RingElement> =
    LazyLock::new(|| RingElement::zero(Representation::IncompleteNTT));

// A pool of preallocated zero matrices for reuse.
// I envision the program running in 3 modes
// 1) Default mode: no preallocation, just allocate as needed
// 2) Warmup mode: no preallocation, but monitor the sizes of matrices being allocated and save the dimensions in a file
// 3) Preallocation mode: preallocate a number of zero matrices of given sizes at the start of the program (based on the data from warmup mode).
static PREALLOCATED_MATRICES: LazyLock<Mutex<HashMap<(usize, usize), Vec<Vec<RingElement>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

impl<T> Index<(usize, usize)> for VerticallyAlignedMatrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        assert!(
            row < self.height && col < self.width,
            "{} < {} && {} < {} failed",
            row,
            self.height,
            col,
            self.width
        );
        &self.data[col * self.height + row]
    }
}

impl<T> IndexMut<(usize, usize)> for VerticallyAlignedMatrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        assert!(
            row < self.height && col < self.width,
            "{} < {} && {} < {} failed",
            row,
            self.height,
            col,
            self.width
        );
        &mut self.data[col * self.height + row]
    }
}

impl<T> ZeroNew<T> for VerticallyAlignedMatrix<T>
where
    T: Clone,
{
    // TODO: maybe it's a bit weird that we cannot
    // implement ::zero for RingElement (many reps)?
    // How to fix that?
    fn new_zero(height: usize, width: usize, zero: &T) -> Self {
        VerticallyAlignedMatrix {
            data: vec![zero.clone(); height * width],
            width,
            height,
        }
    }
}

impl VerticallyAlignedMatrix<RingElement> {
    pub fn new_zero_preallocated(height: usize, width: usize) -> Self {
        let mut pool = PREALLOCATED_MATRICES.lock().expect("pool poisoned");
        let data = pool
            .get_mut(&(height, width))
            .and_then(|v| v.pop())
            .unwrap_or_else(|| {
                println!(
                    "Preallocated matrix pool miss for size {}x{}, allocating new",
                    height, width
                );
                vec![ZERO_REP_INCOMPLETE_NTT.clone(); height * width]
            });

        VerticallyAlignedMatrix {
            data,
            width,
            height,
        }
    }
}

pub fn preallocate_zero_matrices(height: usize, width: usize, count: usize) {
    let mut pool = PREALLOCATED_MATRICES.lock().expect("pool poisoned");
    let entry = pool.entry((height, width)).or_insert_with(Vec::new);
    entry.reserve(count);
    for _ in 0..count {
        entry.push(vec![ZERO_REP_INCOMPLETE_NTT.clone(); height * width]);
    }
}

impl HorizontallyAlignedMatrix<RingElement> {
    pub fn new_zero_preallocated(height: usize, width: usize) -> Self {
        let mut pool = PREALLOCATED_MATRICES.lock().expect("pool poisoned");
        let data = pool
            .get_mut(&(height, width))
            .and_then(|v| v.pop())
            .unwrap_or_else(|| {
                println!(
                    "Preallocated matrix pool miss for size {}x{}, allocating new",
                    height, width
                );
                vec![ZERO_REP_INCOMPLETE_NTT.clone(); height * width]
            });

        HorizontallyAlignedMatrix {
            data,
            width,
            height,
        }
    }
}

pub fn new_vec_zero_preallocated(count: usize) -> Vec<RingElement> {
    let mut pool = PREALLOCATED_MATRICES.lock().expect("pool poisoned");
    let data = pool
        .get_mut(&(1, count))
        .and_then(|v| v.pop())
        .unwrap_or_else(|| {
            println!(
                "Preallocated vector pool miss for size {}, allocating new",
                count
            );
            vec![ZERO_REP_INCOMPLETE_NTT.clone(); count]
        });

    data
}

impl<T> VerticallyAlignedMatrix<T> {
    pub fn col(&self, c: usize) -> &[T] {
        let start = c * self.height;
        let end = start + self.height;
        &self.data[start..end]
    }

    pub fn col_slice(&self, c: usize, start_row: usize, end_row: usize) -> &[T] {
        let col_start = c * self.height + start_row;
        let col_end = c * self.height + end_row;
        &self.data[col_start..col_end]
    }

    pub fn col_slice_mut(&mut self, c: usize, start_row: usize, end_row: usize) -> &mut [T] {
        let col_start = c * self.height + start_row;
        let col_end = c * self.height + end_row;
        &mut self.data[col_start..col_end]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HorizontallyAlignedMatrix<T> {
    pub data: Vec<T>,
    pub width: usize,  // number of cols
    pub height: usize, // number of rows
}

impl<T> Index<(usize, usize)> for HorizontallyAlignedMatrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        assert!(
            row < self.height && col < self.width,
            "{} < {} && {} < {} failed",
            row,
            self.height,
            col,
            self.width
        );
        &self.data[row * self.width + col]
    }
}

impl<T> IndexMut<(usize, usize)> for HorizontallyAlignedMatrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        assert!(
            row < self.height && col < self.width,
            "{} < {} && {} < {} failed",
            row,
            self.height,
            col,
            self.width
        );
        &mut self.data[row * self.width + col]
    }
}

impl<T> ZeroNew<T> for HorizontallyAlignedMatrix<T>
where
    T: Clone,
{
    fn new_zero(height: usize, width: usize, zero: &T) -> Self {
        HorizontallyAlignedMatrix {
            data: vec![zero.clone(); height * width],
            width,
            height,
        }
    }
}

impl<T> HorizontallyAlignedMatrix<T> {
    pub fn row(&self, r: usize) -> &[T] {
        let start = r * self.width;
        let end = start + self.width;
        &self.data[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexing() {
        let mut m = VerticallyAlignedMatrix {
            data: Vec::from([0, 3, 6, 1, 4, 7, 2, 5, 8]),
            width: 3,
            height: 3,
        };

        assert_eq!(m[(0, 0)], 0);
        assert_eq!(m[(0, 2)], 2);
        assert_eq!(m[(2, 0)], 6);
        assert_eq!(m[(2, 1)], 7);
        assert_eq!(m[(2, 2)], 8);

        m[(1, 1)] = 42;
        assert_eq!(m[(1, 1)], 42);
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds() {
        let m = VerticallyAlignedMatrix {
            data: vec![0; 4],
            width: 2,
            height: 2,
        };
        let _ = m[(2, 0)];
    }
}
