use std::ops::{Index, IndexMut};

use crate::common::{
    pool::{
        get_preallocated_quad_vec, get_preallocated_ring_element_vec, preallocate_ring_element_vecs,
    },
    ring_arithmetic::{QuadraticExtension, RingElement},
};

pub trait ZeroNew<T> {
    fn new_zero(height: usize, width: usize, zero: &T) -> Self;
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerticallyAlignedMatrix<T> {
    pub data: Vec<T>,
    pub width: usize,     // number of cols
    pub height: usize,    // number of rows
    pub used_cols: usize, // number of used cols. This is used when witness width is expanded to be power of two but only part of it is used. Optimisation trick. Otherwise same as width.
}

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
            used_cols: width,
        }
    }
}

impl VerticallyAlignedMatrix<RingElement> {
    pub fn new_zero_preallocated(height: usize, width: usize) -> Self {
        let data = get_preallocated_ring_element_vec(height * width);
        VerticallyAlignedMatrix {
            data,
            width,
            height,
            used_cols: width,
        }
    }
}

pub fn preallocate_zero_matrices(height: usize, width: usize, count: usize) {
    preallocate_ring_element_vecs(height * width, count);
}

impl HorizontallyAlignedMatrix<RingElement> {
    pub fn new_zero_preallocated(height: usize, width: usize) -> Self {
        let data = get_preallocated_ring_element_vec(height * width);
        HorizontallyAlignedMatrix {
            data,
            width,
            height,
        }
    }
}

#[inline]
pub fn new_vec_zero_preallocated(count: usize) -> Vec<RingElement> {
    get_preallocated_ring_element_vec(count)
}

#[inline]
pub fn new_vec_zero_field_preallocated(count: usize) -> Vec<QuadraticExtension> {
    get_preallocated_quad_vec(count)
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

    pub fn row_slice_mut(&mut self, r: usize) -> &mut [T] {
        let start = r * self.width;
        let end = start + self.width;
        &mut self.data[start..end]
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
            used_cols: 3,
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
            used_cols: 2,
        };
        let _ = m[(2, 0)];
    }
}
