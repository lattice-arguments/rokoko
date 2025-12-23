use std::ops::{Index, IndexMut};

use num::Zero;

#[derive(Clone, Debug, PartialEq)]
pub struct VerticallyAlignedMatrix<T> {
    pub data: Vec<T>,
    pub width: usize,  // number of cols
    pub height: usize, // number of rows
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

pub struct Row<T> {
    pub ptr: *const T,
    pub len: usize,
}

impl<T> VerticallyAlignedMatrix<T>
where
    T: Clone,
{
    pub fn empty() -> Self {
        VerticallyAlignedMatrix {
            data: Vec::new(),
            width: 0,
            height: 0,
        }
    }

    // TODO: maybe it's a bit weird that we cannot
    // implement ::zero for RingElement (many reps)?
    // How to fix that?
    pub fn new(height: usize, width: usize, zero: &T) -> Self {
        let mut data: Vec<T> = vec![zero.clone(); height * width];

        VerticallyAlignedMatrix {
            data,
            width,
            height,
        }
    }

    pub fn push_col(&mut self, col: &mut Vec<T>) {
        self.width += 1;
        self.data.append(col);
    }

    pub fn get_index(&self, row_idx: usize, col_idx: usize) -> usize {
        return col_idx * self.height + row_idx;
    }

    pub fn get(&self, row_idx: usize, col_idx: usize) -> Option<&T> {
        if row_idx > self.height || col_idx > self.width {
            None
        } else {
            let index = self.get_index(row_idx, col_idx);
            self.data.get(index)
        }
    }

    pub fn get_mut(&mut self, row_idx: usize, col_idx: usize) -> Option<&mut T> {
        if row_idx > self.height || col_idx > self.width {
            None
        } else {
            let index = self.get_index(row_idx, col_idx);
            self.data.get_mut(index)
        }
    }

    pub fn col(&self, c: usize) -> &[T] {
        let start = c * self.height;
        let end = start + self.height;
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
    fn test_get_methods() {
        let mut m = VerticallyAlignedMatrix {
            data: Vec::from([0, 3, 1, 4, 2, 5]),
            width: 3,
            height: 2,
        };

        assert_eq!(m.get(0, 0), Some(&0));
        assert_eq!(m.get(1, 1), Some(&4));
        assert_eq!(m.get(0, 2), Some(&2));
        assert_eq!(m.get(3, 1), None);

        // Test get_mut
        if let Some(val) = m.get_mut(0, 1) {
            *val = 99;
        }
        assert_eq!(m.get(0, 1), Some(&99));
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
