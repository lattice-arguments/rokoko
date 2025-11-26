use std::ops::{Index, IndexMut};

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<T> {
    pub data: Vec<T>,
    pub width: usize,
    pub height: usize,
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (r, c) = index;
        &self.data[r * self.width + c]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (r, c) = index;
        assert!(r < self.width && c < self.height);
        &mut self.data[r * self.width + c]
    }
}

pub struct Row<T> {
    pub ptr: *const T,
    pub len: usize,
}

impl<T> Matrix<T> {
    pub fn empty() -> Self {
        Matrix {
            data: Vec::new(),
            width: 0,
            height: 0,
        }
    }
    pub fn new(width: usize, height: usize) -> Self {
        let mut data: Vec<T> = Vec::with_capacity(width * height);
        unsafe {
            data.set_len(width * height);
        }
        Matrix {
            data,
            width,
            height,
        }
    }

    pub fn push_row(&mut self, row: &mut Vec<T>) {
        self.height += 1;
        self.data.append(row);
    }

    pub fn get_index(&self, x: usize, y: usize) -> usize {
        return y * self.width + x;
    }

    pub fn get(&self, r: usize, c: usize) -> Option<&T> {
        let index = self.get_index(r, c);
        return self.data.get(index);
    }

    pub fn get_mut(&mut self, r: usize, c: usize) -> Option<&mut T> {
        let index = self.get_index(r, c);
        return self.data.get_mut(index);
    }
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
        let mut m = Matrix {
            data: (0..9).collect(),
            width: 3,
            height: 3,
        };

        assert_eq!(m[(0, 0)], 0);
        assert_eq!(m[(0, 2)], 2);
        assert_eq!(m[(2, 0)], 6);
        assert_eq!(m[(2, 2)], 8);

        m[(1, 1)] = 42;
        assert_eq!(m[(1, 1)], 42);
    }

    #[test]
    fn test_get_methods() {
        let mut m = Matrix {
            data: (0..6).collect(),
            width: 3,
            height: 2,
        };

        assert_eq!(m.get(0, 0), Some(&0));
        assert_eq!(m.get(2, 1), Some(&5));
        assert_eq!(m.get(0, 2), None);
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
        let m = Matrix {
            data: vec![0; 4],
            width: 2,
            height: 2,
        };
        let _ = m[(2, 0)];
    }
}
