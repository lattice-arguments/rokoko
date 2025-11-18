use std::ops::{Index, IndexMut};

pub struct Matrix<T> {
    pub data: Vec<T>,
    pub width: usize,
    pub height: usize
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (r, c) = index;
        assert!(r < self.width && c < self.height);
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
    ptr: *const T,
    len: usize,
}

impl<T> Matrix<T> {

    pub fn get_index(&self, x: usize, y: usize) -> usize {
        return y*self.width+x; 
    }

    pub fn get(&self, r: usize, c: usize) -> Option<&T> {
       let index = self.get_index(r, c); 
       return self.data.get(index);
    }

    pub fn get_mut(&mut self, r: usize, c: usize) -> Option<&mut T> {
        let index = self.get_index(r, c); 
        return self.data.get_mut(index);        
    }
    pub fn row(&self, r: usize) -> Row<T> {
        Row {
            ptr: &self.data[r * self.width] as *const T,
            len: self.width,
        }
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
    fn test_row() {
        let m = Matrix {
            data: (0..12).collect(),
            width: 4,
            height: 3,
        };

        let row0 = m.row(0);
        assert_eq!(row0.len, 4);
        unsafe {
            for i in 0..row0.len {
                assert_eq!(*row0.ptr.add(i), i);
            }
        }

        let row2 = m.row(2);
        unsafe {
            for i in 0..row2.len {
                assert_eq!(*row2.ptr.add(i), 8 + i);
            }
        }
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
