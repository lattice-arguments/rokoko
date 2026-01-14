use std::ops::Index;

use crate::common::{
    hash::HashWrapper,
    matrix::{VerticallyAlignedMatrix, ZeroNew},
};

static PROJECTION_BASE_HEIGHT: usize = 8;

// Static storage for return values to avoid returning references to temporaries
static FALSE_FALSE: (bool, bool) = (false, false);
static FALSE_TRUE: (bool, bool) = (false, true);
static TRUE_FALSE: (bool, bool) = (true, false);
static TRUE_TRUE: (bool, bool) = (true, true);

#[derive(Clone)]
pub struct ProjectionSquare {
    // Each byte encodes 4 entries (2 bits per entry)
    data: [u8; PROJECTION_BASE_HEIGHT * PROJECTION_BASE_HEIGHT / 4],
}
pub struct ProjectionMatrix {
    pub projection_height: usize,
    pub projection_width: usize,
    pub projection_ratio: usize,
    pub projection_data: VerticallyAlignedMatrix<ProjectionSquare>,
}

impl ProjectionMatrix {
    pub fn new(projection_ratio: usize, projection_height: usize) -> Self {
        ProjectionMatrix {
            projection_height,
            projection_width: projection_height * projection_ratio,
            projection_ratio,
            projection_data: VerticallyAlignedMatrix::new_zero(
                projection_height / PROJECTION_BASE_HEIGHT,
                projection_height * projection_ratio / PROJECTION_BASE_HEIGHT,
                &ProjectionSquare {
                    data: [0u8; PROJECTION_BASE_HEIGHT * PROJECTION_BASE_HEIGHT / 4],
                },
            ),
        }
    }

    #[cfg(test)]
    pub fn from_i8(data: Vec<Vec<i8>>) -> Self {
        let projection_height = data.len();
        let projection_width = data[0].len();
        let projection_ratio = projection_width / projection_height;
        let mut projection_data = VerticallyAlignedMatrix::new_zero(
            projection_height / PROJECTION_BASE_HEIGHT,
            projection_width / PROJECTION_BASE_HEIGHT,
            &ProjectionSquare {
                data: [0u8; PROJECTION_BASE_HEIGHT * PROJECTION_BASE_HEIGHT / 4],
            },
        );
        for outer_col in 0..projection_data.width {
            for row in 0..projection_data.height {
                let mut square = ProjectionSquare {
                    data: [0u8; PROJECTION_BASE_HEIGHT * PROJECTION_BASE_HEIGHT / 4],
                };
                for inner_col in 0..PROJECTION_BASE_HEIGHT {
                    for inner_row in 0..PROJECTION_BASE_HEIGHT {
                        let value = data[row * PROJECTION_BASE_HEIGHT + inner_row]
                            [outer_col * PROJECTION_BASE_HEIGHT + inner_col];
                        let bits = match value {
                            0 => 0b00,  // (false, false) = not positive, not non-zero
                            1 => 0b11,  // (true, true) = positive, non-zero
                            -1 => 0b01, // (false, true) = not positive, non-zero
                            _ => panic!("Invalid value in projection matrix"),
                        };
                        let byte_index = (inner_col / 4) * PROJECTION_BASE_HEIGHT + inner_row;
                        let bits_offset = (inner_col % 4) * 2;
                        square.data[byte_index] |= bits << bits_offset;
                    }
                }
                projection_data[(row, outer_col)] = square;
            }
        }
        ProjectionMatrix {
            projection_height,
            projection_width,
            projection_ratio,
            projection_data,
        }
    }

    pub fn sample(&mut self, hash_wrapper: &mut HashWrapper) {
        for square in self.projection_data.data.iter_mut() {
            hash_wrapper.fill_from_xof(b"projection-square", &mut square.data);
        }
    }
}

impl Index<(usize, usize)> for ProjectionSquare {
    type Output = (bool, bool);

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        let byte_index = (col / 4) * PROJECTION_BASE_HEIGHT + row;
        let bits_offset = (col % 4) * 2;
        let byte = self.data[byte_index];
        let bits = (byte >> bits_offset) & 0b11;
        match bits {
            0b00 => &FALSE_FALSE,
            0b01 => &FALSE_TRUE,
            0b10 => &TRUE_FALSE,
            0b11 => &TRUE_TRUE,
            _ => unreachable!(),
        }
    }
}

impl Index<(usize, usize)> for ProjectionMatrix {
    // { -1, 0, 1 } is represented as (is_positive, is_non_zero), which automatically imposes a desired bias towards 0
    type Output = (bool, bool);

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        let square_row = row / PROJECTION_BASE_HEIGHT;
        let square_col = col / PROJECTION_BASE_HEIGHT;
        let inner_row = row % PROJECTION_BASE_HEIGHT;
        let inner_col = col % PROJECTION_BASE_HEIGHT;
        &self.projection_data[(square_row, square_col)][(inner_row, inner_col)]
    }
}

#[cfg(test)]
mod tests {
    use crate::common::ring_arithmetic::{Representation, RingElement};

    use super::*;

    #[test]
    fn test_stability_of_sampling() {
        let mut hash_wrapper = HashWrapper::new();
        let mut projection_matrix_1 = ProjectionMatrix::new(4, 8);
        projection_matrix_1.sample(&mut hash_wrapper);

        let mut hash_wrapper_2 = HashWrapper::new();
        let mut projection_matrix_2 = ProjectionMatrix::new(4, 8);
        projection_matrix_2.sample(&mut hash_wrapper_2);

        for outer_col in 0..4 {
            for row in 0..PROJECTION_BASE_HEIGHT {
                for inner_col in 0..PROJECTION_BASE_HEIGHT {
                    assert_eq!(
                        projection_matrix_1[(row, outer_col * PROJECTION_BASE_HEIGHT + inner_col)],
                        projection_matrix_2[(row, outer_col * PROJECTION_BASE_HEIGHT + inner_col)]
                    );
                }
            }
        }
    }

    #[test]
    fn test_instability_with_different_transcript() {
        let mut hash_wrapper = HashWrapper::new();
        let mut projection_matrix_1 = ProjectionMatrix::new(4, 8);
        projection_matrix_1.sample(&mut hash_wrapper);

        let mut hash_wrapper_2 = HashWrapper::new();
        hash_wrapper_2
            .update_with_ring_element(&RingElement::constant(42, Representation::IncompleteNTT));
        let mut projection_matrix_2 = ProjectionMatrix::new(4, 8);
        projection_matrix_2.sample(&mut hash_wrapper_2);

        let mut differences_found = 0;
        for outer_col in 0..4 {
            for row in 0..PROJECTION_BASE_HEIGHT {
                for inner_col in 0..PROJECTION_BASE_HEIGHT {
                    if projection_matrix_1[(row, outer_col * PROJECTION_BASE_HEIGHT + inner_col)]
                        != projection_matrix_2
                            [(row, outer_col * PROJECTION_BASE_HEIGHT + inner_col)]
                    {
                        differences_found += 1;
                    }
                }
            }
        }
        assert!(differences_found > 0);
    }

    #[test]
    fn test_indexing() {
        let mut data = vec![vec![0i8; PROJECTION_BASE_HEIGHT * 4]; PROJECTION_BASE_HEIGHT];
        data[0][0] = 1;
        data[3][1] = -1;
        data[1][4] = 1;
        data[2][3] = 0;
        let projection_matrix = ProjectionMatrix::from_i8(data);
        assert_eq!(projection_matrix[(0, 0)], (true, true));
        assert_eq!(projection_matrix[(3, 1)], (false, true));
        assert_eq!(projection_matrix[(1, 4)], (true, true));
        assert_eq!(projection_matrix[(2, 3)], (false, false));
    }
}
