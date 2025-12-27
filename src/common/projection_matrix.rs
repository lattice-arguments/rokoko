use std::ops::Index;

use crate::common::{config::PROJECTION_HEIGHT, hash::HashWrapper};

#[derive(Clone)]
pub struct ProjectionSquare {
    // Each byte encodes 4 entries (2 bits per entry)
    data: [u8; PROJECTION_HEIGHT * PROJECTION_HEIGHT / 4],
}
pub struct ProjectionMatrix {
    pub projection_ratio: usize,
    pub projection_data: Vec<ProjectionSquare>,
}

impl ProjectionMatrix {
    pub fn new(projection_ratio: usize) -> Self {
        ProjectionMatrix {
            projection_ratio,
            projection_data: vec![
                ProjectionSquare {
                    data: [0u8; PROJECTION_HEIGHT * PROJECTION_HEIGHT / 4]
                };
                projection_ratio
            ],
        }
    }

    pub fn sample(&mut self, hash_wrapper: &mut HashWrapper) {
        for square in self.projection_data.iter_mut() {
            hash_wrapper.fill_from_xof(b"projection-square", &mut square.data);
        }
    }
}

impl Index<(usize, usize)> for ProjectionSquare {
    type Output = (bool, bool);

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        assert!(
            row < PROJECTION_HEIGHT && col < PROJECTION_HEIGHT,
            "{} < {} && {} < {} failed",
            row,
            PROJECTION_HEIGHT,
            col,
            PROJECTION_HEIGHT
        );
        let byte_index = (col / 4) * PROJECTION_HEIGHT + row;
        let bits_offset = (col % 4) * 2;
        let byte = self.data[byte_index];
        let bits = (byte >> bits_offset) & 0b11;
        match bits {
            0b00 => &(false, false),
            0b01 => &(false, true),
            0b10 => &(true, false),
            0b11 => &(true, true),
            _ => unreachable!(),
        }
    }
}

impl Index<(usize, usize)> for ProjectionMatrix {
    // { -1, 0, 1 } is represented as (sign, value_present), which automatically imposes a desired bias towards 0
    type Output = (bool, bool);

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        let inner_col = col % PROJECTION_HEIGHT;
        let outer_col = col / PROJECTION_HEIGHT;
        &self.projection_data[outer_col][(row, inner_col)]
    }
}

#[cfg(test)]
mod tests {
    use crate::common::ring_arithmetic::{Representation, RingElement};

    use super::*;

    #[test]
    fn test_stability_of_sampling() {
        let mut hash_wrapper = HashWrapper::new();
        let mut projection_matrix_1 = ProjectionMatrix::new(4);
        projection_matrix_1.sample(&mut hash_wrapper);

        let mut hash_wrapper_2 = HashWrapper::new();
        let mut projection_matrix_2 = ProjectionMatrix::new(4);
        projection_matrix_2.sample(&mut hash_wrapper_2);

        for outer_col in 0..4 {
            for row in 0..PROJECTION_HEIGHT {
                for inner_col in 0..PROJECTION_HEIGHT {
                    assert_eq!(
                        projection_matrix_1[(row, outer_col * PROJECTION_HEIGHT + inner_col)],
                        projection_matrix_2[(row, outer_col * PROJECTION_HEIGHT + inner_col)]
                    );
                }
            }
        }
    }

    #[test]
    fn test_instability_with_different_transcript() {
        let mut hash_wrapper = HashWrapper::new();
        let mut projection_matrix_1 = ProjectionMatrix::new(4);
        projection_matrix_1.sample(&mut hash_wrapper);

        let mut hash_wrapper_2 = HashWrapper::new();
        hash_wrapper_2
            .update_with_ring_element(&RingElement::constant(42, Representation::IncompleteNTT));
        let mut projection_matrix_2 = ProjectionMatrix::new(4);
        projection_matrix_2.sample(&mut hash_wrapper_2);

        let mut differences_found = 0;
        for outer_col in 0..4 {
            for row in 0..PROJECTION_HEIGHT {
                for inner_col in 0..PROJECTION_HEIGHT {
                    if projection_matrix_1[(row, outer_col * PROJECTION_HEIGHT + inner_col)]
                        != projection_matrix_2[(row, outer_col * PROJECTION_HEIGHT + inner_col)]
                    {
                        differences_found += 1;
                    }
                }
            }
        }
        assert!(differences_found > 0);
    }
}
