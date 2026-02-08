use crate::common::hash::HashWrapper;
use crate::common::matrix::HorizontallyAlignedMatrix;
use std::ops::Index;

static FALSE_FALSE: (bool, bool) = (false, false);
static FALSE_TRUE: (bool, bool) = (false, true);
static TRUE_FALSE: (bool, bool) = (true, false);
static TRUE_TRUE: (bool, bool) = (true, true);

pub struct ProjectionMatrix {
    pub projection_height: usize,
    pub projection_width: usize,
    pub projection_ratio: usize,

    pub width: usize,

    pub pos_masks: HorizontallyAlignedMatrix<u8>,
    pub non_zero_masks: HorizontallyAlignedMatrix<u8>,
}

impl ProjectionMatrix {
    pub fn new(projection_ratio: usize, projection_height: usize) -> Self {
        let projection_width = projection_height * projection_ratio;
        debug_assert!(
            projection_width % 8 == 0,
            "projection_width must be multiple of 8"
        );

        let width = projection_width / 8;

        let pos_masks = HorizontallyAlignedMatrix {
            data: vec![0u8; projection_height * width],
            width: width,
            height: projection_height,
        };
        let non_zero_masks = HorizontallyAlignedMatrix {
            data: vec![0u8; projection_height * width],
            width: width,
            height: projection_height,
        };

        Self {
            projection_height,
            projection_width,
            projection_ratio,
            width,
            pos_masks,
            non_zero_masks,
        }
    }

    #[inline(always)]
    pub fn get_row_masks_u8(&self, row: usize, col_base: usize) -> (u8, u8) {
        debug_assert!(row < self.projection_height);
        debug_assert!(col_base < self.projection_width);
        debug_assert!(col_base % 8 == 0, "col_base must be aligned to 8");

        let chunk = col_base >> 3;
        debug_assert!(chunk < self.width);

        (
            self.pos_masks[(row, chunk)],
            self.non_zero_masks[(row, chunk)],
        )
    }

    pub fn sample(&mut self, hash_wrapper: &mut HashWrapper) {
        hash_wrapper.fill_from_xof(b"projection-plan-sign", &mut self.pos_masks.data);
        hash_wrapper.fill_from_xof(b"projection-plan-value", &mut self.non_zero_masks.data);
    }

    #[cfg(test)]
    pub fn from_i8(data: Vec<Vec<i8>>) -> Self {
        let projection_height = data.len();
        let projection_width = data[0].len();
        let projection_ratio = projection_width / projection_height;

        debug_assert!(projection_width % 8 == 0);
        let width = projection_width / 8;

        let mut pm = ProjectionMatrix {
            projection_height,
            projection_width,
            projection_ratio,
            width,
            pos_masks: HorizontallyAlignedMatrix {
                data: vec![0u8; projection_height * width],
                width: width,
                height: projection_height,
            },
            non_zero_masks: HorizontallyAlignedMatrix {
                data: vec![0u8; projection_height * width],
                width: width,
                height: projection_height,
            },
        };

        for row in 0..projection_height {
            for col in 0..projection_width {
                let v = data[row][col];
                let (is_positive, is_non_zero) = match v {
                    0 => (false, false),
                    1 => (true, true),
                    -1 => (false, true),
                    _ => panic!("Invalid value in projection matrix"),
                };

                let chunk = col >> 3;
                let bit = (col & 7) as u8;

                if is_positive {
                    pm.pos_masks[(row, chunk)] |= 1u8 << bit;
                }
                if is_non_zero {
                    pm.non_zero_masks[(row, chunk)] |= 1u8 << bit;
                }
            }
        }

        pm
    }

    #[inline(always)]
    pub fn row_chunks(&self, row: usize) -> (&[u8], &[u8]) {
        (self.pos_masks.row(row), self.non_zero_masks.row(row))
    }
}

impl Index<(usize, usize)> for ProjectionMatrix {
    type Output = (bool, bool);

    #[inline(always)]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        debug_assert!(row < self.projection_height);
        debug_assert!(col < self.projection_width);

        let chunk = col >> 3;
        let bit = (col & 7) as u8;

        let k_pos = self.pos_masks[(row, chunk)];
        let k_inc = self.non_zero_masks[(row, chunk)];

        let is_positive = ((k_pos >> bit) & 1) == 1;
        let is_non_zero = ((k_inc >> bit) & 1) == 1;

        match (is_positive, is_non_zero) {
            (false, false) => &FALSE_FALSE,
            (false, true) => &FALSE_TRUE,
            (true, false) => &TRUE_FALSE,
            (true, true) => &TRUE_TRUE,
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use crate::common::ring_arithmetic::{Representation, RingElement};

//     use super::*;

//     #[test]
//     fn test_stability_of_sampling() {
//         let mut hash_wrapper = HashWrapper::new();
//         let mut projection_matrix_1 = ProjectionMatrix::new(4, PROJECTION_BASE_HEIGHT * 4);
//         println!(
//             "Matrix 1: height={}, width={}, data.height={}, data.width={}",
//             projection_matrix_1.projection_height,
//             projection_matrix_1.projection_width,
//             projection_matrix_1.projection_data.height,
//             projection_matrix_1.projection_data.width
//         );
//         projection_matrix_1.sample(&mut hash_wrapper);

//         let mut hash_wrapper_2 = HashWrapper::new();
//         let mut projection_matrix_2 = ProjectionMatrix::new(4, PROJECTION_BASE_HEIGHT * 4);
//         projection_matrix_2.sample(&mut hash_wrapper_2);

//         for outer_col in 0..PROJECTION_BASE_HEIGHT * 16 {
//             for row in 0..PROJECTION_BASE_HEIGHT * 4 {
//                 debug_assert_eq!(
//                     projection_matrix_1[(row, outer_col)],
//                     projection_matrix_2[(row, outer_col)]
//                 );
//             }
//         }
//     }

//     #[test]
//     fn test_instability_with_different_transcript() {
//         let mut hash_wrapper = HashWrapper::new();
//         let mut projection_matrix_1 = ProjectionMatrix::new(4, PROJECTION_BASE_HEIGHT * 4);
//         projection_matrix_1.sample(&mut hash_wrapper);

//         let mut hash_wrapper_2 = HashWrapper::new();
//         hash_wrapper_2
//             .update_with_ring_element(&RingElement::constant(42, Representation::IncompleteNTT));
//         let mut projection_matrix_2 = ProjectionMatrix::new(4, PROJECTION_BASE_HEIGHT * 4);
//         projection_matrix_2.sample(&mut hash_wrapper_2);

//         let mut differences_found = 0;
//         for col in 0..PROJECTION_BASE_HEIGHT * 4 * 4 {
//             for row in 0..PROJECTION_BASE_HEIGHT {
//                 if projection_matrix_1[(row, col)] != projection_matrix_2[(row, col)] {
//                     differences_found += 1;
//                 }
//             }
//         }
//         debug_assert!(differences_found > 0);
//     }

//     #[test]
//     fn test_indexing() {
//         let mut data = vec![vec![0i8; PROJECTION_BASE_HEIGHT * 4]; PROJECTION_BASE_HEIGHT];
//         data[0][0] = 1;
//         data[3][1] = -1;
//         data[1][4] = 1;
//         data[2][3] = 0;
//         let projection_matrix = ProjectionMatrix::from_i8(data);
//         debug_assert_eq!(projection_matrix[(0, 0)], (true, true));
//         debug_assert_eq!(projection_matrix[(3, 1)], (false, true));
//         debug_assert_eq!(projection_matrix[(1, 4)], (true, true));
//         debug_assert_eq!(projection_matrix[(2, 3)], (false, false));
//     }
// }
