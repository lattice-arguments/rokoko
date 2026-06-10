use crate::common::hash::HashWrapper;
use crate::common::matrix::HorizontallyAlignedMatrix;
use std::ops::Index;

/// Pre-computed return values for the [`Index`] impl, avoiding repeated
/// construction of `(bool, bool)` tuples on every access.
static FALSE_FALSE: (bool, bool) = (false, false);
static FALSE_TRUE: (bool, bool) = (false, true);
static TRUE_FALSE: (bool, bool) = (true, false);
static TRUE_TRUE: (bool, bool) = (true, true);

/// A ternary projection matrix `J ∈ {-1, 0, +1}^{H × W}` stored as two
/// packed bitmask arrays for SIMD-friendly access.
///
/// # Encoding
///
/// Each entry `J[row, col]` is encoded across two parallel bitmask matrices:
///
/// | `pos_masks` bit | `non_zero_masks` bit | Entry value |
/// |:---------------:|:--------------------:|:-----------:|
/// |        0        |          0           |      0      |
/// |        0        |          1           |     −1      |
/// |        1        |          0           |   (unused)  |
/// |        1        |          1           |     +1      |
///
/// Both masks are stored row-major with **8 columns per byte** (LSB = lowest
/// column index).  The byte at position `row * width + (col / 8)` encodes
/// columns `col..col+8` for that row.
///
/// # Dimensions
///
/// - `projection_height` (`H`) — number of rows (= number of output
///   coefficients when projecting a witness).
/// - `projection_width` (`W = H × projection_ratio`) — number of columns
///   (= number of input witness coefficients consumed per projection block).
/// - `projection_ratio` — fan-in: how many input elements map to one output
///   element.
/// - `width` (`W / 8`) — number of bitmask bytes per row.
///
/// # SIMD usage
///
/// The AVX-512 projection kernels in [`project_fine`](crate::protocol::project_fine)
/// and [`helpers`](crate::protocol::sumchecks::helpers) load `pos_masks` and
/// `non_zero_masks` bytes directly as `__mmask8` values, using:
///
/// - `add_mask = non_zero & pos`       — lanes where `J = +1`
/// - `sub_mask = non_zero & !pos`      — lanes where `J = −1`
///
/// to drive masked `_mm512_mask_add_epi64` / `_mm512_mask_sub_epi64`.
pub struct ProjectionMatrix {
    /// Number of rows in J.
    pub projection_height: usize,
    /// Number of columns in J (= `projection_height × projection_ratio`).
    pub projection_width: usize,
    /// Fan-in ratio: `projection_width / projection_height`.
    pub projection_ratio: usize,

    /// Number of bitmask bytes per row (`projection_width / 8`).
    pub width: usize,

    /// Sign mask: bit set ⇒ entry is positive (+1).
    pub pos_masks: HorizontallyAlignedMatrix<u8>,
    /// Inclusion mask: bit set ⇒ entry is non-zero (±1).
    pub non_zero_masks: HorizontallyAlignedMatrix<u8>,
}

impl ProjectionMatrix {
    /// Creates a zero-initialised projection matrix with the given
    /// `projection_ratio` and `projection_height`.
    ///
    /// All entries start as 0 (both masks cleared).  Call [`sample`](Self::sample)
    /// to fill the matrix from the Fiat-Shamir transcript.
    ///
    /// # Panics
    ///
    /// Panics if `projection_height × projection_ratio` is not a multiple of 8.
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

    /// Returns the raw `(pos, non_zero)` bitmask bytes for 8 consecutive
    /// columns starting at `col_base` in the given `row`.
    ///
    /// This is the primary access path used by AVX-512 kernels: each returned
    /// `u8` can be cast directly to `__mmask8` for masked SIMD operations.
    ///
    /// # Panics
    ///
    /// Panics (debug) if `col_base` is not aligned to 8 or is out of range.
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

    /// Fills both bitmask arrays from the Fiat-Shamir transcript via XOF,
    /// using domain-separated labels `"projection-plan-sign"` (for `pos_masks`)
    /// and `"projection-plan-value"` (for `non_zero_masks`).
    ///
    /// After sampling, each bit is an independent uniform Bernoulli(½), so
    /// every entry is independently +1, −1, or 0 with probabilities ¼, ¼, ½.
    pub fn sample(&mut self, hash_wrapper: &mut HashWrapper) {
        hash_wrapper.fill_from_xof(b"projection-plan-sign", &mut self.pos_masks.data);
        hash_wrapper.fill_from_xof(b"projection-plan-value", &mut self.non_zero_masks.data);
    }

    /// Test helper: builds a [`ProjectionMatrix`] from a dense `i8` matrix
    /// where entries are in `{-1, 0, +1}`.
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

    /// Returns the full row of bitmask byte slices `(pos_row, non_zero_row)`,
    /// each of length [`width`](Self::width).
    ///
    /// Useful for scanning an entire row without per-chunk bounds checks.
    #[inline(always)]
    pub fn row_chunks(&self, row: usize) -> (&[u8], &[u8]) {
        (self.pos_masks.row(row), self.non_zero_masks.row(row))
    }
}

/// Element-wise access: `matrix[(row, col)]` returns `(is_positive, is_non_zero)`.
///
/// - `(true, true)`   → entry is +1
/// - `(false, true)`  → entry is −1
/// - `(_, false)`     → entry is 0
///
/// Returns a reference to a static tuple so the caller can pattern-match
/// without allocation.  For bulk SIMD access, prefer
/// [`get_row_masks_u8`](ProjectionMatrix::get_row_masks_u8) or direct
/// pointer arithmetic on `pos_masks` / `non_zero_masks`.
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

#[cfg(test)]
mod tests {
    use crate::common::ring_arithmetic::{Representation, RingElement};

    use super::*;
    const PROJECTION_BASE_HEIGHT: usize = 128;

    #[test]
    fn test_stability_of_sampling() {
        let mut hash_wrapper = HashWrapper::new();
        let mut projection_matrix_1 = ProjectionMatrix::new(4, PROJECTION_BASE_HEIGHT * 4);
        println!(
            "Matrix 1: height={}, width={}",
            projection_matrix_1.projection_height, projection_matrix_1.projection_width,
        );
        projection_matrix_1.sample(&mut hash_wrapper);

        let mut hash_wrapper_2 = HashWrapper::new();
        let mut projection_matrix_2 = ProjectionMatrix::new(4, PROJECTION_BASE_HEIGHT * 4);
        projection_matrix_2.sample(&mut hash_wrapper_2);

        for outer_col in 0..PROJECTION_BASE_HEIGHT * 16 {
            for row in 0..PROJECTION_BASE_HEIGHT * 4 {
                debug_assert_eq!(
                    projection_matrix_1[(row, outer_col)],
                    projection_matrix_2[(row, outer_col)]
                );
            }
        }
    }

    #[test]
    fn test_instability_with_different_transcript() {
        let mut hash_wrapper = HashWrapper::new();
        let mut projection_matrix_1 = ProjectionMatrix::new(4, PROJECTION_BASE_HEIGHT * 4);
        projection_matrix_1.sample(&mut hash_wrapper);

        let mut hash_wrapper_2 = HashWrapper::new();
        hash_wrapper_2
            .update_with_ring_element(&RingElement::constant(42, Representation::IncompleteNTT));
        let mut projection_matrix_2 = ProjectionMatrix::new(4, PROJECTION_BASE_HEIGHT * 4);
        projection_matrix_2.sample(&mut hash_wrapper_2);

        let mut differences_found = 0;
        for col in 0..PROJECTION_BASE_HEIGHT * 4 * 4 {
            for row in 0..PROJECTION_BASE_HEIGHT {
                if projection_matrix_1[(row, col)] != projection_matrix_2[(row, col)] {
                    differences_found += 1;
                }
            }
        }
        debug_assert!(differences_found > 0);
    }

    #[test]
    fn test_indexing() {
        let mut data = vec![vec![0i8; PROJECTION_BASE_HEIGHT * 4]; PROJECTION_BASE_HEIGHT];
        data[0][0] = 1;
        data[3][1] = -1;
        data[1][4] = 1;
        data[2][3] = 0;
        let projection_matrix = ProjectionMatrix::from_i8(data);
        debug_assert_eq!(projection_matrix[(0, 0)], (true, true));
        debug_assert_eq!(projection_matrix[(3, 1)], (false, true));
        debug_assert_eq!(projection_matrix[(1, 4)], (true, true));
        debug_assert_eq!(projection_matrix[(2, 3)], (false, false));
    }
}
