use std::ops::Index;
use crate::common::hash::HashWrapper;

static FALSE_FALSE: (bool, bool) = (false, false);
static FALSE_TRUE:  (bool, bool) = (false, true);
static TRUE_FALSE:  (bool, bool) = (true,  false);
static TRUE_TRUE:   (bool, bool) = (true,  true);

#[inline(always)]
pub fn plan_idx(inner_row: usize, chunk_idx: usize, chunks_per_row: usize) -> usize {
    inner_row * chunks_per_row + chunk_idx
}

pub struct ProjectionMatrix {
    pub projection_height: usize,
    pub projection_width: usize, // = projection_ratio * projection_height
    pub projection_ratio: usize,
    pub chunks_per_row: usize,

    // Row-major: idx = row*chunks_per_row + chunk
    pub k_pos_plan: Vec<u8>, // sign bits per 8 cols
    pub k_inc_plan: Vec<u8>, // nonzero bits per 8 cols
}

impl ProjectionMatrix {
    pub fn new(projection_ratio: usize, projection_height: usize) -> Self {
        let projection_width = projection_height * projection_ratio;
        debug_assert!(projection_width % 8 == 0, "projection_width must be multiple of 8");
        let chunks_per_row = projection_width / 8;

        let n = projection_height * chunks_per_row;

        let mut k_pos_plan: Vec<u8> = Vec::with_capacity(n);
        let mut k_inc_plan: Vec<u8> = Vec::with_capacity(n);
        unsafe {
            k_pos_plan.set_len(n);
            k_inc_plan.set_len(n);
        }

        Self {
            projection_height,
            projection_width,
            projection_ratio,
            chunks_per_row,
            k_pos_plan,
            k_inc_plan,
        }
    }

    /// Return masks for 8 consecutive columns at a given row.
    /// `col_base` must be 8-aligned.
    #[inline(always)]
    pub fn get_row_masks_u8(&self, row: usize, col_base: usize) -> (u8, u8) {
        debug_assert!(row < self.projection_height);
        debug_assert!(col_base < self.projection_width);
        debug_assert!(col_base % 8 == 0, "col_base must be aligned to 8");

        let chunk = col_base >> 3;
        let idx = plan_idx(row, chunk, self.chunks_per_row);

        unsafe {
            (
                *self.k_pos_plan.get_unchecked(idx),
                *self.k_inc_plan.get_unchecked(idx),
            )
        }
    }

    /// Sampling fills the two plans directly using independent XOF streams.
    pub fn sample(&mut self, hash_wrapper: &mut HashWrapper) {
        // Fill directly into the backing storage (no extra loop, best locality)
        hash_wrapper.fill_from_xof(b"projection-plan-sign", &mut self.k_pos_plan);
        hash_wrapper.fill_from_xof(b"projection-plan-value", &mut self.k_inc_plan);
    }

    #[cfg(test)]
    pub fn from_i8(data: Vec<Vec<i8>>) -> Self {
        let projection_height = data.len();
        let projection_width = data[0].len();
        let projection_ratio = projection_width / projection_height;

        debug_assert!(projection_width % 8 == 0);
        let chunks_per_row = projection_width / 8;

        let n = projection_height * chunks_per_row;

        let mut pm = ProjectionMatrix {
            projection_height,
            projection_width,
            projection_ratio,
            chunks_per_row,
            k_pos_plan: vec![0u8; n],
            k_inc_plan: vec![0u8; n],
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
                let idx = plan_idx(row, chunk, chunks_per_row);

                if is_positive {
                    pm.k_pos_plan[idx] |= 1u8 << bit;
                }
                if is_non_zero {
                    pm.k_inc_plan[idx] |= 1u8 << bit;
                }
            }
        }

        pm
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
        let idx = plan_idx(row, chunk, self.chunks_per_row);

        let k_pos = unsafe { *self.k_pos_plan.get_unchecked(idx) };
        let k_inc = unsafe { *self.k_inc_plan.get_unchecked(idx) };

        let is_positive = ((k_pos >> bit) & 1) == 1;
        let is_non_zero = ((k_inc >> bit) & 1) == 1;

        match (is_positive, is_non_zero) {
            (false, false) => &FALSE_FALSE,
            (false, true)  => &FALSE_TRUE,
            (true,  false) => &TRUE_FALSE,
            (true,  true)  => &TRUE_TRUE,
        }
    }
}
