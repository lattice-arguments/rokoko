pub static DEGREE: usize = 128;
pub static HALF_DEGREE: usize = 64;
pub static MOD_Q: u64 = 1125899906839937;

pub static NOF_BATCHES: usize = 2;
pub const DEBUG: bool = true;

// const WITNESS_DIM: usize = 2usize.pow(14);
pub const WITNESS_DIM: usize = 2usize.pow(14); // most can fit on 64 GB
pub const WITNESS_WIDTH: usize = 2usize;
pub const RANK: usize = 8;

pub const VDF_MATRIX_HEIGHT: usize = 4;
pub const VDF_BITS: usize = 64;
pub const VDF_MATRIX_WIDTH: usize = VDF_BITS * VDF_MATRIX_HEIGHT;

/// Step stride for c-powers: consecutive steps are spaced VDF_MATRIX_HEIGHT apart.
/// Within each step, G uses c^{0..HEIGHT-1} and A uses c^{HEIGHT..2*HEIGHT-1}.
/// A-powers for step i overlap with G-powers for step i+1, giving telescoping.
pub const VDF_STRIDE: usize = VDF_MATRIX_HEIGHT;

pub const NUM_COLUMNS_INITIAL: usize = 2;

pub const PROJECTION_HEIGHT: usize = 256;

pub const MAX_UNSTRUCT_PROJ_RATIO: usize = 8; // to prevent degenerate projection configs with very large proj matrx

pub const LAST_ROUND_THRESHOLD: usize = 256;
