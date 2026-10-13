use crate::common::hash::HashWrapper;
use crate::common::matrix::HorizontallyAlignedMatrix;
use std::ops::Index;
use std::sync::OnceLock;

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
/// Sampled masks are AES-256-CTR keystreams, block `i` of mask `m` (0 `pos`, 1 `non_zero`)
/// encrypting `i ‖ m` as little-endian `u64`s, materialised on first use.
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

    key: Option<[u8; 32]>,
    masks: OnceLock<[HorizontallyAlignedMatrix<u8>; 2]>,
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

        Self {
            projection_height,
            projection_width,
            projection_ratio,
            width: projection_width / 8,
            key: None,
            masks: OnceLock::new(),
        }
    }

    pub fn pos_masks(&self) -> &HorizontallyAlignedMatrix<u8> {
        &self.masks()[0]
    }

    pub fn non_zero_masks(&self) -> &HorizontallyAlignedMatrix<u8> {
        &self.masks()[1]
    }

    fn masks(&self) -> &[HorizontallyAlignedMatrix<u8>; 2] {
        self.masks.get_or_init(|| {
            std::array::from_fn(|mask| {
                let mut data = vec![0u8; self.projection_height * self.width];
                if let Some(key) = self.key {
                    fill_keystream(key, mask as u64, 0, &mut data);
                }
                HorizontallyAlignedMatrix {
                    data,
                    width: self.width,
                    height: self.projection_height,
                }
            })
        })
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
            self.pos_masks()[(row, chunk)],
            self.non_zero_masks()[(row, chunk)],
        )
    }

    /// Draws the AES-256 key of both masks from the transcript (label `"projection-plan-key"`):
    /// every entry is independently +1, −1 or 0 with probabilities ¼, ¼, ½.
    pub fn sample(&mut self, hash_wrapper: &mut HashWrapper) {
        let mut key = [0u8; 32];
        hash_wrapper.fill_from_xof(b"projection-plan-key", &mut key);
        self.key = Some(key);
        self.masks = OnceLock::new();
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

        let mut masks = std::array::from_fn(|_| HorizontallyAlignedMatrix {
            data: vec![0u8; projection_height * width],
            width,
            height: projection_height,
        });

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
                    masks[0][(row, chunk)] |= 1u8 << bit;
                }
                if is_non_zero {
                    masks[1][(row, chunk)] |= 1u8 << bit;
                }
            }
        }

        ProjectionMatrix {
            projection_height,
            projection_width,
            projection_ratio,
            width,
            key: None,
            masks: OnceLock::from(masks),
        }
    }

    /// NEON `(+1, −1)` lane masks of columns `8 chunk .. 8 chunk + 8` in `row`, two per vector.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub(crate) unsafe fn neon_masks(
        &self,
        row: usize,
        chunk: usize,
    ) -> (
        [std::arch::aarch64::uint64x2_t; 4],
        [std::arch::aarch64::uint64x2_t; 4],
    ) {
        use std::arch::aarch64::*;
        static LANE_BITS: [[u64; 2]; 4] = [[1, 2], [4, 8], [16, 32], [64, 128]];
        let at = row * self.width + chunk;
        let pos = *self.pos_masks().data.get_unchecked(at);
        let nz = *self.non_zero_masks().data.get_unchecked(at);
        let add = vdupq_n_u64((nz & pos) as u64);
        let sub = vdupq_n_u64((nz & !pos) as u64);
        let bits = |j: usize| vld1q_u64(LANE_BITS[j].as_ptr());
        (
            std::array::from_fn(|j| vtstq_u64(add, bits(j))),
            std::array::from_fn(|j| vtstq_u64(sub, bits(j))),
        )
    }

    /// Returns the full row of bitmask byte slices `(pos_row, non_zero_row)`,
    /// each of length [`width`](Self::width).
    ///
    /// Useful for scanning an entire row without per-chunk bounds checks.
    #[inline(always)]
    pub fn row_chunks(&self, row: usize) -> (&[u8], &[u8]) {
        (self.pos_masks().row(row), self.non_zero_masks().row(row))
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512bw",
    target_feature = "gfni",
    target_feature = "aes",
    target_feature = "vaes"
))]
impl ProjectionMatrix {
    /// `acc(w, chunk) += Σ_row weights[w][row] · J[row, 8 chunk + l]` in lane `l`, or false unless
    /// `J` is sampled, `16 | height, width` and `weights`, `capacity` cover `J`. Four-Russians over
    /// rows: 4-bit (+1)- and (−1)-column patterns index the 16 subset sums of four weights.
    ///
    /// # Safety
    /// `acc(w, chunk)` for `w < W`, `chunk < capacity` are distinct, aligned, writable vectors.
    pub(crate) unsafe fn accumulate_signed_sums<const W: usize>(
        &self,
        weights: [&[u64]; W],
        capacity: usize,
        acc: impl Fn(usize, usize) -> *mut std::arch::x86_64::__m512i,
    ) -> bool {
        use std::arch::x86_64::*;

        let Some(key) = self.key else {
            return false;
        };
        let width = self.width;
        if self.projection_height % 16 != 0
            || width % 16 != 0
            || self.projection_width != 8 * width
            || width > capacity
            || weights.iter().any(|w| w.len() < self.projection_height)
        {
            return false;
        }
        let round_keys = expand_key(key).map(|k| _mm512_broadcast_i32x4(k));

        let subset_sums: [Vec<[__m512i; 2]>; W] = std::array::from_fn(|w| {
            weights[w]
                .chunks_exact(4)
                .map(|c| {
                    let zero = _mm512_setzero_si512();
                    let c: [__m512i; 4] = std::array::from_fn(|t| _mm512_set1_epi64(c[t] as i64));
                    let lo = _mm512_mask_add_epi64(zero, 0b1010_1010, zero, c[0]);
                    let lo = _mm512_mask_add_epi64(lo, 0b1100_1100, lo, c[1]);
                    let lo = _mm512_mask_add_epi64(lo, 0b1111_0000, lo, c[2]);
                    [lo, _mm512_add_epi64(lo, c[3])]
                })
                .collect()
        });
        let low_nibbles = _mm512_setr_epi64(0, 8, 16, 24, 32, 40, 48, 56);
        let high_nibbles = _mm512_setr_epi64(4, 12, 20, 28, 36, 44, 52, 60);

        let mut tiles = [[_mm512_setzero_si512(); 8]; 4];
        let mut accumulators = [[std::ptr::null_mut::<__m512i>(); W]; 64];
        for start in (0..width).step_by(64) {
            let chunks = (width - start).min(64);
            let mut slots = 0;
            for at in 0..64 {
                if TILE_CHUNK[at] < chunks {
                    accumulators[slots] = std::array::from_fn(|w| acc(w, start + TILE_CHUNK[at]));
                    slots += 1;
                }
            }

            for group in 0..self.projection_height / 16 {
                for half in 0..2 {
                    let rows = keystream_rows(&round_keys, |r| {
                        ((16 * group + 8 * half + r) * width + start) / 16
                    });
                    let (pos, non_zero) = (&rows[..8], &rows[8..]);
                    tiles[2 * half] = transpose_tiles(std::array::from_fn(|r| {
                        _mm512_and_si512(pos[r], non_zero[r])
                    }));
                    tiles[2 * half + 1] = transpose_tiles(std::array::from_fn(|r| {
                        _mm512_andnot_si512(pos[r], non_zero[r])
                    }));
                }
                let tile = tiles.as_ptr() as *const i64;
                let sums: [[__m512i; 8]; W] = std::array::from_fn(|w| {
                    let s = &subset_sums[w][4 * group..4 * group + 4];
                    [
                        s[0][0], s[0][1], s[1][0], s[1][1], s[2][0], s[2][1], s[3][0], s[3][1],
                    ]
                });
                let mut slot = 0;
                for at in 0..64 {
                    if TILE_CHUNK[at] >= chunks {
                        continue;
                    }
                    let nibbles = |first: i64, second: i64| {
                        let (first, second) = (_mm512_set1_epi64(first), _mm512_set1_epi64(second));
                        [
                            _mm512_srlv_epi64(first, low_nibbles),
                            _mm512_srlv_epi64(first, high_nibbles),
                            _mm512_srlv_epi64(second, low_nibbles),
                            _mm512_srlv_epi64(second, high_nibbles),
                        ]
                    };
                    let plus = nibbles(*tile.add(at), *tile.add(128 + at));
                    let minus = nibbles(*tile.add(64 + at), *tile.add(192 + at));
                    for w in 0..W {
                        let s = &sums[w];
                        let look_up = |i: [__m512i; 4]| {
                            _mm512_add_epi64(
                                _mm512_add_epi64(
                                    _mm512_permutex2var_epi64(s[0], i[0], s[1]),
                                    _mm512_permutex2var_epi64(s[2], i[1], s[3]),
                                ),
                                _mm512_add_epi64(
                                    _mm512_permutex2var_epi64(s[4], i[2], s[5]),
                                    _mm512_permutex2var_epi64(s[6], i[3], s[7]),
                                ),
                            )
                        };
                        let acc = accumulators[slot][w];
                        let sum = _mm512_add_epi64(_mm512_load_si512(acc), look_up(plus));
                        _mm512_store_si512(acc, _mm512_sub_epi64(sum, look_up(minus)));
                    }
                    slot += 1;
                }
            }
        }
        true
    }
}

/// AES-256: 14 rounds.
#[cfg(any(
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "aes",
        target_feature = "vaes"
    ),
    all(target_arch = "aarch64", target_feature = "aes")
))]
const ROUND_KEYS: usize = 15;

/// Chunk held by qword `at` of [`transpose_tiles`]'s output.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512bw",
    target_feature = "gfni",
    target_feature = "aes",
    target_feature = "vaes"
))]
const TILE_CHUNK: [usize; 64] = {
    let mut chunk = [0; 64];
    let mut at = 0;
    while at < 64 {
        chunk[at] = 16 * (at % 8 / 2) + 2 * (at / 8) + at % 2;
        at += 1;
    }
    chunk
};

/// Bit `r` of byte `j` of a chunk's tile: row `r`, column `j`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512bw",
    target_feature = "gfni",
    target_feature = "aes",
    target_feature = "vaes"
))]
#[inline(always)]
unsafe fn transpose_tiles(
    rows: [std::arch::x86_64::__m512i; 8],
) -> [std::arch::x86_64::__m512i; 8] {
    use std::arch::x86_64::*;
    let words = [
        _mm512_unpacklo_epi8(rows[7], rows[6]),
        _mm512_unpackhi_epi8(rows[7], rows[6]),
        _mm512_unpacklo_epi8(rows[5], rows[4]),
        _mm512_unpackhi_epi8(rows[5], rows[4]),
        _mm512_unpacklo_epi8(rows[3], rows[2]),
        _mm512_unpackhi_epi8(rows[3], rows[2]),
        _mm512_unpacklo_epi8(rows[1], rows[0]),
        _mm512_unpackhi_epi8(rows[1], rows[0]),
    ];
    let dwords: [[__m512i; 4]; 2] = std::array::from_fn(|h| {
        let (lo, hi) = (4 * h, 4 * h + 2);
        [
            _mm512_unpacklo_epi16(words[lo], words[hi]),
            _mm512_unpackhi_epi16(words[lo], words[hi]),
            _mm512_unpacklo_epi16(words[lo + 1], words[hi + 1]),
            _mm512_unpackhi_epi16(words[lo + 1], words[hi + 1]),
        ]
    });
    let unit = _mm512_set1_epi64(0x8040_2010_0804_0201u64 as i64);
    std::array::from_fn(|k| {
        let rows_of_chunk = if k % 2 == 0 {
            _mm512_unpacklo_epi32(dwords[0][k / 2], dwords[1][k / 2])
        } else {
            _mm512_unpackhi_epi32(dwords[0][k / 2], dwords[1][k / 2])
        };
        _mm512_gf2p8affine_epi64_epi8::<0>(unit, rows_of_chunk)
    })
}

/// Four blocks of each row `r < 8` of `pos`, then of `non_zero`, from block `first_block(r)`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512bw",
    target_feature = "gfni",
    target_feature = "aes",
    target_feature = "vaes"
))]
#[inline(always)]
unsafe fn keystream_rows(
    round_keys: &[std::arch::x86_64::__m512i; ROUND_KEYS],
    first_block: impl Fn(usize) -> usize,
) -> [std::arch::x86_64::__m512i; 16] {
    encrypt(
        round_keys,
        std::array::from_fn(|t| counters(t as u64 / 8, first_block(t % 8) as u64)),
    )
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "aes",
    target_feature = "vaes"
))]
#[inline(always)]
unsafe fn counters(mask: u64, first: u64) -> std::arch::x86_64::__m512i {
    use std::arch::x86_64::*;
    let mask = mask as i64;
    let blocks = _mm512_setr_epi64(0, mask, 1, mask, 2, mask, 3, mask);
    _mm512_add_epi64(blocks, _mm512_maskz_set1_epi64(0b0101_0101, first as i64))
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "aes",
    target_feature = "vaes"
))]
#[inline(always)]
unsafe fn encrypt<const N: usize>(
    round_keys: &[std::arch::x86_64::__m512i; ROUND_KEYS],
    blocks: [std::arch::x86_64::__m512i; N],
) -> [std::arch::x86_64::__m512i; N] {
    use std::arch::x86_64::*;
    let mut blocks = blocks.map(|x| _mm512_xor_si512(x, round_keys[0]));
    for round_key in &round_keys[1..ROUND_KEYS - 1] {
        blocks = blocks.map(|x| _mm512_aesenc_epi128(x, *round_key));
    }
    blocks.map(|x| _mm512_aesenclast_epi128(x, round_keys[ROUND_KEYS - 1]))
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "aes",
    target_feature = "vaes"
))]
unsafe fn expand_key(key: [u8; 32]) -> [std::arch::x86_64::__m128i; ROUND_KEYS] {
    use std::arch::x86_64::*;
    // Word-wise prefix XOR of round key `i - 2`, plus the last word of round key `i - 1` through
    // RotWord, SubWord and the round constant (shuffle 0xff) or through SubWord alone (0xaa).
    macro_rules! next {
        ($k:ident, $i:literal, $shuffle:literal, $rcon:literal) => {{
            let t = _mm_shuffle_epi32::<$shuffle>(_mm_aeskeygenassist_si128::<$rcon>($k[$i - 1]));
            let k = $k[$i - 2];
            let k = _mm_xor_si128(k, _mm_slli_si128::<4>(k));
            let k = _mm_xor_si128(k, _mm_slli_si128::<4>(k));
            let k = _mm_xor_si128(k, _mm_slli_si128::<4>(k));
            $k[$i] = _mm_xor_si128(k, t);
        }};
    }
    let mut k = [_mm_loadu_si128(key.as_ptr() as *const __m128i); ROUND_KEYS];
    k[1] = _mm_loadu_si128(key[16..].as_ptr() as *const __m128i);
    next!(k, 2, 0xff, 0x01);
    next!(k, 3, 0xaa, 0x00);
    next!(k, 4, 0xff, 0x02);
    next!(k, 5, 0xaa, 0x00);
    next!(k, 6, 0xff, 0x04);
    next!(k, 7, 0xaa, 0x00);
    next!(k, 8, 0xff, 0x08);
    next!(k, 9, 0xaa, 0x00);
    next!(k, 10, 0xff, 0x10);
    next!(k, 11, 0xaa, 0x00);
    next!(k, 12, 0xff, 0x20);
    next!(k, 13, 0xaa, 0x00);
    next!(k, 14, 0xff, 0x40);
    k
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "aes",
    target_feature = "vaes"
))]
fn fill_keystream(key: [u8; 32], mask: u64, first_block: u64, out: &mut [u8]) {
    use std::arch::x86_64::*;
    unsafe {
        let round_keys = expand_key(key).map(|k| _mm512_broadcast_i32x4(k));
        for (n, out) in out.chunks_mut(1024).enumerate() {
            let blocks: [__m512i; 16] = encrypt(
                &round_keys,
                std::array::from_fn(|z| counters(mask, first_block + (64 * n + 4 * z) as u64)),
            );
            let mut bytes = [0u8; 1024];
            for (z, block) in blocks.iter().enumerate() {
                _mm512_storeu_si512(bytes.as_mut_ptr().add(64 * z) as *mut __m512i, *block);
            }
            out.copy_from_slice(&bytes[..out.len()]);
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
fn fill_keystream(key: [u8; 32], mask: u64, first_block: u64, out: &mut [u8]) {
    use std::arch::aarch64::*;
    unsafe {
        let round_keys = expand_key(key);
        for (n, out) in out.chunks_mut(128).enumerate() {
            let blocks: [uint8x16_t; 8] = encrypt(
                &round_keys,
                std::array::from_fn(|b| {
                    let mut block = [0u8; 16];
                    block[..8].copy_from_slice(&(first_block + (8 * n + b) as u64).to_le_bytes());
                    block[8..].copy_from_slice(&mask.to_le_bytes());
                    vld1q_u8(block.as_ptr())
                }),
            );
            let mut bytes = [0u8; 128];
            for (b, block) in blocks.iter().enumerate() {
                vst1q_u8(bytes.as_mut_ptr().add(16 * b), *block);
            }
            out.copy_from_slice(&bytes[..out.len()]);
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
unsafe fn encrypt<const N: usize>(
    round_keys: &[std::arch::aarch64::uint8x16_t; ROUND_KEYS],
    blocks: [std::arch::aarch64::uint8x16_t; N],
) -> [std::arch::aarch64::uint8x16_t; N] {
    use std::arch::aarch64::*;
    let mut blocks = blocks;
    for round_key in &round_keys[..ROUND_KEYS - 2] {
        blocks = blocks.map(|x| vaesmcq_u8(vaeseq_u8(x, *round_key)));
    }
    blocks.map(|x| {
        veorq_u8(
            vaeseq_u8(x, round_keys[ROUND_KEYS - 2]),
            round_keys[ROUND_KEYS - 1],
        )
    })
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
unsafe fn expand_key(key: [u8; 32]) -> [std::arch::aarch64::uint8x16_t; ROUND_KEYS] {
    use std::arch::aarch64::*;
    const RCON: [u32; 7] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40];
    let mut w = [0u32; 4 * ROUND_KEYS];
    for (i, word) in key.chunks_exact(4).enumerate() {
        w[i] = u32::from_le_bytes(word.try_into().unwrap());
    }
    for i in 8..4 * ROUND_KEYS {
        let mut t = w[i - 1];
        if i % 4 == 0 {
            // AESE with a zero key and four equal columns is SubBytes alone.
            let column = vreinterpretq_u8_u32(vdupq_n_u32(t));
            t = vgetq_lane_u32::<0>(vreinterpretq_u32_u8(vaeseq_u8(column, vdupq_n_u8(0))));
            if i % 8 == 0 {
                t = t.rotate_right(8) ^ RCON[i / 8 - 1];
            }
        }
        w[i] = w[i - 8] ^ t;
    }
    std::array::from_fn(|r| vreinterpretq_u8_u32(vld1q_u32(w[4 * r..].as_ptr())))
}

#[cfg(not(any(
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "aes",
        target_feature = "vaes"
    ),
    all(target_arch = "aarch64", target_feature = "aes")
)))]
fn fill_keystream(key: [u8; 32], mask: u64, first_block: u64, out: &mut [u8]) {
    fill_keystream_portable(key, mask, first_block, out)
}

#[cfg_attr(
    any(
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "aes",
            target_feature = "vaes"
        ),
        all(target_arch = "aarch64", target_feature = "aes")
    ),
    allow(dead_code)
)]
fn fill_keystream_portable(key: [u8; 32], mask: u64, first_block: u64, out: &mut [u8]) {
    use aes::cipher::{BlockEncrypt, KeyInit};
    let mut blocks: Vec<aes::Block> = (first_block..first_block + out.len().div_ceil(16) as u64)
        .map(|i| {
            let mut block = aes::Block::default();
            block[..8].copy_from_slice(&i.to_le_bytes());
            block[8..].copy_from_slice(&mask.to_le_bytes());
            block
        })
        .collect();
    aes::Aes256::new(&key.into()).encrypt_blocks(&mut blocks);
    for (out, byte) in out.iter_mut().zip(blocks.iter().flatten()) {
        *out = *byte;
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

        let k_pos = self.pos_masks()[(row, chunk)];
        let k_inc = self.non_zero_masks()[(row, chunk)];

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
    fn sampled_masks_fingerprint() {
        let mut projection_matrix = ProjectionMatrix::new(2, 256);
        projection_matrix.sample(&mut HashWrapper::new());
        let mut hasher = blake3::Hasher::new();
        hasher.update(&projection_matrix.pos_masks().data);
        hasher.update(&projection_matrix.non_zero_masks().data);
        assert_eq!(
            hasher.finalize().to_hex().as_str(),
            "5ffac3ceae0164148a47d29644eaa1603f2d4678927df34a651b896fbb70ef28"
        );
    }

    #[test]
    fn keystream_is_aes_256_ctr() {
        let mut block = [0u8; 16];
        let key = std::array::from_fn(|i| i as u8);
        fill_keystream(key, 0xffeeddccbbaa9988, 0x7766554433221100, &mut block);
        assert_eq!(block, 0x8ea2b7ca516745bfeafc49904b496089u128.to_be_bytes());

        let mut hash_wrapper = HashWrapper::new();
        for (len, mask, first_block) in [
            (16, 1, 0),
            (100, 0, 5),
            (1024, 1, 0),
            (5000, 0, 1 << 40),
            (65536 + 48, 1, 0),
        ] {
            let mut key = [0u8; 32];
            hash_wrapper.fill_from_xof(b"key", &mut key);
            let (mut fast, mut portable) = (vec![0u8; len], vec![0u8; len]);
            fill_keystream(key, mask, first_block, &mut fast);
            fill_keystream_portable(key, mask, first_block, &mut portable);
            assert_eq!(
                fast, portable,
                "length {len}, mask {mask}, first block {first_block}"
            );
        }
    }

    #[test]
    fn entries_are_one_half_zero_one_quarter_each_sign() {
        let mut projection_matrix = ProjectionMatrix::new(64, 256);
        projection_matrix.sample(&mut HashWrapper::new());
        let (pos, non_zero) = (
            projection_matrix.pos_masks(),
            projection_matrix.non_zero_masks(),
        );
        let entries = (8 * pos.data.len()) as f64;
        let ones =
            |bytes: &mut dyn Iterator<Item = u8>| bytes.map(u8::count_ones).sum::<u32>() as f64;
        let plus = ones(&mut pos.data.iter().zip(&non_zero.data).map(|(p, n)| p & n));
        let non_zero = ones(&mut non_zero.data.iter().copied());
        assert!((non_zero / entries - 0.5).abs() < 0.005);
        assert!((plus / entries - 0.25).abs() < 0.005);
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512bw",
        target_feature = "gfni",
        target_feature = "aes",
        target_feature = "vaes"
    ))]
    #[test]
    fn generated_rows_match_the_materialised_masks() {
        use std::arch::x86_64::*;
        let mut hash_wrapper = HashWrapper::new();
        for (ratio, height) in [(1, 128), (1, 256), (2, 256), (8, 64), (32, 256)] {
            let mut projection_matrix = ProjectionMatrix::new(ratio, height);
            projection_matrix.sample(&mut hash_wrapper);
            let weights: Vec<u64> = (0..height)
                .map(|_| hash_wrapper.sample_u64() >> 12)
                .collect();
            let mut acc = vec![unsafe { _mm512_setzero_si512() }; projection_matrix.width];
            let at = acc.as_mut_ptr();
            assert!(unsafe {
                projection_matrix
                    .accumulate_signed_sums::<1>([&weights], acc.len(), |_, chunk| at.add(chunk))
            });
            let lanes = acc
                .iter()
                .flat_map(|x| unsafe { std::mem::transmute::<__m512i, [u64; 8]>(*x) });
            for (col, lane) in lanes.enumerate() {
                let expected =
                    (0..height).fold(0u64, |sum, row| match projection_matrix[(row, col)] {
                        (true, true) => sum.wrapping_add(weights[row]),
                        (false, true) => sum.wrapping_sub(weights[row]),
                        _ => sum,
                    });
                assert_eq!(
                    lane, expected,
                    "ratio {ratio}, height {height}, column {col}"
                );
            }
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512bw",
        target_feature = "gfni",
        target_feature = "aes",
        target_feature = "vaes"
    ))]
    #[test]
    fn lookup_kernel_declines_what_it_does_not_cover() {
        let mut hash_wrapper = HashWrapper::new();
        let untouched = |_: usize, _: usize| -> *mut std::arch::x86_64::__m512i {
            panic!("accumulator touched")
        };
        let weights = vec![1u64; 256];
        let unsampled = ProjectionMatrix::new(2, 256);
        assert!(!unsafe { unsampled.accumulate_signed_sums::<1>([&weights[..]], 64, untouched) });
        for (ratio, height, capacity, rows) in [
            (2, 256, 63, 256),
            (2, 256, 64, 255),
            (16, 16, 0, 16),
            (3, 40, 15, 40),
            (1, 24, 3, 24),
        ] {
            let mut projection_matrix = ProjectionMatrix::new(ratio, height);
            projection_matrix.sample(&mut hash_wrapper);
            assert!(!unsafe {
                projection_matrix.accumulate_signed_sums::<1>(
                    [&weights[..rows]],
                    capacity,
                    untouched,
                )
            });
        }
    }

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
