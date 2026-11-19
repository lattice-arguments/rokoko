use crate::ring::config::MOD_Q;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use crate::ring::{config::DEGREE, ring_arithmetic::RingElement};
use crate::rok::proj::coarse::Signed16RingElement;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::arch::x86_64::{
    __m512i, _mm512_abs_epi16, _mm512_add_epi16, _mm512_add_epi32, _mm512_castsi512_si256,
    _mm512_cvtepi16_epi32, _mm512_cvtepi32_epi64, _mm512_extracti64x4_epi64, _mm512_load_si512,
    _mm512_mask_add_epi64, _mm512_max_epu16, _mm512_movepi64_mask, _mm512_set1_epi64,
    _mm512_setzero_si512, _mm512_store_si512, _mm512_storeu_si512, _mm512_sub_epi16,
};
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
use std::arch::x86_64::{
    _mm512_cvtepu16_epi32, _mm512_max_epu32, _mm512_reduce_max_epu32, _mm_prefetch, _MM_HINT_T0,
};

/// Uses `i32` accumulators (to avoid `i16` overflow) and final modular
/// reduction to produce `u64` residues in `[0, Q)`.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub fn project_one_row_i16_to_u64<const DEGREE: usize>(
    subwitness_i16: &[Signed16RingElement],
    pos: &[u16],
    neg: &[u16],
    out_u64: &mut [u64; DEGREE],
) {
    debug_assert!(DEGREE % 16 == 0);

    let mut acc: [i32; DEGREE] = [0; DEGREE];

    for &i in pos {
        let row = &subwitness_i16[i as usize].0;
        for k in 0..DEGREE {
            acc[k] += row[k] as i32;
        }
    }

    for &i in neg {
        let row = &subwitness_i16[i as usize].0;
        for k in 0..DEGREE {
            acc[k] -= row[k] as i32;
        }
    }

    let q = MOD_Q as i64;

    for k in 0..DEGREE {
        let x = acc[k] as i64;
        let mut r = x % q;
        if r < 0 {
            r += q;
        }
        out_u64[k] = r as u64;
    }
}

// Computes out[row] = sum of witness elements listed in pos[row] minus those
// in neg[row], coefficient-wise and exactly, for one witness block.
//
// The loop nest exists to keep the loads cache-hot (L1, spill L2, never L3):
//   k    - one 32-lane (64 B) slice of the coefficients at a time. Each
//          witness element is DEGREE / 32 such lines; a k-pass touches exactly one
//          line per element.
//   tile - the witness is walked in windows of at most 256 elements:
//          256 x 64 B = 16 KB, which stays L1-resident while...
//   row  - ...ALL output rows consume their entries falling inside the
//          window. So a witness line is fetched from memory once per k and
//          then reused ~H/2 times from L1.
//
// A row adds in i16. The adds wrap, but a partial sum over `run` tiles takes
// at most `run * tile` entries, each at most `bound` in magnitude, and the two
// are chosen so that this stays within i16: the partial is exact. Every `run`
// tiles it is widened into the row's i32 sum and restarted. Nothing bounds the
// row's whole sum by i16: a block of base-2^9 digits projects with a standard
// deviation near 2^13, so coefficients past 2^15 do occur.
//
// Because rows are revisited per tile, their sums cannot live in registers;
// the partials live in `scratch` (H x 64 B = 16 KB, also L1-resident) and the
// widened sums in `wide` (H x 128 B), touched once per run.
// The offset lists are sorted, so `pos_cur`/`neg_cur` remember per row how
// far its list has been consumed; the next tile continues from there, and
// "entry belongs to this tile" is a single compare against the tile's end.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub fn project_rows_sparse_tiled<const DEGREE: usize>(
    subwitness_i16: &[Signed16RingElement],
    pos: &[u32],
    pos_bounds: &[usize],
    neg: &[u32],
    neg_bounds: &[usize],
    out: &mut [RingElement],
) {
    const TILE: usize = 256;

    #[repr(align(64))]
    #[derive(Clone, Copy)]
    struct Acc([i16; 32]);

    #[repr(align(64))]
    #[derive(Clone, Copy)]
    struct Wide([i32; 32]);

    debug_assert!(DEGREE % 32 == 0);
    let h = out.len();
    debug_assert_eq!(pos_bounds.len(), h + 1);
    debug_assert_eq!(neg_bounds.len(), h + 1);
    let row_len = subwitness_i16.len();
    let elem_bytes = core::mem::size_of::<Signed16RingElement>();

    let bound = max_abs_i16(subwitness_i16);
    if bound > i16::MAX as usize {
        // An i16::MIN digit negates out of i16 on its own; no partial is exact.
        project_rows_scalar(subwitness_i16, pos, pos_bounds, neg, neg_bounds, out);
        return;
    }
    assert!(
        row_len * bound <= i32::MAX as usize,
        "a {row_len}-element block of digits up to {bound} overflows the i32 row sums"
    );
    // Entries a partial may take, split into `run` tiles of `tile` elements.
    let limit = i16::MAX as usize / bound.max(1);
    let tile = limit.min(TILE);
    let run = limit / tile;

    let mut scratch = vec![Acc([0i16; 32]); h];
    let mut wide = vec![Wide([0i32; 32]); h];
    let mut pos_cur = vec![0usize; h];
    let mut neg_cur = vec![0usize; h];

    let base = subwitness_i16.as_ptr() as *const u8;

    unsafe {
        for k in 0..DEGREE / 32 {
            // The list entries are byte offsets of ELEMENTS; adding them to
            // `chunk` (base shifted to this k-slice) addresses the element's
            // k-th line directly.
            let chunk = base.add(k * 64);
            // Fresh coefficient slice: clear the sums, rewind every row's
            // cursor to the start of its list.
            wide.fill(Wide([0i32; 32]));
            pos_cur.copy_from_slice(&pos_bounds[..h]);
            neg_cur.copy_from_slice(&neg_bounds[..h]);

            let mut tile_start = 0usize;
            let mut tile_index = 0usize;
            while tile_start < row_len {
                // Tile boundary in the same units as the list entries (bytes).
                let tile_end = ((tile_start + tile).min(row_len) * elem_bytes) as u32;
                let opens = tile_index % run == 0;
                let closes = tile_index % run == run - 1 || tile_start + tile >= row_len;
                for row in 0..h {
                    // Two independent accumulators so the adds and the
                    // subtracts form separate dependency chains: a0 continues
                    // this row's partial, a1 collects the negatives, and
                    // a0 - a1 is the partial after this tile.
                    let mut a0 = if opens {
                        _mm512_setzero_si512()
                    } else {
                        _mm512_load_si512(scratch[row].0.as_ptr() as *const __m512i)
                    };
                    let mut a1 = _mm512_setzero_si512();

                    // Consume this row's +1 entries that fall inside the
                    // tile. The load fuses into the add (one vpaddw with a
                    // memory operand) and stays cache-hot: the address is
                    // inside the tile (measured: ~78% of kernel loads L1-hit,
                    // the rest L2-hit, ~nothing reaches L3).
                    let end = pos_bounds[row + 1];
                    let mut i = *pos_cur.get_unchecked(row);
                    while i < end && *pos.get_unchecked(i) < tile_end {
                        a0 = _mm512_add_epi16(
                            a0,
                            _mm512_load_si512(
                                chunk.add(*pos.get_unchecked(i) as usize) as *const __m512i
                            ),
                        );
                        i += 1;
                    }
                    *pos_cur.get_unchecked_mut(row) = i;

                    // Same for the -1 entries.
                    let end = neg_bounds[row + 1];
                    let mut i = *neg_cur.get_unchecked(row);
                    while i < end && *neg.get_unchecked(i) < tile_end {
                        a1 = _mm512_add_epi16(
                            a1,
                            _mm512_load_si512(
                                chunk.add(*neg.get_unchecked(i) as usize) as *const __m512i
                            ),
                        );
                        i += 1;
                    }
                    *neg_cur.get_unchecked_mut(row) = i;

                    let partial = _mm512_sub_epi16(a0, a1);
                    if closes {
                        widen_into(wide[row].0.as_mut_ptr(), partial);
                    } else {
                        _mm512_store_si512(scratch[row].0.as_mut_ptr() as *mut __m512i, partial);
                    }
                }
                tile_start += tile;
                tile_index += 1;
            }

            // All tiles done: `wide` holds the finished sums for this
            // coefficient slice. Lift them to residues in [0, Q) and write
            // them into the output elements.
            for row in 0..h {
                let acc = wide[row].0.as_ptr() as *const __m512i;
                let dst = out[row].v.as_mut_ptr().add(k * 32);
                convert_i32x16_to_u64_mod_q(dst, _mm512_load_si512(acc));
                convert_i32x16_to_u64_mod_q(dst.add(16), _mm512_load_si512(acc.add(1)));
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
pub(crate) const GROUP: usize = 4;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
pub(crate) const GROUP_SUMS: usize = 81;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
pub(crate) const PASS: usize = 4;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
const BLOCK: usize = PASS * GROUP;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
pub(crate) fn project_rows_grouped<const DEGREE: usize>(
    subwitness_i16: &[Signed16RingElement],
    offsets: &[u64],
    lists: (&[u32], &[usize], &[u32], &[usize]),
    out: &mut [RingElement],
) {
    let blocks = subwitness_i16.len() / BLOCK;
    if blocks * BLOCK != subwitness_i16.len()
        || !unsafe { project_rows_in_blocks::<DEGREE>(subwitness_i16, offsets, out) }
    {
        let (pos, pos_bounds, neg, neg_bounds) = lists;
        project_rows_sparse_tiled::<DEGREE>(subwitness_i16, pos, pos_bounds, neg, neg_bounds, out);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
unsafe fn project_rows_in_blocks<const DEGREE: usize>(
    subwitness_i16: &[Signed16RingElement],
    offsets: &[u64],
    out: &mut [RingElement],
) -> bool {
    let lines = Lines {
        base: subwitness_i16.as_ptr() as *const u8,
        stride: core::mem::size_of::<Signed16RingElement>(),
        len: subwitness_i16.len(),
        slices: DEGREE / 32,
    };
    ternary_rows(lines, offsets, out.len(), |k, wide| {
        for (row, out) in out.iter_mut().enumerate() {
            let dst = out.v.as_mut_ptr().add(k * 32);
            convert_i32x16_to_u64_mod_q(dst, wide[2 * row]);
            convert_i32x16_to_u64_mod_q(dst.add(16), wide[2 * row + 1]);
        }
    })
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
pub(crate) fn project_lanes_grouped(
    lanes: &[__m512i],
    offsets: &[u64],
    h: usize,
) -> Option<Vec<__m512i>> {
    let blocks = lanes.len() / BLOCK;
    if blocks * BLOCK != lanes.len() {
        return None;
    }
    let mut sums = vec![unsafe { _mm512_setzero_si512() }; 2 * h];
    let lines = Lines {
        base: lanes.as_ptr() as *const u8,
        stride: 64,
        len: lanes.len(),
        slices: 1,
    };
    unsafe { ternary_rows(lines, offsets, h, |_, wide| sums.copy_from_slice(wide)) }.then_some(sums)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
struct Lines {
    base: *const u8,
    stride: usize,
    len: usize,
    slices: usize,
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
unsafe fn ternary_rows(
    lines: Lines,
    offsets: &[u64],
    h: usize,
    mut emit: impl FnMut(usize, &[__m512i]),
) -> bool {
    let Lines {
        base,
        stride,
        len,
        slices,
    } = lines;
    let blocks = len / BLOCK;
    assert_eq!(offsets.len(), blocks * h);
    if len * (i16::MAX as usize / BLOCK) > i32::MAX as usize {
        return false;
    }

    let mut tables = vec![_mm512_setzero_si512(); PASS * GROUP_SUMS];
    let mut scratch = vec![_mm512_setzero_si512(); h];
    let mut wide = vec![_mm512_setzero_si512(); 2 * h];
    let table = tables.as_mut_ptr();
    let sums = Sums {
        tables: table as *const u8,
        scratch: scratch.as_mut_ptr(),
        wide: wide.as_mut_ptr(),
        rows: h,
    };

    for k in 0..slices {
        let chunk = base.add(k * 64);
        wide.fill(_mm512_setzero_si512());
        let mut budget = 0;
        for block in 0..blocks {
            if block + 1 < blocks {
                let next = chunk.add((block + 1) * BLOCK * stride);
                for e in 0..BLOCK {
                    _mm_prefetch::<_MM_HINT_T0>(next.add(e * stride) as *const i8);
                }
            }
            let mut largest = _mm512_setzero_si512();
            for group in 0..PASS {
                let first = chunk.add((block * PASS + group) * GROUP * stride);
                let w: [__m512i; GROUP] = std::array::from_fn(|c| {
                    let line = _mm512_load_si512(first.add(c * stride) as *const __m512i);
                    largest = _mm512_max_epu16(largest, _mm512_abs_epi16(line));
                    line
                });
                ternary_sums(table.add(group * GROUP_SUMS), w);
            }
            // a row adds at most BLOCK digits per block, so a partial within an i16 budget is exact
            let bound = BLOCK * largest_lane(largest);
            if bound > i16::MAX as usize {
                return false;
            }
            if budget + bound > i16::MAX as usize {
                sums.widen();
                budget = 0;
            }
            let selections = &offsets[block * h..(block + 1) * h];
            if budget == 0 {
                sums.add::<true>(selections);
            } else {
                sums.add::<false>(selections);
            }
            budget += bound;
        }
        sums.widen();
        emit(k, &wide);
    }
    true
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
struct Sums {
    tables: *const u8,
    scratch: *mut __m512i,
    wide: *mut __m512i,
    rows: usize,
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl Sums {
    #[inline(always)]
    unsafe fn add<const OPENS: bool>(&self, selections: &[u64]) {
        const _: () = assert!(PASS == 4);
        for (row, &s) in selections.iter().enumerate() {
            let sum = |group: usize| {
                let offset = (s >> (16 * group)) as u16 as usize;
                _mm512_load_si512(
                    self.tables.add(group * GROUP_SUMS * 64 + offset) as *const __m512i
                )
            };
            let mut partial = _mm512_add_epi16(
                _mm512_add_epi16(sum(0), sum(1)),
                _mm512_add_epi16(sum(2), sum(3)),
            );
            if !OPENS {
                partial = _mm512_add_epi16(partial, _mm512_load_si512(self.scratch.add(row)));
            }
            _mm512_store_si512(self.scratch.add(row), partial);
        }
    }

    #[inline(always)]
    unsafe fn widen(&self) {
        for row in 0..self.rows {
            widen_into(self.wide.add(2 * row) as *mut i32, *self.scratch.add(row));
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn largest_lane(x: __m512i) -> usize {
    let low = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(x));
    let high = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64::<1>(x));
    _mm512_reduce_max_epu32(_mm512_max_epu32(low, high)) as usize
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn ternary_sums(table: *mut __m512i, [w0, w1, w2, w3]: [__m512i; GROUP]) {
    let signed = |s: __m512i, d: usize, w: __m512i| match d {
        0 => _mm512_sub_epi16(s, w),
        1 => s,
        _ => _mm512_add_epi16(s, w),
    };
    let first = [
        signed(_mm512_setzero_si512(), 0, w0),
        _mm512_setzero_si512(),
        w0,
    ];
    let second: [[__m512i; 3]; 3] = std::array::from_fn(|d| first.map(|s| signed(s, d, w1)));
    for d in 0..3 {
        for (i, &s) in second.as_flattened().iter().enumerate() {
            let s = signed(s, d, w2);
            let at = table.add(9 * d + i);
            _mm512_store_si512(at, _mm512_sub_epi16(s, w3));
            _mm512_store_si512(at.add(27), s);
            _mm512_store_si512(at.add(54), _mm512_add_epi16(s, w3));
        }
    }
}

/// `acc[lane] += partial[lane]` over 32 lanes, `acc` as i32.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn widen_into(acc: *mut i32, partial: __m512i) {
    let acc = acc as *mut __m512i;
    let lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(partial));
    let hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(partial));
    _mm512_store_si512(acc, _mm512_add_epi32(_mm512_load_si512(acc), lo));
    _mm512_store_si512(
        acc.add(1),
        _mm512_add_epi32(_mm512_load_si512(acc.add(1)), hi),
    );
}

/// The largest `|x|` over every coefficient of `block`, as the unsigned magnitude (so `i16::MIN`
/// reads as `2^15`).
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn max_abs_i16(block: &[Signed16RingElement]) -> usize {
    unsafe {
        let mut m = _mm512_setzero_si512();
        for element in block {
            for k in 0..DEGREE / 32 {
                let line = _mm512_load_si512(element.0.as_ptr().add(k * 32) as *const __m512i);
                m = _mm512_max_epu16(m, _mm512_abs_epi16(line));
            }
        }
        let mut lanes = [0u16; 32];
        _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, m);
        lanes.into_iter().max().unwrap_or(0) as usize
    }
}

/// `project_rows_sparse_tiled` one coefficient at a time in i64, for blocks the tiled kernel
/// cannot sum exactly.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn project_rows_scalar(
    subwitness_i16: &[Signed16RingElement],
    pos: &[u32],
    pos_bounds: &[usize],
    neg: &[u32],
    neg_bounds: &[usize],
    out: &mut [RingElement],
) {
    let elem_bytes = core::mem::size_of::<Signed16RingElement>();
    for (row, out) in out.iter_mut().enumerate() {
        let mut acc = [0i64; DEGREE];
        for (list, bounds, sign) in [(pos, pos_bounds, 1i64), (neg, neg_bounds, -1i64)] {
            for &offset in &list[bounds[row]..bounds[row + 1]] {
                let element = &subwitness_i16[offset as usize / elem_bytes].0;
                for (a, &x) in acc.iter_mut().zip(element.iter()) {
                    *a += sign * x as i64;
                }
            }
        }
        for (dst, &a) in out.v.iter_mut().zip(acc.iter()) {
            *dst = a.rem_euclid(MOD_Q as i64) as u64;
        }
    }
}

// Centered i32 lanes -> canonical residues in [0, Q): sign-extend, add Q to negatives.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn convert_i32x16_to_u64_mod_q(dst_u64: *mut u64, v32x16: __m512i) {
    let q = _mm512_set1_epi64(MOD_Q as i64);
    for (half, x) in [
        _mm512_castsi512_si256(v32x16),
        _mm512_extracti64x4_epi64::<1>(v32x16),
    ]
    .into_iter()
    .enumerate()
    {
        let w = _mm512_cvtepi32_epi64(x);
        let neg = _mm512_movepi64_mask(w);
        let r = _mm512_mask_add_epi64(w, neg, w, q);
        _mm512_store_si512(dst_u64.add(half * 8) as *mut __m512i, r);
    }
}

/// Wrapper for _mm512_add_epi16 that checks for overflows in debug-decomp, otherwise just adds.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
pub unsafe fn add_epi16_checked(a: __m512i, b: __m512i) -> __m512i {
    #[cfg(feature = "debug-decomp")]
    {
        use std::arch::x86_64::{_mm512_add_epi16, _mm512_cmpgt_epi16_mask, _mm512_set1_epi16};
        let sum = _mm512_add_epi16(a, b);
        let sign_a = _mm512_cmpgt_epi16_mask(a, _mm512_set1_epi16(-1));
        let sign_b = _mm512_cmpgt_epi16_mask(b, _mm512_set1_epi16(-1));
        let sign_sum = _mm512_cmpgt_epi16_mask(sum, _mm512_set1_epi16(-1));
        let same_sign = !(sign_a ^ sign_b); // 1 where same sign
        let overflow = same_sign & (sign_a ^ sign_sum); // 1 where overflow
        if overflow != 0 {
            panic!(
                "add_epi16_checked: overflow detected in SIMD lane(s): {:032b}",
                overflow
            );
        }
        sum
    }
    #[cfg(not(feature = "debug-decomp"))]
    {
        use std::arch::x86_64::_mm512_add_epi16;
        _mm512_add_epi16(a, b)
    }
}

/// Wrapper for _mm512_sub_epi16 that checks for overflows in debug-decomp, otherwise just subtracts.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
pub unsafe fn sub_epi16_checked(a: __m512i, b: __m512i) -> __m512i {
    #[cfg(feature = "debug-decomp")]
    {
        use std::arch::x86_64::{_mm512_cmpgt_epi16_mask, _mm512_set1_epi16, _mm512_sub_epi16};
        let diff = _mm512_sub_epi16(a, b);
        let sign_a = _mm512_cmpgt_epi16_mask(a, _mm512_set1_epi16(-1));
        let sign_b = _mm512_cmpgt_epi16_mask(b, _mm512_set1_epi16(-1));
        let sign_diff = _mm512_cmpgt_epi16_mask(diff, _mm512_set1_epi16(-1));
        let diff_sign = sign_a ^ sign_b; // 1 where different sign
        let overflow = diff_sign & (sign_a ^ sign_diff); // 1 where overflow
        if overflow != 0 {
            panic!(
                "sub_epi16_checked: overflow detected in SIMD lane(s): {:032b}",
                overflow
            );
        }
        diff
    }
    #[cfg(not(feature = "debug-decomp"))]
    {
        use std::arch::x86_64::_mm512_sub_epi16;
        _mm512_sub_epi16(a, b)
    }
}
