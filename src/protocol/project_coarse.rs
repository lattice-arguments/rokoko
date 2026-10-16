//! Coarse JL projection: out[r] = sum_i J[r,i] * w[i], J ternary (~half zeros).
//! Digits are short in the coefficient domain, so: inverse-NTT + narrow the
//! witness to i16 once, drop J's zeros into per-row +/- index lists, accumulate
//! in i16 over L1-resident tiles, lift back to [0, Q) and forward-NTT the image.

use crate::common::{
    arithmetic::centered_i16_from_u64_mod_q,
    config::DEGREE,
    matrix::VerticallyAlignedMatrix,
    projection_matrix::ProjectionMatrix,
    ring_arithmetic::{Representation, RingElement},
};

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
use crate::common::arithmetic::project_one_row_i16_to_u64;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    not(target_feature = "avx512bw")
))]
use crate::common::arithmetic::project_rows_sparse_tiled;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
use crate::common::arithmetic::{project_rows_grouped, GROUP, GROUP_SUMS, PASS};

#[derive(Clone, Copy)]
#[repr(align(64))]
pub struct Signed16RingElement(pub [i16; DEGREE]);

pub fn prepare_i16_witness(
    witness: &VerticallyAlignedMatrix<RingElement>,
) -> VerticallyAlignedMatrix<Signed16RingElement> {
    let mut witness_i16: Vec<Signed16RingElement> =
        vec![Signed16RingElement([0i16; DEGREE]); witness.data.len()];

    #[repr(align(64))]
    struct Buf([u64; DEGREE]);
    let mut temp = Buf([0u64; DEGREE]);

    for col in 0..witness.used_cols {
        let src = witness.col_slice(col, 0, witness.height);
        let dst = &mut witness_i16[col * witness.height..][..witness.height];
        for (out, cr) in dst.iter_mut().zip(src) {
            cr.strided_coefficients_into(&mut temp.0);
            centered_i16_from_u64_mod_q(&mut out.0, &temp.0);
        }
    }

    VerticallyAlignedMatrix::<Signed16RingElement> {
        width: witness.width,
        height: witness.height,
        data: witness_i16,
        used_cols: witness.used_cols,
    }
}

/// Reference projection in ring arithmetic (no i16 path), for debug checks.
pub fn project_ring(
    witness: &VerticallyAlignedMatrix<RingElement>,
    projection_matrix: &ProjectionMatrix,
) -> VerticallyAlignedMatrix<RingElement> {
    let mut image = VerticallyAlignedMatrix {
        data: vec![
            RingElement::zero(Representation::IncompleteNTT);
            witness.height / projection_matrix.projection_ratio * witness.width
        ],
        width: witness.width,
        height: witness.height / projection_matrix.projection_ratio,
        used_cols: witness.width,
    };
    let row_len = projection_matrix.projection_ratio * projection_matrix.projection_height;
    for col in 0..witness.width {
        for chunk in 0..image.height / projection_matrix.projection_height {
            for inner_row in 0..projection_matrix.projection_height {
                let mut acc = RingElement::zero(Representation::IncompleteNTT);
                for i in 0..row_len {
                    let (is_positive, is_non_zero) = projection_matrix[(inner_row, i)];
                    if !is_non_zero {
                        continue;
                    }
                    let src = &witness[(chunk * row_len + i, col)];
                    if is_positive {
                        acc += src;
                    } else {
                        acc -= src;
                    }
                }
                image[(chunk * projection_matrix.projection_height + inner_row, col)] = acc;
            }
        }
    }
    image
}

pub fn project(
    witness_16: &VerticallyAlignedMatrix<Signed16RingElement>,
    projection_matrix: &ProjectionMatrix,
) -> VerticallyAlignedMatrix<RingElement> {
    let mut projection_image = VerticallyAlignedMatrix {
        data: vec![
            RingElement::zero(Representation::IncompleteNTT);
            witness_16.height / projection_matrix.projection_ratio * witness_16.width
        ],
        width: witness_16.width,
        height: witness_16.height / projection_matrix.projection_ratio,
        used_cols: witness_16.width,
    };

    debug_assert_eq!(projection_image.width, witness_16.width);
    debug_assert_eq!(
        projection_image.height * projection_matrix.projection_ratio,
        witness_16.height
    );

    for i in projection_image.data.iter_mut() {
        i.from_incomplete_ntt_to_strided_coefficients();
    }

    let row_len = projection_matrix.projection_ratio * projection_matrix.projection_height;

    // Sparse row lists + L1-tiled kernel: each witness chunk line is loaded
    // once per tile pass and reused by all projection rows.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        let (pos, pos_bounds, neg, neg_bounds) = signed_offset_lists(projection_matrix);
        #[cfg(target_feature = "avx512bw")]
        let offsets = ternary_group_offsets(projection_matrix);

        for col in 0..witness_16.used_cols {
            for rows_chunk in 0..projection_image.height / projection_matrix.projection_height {
                let subwitness_i16 =
                    witness_16.col_slice(col, rows_chunk * row_len, (rows_chunk + 1) * row_len);

                let projection_subimage = projection_image.col_slice_mut(
                    col,
                    rows_chunk * projection_matrix.projection_height,
                    (rows_chunk + 1) * projection_matrix.projection_height,
                );

                #[cfg(target_feature = "avx512bw")]
                project_rows_grouped::<DEGREE>(
                    subwitness_i16,
                    &offsets,
                    (&pos, &pos_bounds, &neg, &neg_bounds),
                    projection_subimage,
                );
                #[cfg(not(target_feature = "avx512bw"))]
                project_rows_sparse_tiled::<DEGREE>(
                    subwitness_i16,
                    &pos,
                    &pos_bounds,
                    &neg,
                    &neg_bounds,
                    projection_subimage,
                );
            }
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        let mut pos_by_row: Vec<Vec<u16>> = (0..projection_matrix.projection_height)
            .map(|_| Vec::<u16>::new())
            .collect();
        let mut neg_by_row: Vec<Vec<u16>> = (0..projection_matrix.projection_height)
            .map(|_| Vec::<u16>::new())
            .collect();

        for inner_row in 0..projection_matrix.projection_height {
            for i in 0..row_len {
                let (is_positive, is_non_zero) = projection_matrix[(inner_row, i)];
                if !is_non_zero {
                    continue;
                }
                if is_positive {
                    pos_by_row[inner_row].push(i as u16);
                } else {
                    neg_by_row[inner_row].push(i as u16);
                }
            }
        }

        for col in 0..witness_16.width {
            for rows_chunk in 0..projection_image.height / projection_matrix.projection_height {
                let subwitness_i16 =
                    witness_16.col_slice(col, rows_chunk * row_len, (rows_chunk + 1) * row_len);

                let projection_subimage = projection_image.col_slice_mut(
                    col,
                    rows_chunk * projection_matrix.projection_height,
                    (rows_chunk + 1) * projection_matrix.projection_height,
                );

                for inner_row in 0..projection_matrix.projection_height {
                    project_one_row_i16_to_u64::<DEGREE>(
                        subwitness_i16,
                        &pos_by_row[inner_row],
                        &neg_by_row[inner_row],
                        &mut projection_subimage[inner_row].v,
                    );
                }
            }
        }
    }

    for i in projection_image.data.iter_mut() {
        i.from_strided_coefficients_to_incomplete_ntt_representation();
    }

    projection_image
}

// Flattens J's two bitplanes into "which witness elements to add / subtract"
// per output row, so the kernel never touches J's zeros (~half of all entries).
//
// For each row the +1 column positions go into one list and the -1 positions
// into another; all rows' lists are glued back-to-back into ONE flat array per
// sign. Example with 3 rows, +1 entries at columns {2,5}, {0}, {1,3,4}:
//
//   pos        = [2, 5, 0, 1, 3, 4]      (times elem_bytes, see below)
//   pos_bounds = [0, 2, 3, 6]            row r owns pos[bounds[r]..bounds[r+1]]
//
// Entries are stored pre-multiplied by the element size, i.e. as BYTE offsets
// into the witness slice, so the kernel's inner loop computes an element
// address as base + offset with no shift. Within a row the offsets ascend
// (built left to right), which lets the kernel walk them with a cursor and
// stop at a tile boundary with one compare.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub fn signed_offset_lists(
    projection_matrix: &ProjectionMatrix,
) -> (Vec<u32>, Vec<usize>, Vec<u32>, Vec<usize>) {
    let h = projection_matrix.projection_height;
    let elem_bytes = core::mem::size_of::<Signed16RingElement>();

    // Pass 1 - count only. J is packed 8 columns per byte in two planes:
    // pos (sign) and nz (nonzero). An entry is +1 where both bits are set
    // (p & n) and -1 where only nz is set (!p & n). Popcounting every byte
    // gives the per-row counts; their running totals are exactly the
    // bounds arrays, and the final totals size the two flat vectors.
    let mut pos_bounds = Vec::with_capacity(h + 1);
    let mut neg_bounds = Vec::with_capacity(h + 1);
    pos_bounds.push(0usize);
    neg_bounds.push(0usize);
    let (mut npos, mut nneg) = (0usize, 0usize);
    for row in 0..h {
        let (pos_row, nz_row) = projection_matrix.row_chunks(row);
        for (&p, &n) in pos_row.iter().zip(nz_row) {
            npos += (p & n).count_ones() as usize;
            nneg += (!p & n).count_ones() as usize;
        }
        pos_bounds.push(npos);
        neg_bounds.push(nneg);
    }

    // Pass 2 - fill, one mask byte (= 8 columns) per compress-store.
    let mut pos: Vec<u32> = Vec::with_capacity(npos);
    let mut neg: Vec<u32> = Vec::with_capacity(nneg);
    unsafe {
        use std::arch::x86_64::{
            _mm256_add_epi32, _mm256_mask_compressstoreu_epi32, _mm256_set1_epi32,
            _mm256_setr_epi32,
        };
        let eb = elem_bytes as i32;
        // Byte offsets of the 8 consecutive columns a mask byte covers,
        // relative to the first of them: [0, eb, 2*eb, ..., 7*eb].
        let lane_offsets = _mm256_setr_epi32(0, eb, 2 * eb, 3 * eb, 4 * eb, 5 * eb, 6 * eb, 7 * eb);
        // Raw write cursors into the (already exactly-sized) vectors.
        let mut pw = pos.as_mut_ptr();
        let mut nw = neg.as_mut_ptr();
        for row in 0..h {
            let (pos_row, nz_row) = projection_matrix.row_chunks(row);
            for (c, (&p, &n)) in pos_row.iter().zip(nz_row).enumerate() {
                // Absolute byte offsets of this byte's 8 columns: byte c
                // starts at column c*8, so base = c*8*eb plus the lane table.
                let offs = _mm256_add_epi32(_mm256_set1_epi32((c * 8) as i32 * eb), lane_offsets);
                // compress-store with the mask byte as __mmask8 writes only
                // the offsets whose bit is set, packed contiguously; advance
                // the cursor by how many that was (popcount).
                let add = p & n;
                _mm256_mask_compressstoreu_epi32(pw as *mut i32, add, offs);
                pw = pw.add(add.count_ones() as usize);
                let sub = !p & n;
                _mm256_mask_compressstoreu_epi32(nw as *mut i32, sub, offs);
                nw = nw.add(sub.count_ones() as usize);
            }
        }
        // Everything was written through raw pointers; the counts from pass 1
        // say how much.
        pos.set_len(npos);
        neg.set_len(nneg);
    }

    (pos, pos_bounds, neg, neg_bounds)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
pub(crate) fn ternary_group_offsets(projection_matrix: &ProjectionMatrix) -> Vec<u64> {
    const _: () = assert!(GROUP == 4 && GROUP_SUMS == 81 && PASS == 4);
    let h = projection_matrix.projection_height;
    let blocks = projection_matrix.projection_width / (GROUP * PASS);
    let digits: [u64; 256] = std::array::from_fn(|masks| {
        (0..GROUP).rev().fold(0, |index, c| {
            let (pos, non_zero) = ((masks >> c) & 1, (masks >> (GROUP + c)) & 1);
            3 * index + if non_zero == 0 { 1 } else { 2 * pos as u64 }
        })
    });
    let mut offsets = vec![0u64; blocks * h];
    for row in 0..h {
        let (pos_row, nz_row) = projection_matrix.row_chunks(row);
        let pairs = pos_row.chunks_exact(2).zip(nz_row.chunks_exact(2));
        for (block, (pos, non_zero)) in pairs.take(blocks).enumerate() {
            let pos = u16::from_le_bytes([pos[0], pos[1]]) as usize;
            let non_zero = u16::from_le_bytes([non_zero[0], non_zero[1]]) as usize;
            offsets[block * h + row] = (0..PASS).fold(0, |packed, group| {
                let shift = GROUP * group;
                let masks = (pos >> shift) & 15 | ((non_zero >> shift) & 15) << GROUP;
                packed | 64 * digits[masks] << (16 * group)
            });
        }
    }
    offsets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{config::MOD_Q, hash::HashWrapper};

    #[test]
    fn test_projection_matches_ring_reference() {
        crate::common::init_common();
        let mut projection_matrix = ProjectionMatrix::new(4, 256);
        projection_matrix.sample(&mut HashWrapper::new());

        let height = 2048;
        let width = 2;
        let witness = VerticallyAlignedMatrix {
            data: (0..height * width)
                .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 16))
                .collect(),
            width,
            height,
            used_cols: width,
        };
        let witness_i16 = prepare_i16_witness(&witness);

        let reference = project_ring(&witness, &projection_matrix);
        let image = project(&witness_i16, &projection_matrix);
        assert_eq!(reference.data, image.data);

        let mut witness_partial = witness;
        for el in witness_partial.data[height..].iter_mut() {
            *el = RingElement::zero(Representation::IncompleteNTT);
        }
        witness_partial.used_cols = 1;
        let witness_partial_i16 = prepare_i16_witness(&witness_partial);
        let reference = project_ring(&witness_partial, &projection_matrix);
        let image = project(&witness_partial_i16, &projection_matrix);
        assert_eq!(reference.data, image.data);
    }

    /// A column of base-2^9 digits, like the one that carries the previous level's projection at
    /// the height-8192 level of a two-opening chain, projects past i16 in some coefficient.
    #[test]
    fn projection_is_exact_past_i16() {
        crate::common::init_common();
        let mut projection_matrix = ProjectionMatrix::new(32, 256);
        projection_matrix.sample(&mut HashWrapper::new());

        let height = 8192;
        let witness = VerticallyAlignedMatrix {
            data: (0..height)
                .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 257))
                .collect(),
            width: 1,
            height,
            used_cols: 1,
        };

        let reference = project_ring(&witness, &projection_matrix);
        let past_i16 = reference.data.iter().any(|element| {
            let mut element = element.clone();
            element.from_incomplete_ntt_to_strided_coefficients();
            element
                .v
                .iter()
                .any(|&c| c.min(MOD_Q - c) > i16::MAX as u64)
        });
        assert!(past_i16, "no coefficient of the image leaves i16");

        let image = project(&prepare_i16_witness(&witness), &projection_matrix);
        assert_eq!(image.data, reference.data);
    }

    /// A digit at `i16::MIN` leaves i16 as soon as a row subtracts it.
    #[test]
    fn projection_is_exact_with_an_i16_min_digit() {
        crate::common::init_common();
        let mut rows = vec![vec![0i8; 512]; 256];
        rows[0][0] = -1;
        rows[1][0] = 1;
        rows[1][1] = 1;
        rows[2][0] = -1;
        rows[2][1] = 1;
        let projection_matrix = ProjectionMatrix::from_i8(rows);

        let all = |x: i64| {
            let mut element = RingElement::all(
                x.rem_euclid(MOD_Q as i64) as u64,
                Representation::Coefficients,
            );
            element.to_representation(Representation::IncompleteNTT);
            element
        };
        let mut data = vec![RingElement::zero(Representation::IncompleteNTT); 512];
        data[0] = all(i16::MIN as i64);
        data[1] = all(i16::MAX as i64);
        let witness = VerticallyAlignedMatrix {
            data,
            width: 1,
            height: 512,
            used_cols: 1,
        };

        let image = project(&prepare_i16_witness(&witness), &projection_matrix);
        assert_eq!(image.data, project_ring(&witness, &projection_matrix).data);
        assert_eq!(image.data[0], all(1 << 15));
        assert_eq!(image.data[2], all((1 << 16) - 1));
    }

    #[test]
    fn test_projection() {
        let projection_height = 256;
        let projection_matrix = ProjectionMatrix::from_i8({
            let mut data = vec![vec![0i8; projection_height * 2]; projection_height];
            data[0][0] = -1;
            data[0][1] = -1;
            data[0][2] = 1;
            data[0][3] = 1;

            data[3][1] = -1;
            data[3][2] = 1;
            data[3][6] = 1;
            data[3][7] = 1;

            data
        });

        let mut witness = VerticallyAlignedMatrix {
            data: vec![
                RingElement::constant(1, Representation::IncompleteNTT);
                projection_matrix.projection_height * 8
            ],
            width: 2,
            height: projection_matrix.projection_height * 4,
            used_cols: 2,
        };

        for i in 0..witness.height * witness.width {
            witness.data[i] = RingElement::constant((i + 1) as u64, Representation::IncompleteNTT);
        }
        let witness_i16 = prepare_i16_witness(&mut witness);

        let projection_image = project(&witness_i16, &projection_matrix);

        debug_assert_eq!(
            projection_image[(0, 0)],
            RingElement::constant(
                (-1i64 * 1 + -1i64 * 2 + 1i64 * 3 + 1i64 * 4) as u64,
                Representation::IncompleteNTT
            )
        );
        debug_assert_eq!(
            projection_image[(projection_matrix.projection_height, 0)],
            RingElement::constant(
                (-1 * (projection_matrix.projection_height as i64 * 2 + 1)
                    + -1 * (projection_matrix.projection_height as i64 * 2 + 2)
                    + 1 * (projection_matrix.projection_height as i64 * 2 + 3)
                    + 1 * (projection_matrix.projection_height as i64 * 2 + 4))
                    as u64,
                Representation::IncompleteNTT
            )
        );

        debug_assert_eq!(
            projection_image[(0, 1)],
            RingElement::constant(
                (-1 * (projection_matrix.projection_height as i64 * 4 + 1)
                    + -1 * (projection_matrix.projection_height as i64 * 4 + 2)
                    + 1 * (projection_matrix.projection_height as i64 * 4 + 3)
                    + 1 * (projection_matrix.projection_height as i64 * 4 + 4))
                    as u64,
                Representation::IncompleteNTT
            )
        );

        debug_assert_eq!(
            projection_image[(projection_matrix.projection_height, 1)],
            RingElement::constant(
                (-1 * (projection_matrix.projection_height as i64 * 6 + 1)
                    + -1 * (projection_matrix.projection_height as i64 * 6 + 2)
                    + 1 * (projection_matrix.projection_height as i64 * 6 + 3)
                    + 1 * (projection_matrix.projection_height as i64 * 6 + 4))
                    as u64,
                Representation::IncompleteNTT
            )
        );

        debug_assert_eq!(
            projection_image[(3, 0)],
            RingElement::constant(
                (-1i64 * 2 + 1 * 3 + 1 * 7 + 1 * 8) as u64,
                Representation::IncompleteNTT
            )
        );

        debug_assert_eq!(
            projection_image[(projection_matrix.projection_height + 3, 0)],
            RingElement::constant(
                (-1 * (projection_matrix.projection_height as i64 * 2 + 2)
                    + 1 * (projection_matrix.projection_height as i64 * 2 + 3)
                    + 1 * (projection_matrix.projection_height as i64 * 2 + 7)
                    + 1 * (projection_matrix.projection_height as i64 * 2 + 8))
                    as u64,
                Representation::IncompleteNTT
            )
        );

        debug_assert_eq!(
            projection_image[(3, 1)],
            RingElement::constant(
                (-1 * (projection_matrix.projection_height as i64 * 4 + 2)
                    + 1 * (projection_matrix.projection_height as i64 * 4 + 3)
                    + 1 * (projection_matrix.projection_height as i64 * 4 + 7)
                    + 1 * (projection_matrix.projection_height as i64 * 4 + 8))
                    as u64,
                Representation::IncompleteNTT
            )
        );

        debug_assert_eq!(
            projection_image[(projection_matrix.projection_height + 3, 1)],
            RingElement::constant(
                (-1 * (projection_matrix.projection_height as i64 * 6 + 2)
                    + 1 * (projection_matrix.projection_height as i64 * 6 + 3)
                    + 1 * (projection_matrix.projection_height as i64 * 6 + 7)
                    + 1 * (projection_matrix.projection_height as i64 * 6 + 8))
                    as u64,
                Representation::IncompleteNTT
            )
        );
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    fn extreme_rows(width: usize) -> Vec<Vec<i8>> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let mut rows: Vec<Vec<i8>> = (0..256)
            .map(|_| (0..width).map(|_| rng.random_range(-1..=1)).collect())
            .collect();
        rows[0].fill(1);
        rows[1].fill(-1);
        rows[2] = (0..width)
            .map(|i| if i < width / 2 { 1 } else { -1 })
            .collect();
        rows[3].fill(0);
        rows[4] = (0..width).map(|i| 1 - 2 * (i & 1) as i8).collect();
        rows[5] = (0..width).map(|i| (i >= width - 16) as i8).collect();
        rows
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    fn extreme_digits(len: usize, cells: usize) -> Vec<(&'static str, Vec<i16>)> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(13);
        let n = len * cells;
        let last = |x: i16| {
            let mut digits = vec![2047i16; n];
            digits[n - 1] = x;
            digits
        };
        vec![
            ("all 2047", vec![2047; n]),
            ("all -2047", vec![-2047; n]),
            ("all -2048", vec![-2048; n]),
            ("2047 then 2048", last(2048)),
            ("2047 then i16::MIN", last(i16::MIN)),
            (
                "ones then 2047",
                (0..n).map(|k| if k < n / 2 { 1 } else { 2047 }).collect(),
            ),
            (
                "blocks of 1000 and 2047",
                (0..n)
                    .map(|k| {
                        if (k / (16 * cells)) & 1 == 0 {
                            1000
                        } else {
                            2047
                        }
                    })
                    .collect(),
            ),
            (
                "alternating 2047",
                (0..n)
                    .map(|k| if (k / cells) & 1 == 0 { 2047 } else { -2047 })
                    .collect(),
            ),
            (
                "random 2047",
                (0..n).map(|_| rng.random_range(-2047..=2047)).collect(),
            ),
            (
                "random 2048",
                (0..n).map(|_| rng.random_range(-2048..=2048)).collect(),
            ),
            (
                "random i16",
                (0..n)
                    .map(|_| rng.random_range(-i16::MAX..=i16::MAX))
                    .collect(),
            ),
        ]
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[test]
    fn grouped_rows_match_the_tiled_kernel_and_the_definition_on_extreme_digits() {
        use crate::common::arithmetic::{project_rows_grouped, project_rows_sparse_tiled};
        crate::common::init_common();
        let width = 256 * 32;
        let rows = extreme_rows(width);
        let projection_matrix = ProjectionMatrix::from_i8(rows.clone());
        let (pos, pos_bounds, neg, neg_bounds) = signed_offset_lists(&projection_matrix);
        let offsets = ternary_group_offsets(&projection_matrix);
        for (name, digits) in extreme_digits(width, DEGREE) {
            let witness: Vec<Signed16RingElement> = digits
                .chunks(DEGREE)
                .map(|c| Signed16RingElement(c.try_into().unwrap()))
                .collect();
            let zero = RingElement::zero(Representation::StridedCoefficients);
            let mut grouped = vec![zero.clone(); rows.len()];
            let mut tiled = vec![zero; rows.len()];
            project_rows_grouped::<DEGREE>(
                &witness,
                &offsets,
                (&pos, &pos_bounds, &neg, &neg_bounds),
                &mut grouped,
            );
            project_rows_sparse_tiled::<DEGREE>(
                &witness,
                &pos,
                &pos_bounds,
                &neg,
                &neg_bounds,
                &mut tiled,
            );
            assert!(grouped == tiled, "{name}");
            for (row, entries) in rows.iter().enumerate().take(8) {
                for t in 0..DEGREE {
                    let sum: i64 = entries
                        .iter()
                        .zip(&witness)
                        .map(|(&j, w)| j as i64 * w.0[t] as i64)
                        .sum();
                    assert_eq!(
                        grouped[row].v[t],
                        sum.rem_euclid(MOD_Q as i64) as u64,
                        "{name}, row {row}"
                    );
                }
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[test]
    fn grouped_lanes_match_the_definition_on_extreme_digits_or_decline() {
        use crate::common::arithmetic::project_lanes_grouped;
        use std::arch::x86_64::__m512i;
        crate::common::init_common();
        let width = 256 * 32;
        let rows = extreme_rows(width);
        let projection_matrix = ProjectionMatrix::from_i8(rows.clone());
        let offsets = ternary_group_offsets(&projection_matrix);
        for (name, digits) in extreme_digits(width, 32) {
            let lanes: Vec<__m512i> = digits
                .chunks(32)
                .map(|c| unsafe {
                    std::mem::transmute::<[i16; 32], __m512i>(c.try_into().unwrap())
                })
                .collect();
            let sums = project_lanes_grouped(&lanes, &offsets, rows.len());
            let fits = digits.iter().all(|&x| x.unsigned_abs() < 2048);
            assert_eq!(sums.is_some(), fits, "{name}");
            let Some(sums) = sums else {
                continue;
            };
            let sums: Vec<i32> = sums
                .into_iter()
                .flat_map(|x| unsafe { std::mem::transmute::<__m512i, [i32; 16]>(x) })
                .collect();
            for (row, entries) in rows.iter().enumerate() {
                for c in 0..32 {
                    let sum: i64 = entries
                        .iter()
                        .zip(digits.chunks(32))
                        .map(|(&j, lane)| j as i64 * lane[c] as i64)
                        .sum();
                    assert_eq!(
                        sums[32 * row + c] as i64,
                        sum,
                        "{name}, row {row}, lane {c}"
                    );
                }
            }
        }
    }
}
