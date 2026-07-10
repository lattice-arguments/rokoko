//! Coarse JL projection: out[r] = sum_i J[r,i] * w[i], J ternary (~half zeros).
//! Digits are short in the coefficient domain, so: inverse-NTT + narrow the
//! witness to i16 once, drop J's zeros into per-row +/- index lists, accumulate
//! in i16 over L1-resident tiles, lift back to [0, Q) and forward-NTT the image.

use crate::common::{
    arithmetic::centered_i16_from_u64_mod_q,
    config::{DEGREE, HALF_DEGREE, MOD_Q},
    matrix::VerticallyAlignedMatrix,
    projection_matrix::ProjectionMatrix,
    ring_arithmetic::{Representation, RingElement},
};
use crate::hexl::bindings::ntt_inverse;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use crate::common::arithmetic::project_rows_sparse_tiled;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
use crate::common::arithmetic::project_one_row_i16_to_u64;

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
            debug_assert!(cr.representation == Representation::IncompleteNTT);
            unsafe {
                ntt_inverse(temp.0.as_mut_ptr(), cr.v.as_ptr(), HALF_DEGREE, MOD_Q);
                ntt_inverse(
                    temp.0.as_mut_ptr().add(HALF_DEGREE),
                    cr.v.as_ptr().add(HALF_DEGREE),
                    HALF_DEGREE,
                    MOD_Q,
                );
            }
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
    let mut image = VerticallyAlignedMatrix { data: vec![RingElement::zero(Representation::IncompleteNTT); witness.height / projection_matrix.projection_ratio * witness.width], width: witness.width, height: witness.height / projection_matrix.projection_ratio, used_cols: witness.width };
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
    let mut projection_image = VerticallyAlignedMatrix { data: vec![RingElement::zero(Representation::IncompleteNTT); witness_16.height / projection_matrix.projection_ratio * witness_16.width], width: witness_16.width, height: witness_16.height / projection_matrix.projection_ratio, used_cols: witness_16.width };

    debug_assert_eq!(projection_image.width, witness_16.width);
    debug_assert_eq!(
        projection_image.height * projection_matrix.projection_ratio,
        witness_16.height
    );

    for i in projection_image.data.iter_mut() {
        i.from_incomplete_ntt_to_even_odd_coefficients();
    }

    let row_len = projection_matrix.projection_ratio * projection_matrix.projection_height;

    // Sparse row lists + L1-tiled kernel: each witness chunk line is loaded
    // once per tile pass and reused by all projection rows.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        let (pos, pos_bounds, neg, neg_bounds) = signed_offset_lists(projection_matrix);

        for col in 0..witness_16.used_cols {
            for rows_chunk in 0..projection_image.height / projection_matrix.projection_height {
                let subwitness_i16 =
                    witness_16.col_slice(col, rows_chunk * row_len, (rows_chunk + 1) * row_len);

                let projection_subimage = projection_image.col_slice_mut(
                    col,
                    rows_chunk * projection_matrix.projection_height,
                    (rows_chunk + 1) * projection_matrix.projection_height,
                );

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
        i.from_even_odd_coefficients_to_incomplete_ntt_representation();
    }

    projection_image
}

// Per-sign column lists as flat byte offsets (index * elem size) with per-row
// bounds, filled via compress-store straight from the packed mask bytes.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn signed_offset_lists(
    projection_matrix: &ProjectionMatrix,
) -> (Vec<u32>, Vec<usize>, Vec<u32>, Vec<usize>) {
    let h = projection_matrix.projection_height;
    let elem_bytes = core::mem::size_of::<Signed16RingElement>();

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

    let mut pos: Vec<u32> = Vec::with_capacity(npos);
    let mut neg: Vec<u32> = Vec::with_capacity(nneg);
    unsafe {
        use std::arch::x86_64::{
            _mm256_add_epi32, _mm256_mask_compressstoreu_epi32, _mm256_set1_epi32,
            _mm256_setr_epi32,
        };
        let eb = elem_bytes as i32;
        let lane_offsets = _mm256_setr_epi32(0, eb, 2 * eb, 3 * eb, 4 * eb, 5 * eb, 6 * eb, 7 * eb);
        let mut pw = pos.as_mut_ptr();
        let mut nw = neg.as_mut_ptr();
        for row in 0..h {
            let (pos_row, nz_row) = projection_matrix.row_chunks(row);
            for (c, (&p, &n)) in pos_row.iter().zip(nz_row).enumerate() {
                let offs = _mm256_add_epi32(
                    _mm256_set1_epi32((c * 8) as i32 * eb),
                    lane_offsets,
                );
                let add = p & n;
                _mm256_mask_compressstoreu_epi32(pw as *mut i32, add, offs);
                pw = pw.add(add.count_ones() as usize);
                let sub = !p & n;
                _mm256_mask_compressstoreu_epi32(nw as *mut i32, sub, offs);
                nw = nw.add(sub.count_ones() as usize);
            }
        }
        pos.set_len(npos);
        neg.set_len(nneg);
    }

    (pos, pos_bounds, neg, neg_bounds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::hash::HashWrapper;

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
}
