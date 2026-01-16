use crate::common::{
    matrix::VerticallyAlignedMatrix,
    arithmetic::{centered_coeffs_u64_to_i64_inplace, pack_i64_to_i16_deg16, project_one_row_i16_to_u64},
    config::{DEGREE, MOD_Q},
    projection_matrix::ProjectionMatrix,
    ring_arithmetic::{Representation, RingElement},
};

// TODO: this projection is very naive and unoptimized
// Some idea:
// (i) Convert witness into EvenOdd rep
// (ii) Convert witness into i64 using e.g.
//     // neg lanes are the big ones (Q - t)
//    __mmask8 neg = _mm512_cmpgt_epu64_mask(a, halfQ);
//
//   // if neg: a = a - Q  (Q - t - Q = -t). else keep a = t
//   __m512i signed64 = _mm512_mask_sub_epi64(a, neg, a, vQ);

// (iii) _mm512_cvtsepi64_epi16 to convert i64 to i16 to 16 bits
// (steps (i) to (iii) can be preprocessed during commitment computation so it doesn't have to be done during opening)
// (iv) Compute the output rows in chunks of 32 (since __m512i holds 32 i16 values) with _mm512_add_epi16 and _mm512_sub_epi16
// (v) _mm512_cvtusepi16_epi64 to convert i16 back to u64
// (vi) Convert back to RingElement in IncompleteNTT representation
//
// Maybe create the same variant for 32 bit and 64 bit too. Add a helper to choose the right one based on the l-inf norm of the witness.

// V = (I \otimes J)W

// w \in [0, MOD_Q)
// w \in [-MOD_Q/2, MOD_Q/2) \in i16 NOT i64


#[derive(Clone)]
pub struct RowPattern {
    pub pos: Vec<u16>,
    pub neg: Vec<u16>,
}

pub struct ProjectionPlan {
    pub projection_ratio: usize,
    pub rows: Vec<RowPattern>
}

pub fn build_plan(pm: &ProjectionMatrix) -> ProjectionPlan {
    let row_len = pm.projection_ratio * pm.projection_height;


    let rows: Vec<RowPattern> = (0..pm.projection_height)
    .map(|inner_row| {
        let mut pos = Vec::<u16>::new();
        let mut neg = Vec::<u16>::new();

        for i in 0..row_len {
            let (is_positive, is_non_zero) = pm[(inner_row, i)];
            if !is_non_zero {
                continue;
            }
            if is_positive {
                pos.push(i as u16);
            } else {
                neg.push(i as u16);
            }
        }

        RowPattern { pos, neg }
    })
    .collect();

    ProjectionPlan {
        projection_ratio: pm.projection_ratio,
        rows,
    }
}

pub fn project(
    witness: &mut VerticallyAlignedMatrix<RingElement>,
    projection_matrix: &ProjectionMatrix,
) -> VerticallyAlignedMatrix<RingElement> {
    let mut projection_image = VerticallyAlignedMatrix::new_zero_preallocated(
        witness.height / projection_matrix.projection_ratio,
        witness.width,
    );
    assert_eq!(projection_image.width, witness.width);

    assert_eq!(
        projection_image.height * projection_matrix.projection_ratio,
        witness.height
    );
    let mut witness_i64 = Vec::<[i64; DEGREE]>::new();


    // TODO: move this steps to commit phase
    for i in projection_image.data.iter_mut() {
        i.from_incomplete_ntt_to_even_odd_coefficients();
    }

    for (i, cr) in witness.data.iter_mut().enumerate() {
        let mut ring_el = [0 as i64; DEGREE];
        cr.from_incomplete_ntt_to_even_odd_coefficients();
        centered_coeffs_u64_to_i64_inplace(&mut ring_el, &cr.v, MOD_Q);
        witness_i64.push(ring_el);
        cr.from_even_odd_coefficients_to_incomplete_ntt_representation();
    }

    let mut witness_i16: Vec<[i16; DEGREE]> = vec![[0i16; DEGREE]; witness_i64.len()];

    for (dst, src) in witness_i16.iter_mut().zip(witness_i64.iter()) {
        unsafe {
            pack_i64_to_i16_deg16(dst, src);
        }
    }

    // build a list of positive and negative rows from projection matrix
    let plan = build_plan(projection_matrix);

    // Create a matrix view for col_slice access (adapt this to your real type)
    let witness_i16_mat = VerticallyAlignedMatrix::<[i16; DEGREE]> {
        width: witness.width,
        height: witness.height,
        data: witness_i16,
    };

    for col in 0..witness_i16_mat.width {
        for rows_chunk in 0..projection_image.height / projection_matrix.projection_height {
            let subwitness_i16 = witness_i16_mat.col_slice(
                col,
                rows_chunk * plan.projection_ratio * projection_matrix.projection_height,
                (rows_chunk + 1) * plan.projection_ratio * projection_matrix.projection_height,
            );

            let projection_subimage = projection_image.col_slice_mut(
                col,
                rows_chunk * projection_matrix.projection_height,
                (rows_chunk + 1) * projection_matrix.projection_height,
            );

            for inner_row in 0..projection_matrix.projection_height {
                let mut out_u64 = [0u64; DEGREE];

                unsafe {
                    project_one_row_i16_to_u64::<DEGREE>(
                        subwitness_i16,
                        &plan.rows[inner_row].pos,
                        &plan.rows[inner_row].neg,
                        MOD_Q,
                        &mut out_u64,
                    );
                }

                projection_subimage[inner_row].v.copy_from_slice(&out_u64);
            }
        }
    }

    for i in projection_image.data.iter_mut() {
        i.from_even_odd_coefficients_to_incomplete_ntt_representation();
    }

    projection_image
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

    let projection_image = project(&mut witness, &projection_matrix);

    assert_eq!(
        projection_image[(0, 0)],
        RingElement::constant(
            (-1i64 * 1 + -1i64 * 2 + 1i64 * 3 + 1i64 * 4) as u64,
            Representation::IncompleteNTT
        )
    );
    assert_eq!(
        projection_image[(projection_matrix.projection_height, 0)],
        RingElement::constant(
            (-1 * (projection_matrix.projection_height as i64 * 2 + 1)
                + -1 * (projection_matrix.projection_height as i64 * 2 + 2)
                + 1 * (projection_matrix.projection_height as i64 * 2 + 3)
                + 1 * (projection_matrix.projection_height as i64 * 2 + 4)) as u64,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        projection_image[(0, 1)],
        RingElement::constant(
            (-1 * (projection_matrix.projection_height as i64 * 4 + 1)
                + -1 * (projection_matrix.projection_height as i64 * 4 + 2)
                + 1 * (projection_matrix.projection_height as i64 * 4 + 3)
                + 1 * (projection_matrix.projection_height as i64 * 4 + 4)) as u64,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        projection_image[(projection_matrix.projection_height, 1)],
        RingElement::constant(
            (-1 * (projection_matrix.projection_height as i64 * 6 + 1)
                + -1 * (projection_matrix.projection_height as i64 * 6 + 2)
                + 1 * (projection_matrix.projection_height as i64 * 6 + 3)
                + 1 * (projection_matrix.projection_height as i64 * 6 + 4)) as u64,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        projection_image[(3, 0)],
        RingElement::constant(
            (-1i64 * 2 + 1 * 3 + 1 * 7 + 1 * 8) as u64,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        projection_image[(projection_matrix.projection_height + 3, 0)],
        RingElement::constant(
            (-1 * (projection_matrix.projection_height as i64 * 2 + 2)
                + 1 * (projection_matrix.projection_height as i64 * 2 + 3)
                + 1 * (projection_matrix.projection_height as i64 * 2 + 7)
                + 1 * (projection_matrix.projection_height as i64 * 2 + 8)) as u64,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        projection_image[(3, 1)],
        RingElement::constant(
            (-1 * (projection_matrix.projection_height as i64 * 4 + 2)
                + 1 * (projection_matrix.projection_height as i64 * 4 + 3)
                + 1 * (projection_matrix.projection_height as i64 * 4 + 7)
                + 1 * (projection_matrix.projection_height as i64 * 4 + 8)) as u64,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        projection_image[(projection_matrix.projection_height + 3, 1)],
        RingElement::constant(
            (-1 * (projection_matrix.projection_height as i64 * 6 + 2)
                + 1 * (projection_matrix.projection_height as i64 * 6 + 3)
                + 1 * (projection_matrix.projection_height as i64 * 6 + 7)
                + 1 * (projection_matrix.projection_height as i64 * 6 + 8)) as u64,
            Representation::IncompleteNTT
        )
    );
}
