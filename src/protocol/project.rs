use crate::common::{
    config::PROJECTION_HEIGHT,
    matrix::{VerticallyAlignedMatrix, ZeroNew},
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
//
// Maybe create the same variant for 32 bit and 64 bit too. Add a helper to choose the right one based on the l-inf norm of the witness.

pub fn project(
    witness: &VerticallyAlignedMatrix<RingElement>,
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

    for col in 0..witness.width {
        for rows_chunk in 0..projection_image.height / PROJECTION_HEIGHT {
            let subwitness = witness.col_slice(
                col,
                rows_chunk * projection_matrix.projection_ratio * PROJECTION_HEIGHT,
                (rows_chunk + 1) * projection_matrix.projection_ratio * PROJECTION_HEIGHT,
            );
            let projection_subimage = projection_image.col_slice_mut(
                col,
                rows_chunk * PROJECTION_HEIGHT,
                (rows_chunk + 1) * PROJECTION_HEIGHT,
            );
            for inner_row in 0..PROJECTION_HEIGHT {
                // compute inner-product
                for i in 0..projection_matrix.projection_ratio * PROJECTION_HEIGHT {
                    let (is_positive, is_non_zero) = &projection_matrix[(inner_row, i)];
                    if !*is_non_zero {
                        continue;
                    }
                    if *is_positive {
                        projection_subimage[inner_row] += &subwitness[i];
                    } else {
                        projection_subimage[inner_row] -= &subwitness[i];
                    }
                }
            }
        }
    }
    projection_image
}

#[test]
fn test_projection() {
    let mut witness = VerticallyAlignedMatrix {
        data: vec![RingElement::constant(1, Representation::IncompleteNTT); PROJECTION_HEIGHT * 8],
        width: 2,
        height: PROJECTION_HEIGHT * 4,
    };

    for i in 0..witness.height * witness.width {
        witness.data[i] = RingElement::constant((i + 1) as u64, Representation::IncompleteNTT);
    }

    let projection_matrix = ProjectionMatrix::from_i8({
        let mut data = vec![vec![0i8; PROJECTION_HEIGHT * 2]; PROJECTION_HEIGHT];
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

    let projection_image = project(&witness, &projection_matrix);

    assert_eq!(
        projection_image[(0, 0)],
        RingElement::constant(
            (-1 * 1 + -1 * 2 + 1 * 3 + 1 * 4) as u64,
            Representation::IncompleteNTT
        )
    );
    assert_eq!(
        projection_image[(PROJECTION_HEIGHT, 0)],
        RingElement::constant(
            (-1 * (PROJECTION_HEIGHT as i64 * 2 + 1)
                + -1 * (PROJECTION_HEIGHT as i64 * 2 + 2)
                + 1 * (PROJECTION_HEIGHT as i64 * 2 + 3)
                + 1 * (PROJECTION_HEIGHT as i64 * 2 + 4)) as u64,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        projection_image[(0, 1)],
        RingElement::constant(
            (-1 * (PROJECTION_HEIGHT as i64 * 4 + 1)
                + -1 * (PROJECTION_HEIGHT as i64 * 4 + 2)
                + 1 * (PROJECTION_HEIGHT as i64 * 4 + 3)
                + 1 * (PROJECTION_HEIGHT as i64 * 4 + 4)) as u64,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        projection_image[(PROJECTION_HEIGHT, 1)],
        RingElement::constant(
            (-1 * (PROJECTION_HEIGHT as i64 * 6 + 1)
                + -1 * (PROJECTION_HEIGHT as i64 * 6 + 2)
                + 1 * (PROJECTION_HEIGHT as i64 * 6 + 3)
                + 1 * (PROJECTION_HEIGHT as i64 * 6 + 4)) as u64,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        projection_image[(3, 0)],
        RingElement::constant(
            (-1 * 2 + 1 * 3 + 1 * 7 + 1 * 8) as u64,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        projection_image[(PROJECTION_HEIGHT + 3, 0)],
        RingElement::constant(
            (-1 * (PROJECTION_HEIGHT as i64 * 2 + 2)
                + 1 * (PROJECTION_HEIGHT as i64 * 2 + 3)
                + 1 * (PROJECTION_HEIGHT as i64 * 2 + 7)
                + 1 * (PROJECTION_HEIGHT as i64 * 2 + 8)) as u64,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        projection_image[(3, 1)],
        RingElement::constant(
            (-1 * (PROJECTION_HEIGHT as i64 * 4 + 2)
                + 1 * (PROJECTION_HEIGHT as i64 * 4 + 3)
                + 1 * (PROJECTION_HEIGHT as i64 * 4 + 7)
                + 1 * (PROJECTION_HEIGHT as i64 * 4 + 8)) as u64,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        projection_image[(PROJECTION_HEIGHT + 3, 1)],
        RingElement::constant(
            (-1 * (PROJECTION_HEIGHT as i64 * 6 + 2)
                + 1 * (PROJECTION_HEIGHT as i64 * 6 + 3)
                + 1 * (PROJECTION_HEIGHT as i64 * 6 + 7)
                + 1 * (PROJECTION_HEIGHT as i64 * 6 + 8)) as u64,
            Representation::IncompleteNTT
        )
    );
}
