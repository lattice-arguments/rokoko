use crate::common::{
    arithmetic::{
        centered_coeffs_u64_to_i64_inplace, pack_i64_to_i16_deg16, project_one_row_i16_to_u64,
    },
    config::{DEGREE, MOD_Q},
    matrix::VerticallyAlignedMatrix,
    projection_matrix::ProjectionMatrix,
    ring_arithmetic::{Representation, RingElement},
};

pub type Signed16RingElement = [i16; DEGREE];
pub type Signed64RingElement = [i64; DEGREE];

pub fn prepare_i16_witness(
    witness: &VerticallyAlignedMatrix<RingElement>,
) -> VerticallyAlignedMatrix<Signed16RingElement> {
    let mut witness_i16: Vec<Signed16RingElement> = vec![[0i16; DEGREE]; witness.data.len()];

    let mut ring_el = [0 as i64; DEGREE];
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    for (i, cr) in witness.data.iter().enumerate() {
        temp.set_from(cr);
        temp.from_incomplete_ntt_to_even_odd_coefficients();
        centered_coeffs_u64_to_i64_inplace(&mut ring_el, &temp.v);
        unsafe {
            pack_i64_to_i16_deg16(&mut witness_i16[i], &mut ring_el);
        }
    }

    VerticallyAlignedMatrix::<Signed16RingElement> {
        width: witness.width,
        height: witness.height,
        data: witness_i16,
        used_cols: witness.width,
    }
}


#[derive(Clone)]
pub struct RowPattern {
    pub pos: Vec<u16>,
    pub neg: Vec<u16>,
}

#[derive(Clone)]
pub struct ProjectionPlan {
    pub projection_ratio: usize,
    pub rows: Vec<RowPattern>,
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
    witness_16: &VerticallyAlignedMatrix<Signed16RingElement>,
    projection_matrix: &ProjectionMatrix,
) -> VerticallyAlignedMatrix<RingElement> {
    let mut projection_image = VerticallyAlignedMatrix::new_zero_preallocated(
        witness_16.height / projection_matrix.projection_ratio,
        witness_16.width,
    );
    debug_assert_eq!(projection_image.width, witness_16.width);

    debug_assert_eq!(
        projection_image.height * projection_matrix.projection_ratio,
        witness_16.height
    );

    for i in projection_image.data.iter_mut() {
        i.from_incomplete_ntt_to_even_odd_coefficients();
    }

    let plan = build_plan(projection_matrix);

    for col in 0..witness_16.width {
        for rows_chunk in 0..projection_image.height / projection_matrix.projection_height {
            let subwitness_i16 = witness_16.col_slice(
                col,
                rows_chunk
                    * projection_matrix.projection_ratio
                    * projection_matrix.projection_height,
                (rows_chunk + 1)
                    * projection_matrix.projection_ratio
                    * projection_matrix.projection_height,
            );

            let mut projection_subimage = projection_image.col_slice_mut(
                col,
                rows_chunk * projection_matrix.projection_height,
                (rows_chunk + 1) * projection_matrix.projection_height,
            );

            for inner_row in 0..projection_matrix.projection_height {
                unsafe {
                    project_one_row_i16_to_u64::<DEGREE>(
                        subwitness_i16,
                        &plan.rows[inner_row].pos,
                        &plan.rows[inner_row].neg,
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
                + 1 * (projection_matrix.projection_height as i64 * 2 + 4)) as u64,
            Representation::IncompleteNTT
        )
    );

    debug_assert_eq!(
        projection_image[(0, 1)],
        RingElement::constant(
            (-1 * (projection_matrix.projection_height as i64 * 4 + 1)
                + -1 * (projection_matrix.projection_height as i64 * 4 + 2)
                + 1 * (projection_matrix.projection_height as i64 * 4 + 3)
                + 1 * (projection_matrix.projection_height as i64 * 4 + 4)) as u64,
            Representation::IncompleteNTT
        )
    );

    debug_assert_eq!(
        projection_image[(projection_matrix.projection_height, 1)],
        RingElement::constant(
            (-1 * (projection_matrix.projection_height as i64 * 6 + 1)
                + -1 * (projection_matrix.projection_height as i64 * 6 + 2)
                + 1 * (projection_matrix.projection_height as i64 * 6 + 3)
                + 1 * (projection_matrix.projection_height as i64 * 6 + 4)) as u64,
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
                + 1 * (projection_matrix.projection_height as i64 * 2 + 8)) as u64,
            Representation::IncompleteNTT
        )
    );

    debug_assert_eq!(
        projection_image[(3, 1)],
        RingElement::constant(
            (-1 * (projection_matrix.projection_height as i64 * 4 + 2)
                + 1 * (projection_matrix.projection_height as i64 * 4 + 3)
                + 1 * (projection_matrix.projection_height as i64 * 4 + 7)
                + 1 * (projection_matrix.projection_height as i64 * 4 + 8)) as u64,
            Representation::IncompleteNTT
        )
    );

    debug_assert_eq!(
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
