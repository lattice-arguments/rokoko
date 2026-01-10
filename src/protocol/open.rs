use std::ops::IndexMut;

use crate::{
    common::{
        config::MOD_Q,
        matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix, ZeroNew},
        ring_arithmetic::{Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    protocol::commitment::{commit_basic_internal, BasicCommitment},
};

pub struct Opening {
    pub rhs: BasicCommitment,
    pub evaluations: Vec<RingElement>, // TODO: those are already public?
    pub evaluation_points_inner: Vec<PreprocessedRow>, // TODO: those are unnecessarily stored like this?
    pub evaluation_points_outer: Vec<PreprocessedRow>, // TODO: those are unnecessarily stored like this?
}

pub fn evaluation_point_to_structured_row(evaluation_point: &Vec<RingElement>) -> StructuredRow {
    StructuredRow {
        tensor_layers: evaluation_point.clone(),
    }
}

pub fn open_at(
    witness: &VerticallyAlignedMatrix<RingElement>,
    evaluation_points_inner: &Vec<Vec<RingElement>>,
    evaluation_points_outer: &Vec<Vec<RingElement>>,
) -> Opening {
    assert_eq!(
        evaluation_points_inner[0].len(),
        witness.height.ilog2() as usize
    );

    let nof_evaluation_points = evaluation_points_inner.len();

    let structured_points_inner = evaluation_points_inner
        .iter()
        .map(evaluation_point_to_structured_row)
        .collect::<Vec<StructuredRow>>();

    let preprocessed_points_inner = structured_points_inner
        .into_iter()
        .map(|sr| PreprocessedRow::from_structured_row(&sr))
        .collect::<Vec<PreprocessedRow>>();

    let structured_points_outer = evaluation_points_outer
        .iter()
        .map(evaluation_point_to_structured_row)
        .collect::<Vec<StructuredRow>>();

    let preprocessed_points_outer = structured_points_outer
        .into_iter()
        .map(|sr| PreprocessedRow::from_structured_row(&sr))
        .collect::<Vec<PreprocessedRow>>();

    // for (i, preprocessed_row_inner) in preprocessed_points_inner.iter().enumerate() {
    //     let mut temp = RingElement::zero(Representation::IncompleteNTT);
    //     for col in 0..witness.width {
    //         for (elem, w_elem) in preprocessed_row_inner
    //             .preprocessed_row
    //             .iter()
    //             .zip(witness.col(col).iter())
    //         {
    //             temp *= (elem, w_elem);
    //             *rhs.index_mut((i, col)) += &temp;
    //         }
    //     }
    // }

    // it's not a commitment, but we can reuse the same structure
    let mut rhs = commit_basic_internal(&preprocessed_points_inner, witness, nof_evaluation_points);

    let mut evaluations =
        vec![RingElement::zero(Representation::IncompleteNTT); nof_evaluation_points];

    for (i, preprocessed_row_outer) in preprocessed_points_outer.iter().enumerate() {
        let mut temp = RingElement::zero(Representation::IncompleteNTT);
        for col in 0..rhs.width {
            temp *= (
                &rhs[(i, col)],
                &preprocessed_row_outer.preprocessed_row[col],
            );
            evaluations[i] += &temp;
        }
    }

    Opening {
        rhs,
        evaluations,
        evaluation_points_inner: preprocessed_points_inner,
        evaluation_points_outer: preprocessed_points_outer,
    }
}

fn claim(
    witness: &VerticallyAlignedMatrix<RingElement>,
    evaluation_point_inner: &StructuredRow,
    evaluation_point_outer: &StructuredRow,
) -> RingElement {
    let preprocessed_row_inner = PreprocessedRow::from_structured_row(evaluation_point_inner);
    let preprocessed_row_outer = PreprocessedRow::from_structured_row(evaluation_point_outer);
    let mut rhs = commit_basic_internal(&vec![preprocessed_row_inner], witness, 1);
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    let mut result = RingElement::zero(Representation::IncompleteNTT);
    for col in 0..rhs.width {
        temp *= (
            &rhs[(0, col)],
            &preprocessed_row_outer.preprocessed_row[col],
        );
        result += &temp;
    }
    result
}

#[test]
fn test_opening() {
    let witness = VerticallyAlignedMatrix {
        height: 4,
        width: 4,
        data: vec![
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(2, Representation::IncompleteNTT),
            RingElement::constant(3, Representation::IncompleteNTT),
            RingElement::constant(4, Representation::IncompleteNTT),
            RingElement::constant(5, Representation::IncompleteNTT),
            RingElement::constant(6, Representation::IncompleteNTT),
            RingElement::constant(7, Representation::IncompleteNTT),
            RingElement::constant(8, Representation::IncompleteNTT),
            RingElement::constant(9, Representation::IncompleteNTT),
            RingElement::constant(10, Representation::IncompleteNTT),
            RingElement::constant(11, Representation::IncompleteNTT),
            RingElement::constant(12, Representation::IncompleteNTT),
            RingElement::constant(13, Representation::IncompleteNTT),
            RingElement::constant(14, Representation::IncompleteNTT),
            RingElement::constant(15, Representation::IncompleteNTT),
            RingElement::constant(16, Representation::IncompleteNTT),
        ],
    };

    let inner_evaluation_points = vec![
        vec![
            RingElement::constant(17, Representation::IncompleteNTT),
            RingElement::constant(18, Representation::IncompleteNTT),
        ],
        vec![
            RingElement::constant(19, Representation::IncompleteNTT),
            RingElement::constant(20, Representation::IncompleteNTT),
        ],
    ];

    let outer_evaluation_points = vec![
        vec![
            RingElement::constant(21, Representation::IncompleteNTT),
            RingElement::constant(22, Representation::IncompleteNTT),
        ],
        vec![
            RingElement::constant(23, Representation::IncompleteNTT),
            RingElement::constant(24, Representation::IncompleteNTT),
        ],
    ];

    let opening = open_at(&witness, &inner_evaluation_points, &outer_evaluation_points);

    assert_eq!(opening.evaluations.len(), 2);

    assert_eq!(
        opening.rhs[(0, 0)],
        RingElement::constant(
            (MOD_Q as i64
                + ((1 - 17) * (1 - 18) * 1
                    + (1 - 17) * (18) * 2
                    + (17) * (1 - 18) * 3
                    + (17) * (18) * 4)) as u64
                % MOD_Q,
            Representation::IncompleteNTT
        )
    );
    assert_eq!(
        opening.rhs[(0, 1)],
        RingElement::constant(
            (MOD_Q as i64
                + ((1 - 17) * (1 - 18) * 5
                    + (1 - 17) * (18) * 6
                    + (17) * (1 - 18) * 7
                    + (17) * (18) * 8)) as u64
                % MOD_Q,
            Representation::IncompleteNTT
        )
    );
    assert_eq!(
        opening.rhs[(0, 2)],
        RingElement::constant(
            (MOD_Q as i64
                + ((1 - 17) * (1 - 18) * 9
                    + (1 - 17) * (18) * 10
                    + (17) * (1 - 18) * 11
                    + (17) * (18) * 12)) as u64
                % MOD_Q,
            Representation::IncompleteNTT
        )
    );
    assert_eq!(
        opening.rhs[(0, 3)],
        RingElement::constant(
            (MOD_Q as i64
                + ((1 - 17) * (1 - 18) * 13
                    + (1 - 17) * (18) * 14
                    + (17) * (1 - 18) * 15
                    + (17) * (18) * 16)) as u64
                % MOD_Q,
            Representation::IncompleteNTT
        )
    );
    assert_eq!(
        opening.rhs[(1, 0)],
        RingElement::constant(
            (MOD_Q as i64
                + ((1 - 19) * (1 - 20) * 1
                    + (1 - 19) * (20) * 2
                    + (19) * (1 - 20) * 3
                    + (19) * (20) * 4)) as u64
                % MOD_Q,
            Representation::IncompleteNTT
        )
    );
    assert_eq!(
        opening.rhs[(1, 1)],
        RingElement::constant(
            (MOD_Q as i64
                + ((1 - 19) * (1 - 20) * 5
                    + (1 - 19) * (20) * 6
                    + (19) * (1 - 20) * 7
                    + (19) * (20) * 8)) as u64
                % MOD_Q,
            Representation::IncompleteNTT
        )
    );
    assert_eq!(
        opening.rhs[(1, 2)],
        RingElement::constant(
            (MOD_Q as i64
                + ((1 - 19) * (1 - 20) * 9
                    + (1 - 19) * (20) * 10
                    + (19) * (1 - 20) * 11
                    + (19) * (20) * 12)) as u64
                % MOD_Q,
            Representation::IncompleteNTT
        )
    );
    assert_eq!(
        opening.rhs[(1, 3)],
        RingElement::constant(
            (MOD_Q as i64
                + ((1 - 19) * (1 - 20) * 13
                    + (1 - 19) * (20) * 14
                    + (19) * (1 - 20) * 15
                    + (19) * (20) * 16)) as u64
                % MOD_Q,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        opening.evaluations[0],
        RingElement::constant(
            (MOD_Q as i64
                + ((1 - 21) * (1 - 22) * opening.rhs[(0, 0)].v[0] as i64
                    + (1 - 21) * (22) * opening.rhs[(0, 1)].v[0] as i64
                    + (21) * (1 - 22) * opening.rhs[(0, 2)].v[0] as i64
                    + (21) * (22) * opening.rhs[(0, 3)].v[0] as i64)) as u64
                % MOD_Q,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        opening.evaluations[1],
        RingElement::constant(
            (MOD_Q as i64
                + ((1 - 23) * (1 - 24) * opening.rhs[(1, 0)].v[0] as i64
                    + (1 - 23) * (24) * opening.rhs[(1, 1)].v[0] as i64
                    + (23) * (1 - 24) * opening.rhs[(1, 2)].v[0] as i64
                    + (23) * (24) * opening.rhs[(1, 3)].v[0] as i64)) as u64
                % MOD_Q,
            Representation::IncompleteNTT
        )
    );
}
