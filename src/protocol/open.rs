use crate::{
    common::{
        matrix::VerticallyAlignedMatrix,
        ring_arithmetic::{Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    protocol::commitment::{commit_basic_internal, BasicCommitment},
};

pub struct Opening {
    pub rhs: BasicCommitment,
    pub evaluation_points_inner: Vec<PreprocessedRow>,
    pub evaluation_points_outer: Vec<PreprocessedRow>,
}

pub fn evaluation_point_to_structured_row(evaluation_point: &[RingElement]) -> StructuredRow {
    StructuredRow {
        tensor_layers: evaluation_point.to_vec(),
    }
}

pub fn evaluation_point_to_structured_row_conjugate(
    evaluation_point: &[RingElement],
) -> StructuredRow {
    StructuredRow {
        tensor_layers: evaluation_point.iter().map(|x| x.conjugate()).collect(),
    }
}

pub fn open_at(
    witness: &VerticallyAlignedMatrix<RingElement>,
    structured_points_inner: &[StructuredRow],
    structured_points_outer: &[StructuredRow],
) -> Opening {
    debug_assert_eq!(
        structured_points_inner[0].tensor_layers.len(),
        witness.height.ilog2() as usize
    );

    let nof_evaluation_points = structured_points_inner.len();

    let preprocessed_points_inner = structured_points_inner
        .into_iter()
        .map(|sr| PreprocessedRow::from_structured_row(&sr))
        .collect::<Vec<PreprocessedRow>>();

    let preprocessed_points_outer = structured_points_outer
        .into_iter()
        .map(|sr| PreprocessedRow::from_structured_row(&sr))
        .collect::<Vec<PreprocessedRow>>();

    // it's not a commitment, but we can reuse the same structure
    let rhs = commit_basic_internal(&preprocessed_points_inner, witness, nof_evaluation_points);

    Opening {
        rhs,                                                // Y
        evaluation_points_inner: preprocessed_points_inner, // we keep it here as well for convenience so we don't have to prerocess again later
        evaluation_points_outer: preprocessed_points_outer,
    }
}

pub fn claim(
    witness: &VerticallyAlignedMatrix<RingElement>,
    evaluation_point_inner: &StructuredRow,
    evaluation_point_outer: &StructuredRow,
) -> RingElement {
    let preprocessed_row_inner = PreprocessedRow::from_structured_row(evaluation_point_inner);
    let preprocessed_row_outer = PreprocessedRow::from_structured_row(evaluation_point_outer);
    let rhs = commit_basic_internal(&vec![preprocessed_row_inner], witness, 1);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::config::MOD_Q;

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
            used_cols: 4,
        };

        let inner_evaluation_points = vec![
            evaluation_point_to_structured_row(&vec![
                RingElement::constant(17, Representation::IncompleteNTT),
                RingElement::constant(18, Representation::IncompleteNTT),
            ]),
            evaluation_point_to_structured_row(&vec![
                RingElement::constant(19, Representation::IncompleteNTT),
                RingElement::constant(20, Representation::IncompleteNTT),
            ]),
        ];

        let outer_evaluation_points = vec![
            evaluation_point_to_structured_row(&vec![
                RingElement::constant(21, Representation::IncompleteNTT),
                RingElement::constant(22, Representation::IncompleteNTT),
            ]),
            evaluation_point_to_structured_row(&vec![
                RingElement::constant(23, Representation::IncompleteNTT),
                RingElement::constant(24, Representation::IncompleteNTT),
            ]),
        ];

        let opening = open_at(&witness, &inner_evaluation_points, &outer_evaluation_points);

        // debug_assert_eq!(opening.evaluations.len(), 2);

        debug_assert_eq!(
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
        debug_assert_eq!(
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
        debug_assert_eq!(
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
        debug_assert_eq!(
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
        debug_assert_eq!(
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
        debug_assert_eq!(
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
        debug_assert_eq!(
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
        debug_assert_eq!(
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

        // debug_assert_eq!(
        //     opening.evaluations[0],
        //     RingElement::constant(
        //         (MOD_Q as i64
        //             + ((1 - 21) * (1 - 22) * opening.rhs[(0, 0)].v[0] as i64
        //                 + (1 - 21) * (22) * opening.rhs[(0, 1)].v[0] as i64
        //                 + (21) * (1 - 22) * opening.rhs[(0, 2)].v[0] as i64
        //                 + (21) * (22) * opening.rhs[(0, 3)].v[0] as i64)) as u64
        //             % MOD_Q,
        //         Representation::IncompleteNTT
        //     )
        // );

        // debug_assert_eq!(
        //     opening.evaluations[1],
        //     RingElement::constant(
        //         (MOD_Q as i64
        //             + ((1 - 23) * (1 - 24) * opening.rhs[(1, 0)].v[0] as i64
        //                 + (1 - 23) * (24) * opening.rhs[(1, 1)].v[0] as i64
        //                 + (23) * (1 - 24) * opening.rhs[(1, 2)].v[0] as i64
        //                 + (23) * (24) * opening.rhs[(1, 3)].v[0] as i64)) as u64
        //             % MOD_Q,
        //         Representation::IncompleteNTT
        //     )
        // );
    }
}
