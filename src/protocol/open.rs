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

fn repeated(row: BasicCommitment, rows: usize) -> BasicCommitment {
    let width = row.width;
    let height = rows.next_power_of_two();
    let mut data = vec![RingElement::zero(Representation::IncompleteNTT); height * width];
    for copy in data.chunks_exact_mut(width).take(rows) {
        copy.clone_from_slice(&row.data[..width]);
    }
    BasicCommitment {
        data,
        width,
        height,
    }
}

pub fn open_at(
    witness: &VerticallyAlignedMatrix<RingElement>,
    structured_points_inner: &[StructuredRow],
    structured_points_outer: &[StructuredRow],
    skip_last_col: bool,
) -> Opening {
    debug_assert!(structured_points_inner
        .iter()
        .all(|p| p.tensor_layers.len() == witness.height.ilog2() as usize));

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
    let mut rhs = match structured_points_inner.split_first() {
        Some((first, rest)) if rest.iter().all(|p| p.tensor_layers == first.tensor_layers) => {
            repeated(
                commit_basic_internal(&preprocessed_points_inner[..1], witness, 1),
                nof_evaluation_points,
            )
        }
        _ => commit_basic_internal(&preprocessed_points_inner, witness, nof_evaluation_points),
    };

    if skip_last_col {
        for i in 0..structured_points_outer.len() {
            rhs[(i, witness.width - 1)].set_zero();
        }
    }

    Opening {
        rhs,                                                // paper: T = matrix-from-rows(l_j) W
        evaluation_points_inner: preprocessed_points_inner, // kept to avoid re-preprocessing
        evaluation_points_outer: preprocessed_points_outer,
    }
}

pub fn opened(
    rhs: BasicCommitment,
    structured_points_inner: &[StructuredRow],
    structured_points_outer: &[StructuredRow],
) -> Opening {
    let preprocessed = |points: &[StructuredRow]| {
        points
            .iter()
            .map(PreprocessedRow::from_structured_row)
            .collect()
    };
    Opening {
        rhs,
        evaluation_points_inner: preprocessed(structured_points_inner),
        evaluation_points_outer: preprocessed(structured_points_outer),
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

pub fn claim_over_inner(
    witness: &VerticallyAlignedMatrix<RingElement>,
    evaluation_point_inner: &StructuredRow,
) -> Vec<RingElement> {
    let preprocessed_row_inner = PreprocessedRow::from_structured_row(evaluation_point_inner);
    let rhs = commit_basic_internal(&vec![preprocessed_row_inner], witness, 1);
    rhs.row(0).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::config::MOD_Q;

    #[test]
    fn coinciding_or_absent_inner_points_open_as_the_general_rows() {
        crate::common::init_common();
        let (height, width) = (8, 4);
        let witness = VerticallyAlignedMatrix {
            height,
            width,
            used_cols: width,
            data: (0..height * width)
                .map(|_| RingElement::random(Representation::IncompleteNTT))
                .collect(),
        };
        let point = |seed: u64| {
            evaluation_point_to_structured_row(
                &(0..3)
                    .map(|i| RingElement::constant(seed + i, Representation::IncompleteNTT))
                    .collect::<Vec<_>>(),
            )
        };
        for inner in [
            vec![],
            vec![point(5)],
            vec![point(5), point(5), point(5)],
            vec![point(5), point(9)],
        ] {
            let opening = open_at(&witness, &inner, &[], false);
            let preprocessed: Vec<PreprocessedRow> = inner
                .iter()
                .map(PreprocessedRow::from_structured_row)
                .collect();
            let direct = commit_basic_internal(&preprocessed, &witness, inner.len());
            assert_eq!(
                (opening.rhs.height, opening.rhs.width),
                (direct.height, direct.width)
            );
            assert_eq!(opening.rhs.data, direct.data, "{} points", inner.len());
        }
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

        let opening = open_at(
            &witness,
            &inner_evaluation_points,
            &outer_evaluation_points,
            false,
        );

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
    }
}
