use std::ops::IndexMut;

use crate::{
    common::{
        ring_arithmetic::{Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
        vertically_aligned_matrix::VerticallyAlignedMatrix,
    },
    subroutines::crs::CRS,
};

struct Commitment {
    // TODO: add recursive layers of commitments
    pub(crate) commitment: VerticallyAlignedMatrix<RingElement>,
}

pub fn commit(crs: &CRS, witness: &VerticallyAlignedMatrix<RingElement>) -> Commitment {
    assert_eq!(crs.ck[0].preprocessed_row.len(), witness.height);
    let mut commitment = VerticallyAlignedMatrix::<RingElement>::new(
        crs.ck[0].preprocessed_row.len(),
        witness.width,
        &RingElement::zero(Representation::IncompleteNTT),
    );

    for (i, row) in crs.ck.iter().enumerate() {
        let mut temp = RingElement::zero(Representation::IncompleteNTT);
        for col in 0..witness.width {
            for (elem, w_elem) in row.preprocessed_row.iter().zip(witness.col(col).iter()) {
                temp *= (elem, w_elem);
                *commitment.index_mut((i, col)) += &temp;
            }
        }
    }

    Commitment { commitment }
}

#[test]
fn test_commitment_computation() {
    let crs = CRS {
        ck: vec![
            PreprocessedRow {
                preprocessed_row: vec![
                    RingElement::constant(1, Representation::IncompleteNTT),
                    RingElement::constant(2, Representation::IncompleteNTT),
                    RingElement::constant(4, Representation::IncompleteNTT),
                    RingElement::constant(8, Representation::IncompleteNTT),
                    RingElement::constant(16, Representation::IncompleteNTT),
                    RingElement::constant(32, Representation::IncompleteNTT),
                    RingElement::constant(64, Representation::IncompleteNTT),
                    RingElement::constant(128, Representation::IncompleteNTT),
                ],
                structured_row: StructuredRow {
                    tensor_layers: vec![], // incorrect but not used here
                },
            },
            PreprocessedRow {
                preprocessed_row: vec![
                    RingElement::constant(1, Representation::IncompleteNTT),
                    RingElement::constant(4, Representation::IncompleteNTT),
                    RingElement::constant(16, Representation::IncompleteNTT),
                    RingElement::constant(64, Representation::IncompleteNTT),
                    RingElement::constant(256, Representation::IncompleteNTT),
                    RingElement::constant(1024, Representation::IncompleteNTT),
                    RingElement::constant(4096, Representation::IncompleteNTT),
                    RingElement::constant(16384, Representation::IncompleteNTT),
                ],
                structured_row: StructuredRow {
                    tensor_layers: vec![], // incorrect but not used here
                },
            },
        ],
    };

    let witness = VerticallyAlignedMatrix {
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
        width: 2,
        height: 8,
    };

    let commitment = commit(&crs, &witness);

    assert_eq!(
        commitment.commitment.get(0, 0).unwrap(),
        &RingElement::constant(
            1 * 1 + 2 * 2 + 4 * 3 + 8 * 4 + 16 * 5 + 32 * 6 + 64 * 7 + 128 * 8,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        commitment.commitment.get(0, 1).unwrap(),
        &RingElement::constant(
            1 * 9 + 2 * 10 + 4 * 11 + 8 * 12 + 16 * 13 + 32 * 14 + 64 * 15 + 128 * 16,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        commitment.commitment.get(1, 0).unwrap(),
        &RingElement::constant(
            1 * 1 + 4 * 2 + 16 * 3 + 64 * 4 + 256 * 5 + 1024 * 6 + 4096 * 7 + 16384 * 8,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        commitment.commitment.get(1, 1).unwrap(),
        &RingElement::constant(
            1 * 9 + 4 * 10 + 16 * 11 + 64 * 12 + 256 * 13 + 1024 * 14 + 4096 * 15 + 16384 * 16,
            Representation::IncompleteNTT
        )
    );
}
