use std::ops::IndexMut;

use crate::{
    common::{
        matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix, ZeroNew},
        ring_arithmetic::{Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    protocol::{
        commitment,
        crs::{CK, CRS},
    },
};

pub struct LevelCommitmentDescriptor {
    pub(crate) decomposition_radix: usize,
    pub(crate) decomposition_chunks: usize,
}

pub struct LevelCommitmentWrapper {
    pub(crate) commitment: Commitment,
    pub(crate) decomposition_radix: usize,
    pub(crate) decomposition_chunks: usize,
}

pub struct Commitment {
    pub(crate) commitment: VerticallyAlignedMatrix<RingElement>,
    pub(crate) recursion: Option<Box<LevelCommitmentWrapper>>,
}

pub fn init_prover_commitment(height: usize, width: usize) -> Commitment {
    Commitment {
        // TODO: think/check which alignment is more efficient
        commitment: VerticallyAlignedMatrix::new_zero_preallocated(height, width),
        recursion: None,
    }
}

pub fn commit(
    ck: &CK,
    witness: &VerticallyAlignedMatrix<RingElement>,
    descriptors: &[LevelCommitmentDescriptor],
) -> Commitment {
    commit_internal(ck, witness, descriptors, 0)
}

// TODO: allow commitment to the prefix of the CK
pub fn commit_internal(
    ck: &CK,
    witness: &VerticallyAlignedMatrix<RingElement>,
    descriptors: &[LevelCommitmentDescriptor],
    current_level: usize,
) -> Commitment {
    let decomposed_witness_width = if current_level < descriptors.len() {
        descriptors[current_level].decomposition_chunks
    } else {
        1
    } * witness.width;
    let mut commitment = init_prover_commitment(ck.len(), decomposed_witness_width);

    assert_eq!(ck[0].preprocessed_row.len(), witness.height);

    for (i, row) in ck.iter().enumerate() {
        let mut temp = RingElement::zero(Representation::IncompleteNTT);
        for col in 0..witness.width {
            for (elem, w_elem) in row.preprocessed_row.iter().zip(witness.col(col).iter()) {
                temp *= (elem, w_elem);
                *commitment.commitment.index_mut((i, col)) += &temp;
            }
        }
    }
    commitment
}


// this is first level commit for FW = Y
pub fn commit_basic(
    crs: &CRS,
    witness: &VerticallyAlignedMatrix<RingElement>,
) -> Commitment {
    let ck = crs.ck_for_wit_dim(witness.height);
    let mut commitment = init_prover_commitment(ck.len(), witness.width);

    for (i, row) in ck.iter().enumerate() {
        for col in 0..witness.width {
            let mut temp = RingElement::zero(Representation::IncompleteNTT);
            for (elem, w_elem) in row.preprocessed_row.iter().zip(witness.col(col).iter()) {
                temp *= (elem, w_elem);
                *commitment.commitment.index_mut((i, col)) += &temp;
            }
        }
    }
    commitment
}

#[test]
fn test_commitment_computation() {
    let ck: CK = vec![
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
    ];

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

    let commitment = commit(&ck, &witness, &[]);

    assert_eq!(
        &commitment.commitment[(0, 0)],
        &RingElement::constant(
            1 * 1 + 2 * 2 + 4 * 3 + 8 * 4 + 16 * 5 + 32 * 6 + 64 * 7 + 128 * 8,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        &commitment.commitment[(0, 1)],
        &RingElement::constant(
            1 * 9 + 2 * 10 + 4 * 11 + 8 * 12 + 16 * 13 + 32 * 14 + 64 * 15 + 128 * 16,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        &commitment.commitment[(1, 0)],
        &RingElement::constant(
            1 * 1 + 4 * 2 + 16 * 3 + 64 * 4 + 256 * 5 + 1024 * 6 + 4096 * 7 + 16384 * 8,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        &commitment.commitment[(1, 1)],
        &RingElement::constant(
            1 * 9 + 4 * 10 + 16 * 11 + 64 * 12 + 256 * 13 + 1024 * 14 + 4096 * 15 + 16384 * 16,
            Representation::IncompleteNTT
        )
    );
}
