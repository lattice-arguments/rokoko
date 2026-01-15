use std::ops::IndexMut;

use crate::{
    common::{
        config::{self, MOD_Q},
        decomposition::decompose,
        matrix::{
            new_vec_zero_preallocated, HorizontallyAlignedMatrix, VerticallyAlignedMatrix, ZeroNew,
        },
        ring_arithmetic::{Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    protocol::{
        commitment,
        crs::{CK, CRS},
        open::Opening,
    },
};

pub type BasicCommitment = HorizontallyAlignedMatrix<RingElement>;

pub fn commit_basic_internal(
    ck: &CK,
    witness: &VerticallyAlignedMatrix<RingElement>,
    rank: usize,
) -> BasicCommitment {
    let mut commitment = HorizontallyAlignedMatrix::new_zero_preallocated(rank, witness.width);

    for (i, row) in ck.iter().take(rank).enumerate() {
        for col in 0..witness.width {
            let mut temp = RingElement::zero(Representation::IncompleteNTT);
            for (elem, w_elem) in row.preprocessed_row.iter().zip(witness.col(col).iter()) {
                temp *= (elem, w_elem);
                *commitment.index_mut((i, col)) += &temp;
            }
        }
    }
    commitment
}

// this is first level commit for FW = Y
pub fn commit_basic(
    crs: &CRS,
    witness: &VerticallyAlignedMatrix<RingElement>,
    rank: usize,
) -> BasicCommitment {
    let ck = crs.ck_for_wit_dim(witness.height);
    commit_basic_internal(ck, witness, rank)
}

#[derive(Clone)]
pub struct Prefix {
    pub prefix: usize,
    pub length: usize,
}

#[derive(Clone)]
pub struct RecursionConfig {
    pub decomposition_base_log: usize,
    pub decomposition_chunks: usize,
    pub rank: usize,
    pub prefix: Prefix,
    pub next: Option<Box<RecursionConfig>>,
}

#[derive(Clone)]
pub struct RecursiveCommitment {
    pub decomposition_base_log: usize,
    pub decomposition_chunks: usize,
    pub committed_data: Vec<RingElement>,
    pub commitment: Vec<RingElement>,
    pub next: Option<Box<RecursiveCommitment>>,
}

impl RecursiveCommitment {
    pub fn most_inner_commitment(&self) -> &Vec<RingElement> {
        match &self.next {
            Some(next_config) => next_config.most_inner_commitment(),
            None => &self.commitment,
        }
    }
}

pub fn recursive_commit(
    crs: &CRS,
    config: &RecursionConfig,
    data: &Vec<RingElement>,
) -> RecursiveCommitment {
    let committed_data = decompose(
        &data,
        config.decomposition_base_log as u64,
        config.decomposition_chunks,
    );

    let ck = crs.ck_for_wit_dim(committed_data.len());

    let mut commitment = new_vec_zero_preallocated(config.rank);

    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    for r in 0..config.rank {
        for (elem, data_elem) in ck[r].preprocessed_row.iter().zip(committed_data.iter()) {
            temp *= (elem, data_elem);
            commitment[r] += &temp;
        }
    }

    let next = match &config.next {
        Some(next_config) => Some(Box::new(recursive_commit(crs, next_config, &commitment))),
        None => None,
    };

    RecursiveCommitment {
        decomposition_base_log: config.decomposition_base_log,
        decomposition_chunks: config.decomposition_chunks,
        committed_data,
        commitment,
        next,
    }
}

#[test]
fn test_recursive_commit() {
    let crs = CRS::gen_crs(256, 2);
    let data = vec![
        RingElement::all(37, Representation::IncompleteNTT),
        RingElement::all(36, Representation::IncompleteNTT),
        RingElement::all(37, Representation::IncompleteNTT),
        RingElement::all(36, Representation::IncompleteNTT),
        RingElement::all(37, Representation::IncompleteNTT),
        RingElement::all(36, Representation::IncompleteNTT),
        RingElement::all(37, Representation::IncompleteNTT),
        RingElement::all(36, Representation::IncompleteNTT),
    ];

    let config = RecursionConfig {
        decomposition_base_log: 3, // base 8
        decomposition_chunks: 4,
        rank: 2,
        prefix: Prefix {
            prefix: 0,
            length: 0,
        },
        next: None,
    };

    let recursive_commitment = recursive_commit(&crs, &config, &data);

    assert_eq!(recursive_commitment.committed_data.len(), 32);
    assert_eq!(recursive_commitment.commitment.len(), 2);
    assert_eq!(
        recursive_commitment.committed_data[0],
        RingElement::all(1, Representation::IncompleteNTT)
    );
    assert_eq!(
        recursive_commitment.committed_data[1],
        RingElement::all(0, Representation::IncompleteNTT)
    );
    assert_eq!(
        recursive_commitment.committed_data[2],
        RingElement::all(MOD_Q - 4, Representation::IncompleteNTT)
    );
    assert_eq!(
        recursive_commitment.committed_data[3],
        RingElement::all(0, Representation::IncompleteNTT)
    );
    assert!(recursive_commitment.next.is_none());
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
            // structured_row: StructuredRow {
            //     tensor_layers: vec![], // incorrect but not used here
            // },
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
            // structured_row: StructuredRow {
            //     tensor_layers: vec![], // incorrect but not used here
            // },
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

    let commitment = commit_basic_internal(&ck, &witness, 2);

    assert_eq!(
        &commitment[(0, 0)],
        &RingElement::constant(
            1 * 1 + 2 * 2 + 4 * 3 + 8 * 4 + 16 * 5 + 32 * 6 + 64 * 7 + 128 * 8,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        &commitment[(0, 1)],
        &RingElement::constant(
            1 * 9 + 2 * 10 + 4 * 11 + 8 * 12 + 16 * 13 + 32 * 14 + 64 * 15 + 128 * 16,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        &commitment[(1, 0)],
        &RingElement::constant(
            1 * 1 + 4 * 2 + 16 * 3 + 64 * 4 + 256 * 5 + 1024 * 6 + 4096 * 7 + 16384 * 8,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        &commitment[(1, 1)],
        &RingElement::constant(
            1 * 9 + 4 * 10 + 16 * 11 + 64 * 12 + 256 * 13 + 1024 * 14 + 4096 * 15 + 16384 * 16,
            Representation::IncompleteNTT
        )
    );
}
