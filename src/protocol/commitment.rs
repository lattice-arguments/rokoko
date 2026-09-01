use std::ops::IndexMut;

#[cfg(feature = "parallel-commitment")]
use rayon::prelude::*;

use crate::{
    common::{
        decomposition::decompose,
        matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        ring_arithmetic::{Representation, RingElement},
    },
    protocol::{
        crs::{CK, CRS},
        project_coarse::Signed16RingElement,
    },
};

pub type BasicCommitment = HorizontallyAlignedMatrix<RingElement>;

// precompute auxiliary witness stored as i16 for faster coarse projections
pub struct CommitmentWithAux {
    pub rc_commitment_with_aux: RecursiveCommitmentWithAux,
    pub witness_i16: Option<VerticallyAlignedMatrix<Signed16RingElement>>,
}

impl CommitmentWithAux {
    pub fn from_rc_commitment_with_aux(rc_commitment_with_aux: RecursiveCommitmentWithAux) -> Self {
        CommitmentWithAux {
            rc_commitment_with_aux,
            witness_i16: None,
        }
    }
}

#[tracing::instrument(skip_all, name = "commit::basic_internal")]
pub fn commit_basic_internal(
    ck: &CK,
    witness: &VerticallyAlignedMatrix<RingElement>,
    rank: usize,
) -> BasicCommitment {
    if rank == 0 {
        return HorizontallyAlignedMatrix {
            data: vec![RingElement::zero(Representation::IncompleteNTT); 0 * witness.width],
            width: witness.width,
            height: 0,
        };
    }
    let mut commitment = HorizontallyAlignedMatrix {
        data: vec![
            RingElement::zero(Representation::IncompleteNTT);
            rank.next_power_of_two() * witness.width
        ],
        width: witness.width,
        height: rank.next_power_of_two(),
    };

    for (i, row) in ck.iter().take(rank).enumerate() {
        for col in 0..witness.used_cols {
            inner_product_into(
                commitment.index_mut((i, col)),
                &row.preprocessed_row,
                witness.col(col),
            );
        }
    }
    commitment
}

/// Accumulates `sum_k ck_row[k] * operand[k]` into `acc`.
fn inner_product_into(acc: &mut RingElement, ck_row: &[RingElement], operand: &[RingElement]) {
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    for (elem, op_elem) in ck_row.iter().zip(operand.iter()) {
        temp *= (elem, op_elem);
        *acc += &temp;
    }
}

/// `commit_basic_internal` with the output cells spread over a rayon pool.
///
/// Every cell `(i, col)` is an independent inner product accumulated in the
/// same order as the serial loop, so the commitment is bit-identical.
#[cfg(feature = "parallel-commitment")]
#[tracing::instrument(skip_all, name = "commit::basic_internal_parallel")]
pub fn commit_basic_internal_parallel(
    ck: &CK,
    witness: &VerticallyAlignedMatrix<RingElement>,
    rank: usize,
) -> BasicCommitment {
    if rank == 0 {
        return HorizontallyAlignedMatrix {
            data: vec![RingElement::zero(Representation::IncompleteNTT); 0 * witness.width],
            width: witness.width,
            height: 0,
        };
    }
    let mut commitment = HorizontallyAlignedMatrix {
        data: vec![
            RingElement::zero(Representation::IncompleteNTT);
            rank.next_power_of_two() * witness.width
        ],
        width: witness.width,
        height: rank.next_power_of_two(),
    };

    let width = commitment.width;
    let used_cols = witness.used_cols;
    commitment
        .data
        .par_chunks_mut(width)
        .take(rank)
        .enumerate()
        .for_each(|(i, commitment_row)| {
            let ck_row = &ck[i].preprocessed_row;
            commitment_row[..used_cols]
                .par_iter_mut()
                .enumerate()
                .for_each(|(col, acc)| inner_product_into(acc, ck_row, witness.col(col)));
        });
    commitment
}

// this is first level commit for FW = Y
pub fn commit_basic(
    crs: &CRS,
    witness: &VerticallyAlignedMatrix<RingElement>,
    rank: usize,
) -> BasicCommitment {
    let ck = crs.ck_for_wit_dim(witness.height);
    let commitment = commit_basic_internal(ck, witness, rank);

    commitment
}

#[cfg(feature = "parallel-commitment")]
pub fn commit_basic_parallel(
    crs: &CRS,
    witness: &VerticallyAlignedMatrix<RingElement>,
    rank: usize,
) -> BasicCommitment {
    let ck = crs.ck_for_wit_dim(witness.height);
    commit_basic_internal_parallel(ck, witness, rank)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Prefix {
    pub prefix: usize,
    pub length: usize,
}

#[derive(Clone, Debug)]
pub struct RecursionConfig {
    pub decomposition_base_log: usize,
    pub decomposition_chunks: usize,
    pub rank: usize,
    pub prefix: Prefix,
    pub next: Option<Box<RecursionConfig>>,
}

impl RecursionConfig {
    pub fn most_inner_config(&self) -> &RecursionConfig {
        match &self.next {
            Some(next_config) => next_config.most_inner_config(),
            None => self,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RecursiveCommitmentWithAux {
    pub decomposition_base_log: usize,
    pub decomposition_chunks: usize,
    pub committed_data: Vec<RingElement>,
    pub commitment: Vec<RingElement>,
    pub next: Option<Box<RecursiveCommitmentWithAux>>,
}

impl RecursiveCommitmentWithAux {
    pub fn most_inner_commitment(&self) -> &RecursiveCommitment {
        match &self.next {
            Some(next_config) => next_config.most_inner_commitment(),
            None => &self.commitment,
        }
    }

    pub fn most_inner_commitment_with_aux(&self) -> &RecursiveCommitmentWithAux {
        match &self.next {
            Some(next_config) => next_config.most_inner_commitment_with_aux(),
            None => self,
        }
    }
}

pub type RecursiveCommitment = Vec<RingElement>;

#[tracing::instrument(skip_all, name = "commit::recursive_layer")]
pub fn recursive_commit(
    crs: &CRS,
    config: &RecursionConfig,
    data: &Vec<RingElement>,
) -> RecursiveCommitmentWithAux {
    let committed_data = decompose(
        &data,
        config.decomposition_base_log as u64,
        config.decomposition_chunks,
    );

    let ck = crs.ck_for_wit_dim(committed_data.len());

    let mut commitment = vec![RingElement::zero(Representation::IncompleteNTT); config.rank];

    for r in 0..config.rank {
        inner_product_into(&mut commitment[r], &ck[r].preprocessed_row, &committed_data);
    }

    let next = match &config.next {
        Some(next_config) => Some(Box::new(recursive_commit(crs, next_config, &commitment))),
        None => None,
    };

    RecursiveCommitmentWithAux {
        decomposition_base_log: config.decomposition_base_log,
        decomposition_chunks: config.decomposition_chunks,
        committed_data,
        commitment,
        next,
    }
}

/// `recursive_commit` with the rank rows of each layer spread over a rayon
/// pool. Each row is an independent inner product accumulated in the same
/// order as the serial loop, so every layer is bit-identical.
#[cfg(feature = "parallel-commitment")]
#[tracing::instrument(skip_all, name = "commit::recursive_layer_parallel")]
pub fn recursive_commit_parallel(
    crs: &CRS,
    config: &RecursionConfig,
    data: &Vec<RingElement>,
) -> RecursiveCommitmentWithAux {
    let committed_data = decompose(
        &data,
        config.decomposition_base_log as u64,
        config.decomposition_chunks,
    );

    let ck = crs.ck_for_wit_dim(committed_data.len());

    let mut commitment = vec![RingElement::zero(Representation::IncompleteNTT); config.rank];

    commitment
        .par_iter_mut()
        .enumerate()
        .for_each(|(r, acc)| inner_product_into(acc, &ck[r].preprocessed_row, &committed_data));

    let next = match &config.next {
        Some(next_config) => Some(Box::new(recursive_commit_parallel(
            crs,
            next_config,
            &commitment,
        ))),
        None => None,
    };

    RecursiveCommitmentWithAux {
        decomposition_base_log: config.decomposition_base_log,
        decomposition_chunks: config.decomposition_chunks,
        committed_data,
        commitment,
        next,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::config::MOD_Q;
    use crate::common::structured_row::PreprocessedRow;

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

        // 8 inputs × 4 chunks each = 32 decomposed elements
        debug_assert_eq!(recursive_commitment.committed_data.len(), 32);
        // rank = 2 → 2 commitment elements
        debug_assert_eq!(recursive_commitment.commitment.len(), 2);

        // Balanced decomposition of 37 with base_log=3 (b=8), radix=4:
        //   k = (b/2) * (1 + b + b² + b³) = 4 * (1 + 8 + 64 + 512) = 2340
        //   37 + k = 2377 → base-8 digits: [1, 1, 5, 4]
        //   balanced (subtract b/2 = 4): [-3, -3, 1, 0]
        debug_assert_eq!(
            recursive_commitment.committed_data[0],
            RingElement::all(MOD_Q - 3, Representation::IncompleteNTT)
        );
        debug_assert_eq!(
            recursive_commitment.committed_data[1],
            RingElement::all(MOD_Q - 3, Representation::IncompleteNTT)
        );
        debug_assert_eq!(
            recursive_commitment.committed_data[2],
            RingElement::all(1, Representation::IncompleteNTT)
        );
        debug_assert_eq!(
            recursive_commitment.committed_data[3],
            RingElement::all(0, Representation::IncompleteNTT)
        );

        // Balanced decomposition of 36:
        //   36 + k = 2376 → base-8 digits: [0, 1, 5, 4]
        //   balanced: [-4, -3, 1, 0]
        debug_assert_eq!(
            recursive_commitment.committed_data[4],
            RingElement::all(MOD_Q - 4, Representation::IncompleteNTT)
        );
        debug_assert_eq!(
            recursive_commitment.committed_data[5],
            RingElement::all(MOD_Q - 3, Representation::IncompleteNTT)
        );
        debug_assert_eq!(
            recursive_commitment.committed_data[6],
            RingElement::all(1, Representation::IncompleteNTT)
        );
        debug_assert_eq!(
            recursive_commitment.committed_data[7],
            RingElement::all(0, Representation::IncompleteNTT)
        );
        debug_assert!(recursive_commitment.next.is_none());
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
            used_cols: 2,
        };

        let commitment = commit_basic_internal(&ck, &witness, 2);

        debug_assert_eq!(
            &commitment[(0, 0)],
            &RingElement::constant(
                1 * 1 + 2 * 2 + 4 * 3 + 8 * 4 + 16 * 5 + 32 * 6 + 64 * 7 + 128 * 8,
                Representation::IncompleteNTT
            )
        );

        debug_assert_eq!(
            &commitment[(0, 1)],
            &RingElement::constant(
                1 * 9 + 2 * 10 + 4 * 11 + 8 * 12 + 16 * 13 + 32 * 14 + 64 * 15 + 128 * 16,
                Representation::IncompleteNTT
            )
        );

        debug_assert_eq!(
            &commitment[(1, 0)],
            &RingElement::constant(
                1 * 1 + 4 * 2 + 16 * 3 + 64 * 4 + 256 * 5 + 1024 * 6 + 4096 * 7 + 16384 * 8,
                Representation::IncompleteNTT
            )
        );

        debug_assert_eq!(
            &commitment[(1, 1)],
            &RingElement::constant(
                1 * 9 + 4 * 10 + 16 * 11 + 64 * 12 + 256 * 13 + 1024 * 14 + 4096 * 15 + 16384 * 16,
                Representation::IncompleteNTT
            )
        );
    }
}
