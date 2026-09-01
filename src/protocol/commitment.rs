use std::ops::IndexMut;

#[cfg(feature = "parallel-commitment")]
use rayon::prelude::*;

use crate::{
    common::{
        decomposition::decompose_chunks_into,
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
    commit_basic_internal_with(ck, witness, rank, false)
}

/// One commitment per (key row, witness column); the cells are independent accumulations.
fn commit_basic_internal_with(
    ck: &CK,
    witness: &VerticallyAlignedMatrix<RingElement>,
    rank: usize,
    parallel: bool,
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

    #[cfg(feature = "parallel-commitment")]
    if parallel {
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
        return commitment;
    }
    let _ = parallel;

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

#[cfg(feature = "parallel-commitment")]
pub fn commit_basic_parallel(
    crs: &CRS,
    witness: &VerticallyAlignedMatrix<RingElement>,
    rank: usize,
) -> BasicCommitment {
    commit_basic_internal_with(crs.ck_for_wit_dim(witness.height), witness, rank, true)
}

/// Accumulates `sum_k ck_row[k] * operand[k]` into `acc`.
/// Accumulates `rank` independent inner products against the same operand.
///
/// Each output accumulates over `k` in ascending order regardless of who computes it, so which
/// thread takes which row does not affect the result.
fn accumulate_rows(
    commitment: &mut [RingElement],
    ck: &CK,
    operand: &[RingElement],
    rank: usize,
    parallel: bool,
) {
    #[cfg(feature = "parallel-commitment")]
    if parallel {
        commitment[..rank]
            .par_iter_mut()
            .enumerate()
            .for_each(|(r, acc)| inner_product_into(acc, &ck[r].preprocessed_row, operand));
        return;
    }
    let _ = parallel;
    for r in 0..rank {
        inner_product_into(&mut commitment[r], &ck[r].preprocessed_row, operand);
    }
}

fn inner_product_into(acc: &mut RingElement, ck_row: &[RingElement], operand: &[RingElement]) {
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    for (elem, op_elem) in ck_row.iter().zip(operand.iter()) {
        temp *= (elem, op_elem);
        *acc += &temp;
    }
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


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Prefix {
    pub prefix: usize,
    pub length: usize,
}

/// The dyadic blocks a component of size `size` is cut into, largest first: one block per set
/// bit of `size`, so a power-of-two component is a single block. Block `k` covers the component's
/// own indices `[offset_k, offset_k + size_k)`, and because the offsets are sums of strictly
/// larger powers of two every block is aligned to its own size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    pub size: usize,
    pub blocks: Vec<Prefix>,
}

/// The binary decomposition of `size`, largest first.
pub fn block_sizes(size: usize) -> Vec<usize> {
    (0..usize::BITS)
        .rev()
        .filter(|bit| (size >> bit) & 1 == 1)
        .map(|bit| 1usize << bit)
        .collect()
}

impl Placement {
    pub fn single(size: usize, prefix: Prefix) -> Self {
        debug_assert!(size.is_power_of_two());
        Placement {
            size,
            blocks: vec![prefix],
        }
    }

    pub fn block_sizes(&self) -> Vec<usize> {
        block_sizes(self.size)
    }

    /// Blocks with the component-local offset each one starts at.
    pub fn blocks_with_offsets(&self) -> Vec<(usize, usize, Prefix)> {
        let mut offset = 0;
        let mut out = Vec::with_capacity(self.blocks.len());
        for (size, prefix) in self.block_sizes().into_iter().zip(self.blocks.iter()) {
            out.push((offset, size, *prefix));
            offset += size;
        }
        out
    }

    /// The prefix addressing the `index`-th of `count` equal dyadic slices of the component.
    /// The slice must be a power of two long and fall inside a single block, which it does
    /// whenever its length divides every block.
    pub fn slice(&self, index: usize, count: usize) -> Prefix {
        debug_assert_eq!(self.size % count, 0, "slice count must divide the component");
        let len = self.size / count;
        debug_assert!(len.is_power_of_two(), "a slice must be dyadic");
        let start = index * len;
        for (offset, size, prefix) in self.blocks_with_offsets() {
            if start < offset + size {
                debug_assert!(
                    start + len <= offset + size,
                    "slice {index}/{count} of a size-{} component crosses a block boundary",
                    self.size
                );
                let parts = size / len;
                return Prefix {
                    prefix: prefix.prefix * parts + (start - offset) / len,
                    length: prefix.length + parts.ilog2() as usize,
                };
            }
        }
        unreachable!("slice {index} of {count} is outside the component")
    }
}

/// One recursion level: a single Ajtai commitment over this level's input, whose `rank` output
/// elements are the next level's input.
///
/// `placements` says where the level's decomposed input lives in the next round's witness, one
/// placement per row. A commitment's input length must be a power of two (`ck_for_wit_dim`
/// indexes the key by `ilog2`), so an `r x width` input is committed over `r.next_power_of_two()`
/// rows and each row over `decomposition_chunks.next_power_of_two()` digit planes. The prover
/// leaves the padding rows and padding planes zero, so only the real ones are placed and the
/// padding costs the round nothing.
///
/// A row is laid out digit-major: plane `j` is the contiguous block of `row_len` `j`-th digits.
/// That keeps every plane dyadically addressable for a radix that is not a power of two, where
/// the element-major stride of `decompose` would not be.
#[derive(Clone, Debug)]
pub struct RecursionConfig {
    pub decomposition_base_log: usize,
    pub decomposition_chunks: usize,
    pub rank: usize,
    /// One placement per row segment of this level's input, in row order.
    pub placements: Vec<Placement>,
    pub next: Option<Box<RecursionConfig>>,
}

impl RecursionConfig {
    pub fn most_inner_config(&self) -> &RecursionConfig {
        match &self.next {
            Some(next_config) => next_config.most_inner_config(),
            None => self,
        }
    }

    /// The placement of a level whose input is a single row.
    pub fn placement(&self) -> &Placement {
        // Checked in release too: on a multi-row level this would return one row's placement and
        // the caller would constrain that row alone, which is a wrong proof rather than a crash.
        assert_eq!(
            self.placements.len(),
            1,
            "this level has one placement per row"
        );
        &self.placements[0]
    }

    /// How many row segments the padded input is cut into: the real rows plus the zero padding
    /// that is committed but never placed.
    pub fn segments(&self) -> usize {
        self.placements.len().next_power_of_two()
    }

    /// How many digit planes the padded row is cut into; the planes past `decomposition_chunks`
    /// are zero, committed but never placed.
    pub fn padded_chunks(&self) -> usize {
        self.decomposition_chunks.next_power_of_two()
    }

    /// The length of one row of this level's (undecomposed) input.
    pub fn row_len(&self) -> usize {
        // Checked in release too: one length speaks for every row, and a level whose rows
        // differed in size would give both parties the same wrong key segments and the same
        // wrong row stride, which is a wrong proof rather than a crash.
        assert!(
            self.placements
                .iter()
                .all(|placement| placement.size == self.placements[0].size),
            "row components of one level are equally sized"
        );
        self.placements[0].size / self.decomposition_chunks
    }

    /// The length of the padded vector this level actually commits to.
    pub fn committed_len(&self) -> usize {
        self.row_len() * self.segments() * self.padded_chunks()
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
    recursive_commit_with(crs, config, data, false)
}

#[cfg(feature = "parallel-commitment")]
#[tracing::instrument(skip_all, name = "commit::recursive_layer_parallel")]
pub fn recursive_commit_parallel(
    crs: &CRS,
    config: &RecursionConfig,
    data: &Vec<RingElement>,
) -> RecursiveCommitmentWithAux {
    recursive_commit_with(crs, config, data, true)
}

fn recursive_commit_with(
    crs: &CRS,
    config: &RecursionConfig,
    data: &Vec<RingElement>,
    parallel: bool,
) -> RecursiveCommitmentWithAux {
    // A commitment's input length must be a power of two, so the input is padded up to
    // `segments()` rows of `padded_chunks()` digit planes. Only the real rows are decomposed:
    // decomposing a zero row does not give zero digits, and the padding has to be genuinely zero
    // for the next round to be allowed to leave it out of its witness.
    let row_len = data.len() / config.segments();
    let chunks = config.decomposition_chunks;
    let padded = config.padded_chunks();

    let mut committed_data =
        vec![RingElement::zero(Representation::IncompleteNTT); data.len() * padded];

    for row in 0..config.placements.len() {
        let start = row * row_len * padded;
        decompose_chunks_into(
            &mut committed_data[start..start + row_len * chunks],
            &data[row * row_len..(row + 1) * row_len],
            config.decomposition_base_log as u64,
            chunks,
        );
    }

    let ck = crs.ck_for_wit_dim(committed_data.len());

    let mut commitment = vec![RingElement::zero(Representation::IncompleteNTT); config.rank];

    accumulate_rows(&mut commitment, ck, &committed_data, config.rank, parallel);

    let next = config.next.as_ref().map(|next_config| {
        Box::new(recursive_commit_with(crs, next_config, &commitment, parallel))
    });

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
            placements: vec![Placement::single(
                32,
                Prefix {
                    prefix: 0,
                    length: 0,
                },
            )],
            next: None,
        };

        let recursive_commitment = recursive_commit(&crs, &config, &data);

        // 8 inputs x 4 chunks each = 32 decomposed elements
        debug_assert_eq!(recursive_commitment.committed_data.len(), 32);
        // rank = 2 -> 2 commitment elements
        debug_assert_eq!(recursive_commitment.commitment.len(), 2);

        // The row is laid out digit-major: plane j holds the j-th digit of all 8 inputs.
        // Balanced decomposition of 37 with base_log=3 (b=8), radix=4:
        //   k = (b/2) * (1 + b + b^2 + b^3) = 4 * (1 + 8 + 64 + 512) = 2340
        //   37 + k = 2377 -> base-8 digits: [1, 1, 5, 4]
        //   balanced (subtract b/2 = 4): [-3, -3, 1, 0]
        debug_assert_eq!(
            recursive_commitment.committed_data[0],
            RingElement::all(MOD_Q - 3, Representation::IncompleteNTT)
        );
        debug_assert_eq!(
            recursive_commitment.committed_data[8],
            RingElement::all(MOD_Q - 3, Representation::IncompleteNTT)
        );
        debug_assert_eq!(
            recursive_commitment.committed_data[16],
            RingElement::all(1, Representation::IncompleteNTT)
        );
        debug_assert_eq!(
            recursive_commitment.committed_data[24],
            RingElement::all(0, Representation::IncompleteNTT)
        );

        // Balanced decomposition of 36:
        //   36 + k = 2376 -> base-8 digits: [0, 1, 5, 4]
        //   balanced: [-4, -3, 1, 0]
        debug_assert_eq!(
            recursive_commitment.committed_data[1],
            RingElement::all(MOD_Q - 4, Representation::IncompleteNTT)
        );
        debug_assert_eq!(
            recursive_commitment.committed_data[9],
            RingElement::all(MOD_Q - 3, Representation::IncompleteNTT)
        );
        debug_assert_eq!(
            recursive_commitment.committed_data[17],
            RingElement::all(1, Representation::IncompleteNTT)
        );
        debug_assert_eq!(
            recursive_commitment.committed_data[25],
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

    #[test]
    fn placement_cuts_a_component_into_its_binary_decomposition() {
        debug_assert_eq!(block_sizes(12), vec![8, 4]);
        debug_assert_eq!(block_sizes(16), vec![16]);
        debug_assert_eq!(block_sizes(1), vec![1]);

        // A size-12 component of a 16-long witness: an 8-block at 0 and a 4-block at 8.
        let placement = Placement {
            size: 12,
            blocks: vec![
                Prefix {
                    prefix: 0,
                    length: 1,
                },
                Prefix {
                    prefix: 2,
                    length: 2,
                },
            ],
        };

        debug_assert_eq!(
            placement.blocks_with_offsets(),
            vec![
                (
                    0,
                    8,
                    Prefix {
                        prefix: 0,
                        length: 1
                    }
                ),
                (
                    8,
                    4,
                    Prefix {
                        prefix: 2,
                        length: 2
                    }
                ),
            ]
        );

        // Three digit planes of length four: two inside the first block, one is the second block.
        debug_assert_eq!(
            placement.slice(0, 3),
            Prefix {
                prefix: 0,
                length: 2
            }
        );
        debug_assert_eq!(
            placement.slice(1, 3),
            Prefix {
                prefix: 1,
                length: 2
            }
        );
        debug_assert_eq!(
            placement.slice(2, 3),
            Prefix {
                prefix: 2,
                length: 2
            }
        );
    }
}