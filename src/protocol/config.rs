use std::{any::Any, sync::LazyLock};

use crate::{
    common::{
        matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        ring_arithmetic::{QuadraticExtension, RingElement},
    },
    protocol::{
        commitment::{Prefix, RecursionConfig, RecursiveCommitment, RecursiveCommitmentWithAux},
        config_generator::{AuxConfig, AuxProjection, AuxRecursionConfig, AuxSumcheckConfig},
        sumcheck_utils::polynomial::Polynomial,
    },
};

// 2^26 = 2^7 (DEGREE) * 2^19
// 2^19 = 2^5 * 2^14

#[derive(Clone)]
pub enum ProjectionType {
    Type0, // full ring elements + no batching
    Type1, // coefficient-wise + batching + commit to constant terms + commit to batched projection
}

#[derive(Clone)]
pub struct Type1ProjectionConfig {
    pub nof_batches: usize,
    pub recursion_constant_term: RecursionConfig, // here we check the norm
    pub recursion_batched_projection: RecursionConfig, // this one is for the actual projection and we need to show consistency
}

pub type Type0ProjectionConfig = RecursionConfig;

#[derive(Clone)]
pub enum Projection {
    Type0(Type0ProjectionConfig),
    Type1(Type1ProjectionConfig),
}

pub static REAL_CONFIG: LazyLock<Config> = LazyLock::new(|| {
    AuxSumcheckConfig {
        witness_height: 2usize.pow(15),   // 2^15
        witness_width: 2usize.pow(6),     // 2^6
        projection_ratio: 2usize.pow(6),  // 2^6
        projection_height: 2usize.pow(8), // 2^8
        basic_commitment_rank: 4,
        nof_openings: 1,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 15, // 2^5 (witness_width) * 2^2 (rank) * 2^2 (decomp) = 2^9
            decomposition_chunks: 4,
            rank: 1,
            next: Some(Box::new(AuxRecursionConfig {
                decomposition_base_log: 7,
                decomposition_chunks: 8, // 1 (rank) * 8 (decomp) = 2^3
                rank: 1,
                next: None,
            })),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 15, // 2^5 (witness_width) * 2^0 (nof openings) * 2^2 (decomp) = 2^7
            decomposition_chunks: 4, // for now, there's no reason why decomposition_chunks here shall be different from commitment_recursion.decomposition_chunks. I will use that assumption in sumcheck.
            rank: 1,
            next: None,
        },
        projection_recursion: AuxProjection::Type0(AuxRecursionConfig {
            // 2^14 (witness_height) * 2^5 (witness_width) / 2^5 (projection_ratio) * 2^0 (decomp) = 2^14
            decomposition_base_log: 20, // no decomposition
            decomposition_chunks: 1,
            rank: 1,
            next: None,
        }),

        witness_decomposition_chunks: 2,
        witness_decomposition_base_log: 10, // no decomposition

        next: Some(Box::new(AuxConfig::Sumcheck(AuxSumcheckConfig {
            witness_height: 2usize.pow(10),
            witness_width: 2usize.pow(7),
            projection_ratio: 2usize.pow(7),
            projection_height: 2usize.pow(8),
            basic_commitment_rank: 2,
            nof_openings: 2,
            commitment_recursion: AuxRecursionConfig {
                decomposition_base_log: 15, // 2^5 (witness_width) * 2^2 (rank) * 2^2 (decomp) = 2^9
                decomposition_chunks: 4,
                rank: 1,
                next: Some(Box::new(AuxRecursionConfig {
                    decomposition_base_log: 7,
                    decomposition_chunks: 8, // 1 (rank) * 8 (decomp) = 2^3
                    rank: 1,
                    next: None,
                })),
            },
            opening_recursion: AuxRecursionConfig {
                decomposition_base_log: 15, // 2^5 (witness_width) * 2^0 (nof openings) * 2^2 (decomp) = 2^7
                decomposition_chunks: 4, // for now, there's no reason why decomposition_chunks here shall be different from commitment_recursion.decomposition_chunks. I will use that assumption in sumcheck.
                rank: 1,
                next: None,
            },
            projection_recursion: AuxProjection::Type1 {
                nof_batches: 2,
                recursion_constant_term: AuxRecursionConfig {
                    decomposition_base_log: 15,
                    decomposition_chunks: 4,
                    rank: 1,
                    next: None,
                },
                recursion_batched_projection: AuxRecursionConfig {
                    decomposition_base_log: 15,
                    decomposition_chunks: 4,
                    rank: 1,
                    next: None,
                },
            },

            witness_decomposition_chunks: 2,
            witness_decomposition_base_log: 10, // no decomposition

            next: None,
        }))),
    }
    .generate_config()
});

pub static TOY_CONFIG: LazyLock<Config> = LazyLock::new(|| {
    AuxSumcheckConfig {
        witness_height: 512,
        witness_width: 16,
        projection_ratio: 32,
        projection_height: 8, // small for testing
        basic_commitment_rank: 2,
        nof_openings: 1,

        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 15,
            decomposition_chunks: 4,
            rank: 1,
            next: Some(Box::new(AuxRecursionConfig {
                decomposition_base_log: 7,
                decomposition_chunks: 8,
                rank: 1,
                next: None,
            })),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 15,
            decomposition_chunks: 4,
            rank: 1,
            next: None,
        },
        projection_recursion: AuxProjection::Type0(AuxRecursionConfig {
            decomposition_base_log: 15,
            decomposition_chunks: 2,
            rank: 1,
            next: None,
        }),

        witness_decomposition_chunks: 2,
        witness_decomposition_base_log: 15,
        next: None,
    }
    .generate_config()
});

pub static TOY_CONFIG_II: LazyLock<Config> = LazyLock::new(|| {
    AuxSumcheckConfig {
        witness_height: 1024,
        witness_width: 16,
        projection_ratio: 32,
        projection_height: 256,
        basic_commitment_rank: 2,
        nof_openings: 1,

        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 15,
            decomposition_chunks: 4,
            rank: 1,
            next: Some(Box::new(AuxRecursionConfig {
                decomposition_base_log: 7,
                decomposition_chunks: 8,
                rank: 1,
                next: None,
            })),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 15,
            decomposition_chunks: 4,
            rank: 1,
            next: None,
        },
        projection_recursion: AuxProjection::Type1 {
            nof_batches: 2,
            recursion_constant_term: AuxRecursionConfig {
                decomposition_base_log: 10,
                decomposition_chunks: 2,
                rank: 1,
                next: None,
            },
            recursion_batched_projection: AuxRecursionConfig {
                decomposition_base_log: 15,
                decomposition_chunks: 4,
                rank: 1,
                next: None,
            },
        },

        witness_decomposition_chunks: 2,
        witness_decomposition_base_log: 15,
        next: Some(Box::new(AuxConfig::Simple(SimpleConfig {
            witness_height: 256,
            witness_width: 16,
            projection_ratio: 128,
            projection_height: 256,
            projection_nof_batches: 2,
            basic_commitment_rank: 2,
        }))),
    }
    .generate_config()
});

pub static CONFIG: LazyLock<Config> = LazyLock::new(|| TOY_CONFIG_II.clone());

#[derive(Clone)]
pub enum Config {
    Sumcheck(SumcheckConfig),
    Simple(SimpleConfig),
}

pub trait ConfigBase: Any {
    fn witness_height(&self) -> usize;
    fn witness_width(&self) -> usize;
    fn projection_ratio(&self) -> usize;
    fn projection_height(&self) -> usize;
    fn basic_commitment_rank(&self) -> usize;
}

#[derive(Clone)]
pub struct SumcheckConfig {
    pub witness_height: usize,
    pub witness_width: usize,
    pub projection_ratio: usize,  // shall be likely the witness_height
    pub projection_height: usize, // likely 256 unless for testing
    pub commitment_recursion: RecursionConfig,
    pub opening_recursion: RecursionConfig,
    pub projection_recursion: Projection,
    pub nof_openings: usize,

    pub witness_decomposition_base_log: usize,
    pub witness_decomposition_chunks: usize,
    pub folded_witness_prefix: Prefix,

    pub basic_commitment_rank: usize,
    pub composed_witness_length: usize,

    pub next: Option<Box<Config>>, // for multiple rounds
}

impl ConfigBase for SumcheckConfig {
    fn witness_height(&self) -> usize {
        self.witness_height
    }

    fn witness_width(&self) -> usize {
        self.witness_width
    }

    fn projection_ratio(&self) -> usize {
        self.projection_ratio
    }

    fn projection_height(&self) -> usize {
        self.projection_height
    }
    fn basic_commitment_rank(&self) -> usize {
        self.basic_commitment_rank
    }
}

#[derive(Clone)]
pub struct SimpleConfig {
    pub witness_height: usize,
    pub witness_width: usize,
    pub projection_ratio: usize,  // shall be likely the witness_height
    pub projection_height: usize, // likely 256 unless for testing
    pub projection_nof_batches: usize,
    pub basic_commitment_rank: usize,
    // pub next: Option<Box<SimpleConfig>>, // for multiple rounds
}

impl ConfigBase for SimpleConfig {
    fn witness_height(&self) -> usize {
        self.witness_height
    }

    fn witness_width(&self) -> usize {
        self.witness_width
    }

    fn projection_ratio(&self) -> usize {
        self.projection_ratio
    }

    fn projection_height(&self) -> usize {
        self.projection_height
    }
    fn basic_commitment_rank(&self) -> usize {
        self.basic_commitment_rank
    }
}

pub enum RoundProof {
    Sumcheck(SumcheckRoundProof),
    Simple(SimpleRoundProof),
}

pub enum NextRoundCommitment {
    Recursive(RecursiveCommitment), // if the next round is sumcheck
    Simple(HorizontallyAlignedMatrix<RingElement>), // if the next round is simple
}

pub struct SumcheckRoundProof {
    pub polys: Vec<Polynomial<QuadraticExtension>>,
    pub claim_over_witness: RingElement,
    pub claim_over_witness_conjugate: RingElement,
    pub norm_claim: RingElement,
    pub rc_opening_inner: Vec<RingElement>,
    pub rc_projection_inner: Option<Vec<RingElement>>,
    pub rcs_projection_1_inner: Option<(Vec<RingElement>, Vec<RingElement>)>,
    pub constant_term_claims: Option<Vec<RingElement>>,
    pub next_round_commitment: Option<NextRoundCommitment>,
    pub next: Option<Box<RoundProof>>,
}

pub struct SimpleRoundProof {
    pub folded_witness: VerticallyAlignedMatrix<RingElement>,
    pub projection_image_ct: VerticallyAlignedMatrix<RingElement>, // cosntant term projection image embedded
    pub batched_projection_image: HorizontallyAlignedMatrix<RingElement>,
    pub opening_rhs: HorizontallyAlignedMatrix<RingElement>,
}

#[inline]
pub fn paste_by_prefix(dest: &mut Vec<RingElement>, src: &Vec<RingElement>, prefix: &Prefix) {
    assert_eq!(
        src.len(),
        1 << dest.len().ilog2() as usize - prefix.length,
        "Pasting failed. Source length does not match prefix length."
    );
    // e.g. if dest.len() = 2048, prefix.length = 4, prefix.prefix = 9 (0b1001)
    // then start = 9 << (11 - 4) = 9 << 7 = 1152 = 10010000000 index to start pasting
    let start = prefix.prefix << (dest.len().ilog2() as usize - prefix.length);
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dest.as_mut_ptr().add(start), src.len());
    }
}

pub fn paste_recursive_commitment(
    dest: &mut Vec<RingElement>,
    commitment: &RecursiveCommitmentWithAux,
    config: &RecursionConfig,
) {
    paste_by_prefix(dest, &commitment.committed_data, &config.prefix);

    if let (Some(next_commitment), Some(next_config)) = (&commitment.next, &config.next) {
        paste_recursive_commitment(dest, next_commitment, next_config);
    }
}

#[inline]
pub fn slice_by_prefix(src: &Vec<RingElement>, prefix: &Prefix) -> Vec<RingElement> {
    let start = prefix.prefix << (src.len().ilog2() as usize - prefix.length);
    let length = 1 << (src.len().ilog2() as usize - prefix.length);
    src[start..start + length].to_vec()
}
