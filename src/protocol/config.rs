use std::sync::LazyLock;

use crate::{
    common::ring_arithmetic::RingElement,
    protocol::{
        commitment::{Prefix, RecursionConfig, RecursiveCommitment},
        config_generator::{AuxConfig, AuxProjection, AuxRecursionConfig},
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

pub static REAL_CONFIG: LazyLock<Config> = LazyLock::new(|| Config {
    witness_height: 2usize.pow(14),   // 2^14
    witness_width: 2usize.pow(5),     // 2^5
    projection_ratio: 2usize.pow(5),  // 2^5
    projection_height: 2usize.pow(8), // 2^8
    basic_commitment_rank: 4,
    nof_openings: 1,

    commitment_recursion: RecursionConfig {
        decomposition_base_log: 15, // 2^5 (witness_width) * 2^2 (rank) * 2^2 (decomp) = 2^9
        decomposition_chunks: 4,
        rank: 1,
        prefix: Prefix {
            prefix: 0b1100000,
            length: 7, // 2^16 / 2^9 = 2^7
        },
        next: Some(Box::new(RecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8, // 1 (rank) * 8 (decomp) = 2^3
            rank: 1,
            prefix: Prefix {
                prefix: 0b1100001010000,
                length: 13, // 2^16 / 2^3 = 2^13
            },
            next: None,
        })),
    },
    opening_recursion: RecursionConfig {
        decomposition_base_log: 15, // 2^5 (witness_width) * 2^0 (nof openings) * 2^2 (decomp) = 2^7
        decomposition_chunks: 4, // for now, there's no reason why decomposition_chunks here shall be different from commitment_recursion.decomposition_chunks. I will use that assumption in sumcheck.
        rank: 1,
        next: None,
        prefix: Prefix {
            prefix: 0b110000100,
            length: 9, // 2^16 / 2^7 = 2^9
        },
    },
    projection_recursion: Projection::Type0(Type0ProjectionConfig {
        // 2^14 (witness_height) * 2^5 (witness_width) / 2^5 (projection_ratio) * 2^0 (decomp) = 2^14
        decomposition_base_log: 20, // no decomposition
        decomposition_chunks: 1,
        rank: 1,
        next: None,
        prefix: Prefix {
            prefix: 0b10,
            length: 2, // 2^16 / 2^14 = 2^2
        },
    }),

    folded_witness_prefix: Prefix {
        //  2^14 (witness_height) * 2^1 (decomp) = 2^15
        prefix: 0b0,
        length: 1, // 2^16 / 2^15 = 2^1
    },
    witness_decomposition_chunks: 2,
    witness_decomposition_base_log: 10, // no decomposition

    composed_witness_length: 2usize.pow(16),

    next: None, // for multiple rounds
});

pub static TOY_CONFIG: LazyLock<Config> = LazyLock::new(|| {
    AuxConfig {
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
    AuxConfig {
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
        witness_decomposition_base_log: 15,
        next: None,
    }
    .generate_config()
});

pub static CONFIG: LazyLock<Config> = LazyLock::new(|| TOY_CONFIG_II.clone());

#[derive(Clone)]
pub struct Config {
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
    commitment: &RecursiveCommitment,
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
