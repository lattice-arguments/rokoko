use std::sync::LazyLock;

use crate::{
    common::ring_arithmetic::RingElement,
    protocol::commitment::{Prefix, RecursionConfig, RecursiveCommitment},
};

pub const BASIC_COMMITMENT_RANK: usize = 2;

pub static CONFIG: LazyLock<Config> = LazyLock::new(|| Config {
    witness_height: 512,
    witness_width: 16,
    projection_ratio: 32,
    basic_commitment_rank: BASIC_COMMITMENT_RANK,
    nof_openings: 1,

    commitment_recursion: RecursionConfig {
        decomposition_base_log: 15,
        decomposition_chunks: 4,
        rank: 1,
        next: None,
        prefix: Prefix {
            prefix: 0b1100,
            length: 4,
        }, // 2048 / 2^4 = 128
    },
    opening_recursion: RecursionConfig {
        decomposition_base_log: 15,
        decomposition_chunks: 4,
        rank: 1,
        next: None,
        prefix: Prefix {
            prefix: 0b11010,
            length: 5,
        }, // 2048 / 2^5 = 64
    },
    projection_recursion: RecursionConfig {
        decomposition_base_log: 15,
        decomposition_chunks: 2,
        rank: 1,
        next: None,
        prefix: Prefix {
            prefix: 0b10,
            length: 2,
        }, // 2048 / 2^2 = 512
    },

    folded_witness_prefix: Prefix {
        prefix: 0b0,
        length: 1,
    }, // 2048 / 2^1 = 1024
    witness_decomposition_chunks: 2,
    witness_decomposition_base_log: 15,

    // committed basic_commitment_len = basic_commitment_rank * witness_width * commitment_recursion.decomposition_chunks = 2 * 16 * 4 = 128

    // committed projection_image_len = witness_height * witness_width / projection_ratio  * projection_recursion.decomposition_chunks = (512 * 16 / 32) * 2 = 512

    // committed opening_len = nof_openings * witness_width * opening_recursion.decomposition_chunks = 1 * 16 * 4 = 64

    // folded_witness len is witness_height * witness_decomposition_chunks = 512 * 2 = 1024

    // in total, we fit into 2048 elements per round
    composed_witness_length: 2048,

    next: None, // for multiple rounds
});

pub struct Config {
    pub witness_height: usize,
    pub witness_width: usize,
    pub projection_ratio: usize, // shall be likely the witness_height
    pub commitment_recursion: RecursionConfig,
    pub opening_recursion: RecursionConfig,
    pub projection_recursion: RecursionConfig,
    pub nof_openings: usize,

    pub witness_decomposition_base_log: usize,
    pub witness_decomposition_chunks: usize,
    pub folded_witness_prefix: Prefix,

    pub basic_commitment_rank: usize,
    pub composed_witness_length: usize,

    pub next: Option<Box<Config>>, // for multiple rounds
}

pub fn paste_by_prefix(dest: &mut Vec<RingElement>, src: &Vec<RingElement>, prefix: &Prefix) {
    assert_eq!(src.len(), 1 << dest.len().ilog2() as usize - prefix.length);
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

pub fn slice_by_prefix(src: &Vec<RingElement>, prefix: &Prefix) -> Vec<RingElement> {
    let start = prefix.prefix << (src.len().ilog2() as usize - prefix.length);
    let length = 1 << (src.len().ilog2() as usize - prefix.length);
    src[start..start + length].to_vec()
}
