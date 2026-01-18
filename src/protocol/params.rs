use std::sync::LazyLock;

use crate::protocol::{
    config::Config,
    config_generator::{AuxProjection, AuxRecursionConfig, AuxSumcheckConfig},
};

pub static P28: LazyLock<Config> = LazyLock::new(|| {
    AuxSumcheckConfig {
        witness_height: 2usize.pow(15),
        witness_width: 2usize.pow(6),
        projection_ratio: 2usize.pow(7),
        projection_height: 2usize.pow(8),
        basic_commitment_rank: 4,
        nof_openings: 1,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 15,
            decomposition_chunks: 4,
            rank: 2,
            next: Some(Box::new(AuxRecursionConfig {
                decomposition_base_log: 7,
                decomposition_chunks: 8,
                rank: 2,
                next: None,
            })),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 15,
            decomposition_chunks: 4,
            rank: 2,
            next: None,
        },
        projection_recursion: AuxProjection::Type0(AuxRecursionConfig {
            decomposition_base_log: 20,
            decomposition_chunks: 1,
            rank: 2,
            next: None,
        }),

        witness_decomposition_chunks: 2,
        witness_decomposition_base_log: 10,

        next: None,
    }
    .generate_config()
});
