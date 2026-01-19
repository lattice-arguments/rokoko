use std::sync::LazyLock;

use crate::protocol::{
    config::{Config, SimpleConfig},
    config_generator::{AuxConfig, AuxProjection, AuxRecursionConfig, AuxSumcheckConfig},
};

pub static DECOMP_8_LAST_LEVEL: AuxRecursionConfig = AuxRecursionConfig {
    decomposition_base_log: 7,
    decomposition_chunks: 8,
    rank: 1,
    next: None,
};

pub static P28: LazyLock<Config> = LazyLock::new(|| {
    AuxSumcheckConfig {
        witness_height: 2usize.pow(15),
        witness_width: 2usize.pow(6),
        projection_ratio: 2usize.pow(7),
        projection_height: 2usize.pow(8),
        basic_commitment_rank: 5,
        nof_openings: 1,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        },
        projection_recursion: AuxProjection::Type0(AuxRecursionConfig {
            decomposition_base_log: 20,
            decomposition_chunks: 1,
            rank: 2,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        }),

        witness_decomposition_chunks: 1,
        witness_decomposition_base_log: 9,

        next: Some(Box::new(AuxConfig::Sumcheck(P28_2.clone()))),
    }
    .generate_config()
});

pub static P28_2: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
    witness_height: 2usize.pow(11),
    witness_width: 2usize.pow(5),
    projection_ratio: 2usize.pow(8),
    projection_height: 2usize.pow(8), // this costs a lot a verification time
    basic_commitment_rank: 6,
    nof_openings: 2,
    commitment_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
    },
    opening_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
    },
    projection_recursion: AuxProjection::Type1 {
        nof_batches: 2,
        recursion_constant_term: AuxRecursionConfig {
            decomposition_base_log: 10,
            decomposition_chunks: 2,
            rank: 2,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        },
        recursion_batched_projection: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        },
    },

    witness_decomposition_chunks: 2,
    witness_decomposition_base_log: 8,

    next: Some(Box::new(AuxConfig::Sumcheck(P28_3.clone()))),
    // To stop here:
    // next: None,
});

pub static P28_3: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
    witness_height: 2usize.pow(8),
    witness_width: 2usize.pow(5),
    projection_ratio: 2usize.pow(6),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 5,
    nof_openings: 2,
    commitment_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
    },
    opening_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
    },
    projection_recursion: AuxProjection::Type1 {
        nof_batches: 2,
        recursion_constant_term: AuxRecursionConfig {
            decomposition_base_log: 10,
            decomposition_chunks: 2,
            rank: 2,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        },
        recursion_batched_projection: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        },
    },

    witness_decomposition_chunks: 2,
    witness_decomposition_base_log: 8,

    next: Some(Box::new(AuxConfig::Sumcheck(P28_4.clone()))),
});

pub static P28_4: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
    witness_height: 2usize.pow(9),
    witness_width: 2usize.pow(3),
    projection_ratio: 2usize.pow(6),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 5,
    nof_openings: 2,
    commitment_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
    },
    opening_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
    },
    projection_recursion: AuxProjection::Type1 {
        nof_batches: 2,
        recursion_constant_term: AuxRecursionConfig {
            decomposition_base_log: 10,
            decomposition_chunks: 2,
            rank: 2,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        },
        recursion_batched_projection: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        },
    },

    witness_decomposition_chunks: 2,
    witness_decomposition_base_log: 8,

    next: Some(Box::new(AuxConfig::Sumcheck(P28_5.clone()))),
});

// pub static P28_5: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig{
//     witness_height: 2usize.pow(9),
//     witness_width: 2usize.pow(3),
//     projection_ratio: 2usize.pow(6),
//     projection_height: 2usize.pow(8),
//     basic_commitment_rank: 6,
//     nof_openings: 2,
//     commitment_recursion: AuxRecursionConfig {
//         decomposition_base_log: 7,
//         decomposition_chunks: 5,
//         rank: 2,
//         next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
//     },
//     opening_recursion: AuxRecursionConfig {
//         decomposition_base_log: 7,
//         decomposition_chunks: 8,
//         rank: 2,
//         next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
//     },
//     projection_recursion: AuxProjection::Type1 {
//         nof_batches: 2,
//         recursion_constant_term: AuxRecursionConfig {
//             decomposition_base_log: 10,
//             decomposition_chunks: 2,
//             rank: 2,
//             next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
//         },
//         recursion_batched_projection: AuxRecursionConfig {
//             decomposition_base_log: 7,
//             decomposition_chunks: 8,
//             rank: 2,
//             next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
//         },
//     },

//     witness_decomposition_chunks: 2,
//     witness_decomposition_base_log: 8,

//     // next: Some(Box::new(AuxConfig::Simple(P28_5.clone()))),
//     next: None,
// });

pub static P28_5: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
    witness_height: 2usize.pow(8),
    witness_width: 2usize.pow(3),
    projection_ratio: 2usize.pow(7),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 5,
    nof_openings: 2,
    commitment_recursion: AuxRecursionConfig {
        decomposition_base_log: 13,
        decomposition_chunks: 4,
        rank: 3,
        next: None,
    },
    opening_recursion: AuxRecursionConfig {
        decomposition_base_log: 13,
        decomposition_chunks: 4,
        rank: 3,
        next: None,
    },
    projection_recursion: AuxProjection::Type1 {
        nof_batches: 2,
        recursion_constant_term: AuxRecursionConfig {
            decomposition_base_log: 10,
            decomposition_chunks: 2,
            rank: 3, // TODO: can it be 3?
            next: None,
        },
        recursion_batched_projection: AuxRecursionConfig {
            decomposition_base_log: 13,
            decomposition_chunks: 4,
            rank: 3,
            next: None,
        },
    },

    witness_decomposition_chunks: 2,
    witness_decomposition_base_log: 8,

    next: Some(Box::new(AuxConfig::Simple(P28_LAST.clone()))),
});

pub static P28_LAST: LazyLock<SimpleConfig> = LazyLock::new(|| SimpleConfig {
    witness_height: 2usize.pow(7),
    witness_width: 2usize.pow(3),
    projection_ratio: 2usize.pow(6),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 6,
    projection_nof_batches: 2,
});
