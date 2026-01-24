use std::sync::LazyLock;

use crate::{
    common::{
        decomposition::decompose,
        matrix::VerticallyAlignedMatrix,
        ring_arithmetic::{Representation, RingElement},
        sampling::sample_random_short_vector,
    },
    protocol::{
        config::{Config, SimpleConfig},
        config_generator::{AuxConfig, AuxProjection, AuxRecursionConfig, AuxSumcheckConfig},
    },
};

pub static DECOMP_8_LAST_LEVEL: AuxRecursionConfig = AuxRecursionConfig {
    decomposition_base_log: 7,
    decomposition_chunks: 8,
    rank: 1,
    next: None,
};

pub static P: LazyLock<Config> = LazyLock::new(|| {
    AuxSumcheckConfig {
        witness_height: 2usize.pow(15), // 2^29 total size, but ths is 2^28 of 2^32 els (as we sample from 2^16 els)
        witness_width: {
            #[cfg(feature = "p-26")]
            {
                2usize.pow(5) // 2^27 total size, but ths is 2^26 of 2^32 els (as we sample from 2^16 els)
            }
            #[cfg(feature = "p-30")]
            {
                2usize.pow(9) // 2^31 total size, but ths is 2^30 of 2^32 els (as we sample from 2^16 els)
            }
            #[cfg(not(any(feature = "p-26", feature = "p-30")))]
            {
                2usize.pow(7) // 2^29 total size, but ths is 2^28 of 2^32 els (as we sample from 2^16 els)
            }
        },
        projection_ratio: 1,              // no-op
        projection_height: 2usize.pow(8), // no-op, TODO: make sure this is not used
        basic_commitment_rank: 8,
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
        projection_recursion: AuxProjection::Skip,

        witness_decomposition_chunks: 4,
        witness_decomposition_base_log: 7,

        next: Some(Box::new(AuxConfig::Sumcheck(P_1.clone()))),
    }
    .generate_config()
});

pub static P_1: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| {
    AuxSumcheckConfig {
        witness_height: 2usize.pow(14),
        witness_width: 2usize.pow(4),
        projection_ratio: 2usize.pow(6),
        projection_height: 2usize.pow(8),
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
        projection_recursion: AuxProjection::Type0(AuxRecursionConfig {
            decomposition_base_log: 10,
            decomposition_chunks: 2,
            rank: 2,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        }),

        witness_decomposition_chunks: 2,
        witness_decomposition_base_log: 8,

        next: Some(Box::new(AuxConfig::Sumcheck(P_2.clone()))),
        // next: None,
    }
});

pub static P_2: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
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

    next: Some(Box::new(AuxConfig::Sumcheck(P_3.clone()))),
    // To stop here:
    // next: None,
});

pub static P_3: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
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

    next: Some(Box::new(AuxConfig::Sumcheck(P_4.clone()))),
});

pub static P_4: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
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

    next: Some(Box::new(AuxConfig::Sumcheck(P_5.clone()))),
});

// pub static P_5: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig{
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

//     // next: Some(Box::new(AuxConfig::Simple(P_5.clone()))),
//     next: None,
// });

pub static P_5: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
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

    next: Some(Box::new(AuxConfig::Simple(P_LAST.clone()))),
});

pub static P_LAST: LazyLock<SimpleConfig> = LazyLock::new(|| SimpleConfig {
    witness_height: 2usize.pow(7),
    witness_width: 2usize.pow(3),
    projection_ratio: 2usize.pow(6),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 7,
    projection_nof_batches: 2,
});

// 2^28 Z_q elements of norm 2^32
// => 2^29 Z_q elements of norm 2^16 (signed 2^15)
// => 2^22 R_q elements
// => height 2^15, width 2^7

pub struct InitialWitnessParams {
    pub height: usize,
    pub width: usize,
    pub decomposition_base_log: usize,
    pub decomposition_chunks: usize,
    pub initial_norm_log: usize,
}

pub static WITNESS_CONFIG: LazyLock<InitialWitnessParams> = LazyLock::new(|| match &*P {
    Config::Sumcheck(config) => InitialWitnessParams {
        height: config.witness_height / 2,
        width: config.witness_width,
        decomposition_base_log: 16,
        decomposition_chunks: 2,
        initial_norm_log: 32,
    },
    _ => panic!("Expected sumcheck config at the top level."),
});

pub fn witness_sampler() -> VerticallyAlignedMatrix<RingElement> {
    let config = &*WITNESS_CONFIG;
    VerticallyAlignedMatrix {
        height: config.height,
        width: config.width,
        data: sample_random_short_vector(
            config.height * config.width,
            2u64.pow(config.initial_norm_log as u32 - 1),
            Representation::IncompleteNTT,
        ),
        used_cols: config.width,
    }
}

pub fn decompose_witness(
    witness: &VerticallyAlignedMatrix<RingElement>,
) -> VerticallyAlignedMatrix<RingElement> {
    let config = &*WITNESS_CONFIG;
    let decomposed_data = decompose(
        &witness.data,
        config.decomposition_base_log as u64,
        config.decomposition_chunks,
    );
    VerticallyAlignedMatrix {
        height: witness.height * config.decomposition_chunks,
        width: witness.width,
        data: decomposed_data,
        used_cols: witness.width,
    }
}
