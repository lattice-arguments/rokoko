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
    diag_blocks: 1,
    next: None,
};

// This config is a bit special as I cannot just handle it in the first round
// Returns `if_p30` if the "p-30" feature is enabled at runtime, otherwise `if_not_p30`.
// I didn't manage to make it a macro that works inside expressions, so a function will do.
#[inline(always)]
#[allow(unused_variables, unreachable_code)]
fn cfg_p30<T>(if_p30: T, if_not_p30: T) -> T {
    #[cfg(feature = "p-30")]
    {
        return if_p30;
    }
    if_not_p30
}

#[inline(always)]
#[allow(unused_variables, unreachable_code)]
fn cfg_p26<T>(if_p26: T, if_not_p26: T) -> T {
    #[cfg(feature = "p-26")]
    {
        return if_p26;
    }
    if_not_p26
}

#[inline(always)]
#[allow(unused_variables, unreachable_code)]
fn per_config<T>(p26_value: T, p28_value: T, p30_value: T) -> T {
    #[cfg(feature = "p-30")]
    {
        return p30_value;
    }
    #[cfg(feature = "p-26")]
    {
        return p26_value;
    }
    p28_value
}

pub static P: LazyLock<Config> = LazyLock::new(|| {
    AuxSumcheckConfig {
        witness_height: per_config(
            2usize.pow(13), // p-26
            2usize.pow(14), // p-28
            2usize.pow(15), // p-30
        ),
        witness_width: per_config(
            2usize.pow(7), // p-26
            2usize.pow(8), // p-28
            2usize.pow(9), // p-30
        ),
        projection_ratio: 1,              // no-op
        projection_height: 2usize.pow(8), // no-op,
        basic_commitment_rank: 16,
        basic_commitment_diag_blocks: 2,
        nof_openings: 1,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 4,
            diag_blocks: 2,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            diag_blocks: 1,
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
        witness_height: cfg_p30(2usize.pow(14), 2usize.pow(13)),
        witness_width: cfg_p26(2usize.pow(3), 2usize.pow(4)),
        projection_ratio: 2usize.pow(5),
        projection_height: 2usize.pow(8),
        basic_commitment_rank: cfg_p30(7, 6),
        basic_commitment_diag_blocks: 1,
        nof_openings: 2,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: cfg_p30(4, 2), // TODO: Add support for non-power-of-two ranks,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: cfg_p30(4, 2),
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        },
        projection_recursion: AuxProjection::Type0(AuxRecursionConfig {
            decomposition_base_log: 10,
            decomposition_chunks: 2,
            rank: cfg_p30(4, 2),
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        }),

        witness_decomposition_chunks: 2,
        witness_decomposition_base_log: 8,

        next: Some(Box::new(AuxConfig::Sumcheck(P_2.clone()))),
        // next: None,
    }
});

pub static P_2: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
    witness_height: cfg_p30(2usize.pow(11), 2usize.pow(10)),
    witness_width: 2usize.pow(5),
    projection_ratio: cfg_p30(2usize.pow(8), 2usize.pow(5)),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 6,
    basic_commitment_diag_blocks: 1,
    nof_openings: 2,
    commitment_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        diag_blocks: 1,
        next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
    },
    opening_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        diag_blocks: 1,
        next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
    },
    projection_recursion: AuxProjection::Type1 {
        nof_batches: 2,
        recursion_constant_term: AuxRecursionConfig {
            decomposition_base_log: 10,
            decomposition_chunks: 2,
            rank: 2,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        },
        recursion_batched_projection: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        },
    },

    witness_decomposition_chunks: 2,
    witness_decomposition_base_log: cfg_p26(10, 9),

    next: Some(Box::new(AuxConfig::Sumcheck(P_3.clone()))),
});

pub static P_3: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
    witness_height: 2usize.pow(8),
    witness_width: 2usize.pow(5),
    projection_ratio: 2usize.pow(6),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 5,
    basic_commitment_diag_blocks: 1,
    nof_openings: 2,
    commitment_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        diag_blocks: 1,
        next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
    },
    opening_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        diag_blocks: 1,
        next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
    },
    projection_recursion: AuxProjection::Type1 {
        nof_batches: 2,
        recursion_constant_term: AuxRecursionConfig {
            decomposition_base_log: 10,
            decomposition_chunks: 2,
            rank: 2,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        },
        recursion_batched_projection: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        },
    },

    witness_decomposition_chunks: 2,
    witness_decomposition_base_log: cfg_p30(8, 9),

    next: Some(Box::new(AuxConfig::Sumcheck(P_4.clone()))),
});

pub static P_4: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
    witness_height: 2usize.pow(9),
    witness_width: 2usize.pow(3),
    projection_ratio: 2usize.pow(6),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 5,
    basic_commitment_diag_blocks: 1,
    nof_openings: 2,
    commitment_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        diag_blocks: 1,
        next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
    },
    opening_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        diag_blocks: 1,
        next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
    },
    projection_recursion: AuxProjection::Type1 {
        nof_batches: 2,
        recursion_constant_term: AuxRecursionConfig {
            decomposition_base_log: 10,
            decomposition_chunks: 2,
            rank: 2,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        },
        recursion_batched_projection: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            diag_blocks: 1,
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
//         rank: 2, diag_blocks: 1,
//         next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
//     },
//     opening_recursion: AuxRecursionConfig {
//         decomposition_base_log: 7,
//         decomposition_chunks: 8,
//         rank: 2, diag_blocks: 1,
//         next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
//     },
//     projection_recursion: AuxProjection::Type1 {
//         nof_batches: 2,
//         recursion_constant_term: AuxRecursionConfig {
//             decomposition_base_log: 10,
//             decomposition_chunks: 2,
//             rank: 2, diag_blocks: 1,
//             next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
//         },
//         recursion_batched_projection: AuxRecursionConfig {
//             decomposition_base_log: 7,
//             decomposition_chunks: 8,
//             rank: 2, diag_blocks: 1,
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
    basic_commitment_diag_blocks: 1,
    nof_openings: 2,
    commitment_recursion: AuxRecursionConfig {
        decomposition_base_log: 13,
        decomposition_chunks: 4,
        rank: 3,
        diag_blocks: 1,
        next: None,
    },
    opening_recursion: AuxRecursionConfig {
        decomposition_base_log: 13,
        decomposition_chunks: 4,
        rank: 3,
        diag_blocks: 1,
        next: None,
    },
    projection_recursion: AuxProjection::Type1 {
        nof_batches: 2,
        recursion_constant_term: AuxRecursionConfig {
            decomposition_base_log: 10,
            decomposition_chunks: 2,
            rank: 3,
            diag_blocks: 1,
            next: None,
        },
        recursion_batched_projection: AuxRecursionConfig {
            decomposition_base_log: 13,
            decomposition_chunks: 4,
            rank: 3,
            diag_blocks: 1,
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
        initial_norm_log: 31,
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
