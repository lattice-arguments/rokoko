use std::sync::LazyLock;

use crate::{
    common::{
        decomposition::decompose,
        matrix::VerticallyAlignedMatrix,
        ring_arithmetic::{Representation, RingElement},
        sampling::sample_random_short_vector,
    },
    protocol::{
        config::{Config, IntermediateConfig, SimpleConfig},
        config_generator::{AuxConfig, AuxProjection, AuxRecursionConfig, AuxSumcheckConfig},
    },
};

pub static DECOMP_8_LAST_LEVEL: AuxRecursionConfig = AuxRecursionConfig {
    decomposition_base_log: 7,
    decomposition_chunks: 8,
    rank: 1,
    next: None,
};
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SizeConfig {
    Small,
    Medium,
    Large,
}

impl SizeConfig {
    #[inline(always)]
    pub fn pick<T>(self, small: T, medium: T, large: T) -> T {
        match self {
            SizeConfig::Small => small,
            SizeConfig::Medium => medium,
            SizeConfig::Large => large,
        }
    }
}

#[inline(always)]
#[allow(unreachable_code)]
pub fn compiled_size() -> SizeConfig {
    #[cfg(feature = "p-30")]
    {
        return SizeConfig::Large;
    }
    #[cfg(feature = "p-26")]
    {
        return SizeConfig::Small;
    }
    SizeConfig::Medium
}

pub fn p_exact_norm_root_aux(size: SizeConfig, nof_openings: usize) -> AuxSumcheckConfig {
    AuxSumcheckConfig {
        witness_height: size.pick(2usize.pow(13), 2usize.pow(14), 2usize.pow(15)),
        witness_width: size.pick(2usize.pow(7), 2usize.pow(8), 2usize.pow(9)),
        projection_ratio: 2usize.pow(5),  // no-op
        projection_height: 2usize.pow(8), // no-op
        basic_commitment_rank: 6,
        nof_openings,
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
        projection_recursion: AuxProjection::Coarse(AuxRecursionConfig {
            decomposition_base_log: 8,
            decomposition_chunks: 2,
            rank: 2,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        }),

        witness_decomposition_chunks: 4,
        witness_decomposition_base_log: size.pick(4, 4, 7),

        next: Some(Box::new(AuxConfig::Sumcheck(p_int(size)))),
    }
}

pub fn p_int(size: SizeConfig) -> AuxSumcheckConfig {
    AuxSumcheckConfig {
        witness_height: size.pick(2usize.pow(14), 2usize.pow(15), 2usize.pow(16)),
        witness_width: size.pick(2usize.pow(3), 2usize.pow(4), 2usize.pow(5)),
        projection_ratio: 2usize.pow(6),
        projection_height: 2usize.pow(8),
        basic_commitment_rank: size.pick(5, 5, 6),
        nof_openings: 2,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: size.pick(2, 2, 4), // TODO: Add support for non-power-of-two ranks
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        },
        projection_recursion: AuxProjection::Coarse(AuxRecursionConfig {
            decomposition_base_log: 9,
            decomposition_chunks: 2,
            rank: 2,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        }),

        witness_decomposition_chunks: 2,
        witness_decomposition_base_log: 7,

        next: Some(Box::new(AuxConfig::Sumcheck(p_1(size)))),
    }
}

pub fn p_root_aux(size: SizeConfig, nof_openings: usize) -> AuxSumcheckConfig {
    AuxSumcheckConfig {
        witness_height: size.pick(2usize.pow(13), 2usize.pow(14), 2usize.pow(15)),
        witness_width: size.pick(2usize.pow(7), 2usize.pow(8), 2usize.pow(9)),
        projection_ratio: 1,              // no-op
        projection_height: 2usize.pow(8), // no-op,
        basic_commitment_rank: 6,
        nof_openings,
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
        witness_decomposition_base_log: size.pick(6, 6, 7),

        next: Some(Box::new(AuxConfig::Sumcheck(p_1(size)))),
    }
}

pub fn p_1(size: SizeConfig) -> AuxSumcheckConfig {
    AuxSumcheckConfig {
        witness_height: size.pick(2usize.pow(13), 2usize.pow(13), 2usize.pow(14)),
        witness_width: size.pick(2usize.pow(3), 2usize.pow(4), 2usize.pow(4)),
        projection_ratio: 2usize.pow(5),
        projection_height: 2usize.pow(8),
        basic_commitment_rank: size.pick(5, 5, 6),
        nof_openings: 2,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: size.pick(2, 2, 4), // TODO: Add support for non-power-of-two ranks
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        },
        projection_recursion: AuxProjection::Coarse(AuxRecursionConfig {
            decomposition_base_log: 10,
            decomposition_chunks: 2,
            rank: 2,
            next: Some(Box::new(DECOMP_8_LAST_LEVEL.clone())),
        }),

        witness_decomposition_chunks: 2,
        // the base-2^6 window measured 2082 against its 2080 cap at p-28
        // (transcript-dependent); base 2^7, already the p-30 value, restores
        // margin at unchanged composed geometry
        witness_decomposition_base_log: 7,

        next: Some(Box::new(AuxConfig::Sumcheck(p_2(size)))),
    }
}

pub fn p_2(size: SizeConfig) -> AuxSumcheckConfig {
    AuxSumcheckConfig {
        witness_height: size.pick(2usize.pow(10), 2usize.pow(10), 2usize.pow(11)),
        witness_width: 2usize.pow(5),
        projection_ratio: size.pick(2usize.pow(5), 2usize.pow(5), 2usize.pow(8)),
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
        projection_recursion: AuxProjection::Fine {
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
    }
}

pub static P_EN_SMALL: LazyLock<Config> =
    LazyLock::new(|| p_exact_norm_root_aux(SizeConfig::Small, 1).generate_config());
pub static P_EN_MEDIUM: LazyLock<Config> =
    LazyLock::new(|| p_exact_norm_root_aux(SizeConfig::Medium, 1).generate_config());
pub static P_EN_LARGE: LazyLock<Config> =
    LazyLock::new(|| p_exact_norm_root_aux(SizeConfig::Large, 1).generate_config()); // never executed, OOM for 64GiB RAM

pub static P_EN: LazyLock<Config> = LazyLock::new(|| match compiled_size() {
    SizeConfig::Small => P_EN_SMALL.clone(),
    SizeConfig::Medium => P_EN_MEDIUM.clone(),
    SizeConfig::Large => P_EN_LARGE.clone(),
});

pub static P_SMALL: LazyLock<Config> = LazyLock::new(|| p_root_aux(SizeConfig::Small, 1).generate_config());
pub static P_MEDIUM: LazyLock<Config> = LazyLock::new(|| p_root_aux(SizeConfig::Medium, 1).generate_config());
pub static P_LARGE: LazyLock<Config> = LazyLock::new(|| p_root_aux(SizeConfig::Large, 1).generate_config()); 

pub static P: LazyLock<Config> = LazyLock::new(|| match compiled_size() {
    SizeConfig::Small => P_SMALL.clone(),
    SizeConfig::Medium => P_MEDIUM.clone(),
    SizeConfig::Large => P_LARGE.clone(),
});

pub static P_3: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
    witness_height: 2usize.pow(8),
    witness_width: 2usize.pow(5),
    projection_ratio: 2usize.pow(5),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 4,
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
    projection_recursion: AuxProjection::Fine {
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
    projection_ratio: 2usize.pow(5),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 4,
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
    projection_recursion: AuxProjection::Fine {
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
    witness_decomposition_base_log: 7,

    next: Some(Box::new(AuxConfig::Sumcheck(P_5.clone()))),
});

pub static P_5: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
    witness_height: 2usize.pow(8),
    witness_width: 2usize.pow(3),
    projection_ratio: 2usize.pow(6),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 3,
    nof_openings: 2,
    commitment_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        next: None,
    },
    opening_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        next: None,
    },
    projection_recursion: AuxProjection::Fine {
        nof_batches: 2,
        recursion_constant_term: AuxRecursionConfig {
            decomposition_base_log: 10,
            decomposition_chunks: 2,
            rank: 2,
            next: None,
        },
        recursion_batched_projection: AuxRecursionConfig {
            decomposition_base_log: 13,
            decomposition_chunks: 4,
            rank: 2,
            next: None,
        },
    },

    witness_decomposition_chunks: 2,
    witness_decomposition_base_log: 7,

    next: Some(Box::new(AuxConfig::Sumcheck(P_6.clone()))),
});

pub static P_6: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
    witness_height: 2usize.pow(7),
    witness_width: 2usize.pow(3),
    projection_ratio: 2usize.pow(6),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 4,
    nof_openings: 2,
    commitment_recursion: AuxRecursionConfig {
        decomposition_base_log: 15,
        decomposition_chunks: 4,
        rank: 2,
        next: None,
    },
    opening_recursion: AuxRecursionConfig {
        decomposition_base_log: 15,
        decomposition_chunks: 4,
        rank: 2,
        next: None,
    },
    projection_recursion: AuxProjection::Fine {
        nof_batches: 2,
        recursion_constant_term: AuxRecursionConfig {
            decomposition_base_log: 11,
            decomposition_chunks: 2,
            rank: 2,
            next: None,
        },
        recursion_batched_projection: AuxRecursionConfig {
            decomposition_base_log: 13,
            decomposition_chunks: 4,
            rank: 2,
            next: None,
        },
    },

    witness_decomposition_chunks: 1,
    witness_decomposition_base_log: 17,

    next: Some(Box::new(AuxConfig::Intermediate(P_INTERMEDIATE.clone()))),
});

pub static P_INTERMEDIATE: LazyLock<IntermediateConfig> = LazyLock::new(|| IntermediateConfig {
    witness_height: 2usize.pow(7),
    witness_width: 2usize.pow(2),
    projection_ratio: 2usize.pow(6),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 4,
    nof_openings: 2,
    projection_nof_batches: 2,
    witness_decomposition_base_log: 11,
    witness_decomposition_chunks: 2,
    next: Some(Box::new(Config::Simple(P_LAST.clone()))),
});

pub static P_LAST: LazyLock<SimpleConfig> = LazyLock::new(|| SimpleConfig {
    witness_height: 2usize.pow(5),
    witness_width: 2usize.pow(3),
    projection_ratio: 2usize.pow(4),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 4,
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
        decomposition_base_log: 18, // change to 8 for EN sets
        decomposition_chunks: 2,
        initial_norm_log: 31, // change to 15 for EN sets
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

/// Sizing rule for targets between compiled parameter sets: keep the compiled
/// set's height and drop column bits (p27 = p28 with one column-bit fewer).
/// Returns the number of witness columns to use; remaining columns stay zero
/// (`used_cols` on the witness matrix).
pub fn witness_cols_for_target(
    witness_height: usize,
    witness_width: usize,
    target_log2_zq_coeffs: usize,
) -> usize {
    use crate::common::config::DEGREE;
    let full_log2 = (witness_height * witness_width * DEGREE).ilog2() as usize;
    assert!(
        target_log2_zq_coeffs <= full_log2,
        "target 2^{} exceeds the compiled parameter set's capacity 2^{}",
        target_log2_zq_coeffs,
        full_log2
    );
    let drop = full_log2 - target_log2_zq_coeffs;
    assert!(
        drop < witness_width.ilog2() as usize,
        "target 2^{} too small for this parameter set; compile a smaller p-XX feature",
        target_log2_zq_coeffs
    );
    witness_width >> drop
}

#[cfg(test)]
mod tests {
    use crate::protocol::config::Config;

    fn assert_chain_dims(mut config: &Config) {
        while let Config::Sumcheck(sc) = config {
            let Some(next) = sc.next.as_deref() else { break };
            let (h, w) = match next {
                Config::Sumcheck(n) => (n.witness_height, n.witness_width),
                Config::Intermediate(n) => (n.witness_height, n.witness_width),
                Config::Simple(n) => (n.witness_height, n.witness_width),
            };
            assert_eq!(
                sc.composed_witness_length,
                h * w,
                "composed 2^{} != next round witness {}x{} = 2^{}",
                sc.composed_witness_length.ilog2(),
                h,
                w,
                (h * w).ilog2(),
            );
            config = next;
        }
    }

    #[test]
    fn test_p_snark_chain_dims() {
        assert_chain_dims(&super::P_EN_MEDIUM);
    }

    #[test]
    fn test_witness_cols_for_target() {
        // p-28-shaped set: 2^13 x 2^8 ring elements = 2^28 Zq coefficients
        assert_eq!(super::witness_cols_for_target(1 << 13, 1 << 8, 28), 1 << 8);
        // p27 rule: one column-bit fewer
        assert_eq!(super::witness_cols_for_target(1 << 13, 1 << 8, 27), 1 << 7);
        assert_eq!(super::witness_cols_for_target(1 << 13, 1 << 8, 25), 1 << 5);
    }
}
