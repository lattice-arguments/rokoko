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

/// Slack over each round's measured max norm; bump if a future witness gets closer.
pub const NORM_MARGIN: f64 = 1.02;

// Per-round `[primary, secondary]` L2-norm maxima over three (deterministic)
// runs, in chain order; `assign_norm_bounds` scales each by NORM_MARGIN onto the
// round config. The two norms per round are (norm claim, most-inner) for a
// sumcheck round, (norm claim, projection image) for the intermediate round, and
// (folded witness, projection image) for the simple round. Standard chain has 9
// rounds (root..P_6, intermediate, simple); the exact-norm chain has 10.
const NB_P_26: [[f64; 2]; 9] = [
    [46889.51181234456, 2242.093664412796],
    [136249.25466218154, 2703.859463803546],
    [88564.70651450272, 3127.9992007671613],
    [51809.033633141626, 3129.3796509851595],
    [35428.87688030768, 3111.4773018616092],
    [195669.4144366973, 195560.31913197524],
    [1602884.0647417393, 1299990.815144861],
    [73660.94992599539, 17077273.72989793],
    [350510.5948127674, 836487.5791151952],
];
const NB_P_28: [[f64; 2]; 9] = [
    [66427.98663966867, 2160.0013888884423],
    [181558.43011548652, 2705.682169065687],
    [95004.44916949942, 3133.253580545309],
    [52846.942182116836, 3145.1373578907487],
    [35438.889895142034, 3128.613750529138],
    [193799.4028705971, 193690.07834166416],
    [1583993.8583391036, 1296847.5245818223],
    [73668.6367459043, 18268958.675824028],
    [349498.9501185948, 809458.9433127538],
];
const NB_P_30: [[f64; 2]; 9] = [
    [146947.061954297, 2201.3457247783685],
    [250426.82932745045, 3131.986111080316],
    [59039.09997620221, 3128.7861863668472],
    [56233.940703102075, 3119.414528401123],
    [35479.55817086791, 3145.0324322652064],
    [193916.4559804041, 193806.1362831425],
    [1612935.666007792, 1308359.9714466962],
    [73796.00136863785, 18564103.121726133],
    [347114.5618581854, 851538.0815265985],
];
const NB_P_EN_26: [[f64; 2]; 10] = [
    [160194.58070733852, 2703.3730782117364],
    [89390.48689877463, 2704.1401590893915],
    [124864.64003071486, 2718.6763323352784],
    [87340.9749716592, 3136.444643222641],
    [52061.72332914077, 3103.024008930643],
    [35395.67282875126, 3073.2464919039603],
    [195646.41070308446, 195537.5162468829],
    [1590113.0789748884, 1299828.2178838095],
    [73439.8848855307, 17699615.97419359],
    [344384.17449412507, 850119.7508974839],
];
const NB_P_EN_28: [[f64; 2]; 10] = [
    [316140.5607273448, 2688.3418681410294],
    [147581.0541092589, 2694.3485297934267],
    [181869.85245773967, 2697.6367435220036],
    [95340.00619362263, 3156.4337471266526],
    [52929.56034202438, 3134.5771644673227],
    [35536.13603080672, 3132.0439013525975],
    [195785.91413837718, 195676.68320727433],
    [1580382.2873349346, 1295429.6467635748],
    [73623.80077800927, 18362161.429342028],
    [345853.3582228167, 836902.1092481485],
];

/// Scale each round's measured maxima by [`NORM_MARGIN`] onto its config,
/// walking the chain root-first. Rounds not covered keep their INFINITY default
/// (e.g. P_EN_30, which OOMs and is never run).
fn assign_norm_bounds(config: &mut Config, bounds: &[[f64; 2]]) {
    fn rec(config: &mut Config, bounds: &[[f64; 2]], i: &mut usize) {
        match config {
            Config::Sumcheck(c) => {
                c.norm_bound = bounds[*i][0] * NORM_MARGIN;
                c.most_inner_norm_bound = bounds[*i][1] * NORM_MARGIN;
                *i += 1;
                if let Some(next) = c.next.as_deref_mut() {
                    rec(next, bounds, i);
                }
            }
            Config::Intermediate(c) => {
                c.norm_bound = bounds[*i][0] * NORM_MARGIN;
                c.projection_norm_bound = bounds[*i][1] * NORM_MARGIN;
                *i += 1;
                if let Some(next) = c.next.as_deref_mut() {
                    rec(next, bounds, i);
                }
            }
            Config::Simple(c) => {
                c.witness_norm_bound = bounds[*i][0] * NORM_MARGIN;
                c.projection_norm_bound = bounds[*i][1] * NORM_MARGIN;
                *i += 1;
            }
        }
    }
    let mut i = 0;
    rec(config, bounds, &mut i);
    assert_eq!(i, bounds.len(), "norm-bound array length must match chain length");
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

pub static P_EN_2_SMALL: LazyLock<Config> = LazyLock::new(|| {
    let mut c = p_exact_norm_root_aux(SizeConfig::Small, 2).generate_config();
    assign_norm_bounds(&mut c, &NB_P_EN_26);
    c
});
pub static P_EN_2_MEDIUM: LazyLock<Config> = LazyLock::new(|| {
    let mut c = p_exact_norm_root_aux(SizeConfig::Medium, 2).generate_config();
    assign_norm_bounds(&mut c, &NB_P_EN_28);
    c
});
pub static P_EN_2_LARGE: LazyLock<Config> =
    LazyLock::new(|| p_exact_norm_root_aux(SizeConfig::Large, 2).generate_config()); // never executed, OOM for 64GiB RAM

pub static P_EN_TWO_EVALS: LazyLock<Config> = LazyLock::new(|| match compiled_size() {
    SizeConfig::Small => P_EN_2_SMALL.clone(),
    SizeConfig::Medium => P_EN_2_MEDIUM.clone(),
    SizeConfig::Large => P_EN_2_LARGE.clone(),
});

pub static P_SMALL: LazyLock<Config> = LazyLock::new(|| {
    let mut c = p_root_aux(SizeConfig::Small, 1).generate_config();
    assign_norm_bounds(&mut c, &NB_P_26);
    c
});
pub static P_MEDIUM: LazyLock<Config> = LazyLock::new(|| {
    let mut c = p_root_aux(SizeConfig::Medium, 1).generate_config();
    assign_norm_bounds(&mut c, &NB_P_28);
    c
});
pub static P_LARGE: LazyLock<Config> = LazyLock::new(|| {
    let mut c = p_root_aux(SizeConfig::Large, 1).generate_config();
    assign_norm_bounds(&mut c, &NB_P_30);
    c
});

pub static P_2_SMALL: LazyLock<Config> = LazyLock::new(|| p_root_aux(SizeConfig::Small, 2).generate_config());
pub static P_2_MEDIUM: LazyLock<Config> = LazyLock::new(|| p_root_aux(SizeConfig::Medium, 2).generate_config());
pub static P_2_LARGE: LazyLock<Config> = LazyLock::new(|| p_root_aux(SizeConfig::Large, 2).generate_config()); 

pub static P: LazyLock<Config> = LazyLock::new(|| match compiled_size() {
    SizeConfig::Small => P_SMALL.clone(),
    SizeConfig::Medium => P_MEDIUM.clone(),
    SizeConfig::Large => P_LARGE.clone(),
});

pub static P_TWO_EVALS: LazyLock<Config> = LazyLock::new(|| match compiled_size() {
    SizeConfig::Small => P_2_SMALL.clone(),
    SizeConfig::Medium => P_2_MEDIUM.clone(),
    SizeConfig::Large => P_2_LARGE.clone(),
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
    norm_bound: f64::INFINITY,
    projection_norm_bound: f64::INFINITY,
    next: Some(Box::new(Config::Simple(P_LAST.clone()))),
});

pub static P_LAST: LazyLock<SimpleConfig> = LazyLock::new(|| SimpleConfig {
    witness_height: 2usize.pow(5),
    witness_width: 2usize.pow(3),
    projection_ratio: 2usize.pow(4),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 4,
    projection_nof_batches: 2,
    witness_norm_bound: f64::INFINITY,
    projection_norm_bound: f64::INFINITY,
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
        decomposition_base_log: 16, // change to 8 for EN sets
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
