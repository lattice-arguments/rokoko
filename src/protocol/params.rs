use std::sync::LazyLock;

use crate::{
    common::{
        matrix::VerticallyAlignedMatrix,
        ring_arithmetic::{Representation, RingElement},
        sampling::sample_random_short_vector,
    },
    protocol::{
        config::{Config, SimpleConfig},
        config_generator::{AuxConfig, AuxProjection, AuxRecursionConfig, AuxSumcheckConfig},
    },
};

pub static DECOMP_11_LAST_LEVEL: AuxRecursionConfig = AuxRecursionConfig {
    decomposition_base_log: 5,
    decomposition_chunks: 11,
    rank: 1,
    diag_blocks: 1,
    next: None,
};
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SizeConfig {
    Micro,
    Tiny,
    Small,
    Medium,
    NarrowLarge,
    Large,
}

impl SizeConfig {
    #[inline(always)]
    pub fn pick<T>(self, small: T, medium: T, narrow_large: T, large: T) -> T {
        match self {
            // p-24 and p-22 share p-26's tail
            SizeConfig::Micro | SizeConfig::Tiny | SizeConfig::Small => small,
            SizeConfig::Medium => medium,
            SizeConfig::NarrowLarge => narrow_large,
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
    #[cfg(feature = "p-29")]
    {
        return SizeConfig::NarrowLarge;
    }
    #[cfg(feature = "p-26")]
    {
        return SizeConfig::Small;
    }
    #[cfg(feature = "p-24")]
    {
        return SizeConfig::Tiny;
    }
    #[cfg(feature = "p-22")]
    {
        return SizeConfig::Micro;
    }
    SizeConfig::Medium
}

pub const NORM_MARGIN: f64 = 1.85; // verifier accepts norms up to this factor times the expected bound

const NB_P_22: [[f64; 3]; 6] = [
    [31586.04321531901, 664.4343458913003, f64::INFINITY],
    [32242.99688304423, 933.3954146019788, f64::INFINITY],
    [40528.49433423354, 941.2194218140635, f64::INFINITY],
    [21462.925103536098, 934.9283394998785, f64::INFINITY],
    [20031.19666919578, 18862.171296009376, f64::INFINITY],
    [93834.292153775, 230467.703379454, f64::INFINITY],
];

const NB_P_24: [[f64; 3]; 6] = [
    [44646.33795732859, 668.5746031670661, f64::INFINITY],
    [42464.83648855839, 933.2770221107986, f64::INFINITY],
    [44926.57335030127, 934.2628109905692, f64::INFINITY],
    [21945.911623808202, 942.6595355694441, f64::INFINITY],
    [20090.157814213406, 18923.73945603775, f64::INFINITY],
    [94437.21929408977, 230698.74991208775, f64::INFINITY],
];

const NB_P_26: [[f64; 3]; 7] = [
    [52962.016615684115, 939.7074012691397, f64::INFINITY],
    [75752.96866790106, 812.7305826656211, f64::INFINITY],
    [42387.67135618563, 931.4075370105182, f64::INFINITY],
    [46470.407336282304, 940.3568471596301, f64::INFINITY],
    [21745.98323829024, 942.3990662134593, f64::INFINITY],
    [20040.361049641797, 18885.146729639142, f64::INFINITY],
    [93821.23664714722, 227687.86020778533, f64::INFINITY],
];

const NB_P_28: [[f64; 3]; 7] = [
    [75056.30693685908, 932.2210038397548, f64::INFINITY],
    [97065.21574693995, 815.1349581511028, f64::INFINITY],
    [53440.1325410033, 935.6115646997957, f64::INFINITY],
    [49837.24499809354, 936.1751972787999, f64::INFINITY],
    [22030.68394308266, 940.0765926242393, f64::INFINITY],
    [20048.31496660006, 18881.50155575557, f64::INFINITY],
    [93816.50766256437, 234175.65906814483, f64::INFINITY],
];

const NB_P_30: [[f64; 3]; 7] = [
    [159046.0282811237, 943.0524905857574, f64::INFINITY],
    [130226.10125086292, 931.624924527033, f64::INFINITY],
    [47236.84904817424, 936.0571563745453, f64::INFINITY],
    [53478.4027154888, 933.0096462523846, f64::INFINITY],
    [22158.50157388807, 936.5121462106084, f64::INFINITY],
    [20022.482063920048, 18852.281347359527, f64::INFINITY],
    [93772.40590920125, 230425.2675077106, f64::INFINITY],
];

const NB_P_EN_26: [[f64; 3]; 8] = [
    [160205.4703872499, 814.1222266957217, 9635740.525794579],
    [108562.36283353453, 814.8852679978943, 1818077.374859772],
    [76199.9080051938, 808.4689233359561, f64::INFINITY],
    [42512.63592157042, 933.0337614470336, f64::INFINITY],
    [46680.69406082133, 937.4342643620405, f64::INFINITY],
    [21804.699011910256, 931.5680329423075, f64::INFINITY],
    [20062.42091573198, 18899.381656551624, f64::INFINITY],
    [94569.6738071989, 215458.37572023048, f64::INFINITY],
];

const NB_P_EN_28: [[f64; 3]; 8] = [
    [316064.95526552765, 2726.532229774664, 19255784.083067354],
    [146363.3975111264, 2698.7339624349784, 3580972.197575122],
    [97279.68811113654, 2717.1343360238925, f64::INFINITY],
    [53533.04769952856, 3103.0016113434426, f64::INFINITY],
    [38637.24035176425, 3136.247120365358, f64::INFINITY],
    [20909.072169754447, 3142.1613580464004, f64::INFINITY],
    [19882.262069492997, 18697.192837428833, f64::INFINITY],
    [93275.05554005314, 237003.21750980514, f64::INFINITY],
];

const NB_P_EN_29: [[f64; 3]; 8] = [
    [255497.9865713231, 2738.004017528097, f64::INFINITY],
    [180884.3985422734, 3162.853142338417, f64::INFINITY],
    [244952.02035908992, 3160.4202252232217, f64::INFINITY],
    [58704.64349606426, 3168.298597039111, f64::INFINITY],
    [56399.73981322964, 3146.806476413826, f64::INFINITY],
    [35765.94076771922, 3153.750782798159, f64::INFINITY],
    [196535.54675681447, 196424.94941325556, f64::INFINITY],
    [943164.8811432708, 2386396.4914190182, f64::INFINITY],
];

fn assign_norm_bounds(config: &mut Config, bounds: &[[f64; 3]]) {
    fn rec(config: &mut Config, bounds: &[[f64; 3]], i: &mut usize) {
        match config {
            Config::Sumcheck(c) => {
                c.norm_bound = bounds[*i][0] * NORM_MARGIN;
                c.most_inner_norm_bound = bounds[*i][1] * NORM_MARGIN;
                c.projection_norm_bound = bounds[*i][2] * NORM_MARGIN;
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
    assert!(
        i <= bounds.len(),
        "norm-bound array length be at least the number of configs in the chain"
    );
}

pub fn p_exact_norm_root_aux(size: SizeConfig, nof_openings: usize) -> AuxSumcheckConfig {
    AuxSumcheckConfig {
        exact_projection_norm: true,
        witness_height: size.pick(
            2usize.pow(13),
            2usize.pow(14),
            2usize.pow(15),
            2usize.pow(15),
        ),
        witness_width: size.pick(2usize.pow(7), 2usize.pow(8), 2usize.pow(8), 2usize.pow(9)),
        projection_ratio: 2usize.pow(5),
        projection_height: 2usize.pow(8),
        basic_commitment_rank: 6,
        basic_commitment_diag_blocks: 1,
        nof_openings,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        projection_recursion: AuxProjection::Coarse(AuxRecursionConfig {
            decomposition_base_log: 8,
            decomposition_chunks: 2,
            rank: 2,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        }),

        witness_decomposition_chunks: 4,
        witness_decomposition_base_log: size.pick(4, 4, 4, 7),

        next: Some(Box::new(AuxConfig::Sumcheck(p_int(size)))),
    }
}

pub fn p_int(size: SizeConfig) -> AuxSumcheckConfig {
    AuxSumcheckConfig {
        exact_projection_norm: true,
        witness_height: size.pick(
            2usize.pow(14),
            2usize.pow(15),
            2usize.pow(16),
            2usize.pow(16),
        ),
        witness_width: size.pick(2usize.pow(3), 2usize.pow(4), 2usize.pow(4), 2usize.pow(5)),
        projection_ratio: 2usize.pow(5),
        projection_height: 2usize.pow(8),
        basic_commitment_rank: size.pick(5, 5, 6, 6),
        basic_commitment_diag_blocks: 1,
        nof_openings: 2,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: size.pick(2, 2, 4, 4),
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        projection_recursion: AuxProjection::Coarse(AuxRecursionConfig {
            decomposition_base_log: 9,
            decomposition_chunks: 2,
            rank: 2,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        }),

        witness_decomposition_chunks: 2,
        witness_decomposition_base_log: 7,

        next: Some(Box::new(AuxConfig::Sumcheck(p_1(size)))),
    }
}

pub fn p_root_aux(size: SizeConfig, nof_openings: usize) -> AuxSumcheckConfig {
    AuxSumcheckConfig {
        exact_projection_norm: false,
        witness_height: size.pick(
            2usize.pow(13),
            2usize.pow(14),
            2usize.pow(15),
            2usize.pow(15),
        ),
        witness_width: size.pick(2usize.pow(7), 2usize.pow(8), 2usize.pow(8), 2usize.pow(9)),
        projection_ratio: 1,              // no-op
        projection_height: 2usize.pow(8), // no-op,
        basic_commitment_rank: size.pick(10, 10, 10, 12),
        basic_commitment_diag_blocks: 1,
        nof_openings,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 4,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 4,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        projection_recursion: AuxProjection::Skip,

        witness_decomposition_chunks: 4,
        witness_decomposition_base_log: size.pick(6, 6, 6, 7),

        next: Some(Box::new(AuxConfig::Sumcheck(p_1(size)))),
    }
}

/// Root of the p-24 and p-22 chains. Its composed witness is already as short as the one p_1
/// composes to (p-22: half of it, hence the shorter p_2), so the chain skips p_1.
pub fn p_root_aux_short(size: SizeConfig, nof_openings: usize) -> AuxSumcheckConfig {
    let tiny = size == SizeConfig::Tiny;
    AuxSumcheckConfig {
        exact_projection_norm: false,
        witness_height: if tiny { 2usize.pow(11) } else { 2usize.pow(10) },
        witness_width: if tiny { 2usize.pow(7) } else { 2usize.pow(6) },
        projection_ratio: 1,              // no-op
        projection_height: 2usize.pow(8), // no-op,
        basic_commitment_rank: 10,
        basic_commitment_diag_blocks: 1,
        nof_openings,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        projection_recursion: AuxProjection::Skip,

        witness_decomposition_chunks: 4,
        witness_decomposition_base_log: 6,

        next: Some(Box::new(AuxConfig::Sumcheck(p_2(size)))),
    }
}

pub fn p_1(size: SizeConfig) -> AuxSumcheckConfig {
    let blocked = size == SizeConfig::Small;
    AuxSumcheckConfig {
        exact_projection_norm: false,
        witness_height: size.pick(
            2usize.pow(13),
            2usize.pow(13),
            2usize.pow(14),
            2usize.pow(14),
        ),
        witness_width: size.pick(2usize.pow(3), 2usize.pow(4), 2usize.pow(4), 2usize.pow(4)),
        projection_ratio: 2usize.pow(5),
        projection_height: 2usize.pow(8),
        basic_commitment_rank: if blocked { 24 } else { size.pick(6, 6, 6, 6) },
        basic_commitment_diag_blocks: if blocked { 4 } else { 1 },
        nof_openings: 2,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: size.pick(2, 2, 4, 4),
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        projection_recursion: AuxProjection::Coarse(AuxRecursionConfig {
            decomposition_base_log: 9,
            decomposition_chunks: 2,
            rank: 2,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        }),

        witness_decomposition_chunks: 2,
        // the base-2^6 window measured 2082 against its 2080 cap at p-28
        // (transcript-dependent); base 2^7, already the p-30 value, restores
        // margin at unchanged composed geometry
        witness_decomposition_base_log: 7,

        next: Some(Box::new(AuxConfig::Sumcheck(p_2(size)))),
        // next: None
    }
}

pub fn p_2(size: SizeConfig) -> AuxSumcheckConfig {
    AuxSumcheckConfig {
        exact_projection_norm: false,
        witness_height: match size {
            SizeConfig::Micro => 2usize.pow(9),
            _ => size.pick(
                2usize.pow(10),
                2usize.pow(10),
                2usize.pow(11),
                2usize.pow(11),
            ),
        },
        witness_width: 2usize.pow(5),
        projection_ratio: size.pick(2usize.pow(6), 2usize.pow(5), 2usize.pow(8), 2usize.pow(8)),
        projection_height: 2usize.pow(8),
        basic_commitment_rank: 6,
        basic_commitment_diag_blocks: 1,
        nof_openings: 2,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        projection_recursion: AuxProjection::Fine {
            nof_batches: 2,
            recursion_constant_term: AuxRecursionConfig {
                decomposition_base_log: 9,
                decomposition_chunks: 2,
                rank: 2,
                diag_blocks: 1,
                next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
            },
            recursion_batched_projection: AuxRecursionConfig {
                decomposition_base_log: 7,
                decomposition_chunks: 8,
                rank: 2,
                diag_blocks: 1,
                next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
            },
        },

        witness_decomposition_chunks: 2,
        witness_decomposition_base_log: 8,

        next: Some(Box::new(AuxConfig::Sumcheck(P_3.clone()))),
        // next: None
    }
}

pub static P_EN_SMALL: LazyLock<Config> = LazyLock::new(|| {
    let mut c = p_exact_norm_root_aux(SizeConfig::Small, 1).generate_config();
    assign_norm_bounds(&mut c, &NB_P_EN_26);
    c
});

pub static P_EN_MEDIUM: LazyLock<Config> = LazyLock::new(|| {
    let mut c = p_exact_norm_root_aux(SizeConfig::Medium, 1).generate_config();
    assign_norm_bounds(&mut c, &NB_P_EN_28);
    c
});
pub static P_EN_NARROW_LARGE: LazyLock<Config> = LazyLock::new(|| {
    let mut c = p_exact_norm_root_aux(SizeConfig::NarrowLarge, 1).generate_config();
    assign_norm_bounds(&mut c, &NB_P_EN_29);
    c
});
pub static P_EN_LARGE: LazyLock<Config> =
    LazyLock::new(|| p_exact_norm_root_aux(SizeConfig::Large, 1).generate_config()); // never executed, OOM for 64GiB RAM

pub static P_EN: LazyLock<Config> = LazyLock::new(|| match compiled_size() {
    SizeConfig::Micro | SizeConfig::Tiny => panic!("no exact-norm chain for p-22 / p-24; use P"),
    SizeConfig::Small => P_EN_SMALL.clone(),
    SizeConfig::Medium => P_EN_MEDIUM.clone(),
    SizeConfig::NarrowLarge => P_EN_NARROW_LARGE.clone(),
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
pub static P_EN_2_NARROW_LARGE: LazyLock<Config> = LazyLock::new(|| {
    let mut c = p_exact_norm_root_aux(SizeConfig::NarrowLarge, 2).generate_config();
    assign_norm_bounds(&mut c, &NB_P_EN_29);
    c
});
pub static P_EN_2_LARGE: LazyLock<Config> =
    LazyLock::new(|| p_exact_norm_root_aux(SizeConfig::Large, 2).generate_config()); // never executed, OOM for 64GiB RAM

pub static P_EN_TWO_EVALS: LazyLock<Config> = LazyLock::new(|| match compiled_size() {
    SizeConfig::Micro | SizeConfig::Tiny => panic!("no exact-norm chain for p-22 / p-24; use P"),
    SizeConfig::Small => P_EN_2_SMALL.clone(),
    SizeConfig::Medium => P_EN_2_MEDIUM.clone(),
    SizeConfig::NarrowLarge => P_EN_2_NARROW_LARGE.clone(),
    SizeConfig::Large => P_EN_2_LARGE.clone(),
});

pub static P_MICRO: LazyLock<Config> = LazyLock::new(|| {
    let mut c = p_root_aux_short(SizeConfig::Micro, 1).generate_config();
    assign_norm_bounds(&mut c, &NB_P_22);
    c
});
pub static P_TINY: LazyLock<Config> = LazyLock::new(|| {
    let mut c = p_root_aux_short(SizeConfig::Tiny, 1).generate_config();
    assign_norm_bounds(&mut c, &NB_P_24);
    c
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

pub static P_2_MICRO: LazyLock<Config> =
    LazyLock::new(|| p_root_aux_short(SizeConfig::Micro, 2).generate_config());
pub static P_2_TINY: LazyLock<Config> =
    LazyLock::new(|| p_root_aux_short(SizeConfig::Tiny, 2).generate_config());
pub static P_2_SMALL: LazyLock<Config> =
    LazyLock::new(|| p_root_aux(SizeConfig::Small, 2).generate_config());
pub static P_2_MEDIUM: LazyLock<Config> =
    LazyLock::new(|| p_root_aux(SizeConfig::Medium, 2).generate_config());
pub static P_2_LARGE: LazyLock<Config> =
    LazyLock::new(|| p_root_aux(SizeConfig::Large, 2).generate_config());

pub static P: LazyLock<Config> = LazyLock::new(|| match compiled_size() {
    SizeConfig::Micro => P_MICRO.clone(),
    SizeConfig::Tiny => P_TINY.clone(),
    SizeConfig::Small => P_SMALL.clone(),
    SizeConfig::Medium => P_MEDIUM.clone(),
    SizeConfig::NarrowLarge => {
        panic!(
            "no calibrated norm bounds for the plain NarrowLarge chain; use P_EN / P_EN_TWO_EVALS"
        )
    }
    SizeConfig::Large => P_LARGE.clone(),
});

pub static P_TWO_EVALS: LazyLock<Config> = LazyLock::new(|| match compiled_size() {
    SizeConfig::Micro => P_2_MICRO.clone(),
    SizeConfig::Tiny => P_2_TINY.clone(),
    SizeConfig::Small => P_2_SMALL.clone(),
    SizeConfig::Medium => P_2_MEDIUM.clone(),
    SizeConfig::NarrowLarge => {
        panic!(
            "no calibrated norm bounds for the plain NarrowLarge chain; use P_EN / P_EN_TWO_EVALS"
        )
    }
    SizeConfig::Large => P_2_LARGE.clone(),
});

pub static P_3: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
    exact_projection_norm: false,
    witness_height: 2usize.pow(8),
    witness_width: 2usize.pow(5),
    projection_ratio: 2usize.pow(5),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 6,
    basic_commitment_diag_blocks: 1,
    nof_openings: 2,
    commitment_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        diag_blocks: 1,
        next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
    },
    opening_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        diag_blocks: 1,
        next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
    },
    projection_recursion: AuxProjection::Fine {
        nof_batches: 2,
        recursion_constant_term: AuxRecursionConfig {
            decomposition_base_log: 10,
            decomposition_chunks: 2,
            rank: 2,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        recursion_batched_projection: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
    },

    witness_decomposition_chunks: 2,
    witness_decomposition_base_log: 8,
    next: Some(Box::new(AuxConfig::Sumcheck(P_4.clone()))),
    // next: None
});

pub static P_4: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
    exact_projection_norm: false,
    witness_height: 2usize.pow(9),
    witness_width: 2usize.pow(3),
    projection_ratio: 2usize.pow(5),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 5,
    basic_commitment_diag_blocks: 1,
    nof_openings: 2,
    commitment_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        diag_blocks: 1,
        next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
    },
    opening_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        diag_blocks: 1,
        next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
    },
    projection_recursion: AuxProjection::Fine {
        nof_batches: 2,
        recursion_constant_term: AuxRecursionConfig {
            decomposition_base_log: 9,
            decomposition_chunks: 2,
            rank: 2,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        recursion_batched_projection: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            diag_blocks: 1,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
    },

    witness_decomposition_chunks: 2,
    witness_decomposition_base_log: 7,

    next: Some(Box::new(AuxConfig::Sumcheck(P_5.clone()))),
});

pub static P_5: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
    exact_projection_norm: false,
    witness_height: 2usize.pow(8),
    witness_width: 2usize.pow(3),
    projection_ratio: 2usize.pow(6),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 4,
    basic_commitment_diag_blocks: 1,
    nof_openings: 2,
    commitment_recursion: AuxRecursionConfig {
        decomposition_base_log: 8,
        decomposition_chunks: 7,
        rank: 2,
        diag_blocks: 1,
        next: None,
    },
    opening_recursion: AuxRecursionConfig {
        decomposition_base_log: 8,
        decomposition_chunks: 7,
        rank: 2,
        diag_blocks: 1,
        next: None,
    },
    projection_recursion: AuxProjection::Fine {
        nof_batches: 2,
        recursion_constant_term: AuxRecursionConfig {
            decomposition_base_log: 9,
            decomposition_chunks: 2,
            rank: 2,
            diag_blocks: 1,
            next: None,
        },
        recursion_batched_projection: AuxRecursionConfig {
            decomposition_base_log: 8,
            decomposition_chunks: 7,
            rank: 2,
            diag_blocks: 1,
            next: None,
        },
    },

    witness_decomposition_chunks: 2,
    witness_decomposition_base_log: 7,
    next: Some(Box::new(AuxConfig::Simple(P_LAST.clone()))),
    // next: None
});

pub static P_LAST: LazyLock<SimpleConfig> = LazyLock::new(|| SimpleConfig {
    witness_height: 2usize.pow(8),
    witness_width: 2usize.pow(2),
    projection_ratio: 2usize.pow(7),
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

#[tracing::instrument(skip_all, name = "commit::decompose_witness")]
pub fn decompose_witness(
    witness: &VerticallyAlignedMatrix<RingElement>,
) -> VerticallyAlignedMatrix<RingElement> {
    decompose_witness_with_digits(witness, None).0
}

/// The decomposition and, when asked, the same digits narrowed to `i16` in the coefficient
/// domain, which is the form the CRT commitment consumes.
#[tracing::instrument(skip_all, name = "commit::decompose_witness")]
pub fn decompose_witness_with_digits(
    witness: &VerticallyAlignedMatrix<RingElement>,
    digits: Option<&mut Vec<crate::protocol::project_coarse::Signed16RingElement>>,
) -> (
    VerticallyAlignedMatrix<RingElement>,
    Option<VerticallyAlignedMatrix<crate::protocol::project_coarse::Signed16RingElement>>,
) {
    let config = &*WITNESS_CONFIG;
    let height = witness.height * config.decomposition_chunks;
    let wanted = digits.is_some();
    let mut sink = digits;
    if let Some(sink) = sink.as_deref_mut() {
        sink.clear();
        sink.reserve(height * witness.width);
    }
    let decomposed_data = crate::common::decomposition::decompose_into(
        &witness.data,
        config.decomposition_base_log as u64,
        config.decomposition_chunks,
        sink.as_deref_mut(),
    );
    let decomposed = VerticallyAlignedMatrix {
        height,
        width: witness.width,
        data: decomposed_data,
        used_cols: witness.width,
    };
    let narrowed = wanted.then(|| VerticallyAlignedMatrix {
        height,
        width: witness.width,
        data: std::mem::take(sink.unwrap()),
        used_cols: witness.width,
    });
    (decomposed, narrowed)
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
            let Some(next) = sc.next.as_deref() else {
                break;
            };
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
    fn test_short_chain_dims() {
        assert_chain_dims(&super::P_MICRO);
        assert_chain_dims(&super::P_TINY);
        assert_chain_dims(&super::P_SMALL);
        assert_chain_dims(&super::P_MEDIUM);
    }

    #[test]
    fn test_p29_chain_dims() {
        assert_chain_dims(&super::P_EN_NARROW_LARGE);
        assert_chain_dims(&super::P_EN_2_NARROW_LARGE);
        assert_chain_dims(&super::p_root_aux(super::SizeConfig::NarrowLarge, 1).generate_config());
        assert_chain_dims(&super::p_root_aux(super::SizeConfig::NarrowLarge, 2).generate_config());
    }

    #[test]
    fn test_p29_front_end_witness_size() {
        let Config::Sumcheck(front) = &*super::P_EN_2_NARROW_LARGE else {
            panic!("expected a sumcheck config at the top level");
        };
        assert_eq!(front.witness_height, 1 << 15);
        assert_eq!(front.witness_width, 1 << 8);
        assert_eq!(
            (front.witness_height * front.witness_width * crate::common::config::DEGREE / 2)
                .ilog2(),
            29
        );
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
