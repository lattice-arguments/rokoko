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
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SizeConfig {
    Small,
    Medium,
    NarrowLarge,
    Large,
}

impl SizeConfig {
    #[inline(always)]
    pub fn pick<T>(self, small: T, medium: T, narrow_large: T, large: T) -> T {
        match self {
            SizeConfig::Small => small,
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
    SizeConfig::Medium
}

pub const NORM_MARGIN: f64 = 1.85; // verifier accepts norms up to this factor times the expected bound

const NB_P_26: [[f64; 3]; 7] = [
    [53005.60869379768, 2187.258786700833, f64::INFINITY],
    [75795.76230238733, 2714.218487889286, f64::INFINITY],
    [42347.903572668154, 3132.204335607752, f64::INFINITY],
    [37207.207420068495, 3114.7304859329324, f64::INFINITY],
    [21489.53142811634, 3131.9711045921226, f64::INFINITY],
    [19936.88313152284, 18768.604210222988, f64::INFINITY],
    [93674.81073372926, 224590.90958674173, f64::INFINITY],
];

const NB_P_28: [[f64; 3]; 7] = [
    [75086.2198009728, 2222.9271243115463, f64::INFINITY],
    [97049.38765391568, 2710.4818759770374, f64::INFINITY],
    [53543.95118031541, 3122.0562134593283, f64::INFINITY],
    [39527.95458912591, 3159.0519147364453, f64::INFINITY],
    [21491.85138604862, 3129.8618180360613, f64::INFINITY],
    [20026.54103933078, 18844.89355236585, f64::INFINITY],
    [94471.88754333217, 227110.0547906235, f64::INFINITY],
];

const NB_P_30: [[f64; 3]; 7] = [
    [155066.22588107316, 2206.3048293470238, f64::INFINITY],
    [127504.2830653151, 3126.8253868740417, f64::INFINITY],
    [46944.76686490199, 3160.822203161703, f64::INFINITY],
    [41239.37173381767, 3127.22848541644, f64::INFINITY],
    [20945.39655867131, 3108.5462840369614, f64::INFINITY],
    [19945.057031756012, 18776.107264286708, f64::INFINITY],
    [93419.08105949234, 230465.33839603735, f64::INFINITY],
];

const NB_P_EN_26: [[f64; 3]; 8] = [
    [160182.94334291652, 2724.094344915389, f64::INFINITY],
    [89330.65946247123, 2711.6249740699764, f64::INFINITY],
    [71240.89427428602, 2719.503263465591, f64::INFINITY],
    [49323.085852367345, 3141.5819263549374, f64::INFINITY],
    [37658.174995610185, 3129.595181489133, f64::INFINITY],
    [20677.989118867434, 3116.819211953109, f64::INFINITY],
    [19947.41073422814, 18776.49674460068, f64::INFINITY],
    [93611.53013384623, 214747.6993869783, f64::INFINITY],
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
        projection_ratio: 2usize.pow(6),
        projection_height: 2usize.pow(8),
        basic_commitment_rank: size.pick(5, 5, 6, 6),
        nof_openings: 2,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: size.pick(2, 2, 4, 4),
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
        witness_decomposition_base_log: size.pick(6, 6, 6, 7),

        next: Some(Box::new(AuxConfig::Sumcheck(p_1(size)))),
    }
}

pub fn p_1(size: SizeConfig) -> AuxSumcheckConfig {
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
        basic_commitment_rank: size.pick(6, 6, 6, 6),
        nof_openings: 2,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: size.pick(2, 2, 4, 4),
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
        witness_height: size.pick(
            2usize.pow(10),
            2usize.pow(10),
            2usize.pow(11),
            2usize.pow(11),
        ),
        witness_width: 2usize.pow(5),
        projection_ratio: size.pick(2usize.pow(6), 2usize.pow(5), 2usize.pow(8), 2usize.pow(8)),
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
        projection_recursion: AuxProjection::Fine {
            nof_batches: 2,
            recursion_constant_term: AuxRecursionConfig {
                decomposition_base_log: 9,
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
    SizeConfig::Small => P_EN_2_SMALL.clone(),
    SizeConfig::Medium => P_EN_2_MEDIUM.clone(),
    SizeConfig::NarrowLarge => P_EN_2_NARROW_LARGE.clone(),
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

pub static P_2_SMALL: LazyLock<Config> =
    LazyLock::new(|| p_root_aux(SizeConfig::Small, 2).generate_config());
pub static P_2_MEDIUM: LazyLock<Config> =
    LazyLock::new(|| p_root_aux(SizeConfig::Medium, 2).generate_config());
pub static P_2_LARGE: LazyLock<Config> =
    LazyLock::new(|| p_root_aux(SizeConfig::Large, 2).generate_config());

pub static P: LazyLock<Config> = LazyLock::new(|| match compiled_size() {
    SizeConfig::Small => P_SMALL.clone(),
    SizeConfig::Medium => P_MEDIUM.clone(),
    SizeConfig::NarrowLarge => {
        panic!("no calibrated norm bounds for the plain NarrowLarge chain; use P_EN / P_EN_TWO_EVALS")
    }
    SizeConfig::Large => P_LARGE.clone(),
});

pub static P_TWO_EVALS: LazyLock<Config> = LazyLock::new(|| match compiled_size() {
    SizeConfig::Small => P_2_SMALL.clone(),
    SizeConfig::Medium => P_2_MEDIUM.clone(),
    SizeConfig::NarrowLarge => {
        panic!("no calibrated norm bounds for the plain NarrowLarge chain; use P_EN / P_EN_TWO_EVALS")
    }
    SizeConfig::Large => P_2_LARGE.clone(),
});

pub static P_3: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
    exact_projection_norm: false,
    witness_height: 2usize.pow(8),
    witness_width: 2usize.pow(5),
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
    // next: None
});

pub static P_4: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
    exact_projection_norm: false,
    witness_height: 2usize.pow(9),
    witness_width: 2usize.pow(3),
    projection_ratio: 2usize.pow(5),
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
            decomposition_base_log: 9,
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
    exact_projection_norm: false,
    witness_height: 2usize.pow(8),
    witness_width: 2usize.pow(3),
    projection_ratio: 2usize.pow(6),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 4,
    nof_openings: 2,
    commitment_recursion: AuxRecursionConfig {
        decomposition_base_log: 8,
        decomposition_chunks: 7,
        rank: 2,
        next: None,
    },
    opening_recursion: AuxRecursionConfig {
        decomposition_base_log: 8,
        decomposition_chunks: 7,
        rank: 2,
        next: None,
    },
    projection_recursion: AuxProjection::Fine {
        nof_batches: 2,
        recursion_constant_term: AuxRecursionConfig {
            decomposition_base_log: 9,
            decomposition_chunks: 2,
            rank: 2,
            next: None,
        },
        recursion_batched_projection: AuxRecursionConfig {
            decomposition_base_log: 8,
            decomposition_chunks: 7,
            rank: 2,
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
