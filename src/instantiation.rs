use std::sync::LazyLock;

use rokoko::protocol::{
    config::{Config, SimpleConfig},
    config_generator::{AuxConfig, AuxProjection, AuxRecursionConfig, AuxSumcheckConfig},
    params::{assign_norm_bounds, InitialWitnessParams, Instantiation},
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ParamSet {
    P22,
    P24,
    P26,
    P28,
    P29,
    P30,
}

impl ParamSet {
    pub const ALL: [ParamSet; 6] = [
        ParamSet::P22,
        ParamSet::P24,
        ParamSet::P26,
        ParamSet::P28,
        ParamSet::P29,
        ParamSet::P30,
    ];

    pub fn name(self) -> &'static str {
        match self {
            ParamSet::P22 => "p-22",
            ParamSet::P24 => "p-24",
            ParamSet::P26 => "p-26",
            ParamSet::P28 => "p-28",
            ParamSet::P29 => "p-29",
            ParamSet::P30 => "p-30",
        }
    }

    #[inline(always)]
    pub fn pick<T>(self, small: T, medium: T, narrow_large: T, large: T) -> T {
        match self {
            // p-24 and p-22 share p-26's tail
            ParamSet::P22 | ParamSet::P24 | ParamSet::P26 => small,
            ParamSet::P28 => medium,
            ParamSet::P29 => narrow_large,
            ParamSet::P30 => large,
        }
    }
}

impl std::str::FromStr for ParamSet {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        ParamSet::ALL
            .into_iter()
            .find(|set| set.name() == s)
            .ok_or_else(|| {
                let names: Vec<_> = ParamSet::ALL.iter().map(|set| set.name()).collect();
                format!(
                    "unknown parameter set {s:?}; expected one of {}",
                    names.join(", ")
                )
            })
    }
}

pub fn instantiation(set: ParamSet) -> Option<Instantiation> {
    Some(Instantiation {
        config: chain(set)?,
        witness: initial_witness(&root_aux(set, 1)),
    })
}

pub fn snark_instantiation(set: ParamSet) -> Option<Instantiation> {
    Some(Instantiation {
        config: exact_norm_chain(set, 2)?,
        witness: initial_witness(&root_aux(set, 1)),
    })
}

/// 2^28 Z_q elements of norm 2^32 => 2^29 Z_q elements of norm 2^16 (signed 2^15)
/// => 2^22 R_q elements => height 2^15, width 2^7
pub fn initial_witness(root: &AuxSumcheckConfig) -> InitialWitnessParams {
    let decomposition_chunks = 2;
    InitialWitnessParams {
        height: root.witness_height / decomposition_chunks,
        width: root.witness_width,
        decomposition_base_log: 16, // change to 8 for EN sets
        decomposition_chunks,
        initial_norm_log: 31, // change to 15 for EN sets
    }
}

fn root_aux(set: ParamSet, nof_openings: usize) -> AuxSumcheckConfig {
    match set {
        ParamSet::P22 | ParamSet::P24 => p_root_aux_short(set, nof_openings),
        _ => p_root_aux(set, nof_openings),
    }
}

pub fn chain(set: ParamSet) -> Option<Config> {
    let bounds: &[[f64; 3]] = match set {
        ParamSet::P22 => &NB_P_22,
        ParamSet::P24 => &NB_P_24,
        ParamSet::P26 => &NB_P_26,
        ParamSet::P28 => &NB_P_28,
        ParamSet::P29 => return None,
        ParamSet::P30 => &NB_P_30,
    };
    let mut c = root_aux(set, 1).generate_config();
    assign_norm_bounds(&mut c, bounds);
    Some(c)
}

pub fn exact_norm_chain(set: ParamSet, nof_openings: usize) -> Option<Config> {
    let bounds: Option<&[[f64; 3]]> = match set {
        ParamSet::P22 | ParamSet::P24 => return None,
        ParamSet::P26 => Some(&NB_P_EN_26),
        ParamSet::P28 => Some(&NB_P_EN_28),
        ParamSet::P29 => Some(&NB_P_EN_29),
        ParamSet::P30 => None, // never executed, OOM for 64GiB RAM
    };
    let mut c = p_exact_norm_root_aux(set, nof_openings).generate_config();
    if let Some(bounds) = bounds {
        assign_norm_bounds(&mut c, bounds);
    }
    Some(c)
}

pub static DECOMP_11_LAST_LEVEL: AuxRecursionConfig = AuxRecursionConfig {
    decomposition_base_log: 5,
    decomposition_chunks: 11,
    rank: 1,
    next: None,
};

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

pub fn p_exact_norm_root_aux(set: ParamSet, nof_openings: usize) -> AuxSumcheckConfig {
    AuxSumcheckConfig {
        exact_projection_norm: true,
        witness_height: set.pick(
            2usize.pow(13),
            2usize.pow(14),
            2usize.pow(15),
            2usize.pow(15),
        ),
        witness_width: set.pick(2usize.pow(7), 2usize.pow(8), 2usize.pow(8), 2usize.pow(9)),
        projection_ratio: 2usize.pow(5),
        projection_height: 2usize.pow(8),
        basic_commitment_rank: 6,
        nof_openings,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        projection_recursion: AuxProjection::Coarse(AuxRecursionConfig {
            decomposition_base_log: 8,
            decomposition_chunks: 2,
            rank: 2,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        }),

        witness_decomposition_chunks: 4,
        witness_decomposition_base_log: set.pick(4, 4, 4, 7),

        next: Some(Box::new(AuxConfig::Sumcheck(p_int(set)))),
    }
}

pub fn p_int(set: ParamSet) -> AuxSumcheckConfig {
    AuxSumcheckConfig {
        exact_projection_norm: true,
        witness_height: set.pick(
            2usize.pow(14),
            2usize.pow(15),
            2usize.pow(16),
            2usize.pow(16),
        ),
        witness_width: set.pick(2usize.pow(3), 2usize.pow(4), 2usize.pow(4), 2usize.pow(5)),
        projection_ratio: 2usize.pow(5),
        projection_height: 2usize.pow(8),
        basic_commitment_rank: set.pick(5, 5, 6, 6),
        nof_openings: 2,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: set.pick(2, 2, 4, 4),
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        projection_recursion: AuxProjection::Coarse(AuxRecursionConfig {
            decomposition_base_log: 9,
            decomposition_chunks: 2,
            rank: 2,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        }),

        witness_decomposition_chunks: 2,
        witness_decomposition_base_log: 7,

        next: Some(Box::new(AuxConfig::Sumcheck(p_1(set)))),
    }
}

pub fn p_root_aux(set: ParamSet, nof_openings: usize) -> AuxSumcheckConfig {
    AuxSumcheckConfig {
        exact_projection_norm: false,
        witness_height: set.pick(
            2usize.pow(13),
            2usize.pow(14),
            2usize.pow(15),
            2usize.pow(15),
        ),
        witness_width: set.pick(2usize.pow(7), 2usize.pow(8), 2usize.pow(8), 2usize.pow(9)),
        projection_ratio: 1,              // no-op
        projection_height: 2usize.pow(8), // no-op,
        basic_commitment_rank: set.pick(10, 10, 10, 12),
        nof_openings,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 4,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 4,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        projection_recursion: AuxProjection::Skip,

        witness_decomposition_chunks: 4,
        witness_decomposition_base_log: set.pick(6, 6, 6, 7),

        next: Some(Box::new(AuxConfig::Sumcheck(p_1(set)))),
    }
}

/// Root of the p-24 and p-22 chains. Its composed witness is already as short as the one p_1
/// composes to (p-22: half of it, hence the shorter p_2), so the chain skips p_1.
pub fn p_root_aux_short(set: ParamSet, nof_openings: usize) -> AuxSumcheckConfig {
    let tiny = set == ParamSet::P24;
    AuxSumcheckConfig {
        exact_projection_norm: false,
        witness_height: if tiny { 2usize.pow(11) } else { 2usize.pow(10) },
        witness_width: if tiny { 2usize.pow(7) } else { 2usize.pow(6) },
        projection_ratio: 1,              // no-op
        projection_height: 2usize.pow(8), // no-op,
        basic_commitment_rank: 10,
        nof_openings,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        projection_recursion: AuxProjection::Skip,

        witness_decomposition_chunks: 4,
        witness_decomposition_base_log: 6,

        next: Some(Box::new(AuxConfig::Sumcheck(p_2(set)))),
    }
}

pub fn p_1(set: ParamSet) -> AuxSumcheckConfig {
    AuxSumcheckConfig {
        exact_projection_norm: false,
        witness_height: set.pick(
            2usize.pow(13),
            2usize.pow(13),
            2usize.pow(14),
            2usize.pow(14),
        ),
        witness_width: set.pick(2usize.pow(3), 2usize.pow(4), 2usize.pow(4), 2usize.pow(4)),
        projection_ratio: 2usize.pow(5),
        projection_height: 2usize.pow(8),
        basic_commitment_rank: set.pick(6, 6, 6, 6),
        nof_openings: 2,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: set.pick(2, 2, 4, 4),
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: set.pick(2, 2, 4, 4),
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        projection_recursion: AuxProjection::Coarse(AuxRecursionConfig {
            decomposition_base_log: set.pick(8, 8, 9, 9),
            decomposition_chunks: 2,
            rank: set.pick(2, 2, 4, 4),
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        }),

        witness_decomposition_chunks: 2,
        // the base-2^6 window measured 2082 against its 2080 cap at p-28
        // (transcript-dependent); base 2^7, already the p-30 value, restores
        // margin at unchanged composed geometry
        witness_decomposition_base_log: 7,

        next: Some(Box::new(AuxConfig::Sumcheck(p_2(set)))),
        // next: None
    }
}

pub fn p_2(set: ParamSet) -> AuxSumcheckConfig {
    AuxSumcheckConfig {
        exact_projection_norm: false,
        witness_height: match set {
            ParamSet::P22 => 2usize.pow(9),
            _ => set.pick(
                2usize.pow(10),
                2usize.pow(10),
                2usize.pow(11),
                2usize.pow(11),
            ),
        },
        witness_width: 2usize.pow(5),
        projection_ratio: set.pick(2usize.pow(6), 2usize.pow(5), 2usize.pow(8), 2usize.pow(8)),
        projection_height: 2usize.pow(8),
        basic_commitment_rank: 6,
        nof_openings: 2,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        projection_recursion: AuxProjection::Fine {
            nof_batches: 2,
            recursion_constant_term: AuxRecursionConfig {
                decomposition_base_log: 9,
                decomposition_chunks: 2,
                rank: 2,
                next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
            },
            recursion_batched_projection: AuxRecursionConfig {
                decomposition_base_log: 7,
                decomposition_chunks: 8,
                rank: 2,
                next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
            },
        },

        witness_decomposition_chunks: 2,
        witness_decomposition_base_log: 8,

        next: Some(Box::new(AuxConfig::Sumcheck(P_3.clone()))),
        // next: None
    }
}

pub static P_3: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
    exact_projection_norm: false,
    witness_height: 2usize.pow(8),
    witness_width: 2usize.pow(5),
    projection_ratio: 2usize.pow(5),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 6,
    nof_openings: 2,
    commitment_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
    },
    opening_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
    },
    projection_recursion: AuxProjection::Fine {
        nof_batches: 2,
        recursion_constant_term: AuxRecursionConfig {
            decomposition_base_log: 10,
            decomposition_chunks: 2,
            rank: 2,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        recursion_batched_projection: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
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
    nof_openings: 2,
    commitment_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
    },
    opening_recursion: AuxRecursionConfig {
        decomposition_base_log: 7,
        decomposition_chunks: 8,
        rank: 2,
        next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
    },
    projection_recursion: AuxProjection::Fine {
        nof_batches: 2,
        recursion_constant_term: AuxRecursionConfig {
            decomposition_base_log: 9,
            decomposition_chunks: 2,
            rank: 2,
            next: Some(Box::new(DECOMP_11_LAST_LEVEL.clone())),
        },
        recursion_batched_projection: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: 2,
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

#[cfg(all(test, ring_degree = "128"))]
mod tests {
    use super::{exact_norm_chain, instantiation, p_root_aux, ParamSet};
    use rokoko::common::init_common;
    use rokoko::protocol::config::Config;
    use rokoko::protocol::parties::executor::{execute, execute_to_boundary};
    use std::num::NonZeroUsize;

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
        assert_chain_dims(&exact_norm_chain(ParamSet::P28, 1).unwrap());
    }

    #[test]
    fn test_short_chain_dims() {
        assert_chain_dims(&super::chain(ParamSet::P22).unwrap());
        assert_chain_dims(&super::chain(ParamSet::P24).unwrap());
    }

    #[test]
    fn test_p29_chain_dims() {
        assert_chain_dims(&exact_norm_chain(ParamSet::P29, 1).unwrap());
        assert_chain_dims(&exact_norm_chain(ParamSet::P29, 2).unwrap());
        assert_chain_dims(&p_root_aux(ParamSet::P29, 1).generate_config());
        assert_chain_dims(&p_root_aux(ParamSet::P29, 2).generate_config());
    }

    #[test]
    fn test_p29_front_end_witness_size() {
        let Some(Config::Sumcheck(front)) = exact_norm_chain(ParamSet::P29, 2) else {
            panic!("expected a sumcheck config at the top level");
        };
        assert_eq!(front.witness_height, 1 << 15);
        assert_eq!(front.witness_width, 1 << 8);
        assert_eq!(
            (front.witness_height * front.witness_width * rokoko::common::config::DEGREE / 2)
                .ilog2(),
            29
        );
    }

    #[test]
    fn initial_witness_fills_the_root() {
        for inst in ParamSet::ALL.into_iter().filter_map(instantiation) {
            let root = inst.root();
            assert_eq!(
                inst.witness.height * inst.witness.decomposition_chunks,
                root.witness_height
            );
            assert_eq!(inst.witness.width, root.witness_width);
        }
    }

    /// The boundary tests stop a few rounds in, so they never reach the last sumcheck round,
    /// whose recursions are single levels and whose level 0 is therefore itself a leaf. Only a
    /// whole-chain run covers it.
    #[test]
    fn full_chain_verifies() {
        init_common();
        execute(&instantiation(ParamSet::P28).unwrap());
    }

    #[test]
    fn round_boundary_extraction() {
        init_common();
        let inst = instantiation(ParamSet::P28).unwrap();
        let mut run = execute_to_boundary(&inst, NonZeroUsize::new(3).unwrap());

        assert_eq!(run.prover.witness.height, 256);
        assert_eq!(run.prover.witness.width, 32);
        assert_eq!(run.verifier.commitment_root.len(), 1);
        assert_eq!(run.prover.claims.len(), 2);
        assert_eq!(run.verifier.claims.len(), 2);
        assert_eq!(run.prover.evaluation_points, run.verifier.evaluation_points);

        let mut prover_bytes = [0u8; 16];
        let mut verifier_bytes = [0u8; 16];
        run.prover
            .transcript
            .fill_from_xof(b"round-boundary-test", &mut prover_bytes);
        run.verifier
            .transcript
            .fill_from_xof(b"round-boundary-test", &mut verifier_bytes);
        assert_eq!(prover_bytes, verifier_bytes);

        assert_eq!(run.crs.cks.len(), run.verifier_crs.structured_cks.len());
        let first_row = &run.verifier_crs.structured_cks[0][0];
        assert_eq!(first_row.tensor_layers.len(), 1);

        let run4 = execute_to_boundary(&inst, NonZeroUsize::new(4).unwrap());
        assert_eq!(run4.prover.witness.height, 512);
        assert_eq!(run4.prover.witness.width, 8);
        assert_eq!(
            run4.prover.evaluation_points,
            run4.verifier.evaluation_points
        );
    }
}
