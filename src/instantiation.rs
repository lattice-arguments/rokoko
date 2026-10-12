use std::sync::LazyLock;

use rokoko::common::config::NOF_BATCHES;
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
        witness: initial_witness(set),
    })
}

pub fn snark_instantiation(set: ParamSet) -> Option<Instantiation> {
    Some(Instantiation {
        config: exact_norm_chain(set, 2)?,
        witness: initial_witness(set),
    })
}

/// 2^28 Z_q elements of norm 2^32 => 2^29 Z_q elements of norm 2^16 (signed 2^15)
/// => 2^22 R_q elements => height 2^15, width 2^7
pub(crate) fn initial_witness(set: ParamSet) -> InitialWitnessParams {
    let root = root_aux(set, 1);
    let decomposition_chunks = 2;
    InitialWitnessParams {
        height: root.witness_height / decomposition_chunks,
        width: root.witness_width,
        decomposition_base_log: 16, // change to 8 for EN sets
        decomposition_chunks,
        initial_norm_log: 31, // change to 15 for EN sets
    }
}

pub(crate) fn root_aux(set: ParamSet, nof_openings: usize) -> AuxSumcheckConfig {
    match set {
        ParamSet::P22 | ParamSet::P24 => p_root_aux_short(set, nof_openings),
        _ => p_root_aux(set, nof_openings),
    }
}

pub fn chain(set: ParamSet) -> Option<Config> {
    assert_two_batches();
    let bounds: &[[f64; 3]] = match set {
        ParamSet::P22 => &NB_P_22,
        ParamSet::P24 => &NB_P_24,
        ParamSet::P26 => &NB_P_26,
        ParamSet::P28 => &NB_P_28,
        ParamSet::P29 => return None,
        ParamSet::P30 => &NB_P_30,
    };
    let mut root = root_aux(set, 1);
    root.next = Some(Box::new(AuxConfig::Sumcheck(match set {
        ParamSet::P22 => TAIL_14.clone(),
        _ => p_head(set),
    })));
    // only these sets' verifier budget covers the 2^7 terminal ratio
    if matches!(set, ParamSet::P22 | ParamSet::P26 | ParamSet::P30) {
        terminal(&mut root).projection_ratio = 2usize.pow(7);
    }
    let mut c = root.generate_config();
    assign_norm_bounds(&mut c, bounds);
    Some(c)
}

pub(crate) fn terminal(round: &mut AuxSumcheckConfig) -> &mut SimpleConfig {
    match round.next.as_deref_mut() {
        Some(AuxConfig::Sumcheck(next)) => terminal(next),
        Some(AuxConfig::Simple(last)) => last,
        _ => unreachable!("a plain chain ends in a simple round"),
    }
}

pub fn exact_norm_chain(set: ParamSet, nof_openings: usize) -> Option<Config> {
    assert_two_batches();
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

/// The norm tables below are measured with two projection batches.
fn assert_two_batches() {
    assert_eq!(
        NOF_BATCHES, 2,
        "src/instantiation.rs is measured for 2 projection batches and the ring spec sets \
         {NOF_BATCHES}; src/main.rs selects the instantiation of a ring"
    );
}

pub static DECOMP_11_LAST_LEVEL: AuxRecursionConfig = AuxRecursionConfig {
    decomposition_base_log: 5,
    decomposition_chunks: 11,
    rank: 1,
    next: None,
};

const NB_P_22: [[f64; 3]; 6] = [
    [31582.8548424616, 664.4343458913003, f64::INFINITY],
    [10287.970062164839, 815.399288692356, f64::INFINITY],
    [7814.129510060606, 811.9981527072583, f64::INFINITY],
    [8158.630522336454, 7901.370577311255, f64::INFINITY],
    [48574.82206863963, 810.5436447224788, f64::INFINITY],
    [227045.92549085748, 514965.5385227248, f64::INFINITY],
];

const NB_P_24: [[f64; 3]; 7] = [
    [44636.17723327122, 668.5746031670661, f64::INFINITY],
    [13721.285289651258, 809.9283918964688, f64::INFINITY],
    [7699.331009899496, 812.3195184162449, f64::INFINITY],
    [7628.817929404267, 806.9405182539789, f64::INFINITY],
    [8121.582727523004, 7867.218631765612, f64::INFINITY],
    [48718.81122318155, 811.8873074509787, f64::INFINITY],
    [231712.41338780278, 560532.1830947801, f64::INFINITY],
];

const NB_P_26: [[f64; 3]; 7] = [
    [52972.47329509922, 939.7074012691397, f64::INFINITY],
    [17650.77610191688, 812.8093257339018, f64::INFINITY],
    [8923.67379502411, 802.1022378724548, f64::INFINITY],
    [7734.239846293881, 805.8120128168853, f64::INFINITY],
    [8180.561227690921, 7925.433363545491, f64::INFINITY],
    [48410.08584995486, 810.5467290662519, f64::INFINITY],
    [220999.3084740312, 561130.5429461134, f64::INFINITY],
];

const NB_P_28: [[f64; 3]; 7] = [
    [75055.5900383176, 932.2210038397548, f64::INFINITY],
    [20666.08366381981, 809.8950549299582, f64::INFINITY],
    [9713.90065833494, 809.8641861448128, f64::INFINITY],
    [7775.1501593216835, 811.0764452257259, f64::INFINITY],
    [8136.812582332224, 7880.73651380377, f64::INFINITY],
    [48374.699285887036, 810.2283628706168, f64::INFINITY],
    [227752.90103970136, 541265.3785519631, f64::INFINITY],
];

const NB_P_30: [[f64; 3]; 7] = [
    [159050.3893173481, 943.0524905857574, f64::INFINITY],
    [39818.32020565408, 816.7012917829873, f64::INFINITY],
    [12516.888271451495, 812.7022825118679, f64::INFINITY],
    [7955.511925702833, 813.5404107971527, f64::INFINITY],
    [8136.906967638256, 7875.548488835555, f64::INFINITY],
    [48339.03515793422, 808.555502114728, f64::INFINITY],
    [226074.88245711863, 527773.3843952345, f64::INFINITY],
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
        binary_tops: false,
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
        binary_tops: false,
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
        binary_tops: true,
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
        binary_tops: true,
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
        binary_tops: true,
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
        binary_tops: true,
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
        projection_recursion: AuxProjection::FineBatchedInOpening {
            recursion_constant_term: AuxRecursionConfig {
                decomposition_base_log: 9,
                decomposition_chunks: 2,
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
    binary_tops: true,
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
    projection_recursion: AuxProjection::FineBatchedInOpening {
        recursion_constant_term: AuxRecursionConfig {
            decomposition_base_log: 10,
            decomposition_chunks: 2,
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
    binary_tops: false,
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
    projection_recursion: AuxProjection::FineBatchedInOpening {
        recursion_constant_term: AuxRecursionConfig {
            decomposition_base_log: 9,
            decomposition_chunks: 2,
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
    binary_tops: false,
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
    projection_recursion: AuxProjection::FineBatchedInOpening {
        recursion_constant_term: AuxRecursionConfig {
            decomposition_base_log: 9,
            decomposition_chunks: 2,
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
    witness_norm_bound: f64::INFINITY,
    projection_norm_bound: f64::INFINITY,
});

fn recursion(base_log: usize, chunks: usize, two_level: bool) -> AuxRecursionConfig {
    AuxRecursionConfig {
        decomposition_base_log: base_log,
        decomposition_chunks: chunks,
        rank: 2,
        next: two_level.then(|| Box::new(DECOMP_11_LAST_LEVEL.clone())),
    }
}

/// The round after the root in the plain chains of `chain`, which p-22 skips for `TAIL_14`.
pub fn p_head(set: ParamSet) -> AuxSumcheckConfig {
    let (height_log, width_log, ratio_log, rank, digits, constant_term) = match set {
        ParamSet::P24 => (10, 5, 4, 4, (5, 10), (6, 2)),
        ParamSet::P26 => (11, 5, 4, 4, (5, 10), (6, 2)),
        ParamSet::P28 => (12, 5, 6, 6, (5, 10), (6, 2)),
        ParamSet::P30 => (12, 6, 8, 6, (6, 9), (6, 2)),
        ParamSet::P22 | ParamSet::P29 => unreachable!("{} has no plain head round", set.name()),
    };
    let recursion = |(base_log, chunks)| recursion(base_log, chunks, true);
    AuxSumcheckConfig {
        exact_projection_norm: false,
        binary_tops: true,
        witness_height: 1 << height_log,
        witness_width: 1 << width_log,
        projection_ratio: 1 << ratio_log,
        projection_height: 2usize.pow(8),
        basic_commitment_rank: rank,
        nof_openings: 2,
        commitment_recursion: recursion(digits),
        opening_recursion: recursion(digits),
        projection_recursion: AuxProjection::FineBatchedInOpening {
            recursion_constant_term: recursion(constant_term),
        },
        witness_decomposition_chunks: 2,
        witness_decomposition_base_log: 5,
        next: Some(Box::new(AuxConfig::Sumcheck(TAIL_14.clone()))),
    }
}

pub static TAIL_14: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
    exact_projection_norm: false,
    binary_tops: true,
    witness_height: 2usize.pow(10),
    witness_width: 2usize.pow(4),
    projection_ratio: 2usize.pow(6),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 5,
    nof_openings: 2,
    commitment_recursion: recursion(6, 9, true),
    opening_recursion: recursion(6, 9, true),
    projection_recursion: AuxProjection::FineBatchedInOpening {
        recursion_constant_term: recursion(6, 2, true),
    },
    witness_decomposition_chunks: 2,
    witness_decomposition_base_log: 5,
    next: Some(Box::new(AuxConfig::Sumcheck(TAIL_12.clone()))),
});

pub static TAIL_12: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
    exact_projection_norm: false,
    binary_tops: true,
    witness_height: 2usize.pow(9),
    witness_width: 2usize.pow(3),
    projection_ratio: 2usize.pow(5),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 4,
    nof_openings: 2,
    commitment_recursion: recursion(6, 8, true),
    opening_recursion: recursion(6, 9, true),
    projection_recursion: AuxProjection::FineBatchedInOpening {
        recursion_constant_term: recursion(6, 2, true),
    },
    witness_decomposition_chunks: 2,
    witness_decomposition_base_log: 5,
    next: Some(Box::new(AuxConfig::Sumcheck(TAIL_11.clone()))),
});

/// Eight base-2^6 digits window 48 bits of a residue; the top digit carries the rest.
pub static TAIL_11: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
    exact_projection_norm: false,
    binary_tops: false,
    witness_height: 2usize.pow(8),
    witness_width: 2usize.pow(3),
    projection_ratio: 2usize.pow(6),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 3,
    nof_openings: 2,
    commitment_recursion: recursion(6, 8, false),
    opening_recursion: recursion(6, 8, false),
    projection_recursion: AuxProjection::FineBatchedInOpening {
        recursion_constant_term: recursion(7, 2, false),
    },
    witness_decomposition_chunks: 2,
    witness_decomposition_base_log: 5,
    next: Some(Box::new(AuxConfig::Sumcheck(TAIL_10.clone()))),
});

pub static TAIL_10: LazyLock<AuxSumcheckConfig> = LazyLock::new(|| AuxSumcheckConfig {
    exact_projection_norm: false,
    binary_tops: false,
    witness_height: 2usize.pow(8),
    witness_width: 2usize.pow(2),
    projection_ratio: 2usize.pow(6),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 3,
    nof_openings: 2,
    commitment_recursion: recursion(10, 5, true),
    opening_recursion: recursion(9, 6, true),
    projection_recursion: AuxProjection::FineBatchedInOpening {
        recursion_constant_term: recursion(8, 2, true),
    },
    witness_decomposition_chunks: 1,
    witness_decomposition_base_log: 16,
    next: Some(Box::new(AuxConfig::Simple(TAIL_LAST.clone()))),
});

pub static TAIL_LAST: LazyLock<SimpleConfig> = LazyLock::new(|| SimpleConfig {
    witness_height: 2usize.pow(7),
    witness_width: 2usize.pow(2),
    projection_ratio: 2usize.pow(6),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 4,
    witness_norm_bound: f64::INFINITY,
    projection_norm_bound: f64::INFINITY,
});

// The chains of an n128_d4 build are those of `crate::instantiation_n128_d4`.
#[cfg(all(test, not(rokoko_ring = "n128_d4")))]
mod tests {
    use super::{exact_norm_chain, p_root_aux, ParamSet};
    use rokoko::protocol::config::Config;

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
}
