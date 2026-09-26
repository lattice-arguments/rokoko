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

const NB_P_22: [[f64; 3]; 5] = [
    [32780.13553968318, 658.8581030844199, f64::INFINITY],
    [47419.75602003874, 937.2283606464329, f64::INFINITY],
    [19652.954281735863, 17184.076408116904, f64::INFINITY],
    [82819.62027201042, 17959.385679916784, f64::INFINITY],
    [342150.11295044166, 906577.6996612039, f64::INFINITY],
];

const NB_P_24: [[f64; 3]; 6] = [
    [46380.63681106589, 658.2780567510966, f64::INFINITY],
    [52495.27435874585, 933.6910623969793, f64::INFINITY],
    [36664.73233231084, 932.7829329484969, f64::INFINITY],
    [20123.63543199886, 17723.635970082436, f64::INFINITY],
    [84691.22217798016, 18058.99352123479, f64::INFINITY],
    [343553.17282772984, 974675.9334322357, f64::INFINITY],
];

const NB_P_26: [[f64; 3]; 6] = [
    [65626.87609508775, 935.359289257342, f64::INFINITY],
    [81826.37432393055, 1329.583017340399, f64::INFINITY],
    [40182.334352299644, 936.3690511758705, f64::INFINITY],
    [20021.812655201826, 17588.60375356725, f64::INFINITY],
    [84896.52137160863, 18074.12905232227, f64::INFINITY],
    [354224.9021186963, 975599.2551206668, f64::INFINITY],
];

const NB_P_28: [[f64; 3]; 6] = [
    [77066.24937675377, 934.5346435526079, f64::INFINITY],
    [70699.86742561827, 1323.6721648504965, f64::INFINITY],
    [54203.87920619704, 935.0802104632522, f64::INFINITY],
    [20145.34559147596, 17696.098214013167, f64::INFINITY],
    [84928.53727104924, 17998.61655794689, f64::INFINITY],
    [367344.54147979384, 958576.8451699634, f64::INFINITY],
];

const NB_P_30: [[f64; 3]; 6] = [
    [161637.21955663554, 934.2644165331354, f64::INFINITY],
    [117458.72637654471, 1318.812344497882, f64::INFINITY],
    [54874.59558848703, 935.467262922653, f64::INFINITY],
    [20188.107142572826, 17747.034935447668, f64::INFINITY],
    [85814.1120737143, 18068.599198609725, f64::INFINITY],
    [352581.0992651194, 938639.6995173388, f64::INFINITY],
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
        nof_openings: 2,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 7,
            decomposition_chunks: 8,
            rank: size.pick(2, 2, 4, 4),
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

        next: Some(Box::new(AuxConfig::Sumcheck(p_1(size)))),
    }
}

pub fn p_root_aux(size: SizeConfig, nof_openings: usize) -> AuxSumcheckConfig {
    chained(AuxSumcheckConfig {
        exact_projection_norm: false,
        witness_height: size.pick(
            2usize.pow(11),
            2usize.pow(13),
            2usize.pow(14),
            2usize.pow(13),
        ),
        witness_width: size.pick(2usize.pow(8), 2usize.pow(8), 2usize.pow(8), 2usize.pow(10)),
        projection_ratio: 1,              // no-op
        projection_height: 2usize.pow(8), // no-op,
        basic_commitment_rank: size.pick(5, 5, 5, 6),
        nof_openings,
        commitment_recursion: two_level(7, 8, 2),
        opening_recursion: two_level(7, 8, 2),
        projection_recursion: AuxProjection::Skip,

        witness_decomposition_chunks: 4,
        witness_decomposition_base_log: size.pick(6, 6, 6, 7),

        next: None,
    })
}

/// Root of the p-24 and p-22 chains, narrow enough that its composed witness is already short.
pub fn p_root_aux_short(size: SizeConfig, nof_openings: usize) -> AuxSumcheckConfig {
    let tiny = size == SizeConfig::Tiny;
    chained(AuxSumcheckConfig {
        exact_projection_norm: false,
        witness_height: if tiny { 2usize.pow(10) } else { 2usize.pow(9) },
        witness_width: if tiny { 2usize.pow(7) } else { 2usize.pow(6) },
        projection_ratio: 1,              // no-op
        projection_height: 2usize.pow(8), // no-op,
        basic_commitment_rank: 5,
        nof_openings,
        commitment_recursion: two_level(7, 8, 1),
        opening_recursion: two_level(7, 8, 1),
        projection_recursion: AuxProjection::Skip,

        witness_decomposition_chunks: 4,
        witness_decomposition_base_log: 6,

        next: None,
    })
}

/// The round after the plain root of `size`.
pub fn p_1(size: SizeConfig) -> AuxSumcheckConfig {
    let root = p_root_aux(size, 1);
    chained(round_for(composed_length(&root)))
}

fn level(
    decomposition_base_log: usize,
    decomposition_chunks: usize,
    rank: usize,
    next: Option<AuxRecursionConfig>,
) -> AuxRecursionConfig {
    AuxRecursionConfig {
        decomposition_base_log,
        decomposition_chunks,
        rank,
        next: next.map(Box::new),
    }
}

fn two_level(
    decomposition_base_log: usize,
    decomposition_chunks: usize,
    rank: usize,
) -> AuxRecursionConfig {
    level(
        decomposition_base_log,
        decomposition_chunks,
        rank,
        Some(DECOMP_11_LAST_LEVEL.clone()),
    )
}

/// A round with a fine projection whose commitment, opening and batched-projection recursions
/// share one shape.
struct FineRound {
    height: usize,
    width: usize,
    ratio: usize,
    rank: usize,
    witness_base_log: usize,
    witness_chunks: usize,
    recursion: AuxRecursionConfig,
    constant_term: AuxRecursionConfig,
}

impl FineRound {
    fn aux(self) -> AuxSumcheckConfig {
        AuxSumcheckConfig {
            exact_projection_norm: false,
            witness_height: self.height,
            witness_width: self.width,
            projection_ratio: self.ratio,
            projection_height: 2usize.pow(8),
            basic_commitment_rank: self.rank,
            nof_openings: 2,
            commitment_recursion: self.recursion.clone(),
            opening_recursion: self.recursion.clone(),
            projection_recursion: AuxProjection::Fine {
                nof_batches: 2,
                recursion_constant_term: self.constant_term,
                recursion_batched_projection: self.recursion,
            },
            witness_decomposition_chunks: self.witness_chunks,
            witness_decomposition_base_log: self.witness_base_log,
            next: None,
        }
    }
}

fn composed_length(round: &AuxSumcheckConfig) -> usize {
    let mut open = round.clone();
    open.next = None;
    match open.generate_config() {
        Config::Sumcheck(config) => config.composed_witness_length,
        _ => unreachable!("a sumcheck round generates a sumcheck config"),
    }
}

/// `round` followed by the rounds that take its composed witness down to the terminal one.
fn chained(mut round: AuxSumcheckConfig) -> AuxSumcheckConfig {
    let length = composed_length(&round);
    round.next = Some(Box::new(
        if length == P_LAST.witness_height * P_LAST.witness_width {
            AuxConfig::Simple(P_LAST.clone())
        } else {
            AuxConfig::Sumcheck(chained(round_for(length)))
        },
    ));
    round
}

/// The round that takes a witness of `length` ring elements.
fn round_for(length: usize) -> AuxSumcheckConfig {
    match length.ilog2() {
        17 => FineRound {
            height: 2usize.pow(11),
            width: 2usize.pow(6),
            ratio: 2usize.pow(8),
            rank: 3,
            witness_base_log: 7,
            witness_chunks: 2,
            recursion: two_level(9, 6, 2),
            constant_term: two_level(9, 2, 2),
        },
        16 => FineRound {
            height: 2usize.pow(11),
            width: 2usize.pow(5),
            ratio: 2usize.pow(6),
            rank: 3,
            witness_base_log: 7,
            witness_chunks: 2,
            recursion: two_level(7, 8, 2),
            constant_term: two_level(9, 2, 2),
        },
        15 => FineRound {
            height: 2usize.pow(10),
            width: 2usize.pow(5),
            ratio: 2usize.pow(7),
            rank: 3,
            witness_base_log: 7,
            witness_chunks: 2,
            recursion: two_level(9, 6, 2),
            constant_term: two_level(9, 2, 2),
        },
        14 => FineRound {
            height: 2usize.pow(9),
            width: 2usize.pow(5),
            ratio: 2usize.pow(5),
            rank: 3,
            witness_base_log: 8,
            witness_chunks: 2,
            recursion: two_level(7, 8, 1),
            constant_term: two_level(9, 2, 1),
        },
        13 => FineRound {
            height: 2usize.pow(9),
            width: 2usize.pow(4),
            ratio: 2usize.pow(7),
            rank: 3,
            witness_base_log: 8,
            witness_chunks: 2,
            recursion: two_level(8, 7, 1),
            constant_term: two_level(10, 2, 1),
        },
        12 => FineRound {
            height: 2usize.pow(9),
            width: 2usize.pow(3),
            ratio: 2usize.pow(4),
            rank: 3,
            witness_base_log: 7,
            witness_chunks: 2,
            recursion: two_level(7, 8, 1),
            constant_term: two_level(9, 2, 1),
        },
        11 => FineRound {
            height: 2usize.pow(8),
            width: 2usize.pow(3),
            ratio: 2usize.pow(6),
            rank: 3,
            witness_base_log: 7,
            witness_chunks: 2,
            recursion: level(7, 8, 1, None),
            constant_term: level(9, 2, 1, None),
        },
        10 => FineRound {
            height: 2usize.pow(8),
            width: 2usize.pow(2),
            ratio: 2usize.pow(6),
            rank: 3,
            // the folded witness goes whole into the terminal round: as two digit planes it
            // would leave the round no room to halve the witness
            witness_base_log: 16,
            witness_chunks: 1,
            recursion: level(8, 7, 1, None),
            constant_term: level(9, 2, 1, None),
        },
        log => panic!("no round takes a witness of 2^{log} ring elements"),
    }
    .aux()
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

pub static P_LAST: LazyLock<SimpleConfig> = LazyLock::new(|| SimpleConfig {
    witness_height: 2usize.pow(7),
    witness_width: 2usize.pow(2),
    projection_ratio: 2usize.pow(7),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 2,
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
        // p-28-shaped set: 2^12 x 2^8 ring elements = 2^28 Zq coefficients
        assert_eq!(super::witness_cols_for_target(1 << 12, 1 << 8, 28), 1 << 8);
        // p27 rule: one column-bit fewer
        assert_eq!(super::witness_cols_for_target(1 << 12, 1 << 8, 27), 1 << 7);
        assert_eq!(super::witness_cols_for_target(1 << 12, 1 << 8, 25), 1 << 5);
    }
}
