//! The chains of `crate::instantiation` retuned for `rings/n128_d4.toml`.

use rokoko::common::config::{DEGREE, MOD_Q, NOF_BATCHES, SLOT_DEGREE};
use rokoko::common::short_challenge::{TAU, T_OP_NORM_BOUND};
use rokoko::protocol::{
    config::Config,
    config_generator::{AuxConfig, AuxProjection, AuxRecursionConfig, AuxSumcheckConfig},
    params::{assign_norm_bounds, Instantiation},
};

use crate::instantiation::{initial_witness, root_aux, ParamSet};

const _: () = assert!(
    DEGREE == 128
        && SLOT_DEGREE == 4
        && MOD_Q == 1125899906842177
        && NOF_BATCHES == 3
        && TAU == 22
        && T_OP_NORM_BOUND == 9.8,
    "src/instantiation_n128_d4.rs is tuned for the ring of rings/n128_d4.toml"
);

pub fn instantiation(set: ParamSet) -> Option<Instantiation> {
    Some(Instantiation {
        config: chain(set)?,
        witness: initial_witness(set),
    })
}

pub fn snark_instantiation(_: ParamSet) -> Option<Instantiation> {
    None
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
    let mut root = root_aux(set, 1);
    // base-2^9 digits keep P_5's composed witness at 2^10 with the third batch's row
    let p_5 = from_last(&mut root, 0);
    set_digits(&mut p_5.commitment_recursion, 9, 6);
    set_digits(batched_projection(p_5), 9, 6);
    let mut c = root.generate_config();
    assign_norm_bounds(&mut c, bounds);
    Some(c)
}

/// The sumcheck round of `round`'s chain `k` places from its last one, `P_5`.
fn from_last(round: &mut AuxSumcheckConfig, k: usize) -> &mut AuxSumcheckConfig {
    let mut depth = 0;
    let mut r = &*round;
    while let Some(AuxConfig::Sumcheck(next)) = r.next.as_deref() {
        r = next;
        depth += 1;
    }
    let mut r = round;
    for _ in k..depth {
        let Some(AuxConfig::Sumcheck(next)) = r.next.as_deref_mut() else {
            unreachable!();
        };
        r = next;
    }
    r
}

fn batched_projection(round: &mut AuxSumcheckConfig) -> &mut AuxRecursionConfig {
    match &mut round.projection_recursion {
        AuxProjection::Fine {
            recursion_batched_projection,
            ..
        } => recursion_batched_projection,
        _ => unreachable!("the rounds from P_2 on have fine projections"),
    }
}

fn set_digits(recursion: &mut AuxRecursionConfig, base_log: usize, chunks: usize) {
    recursion.decomposition_base_log = base_log;
    recursion.decomposition_chunks = chunks;
}

const NB_P_22: [[f64; 3]; 6] = [
    [31597.41104900843, 664.8593836293506, f64::INFINITY],
    [33556.08104353069, 939.364146643888, f64::INFINITY],
    [43133.05307997569, 933.3718444435743, f64::INFINITY],
    [22455.68063987373, 935.882471253736, f64::INFINITY],
    [31448.348064723526, 30722.603193739946, f64::INFINITY],
    [148545.0091117167, 349407.17405914835, f64::INFINITY],
];

const NB_P_24: [[f64; 3]; 6] = [
    [44642.79829938979, 672.1926807099286, f64::INFINITY],
    [43422.76602198437, 944.8745948537298, f64::INFINITY],
    [46937.96281263174, 937.4406647889774, f64::INFINITY],
    [22765.535003596993, 933.2282678959098, f64::INFINITY],
    [31315.171977812926, 30584.383204504877, f64::INFINITY],
    [146856.47095719003, 351079.102189236, f64::INFINITY],
];

const NB_P_26: [[f64; 3]; 7] = [
    [52992.09614650094, 940.4302206968894, f64::INFINITY],
    [50805.29161416161, 804.5396199069378, f64::INFINITY],
    [43322.00847144555, 929.4659757086324, f64::INFINITY],
    [47378.804997593594, 934.4056934758049, f64::INFINITY],
    [22756.54951876492, 938.3368265180686, f64::INFINITY],
    [31587.514495445823, 30858.00533086998, f64::INFINITY],
    [149040.65252809384, 359767.9313321297, f64::INFINITY],
];

const NB_P_28: [[f64; 3]; 7] = [
    [75052.57537220158, 933.2786293492421, f64::INFINITY],
    [59601.146264144954, 806.4025049564268, f64::INFINITY],
    [54336.72657604615, 940.0324462485324, f64::INFINITY],
    [50225.83905720242, 942.6982550105839, f64::INFINITY],
    [23042.27391122673, 935.2646684227947, f64::INFINITY],
    [31535.92470500905, 30806.53016813156, f64::INFINITY],
    [147562.35631081526, 354921.2714673495, f64::INFINITY],
];

const NB_P_30: [[f64; 3]; 7] = [
    [159064.19116193312, 934.5843996130045, f64::INFINITY],
    [130534.53854440211, 1144.7982354982908, f64::INFINITY],
    [48033.36796436411, 934.3216790806044, f64::INFINITY],
    [54963.47088749036, 937.2054203855204, f64::INFINITY],
    [22920.521656367247, 933.0546607782419, f64::INFINITY],
    [31567.11979259432, 30833.5111526404, f64::INFINITY],
    [148917.9047999266, 349812.46915454575, f64::INFINITY],
];
