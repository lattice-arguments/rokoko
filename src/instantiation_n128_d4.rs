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
        && TAU == 31
        && T_OP_NORM_BOUND == 11.3,
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
    if set == ParamSet::P28 {
        // the fold outgrows four base-2^6 digits, and base 2^7 needs rank 12 for 128-bit SIS
        root.basic_commitment_rank = 12;
        root.witness_decomposition_base_log = 7;
    }
    let p_3 = from_last(&mut root, 2);
    // base-2^10 digits overflow P_4's base-2^7 fold window
    constant_term(p_3).decomposition_base_log = 9;
    // with rank 5 for 128-bit SIS, P_5's 2^10 composed witness fits only these digits
    let p_5 = from_last(p_3, 0);
    p_5.basic_commitment_rank = 5;
    set_digits(&mut p_5.commitment_recursion, 9, 6);
    set_digits(&mut p_5.opening_recursion, 10, 5);
    set_digits(batched_projection(p_5), 10, 5);
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

fn constant_term(round: &mut AuxSumcheckConfig) -> &mut AuxRecursionConfig {
    match &mut round.projection_recursion {
        AuxProjection::Fine {
            recursion_constant_term,
            ..
        } => recursion_constant_term,
        _ => unreachable!("the rounds from P_2 on have fine projections"),
    }
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
    [31598.375606983343, 664.8593836293506, f64::INFINITY],
    [33585.30718930527, 937.0949791776712, f64::INFINITY],
    [29566.72037950777, 939.3785179574845, f64::INFINITY],
    [22522.740552605937, 941.50305363286, f64::INFINITY],
    [53993.15626077068, 53569.69525767344, f64::INFINITY],
    [301167.36895121954, 613877.4225046886, f64::INFINITY],
];

const NB_P_24: [[f64; 3]; 6] = [
    [44652.98684298733, 672.1926807099286, f64::INFINITY],
    [43428.70740650705, 939.8467960258204, f64::INFINITY],
    [31946.698593125395, 931.3066090176746, f64::INFINITY],
    [22855.352327190234, 933.4655858680597, f64::INFINITY],
    [53977.22650340605, 53550.77276753343, f64::INFINITY],
    [301487.444141543, 608682.5689856742, f64::INFINITY],
];

const NB_P_26: [[f64; 3]; 7] = [
    [53031.955319410954, 940.4302206968894, f64::INFINITY],
    [50785.180043000735, 813.0202949496403, f64::INFINITY],
    [43678.40070332246, 937.0037353180616, f64::INFINITY],
    [32071.432786827594, 936.8991407830407, f64::INFINITY],
    [22792.602286706973, 945.7113724599066, f64::INFINITY],
    [54238.60899765037, 53816.696721370776, f64::INFINITY],
    [302487.0330278638, 633960.9730827285, f64::INFINITY],
];

const NB_P_28: [[f64; 3]; 7] = [
    [112206.82262233434, 934.7357915475367, f64::INFINITY],
    [59446.82708101417, 806.3950644690232, f64::INFINITY],
    [55693.657376760595, 934.0936783856317, f64::INFINITY],
    [33592.34262447322, 932.8504703327324, f64::INFINITY],
    [23116.382069865518, 937.7824907727804, f64::INFINITY],
    [53992.249490088856, 53565.316044993146, f64::INFINITY],
    [305014.437115688, 633967.7738584194, f64::INFINITY],
];

const NB_P_30: [[f64; 3]; 7] = [
    [159143.6493831909, 934.5843996130045, f64::INFINITY],
    [130885.52546022803, 1150.1560763652906, f64::INFINITY],
    [48072.66581956944, 944.2060156554818, f64::INFINITY],
    [35310.26822328032, 935.7825602136428, f64::INFINITY],
    [22991.685866851956, 929.1205519199325, f64::INFINITY],
    [54067.290888299554, 53642.33305888177, f64::INFINITY],
    [299448.58448154334, 593458.5268685588, f64::INFINITY],
];
