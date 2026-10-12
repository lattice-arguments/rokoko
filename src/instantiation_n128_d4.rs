//! The chains of `crate::instantiation` retuned for `rings/n128_d4.toml`.

use rokoko::common::config::{DEGREE, MOD_Q, NOF_BATCHES, SLOT_DEGREE};
use rokoko::common::short_challenge::{TAU, T_OP_NORM_BOUND};
use rokoko::protocol::{
    config::Config,
    config_generator::{AuxConfig, AuxProjection, AuxRecursionConfig, AuxSumcheckConfig},
    params::{assign_norm_bounds, Instantiation},
};

use crate::instantiation::{
    initial_witness, p_head, root_aux, terminal, ParamSet, TAIL_10, TAIL_11, TAIL_12, TAIL_14,
};

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
    let mut root = then(
        root,
        match set {
            ParamSet::P22 => tails(set),
            _ => then(head(set), tails(set)),
        },
    );
    // only these sets' verifier budget covers the 2^7 terminal ratio
    if matches!(set, ParamSet::P22 | ParamSet::P26 | ParamSet::P30) {
        terminal(&mut root).projection_ratio = 2usize.pow(7);
    }
    let mut c = root.generate_config();
    assign_norm_bounds(&mut c, bounds);
    Some(c)
}

fn head(set: ParamSet) -> AuxSumcheckConfig {
    let mut head = p_head(set);
    match set {
        ParamSet::P26 => head.projection_ratio = 2usize.pow(5),
        ParamSet::P30 => head.basic_commitment_rank = 5,
        _ => {}
    }
    head
}

/// Each round sheds the opening row the third batch adds.
fn tails(set: ParamSet) -> AuxSumcheckConfig {
    let mut tail_14 = TAIL_14.clone();
    tail_14.basic_commitment_rank = 4;
    let mut tail_12 = TAIL_12.clone();
    if set == ParamSet::P24 {
        tail_12.projection_ratio = 2usize.pow(6);
    } else {
        set_digits(&mut tail_12.opening_recursion, 7, 7);
    }
    let mut tail_11 = TAIL_11.clone();
    set_digits(&mut tail_11.commitment_recursion, 7, 7);
    set_digits(&mut tail_11.opening_recursion, 7, 7);
    tail_11.witness_decomposition_base_log = 4;
    let mut tail_10 = TAIL_10.clone();
    *constant_term(&mut tail_10) = single_level(8, 2);
    if set == ParamSet::P28 {
        tail_10.projection_ratio = 2usize.pow(5);
        tail_10.commitment_recursion = single_level(8, 6);
        tail_10.opening_recursion = single_level(8, 6);
    } else {
        tail_10.commitment_recursion = single_level(7, 7);
        tail_10.opening_recursion = single_level(7, 7);
    }
    then(tail_14, then(tail_12, then(tail_11, tail_10)))
}

fn then(mut round: AuxSumcheckConfig, next: AuxSumcheckConfig) -> AuxSumcheckConfig {
    round.next = Some(Box::new(AuxConfig::Sumcheck(next)));
    round
}

fn single_level(base_log: usize, chunks: usize) -> AuxRecursionConfig {
    AuxRecursionConfig {
        decomposition_base_log: base_log,
        decomposition_chunks: chunks,
        rank: 2,
        next: None,
    }
}

fn constant_term(round: &mut AuxSumcheckConfig) -> &mut AuxRecursionConfig {
    match &mut round.projection_recursion {
        AuxProjection::FineBatchedInOpening {
            recursion_constant_term,
        } => recursion_constant_term,
        _ => unreachable!("the tails batch their projection in the opening tree"),
    }
}

fn set_digits(recursion: &mut AuxRecursionConfig, base_log: usize, chunks: usize) {
    recursion.decomposition_base_log = base_log;
    recursion.decomposition_chunks = chunks;
}

const NB_P_22: [[f64; 3]; 6] = [
    [31597.69637172938, 664.8593836293506, f64::INFINITY],
    [10641.773583383552, 806.8413722659492, f64::INFINITY],
    [11009.769797775065, 807.5239934515878, f64::INFINITY],
    [11552.691374740347, 10863.416635663018, f64::INFINITY],
    [65186.99179284162, 8214.84004713421, f64::INFINITY],
    [337408.247544425, 732389.6760775646, f64::INFINITY],
];

const NB_P_24: [[f64; 3]; 7] = [
    [44650.16811838451, 672.1926807099286, f64::INFINITY],
    [14433.937196759587, 817.1272850664087, f64::INFINITY],
    [7877.342572213043, 810.0203701142336, f64::INFINITY],
    [7702.501087309239, 799.2659131978543, f64::INFINITY],
    [11223.849651523313, 10872.278877953784, f64::INFINITY],
    [63266.023914894475, 8223.424773170847, f64::INFINITY],
    [363359.894182063, 723773.615755645, f64::INFINITY],
];

const NB_P_26: [[f64; 3]; 7] = [
    [53033.34391682274, 940.4302206968894, f64::INFINITY],
    [16711.645610172567, 812.3435233938903, f64::INFINITY],
    [8595.102966224431, 813.8286060344648, f64::INFINITY],
    [10893.781069949955, 811.305737684629, f64::INFINITY],
    [11554.574288999142, 10876.618500250894, f64::INFINITY],
    [64973.82701519128, 8180.513186836141, f64::INFINITY],
    [359326.17593072733, 683230.1251928225, f64::INFINITY],
];

const NB_P_28: [[f64; 3]; 7] = [
    [112193.01951101949, 934.7357915475367, f64::INFINITY],
    [30187.998559030042, 816.2064689770598, f64::INFINITY],
    [11340.855964167784, 813.5041487294334, f64::INFINITY],
    [11039.993478258943, 806.8512874129904, f64::INFINITY],
    [11550.135497040717, 10866.35881977031, f64::INFINITY],
    [68205.83708158709, 22046.283723113065, f64::INFINITY],
    [385136.6771939541, 757508.6818248356, f64::INFINITY],
];

const NB_P_30: [[f64; 3]; 7] = [
    [159147.59168457435, 934.5843996130045, f64::INFINITY],
    [42342.502051721036, 806.4136655588123, f64::INFINITY],
    [13573.757954229182, 813.0079950406392, f64::INFINITY],
    [11189.345065731059, 808.4905688998481, f64::INFINITY],
    [11598.331690376854, 10882.943535643286, f64::INFINITY],
    [65012.39146501227, 8246.859038446091, f64::INFINITY],
    [377094.79181235057, 658370.5562849846, f64::INFINITY],
];
