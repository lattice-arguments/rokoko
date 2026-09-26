//! The parameter sets for the degree-256 ring. Every round after the root is picked by the length
//! of the witness it takes, so a chain composes by construction and ends at a 128x4 terminal round.

use std::sync::LazyLock;

use rokoko::protocol::{
    config::{Config, SimpleConfig},
    config_generator::{AuxConfig, AuxProjection, AuxRecursionConfig, AuxSumcheckConfig},
    params::{assign_norm_bounds, Instantiation},
};

use crate::instantiation::{initial_witness, ParamSet, DECOMP_11_LAST_LEVEL};

pub fn instantiation(set: ParamSet) -> Option<Instantiation> {
    let root = root_aux(set, 1)?;
    Some(Instantiation {
        witness: initial_witness(&root),
        config: chain(set)?,
    })
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
    let mut c = root_aux(set, 1)?.generate_config();
    assign_norm_bounds(&mut c, bounds);
    Some(c)
}

fn root_aux(set: ParamSet, nof_openings: usize) -> Option<AuxSumcheckConfig> {
    let (height, width, rank, recursion_rank, base_log) = match set {
        ParamSet::P22 => (2usize.pow(9), 2usize.pow(6), 5, 1, 6),
        ParamSet::P24 => (2usize.pow(10), 2usize.pow(7), 5, 1, 6),
        ParamSet::P26 => (2usize.pow(11), 2usize.pow(8), 5, 2, 6),
        ParamSet::P28 => (2usize.pow(13), 2usize.pow(8), 5, 2, 6),
        ParamSet::P29 => return None,
        ParamSet::P30 => (2usize.pow(13), 2usize.pow(10), 6, 2, 7),
    };
    Some(chained(AuxSumcheckConfig {
        exact_projection_norm: false,
        witness_height: height,
        witness_width: width,
        projection_ratio: 1,
        projection_height: 2usize.pow(8),
        basic_commitment_rank: rank,
        nof_openings,
        commitment_recursion: two_level(7, 8, recursion_rank),
        opening_recursion: two_level(7, 8, recursion_rank),
        projection_recursion: AuxProjection::Skip,

        witness_decomposition_chunks: 4,
        witness_decomposition_base_log: base_log,

        next: None,
    }))
}

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

static P_LAST: LazyLock<SimpleConfig> = LazyLock::new(|| SimpleConfig {
    witness_height: 2usize.pow(7),
    witness_width: 2usize.pow(2),
    projection_ratio: 2usize.pow(7),
    projection_height: 2usize.pow(8),
    basic_commitment_rank: 2,
    projection_nof_batches: 2,
    witness_norm_bound: f64::INFINITY,
    projection_norm_bound: f64::INFINITY,
});

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

#[cfg(all(test, ring_degree = "256"))]
mod tests {
    use super::{chain, instantiation, ParamSet};
    use rokoko::common::init_common;
    use rokoko::protocol::config::{config_base_from_config, Config};
    use rokoko::protocol::parties::executor::{execute, execute_to_boundary};
    use std::num::NonZeroUsize;

    fn assert_chain_dims(mut config: &Config) {
        while let Config::Sumcheck(sc) = config {
            let Some(next) = sc.next.as_deref() else {
                break;
            };
            let shape = config_base_from_config(next);
            assert_eq!(
                sc.composed_witness_length,
                shape.witness_height() * shape.witness_width(),
                "composed 2^{} != next round witness {}x{}",
                sc.composed_witness_length.ilog2(),
                shape.witness_height(),
                shape.witness_width(),
            );
            config = next;
        }
    }

    #[test]
    fn chains_compose() {
        for set in ParamSet::ALL {
            if let Some(config) = chain(set) {
                assert_chain_dims(&config);
            }
        }
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

    #[test]
    fn full_chain_verifies() {
        init_common();
        execute(&instantiation(ParamSet::P26).unwrap());
    }

    #[test]
    fn round_boundary_extraction() {
        init_common();
        let inst = instantiation(ParamSet::P26).unwrap();
        let shape_at = |boundary: usize| {
            let mut round = &inst.config;
            for _ in 0..boundary {
                let Config::Sumcheck(config) = round else {
                    panic!("expected a sumcheck round before the boundary");
                };
                round = config.next.as_deref().unwrap();
            }
            let shape = config_base_from_config(round);
            (shape.witness_height(), shape.witness_width())
        };

        let mut run = execute_to_boundary(&inst, NonZeroUsize::new(3).unwrap());
        assert_eq!(
            (run.prover.witness.height, run.prover.witness.width),
            shape_at(3)
        );
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

        let run4 = execute_to_boundary(&inst, NonZeroUsize::new(4).unwrap());
        assert_eq!(
            (run4.prover.witness.height, run4.prover.witness.width),
            shape_at(4)
        );
        assert_eq!(
            run4.prover.evaluation_points,
            run4.verifier.evaluation_points
        );
    }
}
