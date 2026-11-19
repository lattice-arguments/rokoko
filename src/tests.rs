//! End-to-end tests of single rounds: honest proofs verify and round-trip through the wire
//! format, proof and statement shapes follow the config, a context serves one proof, and cut,
//! out-of-bounds or non-canonical proofs are rejected.

use crate::common::init_common;

use crate::ring::ring_arithmetic::RingElement;
use crate::common::structured_row::StructuredRow;
use crate::rok::config::{
    Config, IntermediateRoundProof, RoundProof, SimpleRoundProof, SumcheckConfig,
    SumcheckRoundProof,
};
use crate::commitment::crs::{VerifierCRS, CRS};
use crate::rok::constraints::sumcheck_round::{
    builder::init_sumcheck, context::SumcheckContext,
};
use crate::rok::constraints::sumcheck_round::{
    builder_verifier::init_verifier, context_verifier::VerifierSumcheckContext,
};
use crate::rok::wire;

/// An honest proof of one round of `config`, with the statement a verifier checks it against.
/// The input witness is decomposed into two element-major chunks, which is the top-level
/// hypercube and independent of the round's own `witness_decomposition_chunks`.
struct Honest {
    config: SumcheckConfig,
    crs: VerifierCRS,
    rc_commitment: Vec<RingElement>,
    inner: Vec<StructuredRow>,
    outer: Vec<StructuredRow>,
    claims: Vec<RingElement>,
    proof: SumcheckRoundProof,
}

impl Honest {
    fn prove(config: &SumcheckConfig) -> Honest {
        Honest::prove_witness(config, short_witness)
    }

    /// A proof whose input witness, before its decomposition, is `raw(len)`.
    fn prove_witness(
        config: &SumcheckConfig,
        raw: impl FnOnce(usize) -> Vec<RingElement>,
    ) -> Honest {
        let crs = CRS::gen_prover_crs(config);
        Honest::prove_in(config, &crs, &mut init_sumcheck(&crs, config), raw)
    }

    fn prove_in(
        config: &SumcheckConfig,
        crs: &CRS,
        sumcheck_context: &mut SumcheckContext,
        raw: impl FnOnce(usize) -> Vec<RingElement>,
    ) -> Honest {
        use crate::common::{decomposition::decompose, matrix::VerticallyAlignedMatrix};
        use crate::{
            committer::commit, prover::prover_round,
            rok::evaluation_point_sampler::sample_initial_evaluation_points,
        };

        let input_chunks = 2;
        let base_log = 15;
        let height = config.witness_height / input_chunks;
        let width = config.witness_width;

        let sampled = (0..config.nof_openings)
            .map(|_| sample_initial_evaluation_points(height, width, base_log, input_chunks))
            .collect::<Vec<_>>();
        let inner = sampled
            .iter()
            .flat_map(|points| points.inner.iter().cloned())
            .collect::<Vec<_>>();
        let outer = sampled
            .iter()
            .flat_map(|points| points.outer.iter().cloned())
            .collect::<Vec<_>>();

        let raw = raw(height * width);
        let witness = VerticallyAlignedMatrix {
            height: config.witness_height,
            width,
            used_cols: width,
            data: decompose(&raw, base_log as u64, input_chunks),
        };

        let (commitment_with_aux, rc_commitment) = commit(crs, config, &witness);

        let (proof, claims) = prover_round(
            crs,
            config,
            &commitment_with_aux,
            &witness,
            &inner,
            &outer,
            sumcheck_context,
            true,
            None,
            None,
        );
        let claims = claims.expect("prover must return claims");
        assert_eq!(claims.len(), config.nof_openings);

        Honest {
            config: config.clone(),
            crs: CRS::gen_verifier_crs(config),
            rc_commitment,
            inner,
            outer,
            claims,
            proof,
        }
    }

    /// A copy of the proof, through the wire.
    fn copy(&self) -> SumcheckRoundProof {
        let bytes = wire::to_bytes(&self.proof);
        let copy = wire::from_bytes_for(&self.config, &bytes).expect("an honest proof decodes");
        assert_eq!(wire::to_bytes(&copy), bytes);
        copy
    }

    fn verify(&self, proof: &SumcheckRoundProof) {
        self.verify_in(
            &mut init_verifier(&self.crs, &self.config),
            &self.rc_commitment,
            &self.claims,
            proof,
        );
    }

    fn verify_in(
        &self,
        sumcheck_context: &mut VerifierSumcheckContext,
        rc_commitment: &[RingElement],
        claims: &[RingElement],
        proof: &SumcheckRoundProof,
    ) {
        use crate::verifier::verifier_round;

        verifier_round(
            &self.crs,
            &self.config,
            rc_commitment,
            proof,
            &self.inner,
            &self.outer,
            claims,
            sumcheck_context,
            None,
            None,
        );
    }
}

fn short_witness(len: usize) -> Vec<RingElement> {
    use crate::{
        common::sampling::sample_random_short_vector, ring::ring_arithmetic::Representation,
    };

    sample_random_short_vector(len, 1u64 << 12, Representation::IncompleteNTT)
}

/// Prove and verify one round of `config` end to end.
fn round_trip(config: &SumcheckConfig) {
    round_trip_with(config, |_| {});
}

fn round_trip_with(config: &SumcheckConfig, tamper: impl FnOnce(&mut SumcheckRoundProof)) {
    let mut honest = Honest::prove(config);
    tamper(&mut honest.proof);
    honest.verify(&honest.proof);
}

/// The message a verification panics with, or a panic of its own when it accepts.
fn refusal(verify: impl FnOnce()) -> String {
    let payload = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(verify)) {
        Ok(()) => panic!("the verifier accepted"),
        Err(payload) => payload,
    };
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn sumcheck_config(
    aux: crate::rok::config::generator::AuxSumcheckConfig,
) -> SumcheckConfig {
    match aux.generate_config() {
        Config::Sumcheck(config) => config,
        _ => panic!("expected a sumcheck config"),
    }
}

/// The first round below `proof` that is not a sumcheck round.
fn below(proof: &mut SumcheckRoundProof) -> &mut RoundProof {
    match proof.next.as_deref_mut() {
        Some(RoundProof::Sumcheck(next)) => below(next),
        Some(round) => round,
        None => panic!("the chain ends in a sumcheck round"),
    }
}

fn simple(proof: &mut SumcheckRoundProof) -> &mut SimpleRoundProof {
    match below(proof) {
        RoundProof::Simple(simple) => simple,
        RoundProof::Intermediate(intermediate) => match intermediate.next.as_deref_mut() {
            Some(RoundProof::Simple(simple)) => simple,
            _ => panic!("expected a simple round after the intermediate one"),
        },
        RoundProof::Sumcheck(_) => unreachable!(),
    }
}

fn intermediate(proof: &mut SumcheckRoundProof) -> &mut IntermediateRoundProof {
    match below(proof) {
        RoundProof::Intermediate(intermediate) => intermediate,
        _ => panic!("expected an intermediate round"),
    }
}

/// A round with a non-power-of-two number of openings: the opening commitment is padded to
/// four rows, three of which are placed, one placement each.
#[test]
fn three_openings_round_trip() {
    init_common();

    let generated = three_openings_aux().generate_config();
    let config = match &generated {
        crate::rok::config::Config::Sumcheck(config) => config,
        _ => panic!("expected a sumcheck config"),
    };

    assert_eq!(config.opening_recursion.placements.len(), 3);
    assert_eq!(config.opening_recursion.segments(), 4);

    round_trip(config);
}

#[test]
#[should_panic(expected = "proof ends before the configured round chain")]
fn a_proof_cut_before_the_last_round_is_rejected() {
    init_common();

    let generated = three_openings_aux().generate_config();
    let config = match &generated {
        crate::rok::config::Config::Sumcheck(config) => config,
        _ => panic!("expected a sumcheck config"),
    };

    round_trip_with(config, |proof| proof.next = None);
}

/// The verifier takes every size from the config and none from the proof: a proof whose
/// shapes disagree with the config is refused before any of its entries is used, at every
/// round kind, including shapes that would otherwise only skip a check.
#[test]
fn proof_shapes_follow_the_config() {
    use crate::ring::ring_arithmetic::Representation;
    use crate::rok::config::NextRoundCommitment;

    type Tamper = fn(&mut SumcheckRoundProof);
    fn one() -> RingElement {
        RingElement::constant(1, Representation::IncompleteNTT)
    }
    fn simple_commitment(
        commitment: &mut Option<NextRoundCommitment>,
    ) -> &mut crate::common::matrix::HorizontallyAlignedMatrix<RingElement> {
        match commitment {
            Some(NextRoundCommitment::Simple(commitment)) => commitment,
            _ => panic!("expected a basic next-round commitment"),
        }
    }

    let sumcheck_then_simple: &[(&str, Tamper)] = &[
        ("one round polynomial too few", |p| drop(p.polys.pop())),
        ("a quartic round polynomial", |p| {
            p.polys[0].num_coefficients = 5
        }),
        ("a round polynomial with a non-zero tail", |p| {
            p.polys[0].num_coefficients = 1
        }),
        ("an extra opening root entry", |p| {
            p.rc_opening_inner.push(one())
        }),
        ("no opening root", |p| p.rc_opening_inner.clear()),
        ("no constant-term claims", |p| p.constant_term_claims = None),
        ("one constant-term claim too few", |p| {
            drop(p.constant_term_claims.as_mut().unwrap().pop())
        }),
        ("no fine projection roots", |p| {
            p.rc_fine_projection_inner = None
        }),
        ("a coarse projection root under a fine projection", |p| {
            p.rc_coarse_projection_inner = Some(vec![])
        }),
        ("a projection norm claim the config does not ask for", |p| {
            p.projection_norm_claim = Some(one())
        }),
        ("no next-round commitment", |p| {
            p.next_round_commitment = None
        }),
        (
            "a recursive next-round commitment before a simple round",
            |p| p.next_round_commitment = Some(NextRoundCommitment::Recursive(vec![one()])),
        ),
        ("a narrower next-round commitment", |p| {
            let commitment = simple_commitment(&mut p.next_round_commitment);
            commitment.width -= 1;
            commitment
                .data
                .truncate(commitment.height * commitment.width);
        }),
        ("a batched projection image of height zero", |p| {
            simple(p).batched_projection_image.height = 0
        }),
        ("a projection image with no used columns", |p| {
            simple(p).projection_image_ct.used_cols = 0
        }),
        (
            "a folded witness with more used columns than columns",
            |p| simple(p).folded_witness.used_cols = 77,
        ),
        ("a folded witness one row short", |p| {
            let folded = &mut simple(p).folded_witness;
            folded.height -= 1;
            folded.data.pop();
        }),
        ("an extra opening row", |p| {
            let opening = &mut simple(p).opening_rhs;
            opening.height += 1;
            opening.data.extend((0..opening.width).map(|_| one()));
        }),
        ("a non-zero last opening column", |p| {
            let opening = &mut simple(p).opening_rhs;
            let last = opening.width - 1;
            opening[(0, last)] = one();
        }),
    ];
    let through_an_intermediate_round: &[(&str, Tamper)] = &[
        ("one intermediate round polynomial too few", |p| {
            drop(intermediate(p).polys.pop())
        }),
        ("one intermediate opening row too few", |p| {
            let opening = &mut intermediate(p).opening_rhs;
            opening.height -= 1;
            opening.data.truncate(opening.height * opening.width);
        }),
        (
            "an intermediate batched projection image of height zero",
            |p| intermediate(p).batched_projection_image.height = 0,
        ),
        (
            "an intermediate projection image with no used columns",
            |p| intermediate(p).projection_image_ct.used_cols = 0,
        ),
        ("a recursive commitment out of an intermediate round", |p| {
            intermediate(p).next_round_commitment =
                Some(NextRoundCommitment::Recursive(vec![one()]))
        }),
        ("a non-zero commitment padding row", |p| {
            let commitment = simple_commitment(&mut intermediate(p).next_round_commitment);
            *commitment.data.last_mut().unwrap() = one();
        }),
        (
            "a batched projection image of height zero after an intermediate round",
            |p| simple(p).batched_projection_image.height = 0,
        ),
    ];

    for (aux, cases) in [
        (three_openings_aux(), sumcheck_then_simple),
        (intermediate_aux(), through_an_intermediate_round),
    ] {
        init_common();
        let honest = Honest::prove(&sumcheck_config(aux));
        honest.verify(&honest.copy());
        for (case, tamper) in cases {
            let mut proof = honest.copy();
            tamper(&mut proof);
            let refusal = refusal(|| honest.verify(&proof));
            assert!(refusal.starts_with("proof shape"), "{case}: {refusal}");
        }
    }
}

/// The statement a round is checked against is sized by the config too: below the root it is
/// the previous round's output.
#[test]
fn statement_shapes_follow_the_config() {
    init_common();
    let honest = Honest::prove(&sumcheck_config(three_openings_aux()));
    let proof = honest.copy();
    for (case, rc_commitment, claims) in [
        (
            "one claim too few",
            &honest.rc_commitment[..],
            &honest.claims[1..],
        ),
        ("no commitment root", &[][..], &honest.claims[..]),
    ] {
        let refusal = refusal(|| {
            honest.verify_in(
                &mut init_verifier(&honest.crs, &honest.config),
                rc_commitment,
                claims,
                &proof,
            )
        });
        assert!(refusal.starts_with("proof shape"), "{case}: {refusal}");
    }
}

fn three_openings_aux() -> crate::rok::config::generator::AuxSumcheckConfig {
    use crate::rok::config::SimpleConfig;
    use crate::rok::config::generator::{
        AuxConfig, AuxProjection, AuxRecursionConfig, AuxSumcheckConfig,
    };

    AuxSumcheckConfig {
        exact_projection_norm: false,
        witness_height: 1024,
        witness_width: 16,
        projection_ratio: 32,
        projection_height: 256,
        basic_commitment_rank: 3,
        nof_openings: 3,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 15,
            decomposition_chunks: 4,
            rank: 1,
            next: Some(Box::new(AuxRecursionConfig {
                decomposition_base_log: 7,
                decomposition_chunks: 8,
                rank: 1,
                next: None,
            })),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 15,
            decomposition_chunks: 4,
            rank: 1,
            next: None,
        },
        projection_recursion: AuxProjection::Fine {
            recursion_constant_term: AuxRecursionConfig {
                decomposition_base_log: 15,
                decomposition_chunks: 2,
                rank: 1,
                next: None,
            },
            recursion_batched_projection: AuxRecursionConfig {
                decomposition_base_log: 15,
                decomposition_chunks: 4,
                rank: 1,
                next: None,
            },
        },
        witness_decomposition_chunks: 2,
        witness_decomposition_base_log: 15,
        next: Some(Box::new(AuxConfig::Simple(SimpleConfig {
            witness_height: 256,
            witness_width: 16,
            projection_ratio: crate::ring::config::DEGREE,
            projection_height: 256,
            basic_commitment_rank: 2,
            witness_norm_bound: f64::INFINITY,
            projection_norm_bound: f64::INFINITY,
        }))),
    }
}

/// An honest proof decodes and verifies whatever the witness, including one that is all
/// zero or zero but for one entry, whose proofs carry roles of a single octave.
#[test]
fn honest_proofs_of_sparse_witnesses_decode_and_verify() {
    use crate::ring::ring_arithmetic::Representation;

    fn zero(len: usize) -> Vec<RingElement> {
        vec![RingElement::zero(Representation::IncompleteNTT); len]
    }
    fn one_entry(len: usize) -> Vec<RingElement> {
        let mut witness = zero(len);
        witness[len / 3] = RingElement::constant(5, Representation::IncompleteNTT);
        witness
    }

    for aux in [three_openings_aux(), intermediate_aux()] {
        let config = sumcheck_config(aux);
        for witness in [zero, one_entry] {
            init_common();
            let honest = Honest::prove_witness(&config, witness);
            honest.verify(&honest.copy());
        }
    }
}

/// The decoder allows exactly what an honest proof of the config declares: one coefficient or
/// one round less refuses it, and a proof declaring more is refused before it is allocated,
/// however few bytes its zeros take.
#[test]
fn a_proof_decodes_within_what_its_config_allows() {
    use crate::ring::ring_arithmetic::Representation;
    use crate::rok::wire::{from_bytes_within, Allowance, WireError};

    for aux in [three_openings_aux(), intermediate_aux()] {
        init_common();
        let honest = Honest::prove(&sumcheck_config(aux));
        let bytes = wire::to_bytes(&honest.proof);
        let allowance = Allowance::of(&honest.config);
        assert!(from_bytes_within(&bytes, allowance).is_ok());
        for short in [
            Allowance {
                cells: allowance.cells - 1,
                ..allowance
            },
            Allowance {
                rounds: allowance.rounds - 1,
                ..allowance
            },
        ] {
            assert_eq!(
                from_bytes_within(&bytes, short).err(),
                Some(WireError::Malformed)
            );
        }

        let mut flooded = honest.copy();
        flooded.rc_opening_inner.extend(
            std::iter::repeat_with(|| RingElement::zero(Representation::IncompleteNTT))
                .take(1 << 14),
        );
        assert_eq!(
            wire::from_bytes_for(&honest.config, &wire::to_bytes(&flooded)).err(),
            Some(WireError::Malformed)
        );
    }
}

#[test]
fn a_proof_decodes_only_from_its_own_encoding() {
    for (seed, aux) in [three_openings_aux(), intermediate_aux()]
        .into_iter()
        .enumerate()
    {
        init_common();
        let honest = Honest::prove(&sumcheck_config(aux));
        let bytes = wire::to_bytes(&honest.proof);
        let mut accepted = 0;
        for mutated in wire::tests::mutations(&bytes, 8000, seed as u64) {
            if let Ok(decoded) = wire::from_bytes_for(&honest.config, &mutated) {
                assert_eq!(wire::to_bytes(&decoded), mutated);
                accepted += 1;
            }
        }
        assert!(accepted > 0);
    }
}

/// A context serves one proof: its tables fold and its evaluations cache in place, so a
/// second proof through the same context is refused outright rather than run against what the
/// first one left behind.
#[test]
fn a_context_serves_one_proof() {
    init_common();
    let config = sumcheck_config(three_openings_aux());
    let crs = CRS::gen_prover_crs(&config);
    let mut prover_context = init_sumcheck(&crs, &config);
    let first = Honest::prove_in(&config, &crs, &mut prover_context, short_witness);
    let refusal_to_prove = refusal(|| {
        drop(Honest::prove_in(
            &config,
            &crs,
            &mut prover_context,
            short_witness,
        ))
    });
    assert!(
        refusal_to_prove.starts_with("a sumcheck context serves one proof"),
        "{refusal_to_prove}"
    );

    let second = Honest::prove(&config);
    let mut verifier_context = init_verifier(&first.crs, &config);
    first.verify_in(
        &mut verifier_context,
        &first.rc_commitment,
        &first.claims,
        &first.proof,
    );
    let refusal_to_verify = refusal(|| {
        second.verify_in(
            &mut verifier_context,
            &second.rc_commitment,
            &second.claims,
            &second.proof,
        )
    });
    assert!(
        refusal_to_verify.starts_with("a sumcheck context serves one proof"),
        "{refusal_to_verify}"
    );
    second.verify(&second.proof);
}

/// `three_openings_aux` with an intermediate round before the last one, whose basic
/// commitment has a rank that is not a power of two and so a zero padding row.
fn intermediate_aux() -> crate::rok::config::generator::AuxSumcheckConfig {
    use crate::ring::config::DEGREE;
    use crate::rok::config::{IntermediateConfig, SimpleConfig};
    use crate::rok::config::generator::AuxConfig;

    let mut aux = three_openings_aux();
    aux.next = Some(Box::new(AuxConfig::Intermediate(IntermediateConfig {
        witness_height: 256,
        witness_width: 16,
        projection_ratio: DEGREE,
        projection_height: 256,
        nof_openings: 2,
        basic_commitment_rank: 2,
        witness_decomposition_base_log: 15,
        witness_decomposition_chunks: 2,
        norm_bound: f64::INFINITY,
        projection_norm_bound: f64::INFINITY,
        next: Some(Box::new(Config::Simple(SimpleConfig {
            witness_height: 256,
            witness_width: 2,
            projection_ratio: DEGREE,
            projection_height: 256,
            basic_commitment_rank: 3,
            witness_norm_bound: f64::INFINITY,
            projection_norm_bound: f64::INFINITY,
        }))),
    })));
    aux
}

#[test]
fn intermediate_round_trip() {
    init_common();
    round_trip(&sumcheck_config(intermediate_aux()));
}

/// A round whose folded witness is decomposed into three digit planes: the component is two
/// dyadic blocks and the recomposition is a weighted sum over the planes.
#[test]
fn three_decomposition_chunks_round_trip() {
    use crate::rok::config::SimpleConfig;
    use crate::rok::config::generator::{
        AuxConfig, AuxProjection, AuxRecursionConfig, AuxSumcheckConfig,
    };

    init_common();

    let aux = AuxSumcheckConfig {
        exact_projection_norm: false,
        witness_height: 512,
        witness_width: 16,
        projection_ratio: 32,
        projection_height: 256,
        basic_commitment_rank: 3,
        nof_openings: 1,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 15,
            decomposition_chunks: 4,
            rank: 1,
            next: Some(Box::new(AuxRecursionConfig {
                decomposition_base_log: 7,
                decomposition_chunks: 8,
                rank: 1,
                next: None,
            })),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 15,
            decomposition_chunks: 4,
            rank: 1,
            next: None,
        },
        projection_recursion: AuxProjection::Fine {
            recursion_constant_term: AuxRecursionConfig {
                decomposition_base_log: 15,
                decomposition_chunks: 2,
                rank: 1,
                next: None,
            },
            recursion_batched_projection: AuxRecursionConfig {
                decomposition_base_log: 15,
                decomposition_chunks: 4,
                rank: 1,
                next: None,
            },
        },
        witness_decomposition_chunks: 3,
        witness_decomposition_base_log: 15,
        next: Some(Box::new(AuxConfig::Simple(SimpleConfig {
            witness_height: 256,
            witness_width: 16,
            projection_ratio: crate::ring::config::DEGREE,
            projection_height: 256,
            basic_commitment_rank: 2,
            witness_norm_bound: f64::INFINITY,
            projection_norm_bound: f64::INFINITY,
        }))),
    };

    let generated = aux.generate_config();
    let config = match &generated {
        crate::rok::config::Config::Sumcheck(config) => config,
        _ => panic!("expected a sumcheck config"),
    };

    // 512 * 3 is not a power of two, so the folded witness is two dyadic blocks.
    assert_eq!(config.folded_witness_placement.size, 1536);
    assert_eq!(config.folded_witness_placement.blocks.len(), 2);
    assert_eq!(
        config.folded_witness_placement.blocks[1].length,
        config.folded_witness_placement.blocks[0].length + 1
    );

    round_trip(config);
}

/// A coarse-projection round whose image leaves both i16 and the window of its two base-2^8
/// digits, as the height-8192 level of a two-opening chain does in a few proofs out of a
/// hundred: the round verifies only if the projection and the decomposition of the image both
/// stay exact.
#[test]
fn coarse_projection_past_i16_round_trip() {
    use crate::rok::config::generator::{AuxProjection, AuxRecursionConfig, AuxSumcheckConfig};

    init_common();

    // The height-8192 level's recursions: base-2^7 commitments and base-2^8 projection
    // digits, each ending in one base-2^5 tail.
    let recursion = |decomposition_base_log, decomposition_chunks| AuxRecursionConfig {
        decomposition_base_log,
        decomposition_chunks,
        rank: 2,
        next: Some(Box::new(AuxRecursionConfig {
            decomposition_base_log: 5,
            decomposition_chunks: 11,
            rank: 1,
            next: None,
        })),
    };
    let aux = AuxSumcheckConfig {
        exact_projection_norm: false,
        witness_height: 1024,
        witness_width: 16,
        projection_ratio: 4,
        projection_height: 256,
        basic_commitment_rank: 2,
        nof_openings: 2,
        commitment_recursion: recursion(7, 8),
        opening_recursion: recursion(7, 8),
        projection_recursion: AuxProjection::Coarse(recursion(8, 2)),
        witness_decomposition_chunks: 2,
        witness_decomposition_base_log: 15,
        next: None,
    };

    let generated = aux.generate_config();
    let config = match &generated {
        crate::rok::config::Config::Sumcheck(config) => config,
        _ => panic!("expected a sumcheck config"),
    };

    round_trip(config);
}

/// A component whose size is not a power of two occupies the blocks of its binary
/// decomposition rather than one block of the next size up: three openings cost three
/// opening rows, not four.
#[test]
fn non_power_of_two_component_does_not_round_up() {
    use crate::rok::config::generator::{AuxProjection, AuxRecursionConfig, AuxSumcheckConfig};

    init_common();

    let layout = |nof_openings: usize| AuxSumcheckConfig {
        exact_projection_norm: false,
        witness_height: 256,
        witness_width: 256,
        projection_ratio: 32,
        projection_height: 8,
        basic_commitment_rank: 1,
        nof_openings,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 15,
            decomposition_chunks: 2,
            rank: 1,
            next: None,
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 15,
            decomposition_chunks: 4,
            rank: 1,
            next: None,
        },
        projection_recursion: AuxProjection::Skip,
        witness_decomposition_chunks: 2,
        witness_decomposition_base_log: 15,
        next: None,
    };

    let composed = |nof_openings: usize| match layout(nof_openings).generate_config() {
        crate::rok::config::Config::Sumcheck(config) => config.composed_witness_length,
        _ => panic!("expected a sumcheck config"),
    };

    // folded witness 256*2, one commitment row 256*2, and `nof_openings` opening rows 256*4.
    let demanded = |nof_openings: usize| 512 + 512 + nof_openings * 1024;

    assert_eq!(demanded(3), 4096);
    assert_eq!(composed(3), 4096);

    assert_eq!(demanded(4), 5120);
    assert_eq!(composed(4), 8192);
}
