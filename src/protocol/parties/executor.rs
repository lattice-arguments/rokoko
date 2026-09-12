use std::num::NonZeroUsize;

use crate::{
    common::{matrix::VerticallyAlignedMatrix, ring_arithmetic::RingElement},
    protocol::{
        boundary::{BoundaryCapture, ProverBoundary, VerifierBoundary},
        config::{to_kb, Config, SizeableProof, CONFIG},
        crs::{VerifierCRS, CRS},
        evaluation_point_sampler::{sample_initial_evaluation_points, InitialEvaluationPoints},
        open::claim,
        params::{witness_sampler, WITNESS_CONFIG},
        parties::{commiter::commit, prover::prover_round, verifier::verifier_round},
        sumcheck::init_sumcheck,
        sumchecks::builder_verifier::init_verifier,
        wire,
    },
};

pub struct BoundaryRun {
    pub prover: ProverBoundary,
    pub verifier: VerifierBoundary,
    pub crs: CRS,
    pub verifier_crs: VerifierCRS,
    pub proof_size_bits: usize,
}

fn run(
    cut: Option<NonZeroUsize>,
) -> (
    usize,
    Option<ProverBoundary>,
    Option<VerifierBoundary>,
    CRS,
    VerifierCRS,
) {
    // check_prefixing_correctness(&CONFIG);
    let config = match &*CONFIG {
        Config::Sumcheck(config) => config,
        _ => panic!("Expected sumcheck config at the top level."),
    };

    let witness_config = &*WITNESS_CONFIG;

    let evaluation_points = sample_initial_evaluation_points(
        witness_config.height,
        witness_config.width,
        witness_config.decomposition_base_log,
        witness_config.decomposition_chunks,
    );

    tracing::debug!("Generating CRS...");

    let crs_start = std::time::Instant::now();
    let crs = CRS::gen_prover_crs(&config);
    let verifier_crs = CRS::gen_verifier_crs(&config);
    let crs_duration = crs_start.elapsed().as_nanos();
    println!("TOTAL CRS gen time: {:?} ns", crs_duration);

    let mut sumcheck_context = init_sumcheck(&crs, &config);
    let mut sumcheck_context_verifier = init_verifier(&verifier_crs, &config);
    let witness = witness_sampler();

    let start = std::time::Instant::now();

    let commit_span = tracing::info_span!("commit").entered();
    #[cfg(not(feature = "crt-commitment"))]
    let (witness_decomposed, commitment_with_aux, rc_commitment) = {
        let witness_decomposed = crate::protocol::params::decompose_witness(&witness);
        let (commitment_with_aux, rc_commitment) = commit(&crs, &config, &witness_decomposed, None);
        (witness_decomposed, commitment_with_aux, rc_commitment)
    };
    #[cfg(feature = "crt-commitment")]
    let (witness_decomposed, commitment_with_aux, rc_commitment) = {
        // Handing the digits over beats re-deriving them, but only while the array is small
        // enough to allocate cheaply; past that the commitment narrows a column tile at a time.
        const DIGIT_BUDGET: usize = 2 << 30;
        let wanted = witness.data.len()
            * WITNESS_CONFIG.decomposition_chunks
            * std::mem::size_of::<crate::protocol::project_coarse::Signed16RingElement>()
            <= DIGIT_BUDGET;
        let mut sink = Vec::new();
        let (witness_decomposed, digits) = crate::protocol::params::decompose_witness_with_digits(
            &witness,
            wanted.then_some(&mut sink),
        );
        let (commitment_with_aux, rc_commitment) =
            commit(&crs, &config, &witness_decomposed, digits.as_ref());
        (witness_decomposed, commitment_with_aux, rc_commitment)
    };
    drop(commit_span);

    let commit_duration = start.elapsed().as_nanos();
    println!("TOTAL Commit time: {:?} ns", commit_duration);

    let boundary_note = if cut.is_some() { " (to boundary)" } else { "" };

    let start = std::time::Instant::now();

    let mut prover_boundary = None;
    let prover_span = tracing::info_span!("prover").entered();
    let (proof, claims) = prover_round(
        &crs,
        &config,
        &commitment_with_aux,
        &witness_decomposed,
        &evaluation_points.inner,
        &evaluation_points.outer,
        &mut sumcheck_context,
        true,
        None,
        cut.map(|cut| BoundaryCapture {
            cut,
            slot: &mut prover_boundary,
        }),
    );
    drop(prover_span);
    let claims = claims.expect("Prover round must return claims when with_claims is true.");
    {
        let _s = tracing::info_span!("verify_claims").entered();
        check_prover_claims_match_witness(&witness, &evaluation_points, &claims);
    }

    let prover_duration = start.elapsed().as_nanos();
    println!(
        "TOTAL Prover time{}: {:?} ns",
        boundary_note, prover_duration
    );

    let proof_size_bits = proof.size_in_bits();
    println!(
        "Total proof size{}: {} KB",
        boundary_note,
        to_kb(proof_size_bits)
    );

    let start = std::time::Instant::now();
    let bytes = wire::to_bytes(&proof);
    let serialise = start.elapsed();
    let start = std::time::Instant::now();
    let proof = wire::from_bytes(&bytes).expect("the proof must deserialise");
    let deserialise = start.elapsed();
    println!(
        "Wire proof size: {} KB (serialise {:.3} ms, deserialise {:.3} ms)",
        bytes.len() as f64 / 1024.0,
        serialise.as_secs_f64() * 1e3,
        deserialise.as_secs_f64() * 1e3
    );

    let start = std::time::Instant::now();
    let mut verifier_boundary = None;
    let verifier_span = tracing::info_span!("verifier").entered();
    verifier_round(
        &verifier_crs,
        &config,
        &rc_commitment,
        &proof,
        &evaluation_points.inner,
        &evaluation_points.outer,
        &claims,
        &mut sumcheck_context_verifier,
        None,
        cut.map(|cut| BoundaryCapture {
            cut,
            slot: &mut verifier_boundary,
        }),
    );
    drop(verifier_span);

    let verifier_duration = start.elapsed().as_nanos();
    println!(
        "TOTAL Verifier time{}: {:?} ns",
        boundary_note, verifier_duration
    );

    (
        proof_size_bits,
        prover_boundary,
        verifier_boundary,
        crs,
        verifier_crs,
    )
}

pub fn execute() {
    run(None);
}

pub fn execute_to_boundary(cut: NonZeroUsize) -> BoundaryRun {
    let (proof_size_bits, prover_boundary, verifier_boundary, crs, verifier_crs) = run(Some(cut));
    BoundaryRun {
        prover: prover_boundary.expect("execute_to_boundary must populate the prover boundary"),
        verifier: verifier_boundary
            .expect("execute_to_boundary must populate the verifier boundary"),
        crs,
        verifier_crs,
        proof_size_bits,
    }
}

fn check_prover_claims_match_witness(
    witness: &VerticallyAlignedMatrix<RingElement>,
    evaluation_points: &InitialEvaluationPoints,
    prover_claims: &[RingElement],
) {
    assert_eq!(
        prover_claims.len(),
        evaluation_points.witness_inner.len(),
        "Prover returned a different number of claims than sampled witness points."
    );

    for (i, ((inner, outer), prover_claim)) in evaluation_points
        .witness_inner
        .iter()
        .zip(evaluation_points.outer.iter())
        .zip(prover_claims.iter())
        .enumerate()
    {
        let mut expected_claim = claim(witness, inner, outer);
        expected_claim *= &evaluation_points.witness_claim_scale;
        assert_eq!(
            &expected_claim, prover_claim,
            "Prover claim {i} does not match the direct witness claim."
        );
    }
}

/// SNARK mode: prove user-supplied sumcheck claims about a committed witness,
/// then run the PCS chain on the resulting evaluation claims.
pub fn execute_snark() {
    use crate::common::{
        hash::HashWrapper,
        ring_arithmetic::{Representation, RingElement},
        sampling::sample_random_short_vector,
    };
    use crate::protocol::params::P_EN_TWO_EVALS;
    use crate::protocol::snark::{eq, prove_claims, verify_claims, witness_in, Claim, Region};

    let config = match &*P_EN_TWO_EVALS {
        Config::Sumcheck(config) => config,
        _ => panic!("Expected sumcheck config at the top level."),
    };

    tracing::debug!("Generating CRS...");
    let crs = CRS::gen_prover_crs(&config);
    let verifier_crs = CRS::gen_verifier_crs(&config);

    let mut sumcheck_context = init_sumcheck(&crs, &config);
    let mut sumcheck_context_verifier = init_verifier(&verifier_crs, &config);

    let witness = VerticallyAlignedMatrix {
        height: config.witness_height,
        width: config.witness_width,
        used_cols: config.witness_width,
        data: sample_random_short_vector(
            config.witness_height * config.witness_width,
            2u64.pow(7),
            crate::common::ring_arithmetic::Representation::IncompleteNTT,
        ),
    };

    let start = std::time::Instant::now();
    let _commit_span = tracing::info_span!("commit").entered();
    let (commitment_with_aux, rc_commitment) = commit(&crs, &config, &witness, None);
    drop(_commit_span);
    println!("TOTAL Commit time: {:?} ns", start.elapsed().as_nanos());

    let total_vars = (config.witness_height * config.witness_width).ilog2() as usize;
    let n = config.witness_height * config.witness_width;
    let everything = Region::whole(n);

    let structured_point: Vec<RingElement> = (0..total_vars)
        .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 1 << 10))
        .collect();
    let t1 = {
        use crate::common::structured_row::PreprocessedRow;
        let expanded = PreprocessedRow::from_layers(&structured_point).preprocessed_row;
        let mut acc = RingElement::zero(Representation::IncompleteNTT);
        let mut temp = RingElement::zero(Representation::IncompleteNTT);
        for (a, w) in expanded.iter().zip(witness.data.iter()) {
            temp *= (a, w);
            acc += &temp;
        }
        acc
    };
    let claim_linear = Claim::sums_to(eq(structured_point) * witness_in(everything), t1);

    let segment = Region::new(n / 4, n / 4, n);
    let mut t2 = RingElement::zero(Representation::IncompleteNTT);
    {
        let mut temp = RingElement::zero(Representation::IncompleteNTT);
        for w in &witness.data[segment.range()] {
            temp *= (w, w);
            t2 += &temp;
        }
    }
    let claim_square = Claim::sums_to(witness_in(segment) * witness_in(segment), t2);

    // P_EN_TWO_EVALS is compiled for two openings, so the statement must use
    // the conjugate; the norm claim is the natural way.
    let mut t3 = RingElement::zero(Representation::IncompleteNTT);
    {
        let mut temp = RingElement::zero(Representation::IncompleteNTT);
        for w in &witness.data {
            temp *= (w, &w.conjugate());
            t3 += &temp;
        }
    }
    let claim_norm = Claim::sums_to(
        witness_in(everything) * witness_in(everything).conjugate(),
        t3,
    );

    let claims = vec![claim_linear, claim_square, claim_norm];

    let start = std::time::Instant::now();
    let _prover_span = tracing::info_span!("prover").entered();

    let mut hash_wrapper = HashWrapper::new();
    hash_wrapper.update_with_ring_element_slice(
        &commitment_with_aux
            .rc_commitment_with_aux
            .most_inner_commitment(),
    );

    let (initial_proof, chain_inputs) = {
        let _s = tracing::info_span!("prover::claims").entered();
        prove_claims(&witness, &claims, &mut hash_wrapper)
    };

    println!(
        "Initial claims sumcheck done: {} ms",
        start.elapsed().as_millis()
    );

    let (proof, _) = prover_round(
        &crs,
        &config,
        &commitment_with_aux,
        &witness,
        &chain_inputs.evaluation_points_inner,
        &chain_inputs.evaluation_points_outer,
        &mut sumcheck_context,
        false,
        Some(hash_wrapper),
        None,
    );
    drop(_prover_span);
    println!("TOTAL Prover time: {:?} ns", start.elapsed().as_nanos());

    let proof_size_bits = proof.size_in_bits() + initial_proof.size_in_bits();
    println!("Total proof size: {} KB", to_kb(proof_size_bits));

    let start = std::time::Instant::now();
    let bytes = wire::to_bytes(&proof);
    let initial_bytes = wire::initial_to_bytes(&initial_proof);
    let serialise = start.elapsed();
    let start = std::time::Instant::now();
    let proof = wire::from_bytes(&bytes).expect("the proof must deserialise");
    let initial_proof =
        wire::initial_from_bytes(&initial_bytes).expect("the initial proof must deserialise");
    let deserialise = start.elapsed();
    println!(
        "Wire proof size: {} KB (serialise {:.3} ms, deserialise {:.3} ms)",
        (bytes.len() + initial_bytes.len()) as f64 / 1024.0,
        serialise.as_secs_f64() * 1e3,
        deserialise.as_secs_f64() * 1e3
    );

    let start = std::time::Instant::now();
    let _verifier_span = tracing::info_span!("verifier").entered();

    let mut hash_wrapper_verifier = HashWrapper::new();
    hash_wrapper_verifier.update_with_ring_element_slice(&rc_commitment);

    let chain_inputs_verifier = {
        let _s = tracing::info_span!("verifier::claims").entered();
        verify_claims(
            (config.witness_height, config.witness_width),
            &claims,
            &initial_proof,
            &mut hash_wrapper_verifier,
        )
    };

    verifier_round(
        &verifier_crs,
        &config,
        &rc_commitment,
        &proof,
        &chain_inputs_verifier.evaluation_points_inner,
        &chain_inputs_verifier.evaluation_points_outer,
        &chain_inputs_verifier.claims,
        &mut sumcheck_context_verifier,
        Some(hash_wrapper_verifier),
        None,
    );
    drop(_verifier_span);
    println!("TOTAL Verifier time: {:?} ns", start.elapsed().as_nanos());
}

#[cfg(test)]
mod tests {
    use super::execute_to_boundary;
    use crate::common::init_common;
    use std::num::NonZeroUsize;

    /// The boundary tests stop a few rounds in, so they never reach the last sumcheck round,
    /// whose recursions are single levels and whose level 0 is therefore itself a leaf. Only a
    /// whole-chain run covers it.
    #[cfg(not(feature = "p-29"))]
    #[test]
    fn full_chain_verifies() {
        init_common();
        super::execute();
    }

    #[cfg(not(any(feature = "p-22", feature = "p-24", feature = "p-29")))]
    #[test]
    fn round_boundary_extraction() {
        init_common();
        let mut run = execute_to_boundary(NonZeroUsize::new(3).unwrap());

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

        // Both parties read the same rows off the public seed.
        assert_eq!(run.crs.cks.len(), run.verifier_crs.cks.len());
        for (prover, verifier) in run.crs.cks.iter().zip(&run.verifier_crs.cks) {
            assert_eq!(prover.len(), verifier.len());
            assert!(prover
                .iter()
                .zip(verifier)
                .all(|(a, b)| a.preprocessed_row == b.preprocessed_row));
        }

        let run4 = execute_to_boundary(NonZeroUsize::new(4).unwrap());
        assert_eq!(run4.prover.witness.height, 512);
        assert_eq!(run4.prover.witness.width, 8);
        assert_eq!(
            run4.prover.evaluation_points,
            run4.verifier.evaluation_points
        );
    }

    /// Prove and verify one round of `config` end to end. The input witness is decomposed into
    /// two element-major chunks, which is the top-level hypercube and independent of the round's
    /// own `witness_decomposition_chunks`.
    fn round_trip(config: &crate::protocol::config::SumcheckConfig) {
        assert!(
            round_trip_perturbed(config, |_| {}),
            "the honest round must verify"
        );
    }

    /// The same round, with `perturb` free to corrupt the commitment the prover proves about
    /// after the public value the verifier anchors to has been taken off it. Answers whether the
    /// verifier accepted.
    fn round_trip_perturbed(
        config: &crate::protocol::config::SumcheckConfig,
        perturb: impl FnOnce(&mut crate::protocol::commitment::CommitmentWithAux),
    ) -> bool {
        use crate::common::{
            decomposition::decompose, matrix::VerticallyAlignedMatrix,
            ring_arithmetic::Representation, sampling::sample_random_short_vector,
        };
        use crate::protocol::{
            crs::CRS,
            evaluation_point_sampler::sample_initial_evaluation_points,
            parties::{commiter::commit, prover::prover_round, verifier::verifier_round},
            sumcheck::init_sumcheck,
            sumchecks::builder_verifier::init_verifier,
        };

        let crs = CRS::gen_prover_crs(config);
        let verifier_crs = CRS::gen_verifier_crs(config);
        let mut sumcheck_context = init_sumcheck(&crs, config);
        let mut sumcheck_context_verifier = init_verifier(&verifier_crs, config);

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

        let raw =
            sample_random_short_vector(height * width, 1u64 << 12, Representation::IncompleteNTT);
        let witness = VerticallyAlignedMatrix {
            height: config.witness_height,
            width,
            used_cols: width,
            data: decompose(&raw, base_log as u64, input_chunks),
        };

        let (mut commitment_with_aux, rc_commitment) = commit(&crs, config, &witness, None);
        perturb(&mut commitment_with_aux);

        let (proof, claims) = prover_round(
            &crs,
            config,
            &commitment_with_aux,
            &witness,
            &inner,
            &outer,
            &mut sumcheck_context,
            true,
            None,
            None,
        );
        let claims = claims.expect("prover must return claims");
        assert_eq!(claims.len(), config.nof_openings);

        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            verifier_round(
                &verifier_crs,
                config,
                &rc_commitment,
                &proof,
                &inner,
                &outer,
                &claims,
                &mut sumcheck_context_verifier,
                None,
                None,
            );
        }))
        .is_ok()
    }

    /// A round with a non-power-of-two number of openings: the opening commitment is padded to
    /// four rows, three of which are placed, one placement each.
    #[test]
    fn three_openings_round_trip() {
        use crate::protocol::config::SimpleConfig;
        use crate::protocol::config_generator::{
            AuxConfig, AuxProjection, AuxRecursionConfig, AuxSumcheckConfig,
        };

        init_common();

        let aux = AuxSumcheckConfig {
            exact_projection_norm: false,
            witness_height: 1024,
            witness_width: 16,
            projection_ratio: 32,
            projection_height: 256,
            basic_commitment_rank: 3,
            basic_commitment_diag_blocks: 1,
            nof_openings: 3,
            commitment_recursion: AuxRecursionConfig {
                decomposition_base_log: 15,
                decomposition_chunks: 4,
                rank: 1,
                diag_blocks: 1,
                next: Some(Box::new(AuxRecursionConfig {
                    decomposition_base_log: 7,
                    decomposition_chunks: 8,
                    rank: 1,
                    diag_blocks: 1,
                    next: None,
                })),
            },
            opening_recursion: AuxRecursionConfig {
                decomposition_base_log: 15,
                decomposition_chunks: 4,
                rank: 1,
                diag_blocks: 1,
                next: None,
            },
            projection_recursion: AuxProjection::Fine {
                nof_batches: 2,
                recursion_constant_term: AuxRecursionConfig {
                    decomposition_base_log: 15,
                    decomposition_chunks: 2,
                    rank: 1,
                    diag_blocks: 1,
                    next: None,
                },
                recursion_batched_projection: AuxRecursionConfig {
                    decomposition_base_log: 15,
                    decomposition_chunks: 4,
                    rank: 1,
                    diag_blocks: 1,
                    next: None,
                },
            },
            witness_decomposition_chunks: 2,
            witness_decomposition_base_log: 15,
            next: Some(Box::new(AuxConfig::Simple(SimpleConfig {
                witness_height: 256,
                witness_width: 16,
                projection_ratio: 128,
                projection_height: 256,
                projection_nof_batches: 2,
                basic_commitment_rank: 2,
                witness_norm_bound: f64::INFINITY,
                projection_norm_bound: f64::INFINITY,
            }))),
        };

        let generated = aux.generate_config();
        let config = match &generated {
            crate::protocol::config::Config::Sumcheck(config) => config,
            _ => panic!("expected a sumcheck config"),
        };

        assert_eq!(config.opening_recursion.placements.len(), 3);
        assert_eq!(config.opening_recursion.segments(), 4);

        round_trip(config);
    }

    /// A round whose folded witness is decomposed into three digit planes: the component is two
    /// dyadic blocks and the recomposition is a weighted sum over the planes.
    #[test]
    fn three_decomposition_chunks_round_trip() {
        use crate::protocol::config::SimpleConfig;
        use crate::protocol::config_generator::{
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
            basic_commitment_diag_blocks: 1,
            nof_openings: 1,
            commitment_recursion: AuxRecursionConfig {
                decomposition_base_log: 15,
                decomposition_chunks: 4,
                rank: 1,
                diag_blocks: 1,
                next: Some(Box::new(AuxRecursionConfig {
                    decomposition_base_log: 7,
                    decomposition_chunks: 8,
                    rank: 1,
                    diag_blocks: 1,
                    next: None,
                })),
            },
            opening_recursion: AuxRecursionConfig {
                decomposition_base_log: 15,
                decomposition_chunks: 4,
                rank: 1,
                diag_blocks: 1,
                next: None,
            },
            projection_recursion: AuxProjection::Fine {
                nof_batches: 2,
                recursion_constant_term: AuxRecursionConfig {
                    decomposition_base_log: 15,
                    decomposition_chunks: 2,
                    rank: 1,
                    diag_blocks: 1,
                    next: None,
                },
                recursion_batched_projection: AuxRecursionConfig {
                    decomposition_base_log: 15,
                    decomposition_chunks: 4,
                    rank: 1,
                    diag_blocks: 1,
                    next: None,
                },
            },
            witness_decomposition_chunks: 3,
            witness_decomposition_base_log: 15,
            next: Some(Box::new(AuxConfig::Simple(SimpleConfig {
                witness_height: 256,
                witness_width: 16,
                projection_ratio: 128,
                projection_height: 256,
                projection_nof_batches: 2,
                basic_commitment_rank: 2,
                witness_norm_bound: f64::INFINITY,
                projection_norm_bound: f64::INFINITY,
            }))),
        };

        let generated = aux.generate_config();
        let config = match &generated {
            crate::protocol::config::Config::Sumcheck(config) => config,
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

    /// A round that commits block-diagonally, at the basic commitment and at a recursion level:
    /// the witness is cut in two, one short key meets both halves, and the commitment keeps its
    /// full rank.
    fn diag_blocks_config(
        basic_commitment_rank: usize,
        opening_blocks: usize,
    ) -> crate::protocol::config_generator::AuxSumcheckConfig {
        use crate::protocol::config::SimpleConfig;
        use crate::protocol::config_generator::{
            AuxConfig, AuxProjection, AuxRecursionConfig, AuxSumcheckConfig,
        };

        AuxSumcheckConfig {
            exact_projection_norm: false,
            witness_height: 1024,
            witness_width: 16,
            projection_ratio: 32,
            projection_height: 256,
            basic_commitment_rank,
            basic_commitment_diag_blocks: 2,
            nof_openings: 1,
            commitment_recursion: AuxRecursionConfig {
                decomposition_base_log: 15,
                decomposition_chunks: 4,
                rank: 2,
                diag_blocks: 2,
                next: Some(Box::new(AuxRecursionConfig {
                    decomposition_base_log: 7,
                    decomposition_chunks: 8,
                    rank: 1,
                    diag_blocks: 1,
                    next: None,
                })),
            },
            // A single placed block is one piece of the input, so a block cuts the piece in
            // two and each half meets the whole key row.
            opening_recursion: AuxRecursionConfig {
                decomposition_base_log: 15,
                decomposition_chunks: 4,
                rank: opening_blocks,
                diag_blocks: opening_blocks,
                next: None,
            },
            projection_recursion: AuxProjection::Fine {
                nof_batches: 2,
                recursion_constant_term: AuxRecursionConfig {
                    decomposition_base_log: 15,
                    decomposition_chunks: 2,
                    rank: 1,
                    diag_blocks: 1,
                    next: None,
                },
                recursion_batched_projection: AuxRecursionConfig {
                    decomposition_base_log: 15,
                    decomposition_chunks: 4,
                    rank: 1,
                    diag_blocks: 1,
                    next: None,
                },
            },
            witness_decomposition_chunks: 2,
            witness_decomposition_base_log: 15,
            next: Some(Box::new(AuxConfig::Simple(SimpleConfig {
                witness_height: 256,
                witness_width: 16,
                projection_ratio: 128,
                projection_height: 256,
                projection_nof_batches: 2,
                basic_commitment_rank: 2,
                witness_norm_bound: f64::INFINITY,
                projection_norm_bound: f64::INFINITY,
            }))),
        }
    }

    #[test]
    fn diag_blocks_round_trip() {
        init_common();

        let generated = diag_blocks_config(4, 2).generate_config();
        let config = match &generated {
            crate::protocol::config::Config::Sumcheck(config) => config,
            _ => panic!("expected a sumcheck config"),
        };

        assert_eq!(config.composed_witness_length, 4096);
        assert_eq!(config.commitment_recursion.block_len(), 128);
        assert_eq!(config.commitment_recursion.blockwise_rank(), 1);

        round_trip(config);
    }

    /// One key row meets all four blocks of a recursion level, so the verifier evaluates its
    /// slices once and reuses them; only the per-block selector differs.
    #[test]
    fn four_blocks_recursion_level_round_trip() {
        init_common();

        let generated = diag_blocks_config(4, 4).generate_config();
        let config = match &generated {
            crate::protocol::config::Config::Sumcheck(config) => config,
            _ => panic!("expected a sumcheck config"),
        };

        assert_eq!(config.opening_recursion.diag_blocks, 4);
        assert_eq!(config.opening_recursion.blockwise_rank(), 1);
        assert_eq!(config.opening_recursion.block_len(), 16);
        assert_eq!(config.composed_witness_length, 4096);

        round_trip(config);
    }

    /// The basic commitment's rows are independent constraints, so its rank need not be dyadic
    /// when it commits block-diagonally: three key rows meet each of the two halves.
    #[test]
    fn non_dyadic_rank_with_blocks_round_trip() {
        init_common();

        let generated = diag_blocks_config(6, 2).generate_config();
        let config = match &generated {
            crate::protocol::config::Config::Sumcheck(config) => config,
            _ => panic!("expected a sumcheck config"),
        };

        assert_eq!(config.basic_commitment_rank, 6);
        assert_eq!(config.basic_commitment_diag_blocks, 2);
        assert_eq!(config.composed_witness_length, 4096);

        round_trip(config);
    }

    /// The rows of a commitment are one sumcheck constraint between them, so the control that
    /// every row is still constrained is that corrupting any single one of them is rejected. The
    /// first digit of a row is placed and carries radix one, and touching it breaks that row's
    /// CommitmentFold term and the layer that commits it alike.
    #[test]
    fn a_corrupt_commitment_row_is_rejected() {
        use crate::common::ring_arithmetic::{Representation, RingElement};

        init_common();

        let generated = diag_blocks_config(6, 2).generate_config();
        let config = match &generated {
            crate::protocol::config::Config::Sumcheck(config) => config,
            _ => panic!("expected a sumcheck config"),
        };

        let bump = |data: &mut Vec<RingElement>, rows: usize, row: usize| {
            let stride = data.len() / rows.next_power_of_two();
            data[row * stride] += &RingElement::constant(1, Representation::IncompleteNTT);
        };

        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        let rows = config.basic_commitment_rank;
        let outcomes = (0..rows)
            .map(|row| {
                round_trip_perturbed(config, |aux| {
                    bump(&mut aux.rc_commitment_with_aux.committed_data, rows, row)
                })
            })
            .collect::<Vec<_>>();

        let elements = config.commitment_recursion.rank;
        let inner_outcomes = (0..elements)
            .map(|element| {
                round_trip_perturbed(config, |aux| {
                    let child = aux.rc_commitment_with_aux.next.as_mut().unwrap();
                    bump(&mut child.committed_data, elements, element)
                })
            })
            .collect::<Vec<_>>();

        std::panic::set_hook(hook);

        for (row, accepted) in outcomes.iter().enumerate() {
            assert!(
                !accepted,
                "a corrupt digit of commitment row {row} verified"
            );
        }
        for (element, accepted) in inner_outcomes.iter().enumerate() {
            assert!(
                !accepted,
                "a corrupt digit of recursion element {element} verified"
            );
        }
    }

    /// A component whose size is not a power of two occupies the blocks of its binary
    /// decomposition rather than one block of the next size up: three openings cost three
    /// opening rows, not four.
    #[test]
    fn non_power_of_two_component_does_not_round_up() {
        use crate::protocol::config_generator::{
            AuxProjection, AuxRecursionConfig, AuxSumcheckConfig,
        };

        init_common();

        let layout = |nof_openings: usize| AuxSumcheckConfig {
            exact_projection_norm: false,
            witness_height: 256,
            witness_width: 256,
            projection_ratio: 32,
            projection_height: 8,
            basic_commitment_rank: 1,
            basic_commitment_diag_blocks: 1,
            nof_openings,
            commitment_recursion: AuxRecursionConfig {
                decomposition_base_log: 15,
                decomposition_chunks: 2,
                rank: 1,
                diag_blocks: 1,
                next: None,
            },
            opening_recursion: AuxRecursionConfig {
                decomposition_base_log: 15,
                decomposition_chunks: 4,
                rank: 1,
                diag_blocks: 1,
                next: None,
            },
            projection_recursion: AuxProjection::Skip,
            witness_decomposition_chunks: 2,
            witness_decomposition_base_log: 15,
            next: None,
        };

        let composed = |nof_openings: usize| match layout(nof_openings).generate_config() {
            crate::protocol::config::Config::Sumcheck(config) => config.composed_witness_length,
            _ => panic!("expected a sumcheck config"),
        };

        // folded witness 256*2, one commitment row 256*2, and `nof_openings` opening rows 256*4.
        let demanded = |nof_openings: usize| 512 + 512 + nof_openings * 1024;

        assert_eq!(demanded(3), 4096);
        assert_eq!(composed(3), 4096);

        assert_eq!(demanded(4), 5120);
        assert_eq!(composed(4), 8192);
    }
}
