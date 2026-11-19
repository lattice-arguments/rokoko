use std::num::NonZeroUsize;

use rokoko::{
    commitment::crs::{VerifierCRS, CRS},
    committer::commit,
    common::matrix::VerticallyAlignedMatrix,
    prover::prover_round,
    ring::ring_arithmetic::RingElement,
    rok::{
        config::{
            params::{decompose_witness, witness_sampler, Instantiation},
            to_kb, SizeableProof,
        },
        constraints::sumcheck_round::{builder::init_sumcheck, builder_verifier::init_verifier},
        evaluation_point_sampler::{sample_initial_evaluation_points, InitialEvaluationPoints},
        open::claim,
        wire,
    },
    tools::boundary::{BoundaryCapture, ProverBoundary, VerifierBoundary},
    verifier::verifier_round,
};

pub struct BoundaryRun {
    pub prover: ProverBoundary,
    pub verifier: VerifierBoundary,
    pub crs: CRS,
    pub verifier_crs: VerifierCRS,
    pub proof_size_bits: usize,
}

fn run(
    inst: &Instantiation,
    cut: Option<NonZeroUsize>,
) -> (
    usize,
    Option<ProverBoundary>,
    Option<VerifierBoundary>,
    CRS,
    VerifierCRS,
) {
    let config = inst.root();
    let witness_config = &inst.witness;

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
    let witness = witness_sampler(witness_config);

    let start = std::time::Instant::now();

    let commit_span = tracing::info_span!("commit").entered();
    let witness_decomposed = decompose_witness(&witness, witness_config);
    let (commitment_with_aux, rc_commitment) = commit(&crs, &config, &witness_decomposed);
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
    let proof = wire::from_bytes_for(&config, &bytes).expect("the proof must deserialise");
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

/// A run from a Ξ^lin start: run the chain on evaluation claims about a sampled witness.
pub fn execute(inst: &Instantiation) {
    run(inst, None);
}

pub fn execute_to_boundary(inst: &Instantiation, cut: NonZeroUsize) -> BoundaryRun {
    let (proof_size_bits, prover_boundary, verifier_boundary, crs, verifier_crs) =
        run(inst, Some(cut));
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

/// A run from a Ξ^sum start: prove three fixed example sumcheck claims about a committed
/// witness, then run the chain on the resulting evaluation claims (a Ξ^lin statement).
pub fn execute_snark(inst: &Instantiation) {
    use rokoko::{
        common::{hash::HashWrapper, sampling::sample_random_short_vector},
        ring::ring_arithmetic::{Representation, RingElement},
    };
    use rokoko::rok::claims::{eq, prove_claims, verify_claims, witness_in, Claim, Region};

    let config = inst.root();

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
            rokoko::ring::ring_arithmetic::Representation::IncompleteNTT,
        ),
    };

    let start = std::time::Instant::now();
    let _commit_span = tracing::info_span!("commit").entered();
    let (commitment_with_aux, rc_commitment) = commit(&crs, &config, &witness);
    drop(_commit_span);
    println!("TOTAL Commit time: {:?} ns", start.elapsed().as_nanos());

    let total_vars = (config.witness_height * config.witness_width).ilog2() as usize;
    let n = config.witness_height * config.witness_width;
    let everything = Region::whole(n);

    let structured_point: Vec<RingElement> = (0..total_vars)
        .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 1 << 10))
        .collect();
    let t1 = {
        use rokoko::common::structured_row::PreprocessedRow;
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

    // the exact-norm chain has two openings, so the statement must use
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
    let proof = wire::from_bytes_for(&config, &bytes).expect("the proof must deserialise");
    let initial_proof = wire::initial_from_bytes_for(
        (config.witness_height, config.witness_width),
        &initial_bytes,
    )
    .expect("the initial proof must deserialise");
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
