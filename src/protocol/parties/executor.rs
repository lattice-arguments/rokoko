use crate::{
    common::{
        config::*,
        hash::HashWrapper,
        matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        ring_arithmetic::{Representation, RingElement},
    },
    protocol::{
        commitment::commit_basic,
        config::{to_kb, SizeableProof, CONFIG},
        crs::CRS,
        parties::{prover::prover_round, verifier::verifier_round},
        sumchecks::{builder::init_prover_sumcheck, builder_verifier::init_verifier_sumcheck},
        vdf::{run_vdf, vdf_init},
    },
};

pub struct VDFOutput {
    y_int: [RingElement; VDF_MATRIX_HEIGHT], // TODO: this y_int is not needed but let's keep it for now
    y_t: [RingElement; VDF_MATRIX_HEIGHT],
    trace_witness: VerticallyAlignedMatrix<RingElement>,
}
fn sample_random_binary_vector(len: usize) -> Vec<RingElement> {
    (0..len)
        .map(|_| RingElement::random_bounded_unsigned(Representation::IncompleteNTT, 2))
        .collect()
}

pub fn binary_witness_sampler() -> VerticallyAlignedMatrix<RingElement> {
    VerticallyAlignedMatrix {
        height: WITNESS_DIM,
        width: WITNESS_WIDTH,
        data: sample_random_binary_vector(WITNESS_DIM * WITNESS_WIDTH),
        // data: vec![RingElement::all(0, Representation::IncompleteNTT); WITNESS_DIM * WITNESS_WIDTH],
        used_cols: WITNESS_WIDTH,
    }
}

/// Decomposes a RingElement into 64 bit-plane RingElements, writing into `target`.
/// target\[b\].v\[j\] = (element.v\[j\] >> b) & 1 for each coefficient j and bit b.
/// The input is assumed to be in IncompleteNTT; we convert to EvenOddCoefficients
/// to access raw coefficients, decompose, then convert each result back.
pub fn decompose_binary_into(element: &RingElement, target: &mut [RingElement]) {
    assert!(
        target.len() >= 64,
        "target slice must have at least 64 elements"
    );

    let mut tmp = element.clone();
    tmp.from_incomplete_ntt_to_even_odd_coefficients();

    for bit_elem in target[..64].iter_mut() {
        *bit_elem = RingElement::zero(Representation::EvenOddCoefficients);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        use std::arch::x86_64::*;
        unsafe {
            let one = _mm512_set1_epi64(1);
            // Process 8 coefficients at a time
            for chunk_start in (0..DEGREE).step_by(8) {
                let coeffs = _mm512_loadu_epi64(tmp.v[chunk_start..].as_ptr() as *const i64);
                for b in 0..64u64 {
                    let shift_amt = _mm512_set1_epi64(b as i64);
                    let shifted = _mm512_srlv_epi64(coeffs, shift_amt);
                    let masked = _mm512_and_epi64(shifted, one);
                    _mm512_storeu_epi64(
                        target[b as usize].v[chunk_start..].as_mut_ptr() as *mut i64,
                        masked,
                    );
                }
            }
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        for j in 0..DEGREE {
            let val = tmp.v[j];
            for b in 0..64usize {
                target[b].v[j] = (val >> b) & 1;
            }
        }
    }

    for bit_elem in target[..64].iter_mut() {
        bit_elem.from_even_odd_coefficients_to_incomplete_ntt_representation();
    }
}

pub fn execute() {
    println!("Generating CRS...");

    let crs = CRS::gen_crs(WITNESS_DIM, 8);
    let vdf_crs = vdf_init();

    println!("CRS generated. Starting execution...");
    let y_0: [RingElement; VDF_MATRIX_HEIGHT] =
        std::array::from_fn(|_| RingElement::random(Representation::IncompleteNTT)); // TODO: from hash
    let vdf_output = run_vdf(&y_0, WITNESS_DIM, &vdf_crs);

    let mut sumcheck_context = init_prover_sumcheck(&crs, &CONFIG);

    println!("===== COMMITTING WITNESS =====");
    let start = std::time::Instant::now();

    let commitment = commit_basic(&crs, &vdf_output.trace_witness, RANK);

    let commit_duration = start.elapsed().as_nanos();
    println!("TOTAL Commit time: {:?} ns", commit_duration);

    let no_claims = HorizontallyAlignedMatrix {
        height: 0,
        width: 2,
        data: vec![],
    };

    println!("===== STARTING PROVER =====");
    let start = std::time::Instant::now();
    let proof = prover_round(
        &crs,
        &vdf_output.trace_witness,
        &CONFIG,
        &mut sumcheck_context,
        &vec![], // no evaluation points for first round
        &no_claims,
        &mut HashWrapper::new(),
        Some((&y_0, &vdf_output.y_t, &vdf_crs)),
    );
    let prove_duration = start.elapsed().as_millis();
    println!("TOTAL Prove time: {:?} ms", prove_duration);

    println!("===== PROOF SIZE =====");
    let proof_size_bits = proof.size_in_bits();
    println!("Total proof size: {:.2} KB", to_kb(proof_size_bits));

    println!("===== STARTING VERIFIER =====");
    let start = std::time::Instant::now();
    let mut verifier_context = init_verifier_sumcheck(&CONFIG);
    verifier_round(
        &CONFIG,
        &crs,
        &mut verifier_context,
        &commitment,
        &proof,
        &[],        // no evaluation points for first round
        &no_claims, // no claims for first round
        &mut HashWrapper::new(),
        Some(&vdf_crs),
        Some((&y_0, &vdf_output.y_t)),
        0,
    );
    let verify_duration = start.elapsed().as_nanos();
    println!("TOTAL Verify time: {:?} ns", verify_duration);
    println!("===== VERIFICATION PASSED =====");
}
