use crate::protocol::commitment::commit_basic;
use crate::protocol::config::{to_kb, SizeableProof, CONFIG};
use crate::protocol::crs::CRS;
use crate::protocol::parties::prover::{prover_round, vdf_crs};
use crate::protocol::parties::verifier::verifier_round;
use crate::protocol::sumchecks::builder::init_prover_sumcheck;
use crate::protocol::sumchecks::builder_verifier::init_verifier_sumcheck;
use crate::{
    common::{
        config::*,
        hash::HashWrapper,
        matrix::{new_vec_zero_preallocated, HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        ring_arithmetic::{Representation, RingElement},
    },
    protocol::config::RoundConfig,
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

// ==========================
// Verifier Sumcheck Context
// ==========================

/// Computes ip_vdf_claim = Σ_r c^r·(-y_0[r]) + c^{VDF_STRIDE·2K+r}·y_t[r] from the VDF challenge and outputs.
pub fn compute_ip_vdf_claim(
    config: &RoundConfig,
    vdf_challenge: Option<&RingElement>,
    vdf_params: Option<(
        &[RingElement; VDF_MATRIX_HEIGHT],
        &[RingElement; VDF_MATRIX_HEIGHT],
        &vdf_crs,
    )>,
) -> Option<RingElement> {
    if !config.vdf {
        return None;
    }
    let c = vdf_challenge.expect("VDF enabled but no challenge");
    let (y_0, y_t, _) = vdf_params.expect("VDF enabled but no params");
    let two_k = config.extended_witness_length / 2 / VDF_MATRIX_WIDTH;

    // Compute c^{VDF_STRIDE * 2K}
    let mut c_stride = RingElement::constant(1, Representation::IncompleteNTT);
    for _ in 0..VDF_STRIDE {
        c_stride *= c;
    }
    let mut c_stride_2k = RingElement::constant(1, Representation::IncompleteNTT);
    for _ in 0..two_k {
        c_stride_2k *= &c_stride;
    }

    // claim = Σ_r c^r · (-y_0[r]) + Σ_r c^{VDF_STRIDE·2K + r} · y_t[r]
    let mut claim = RingElement::zero(Representation::IncompleteNTT);
    let mut c_power = RingElement::constant(1, Representation::IncompleteNTT); // c^r
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    for r in 0..VDF_MATRIX_HEIGHT {
        // -c^r · y_0[r]
        temp *= (&c_power, &y_0[r]);
        claim -= &temp;
        // c^{VDF_STRIDE·2K + r} · y_t[r]
        temp *= (&c_stride_2k, &c_power); // temp = c^{VDF_STRIDE*2K + r}
        temp *= &y_t[r];
        claim += &temp;
        c_power *= c;
    }
    Some(claim)
}

pub fn vdf_init() -> vdf_crs {
    println!("Initializing VDF CRS...");
    let A = HorizontallyAlignedMatrix {
        height: VDF_MATRIX_HEIGHT,
        width: VDF_MATRIX_WIDTH,
        data: (0..VDF_MATRIX_HEIGHT * VDF_MATRIX_WIDTH)
            .map(|_| RingElement::random(Representation::IncompleteNTT))
            .collect(),
    };
    vdf_crs { A }
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

pub fn execute_vdf(
    y_0: &[RingElement; VDF_MATRIX_HEIGHT],
    dim: usize,
    vdf_crs: &vdf_crs,
) -> VDFOutput {
    let vdf_crs_ref = vdf_crs;

    // VDF with G = I_{HEIGHT} ⊗ g^T (gadget) and A (HEIGHT × WIDTH CRS matrix).
    //
    // Per step:
    //   w_step = G^{-1}(-y_step)   — decompose each component of y_step into VDF_BITS binary planes
    //   y_{step+1} = A · w_step    — full matrix-vector product giving HEIGHT outputs
    //
    // The witness is split into two columns (matching vertical memory alignment).
    // y_int is the intermediate value at the column boundary.

    let mut trace_witness = VerticallyAlignedMatrix {
        height: dim,
        width: 2,
        data: new_vec_zero_preallocated(dim * 2),
        used_cols: 2,
    };

    let steps_per_col = dim / VDF_MATRIX_WIDTH;
    let total_steps = steps_per_col * 2;

    let mut neg_y: [RingElement; VDF_MATRIX_HEIGHT] = std::array::from_fn(|r| y_0[r].negate());
    let mut y_int: [RingElement; VDF_MATRIX_HEIGHT] =
        std::array::from_fn(|_| RingElement::zero(Representation::IncompleteNTT));
    let mut temp = RingElement::zero(Representation::IncompleteNTT);

    println!("Executing delay function with {} steps", total_steps);
    // y_{step+1} = A · w_step: full matrix-vector product
    let mut y_next: [RingElement; VDF_MATRIX_HEIGHT] =
        std::array::from_fn(|_| RingElement::zero(Representation::IncompleteNTT));
    let vdf_start = std::time::Instant::now();
    for step in 0..total_steps {
        let col = step / steps_per_col;
        let row_in_col = step % steps_per_col;
        let base_row = row_in_col * VDF_MATRIX_WIDTH;

        // w_step = G^{-1}(-y_step): decompose each component into VDF_BITS binary planes
        let data_offset = col * dim + base_row;
        for r in 0..VDF_MATRIX_HEIGHT {
            decompose_binary_into(
                &neg_y[r],
                &mut trace_witness.data
                    [data_offset + r * VDF_BITS..data_offset + (r + 1) * VDF_BITS],
            );
        }

        for r in 0..VDF_MATRIX_HEIGHT {
            for j in 0..VDF_MATRIX_WIDTH {
                temp *= (&vdf_crs_ref.A[(r, j)], &trace_witness.data[data_offset + j]);
                if j == 0 {
                    y_next[r].set_from(&temp);
                } else {
                    y_next[r] += &temp;
                }
            }
        }

        if step == steps_per_col - 1 {
            y_int = y_next.clone();
        }

        neg_y = std::array::from_fn(|r| y_next[r].negate());
    }
    let vdf_duration = vdf_start.elapsed().as_micros();
    println!("Delay function executed in {:?} µs", vdf_duration);
    println!(
        "Avg step time: {:?} µs",
        vdf_duration as f64 / (total_steps as f64)
    );

    let y_t: [RingElement; VDF_MATRIX_HEIGHT] = std::array::from_fn(|r| neg_y[r].negate());

    VDFOutput {
        y_int,
        y_t,
        trace_witness,
    }
}

pub fn execute() {
    println!("Generating CRS...");

    let crs = CRS::gen_crs(WITNESS_DIM, 8);
    let vdf_crs = vdf_init();

    println!("CRS generated. Starting execution...");
    let y_0: [RingElement; VDF_MATRIX_HEIGHT] =
        std::array::from_fn(|_| RingElement::random(Representation::IncompleteNTT)); // TODO: from hash
    let vdf_output = execute_vdf(&y_0, WITNESS_DIM, &vdf_crs);

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
        &vec![],    // no evaluation points for first round
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
