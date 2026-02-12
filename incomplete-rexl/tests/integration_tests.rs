//! Integration tests for the incomplete-rexl standalone library.
//!
//! These tests exercise the public API surface: NTT transforms,
//! element-wise modular arithmetic, incomplete-NTT helpers, and
//! the fused incomplete-NTT multiplication kernel.

use incomplete_rexl::*;

// A small NTT-friendly prime that fits comfortably in the AVX-512 float path (< 2^50).
// modulus - 1 = 2^20 * 3 * ... so NTT supports degrees up to 2^20.
const MOD_SMALL: u64 = 1125899906826241;
// Half-degrees to test (must be powers of two, >= 8, and 2*n must divide modulus-1).
const TEST_HALF_DEGREES: &[usize] = &[8, 16, 32, 64, 128, 256, 512];

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

fn random_vec(n: usize, modulus: u64) -> Vec<u64> {
    use rand::Rng;
    let mut rng = rand::rng();
    (0..n).map(|_| rng.random::<u64>() % modulus).collect()
}

/// Schoolbook polynomial multiplication mod (X^degree + 1), mod `modulus`.
/// Both `a` and `b` are in coefficient form, length = `degree`.
fn poly_mul_schoolbook(a: &[u64], b: &[u64], degree: usize, modulus: u64) -> Vec<u64> {
    let mut result = vec![0u64; degree];
    for i in 0..degree {
        for j in 0..degree {
            let prod = multiply_mod(a[i], b[j], modulus);
            let idx = i + j;
            if idx < degree {
                result[idx] = add_mod(result[idx], prod, modulus);
            } else {
                // X^degree ≡ -1 (mod X^degree + 1)
                let idx = idx - degree;
                result[idx] = sub_mod(result[idx], prod, modulus);
            }
        }
    }
    result
}

// ──────────────────────────────────────────────────────────────────────────────
// Basic modular arithmetic
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_add_mod() {
    assert_eq!(add_mod(3, 4, 5), 2);
    assert_eq!(add_mod(0, 0, 7), 0);
    assert_eq!(add_mod(6, 0, 7), 6);
    assert_eq!(add_mod(6, 1, 7), 0);
}

#[test]
fn test_sub_mod() {
    assert_eq!(sub_mod(4, 3, 5), 1);
    assert_eq!(sub_mod(0, 0, 7), 0);
    assert_eq!(sub_mod(0, 1, 7), 6);
}

#[test]
fn test_multiply_mod() {
    assert_eq!(multiply_mod(3, 4, 5), 2); // 12 mod 5
    assert_eq!(multiply_mod(0, 100, MOD_SMALL), 0);
    assert_eq!(multiply_mod(1, 42, MOD_SMALL), 42);
}

#[test]
fn test_power_mod() {
    assert_eq!(power_mod(2, 10, 1000000007), 1024);
    assert_eq!(power_mod(3, 0, 7), 1);
    assert_eq!(power_mod(5, 1, 7), 5);
}

#[test]
fn test_inv_mod() {
    let modulus = 1000000007u64;
    for a in [1, 2, 3, 17, 9999, 123456789] {
        let a_inv = inv_mod(a, modulus);
        assert_eq!(
            multiply_mod(a, a_inv, modulus),
            1,
            "inv_mod failed for a={a}"
        );
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Element-wise operations
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_eltwise_add_mod() {
    let modulus = 17u64;
    let a = vec![1, 5, 16, 0, 10, 12, 8, 3];
    let b = vec![3, 12, 1, 0, 7, 5, 9, 14];
    let mut result = vec![0u64; 8];
    eltwise_add_mod(&mut result, &a, &b, modulus);
    for i in 0..8 {
        assert_eq!(result[i], (a[i] + b[i]) % modulus);
    }
}

#[test]
fn test_eltwise_sub_mod() {
    let modulus = 17u64;
    let a = vec![1, 5, 16, 0, 10, 12, 8, 3];
    let b = vec![3, 12, 1, 0, 7, 5, 9, 14];
    let mut result = vec![0u64; 8];
    eltwise_sub_mod(&mut result, &a, &b, modulus);
    for i in 0..8 {
        let expected = if a[i] >= b[i] {
            a[i] - b[i]
        } else {
            a[i] + modulus - b[i]
        };
        assert_eq!(result[i], expected);
    }
}

#[test]
fn test_eltwise_mult_mod() {
    let modulus = MOD_SMALL;
    let n = 64;
    let a = random_vec(n, modulus);
    let b = random_vec(n, modulus);
    let mut result = vec![0u64; n];
    eltwise_mult_mod(&mut result, &a, &b, modulus);
    for i in 0..n {
        let expected = multiply_mod(a[i], b[i], modulus);
        assert_eq!(result[i], expected, "mismatch at index {i}");
    }
}

#[test]
fn test_eltwise_reduce_mod() {
    let modulus = 17u64;
    let operand: Vec<u64> = vec![0, 1, 16, 17, 18, 33, 34, 100];
    let mut result = vec![0u64; operand.len()];
    eltwise_reduce_mod(&mut result, &operand, modulus);
    for i in 0..operand.len() {
        assert_eq!(result[i], operand[i] % modulus, "mismatch at index {i}");
    }
}

#[test]
fn test_eltwise_fma_mod() {
    let modulus = MOD_SMALL;
    let n = 64;
    let a = random_vec(n, modulus);
    let scalar = 42u64;
    let c = random_vec(n, modulus);
    let mut result = vec![0u64; n];
    eltwise_fma_mod(&mut result, &a, scalar, &c, modulus);
    for i in 0..n {
        let expected = add_mod(multiply_mod(a[i], scalar, modulus), c[i], modulus);
        assert_eq!(result[i], expected, "mismatch at index {i}");
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// NTT forward / inverse round-trip
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ntt_roundtrip() {
    let modulus = MOD_SMALL;
    for &n in TEST_HALF_DEGREES {
        let original = random_vec(n, modulus);
        let mut data = original.clone();
        ntt_forward_in_place(&mut data, n, modulus);
        // NTT output should differ from input (except for trivial inputs)
        assert_ne!(data, original, "NTT didn't change data for n={n}");
        ntt_inverse_in_place(&mut data, n, modulus);
        assert_eq!(data, original, "NTT round-trip failed for n={n}");
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// incomplete_ntt_forward / inverse round-trip
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_incomplete_ntt_roundtrip() {
    let modulus = MOD_SMALL;
    for &n in TEST_HALF_DEGREES {
        let degree = 2 * n;
        let original = random_vec(degree, modulus);
        let mut data = original.clone();

        incomplete_ntt_forward_in_place(&mut data, n, modulus);
        assert_ne!(
            data, original,
            "incomplete NTT didn't change data for n={n}"
        );

        incomplete_ntt_inverse_in_place(&mut data, n, modulus);
        assert_eq!(data, original, "incomplete NTT round-trip failed for n={n}");
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// fused_incomplete_ntt_mult correctness
// ──────────────────────────────────────────────────────────────────────────────

/// Verify the fused kernel against separate eltwise calls (the reference algorithm).
#[test]
fn test_fused_incomplete_ntt_mult_vs_separate() {
    let modulus = MOD_SMALL;

    for &n in TEST_HALF_DEGREES {
        for _ in 0..5 {
            let op1 = random_vec(2 * n, modulus);
            let op2 = random_vec(2 * n, modulus);

            // Compute shift factors the classic way: NTT([0,1,0,…])
            let mut shift = vec![0u64; n];
            shift[1] = 1;
            ntt_forward_in_place(&mut shift, n, modulus);

            // --- Reference: separate eltwise calls ---
            let mut ref_result = vec![0u64; 2 * n];
            let mut tmp = vec![0u64; n];

            // ref_even = op1_even * op2_even
            eltwise_mult_mod(&mut ref_result[..n], &op1[..n], &op2[..n], modulus);
            // ref_odd = op1_odd * op2_even
            eltwise_mult_mod(&mut ref_result[n..], &op1[n..], &op2[..n], modulus);
            // tmp = op1_odd * op2_odd
            eltwise_mult_mod(&mut tmp, &op1[n..], &op2[n..], modulus);
            // tmp *= shift_factors
            let mut stmp = vec![0u64; n];
            eltwise_mult_mod(&mut stmp, &tmp, &shift, modulus);
            // ref_even += stmp
            let ref_even_copy: Vec<u64> = ref_result[..n].to_vec();
            eltwise_add_mod(&mut ref_result[..n], &ref_even_copy, &stmp, modulus);
            // tmp = op1_even * op2_odd
            eltwise_mult_mod(&mut tmp, &op1[..n], &op2[n..], modulus);
            // ref_odd += tmp
            let ref_odd_copy: Vec<u64> = ref_result[n..].to_vec();
            eltwise_add_mod(&mut ref_result[n..], &ref_odd_copy, &tmp, modulus);

            // --- Fused path ---
            let mut fused_result = vec![0u64; 2 * n];
            fused_incomplete_ntt_mult(&mut fused_result, &op1, &op2, n, modulus);

            assert_eq!(
                ref_result, fused_result,
                "Fused mult diverged from reference at n={n}"
            );
        }
    }
}

/// End-to-end: multiply two polynomials in coefficient form via incomplete NTT,
/// compare against schoolbook multiplication in Z_q[X]/(X^degree + 1).
#[test]
fn test_incomplete_ntt_mult_end_to_end() {
    let modulus = MOD_SMALL;

    // Use smaller degrees for schoolbook (O(n²) cost).
    for &n in &[8, 16, 32] {
        let degree = 2 * n;
        for _ in 0..3 {
            let a_coeffs = random_vec(degree, modulus);
            let b_coeffs = random_vec(degree, modulus);

            // Schoolbook reference
            let expected = poly_mul_schoolbook(&a_coeffs, &b_coeffs, degree, modulus);

            // Incomplete NTT path
            let mut a_ntt = a_coeffs.clone();
            let mut b_ntt = b_coeffs.clone();
            incomplete_ntt_forward_in_place(&mut a_ntt, n, modulus);
            incomplete_ntt_forward_in_place(&mut b_ntt, n, modulus);

            let mut c_ntt = vec![0u64; degree];
            fused_incomplete_ntt_mult(&mut c_ntt, &a_ntt, &b_ntt, n, modulus);

            incomplete_ntt_inverse_in_place(&mut c_ntt, n, modulus);

            assert_eq!(
                c_ntt, expected,
                "End-to-end polynomial multiplication failed at degree={degree}"
            );
        }
    }
}

/// Verify that result can alias operand1 (in-place multiplication).
#[test]
fn test_fused_incomplete_ntt_mult_in_place() {
    let modulus = MOD_SMALL;

    for &n in TEST_HALF_DEGREES {
        let op1 = random_vec(2 * n, modulus);
        let op2 = random_vec(2 * n, modulus);

        // Out-of-place reference
        let mut ref_result = vec![0u64; 2 * n];
        fused_incomplete_ntt_mult(&mut ref_result, &op1, &op2, n, modulus);

        // In-place: result aliases operand1
        let mut in_place = op1.clone();
        let in_place_copy = in_place.clone();
        fused_incomplete_ntt_mult(&mut in_place, &in_place_copy, &op2, n, modulus);

        assert_eq!(ref_result, in_place, "In-place aliasing broke at n={n}");
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Shift factors cached correctly
// ──────────────────────────────────────────────────────────────────────────────

/// Verify that the internally cached shift factors match the manual computation.
#[test]
fn test_shift_factors_cached_correctly() {
    let modulus = MOD_SMALL;
    for &n in TEST_HALF_DEGREES {
        // Manual: NTT([0, 1, 0, 0, …])
        let mut manual = vec![0u64; n];
        manual[1] = 1;
        ntt_forward_in_place(&mut manual, n, modulus);

        // Now verify via the fused mult: multiply (1,0,0,…) × (0,0,…,0 | 0,0,…,1_at_pos_i,…)
        // Actually, simplest check: use the identity element.
        // For the even-odd representation, the identity is [1,1,1,…,1 | 0,0,…,0].
        // Multiplying identity × [0,…,0 | 0,…,0,1_at_0,0,…] gives:
        //   result_even[i] = shift[i] * 1 = shift[i]
        //   result_odd[i]  = 1 * 1 = ... (we only care about result_even here)
        //
        // Instead, let's just pick two known vectors and check the formula directly.

        // Use op1 = [1,0,0,… | 0,0,…,0]  (e_0 in even half)
        //     op2 = [0,0,0,… | 1,0,…,0]  (e_0 in odd half)
        // result_even[i] = op1_even[i]*op2_even[i] + shift[i]*op1_odd[i]*op2_odd[i] = 0
        // result_odd[i]  = op1_odd[i]*op2_even[i] + op1_even[i]*op2_odd[i]
        //   for i=0: = 0 + 1*1 = 1; for i>0: = 0
        // This doesn't test shift factors. Let's do:
        // op1 = [0,…,0 | 1,0,…,0], op2 = [0,…,0 | 1,0,…,0]
        // result_even[i] = 0 + shift[i]*(1 if i==0 else 0)*(1 if i==0 else 0) = shift[0] if i==0
        // That only gives shift[0].

        // Best: op1 = [0,…,0 | 1,1,…,1], op2 = [0,…,0 | 1,0,…,0]
        // result_even[i] = shift[i]*1*... hmm this gets complicated.
        // Let's just verify by constructing the shift factors from NTT and comparing
        // with what fused_incomplete_ntt_mult produces vs the reference formula.
        // We already test this in test_fused_incomplete_ntt_mult_vs_separate.

        // Direct verification: for each i, set op1 = op2 = unit vector at odd[i].
        // op1 = [0..0 | e_i], op2 = [0..0 | e_i]
        // result_even[j] = shift[j] * (i==j ? 1 : 0) * (i==j ? 1 : 0) = shift[i] if j==i
        for test_i in 0..std::cmp::min(n, 8) {
            let mut op1 = vec![0u64; 2 * n];
            op1[n + test_i] = 1;
            let op2 = op1.clone();
            let mut result = vec![0u64; 2 * n];
            fused_incomplete_ntt_mult(&mut result, &op1, &op2, n, modulus);
            assert_eq!(
                result[test_i], manual[test_i],
                "Cached shift factor mismatch at n={n}, i={test_i}: got {}, expected {}",
                result[test_i], manual[test_i]
            );
        }
    }
}
