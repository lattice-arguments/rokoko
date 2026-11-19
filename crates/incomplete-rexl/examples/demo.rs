//! Comprehensive example demonstrating every public feature of incomplete-rexl.
//!
//! Run with:
//!     cargo run --release --example demo

use incomplete_rexl::{
    // Scalar modular arithmetic
    add_mod,
    // CPU feature detection
    cpu_features,
    // Element-wise vector operations
    eltwise_add_mod,
    eltwise_fma_mod,
    eltwise_mult_mod,
    eltwise_reduce_mod,
    eltwise_sub_mod,
    // Fused ring multiplication
    fused_incomplete_ntt_mult,
    // Incomplete-NTT helpers
    incomplete_ntt_forward_in_place,
    incomplete_ntt_inverse_in_place,
    inv_mod,
    multiply_mod,
    // NTT transforms
    ntt_forward_in_place,
    ntt_inverse_in_place,
    power_mod,
    sub_mod,
};

/// 50-bit NTT-friendly prime (supports NTT up to degree 16384).
const P: u64 = 1125899906826241;

fn main() {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║       incomplete-rexl  feature demo             ║");
    println!("╚══════════════════════════════════════════════════╝\n");

    // ─── CPU features ────────────────────────────────────────────────────
    println!("── CPU features ──");
    cpu_features::print_features();
    println!();

    // ─── 1. Scalar modular arithmetic ────────────────────────────────────
    println!("── 1. Scalar modular arithmetic (mod p = {P}) ──");

    let a = 123_456_789;
    let b = 987_654_321;

    let sum = add_mod(a, b, P);
    let diff = sub_mod(a, b, P);
    let prod = multiply_mod(a, b, P);
    let inv = inv_mod(a, P);
    let pow = power_mod(a, 1000, P);

    println!("  add_mod({a}, {b})      = {sum}");
    println!("  sub_mod({a}, {b})      = {diff}");
    println!("  multiply_mod({a}, {b}) = {prod}");
    println!("  inv_mod({a})                    = {inv}");
    println!("  power_mod({a}, 1000)            = {pow}");

    // Verify: a * inv(a) ≡ 1
    assert_eq!(multiply_mod(a, inv, P), 1);
    println!("  ✓ a * inv(a) ≡ 1");

    // Verify: Fermat's little theorem  a^(p-1) ≡ 1
    assert_eq!(power_mod(a, P - 1, P), 1);
    println!("  ✓ a^(p-1) ≡ 1  (Fermat)");
    println!();

    // ─── 2. Element-wise vector operations ───────────────────────────────
    println!("── 2. Element-wise vector operations ──");

    let n = 8;
    let x: Vec<u64> = (1..=n as u64).collect(); // [1,2,…,8]
    let y: Vec<u64> = (1..=n as u64).rev().collect(); // [8,7,…,1]
    let mut result = vec![0u64; n];

    eltwise_add_mod(&mut result, &x, &y, P);
    println!("  add  {x:?} + {y:?} = {result:?}");

    eltwise_sub_mod(&mut result, &x, &y, P);
    println!("  sub  {x:?} - {y:?} = {result:?}");

    eltwise_mult_mod(&mut result, &x, &y, P);
    println!("  mult {x:?} * {y:?} = {result:?}");

    // FMA: result = x * 3 + y  (arg2 is a scalar)
    eltwise_fma_mod(&mut result, &x, 3, &y, P);
    println!("  fma  x*3 + y = {result:?}");

    // Reduce: bring values > P back into [0, P)
    let big: Vec<u64> = x.iter().map(|&v| v + P).collect();
    eltwise_reduce_mod(&mut result, &big, P);
    println!("  reduce (x + P) mod P = {result:?}");
    assert_eq!(result, x);
    println!("  ✓ reduction matches original");
    println!();

    // ─── 3. NTT forward / inverse round-trip ────────────────────────────
    println!("── 3. NTT round-trip (n = 1024) ──");

    let ntt_n = 1024;
    let mut data = vec![0u64; ntt_n];
    data[0] = 42;
    data[1] = 99;
    data[ntt_n - 1] = 7;
    let original = data.clone();

    ntt_forward_in_place(&mut data, ntt_n, P);
    println!(
        "  After forward NTT: [{}, {}, {}, …, {}]",
        data[0],
        data[1],
        data[2],
        data[ntt_n - 1]
    );

    ntt_inverse_in_place(&mut data, ntt_n, P);
    assert_eq!(data, original);
    println!(
        "  After inverse NTT: [{}, {}, {}, …, {}]",
        data[0],
        data[1],
        data[2],
        data[ntt_n - 1]
    );
    println!("  ✓ NTT round-trip exact");
    println!();

    // ─── 4. Point-wise multiplication in full-NTT domain ────────────────
    println!("── 4. Point-wise multiplication in full-NTT domain ──");

    // Multiply (1 + x) * (1 + x) = 1 + 2x + x^2  using full NTT
    let mut f = vec![0u64; ntt_n];
    f[0] = 1;
    f[1] = 1;
    let mut g = f.clone();

    ntt_forward_in_place(&mut f, ntt_n, P);
    ntt_forward_in_place(&mut g, ntt_n, P);

    let mut h = vec![0u64; ntt_n];
    eltwise_mult_mod(&mut h, &f, &g, P);

    ntt_inverse_in_place(&mut h, ntt_n, P);
    println!(
        "  (1+x)² = {} + {}x + {}x²  (rest zeros: {})",
        h[0],
        h[1],
        h[2],
        h[3..].iter().all(|&v| v == 0)
    );
    assert_eq!((h[0], h[1], h[2]), (1, 2, 1));
    println!("  ✓ correct");
    println!();

    // ─── 5. Incomplete-NTT forward / inverse round-trip ─────────────────
    println!("── 5. Incomplete-NTT round-trip ──");

    let half_n: usize = 512;
    let degree = 2 * half_n;
    let mut poly = vec![0u64; degree];
    poly[0] = 5;
    poly[1] = 3;
    poly[degree - 1] = 1;
    let poly_orig = poly.clone();

    incomplete_ntt_forward_in_place(&mut poly, half_n, P);
    println!(
        "  Forward: first 4 = [{}, {}, {}, {}]",
        poly[0], poly[1], poly[2], poly[3]
    );

    incomplete_ntt_inverse_in_place(&mut poly, half_n, P);
    assert_eq!(poly, poly_orig);
    println!("  ✓ Incomplete-NTT round-trip exact");
    println!();

    // ─── 6. Fused incomplete-NTT ring multiplication ─────────────────────
    println!(
        "── 6. Ring multiplication in Z_p[x]/(x^{{{}}} + 1) ──",
        degree
    );

    //   a(x) = 5 + 3x
    //   b(x) = 2 + 7x + x^{1023}
    let mut a_poly = vec![0u64; degree];
    let mut b_poly = vec![0u64; degree];
    a_poly[0] = 5;
    a_poly[1] = 3;
    b_poly[0] = 2;
    b_poly[1] = 7;
    b_poly[degree - 1] = 1;

    // Convert to incomplete-NTT form
    incomplete_ntt_forward_in_place(&mut a_poly, half_n, P);
    incomplete_ntt_forward_in_place(&mut b_poly, half_n, P);

    // Multiply (shift factors cached automatically)
    let mut c_poly = vec![0u64; degree];
    fused_incomplete_ntt_mult(&mut c_poly, &a_poly, &b_poly, half_n, P);

    // Convert back to coefficients
    incomplete_ntt_inverse_in_place(&mut c_poly, half_n, P);

    //   Hand computation:
    //     (5 + 3x)(2 + 7x + x^1023)
    //     = 10 + 35x + 5x^1023  +  6x + 21x^2 + 3x^1024
    //     x^1024 ≡ −1  ⟹  3x^1024 = −3
    //     = (10−3) + (35+6)x + 21x^2 + 5x^1023
    //     = 7 + 41x + 21x^2 + 5x^1023
    assert_eq!(c_poly[0], 7);
    assert_eq!(c_poly[1], 41);
    assert_eq!(c_poly[2], 21);
    assert_eq!(c_poly[degree - 1], 5);
    assert!(c_poly[3..degree - 1].iter().all(|&v| v == 0));

    println!("  a(x) = 5 + 3x");
    println!("  b(x) = 2 + 7x + x^1023");
    println!(
        "  a·b  = {} + {}x + {}x² + {}x^1023",
        c_poly[0],
        c_poly[1],
        c_poly[2],
        c_poly[degree - 1]
    );
    println!("  ✓ matches hand computation");
    println!();

    // ─── 7. In-place multiplication (result aliases operand) ────────────
    println!("── 7. In-place multiplication (accumulate into operand) ──");

    //  Square a(x) = 1 + x  in-place
    let mut acc = vec![0u64; degree];
    acc[0] = 1;
    acc[1] = 1;
    incomplete_ntt_forward_in_place(&mut acc, half_n, P);
    let acc_copy = acc.clone();
    fused_incomplete_ntt_mult(&mut acc, &acc_copy, &acc_copy, half_n, P);
    incomplete_ntt_inverse_in_place(&mut acc, half_n, P);

    // (1+x)^2 mod (x^1024+1) = 1 + 2x + x^2
    assert_eq!((acc[0], acc[1], acc[2]), (1, 2, 1));
    assert!(acc[3..].iter().all(|&v| v == 0));
    println!("  (1+x)² = {} + {}x + {}x²", acc[0], acc[1], acc[2]);
    println!("  ✓ in-place aliasing works");
    println!();

    println!("══════════════════════════════════════════════════");
    println!("  All checks passed!");
    println!("══════════════════════════════════════════════════");
}
