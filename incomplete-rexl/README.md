# incomplete-rexl

A pure-Rust library for high-performance modular arithmetic and Number Theoretic
Transforms (NTT), with AVX-512 acceleration.  Designed as a standalone
replacement for Intel HEXL, with additional support for *incomplete NTT*
multiplication. 

> **Requires Rust nightly** (`#![feature(target_feature_inline_always)]`).

## Features

- **Scalar modular arithmetic** – `add_mod`, `sub_mod`, `multiply_mod`,
  `power_mod`, `inv_mod`.
- **Element-wise vector operations** – `eltwise_add_mod`, `eltwise_sub_mod`,
  `eltwise_mult_mod`, `eltwise_fma_mod`, `eltwise_reduce_mod`.
- **Forward / inverse NTT** – `ntt_forward_in_place`, `ntt_inverse_in_place`.
- **Incomplete-NTT helpers** – `incomplete_ntt_forward_in_place`,
  `incomplete_ntt_inverse_in_place` for converting between coefficient and
  even-odd NTT representation.
- **Fused incomplete-NTT multiplication** – `fused_incomplete_ntt_mult`,
  a single-call ring multiplication over incomplete-NTT operands with
  internally cached shift factors.
- **AVX-512 fast paths** – transparent runtime dispatch to AVX-512 kernels
  (avx512f, avx512dq, avx512ifma) when supported; falls back to portable
  scalar code otherwise.

## Quick start

Add the dependency (path or git):

```toml
[dependencies]
incomplete-rexl = { path = "../incomplete-rexl" }
```

### Basic modular arithmetic

```rust
use incomplete_rexl::{add_mod, multiply_mod, power_mod, inv_mod};

let p = 1125899906826241u64; // 50-bit NTT-friendly prime

assert_eq!(add_mod(10, 20, p), 30);
assert_eq!(multiply_mod(7, inv_mod(7, p), p), 1);
assert_eq!(power_mod(3, p - 1, p), 1); // Fermat's little theorem
```

### NTT round-trip

```rust
use incomplete_rexl::{ntt_forward_in_place, ntt_inverse_in_place};

let p: u64 = 1125899906826241;
let n: usize = 1024;
let mut data = vec![0u64; n];
data[0] = 1;
data[1] = 2;

ntt_forward_in_place(&mut data, n, p);
ntt_inverse_in_place(&mut data, n, p);

assert_eq!(data[0], 1);
assert_eq!(data[1], 2);
assert!(data[2..].iter().all(|&x| x == 0));
```

### Incomplete-NTT polynomial multiplication

This is the main high-level workflow for ring multiplication in
$\mathbb{Z}_p[x]/(x^N + 1)$ where $N = 2n$.  The *incomplete NTT* splits each
degree-$N$ polynomial into two halves of size $n$ (even- and odd-indexed
coefficients) and applies a size-$n$ forward NTT to each half independently.
Multiplication then uses a Karatsuba-style formula per coefficient, with
internally cached *shift factors* (the twiddle factors from the omitted last
butterfly level).

```rust
use incomplete_rexl::{
    add_mod, sub_mod, multiply_mod,
    incomplete_ntt_forward_in_place,
    incomplete_ntt_inverse_in_place,
    fused_incomplete_ntt_mult,
};

let p: u64 = 1125899906826241;   // 50-bit NTT-friendly prime
let n: usize = 512;              // half-degree
let degree = 2 * n;              // ring degree  (x^{degree} + 1)

// ── Build two small polynomials in coefficient form ──
//   a(x) = 5 + 3x
//   b(x) = 2 + 7x + x^{degree-1}
let mut a = vec![0u64; degree];
let mut b = vec![0u64; degree];
a[0] = 5; a[1] = 3;
b[0] = 2; b[1] = 7; b[degree - 1] = 1;

// ── Step 1: Convert to incomplete-NTT representation ──
// (de-interleave even/odd coefficients, then NTT each half)
incomplete_ntt_forward_in_place(&mut a, n, p);
incomplete_ntt_forward_in_place(&mut b, n, p);

// ── Step 2: Multiply in the incomplete-NTT domain ──
// Shift factors are computed once and cached per (n, p).
let mut c = vec![0u64; degree];
fused_incomplete_ntt_mult(&mut c, &a, &b, n, p);

// ── Step 3: Convert back to coefficient form ──
incomplete_ntt_inverse_in_place(&mut c, n, p);

// ── Verify against hand computation in Z_p[x]/(x^{degree} + 1) ──
//
//   a·b = (5 + 3x)(2 + 7x + x^{1023})
//       =  10 + 35x + 5x^{1023}
//       +  6x + 21x² + 3x^{1024}
//
// Now reduce mod (x^{1024} + 1):  x^{1024} ≡ −1  (mod x^{1024}+1)
//
//       = (10 − 3) + (35 + 6)x + 21x² + 5x^{1023}
//       = 7 + 41x + 21x² + 5x^{1023}
//
assert_eq!(c[0], 7);
assert_eq!(c[1], 41);
assert_eq!(c[2], 21);
assert_eq!(c[degree - 1], 5);
// all other coefficients are zero
assert!(c[3..degree - 1].iter().all(|&v| v == 0));
```

## API reference

### Scalar operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `add_mod` | `(a, b, modulus) → u64` | `(a + b) mod modulus` |
| `sub_mod` | `(a, b, modulus) → u64` | `(a − b) mod modulus` |
| `multiply_mod` | `(a, b, modulus) → u64` | `(a × b) mod modulus` |
| `power_mod` | `(base, exp, modulus) → u64` | `base^exp mod modulus` |
| `inv_mod` | `(a, modulus) → u64` | Modular inverse of `a` |

### Element-wise vector operations

All operate on slices of equal length and write to `result`.

| Function | Description |
|----------|-------------|
| `eltwise_add_mod` | Vector addition mod `p` |
| `eltwise_sub_mod` | Vector subtraction mod `p` |
| `eltwise_mult_mod` | Vector multiplication mod `p` |
| `eltwise_fma_mod` | Fused multiply-add: `a*b + c mod p` |
| `eltwise_reduce_mod` | Reduce each element mod `p` |

### NTT

| Function | Description |
|----------|-------------|
| `ntt_forward_in_place` | In-place forward NTT |
| `ntt_inverse_in_place` | In-place inverse NTT |
| `incomplete_ntt_forward_in_place` | Coefficient → even-odd incomplete-NTT |
| `incomplete_ntt_inverse_in_place` | Even-odd incomplete-NTT → coefficient |

### Ring multiplication

| Function | Description |
|----------|-------------|
| `fused_incomplete_ntt_mult` | Multiply two polynomials in incomplete-NTT form. Shift factors are computed once and cached per `(n, modulus)` pair. |

## Running the demo

A self-contained example exercising every feature is included in `examples/demo.rs`:

```bash
cargo run --release --example demo
```

## Requirements

- **Rust nightly** (uses `target_feature_inline_always`)
- AVX-512 optional but recommended for peak performance on x86-64; portable scalar fallbacks run on any platform

## License

Licensed under the Apache License, Version 2.0 — see [LICENSE](LICENSE) for details.

This project is derived from [Intel HEXL](https://github.com/IntelLabs/hexl),
which is Copyright (c) Intel Corporation and licensed under Apache-2.0.
