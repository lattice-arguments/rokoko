## Benchmarks

Run the eltwise multiplication benchmarks (requires AVX-512):

```bash
cargo bench --bench eltwise_bench
```

This benchmarks three kernels across polynomial degrees 2^6 through 2^13:
- `hexl_rust/eltwise_mult_mod` — single element-wise modular multiply
- `bindings/eltwise_mult_mod` — the same multiply through the raw-pointer wrappers in `rokoko::hexl::bindings`
- `hexl_rust/fused_incomplete_ntt_mult` — fused incomplete-NTT multiplication (split_degree=2 Karatsuba)


# Comparison with proof-friendly-CKKS (C++)

To compare against the C++ [proof-friendly-CKKS](https://github.com/vfhe/proof-friendly-CKKS) decomposed polynomial multiply:

In `../proof-friendly-CKKS/lib/main_benchmark.cpp`, make two changes:

**Line 95** — change `L`, `N`, and `split_degree`:
```c
// FROM:
const uint64_t L = 3, N = 1ULL<<13, split_degree = 4;
// TO:
const uint64_t L = 1, N = 1ULL<<13, split_degree = 2;
```

**Line 163** — uncomment `test_arith()`:
```c
// FROM:
test_encoding_mp();
// test_arith();
// TO:
// test_encoding_mp();
test_arith();
```

## Build and run

```bash
make hexl
make main
LD_LIBRARY_PATH=./src/third-party/hexl/build/hexl/lib ./main
```

Change `N` on line 95 to test different sizes (e.g. `1ULL<<7` for N=128, `1ULL<<14` for N=16384).

## Fused incomplete-NTT multiply (split_degree=2, L=1)

The C++ benchmark reports `N` as the full polynomial degree, while the Rust benchmarks use `n = N / split_degree` (the chunk size). For a direct comparison:

| N     | n=N/2 | C++ CKKS (ns) | Rust fused (ns) | Speedup |
|-------|-------|---------------|-----------------|---------|
| 128   | 64    | 134           | 95              | 1.41×   |
| 256   | 128   | 260           | 180             | 1.44×   |
| 512   | 256   | 548           | 373             | 1.47×   |
| 1024  | 512   | 1019          | 747             | 1.36×   |
| 2048  | 1024  | 2112          | 1481            | 1.43×   |
| 4096  | 2048  | 3985          | 2921            | 1.36×   |
| 8192  | 4096  | 9737          | 5972            | 1.63×   |
| 16384 | 8192  | 18159         | 11970           | 1.52×   |
