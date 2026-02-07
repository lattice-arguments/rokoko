
![Project Banner](banner.png)
# RoKoko

## Benchmarks

Run the eltwise multiplication benchmarks (requires AVX-512):

```bash
cargo bench --bench eltwise_bench --features rust-hexl
```

This benchmarks three kernels across polynomial degrees 2^6 through 2^13:
- `hexl_rust/eltwise_mult_mod` — single element-wise modular multiply
- `bindings/eltwise_mult_mod` — C++ HEXL FFI element-wise modular multiply
- `hexl_rust/fused_incomplete_ntt_mult` — fused incomplete-NTT multiplication (split_degree=2 schoolbook)

### rdtsc microbenchmark

For cycle-accurate measurements (useful for comparison with the C++ proof-friendly-CKKS benchmark):

```bash
cargo bench --bench rdtsc_bench --features rust-hexl
```

This uses the x86 `rdtsc` instruction to report both CPU cycle counts and wall-clock nanoseconds per iteration. It measures `eltwise_mult_mod` and `fused_incomplete_ntt_mult` across sizes 2^6 through 2^13 with 50k warmup iterations and 200k measurement iterations, reporting best and average cycles.

### Comparison with proof-friendly-CKKS (C++)

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

#### 3. Build and run

```bash
make main
LD_LIBRARY_PATH=./src/third-party/hexl/build/hexl/lib ./main
```

Change `N` on line 95 to test different sizes (e.g. `1ULL<<7` for N=128, `1ULL<<14` for N=16384).

**Fused incomplete-NTT multiply** (split_degree=2, L=1):

The C++ benchmark reports `N` as the full polynomial degree, while the Rust benchmarks use `n = N / split_degree` (the chunk size). For a direct comparison:

| N     | n=N/2 | C++ CKKS (ns) | Rust fused (ns) | Speedup |
|-------|-------|---------------|-----------------|---------|
| 128   | 64    | 134           | 93              | 1.44×   |
| 256   | 128   | 260           | 190             | 1.37×   |
| 512   | 256   | 548           | 378             | 1.45×   |
| 1024  | 512   | 1019          | 755             | 1.35×   |
| 2048  | 1024  | 2112          | 1508            | 1.40×   |
| 4096  | 2048  | 3985          | 3035            | 1.31×   |
| 8192  | 4096  | 9737          | 6071            | 1.60×   |
| 16384 | 8192  | 18159         | 12038           | 1.51×   |
