# Benchmark Results — 2026-04-24 12:19

Command: `cargo run --release --features {p-XX},incomplete-rexl,unsafe-sumcheck`

Protocol: 1 warmup run (discarded) + 3 timed runs per feature.

Notes:
- **Commit time** = `TOTAL Commit time` printed by the witness-commit stage.
- **Prover time** = `TOTAL Prover time` (sumcheck/prover phase, excludes commit time).
- Proof size is deterministic across runs for a given feature set.

## Per-run measurements

| Feature | Run | Commit | Prover  | Verifier | Proof size (KB) |
|---------|-----|-------:|--------:|---------:|----------------:|
| p-26    | 1   | 1.41s  | 1.62s   | 18.2ms   | 187             |
| p-26    | 2   | 1.40s  | 1.63s   | 17.9ms   | 187             |
| p-26    | 3   | 1.41s  | 1.61s   | 17.9ms   | 187             |
| p-28    | 1   | 5.78s  | 3.83s   | 24.0ms   | 195             |
| p-28    | 2   | 5.71s  | 3.79s   | 23.9ms   | 195             |
| p-28    | 3   | 5.76s  | 3.80s   | 24.1ms   | 195             |
| p-30    | 1   | 23.0s  | 12.9s   | 43.4ms   | 210             |
| p-30    | 2   | 23.2s  | 12.8s   | 43.9ms   | 210             |
| p-30    | 3   | 23.1s  | 13.0s   | 44.9ms   | 210             |

## Averages

| Feature | Commit | Prover  | Verifier | Proof size (KB) |
|---------|-------:|--------:|---------:|----------------:|
| p-26    | 1.41s  | 1.62s   | 18.0ms   | 187             |
| p-28    | 5.75s  | 3.80s   | 24.0ms   | 195             |
| p-30    | 23.1s  | 12.9s   | 44.1ms   | 210             |

## Paper row (LaTeX)

```
& 1.41s & 1.62s & 18.0ms  & 187 & 5.75s & 3.80s & 24.0ms  & 195 & 23.1s & 12.9s & 44.1ms  & 210\\
```

## Machine config

- **CPU**: 11th Gen Intel(R) Core(TM) i7-11850H @ 2.50GHz (Tiger Lake-H, released 2021)
- **Architecture**: x86_64
- **RAM**: 62 GiB
- **Instruction sets (relevant)**: AVX, AVX2, AVX-512 (AVX512F, AVX512DQ, AVX512BW, AVX512VL, AVX512CD, AVX512VBMI, AVX512VBMI2, AVX512IFMA, AVX512VNNI, AVX512BITALG, AVX512VPOPCNTDQ, AVX512_VP2INTERSECT), AES-NI, SHA-NI, VAES, VPCLMULQDQ, GFNI, BMI1, BMI2, FMA
