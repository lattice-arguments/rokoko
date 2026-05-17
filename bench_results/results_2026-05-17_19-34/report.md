# Benchmark report — 2026-05-17 19:34

## Command

```
cargo run --release --features {FEATURE},incomplete-rexl,unsafe-sumcheck
```

`{FEATURE}` ∈ `p-26`, `p-28`, `p-30`.

## Protocol

Per `ai_instructions/HOW_TO_RUN.md`: for each feature, in order p-26 → p-28 → p-30,
1 warmup run (discarded) followed by 3 timed runs, averaged. Runs executed
sequentially, never in parallel. Changing the `p-XX` feature triggers a full
workspace rebuild.

## Per-run measurements

| Feature | Run | Commit | Prover | Verifier | Proof size |
|---------|-----|--------|--------|----------|------------|
| p-26    | 1   | 1.13s  | 1.34s  | 7.36ms   | 157 KB     |
| p-26    | 2   | 1.14s  | 1.37s  | 7.65ms   | 157 KB     |
| p-26    | 3   | 1.13s  | 1.37s  | 7.49ms   | 157 KB     |
| p-28    | 1   | 4.29s  | 2.95s  | 8.07ms   | 157 KB     |
| p-28    | 2   | 4.31s  | 2.96s  | 7.80ms   | 157 KB     |
| p-28    | 3   | 4.34s  | 2.99s  | 8.04ms   | 157 KB     |
| p-30    | 1   | 17.9s  | 8.78s  | 13.0ms   | 157 KB     |
| p-30    | 2   | 18.1s  | 9.00s  | 13.3ms   | 157 KB     |
| p-30    | 3   | 18.1s  | 9.00s  | 13.1ms   | 157 KB     |

## Averages (across 3 timed runs)

| Feature | Commit | Prover | Verifier | Proof size |
|---------|--------|--------|----------|------------|
| p-26    | 1.13s  | 1.36s  | 7.50ms   | 157 KB     |
| p-28    | 4.31s  | 2.97s  | 7.97ms   | 157 KB     |
| p-30    | 18.1s  | 8.92s  | 13.1ms   | 157 KB     |

Prover time excludes commit time; the two are reported separately.

## LaTeX paper row

```
& 1.13s & 1.36s & 7.50ms  & 157 & 4.31s & 2.97s & 7.97ms & 157 & 18.1s & 8.92s & 13.1ms  & 157\\
```

## Machine config

- CPU: 11th Gen Intel Core i7-11850H @ 2.50GHz (Tiger Lake, 2021)
- Architecture: x86_64
- RAM: 62 GiB total (~54 GiB available at run time)
- AVX-512 instruction sets: avx512f, avx512bw, avx512cd, avx512dq, avx512vl,
  avx512ifma, avx512vbmi, avx512_vbmi2, avx512_vnni, avx512_vpopcntdq,
  avx512_bitalg, avx512_vp2intersect
