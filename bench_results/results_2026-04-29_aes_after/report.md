# AES-256-CTR CRS sampler — 2026-04-29

Switched the public-seed CRS sampler from SHAKE128 to AES-256-CTR
(key = `blake3(seed)`, IV = 0). Same `from_seed` / `fill_ring_element`
shape, same per-coefficient u64 rejection sampling against
`u64::MAX − (u64::MAX % MOD_Q)`.

## Why

Microbenched on this Xeon (AES-NI present):

| PRG          | 16 MB throughput |
|--------------|-----------------:|
| SHAKE128     | 302 MB/s         |
| AES-256-CTR  | **9374 MB/s**    |

~31× headroom. SHAKE128 was the wall on sample-heavy CRS shapes (e.g.
the unstruct-ck branch needs ~2.7 GB of XOF output for p-26).

## Per-run CRS gen time (ms)

`main` (current branch):

| Feature | run 1 | run 2 | run 3 | avg of 3 |
|---------|------:|------:|------:|---------:|
| p-26    |  778  |  783  |  781  |   781    |
| p-28    | 1746  | 1751  | 1754  |  1750    |
| p-30    | 3716  | 3746  | 3733  |  3731    |

`@osdnk/unstruct-ck` (worktree, p-26 only):

| Feature | run 1 | run 2 | run 3 | avg of 3 |
|---------|------:|------:|------:|---------:|
| p-26    | 1317  | 1320  | 1301  |  1313    |

## Comparison

| Branch / p-26 CRS gen | rand baseline | SHAKE128 | AES-CTR | Δ vs rand |
|-----------------------|--------------:|---------:|--------:|----------:|
| `main`                |     774 ms    |  778 ms  | 781 ms  | +0.9%     |
| `@osdnk/unstruct-ck`  |    1356 ms    | 9920 ms  | 1313 ms | −3.2%     |

AES-CTR matches `rand` (StdRng) on `main` (sampling is a few % of total — tensor expansion dominates) and **beats `rand` on unstruct-ck** while still being a publicly-derived deterministic stream from `PUBLIC_CRS_SEED`.

## Pipeline (3-run averages on main)

| Feature | Commit | Prover | Verifier | Proof size (KB) |
|---------|-------:|-------:|---------:|----------------:|
| p-26    | 1.439s | 1.444s | 6.78 ms  | 186.55          |
| p-28    | 5.875s | 2.975s | 7.09 ms  | 186.57          |
| p-30    | 23.87s | 8.302s | 11.72 ms | 186.72          |

End-to-end pipeline runs cleanly across p-26/p-28/p-30; commit/prover/verifier and proof size all within run noise vs the SHAKE128 variant. Proof size shifts only because the CRS contents differ.

## Machine config

```
Architecture:                         x86_64
Model name:                           Intel(R) Xeon(R) Platinum 8488C
Flags (relevant):                     aes (AES-NI), avx512f, avx512dq, avx512vbmi2, sha_ni
Mem:                                  1.0 TiB
```
