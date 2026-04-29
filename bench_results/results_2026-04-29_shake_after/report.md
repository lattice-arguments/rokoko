# SHAKE128 CRS perf comparison — 2026-04-29

CRS public-key sampling switched from the global `rand` RNG (StdRng seeded
via Blake3) to SHAKE128 squeezed from a public seed
(`PUBLIC_CRS_SEED = b"rokoko-CRS-v1/SHAKE128 public seed"`).

## Command

```
cargo run --release --features {p-26|p-28|p-30},incomplete-rexl,unsafe-sumcheck
```

## Protocol

For each feature: 1 warmup + 3 timed runs, sequentially. Same machine, same
binary configuration. Baseline runs (`results_2026-04-29_shake_before/`)
were taken with the unmodified `sample_random_vector` path; `*_after`
captures the SHAKE128 path. CRS gen timing comes from a new
`TOTAL CRS gen time:` line added around `CRS::gen_crs` in
`src/protocol/parties/executor.rs`.

## Per-run CRS gen time (ms)

| Feature | Path    | warmup | run 1 | run 2 | run 3 | avg of 3 |
|---------|---------|-------:|------:|------:|------:|---------:|
| p-26    | before  |  787   |  781  |  771  |  772  |   774    |
| p-26    | after   |  781   |  773  |  777  |  783  |   778    |
| p-28    | before  | 1714   | 1730  | 1714  | 1737  |  1727    |
| p-28    | after   | 1723   | 1716  | 1755  | 1730  |  1733    |
| p-30    | before  | 3711   | 3681  | 3691  | 3671  |  3681    |
| p-30    | after   | 3712   | 3710  | 3662  | 3746  |  3706    |

## Averages (3 timed runs)

| Feature | CRS before | CRS after | Δ      |
|---------|-----------:|----------:|-------:|
| p-26    |    774 ms  |   778 ms  | +0.5%  |
| p-28    |   1727 ms  |  1733 ms  | +0.4%  |
| p-30    |   3681 ms  |  3706 ms  | +0.7%  |

## Pipeline sanity (3-run averages)

The full pipeline still runs end-to-end with the new sampler, with
commit/prover/verifier times and proof sizes essentially unchanged
(differences within run-to-run noise; proof size changes only because the
CRS contents differ, which shifts a few bit-level signs in the projection):

| Feature | Commit before | Commit after | Prover before | Prover after | Verifier before | Verifier after | Proof before (KB) | Proof after (KB) |
|---------|--------------:|-------------:|--------------:|-------------:|----------------:|---------------:|------------------:|-----------------:|
| p-26    | 1.397s        | 1.440s       | 1.398s        | 1.447s       | 6.81 ms         | 6.87 ms        | 186.57            | 186.56           |
| p-28    | 5.353s        | 5.505s       | 2.848s        | 2.923s       | 6.94 ms         | 7.03 ms        | 186.63            | 186.68           |
| p-30    | 22.361s       | 22.361s      | 8.166s        | 8.169s       | 11.51 ms        | 11.39 ms       | 186.69            | 186.70           |

## Conclusion

CRS gen time is unchanged (within ~1%). The random sampling step is a
small fraction of total CRS gen time — `PreprocessedRow::from_structured_row`
(the tensor-product expansion) dominates — so the choice of PRG matters
much less than its rate. SHAKE128 keeps up with `StdRng` here while making
the CRS reproducible from a single public seed.

## Machine config

```
Architecture:                         x86_64
Model name:                           Intel(R) Xeon(R) Platinum 8488C
Flags:                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid tsc_known_freq pni pclmulqdq vmx ssse3 fma cx16 pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves avx512_bf16 wbnoinvd arat avx512vbmi umip avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx512_vpopcntdq la57 rdpid bus_lock_detect cldemote movdiri movdir64b enqcmd fsrm md_clear serialize tsxldtrk avx512_fp16 arch_capabilities
Mem (free -h, line 2):
total        used        free      shared  buff/cache   available
Mi:           1.0Ti        17Gi        67Gi        20Mi       955Gi       990Gi
```

(AVX-512F + AVX-512DQ + AVX-512VBMI2 present, which is what the kernels rely on.)
