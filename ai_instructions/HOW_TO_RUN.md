# How to run benchmarks

Reference for future experiment runs. Follow this procedure exactly unless instructed otherwise.

## The command

```
cargo run --release -- {SET}
```

Where `{SET}` is one of `p-22`, `p-24`, `p-26`, `p-28`, `p-29`, `p-30` (or whichever is requested);
without an argument the binary runs `p-28`. `p-29` has only the exact-norm chain, so it runs with
`--features snark`. The parameter set is a runtime argument, so one build serves every set.

The sets themselves live in `src/instantiation.rs` (binary crate); the library only receives the
chain as a `rokoko::protocol::params::Instantiation`. A build for `rings/n128_d4.toml` uses
`src/instantiation_n128_d4.rs` instead, chosen in `src/main.rs` by the spec's file name.

Features are defined in `Cargo.toml`; the defaults already include:
- `unsafe-sumcheck` — turns on unsafe sumcheck optimizations
- `crt-commitment` — the root commitment through the CRT kernel

Other useful features: `snark` (exact-norm SNARK mode), `calibration` (prints the measured norms
as a paste-ready `NB_*` table, e.g. `cargo run --release --features calibration -- p-26`), `profile`.

## The ring

The ring degree, modulus and challenge-set constants are fixed at build time from a small spec
file; `rings/default.toml` is used unless `ROKOKO_RING` names another one, e.g.
`ROKOKO_RING=rings/n256.toml cargo build --release`. Changing it rebuilds the crate.

## Protocol

For each parameter set, in order (default is p-26 → p-28 → p-30):

1. **1 warmup run** — discard the numbers (caches cold, JIT effects, etc).
2. **3 timed runs** — record the metrics, average across the 3.

Run them **sequentially**, not in parallel — parallel runs contaminate timing.

Build once with `cargo build --release`, then run `target/release/rokoko {SET}` per set; changing the set does not rebuild.

## Metrics to collect

Parse the following lines from stdout:

| Metric        | Log line                                      |
|---------------|-----------------------------------------------|
| Commit time   | `Witness decomposed. TOTAL Commit time: N ns` |
| Prover time   | `TOTAL Prover time: N ns`                     |
| Verifier time | `TOTAL Verifier time: N ns`                   |
| Proof size    | `Total proof size: N KB`                      |

Prover time does **not** include commit time — they are separate. Report both.

Proof size is deterministic across runs for a given parameter set; only times vary.

## Output format

### Presentation units (in the per-run + averages tables)

- Commit / Prover times: **seconds, 2–3 sig figs** (e.g. `1.41s`, `23.1s`).
- Verifier time: **milliseconds, 3 sig figs** (e.g. `18.0ms`, `44.1ms`).
- Proof size: **integer KB** (round to nearest, e.g. `187`).

### LaTeX paper row

A single LaTeX row covers all parameter sets side-by-side for paste into the paper. Format (4 columns per set: commit, prover, verifier, proof size):

```
& <commit>s & <prover>s & <verifier>s  & <proof> & ... \\
```

- Commit and prover in **seconds** (e.g. `1.41s`), verifier in **milliseconds** (e.g. `18.0ms`).
- Proof size is an integer, no unit in the row.
- Example (p-26 / p-28 / p-30):

```
& 1.41s & 1.62s & 18.0ms  & 187 & 5.75s & 3.80s & 24.0ms & 195 & 23.1s & 12.9s & 44.1ms  & 210\\
```

## Output layout

Each experiment gets its own directory under `bench_results/` named with the date/time of the run: `bench_results/results_{YYYY-MM-DD}_{HH-MM}/`.

Inside that directory:
- `report.md` — the markdown report.
- `p{XX}_warmup.log`, `p{XX}_run{1,2,3}.log` — raw stdout captures for every run (warmup included for completeness).

Save raw logs directly into this directory from the start — don't stage them at the top level and then move them.

File structure:
1. Title with date/time.
2. Command used.
3. Protocol description.
4. Per-run measurements table.
5. Averages table.
6. LaTeX paper row.
7. Machine config (CPU model + year, arch, RAM, key instruction sets — especially AVX-512 variants since the codebase uses them).

Gather machine config with:
```
lscpu | grep -E "Model name|Architecture|Flags"
free -h | head -2
```

## Raw logs

Per-run logs live alongside `report.md` inside the experiment directory (see Output layout) so the numbers can be re-parsed / audited later.

## Driving this from an agent

- Use the `Bash` tool with `run_in_background: true` for each `cargo run` so timing is precise and a completion notification fires.
- Run them one at a time (wait for each notification before launching the next).
- The first build takes much longer than subsequent runs — that's expected.
- Do **not** sleep-poll; let the notification drive you.

## Checklist before reporting back

- [ ] 3 parameter sets, each: 1 warmup + 3 timed runs (9 timed runs total).
- [ ] Per-run table.
- [ ] Averages table.
- [ ] LaTeX paper row.
- [ ] Machine config section.
- [ ] Experiment directory named with date/time stamp; contains `report.md` + all raw logs.
