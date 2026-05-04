# Profiling

The `rokoko-profiling` crate provides `tracing`-based instrumentation for the
rokoko prover and verifier. Every protocol round and sub-step is recorded as
a span; protocol diagnostics (proof sizes, verifier norms, per-round shapes)
fire as `tracing::debug!` / `tracing::trace!` events you can opt into via
`RUST_LOG`.

## Enabling

The instrumentation is gated by the `profile` Cargo feature on the `rokoko`
crate:

```bash
cargo +nightly run --release \
  --features incomplete-rexl,p-26,unsafe-sumcheck,profile
```

Without `profile`, all `info_span!` / `instrument` calls compile to no-ops at
runtime (no subscriber installed) and the binary produces no timing output.

## Outputs

With `profile` enabled, four artifacts are produced:

### 1. Live console stream

Indented hierarchical timing rendered as the program runs. Spans at depth
≤ 1 print as they close (so you see `commit`, `prover_round`, `verifier`
total times during the run). Deeper spans are tracked but suppressed from
the live stream — they appear in the end-of-run summary and the Chrome JSON.

### 2. End-of-run profile summary

Printed once when the program exits. Phase-segmented hierarchical tree of
every span with **edge-specific aggregation** — when child Y is shown under
parent X, the time and call count come from the (X, Y) edge, not the
global total for Y. So percentages never exceed 100% of parent. Names are
stripped of their phase prefix where applicable; recursive cycles are
stubbed with `…name`.

### 3. Chrome / Perfetto JSON

Written to `bench_results/traces/{name}.json` (where `{name}` is `p26` /
`p28` / `p30` based on the active parameter feature). Open at
<https://ui.perfetto.dev/> for an interactive flame-graph-style timeline.
Click any span to see start/end timestamps, source location, and field
values; the search bar highlights all matching spans.

### 4. Snapshot JSON

Written to `bench_results/snapshots/{name}.json`. Flat aggregation of every
span (`total_ns`, `calls`) keyed by name, plus metadata (git SHA, ISO date,
active features, machine string). Designed for diffing across runs —
compact, stable, human-readable.

## Filtering with `RUST_LOG`

The default filter is `info`, which mutes `debug!` / `trace!` events. Set
`RUST_LOG=debug` or `RUST_LOG=trace` to bring them back:

```bash
# adds proof-size, verifier norms, per-round shape:
RUST_LOG=debug cargo +nightly run --release --features ...,profile

# adds composed-witness layout tables:
RUST_LOG=trace cargo +nightly run --release --features ...,profile
```

Per-module filtering works too:

```bash
# only proof-size events:
RUST_LOG=info,rokoko::protocol::config=debug cargo run ...
```

## Diffing two runs

Each parameter set has its own committed baseline snapshot:

- `bench_results/snapshots/p26-baseline.json`
- `bench_results/snapshots/p28-baseline.json`
- `bench_results/snapshots/p30-baseline.json`

Per-PR workflow:

```bash
# 1. on your optimization branch, take a fresh snapshot
cargo +nightly run --release --features incomplete-rexl,p-26,unsafe-sumcheck,profile
# ↳ writes bench_results/snapshots/p26.json (gitignored)

# 2. diff against the committed baseline
python3 bench_results/diff_snapshots.py \
  bench_results/snapshots/p26-baseline.json \
  bench_results/snapshots/p26.json

# 3. paste the markdown table into the PR description
```

Refreshing the baseline (occasional maintenance — only when `main` moves
significantly enough that the existing baseline is misleading):

```bash
git checkout main
cargo +nightly run --release --features incomplete-rexl,p-26,unsafe-sumcheck,profile
mv bench_results/snapshots/p26.json bench_results/snapshots/p26-baseline.json
git checkout -b refresh-p26-baseline
git add bench_results/snapshots/p26-baseline.json
git commit -m "refresh p26 baseline"
```

## Extending

Add a span anywhere in the codebase:

```rust
let _s = tracing::info_span!("my::component").entered();
// ... timed code ...
```

Or attribute a whole function:

```rust
#[tracing::instrument(skip_all, name = "my::component")]
fn my_function(...) { ... }
```

The new span shows up in all four artifacts automatically — no schema
changes, no other plumbing.

## A note on inclusive vs self time

Span totals are **inclusive** — a parent's `total_ns` includes the time
spent in its children. To compute self-time:

```
self_ns(parent) = parent_ns - sum(direct_child_ns)
```

Both the snapshot JSON and the Chrome JSON expose this directly. The
end-of-run summary tree shows inclusive times throughout, which is the
standard profiler convention.
