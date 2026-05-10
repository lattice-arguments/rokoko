# Profiling

The `rokoko-profiling` crate provides `tracing`-based instrumentation. Every
protocol round and sub-step is recorded as a span; protocol diagnostics
(proof sizes, verifier norms, per-round shapes) fire as `tracing::debug!` /
`tracing::trace!` events you can opt into via `RUST_LOG`.

## Enabling

Gated by the `profile` Cargo feature on the `rokoko` crate:

```bash
cargo +nightly run --release \
  --features incomplete-rexl,p-26,unsafe-sumcheck,profile
```

Without `profile`, no subscriber is installed and the spans are runtime
no-ops.

## Outputs

With `profile` enabled, three artifacts are produced in `bench_results/`
(all gitignored — runtime artifacts are machine-specific):

- **Live console stream + end-of-run summary** — depth-limited live ticks
  as spans close, then a phase-segmented tree on exit. The tree uses
  edge-specific aggregation: when child Y is shown under parent X, the time
  comes from the (X, Y) edge, not the global total for Y, so percentages
  never exceed 100% of parent. Recursive cycles are stubbed with `…name`.
- **Chrome / Perfetto JSON** at `bench_results/traces/{name}.json`. Open at
  <https://ui.perfetto.dev/> for a flame-graph view.
- **Snapshot JSON** at `bench_results/snapshots/{name}.json`. Flat
  `(total_ns, calls)` per span name plus metadata (git SHA, date, features,
  machine).

`{name}` is `p26` / `p28` / `p30` based on the active parameter feature.

## Subtree focus: `ROKOKO_PROFILE_FOCUS`

When optimizing one component, scope the console summary and snapshot to a
subtree (the Chrome JSON stays unfiltered):

```bash
ROKOKO_PROFILE_FOCUS=commit cargo +nightly run --release --features ...,profile
ROKOKO_PROFILE_FOCUS=commit,verifier cargo +nightly run --release --features ...,profile
```

A span matches token `tok` iff its name (or any ancestor's name) equals
`tok` or starts with `tok::`.

## `RUST_LOG`

Default filter is `info`. Bring back protocol diagnostics with
`RUST_LOG=debug` (proof sizes, verifier norms, per-round shape) or
`RUST_LOG=trace` (composed-witness layout tables). Per-module filters work
too: `RUST_LOG=info,rokoko::protocol::config=debug`.

## Diffing two runs

Snapshots are not committed (different machines → noisy deltas). Collect a
local baseline, then diff:

```bash
# baseline from main
git checkout main
cargo +nightly run --release --features incomplete-rexl,p-26,unsafe-sumcheck,profile
mv bench_results/snapshots/p26.json /tmp/p26-main.json

# candidate from your branch
git checkout my-branch
cargo +nightly run --release --features incomplete-rexl,p-26,unsafe-sumcheck,profile

# diff (markdown table sorted by |Δ|)
python3 bench_results/diff_snapshots.py /tmp/p26-main.json bench_results/snapshots/p26.json
```

## Adding a span

```rust
let _s = tracing::info_span!("my::component").entered();
// ... timed code ...
```

Or attribute a whole function:

```rust
#[tracing::instrument(skip_all, name = "my::component")]
fn my_function(...) { ... }
```

The new span shows up in all three artifacts automatically.

## Inclusive vs self time

Span totals are inclusive — a parent's `total_ns` includes its children.
Self-time = `parent_ns - sum(direct_child_ns)`. The end-of-run tree shows
inclusive times throughout (standard profiler convention).
