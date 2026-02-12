![Project Banner](banner.png)
# RoKoko

A Rust implementation of RoKoko, an efficient lattice-based succint argument system.

## Build and run instructions

For more a detailed comparison about the backends, check below.

#### Using `rust-hexl` feature (pure Rust backend)
The protocol can be directly compiled and run with 
```
cargo +nightly run --release --features rust-hexl
```

#### Using HEXL C++ bindings
It is first required to build the library submodule separetely.

It is necessary to first clone and build the HEXL submodule. Run 
```
git submodule update --init --recursive
```
Then run
```
make hexl
make wrapper
export LD_LIBRARY_PATH=./hexl-bindings/hexl/build/hexl/lib:$(pwd)
```
And finally simply run

```
cargo +nightly run --release
```

## API

### Commiter
```rust
pub fn commit(
    crs: &CRS,
    config: &SumcheckConfig,
    witness: &VerticallyAlignedMatrix<RingElement>,
) -> (CommitmentWithAux, Vec<RingElement>)
```
Performs the basic commitment via `commit_basic`, and then outputs a tuple consisting of the recursive commitment (including auxiliary data) in `CommitmentWithAux` and the most inner commitment.

### Prover
```rust
pub fn prover_round(
    crs: &CRS,
    config: &SumcheckConfig,
    commitment_with_aux: &CommitmentWithAux,
    witness: &VerticallyAlignedMatrix<RingElement>,
    evaluation_points_inner: &Vec<StructuredRow>,
    evaluation_points_outer: &Vec<StructuredRow>,
    sumcheck_context: &mut SumcheckContext,
    with_claims: bool,
    hash_wrapper: Option<HashWrapper>,
) -> (SumcheckRoundProof, Option<Vec<RingElement>>)
```

The prover takes as input the CRS, `SumcheckConfig`, recursive commitment (plus auxiliary data), witness, structured evaluations points. Additionally, a `with_claims` flag can be provided, deciding to output evaluation claims. A initialized Fiat-Shamir transcript may be provided via `hash_wrapper`, and otherwise newly initiliazed inside the round. 
Note that prover is recursively called, as `SumcheckConfig` can define multiple "standard" or a "simple" rounds.

```rust
pub fn prover_round_simple(
    config: &SimpleConfig,
    commitment: &BasicCommitment,
    witness: &VerticallyAlignedMatrix<RingElement>,
    evaluation_points_inner: &Vec<StructuredRow>,
    evaluation_points_outer: &Vec<StructuredRow>,
    hash_wrapper: Option<HashWrapper>,
) -> SimpleRoundProof
```
Simple prover rounds require a non-recursive `BasicCommitment` and no `SumcheckContext`.
### Verifier
```rust
pub fn verifier_round(
    crs: &CRS,
    config: &SumcheckConfig,
    rc_commitment: &[RingElement],
    round_proof: &SumcheckRoundProof,
    evaluation_points_inner: &[StructuredRow],
    evaluation_points_outer: &[StructuredRow],
    claims: &[RingElement],
    sumcheck_context_verifier: &mut VerifierSumcheckContext,
    hash_wrapper_verifier: Option<HashWrapper>,
)
```
The verifier interface, similarly to the prover, requires a CRS, `SumcheckConfig`, structured evaluations points and an (optionally pre-initialized) Fiat-Shamir transcript. Additionally, it takes as input the claimed polynomial evalutations to be checked and a mutable `VerifierSumcheckContext`.

Just as the prover, `verifier_round` is recursively called, and the prover sumcheck and simple rounds are checked against the respective interface.
```rust
pub fn verifier_round(
    crs: &CRS,
    config: &SumcheckConfig,
    rc_commitment: &[RingElement],
    round_proof: &SumcheckRoundProof,
    evaluation_points_inner: &[StructuredRow],
    evaluation_points_outer: &[StructuredRow],
    claims: &[RingElement],
    sumcheck_context_verifier: &mut VerifierSumcheckContext,
    hash_wrapper_verifier: Option<HashWrapper>,
)
```

## Cofiguration and structure

Ring degrees `DEGREE`, modulus `MOD_Q` and number of batches `NOF_BATCHED` are defined as constants in `src/common.config.rs`.

Protocol configuration is defined in `src/protocol/config.rs`. Currently, parameters for the configuration are concretely defined in `src/procotol/params.rs`. In the future, we plan to provide automic selection.

Each run executed by the prover or verifier consists of one or more **rounds**. Each round is either:

- `Config::Sumcheck(SumcheckConfig)` — the main sumcheck-based round, optionally chaining into another round(s)
- `Config::Simple(SimpleConfig)` — sumcheck-less round with plain folded witness, executed last

### Core parameters

The following parameters are shared by both `SumcheckConfig` and `SimpleConfig`.

- `witness_height`: number of rows in the witness matrix.
- `witness_width`: number of columns in the witness matrix.
- `projection_ratio`: target witness height reduction by projections
- `projection_height`: height of the projection image
- `basic_commitment_rank`: rank of the (non recursive) commitment

### Sumcheck configuration

The following parameters are sumcheck-specific and defined in `SumcheckConfig`.

Sumcheck rounds
- `commitment_recursion: RecursionConfig`: controls how witness commitments are recursively represented via decomposition + prefix.

- `opening_recursion: RecursionConfig`  Same idea, but for opening proofs. In many setups it mirrors `commitment_recursion`.

- `projection_recursion: Projection`: selects which (if any) projection to run.

- `nof_openings`: number of openings per round.

- `next_level_usage_ratio`  define usage of witness width for the next level (as a fraction)

- Witness decomposition related settings:
  - `witness_decomposition_base_log`
  - `witness_decomposition_chunks`
  - `folded_witness_prefix: Prefix`
  - `composed_witness_length`

Different kind of projections can be selected through:
```rust
pub enum Projection {
    Type0(Type0ProjectionConfig),
    Type1(Type1ProjectionConfig),
    Skip,
}
```
where `Type0` defines the random projection over the full ring elements, and `Type1` the random projections over the ring coefficient.

## Features

* `rust-hexl`: enable pure-Rust ring arithmetic backend
* `p-26, p-28, p-30`: parameters for polynomial degrees 2^26, 2^28, 2^30 respectively
* `unsafe-sumcheck`: enables zero-cost borrow checking by using `UnsafeCell` instead of `RefCell` in sumcheck subprotocols
* `debug-hardness`: additional checks and prints for L2 norm in prover
* `debug-decomp`: additional checks for decomposition and overflows in type 0 projections

## Platform and requirements
The project supports two interchangeable backends for ring arithmetic:

- `rust-hexl` — a pure Rust implementation for modular arithmetic and NTT operations
- HEXL C++ bindings — native bindings to the Intel HEXL library

Unlike HEXL, `rust-hexl` runs on any Rust-supported platform. On non-`x86_64` architectures, or when AVX-512 is unavailable, it automatically falls back to weaker SIMD instructions or naive multiplication. This ensures portability at the cost of reduced performance.

It is necessary to manually enable the different AVX-512 features for the Rust compiler.
```
export RUSTFLAGS="-C target-feature=+avx512f,+avx512bw,+avx512dq,+avx512vbmi2 -C linker=gcc"
```
Note that even if your processor advertises AVX-512 support, it may not support all AVX-512 instruction subsets, as [listed here](https://en.wikipedia.org/wiki/AVX-512#CPUs_with_AVX-512).
If your platform does not support some of the listed target features, remove the unsupported ones. Performance will degrade accordingly.

## License

RoKoko is licensed under the Apache 2.0 License.
