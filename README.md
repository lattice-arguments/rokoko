![Project Banner](banner.png)
# RoKoko

A Rust implementation of RoKoko, an efficient lattice-based succint argument system. 

## Intro

Our protocol is run over power-of-two cyclotomic rings, and parameters are selected such that the ring splits into factors of 2-degree ("almost splitting"), which allows to leverage incomplete NTT for efficient multiplication.

The sumcheck protocol efficiently enforces a collection of algebraic constraints over committed and folded witnesses. A general, highly modular interface for sumcheck protocols is provided, which supports different constraints and may be used for different relations.

We implement vectorized random projections for full ring elements and coefficients, and specifically achieve a higher degree of vectorization for the first kind by leveraging smaller registers and thus utilizing a higher number of lanes.


## Build and run instructions

The project supports two interchangeable backends for ring arithmetic:

- `rust-hexl` — a [pure Rust implementation](incomplete-rexl/README.md) for modular arithmetic and NTT operations
- HEXL C++ bindings — native bindings to the Intel HEXL library

Unlike HEXL, `rust-hexl` can run on any Rust-supported platform (at degraded performance).

For the best performance, it is necessary to manually enable the different AVX-512 features flags for the Rust compiler.
```
export RUSTFLAGS="-C target-feature=+avx512f,+avx512bw,+avx512dq,+avx512vbmi2 -C linker=gcc"
```
Note that even if your processor advertises AVX-512 support, it may not support all AVX-512 instruction subsets, as [listed here](https://en.wikipedia.org/wiki/AVX-512#CPUs_with_AVX-512).
If your platform does not support some of the listed target features, remove the unsupported ones. Performance will degrade accordingly.

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
pub fn verifier_round_simple(
    crs: &CRS,
    config: &SimpleConfig,
    commitment: &BasicCommitment,
    round_proof: &SimpleRoundProof,
    evaluation_points_inner: &[StructuredRow],
    evaluation_points_outer: &[StructuredRow],
    claims: &[RingElement],
    hash_wrapper: Option<HashWrapper>,
)
```

### Sumcheck interface

We support different constraint type, with each encoding a specific semantic guarantee:

* `Type0`: Basic commitment correctness - verifies `CK · folded_witness = commitment · fold_challenge`
* `Type1`: Inner evaluation consistency - verifies opening RHS matches witness evaluation
* `Type2`: Outer evaluation consistency - verifies opening produces the claimed scalar result
* `Type3`: Projection validity (block-diagonal) - verifies projection image is correctly computed from witness
* `Type3_1`: Projection validity (Kronecker) - verifies `c^T (I ⊗ P) · witness = c^T projection_image · fold_challenge`
* `Type4`: Recursive commitment well-formedness - verifies the entire recursive commitment  tree structure
* `Type5`: Witness norm check - verifies `<combined_witness, conjugated_witness> = norm_claim`

Sumcheck contexts are initiliazed by `init_sumcheck`, which currently builds intializes all sumcheck gadges together.
The sumcheck protocol (over all constraint types) then can be run by the runner, which interface is defined as
```rust
pub fn sumcheck(
    config: &SumcheckConfig,
    combined_witness: &Vec<RingElement>,
    projection_matrix: &ProjectionMatrix,
    folding_challenges: &Vec<RingElement>,
    challenges_batching_projection_1: &Option<&[BatchedProjectionChallenges; NOF_BATCHES]>,
    opening: &Opening,
    sumcheck_context: &mut SumcheckContext,
    hash_wrapper: &mut HashWrapper,
) -> (
    RingElement,
    RingElement,
    RingElement,
    RingElement,
    Vec<Polynomial<QuadraticExtension>>,
    Vec<RingElement>,
    Option<Vec<RingElement>>,
)
```
In order, different claim over the witness and conjugated witness are returned, alongside with norm and inner norm claims. Additionally, the sumcheck runner returns the round polynomials, evaluations points and finally optional constant claims.

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

## Experiments
This codebase has been benchmarked on a Precision 750, which features a Intel Core i7-11850H and 64GB of Memory. The benchmarks logs have been placed under `experiments/tiger_lake` folder.

Additionally, benchmarks of [Greyhound](https://github.com/lattice-dogs/labrador) and [SALSAA](https://github.com/lattice-arguments/salsaaa) have been recorded on the same machine for polynomial degrees 2^26, 2^28.

Due to memory requirements for polynomial degrees 2^30 exceeding 64GB, respective benchmarks for Greyhound and SALSAA have been ran on a different machine (Dell PowerEdge XE8640 with Xeon Platinum 8468) and placed in the `sapphire_rapids` folder.

## Features

* `rust-hexl`: enable pure-Rust ring arithmetic backend
* `p-26, p-28, p-30`: parameters for polynomial degrees 2^26, 2^28, 2^30 respectively
* `unsafe-sumcheck`: enables zero-cost borrow checking by using `UnsafeCell` instead of `RefCell` in sumcheck subprotocols
* `debug-hardness`: additional checks and prints for L2 norm in prover
* `debug-decomp`: additional checks for decomposition and overflows in type 0 projections

## License

RoKoko is licensed under the Apache 2.0 License.
