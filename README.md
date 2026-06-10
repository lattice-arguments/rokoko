![Project Banner](banner.png)

# RoKoko

A Rust implementation of a SNARK/PCS built from the argument system presented in _RoKoko: Lattice-based Succinct Arguments, a Committed Refinement_.

## Intro

Our protocol is run over power-of-two cyclotomic rings, and parameters are selected such that the ring splits into factors of degree 2 ("almost splitting"), which allows us to leverage incomplete NTT for efficient multiplication.

The sumcheck protocol efficiently enforces a collection of algebraic constraints over committed and folded witnesses. A general, highly modular interface for sumcheck protocols is provided, which supports different constraints and may be used for different relations.

We implement two variants of committed random projections based on the Johnson-Lindestrauss lemma, a coarse variant (`Projection::Coarse`, paper: Π^proj-c) applying a projection matrix to full ring elements, and a fine variant (`Projection::Fine`, paper: Π^proj-f) projecting only the coefficients of the witness ring elements. Both implementations are efficient and vectorised, specifically achieving a higher degree of vectorisation for the coarse projection by leveraging smaller registers and thus utilising a greater number of lanes.

## Build and Run Instructions

The project supports two interchangeable back-ends for ring arithmetic:

- `incomplete-rexl` — a [pure Rust implementation](incomplete-rexl/README.md) for modular arithmetic and NTT operations.
- HEXL C++ bindings — native bindings to the Intel HEXL library  

Unlike HEXL, `incomplete-rexl` can run on any Rust-supported platform (with degraded performance).

For the best performance, it is required to compile and run the project on an AVX-512-enabled processor.
Note that your processor may not support all AVX-512 instruction subsets, as listed here: https://en.wikipedia.org/wiki/AVX-512#CPUs_with_AVX-512.  
If your platform does not support some of the instruction subsets (such as `avx512dq` or `avx512vbmi2`), performance will degrade accordingly.

#### Using the `incomplete-rexl` feature (pure Rust back-end)

The protocol can be compiled and run directly with:

```
cargo +nightly run --release --features incomplete-rexl
```


#### Using HEXL C++ bindings

It is first necessary to build the library submodule separately.

Clone and build the HEXL submodule:

```
git submodule update --init --recursive
```

Then run:

```
make hexl
make wrapper
export LD_LIBRARY_PATH=./hexl-bindings/hexl/build/hexl/lib:$(pwd)
```

Finally, run:

```
cargo +nightly run --release
```

## Cached allocations
For the best performance, it is advisable to run the protocol twice. During the first run, the protocol collects the allocation descriptions (and stores them as a file, while printing a number of warnings about an unpopulated cache). On the next run, those allocations will be done in advance, which impact especially the commitment and verifier performance. 

## API

### Committer

```rust
pub fn commit(
    crs: &CRS,
    config: &SumcheckConfig,
    witness: &VerticallyAlignedMatrix<RingElement>,
) -> (CommitmentWithAux, Vec<RingElement>)
```

Performs the basic commitment via `commit_basic`, and then outputs a tuple consisting of the recursive commitment (including auxiliary data) in `CommitmentWithAux` and the commitment.

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

The prover takes as input the CRS, `SumcheckConfig`, the recursive commitment (plus auxiliary data), the witness, and structured evaluation points (corresponding to left-right constraints on the witness used to construct a PCS). Additionally, a `with_claims` flag may be provided to determine whether to output left-right evaluation claims. An initialised Fiat–Shamir transcript may be provided via `hash_wrapper`; otherwise, it is newly initialised within the round.

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

The verifier interface, similarly to the prover, requires a CRS, `SumcheckConfig`, structured evaluation points, and an (optionally pre-initialised) Fiat–Shamir transcript. Additionally, it takes as input the claimed polynomial evaluations to be checked and a mutable `VerifierSumcheckContext`.

### Sumcheck Interface

We support different constraint types, each encoding a specific semantic guarantee:

Each constraint family is implemented as a sumcheck gadget named after the paper constraint it enforces (former `TypeN` codenames in parentheses):

* `CommitmentFold` (Type0): basic commitment correctness - verifies `CK · folded_witness = commitment · fold_challenge`
* `InnerEvalFold` (Type1): inner evaluation consistency - verifies the opening RHS matches the witness evaluation; the claim ``matrix-from-rows(l) W = T'' from the publication.
* `OuterEvalClaim` (Type2): outer evaluation consistency - verifies the evaluation of the rows of the matrix T at the outer evaluation points; these inner products are known to the verifier.
* `CoarseProj` (Type3): coarse projection validity (block-diagonal) - verifies the projection image is correctly computed from the witness
* `FineProj` (Type3_1): fine projection validity - verifies `c^T (I ⊗ P) · witness = c^T projection_image · fold_challenge` (batched), plus the correspondence of the constant terms of the fine projection.
* `ComVerify` (Type4): recursive commitment well-formedness - verifies the recursive commitment tree structure (paper: `COM.Verify`)
* `NormCheck` (Type5): witness norm check - verifies `<combined_witness, conjugated_witness> = norm_claim` (paper: `v = <w-hat, conj(w-hat)>`), with an additional check for the innermost commitment layer.

Currently, a framework for supporting different kinds of relations is not fully exposed. Yet, all of those checks have been (generally) built from composable blocks, not specific to our relation in general. 

## Code ↔ paper notation

| Code | Paper |
|---|---|
| `witness` (`witness_height` × `witness_width`) | `W ∈ R^{m_w × r}` |
| `fold_challenge` | folding challenge `c ∈ C^r` |
| `folded_witness` / decomposed | `W c` / `w̃ = G^{-1}(W c)` |
| `next_round_data` (packed) | `ŵ = pack(w̃, Y_i, x_i, …)` |
| `next_round_witness` | `U = reshape_{r'}(ŵ)` |
| `opening.rhs` | `T = matrix-from-rows(l_j) · W` |
| `evaluation_points_inner` / `outer` | left/right vectors `l_j` / `r_j` |
| `claims` (`outer_eval_claims`) | `t_j = l_j^T W r_j` |
| `rc_commitment` / `rc_opening` | recursive commitments `com_i` of `vec(Y_i)`, aux `x_i` |
| `projection_matrix` | `J ← χ^{n_rp × m_rp}` (block-diagonal `I ⊗ J`) |
| `norm_claim` | `v = ⟨ŵ, conj(ŵ)⟩`, norm via `ct(v)` |
| `claim_over_witness`, `…_conjugate` | `z_0 = MLE[w](c)`, `z_1 = MLE[w](conj(c))` |
| sumcheck `combination` challenges | batching `γ` (via `eq(bin(i), γ)`) |
| `combination_to_field` (`RingToFieldCombiner`) | the F_{q^a}-linear map `Φ = δ^T ∘ θ_a` |
| `Prefix` selectors | packing prefixes `p_i` with `eq(p_i, ·)` |

## Configuration and Structure

Ring degrees `DEGREE`, modulus `MOD_Q`, and number of batches `NOF_BATCHED` are defined as constants in `src/common.config.rs`.

Protocol configuration is defined in `src/protocol/config.rs`. Currently, parameters for the configuration are concretely defined in `src/protocol/params.rs`. In the future, we plan to provide automatic selection.

Each run executed by the prover or verifier consists of one or more **rounds**. Each round is either:

- `Config::Sumcheck(SumcheckConfig)` — the main sumcheck-based round, optionally chaining into further round(s)
- `Config::Simple(SimpleConfig)` — a sumcheck-less round with a plain folded witness, executed last

### Core Parameters

The following parameters are shared by both `SumcheckConfig` and `SimpleConfig`:

- `witness_height`: number of rows in the witness matrix
- `witness_width`: number of columns in the witness matrix
- `projection_ratio`: target witness height reduction by projections
- `projection_height`: height of the projection image (typically, 256) 
- `basic_commitment_rank`: rank of the (non-recursive) commitment (`F_0` part of the committed linear relation from the publication)

### Sumcheck Configuration

The following parameters are sumcheck-specific and defined in `SumcheckConfig`.

Sumcheck rounds:

- `commitment_recursion: RecursionConfig`: controls how witness commitments are recursively represented via decomposition and prefix
- `opening_recursion: RecursionConfig`: same idea, but for left-constraint opening (T). In many setups, it mirrors `commitment_recursion`
- `projection_recursion: Projection`: selects which (if any) projection to run
- `nof_openings`: number of openings per round

Witness decomposition-related settings:

- `witness_decomposition_base_log`
- `witness_decomposition_chunks`
- `composed_witness_length`

The different variants of projections can be selected through:

```rust
pub enum Projection {
    Coarse(CoarseProjectionConfig),
    Fine(FineProjectionConfig),
    Skip,
}
```

where `Coarse` and `Fine` are the two random projection variants (paper: Π^proj-c and Π^proj-f).

## Experiments

This codebase has been benchmarked on a Precision 750, which features an Intel Core i7-11850H and 64 GB of memory. The benchmarks have been run using the pure-Rust back-end, specifically with the features `unsafe-sumcheck` and `incomplete-rexl` enabled. Logs have been placed under the [experiments/tiger_lake](experiments/tiger_lake) folder.

Additionally, benchmarks of [Greyhound](https://github.com/lattice-dogs/labrador) and [SALSAA](https://github.com/lattice-arguments/salsaaa) have been recorded on the same machine for polynomial degrees 2^26 and 2^28.

Due to memory requirements for polynomial degree 2^30 exceeding 64 GB, the respective benchmarks for Greyhound and SALSAA were run on a different machine (Dell PowerEdge XE8640 with Xeon Platinum 8468) and placed in the [experiments/sapphire_rapids](experiments/sapphire_rapids) folder.

## Features

* `incomplete-rexl`: enables the pure-Rust ring arithmetic back-end
* `p-26`, `p-28`, `p-30`: parameters for polynomial degrees 2^26, 2^28, and 2^30 respectively
* `unsafe-sumcheck`: enables zero-cost borrow checking by using `UnsafeCell` instead of `RefCell` in sumcheck subprotocols
* `debug-hardness`: verifies the hardness of underlying SIS instances (requires [Lattice Estimator](https://github.com/malb/lattice-estimator) cloned as a submodule and [SageMath](https://www.sagemath.org/) installed)
* `debug-decomp`: additional checks for decomposition and overflows in type 0 projections

## License

RoKoko is licensed under the Apache 2.0 Licence.
