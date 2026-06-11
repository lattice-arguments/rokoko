# The SNARK front end

The argument's native object is one committed vector; its native statement is
a batch of sumcheck claims about that vector. The flow is three steps:

1. **Commit.** Arrange the witness as a matrix `W` of ring elements
   (`VerticallyAlignedMatrix`, `height x width` a power of two) and commit it
   (`commiter::commit`). The Ajtai commitment is binding only for short
   vectors.
2. **Entry sumcheck.** Build a `Vec<SnarkClaim>` and run
   `prove_initial_claims(&witness, &claims, &mut hash_wrapper)`. All claims
   are batched with transcript randomness into one sumcheck; after
   `nu = log2(height * width)` rounds, everything left to prove is a set of
   evaluation claims at one random point: the two standard witness
   evaluations plus one opening per distinct segment, returned as
   `ChainInputs`.
3. **The chain.** `prover_round` / `verifier_round` prove those openings
   against the commitment and certify an aggregate l2 bound on the committed
   vector. The verifier side mirrors the flow: rebuild the same claims (same
   transcript state), `verify_initial_claims`, `verifier_round`.

A minimal end-to-end program is `execute_snark` in `parties/executor.rs`
(`ROKOKO_MODE=snark cargo run ...`).

## The statement

Each `SnarkClaim` asserts

```text
sum_{z in {0,1}^nu} sum_t coeff_t * prod_{f in t} factor_f(z)  =  value
```

over the full cube of `nu` variables. A term is a coefficient (a ring
element) times a product of factors; a degree-two term multiplies two
committed values at the same cube position. There is no other multiplication:
the claim language is coordinatewise, and relations that multiply values at
*different* positions must first align them (segments, shifted segments,
operand lists).

## Shortness is the caller's

The witness commits **as given**: the front end never decomposes it
(nonlinear claims do not commute with the chain's norm decomposition), and
the only smallness statement the system proves is the aggregate l2 norm of
the whole committed vector. Consequences for relation design:

- Provide the witness already short. For full-range values, commit balanced
  digits (`common::decomposition::decompose(values, base_log, radix)`) and
  state the value as the recomposition `sum_j base^j * digit_j` inside your
  claims; `weighted_layer` turns the powers of the base into tensor layers.
- No per-coordinate range is proven. The certified bound is one l2 number for
  the entire vector; if your relation's soundness needs per-value ranges,
  encode them (bit decompositions tested through the conjugate constant-term
  pattern, see below) or account for the aggregate bound in the security
  analysis.

## Factors

Committed factors (`ClaimFactor`):

| variant | reads | opening cost |
|---|---|---|
| `Witness` | the full vector | none beyond the standard `z_0` |
| `ConjWitness` | the conjugated vector (`X -> X^-1` per element) | none beyond the standard `z_1` |
| `WitnessSegment(prefix)` | the sub-vector under a binary prefix, as an oracle over the low variables | one opening per distinct prefix |
| `WitnessSegmentShifted(prefix, k)` | the same, data variables sitting above `k` low variables in which it is constant | one opening (the point's matching slice) |
| `ConjWitnessSegment(prefix)` | conjugate of a segment | one opening |
| `WitnessSegmentsScaled(parts, k)` | the virtual combination `sum_i scale_i * segment_i` over a shared layout | none: derived from the parts' openings |

Public factors (`PublicFactor`), chosen by weight structure:

| variant | weights | prover cost | verifier cost |
|---|---|---|---|
| `FieldTensor { layers, .. }` | product-structured (eq-tensors, geometric scales) | one dense expansion per `Arc` | `O(layers)` |
| `DensePrefixed(prefix, suffix, data)` | small arbitrary tables | the table itself | linear in the table |
| `LazyPrefixed { data, eval, .. }` | large tables with a closed form | `data` (set `None` on the verifier) | the `eval` closure at the final point |
| `Structured` / `Selector` | raw tensor rows / `eq(prefix, .)` | none | `O(nu)` |

## Conventions

- **Full-cube normalization.** Every claim sums over all `nu` variables. A
  factor constant in `k` of them contributes to both halves of each unused
  variable's sum, so the term picks up `2^k`; cancel it in the coefficient
  with `inv_pow2_q(k)` (the segment conventions assume the coefficient
  carries `inv_pow2_q(prefix_len)`).
- **Tensor layers are MSB-first.** Layer `j` weighs index bit `j` counted
  from the top of the oracle's variable block; entry `i` weighs
  `prod_j ((1-a_j)(1-i_j) + a_j*i_j)`. Per-index scales fold into layers:
  `(1, w)` is `weighted_layer(w)`, an eq layer plus a scalar for the
  coefficient.
- **Coefficients and values are ring elements.** A fixed public element
  multiplying a whole term (a conjugate element, a packed constant) rides in
  the coefficient at no oracle cost; claim equality is checked as ring
  elements, so per-coefficient data batches through the value.
- **Witness-dependent claim values are yours.** With
  `ct(u * conj(v)) = sum_c u_c v_c`, integer statements about coefficients
  (binariness: `sum x_c(x_c - 1) = 0`) become claims whose value the
  verifier cannot compute. The front end takes `value` as given on both
  sides: ship such values in your own envelope, absorb them into the
  transcript before proving and verifying, and perform the structural check
  (a zero constant coefficient, say) on the verifier side yourself. The
  no-wrap precondition for integer readings comes from the certified l2
  bound.
- **Transcript order.** Absorb the commitment before building claims (their
  batching randomness samples from the transcript). The verifier must
  rebuild claims in exactly the prover's order.

## Building a relation

Lay out the witness, keeping each logical region a power-of-two-aligned
segment:

```rust
let start = cursor.next_multiple_of(size);            // size a power of two
data[start..start + size].clone_from_slice(region);
let seg = Prefix { prefix: start / size, length: total_vars - size.ilog2() as usize };
```

A linear claim contracting a segment against an eq-tensor of challenges,
value zero:

```rust
let layers = sample_qe_layers(&mut hw, seg_vars);     // MSB-first
SnarkClaim {
    terms: vec![ClaimTerm::scaled(
        RingElement::constant(inv_pow2_q(seg.length), Representation::IncompleteNTT),
        vec![
            ClaimFactor::Public(PublicFactor::FieldTensor {
                prefix_len: seg.length, suffix_len: 0, layers: Arc::new(layers),
            }),
            ClaimFactor::WitnessSegment(seg.clone()),
        ],
    )],
    value: RingElement::zero(Representation::IncompleteNTT),
}
```

A degree-two term multiplies two committed factors under a public weight; a
recomposition (digits to value) is the same linear shape with
`weighted_layer(base)` layers on the digit index and the scalar scales folded
into the coefficient. Values the verifier must compute (public boundary data)
go into `value` with the same tensor weights, e.g.
`embed_qe(&tensor_at(&layers, i)) * public_i` summed over the public rows.

## Helpers

| helper | gives |
|---|---|
| `sample_qe_layers(hw, n)` | `n` transcript challenges for tensor layers |
| `tensor_at(layers, i)` | entry `i` of the eq-tensor |
| `eq_layers_qe(a, z)` | `eq(a, z)` over layer/point slices |
| `weighted_layer(w)` | the pair `(1, w)` as a layer plus its coefficient scale |
| `embed_qe(v)` | the field scalar as a ring element |
| `qe_mul`, `qe_one_minus` | field arithmetic on challenges |
| `inv_pow2_q(k)` | the full-cube normalization constant |
| `expand_field_tensor(layers)` | the dense tensor, for prover-side tables |

## Parameters

`p_root_aux(nof_openings).generate_config()` is the chain as compiled with a
chosen opening budget (`P_SNARK` is `p_root_aux(2)`); the opening count is
padded to a power of two, one slot per distinct segment plus the two standard
evaluations. For witness sizes between the compiled sets, keep the set's
height and drop column bits (`params::witness_cols_for_target`).
