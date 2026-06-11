# The SNARK front end

The argument commits a single vector and proves a batch of sumcheck claims
about it. A proof has three stages.

1. Commitment. Arrange the witness as a matrix `W` of ring elements
   (`VerticallyAlignedMatrix`, `height` and `width` powers of two) and commit
   it with `commiter::commit`. The Ajtai commitment is binding only for short
   vectors; the shortness contract below says what this requires of the
   witness.
2. Entry round. Build a `Vec<SnarkClaim>` and run
   `prove_initial_claims(&witness, &claims, &mut hash_wrapper)`. All claims
   are batched with transcript randomness into one sumcheck; after
   `nu = log2(height * width)` rounds, everything left to prove is two
   evaluation claims at one random point: the standard witness evaluations
   `z_0` and `z_1`, returned as `ChainInputs`.
3. Chain. The chain proves those openings against the commitment and
   certifies an aggregate l2 bound on the committed vector; `prover_round`
   and `verifier_round` run its two sides. Pass them the same `HashWrapper`
   that ran the entry round (the `Some(hash_wrapper)` argument), so the
   transcript continues unbroken.

The verifier mirrors the whole flow: rebuild the same claims at the same
transcript state, then `verify_initial_claims` and `verifier_round`.

On both sides, the entry round passes the openings to the chain in a fixed
order: slot 0 holds the witness evaluation `z_0` and slot 1 the conjugate
evaluation `conj(z_1)`, checked against the conjugated evaluation rows.

A minimal end-to-end program is `execute_snark` in `protocol/parties/executor.rs`
(`ROKOKO_MODE=snark cargo run ...`).

## The statement

Each `SnarkClaim` asserts

```text
sum_{z in {0,1}^nu} sum_t coeff_t * prod_{f in t} factor_f(z)  =  value
```

over the full cube of `nu` variables. A term is a coefficient (a ring
element) times a product of factors; a degree-two term multiplies two
committed values at the same cube position. There is no other multiplication:
the claim language is coordinatewise, and a relation that multiplies values
at different positions must first align them, by committing copies of the
rearranged data and tying each copy to its source entries with linear copy
claims.

## The shortness contract

The witness is committed as given: the front end never decomposes it
(nonlinear claims do not commute with the chain's norm decomposition), and
the only smallness statement the system proves is an aggregate l2 bound on
the whole committed vector. Two consequences follow for relation design.

The witness must be short as committed. For full-range values, commit
balanced digits (`common::decomposition::decompose(values, base_log, radix)`)
and state each value inside your claims as the recomposition
`sum_j base^j * digit_j`; `weighted_layer` turns the powers of the base into
tensor layers.

No per-coordinate range is proved. The certified bound is a single bound on
the l2 norm of the entire vector; if a relation's soundness needs per-value
ranges, encode them in the claims (bit decompositions tested through the
conjugate constant-term pattern described under Conventions) or account for
the aggregate bound in the security analysis.

## Factors

Committed factors (`ClaimFactor`):

| variant | reads | opening cost |
|---|---|---|
| `Witness` | the full vector | the standard `z_0` |
| `ConjWitness` | the conjugated vector (`X -> X^-1` per element) | the standard `z_1` |
| `WitnessSegment(prefix)` | the sub-vector under a binary prefix; the term sums over its block | none (lowers to `eq(prefix, .) x Witness`) |
| `ConjWitnessSegment(prefix)` | conjugate of a segment | none (lowers to `eq(prefix, .) x ConjWitness`) |

Public factors (`PublicFactor`), chosen by weight structure:

| variant | weights | prover cost | verifier cost |
|---|---|---|---|
| `FieldTensor { layers, .. }` | product-structured (eq-tensors, geometric scales) | one dense expansion per `Arc` | `O(layers)` |
| `Dense(data)` | arbitrary full-cube tables (tests, small relations) | the table itself | linear in the table |
| `DensePrefixed(prefix, suffix, data)` | small arbitrary tables over a segment | the table itself | linear in the table |
| `LazyPrefixed { data, eval, .. }` | large tables with a closed form | `data` (set `None` on the verifier) | the `eval` closure at the final point |
| `Structured` | raw tensor rows over all variables | one dense expansion per use (not shared) | `O(nu)` |
| `Selector` | `eq(prefix, .)` | none (lazy gadget) | `O(nu)` |

The closure of `LazyPrefixed` has type
`Arc<dyn Fn(&[RingElement], &[QuadraticExtension]) -> RingElement + Send + Sync>`.
It receives only the middle (data-variable) slice of the final point, in both
ring and field form, and the slice arrives LS-first (least-significant
variable first, the round order), the reverse of the MSB-first layer
convention.

## Conventions

A term holding a `WitnessSegment(prefix)` sums over that segment's block
exactly once. The segment lowers internally to `eq(prefix, .)` times the
full-vector oracle, so the claim `value` is the plain block sum and no
power-of-two bookkeeping arises anywhere. The lowering costs one factor of
term degree (a segment counts as two of the three factors a term may hold)
and no opening, since the final evaluation reduces to the standard `z_0` and
`z_1`. Two factors restricted to the same block share one selector.

Tensor layers are MSB-first: layer `j` weights index bit `j` counted from the
top of the oracle's variable block, and entry `i` carries the weight
`prod_j ((1-a_j)(1-i_j) + a_j*i_j)`. A per-index scale folds into a layer:
`weighted_layer(w)` returns the pair `(1, w)` as an eq layer together with
the scalar that moves to the coefficient.

Coefficients and values are full ring elements. A fixed public element that
multiplies a whole term (a conjugate element, a packed constant) folds into
the coefficient at no oracle cost, and claim equality is checked over full
ring elements, so several scalar statements can be packed into distinct
coefficients of a single claim value.

Witness-dependent claim values are the caller's responsibility. Writing
`ct(x)` for the constant coefficient of a ring element, the identity
`ct(u * conj(v)) = sum_c u_c v_c` turns integer statements about coefficients
(binariness: `sum x_c(x_c - 1) = 0`) into claims whose value the verifier
cannot compute. Reading such a ring identity as an integer statement requires
that no coefficient arithmetic wraps modulo q; the certified l2 bound
supplies that precondition. The front end takes `value` as given on both
sides and binds it to the transcript before sampling the batching
randomness. It cannot rebuild the value on the verifier side, so transmit it
to the verifier yourself; nor can it check the value's internal structure (a
zero constant coefficient, say), so perform that check when verifying.

The module binds every other hashable claim component the same way: factor
kinds, prefixes, window dimensions, coefficients, and public tables are all
absorbed before the batching randomness is sampled. The one thing it cannot
see is the content behind a `LazyPrefixed` closure, which is bound only
through its window dimensions; leaving it unbound is safe when the
verifier derives the closure itself, and anything the verifier instead
receives must be absorbed into the transcript by the caller before the prove
and verify calls.

The transcript fixes the order of operations: absorb the commitment before
building claims, since the challenges a claim embeds, like the batching
randomness after them, are sampled from the transcript and must come after
the commitment; and rebuild claims on the verifier side in exactly the
prover's order.

## Building a relation

Lay out the witness, keeping each logical region a power-of-two-aligned
segment:

```rust
let start = cursor.next_multiple_of(size);            // size a power of two
data[start..start + size].clone_from_slice(region);
let seg = Prefix { prefix: start / size, length: total_vars - size.ilog2() as usize };
```

A linear claim contracting a segment against an eq-tensor of challenges,
with value zero:

```rust
let layers = sample_qe_layers(&mut hw, seg_vars);     // MSB-first
SnarkClaim {
    terms: vec![ClaimTerm::scaled(
        RingElement::constant(1, Representation::IncompleteNTT),
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
term holds at most three factors, because the round polynomials have degree
three. A recomposition (digits to value) is the same linear shape, with one
weighted layer per digit-index bit: `weighted_layer(base.pow(1 << l))` for
bit `l`, MSB-first. The scalar scale of each layer folds into the
coefficient. Values the verifier computes itself (public boundary data) go
into `value` with the same tensor weights, accumulating
`&embed_qe(&tensor_at(&layers, i)) * &public_i` over the public rows.

## Helpers

| helper | returns |
|---|---|
| `sample_qe_layers(hw, n)` | `n` transcript challenges for tensor layers |
| `tensor_at(layers, i)` | entry `i` of the eq-tensor |
| `eq_layers_qe(a, z)` | `eq(a, z)` over layer/point slices |
| `weighted_layer(w)` | the pair `(1, w)` as a layer plus its coefficient scale |
| `embed_qe(v)` | the field scalar as a ring element |
| `qe_mul`, `qe_one_minus` | challenge products and one-minus complements |
| `expand_field_tensor(layers)` | the dense tensor, for prover-side tables |

## Parameters

`p_root_aux(nof_openings).generate_config()` is the chain as compiled with a
chosen opening budget; the chain requires a power-of-two opening count, and
the entry round always emits exactly two, so `P_SNARK` (`p_root_aux(2)`)
fits every relation. For witness sizes between the compiled `height x width`
configurations, keep the compiled `height` and `width` and set `used_cols`
of the witness matrix to `params::witness_cols_for_target(...)`, leaving the
remaining columns zero; shrinking `width` itself changes the variable count
and breaks the compiled chain.
