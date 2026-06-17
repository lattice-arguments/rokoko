# The SNARK front end (EXPERIMENTAL)

The argument's native object is one committed vector; its native statement is
a batch of sumcheck claims about that vector. The flow is three steps:

1. Commit. Arrange the witness as a matrix `W` of ring elements
   (`VerticallyAlignedMatrix`, `height x width` a power of two) and commit it
   (`commiter::commit`). The Ajtai commitment is binding only for short
   vectors.
2. Entry sumcheck. Build a `Vec<SnarkClaim>` and run
   `prove_initial_claims(&witness, &claims, &mut hash_wrapper)`. All claims
   are batched with transcript randomness into one sumcheck; after
   `nu = log2(height * width)` rounds, everything left to prove is two
   evaluation claims at one random point, the witness evaluation `z_0` and
   its conjugate `z_1`, returned as `ChainInputs`.
3. The chain. `prover_round` / `verifier_round` prove those openings
   against the commitment and certify an aggregate l2 bound on the committed
   vector; pass them the same `HashWrapper` that ran the entry round (the
   `Some(hash_wrapper)` argument), so the transcript continues unbroken. The
   verifier side mirrors the flow: rebuild the same claims (same transcript
   state), `verify_initial_claims`, `verifier_round`.

   The openings hand over in a fixed order and there are always exactly
   two: slot 0 the witness evaluation, slot 1 the conjugated one - the
   compiled chain `P_EN_TWO_EVALS` (two openings) fits every relation.

A minimal end-to-end program is `execute_snark` in `parties/executor.rs`
(`cargo run --features snark ...`).

## The statement

Each `SnarkClaim` is one expression `expr` and a `value`, and asserts

```text
sum_{z in {0,1}^nu} expr(z)  =  value
```

over the full cube of `nu` variables. `expr` is a `ClaimExpr`: a tree of public
and private (witness) leaves, closed under product, sum, difference and
scaling. A degree-two product multiplies two committed values at the same cube
position. There is no other multiplication: the claim language is
coordinatewise, and relations that multiply values at *different* positions
must first align them - commit copies of the rearranged data and tie them back
with copy claims.

## Composing claims

A `ClaimExpr` is built from leaf constructors and combined with the `+ - *`
operators (or the matching `add`/`sub`/`mul` methods):

- leaves: `ClaimExpr::witness()`, `conj_witness()`, `segment(prefix)`,
  `conj_segment(prefix)`, `public(factor)`, `constant(r)`;
- `a * b` is a product, `a + b` a sum, `a - b` a difference; `a.scale(&c)`
  multiplies by a ring scalar and `-a` flips the sign.

These lower 1:1 to the sumcheck combinators - product to `ProductSumcheck`, sum
to `SumSumcheck`, difference to `DiffSumcheck`. The degree-three cap is
per-variable: a product adds the per-variable degrees of its factors, a sum or
difference takes the max, so factors on disjoint blocks multiply freely. The
common use of a difference is an equality constraint - with `value` zero,
`a - b` states `a == b` as one zero-claim.

## Shortness is the caller's

The witness commits as given: the front end never decomposes it. Consequences for relation design:

- Provide the witness already short. For full-range values, commit balanced
  digits (`common::decomposition::decompose(values, base_log, radix)`) and
  state the value as the recomposition `sum_j base^j * digit_j` inside your
  claims; `weighted_layer` turns the powers of the base into tensor layers.
- No per-coordinate range is proven. The certified bound is one L2 number for
  the entire vector in the exact-norm mode.

## Factors

Committed factors (`ClaimFactor`):

| variant | reads | opening cost |
|---|---|---|
| `Witness` | the full vector | the standard `z_0` |
| `ConjWitness` | the conjugated vector (`X -> X^-1` per element) | the standard `z_1` |
| `WitnessSegment(prefix)` | the sub-vector under a binary prefix; the term sums over its block | none (lowers to `eq(prefix, .) x Witness`) |
| `ConjWitnessSegment(prefix)` | conjugate of a segment | none (lowers against `ConjWitness`) |

A public factor is a `PublicFactor`: a `weights` shape and representation,
placed on the cube with `.over_middle(prefix_len, suffix_len)` (constant on the
top/bottom variables, varying over the middle; the full cube is the default).

| constructor | weights | verifier cost |
|---|---|---|
| `tensor_ring(layers)` / `tensor_field(layers)` | product eq-tensor, MSB-first layers | `O(layers)` |
| `dense_ring(table)` / `dense_field(table)` | arbitrary table over the middle | linear in the table |
| `selector(bits, length)` | `eq(bits, .)` on the leading `length` variables | `O(length)`, zero prover cost |

Both `tensor_*` and `dense_*` take any placement; `_field` weights evaluate
faster on the verifier than `_ring`. `selector` is anchored to the leading
variables.

## Conventions

- Segment terms are localised. A term holding a `WitnessSegment(prefix)` sums
  over that segment's block exactly once, lowering to `eq(prefix, .)` times the
  full-vector oracle: it adds no opening (the final check still reduces to
  `z_0`/`z_1`), it contributes the selector to the leading variables and the
  oracle to all of them, and two factors over the same block share one selector
  (`eq` is 0/1 on the cube, so `eq^2 = eq`).
- Tensor layers are MSB-first. Layer `j` weighs index bit `j` counted
  from the top of the oracle's variable block; entry `i` weighs
  `prod_j ((1-a_j)(1-i_j) + a_j*i_j)`. Per-index scales fold into layers:
  `(1, w)` is `weighted_layer(w)`, an eq layer plus a scalar for the
  coefficient.
- Coefficients and values are ring elements. A fixed public element
  multiplying a whole term (a conjugate element, a packed constant) rides in
  the coefficient at no oracle cost; claim equality is checked as ring
  elements, so per-coefficient data batches through the value.
- Witness-dependent Z_q values are per-user. Using `ct(u * conj(v)) = sum_c u_c v_c`,
  a claim can state an integer fact about coefficients (binariness:
  `sum x_c(x_c - 1) = 0`); its `value` then depends on the secret witness, so
  the verifier cannot recompute it. The front end only proves
  `sum_z expr = value` for the `value` you supply, and hashes each value into
  the transcript before drawing the batching randomness. Two things stay
  yours: ship the value to the verifier, and check its shape (say, a zero
  constant term) - the front end checks ring-element equality, not structure.
  More generally, anything the prover sends instead of the verifier deriving
  it - a coefficient, a public table - must be hashed into the transcript
  before the prove and verify calls, or Fiat-Shamir is unsound. Reading a
  constant term as a true integer also needs no wraparound mod q, which the
  certified l2 bound guarantees.
- Build claims in a fixed order, after absorbing the commitment. The claim
  batching randomness is drawn from the transcript, so the commitment must go
  in first, and the verifier must rebuild the claims in exactly the prover's
  order.

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
    expr: ClaimExpr::public(PublicFactor::tensor_field(layers).over_middle(seg.length, 0))
        * ClaimExpr::segment(seg.clone()),
    value: RingElement::zero(Representation::IncompleteNTT),
}
```

A degree-two product multiplies two committed factors under a public weight. The
limit is per-variable, not a raw factor count: a round polynomial's degree is
the number of factors depending on that variable, and the round polynomials
carry degree at most three. Factors placed on disjoint variable blocks (a node
eq-tensor over one block, a table over another) therefore stack freely - a
localized `tensor * dense * segment` lowers to four factors yet every variable
sees at most two. A recomposition (digits to value) is the same linear shape
with one weighted layer per digit-index bit, `weighted_layer(base.pow(1 << l))`
for bit `l` MSB-first, the scalar scales folded into a constant. Values the verifier must compute (public boundary data)
go into `value` with the same tensor weights, accumulating
`&embed_qe(&tensor_at(&layers, i)) * &public_i` over the public rows.

## Helpers

| helper | gives |
|---|---|
| `sample_qe_layers(hw, n)` | `n` transcript challenges for tensor layers |
| `tensor_at(layers, i)` | entry `i` of the eq-tensor |
| `eq_layers_qe(a, z)` | `eq(a, z)` over layer/point slices |
| `weighted_layer(w)` | the pair `(1, w)` as a layer plus its coefficient scale |
| `embed_qe(v)` | the field scalar as a ring element |
| `qe_mul`, `qe_one_minus` | field arithmetic on challenges |
| `expand_field_tensor(layers)` | the dense tensor, for prover-side tables |

## Parameters

The witness size is fixed at compile time by the `p-26` / `p-28` / `p-30`
features: `compiled_size()` maps them to `SizeConfig::Small` (`2^26`) /
`Medium` (`2^28`, the default) / `Large` (`2^30`), and that `size` feeds every
builder below.

The chain always proves exactly two openings (`z_0`, `z_1`), so the front end
needs a parameter set compiled with `nof_openings = 2` - a `_2` set. Two norm
flavors, picked by the witness's smallness budget:

- Exact norm: `P_EN_2_SMALL` / `P_EN_2_MEDIUM` / `P_EN_2_LARGE` (one per mode,
  `p_exact_norm_root_aux(size, 2)`). Witness coefficients must be
  `<= 2^7`.
- Non-exact norm: `P_2_SMALL` / `P_2_MEDIUM` / `P_2_LARGE` (one per mode,
  `p_root_aux(size, 2)`). Witness coefficients must be `<= 2^15`.

For witness sizes between the compiled sets, keep the compiled `height` and
`width` and set `used_cols` of the witness matrix to
`params::witness_cols_for_target(...)`, leaving the remaining columns zero;
shrinking `width` itself changes the variable count and breaks the compiled
chain.
