# The SNARK front end (EXPERIMENTAL)

You commit to one vector of ring elements, then prove statements of the form

```text
sum over the whole vector of   weight(i) * witness(i) * ...   =   value
```

Everything happens in `rokoko::protocol::snark`. A runnable end-to-end program
is `examples/claims.rs` (`cargo run --release --example claims`); the full
pipeline with the commitment chain is `execute_snark` in `parties/executor.rs`.

## The five-minute version

```rust
use rokoko::protocol::snark::*;

let mut layout = WitnessBuilder::new(256, 8);
let balances_at = layout.push(&balances);
let witness = layout.finish();

let revenue = (table(prices.clone()).on(balances_at) * witness_in(balances_at)).sum(&witness);

let claims = vec![Claim::sums_to(
    table(prices).on(balances_at) * witness_in(balances_at),
    revenue,
)];

let mut transcript = Transcript::new();
let (proof, openings) = prove_claims(&witness, &claims, &mut transcript);
```

The verifier rebuilds the same `claims` (same transcript state) and calls
`verify_claims(WitnessShape::new(height, width), &claims, &proof, &mut
transcript)` - a `(height, width)` tuple or `&witness` also convert into the
shape. Both calls return the `ChainInputs` that the commitment chain then
proves (see [Running the chain](#running-the-chain)).

## The mental model

The witness is one long vector `w` of ring elements; its length is a power of
two, so an index is a string of bits. A claim multiplies a few *factors*
together at every index, sums over all indices, and asserts the result:

- **witness factors** read committed entries: `witness_in(region)`, and its
  conjugate for norm-style statements;
- **weight factors** are public: a `table` of values, an `eq` point, a
  `powers` progression.

There is no other multiplication: factors combine *pointwise, at the same
index*. Relations between entries at different positions are a layout
problem - put the data where the claim needs it (commit a rearranged copy and
tie it back with a copy claim; see [Recipes](#recipes)).

## Laying out the witness

`WitnessBuilder` places each `push` at an offset aligned to its own length
(lengths must be powers of two) and returns a `Region`; unused space stays
zero:

```rust
let mut layout = WitnessBuilder::new(height, width);   // height*width entries
let balances_at = layout.push(&balances);
let digits_at = layout.push(&digits);
let witness = layout.finish();
```

A `Region` is how claims refer to that block. Its entries are addressed by
`region.vars()` - the block's index bits, most-significant first. Splitting
those bits views the region as a table:

```rust
let (row, column) = region.vars().split_at(row_bits);
```

`Region::whole(n)` is the entire witness; `Region::new(start, len, n)` names
an existing aligned block without the builder.

## Reading the witness

| expression | reads |
|---|---|
| `witness_in(region)` | the region's entries; the term sums over that region only |
| `witness_in(region).conjugate()` | the same entries under `X -> X^{-1}` |
| `witness()` | shorthand for `witness_in(Region::whole(n))` |

Restricting to a region is free: it lowers to an internal selector and adds
no proof material and no opening. Two factors over the same region share one
selector, so `witness_in(r) * witness_in(r)` costs the same selectors as one.

Conjugation is the tool for coefficient-level statements: the constant
coefficient of `sum_i w_i * conj(w_i)` is the squared l2 norm of all
coefficients, and `ct(u * conj(v)) = sum_c u_c * v_c` in general.

## Public weights

| constructor | entry `i` | verifier cost |
|---|---|---|
| `table(values)` | `values[i]` | linear in the table |
| `eq(point)` | `eq(point, bits(i))` | `O(point.len())` |
| `powers(ratio, k)` | `ratio^i` (over `2^k` entries) | `O(k)` |

Pass weight entries as whatever you have - `Vec<u64>`, transcript challenges
(`Vec<QuadraticExtension>`), or `Vec<RingElement>`. Scalar and challenge
weights automatically take a fast verifier path; ring-element weights are for
genuinely ring-valued tables.

A weight spans the whole witness by default. `.on(...)` places it on a region
or on a block of variables, where it varies; everywhere else it is constant:

```rust
table(prices).on(balances_at)                       // one weight per entry
eq(alpha).on(row) * table(k).on(column)             // rows weighted by alpha, columns by k
```

Weights on non-overlapping variable blocks stack freely in one product - that
`eq * table` pair costs degree one per variable, not two.

`challenge_point(&mut transcript, num_vars)` draws an `eq` point from the
transcript; with such a point, `eq` is the standard random-linear-combination
weight ("check a random mixture instead of every entry"). Points read
MSB-first: coordinate `j` weighs index bit `j` counted from the top of the
block.

## Stating claims

```rust
Claim::sums_to(expr, value)     // sum over the cube of expr = value
Claim::sums_to_zero(expr)       // ... = 0, the shape of every equality
```

Expressions combine with `*`, `+`, `-`, unary `-`, and `expr.scale(&c)` (or
`7 * expr` for small scalars). All claims passed to one `prove_claims` call
are batched into a single sumcheck with transcript randomness; more claims
mean more prover passes but only one shared round trip.

**The degree rule.** Each *variable* may be touched by at most three factors
in any term. A product adds its factors' per-variable degrees, a sum takes
the max, and a region restriction contributes its selector to the region's
prefix variables. Factors on disjoint variable blocks therefore multiply
freely; three full-witness factors (`w * w * w`) are the ceiling.
`prove_claims` checks this up front and panics with the offending shape.

**Computing values.** `expr.sum(&witness)` returns the true sum - the value a
correct prover ships. It is a plain prover-side pass over the witness: use it
to build claims and tests, not in the verifier (which must never need the
witness). For values the verifier derives from public data, see the recipes.

## Who computes the value?

Three cases, from strongest to weakest verifier position:

1. **The verifier computes it from public data.** Example: "this region holds
   the digits of these public totals" - the value is `eq_weighted_sum(&point,
   &totals)`, computed by both sides. Nothing ships.
2. **The value is zero by construction.** Copy claims, equality claims,
   anything of the form `a - b`. Nothing ships.
3. **The prover ships it.** Norm accumulators, binariness sums - anything
   depending on the secret witness. The front end proves `sum = value` for
   the value you supply and absorbs every claim value into the transcript
   before drawing batching randomness. Two duties stay yours: deliver the
   value to the verifier, and check its *structure* there (e.g. a zero
   constant coefficient for a binariness claim) - the front end checks
   ring-element equality, not meaning. Reading a constant coefficient as a
   true integer also needs no wraparound mod q, which the chain's certified
   l2 bound guarantees.

## Fiat-Shamir, in one paragraph

Prover and verifier drive one shared transcript and must stay in lockstep:
absorb the commitment first, build the claims in the same order, draw
`challenge_point`s at the same transcript states, and absorb anything the
prover ships (claim values are absorbed for you; other shipped data - a
table, a coefficient - you must `update` yourself before proving). The
verifier panics on any mismatch.

## Recipes

**A weighted sum over a region** (dot product with public weights):

```rust
let value = (table(prices.clone()).on(orders) * witness_in(orders)).sum(&witness);
Claim::sums_to(table(prices).on(orders) * witness_in(orders), value)
```

**Tie two regions together** (copy claim; the alignment tool):

```rust
let point = challenge_point(&mut transcript, original.vars().len());
Claim::sums_to_zero(eq(&point).on(original) * (witness_in(original) - witness_in(mirror)))
```

**A hash-style layer relation** (the ajtai-merkle shape: nodes x slots, an
`eq` over the node index, a matrix-derived table over the slot index):

```rust
let (node, slot) = layer.vars().split_at(node_bits);
let alpha = challenge_point(&mut transcript, node.len());
Claim::sums_to_zero(
    eq(&alpha).on(node) * table(folded_matrix_row).on(slot) * witness_in(layer)
        - /* the layer's outputs, stated the same way over the next region */
)
```

**Full-range values via digits.** The witness must stay short, so a
full-range value enters as its balanced digits
(`common::decomposition::decompose(values, base_log, radix)`) and the value
is only ever *stated*, never committed:

```rust
let (value_index, digit_index) = digits_at.vars().split_at(value_bits);
let point = challenge_point(&mut transcript, value_index.len());
Claim::sums_to(
    eq(&point).on(value_index)
        * powers(1 << base_log, digit_index.len()).on(digit_index)
        * witness_in(digits_at),
    eq_weighted_sum(&point, &public_totals),
)
```

**The witness energy** (squared l2 norm in the constant coefficient; the
value ships):

```rust
let energy = (witness() * witness().conjugate()).sum(&witness);
Claim::sums_to(witness() * witness().conjugate(), energy)
```

**Binariness of a region's coefficients**: `sum x(x-1) = 0` per coefficient,
stated as `witness_in(r) * witness_in(r).conjugate() - table(ones_conj).on(r)
* witness_in(r)` summing to a shipped value whose constant coefficient the
verifier checks is zero.

## Running the chain

`prove_claims` reduces every claim to openings of the committed vector at one
random point (`ChainInputs`): the witness evaluation always, plus a
conjugated one only when some claim conjugates - a conjugate-free statement
emits a single opening and skips the conjugate fold entirely. The chain
proves those openings against the commitment and certifies an aggregate l2
bound:

```rust
let (chain_proof, _) = prover_round(&crs, &config, &commitment_with_aux, &witness,
    &inputs.evaluation_points_inner, &inputs.evaluation_points_outer,
    &mut sumcheck_context, false, Some(transcript));
```

Pass the *same* transcript that ran `prove_claims`, so it continues unbroken;
mirror with `verify_claims` + `verifier_round`. The commitment must be
absorbed into the transcript before any claim is built.

The compiled chain must match the opening count; the `_2` sets fit the
two-opening (conjugate-using) statements:

- Exact norm: `P_EN_2_SMALL` / `P_EN_2_MEDIUM` / `P_EN_2_LARGE`; witness
  coefficients `<= 2^7`. Conjugate-free statements pair with the
  single-opening sets `P_EN_SMALL` / `P_EN_MEDIUM` / `P_EN_LARGE` instead.
- Non-exact norm: `P_2_SMALL` / `P_2_MEDIUM` / `P_2_LARGE`; coefficients
  `<= 2^15`.

The compiled witness size comes from the `p-26` / `p-28` (default) / `p-30`
features. For sizes in between, keep the compiled `height`/`width` and shrink
`used_cols` via `params::witness_cols_for_target(...)`; changing `width`
itself breaks the compiled chain.

The commitment is binding only for short vectors, and the front end never
decomposes anything for you: whatever you `push` must already be short.
