# Proof serialisation

`protocol::wire::{to_bytes, from_bytes}` turn the round-proof chain into a byte string and back.
The verifier in `parties::executor` runs off the deserialised proof, so the roundtrip is on the
honest path; its two timings are printed on their own line and are not part of the prover or
verifier totals.

## The code

Every number that travels is a residue modulo `q`, and the proof mixes two populations of them.
The commitments, the openings and the sumcheck polynomials are uniform. The folded witness, the
two norm claims and the constant-term projection image are short — a fold of challenge-weighted
columns, standard deviation a few hundred against a 50-bit `q` — and a code that spends 50 bits
on them wastes four fifths of the proof.

One mechanism covers both. A coefficient `c` is split into its octave `b = bitlen(|c|)`, coded
against a measured distribution, and a `b`-bit field holding the mantissa `|c| - 2^(b-1)` and the
sign, written raw. Uniform data lands on the octave law `Pr[b = k] = 2^(k-50)`, whose entropy is
2 bits, and pays `2 + 48 = 50` bits: the mechanism costs it nothing. Short data lands on the
octaves its own scale occupies. Zero is octave 0 and carries no field, which is what makes the
all-zero rows of `opening_rhs` free.

The distributions are measured on the proof being sent, never derived from the norm bounds: one
octave histogram per field role — the folded witness, the norm claims, the commitments, ... —
taken over that role's coefficients across every round, quantised to `SCALE` by largest remainder
with every occupied octave floored at one, and written in the header as Elias gamma codes. A
proof whose folded witness came out wider than the schedule predicted still encodes, at its own
measured width. A role whose histogram does not pay for itself — the coded cost plus the table
must beat `ceil(log2 q)` per coefficient by 2% — is written raw instead, and its table is one
bit.

The octave code is a 32-bit rANS with 16-bit renormalisation over a `SCALE`-entry table,
interleaved across `LANES` streams: coefficient `i` belongs to lane `i mod LANES`, so one
dependency chain of `n` steps becomes `LANES` chains of `n / LANES`. The encoder walks the
coefficients backwards and pushes its renormalisation words onto one vector, whose reversal is
the order a forward decoder pulls them in, so the wire carries the `LANES` final states and then
the words, with no per-lane lengths. The mantissa fields ride a separate bit stream that both
sides walk forwards, so neither code has to interleave with the other.

A ring element is coded in the representation it already holds, except where the coefficient
representation is genuinely shorter — the folded witness and the norm claims, held in NTT form
where their coefficients look uniform. The inverse NTT is not free, so it is taken only when it
saves more than a bit per coefficient, decided once per region, and one bit on the wire says
whether it was taken. The reverse direction never wins: measured over a p-26 proof, no element
was shorter in NTT form than in coefficients, and 310 were shorter in coefficients.

Shapes travel with the data: lengths and matrix dimensions as varints, the round kind and the
optional fields as flags, representations as two bits per region. Re-deriving them from `Config`
would put a second copy of the recursive commitment layout in the codec, to drift against the
first.

## What it costs against the reported proof size

`SizeableProof::size_in_bits` charges `min(bitlen(v), bitlen(centre(v)) + 1) ≈ log2|c| + 0.5` per
coefficient. That is the ideal code length for a scale-free source, `p(c) ∝ 1/|c|`. Real
coefficients are Gaussian or uniform, and the octave a coefficient lands in has to be paid for —
about `H(octave) - 0.5 ≈ 1.5` bits of it, for every nonzero coefficient, whatever its scale.

Measured on a p-26 proof (45626 coefficients, 100.47 KB reported):

| field | kind | coeffs | reported | entropy floor |
|---|---|---|---|---|
| folded witness | short, σ ≈ 514 | 32768 | 36.12 | 43.98 |
| next commitment | uniform | 2816 | 16.65 | 17.17 |
| rc opening | uniform | 2176 | 12.85 | 13.26 |
| batched projection | uniform | 1024 | 6.05 | 6.24 |
| constant-term claims | uniform | 1024 | 6.02 | 6.20 |
| claim, conjugate | uniform | 1536 | 9.09 | 9.37 |
| opening rhs | uniform, 2 rows zero | 1024 | 4.54 | 4.68 |
| sumcheck polys | uniform | 542 | 3.21 | 3.22 |
| norm claim | short, σ ≈ 2²⁸ | 768 | 2.06 | 2.26 |
| inner norm claim | short, σ ≈ 2¹⁹ | 768 | 1.62 | 1.81 |
| projection image ct | short | 1024 | 1.50 | 1.75 |
| rc coarse | uniform | 128 | 0.76 | 0.78 |
| **total** | | **45626** | **100.47** | **110.74** |

The floor column is the exact zero-order entropy of the coefficients actually sent, modelled per
element with no header cost, so it is below anything implementable. The folded witness is 77% of
the gap because it is 72% of the coefficients; it carries no structure to exploit (per-index
modelling costs 11.08 bits per coefficient and delta coding 11.52, against 11.00 for the plain
octave code).

So the wire form is about 11.7 kB above the reported size, and roughly 10 kB of that is not
recoverable by any encoder.
