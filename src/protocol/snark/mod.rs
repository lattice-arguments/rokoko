//! SNARK claim language; guide in `docs/snark.md`, demo in `examples/claims.rs`.

mod lowering;

pub use crate::common::hash::HashWrapper as Transcript;
pub use lowering::InitialSumcheckProof as ClaimsProof;
pub use lowering::SnarkClaim as Claim;
pub use lowering::{
    expand_field_tensor, prove_claims, prove_claims_with_conjugate, verify_claims, ChainInputs,
    ClaimExpr, WitnessShape,
};

use lowering::{ClaimFactor, Coeffs, PublicFactor, SnarkClaim, Weights};

use crate::common::config::MOD_Q;
use crate::common::matrix::VerticallyAlignedMatrix;
use crate::common::ring_arithmetic::{QuadraticExtension, Representation, RingElement};
use crate::protocol::commitment::Prefix;
use std::sync::Arc;

fn zero() -> RingElement {
    RingElement::zero(Representation::IncompleteNTT)
}

fn one() -> RingElement {
    RingElement::constant(1, Representation::IncompleteNTT)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    start: usize,
    len: usize,
    witness_len: usize,
}

impl Region {
    pub fn new(start: usize, len: usize, witness_len: usize) -> Region {
        assert!(len.is_power_of_two(), "region length must be a power of two");
        assert!(witness_len.is_power_of_two(), "witness length must be a power of two");
        assert_eq!(start % len, 0, "region start must be aligned to its length");
        assert!(start + len <= witness_len, "region exceeds the witness");
        Region { start, len, witness_len }
    }

    pub fn whole(witness_len: usize) -> Region {
        Region::new(0, witness_len, witness_len)
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn range(&self) -> std::ops::Range<usize> {
        self.start..self.start + self.len
    }

    /// The `log2(len)` index variables addressing the region's entries.
    pub fn vars(&self) -> Vars {
        let total = self.witness_len.ilog2() as usize;
        let len = self.len.ilog2() as usize;
        Vars { skip: total - len, len, total }
    }

    pub fn prefix(&self) -> Prefix {
        let total = self.witness_len.ilog2() as usize;
        let len = self.len.ilog2() as usize;
        Prefix { prefix: self.start / self.len, length: total - len }
    }
}

/// Consecutive index variables (witness index bits, most-significant first).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vars {
    skip: usize,
    len: usize,
    total: usize,
}

impl Vars {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn split_at(self, num_vars: usize) -> (Vars, Vars) {
        assert!(num_vars <= self.len, "cannot split off more variables than the block has");
        (
            Vars { skip: self.skip, len: num_vars, total: self.total },
            Vars { skip: self.skip + num_vars, len: self.len - num_vars, total: self.total },
        )
    }

    pub fn leading(self, num_vars: usize) -> Vars {
        self.split_at(num_vars).0
    }

    pub fn trailing(self, num_vars: usize) -> Vars {
        self.split_at(self.len - num_vars).1
    }
}

impl From<Region> for Vars {
    fn from(region: Region) -> Vars {
        region.vars()
    }
}

/// Each `push` self-aligns and returns its [`Region`]; gaps stay zero.
pub struct WitnessBuilder {
    height: usize,
    width: usize,
    data: Vec<RingElement>,
    cursor: usize,
}

impl WitnessBuilder {
    pub fn new(height: usize, width: usize) -> WitnessBuilder {
        let n = height * width;
        assert!(n.is_power_of_two(), "witness size must be a power of two");
        WitnessBuilder {
            height,
            width,
            data: vec![zero(); n],
            cursor: 0,
        }
    }

    pub fn push(&mut self, values: &[RingElement]) -> Region {
        assert!(!values.is_empty() && values.len().is_power_of_two(), "pushed data length must be a nonzero power of two");
        let start = self.cursor.next_multiple_of(values.len());
        assert!(start + values.len() <= self.data.len(), "witness is full");
        self.data[start..start + values.len()].clone_from_slice(values);
        self.cursor = start + values.len();
        Region::new(start, values.len(), self.data.len())
    }

    pub fn finish(self) -> VerticallyAlignedMatrix<RingElement> {
        VerticallyAlignedMatrix {
            height: self.height,
            width: self.width,
            used_cols: self.width,
            data: self.data,
        }
    }
}

/// Weight entries; `u64`/field scalars auto-select the fast subfield verifier path.
#[derive(Clone)]
pub enum Scalars {
    Ring(Arc<Vec<RingElement>>),
    Field(Arc<Vec<QuadraticExtension>>),
}

impl Scalars {
    fn len(&self) -> usize {
        match self {
            Scalars::Ring(v) => v.len(),
            Scalars::Field(v) => v.len(),
        }
    }
}

impl From<Vec<RingElement>> for Scalars {
    fn from(v: Vec<RingElement>) -> Scalars {
        Scalars::Ring(Arc::new(v))
    }
}

impl From<Arc<Vec<RingElement>>> for Scalars {
    fn from(v: Arc<Vec<RingElement>>) -> Scalars {
        Scalars::Ring(v)
    }
}

impl From<&[RingElement]> for Scalars {
    fn from(v: &[RingElement]) -> Scalars {
        Scalars::Ring(Arc::new(v.to_vec()))
    }
}

impl From<&Vec<RingElement>> for Scalars {
    fn from(v: &Vec<RingElement>) -> Scalars {
        Scalars::Ring(Arc::new(v.clone()))
    }
}

impl From<Vec<QuadraticExtension>> for Scalars {
    fn from(v: Vec<QuadraticExtension>) -> Scalars {
        Scalars::Field(Arc::new(v))
    }
}

impl From<Arc<Vec<QuadraticExtension>>> for Scalars {
    fn from(v: Arc<Vec<QuadraticExtension>>) -> Scalars {
        Scalars::Field(v)
    }
}

impl From<&[QuadraticExtension]> for Scalars {
    fn from(v: &[QuadraticExtension]) -> Scalars {
        Scalars::Field(Arc::new(v.to_vec()))
    }
}

impl From<&Vec<QuadraticExtension>> for Scalars {
    fn from(v: &Vec<QuadraticExtension>) -> Scalars {
        Scalars::Field(Arc::new(v.clone()))
    }
}

impl From<Vec<u64>> for Scalars {
    fn from(v: Vec<u64>) -> Scalars {
        Scalars::Field(Arc::new(
            v.into_iter().map(|x| QuadraticExtension { coeffs: [x % MOD_Q, 0] }).collect(),
        ))
    }
}

impl From<&[u64]> for Scalars {
    fn from(v: &[u64]) -> Scalars {
        v.to_vec().into()
    }
}

#[derive(Clone)]
enum WeightKind {
    Eq(Scalars),
    Table(Scalars),
}

/// Public weight factor; spans the whole witness unless placed with [`Weight::on`].
#[derive(Clone)]
pub struct Weight {
    kind: WeightKind,
    placement: Option<Vars>,
    coefficient: Option<RingElement>,
}

/// The weight `eq(point, index)`, one coordinate per index bit (MSB-first);
/// verifier cost `O(point.len())`.
pub fn eq(point: impl Into<Scalars>) -> Weight {
    Weight { kind: WeightKind::Eq(point.into()), placement: None, coefficient: None }
}

/// Arbitrary weight table, entry `i` is `values[i]`; verifier cost linear in it.
pub fn table(values: impl Into<Scalars>) -> Weight {
    Weight { kind: WeightKind::Table(values.into()), placement: None, coefficient: None }
}

/// The weight `ratio^i` over `2^num_vars` entries (recomposes base-`ratio`
/// digits); verifier cost `O(num_vars)`.
pub fn powers(ratio: u64, num_vars: usize) -> Weight {
    use crate::common::arithmetic::pow_mod;
    let mut layers = Vec::with_capacity(num_vars);
    let mut scale: u128 = 1;
    for t in (0..num_vars).rev() {
        let (layer, s) = lowering::weighted_layer(pow_mod(ratio, 1u64 << t));
        layers.push(layer);
        scale = scale * s as u128 % MOD_Q as u128;
    }
    Weight {
        kind: WeightKind::Eq(Scalars::Field(Arc::new(layers))),
        placement: None,
        coefficient: Some(RingElement::constant(scale as u64, Representation::IncompleteNTT)),
    }
}

impl Weight {
    /// Vary over this variable block, constant elsewhere; non-overlapping
    /// blocks stack freely in one product.
    pub fn on(mut self, at: impl Into<Vars>) -> Weight {
        let at = at.into();
        match &self.kind {
            WeightKind::Eq(point) => assert_eq!(
                point.len(),
                at.len,
                "eq weight has {} coordinates but the block has {} variables",
                point.len(),
                at.len
            ),
            WeightKind::Table(values) => assert_eq!(
                values.len(),
                1usize << at.len,
                "table has {} entries but the block addresses {}",
                values.len(),
                1usize << at.len
            ),
        }
        self.placement = Some(at);
        self
    }
}

impl From<Weight> for ClaimExpr {
    fn from(w: Weight) -> ClaimExpr {
        let weights = match w.kind {
            WeightKind::Eq(Scalars::Ring(v)) => Weights::Tensor(Coeffs::Ring(v)),
            WeightKind::Eq(Scalars::Field(v)) => Weights::Tensor(Coeffs::Field(v)),
            WeightKind::Table(Scalars::Ring(v)) => Weights::Dense(Coeffs::Ring(v)),
            WeightKind::Table(Scalars::Field(v)) => Weights::Dense(Coeffs::Field(v)),
        };
        let (prefix_len, suffix_len) = match w.placement {
            Some(at) => (at.skip, at.total - at.skip - at.len),
            None => (0, 0),
        };
        let expr = ClaimExpr::public(PublicFactor { prefix_len, suffix_len, weights });
        match w.coefficient {
            Some(c) => expr.scale(&c),
            None => expr,
        }
    }
}

impl std::ops::Mul for Weight {
    type Output = ClaimExpr;
    fn mul(self, rhs: Weight) -> ClaimExpr {
        ClaimExpr::from(self) * ClaimExpr::from(rhs)
    }
}

impl std::ops::Mul<ClaimExpr> for Weight {
    type Output = ClaimExpr;
    fn mul(self, rhs: ClaimExpr) -> ClaimExpr {
        ClaimExpr::from(self) * rhs
    }
}

impl std::ops::Mul<Weight> for ClaimExpr {
    type Output = ClaimExpr;
    fn mul(self, rhs: Weight) -> ClaimExpr {
        self * ClaimExpr::from(rhs)
    }
}

impl std::ops::Mul<ClaimExpr> for u64 {
    type Output = ClaimExpr;
    fn mul(self, rhs: ClaimExpr) -> ClaimExpr {
        rhs.scale(&RingElement::constant(self % MOD_Q, Representation::IncompleteNTT))
    }
}

/// The full committed vector.
pub fn witness() -> ClaimExpr {
    ClaimExpr::witness()
}

/// One region's entries; the term sums over that region only, at no extra opening.
pub fn witness_in(region: Region) -> ClaimExpr {
    if region.len == region.witness_len {
        ClaimExpr::witness()
    } else {
        ClaimExpr::segment(region.prefix())
    }
}

impl ClaimExpr {
    /// Ring conjugation `X -> X^{-1}` of committed/constant factors;
    /// `witness() * witness().conjugate()` sums to ct = squared l2 norm.
    pub fn conjugate(self) -> ClaimExpr {
        match self {
            ClaimExpr::Factor(ClaimFactor::Witness) => ClaimExpr::conj_witness(),
            ClaimExpr::Factor(ClaimFactor::ConjWitness) => ClaimExpr::witness(),
            ClaimExpr::Factor(ClaimFactor::WitnessSegment(p)) => ClaimExpr::conj_segment(p),
            ClaimExpr::Factor(ClaimFactor::ConjWitnessSegment(p)) => ClaimExpr::segment(p),
            ClaimExpr::Constant(c) => ClaimExpr::Constant(c.conjugate()),
            ClaimExpr::Product(a, b) => a.conjugate() * b.conjugate(),
            ClaimExpr::Sum(a, b) => a.conjugate() + b.conjugate(),
            ClaimExpr::Diff(a, b) => a.conjugate() - b.conjugate(),
            ClaimExpr::Scale(c, x) => x.conjugate().scale(&c.conjugate()),
            ClaimExpr::Factor(ClaimFactor::Public(_)) => {
                panic!("conjugate() does not reach into public weights; conjugate the weight data itself")
            }
        }
    }

    /// The true sum over the witness cube - the `value` a correct prover
    /// ships. Prover-side only.
    pub fn sum(&self, witness: &VerticallyAlignedMatrix<RingElement>) -> RingElement {
        let n = witness.data.len();
        assert!(n.is_power_of_two());
        let total_vars = n.ilog2() as usize;
        let mut acc = zero();
        for index in 0..n {
            let v = eval_at(self, &witness.data, index, total_vars);
            acc += &v;
        }
        acc
    }
}

fn in_prefix(index: usize, p: &Prefix, total_vars: usize) -> bool {
    p.length == 0 || (index >> (total_vars - p.length)) == p.prefix
}

fn eval_at(expr: &ClaimExpr, data: &[RingElement], index: usize, total_vars: usize) -> RingElement {
    match expr {
        ClaimExpr::Factor(ClaimFactor::Witness) => data[index].clone(),
        ClaimExpr::Factor(ClaimFactor::ConjWitness) => data[index].conjugate(),
        ClaimExpr::Factor(ClaimFactor::WitnessSegment(p)) => {
            if in_prefix(index, p, total_vars) { data[index].clone() } else { zero() }
        }
        ClaimExpr::Factor(ClaimFactor::ConjWitnessSegment(p)) => {
            if in_prefix(index, p, total_vars) { data[index].conjugate() } else { zero() }
        }
        ClaimExpr::Factor(ClaimFactor::Public(pf)) => eval_public_at(pf, index, total_vars),
        ClaimExpr::Constant(c) => c.clone(),
        ClaimExpr::Scale(c, x) => {
            let mut v = eval_at(x, data, index, total_vars);
            v *= c;
            v
        }
        ClaimExpr::Product(a, b) => {
            let mut v = eval_at(a, data, index, total_vars);
            v *= &eval_at(b, data, index, total_vars);
            v
        }
        ClaimExpr::Sum(a, b) => {
            let mut v = eval_at(a, data, index, total_vars);
            v += &eval_at(b, data, index, total_vars);
            v
        }
        ClaimExpr::Diff(a, b) => {
            let mut v = eval_at(a, data, index, total_vars);
            v -= &eval_at(b, data, index, total_vars);
            v
        }
    }
}

fn eval_public_at(pf: &PublicFactor, index: usize, total_vars: usize) -> RingElement {
    if let Weights::Selector { bits, length } = &pf.weights {
        let p = Prefix { prefix: *bits, length: *length };
        return if in_prefix(index, &p, total_vars) { one() } else { zero() };
    }
    let middle_vars = total_vars - pf.prefix_len - pf.suffix_len;
    let middle = (index >> pf.suffix_len) & ((1usize << middle_vars) - 1);
    match &pf.weights {
        Weights::Selector { .. } => unreachable!(),
        Weights::Dense(Coeffs::Ring(v)) => v[middle].clone(),
        Weights::Dense(Coeffs::Field(v)) => lowering::embed_qe(&v[middle]),
        Weights::Tensor(Coeffs::Field(layers)) => {
            lowering::embed_qe(&lowering::tensor_at(layers, middle))
        }
        Weights::Tensor(Coeffs::Ring(layers)) => {
            let mut acc = one();
            for (j, a) in layers.iter().enumerate() {
                let bit = (middle >> (layers.len() - 1 - j)) & 1;
                if bit == 1 {
                    acc *= a;
                } else {
                    let mut one_minus = one();
                    one_minus -= a;
                    acc *= &one_minus;
                }
            }
            acc
        }
    }
}

impl SnarkClaim {
    pub fn sums_to(expr: impl Into<ClaimExpr>, value: RingElement) -> SnarkClaim {
        SnarkClaim { expr: expr.into(), value }
    }

    pub fn sums_to_zero(expr: impl Into<ClaimExpr>) -> SnarkClaim {
        SnarkClaim::sums_to(expr, zero())
    }
}

/// Transcript-drawn point, one coordinate per variable; both sides must draw
/// at the same transcript state.
pub fn challenge_point(transcript: &mut Transcript, num_vars: usize) -> Vec<QuadraticExtension> {
    lowering::sample_qe_layers(transcript, num_vars)
}

/// `sum_i eq(point, i) * values[i]`: the claim value both sides compute from
/// public boundary data matching `eq(point) * witness_in(region)`.
pub fn eq_weighted_sum(point: &[QuadraticExtension], values: &[RingElement]) -> RingElement {
    let expanded = lowering::expand_field_tensor(point);
    assert_eq!(expanded.len(), values.len(), "point addresses {} entries, got {}", expanded.len(), values.len());
    let mut acc = zero();
    let mut term = zero();
    for (e, v) in expanded.iter().zip(values.iter()) {
        term *= (e, v);
        acc += &term;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::decomposition::decompose;
    use crate::common::{init_common, sampling::sample_random_short_vector};

    fn short(n: usize, bound: u64) -> Vec<RingElement> {
        sample_random_short_vector(n, bound, Representation::IncompleteNTT)
    }

    fn roundtrip(witness: &VerticallyAlignedMatrix<RingElement>, make: impl Fn(&mut Transcript) -> Vec<Claim>) {
        let mut tp = Transcript::new();
        let claims_p = make(&mut tp);
        let (proof, chain_p) = prove_claims(witness, &claims_p, &mut tp);

        let mut tv = Transcript::new();
        let claims_v = make(&mut tv);
        let chain_v = verify_claims(witness, &claims_v, &proof, &mut tv);

        assert_eq!(chain_p.claims, chain_v.claims);
        for j in 0..chain_p.claims.len() {
            let direct = crate::protocol::open::claim(
                witness,
                &chain_p.evaluation_points_inner[j],
                &chain_p.evaluation_points_outer[j],
            );
            assert_eq!(direct, chain_p.claims[j], "opening {}", j);
        }
    }

    #[test]
    fn test_builder_aligns_regions() {
        init_common();
        let mut layout = WitnessBuilder::new(256, 8);
        let a = layout.push(&short(512, 10));
        let b = layout.push(&short(256, 10));
        let c = layout.push(&short(512, 10));
        assert_eq!((a.start(), a.len()), (0, 512));
        assert_eq!((b.start(), b.len()), (512, 256));
        assert_eq!((c.start(), c.len()), (1024, 512));
        assert_eq!(c.prefix(), Prefix { prefix: 2, length: 2 });
        let (rows, cols) = a.vars().split_at(3);
        assert_eq!((rows.len(), cols.len()), (3, 6));
        let w = layout.finish();
        assert!(w.data[768..1024].iter().all(|x| x == &zero()));
    }

    #[test]
    fn test_table_dot_product_roundtrip() {
        init_common();
        let w = WitnessBuilder {
            height: 64,
            width: 4,
            data: short(256, 100),
            cursor: 256,
        }
        .finish();
        let weights = short(256, 50);

        let expr = table(&weights) * witness();
        let mut direct = zero();
        let mut term = zero();
        for (a, w) in weights.iter().zip(w.data.iter()) {
            term *= (a, w);
            direct += &term;
        }
        assert_eq!(expr.sum(&w), direct);

        roundtrip(&w, |_| vec![Claim::sums_to(table(&weights) * witness(), direct.clone())]);
    }

    #[test]
    fn test_localized_disjoint_weights_roundtrip() {
        init_common();
        let mut layout = WitnessBuilder::new(64, 4);
        let nodes_data = short(64, 100);
        let layer = layout.push(&nodes_data);
        layout.push(&short(128, 100));
        let w = layout.finish();

        let per_slot: Vec<u64> = (0..8).map(|i| 3 * i + 1).collect();
        let make = |t: &mut Transcript| {
            let (node, slot) = layer.vars().split_at(3);
            let alpha = challenge_point(t, node.len());
            let expr = eq(&alpha).on(node) * table(per_slot.clone()).on(slot) * witness_in(layer);
            let value = expr.sum(&w);
            vec![Claim::sums_to(eq(&alpha).on(node) * table(per_slot.clone()).on(slot) * witness_in(layer), value)]
        };
        roundtrip(&w, make);
    }

    #[test]
    fn test_powers_recompose_digits() {
        init_common();
        let base_log = 8u64;
        let digits_per_value = 8usize;
        let values: Vec<RingElement> =
            (0..32).map(|_| RingElement::random(Representation::IncompleteNTT)).collect();
        let digits = decompose(&values, base_log, digits_per_value);

        let mut layout = WitnessBuilder::new(64, 4);
        let digits_at = layout.push(&digits);
        let w = layout.finish();

        let make = |t: &mut Transcript| {
            let (value_index, digit_index) = digits_at.vars().split_at(5);
            let point = challenge_point(t, value_index.len());
            let recomposed = eq(&point).on(value_index)
                * powers(1 << base_log, digit_index.len()).on(digit_index)
                * witness_in(digits_at);
            vec![Claim::sums_to(recomposed, eq_weighted_sum(&point, &values))]
        };
        roundtrip(&w, make);
    }

    #[test]
    fn test_copy_regions_roundtrip() {
        init_common();
        let data = short(128, 100);
        let mut layout = WitnessBuilder::new(64, 4);
        let original = layout.push(&data);
        let mirror = layout.push(&data);
        let w = layout.finish();

        let make = |t: &mut Transcript| {
            let point = challenge_point(t, original.vars().len());
            vec![Claim::sums_to_zero(
                eq(&point).on(original) * (witness_in(original) - witness_in(mirror)),
            )]
        };
        roundtrip(&w, make);
    }

    #[test]
    #[should_panic(expected = "round claim mismatch")]
    fn test_copy_regions_tampered_rejected() {
        init_common();
        let data = short(128, 100);
        let mut tampered = data.clone();
        tampered[17] += &one();
        let mut layout = WitnessBuilder::new(64, 4);
        let original = layout.push(&data);
        let mirror = layout.push(&tampered);
        let w = layout.finish();

        let make = |t: &mut Transcript| {
            let point = challenge_point(t, original.vars().len());
            vec![Claim::sums_to_zero(
                eq(&point).on(original) * (witness_in(original) - witness_in(mirror)),
            )]
        };
        let mut tp = Transcript::new();
        let claims_p = make(&mut tp);
        let (proof, _) = prove_claims(&w, &claims_p, &mut tp);
        let mut tv = Transcript::new();
        let claims_v = make(&mut tv);
        verify_claims(&w, &claims_v, &proof, &mut tv);
    }

    #[test]
    fn test_norm_claim_roundtrip() {
        init_common();
        let w = WitnessBuilder {
            height: 64,
            width: 4,
            data: short(256, 100),
            cursor: 256,
        }
        .finish();

        let energy = (witness() * witness().conjugate()).sum(&w);

        let mut norm_sq: u128 = 0;
        for e in &w.data {
            let mut c = e.clone();
            c.to_representation(Representation::Coefficients);
            for &x in c.v.iter() {
                let signed = if x > MOD_Q / 2 { MOD_Q - x } else { x };
                norm_sq += signed as u128 * signed as u128;
            }
        }
        let mut ct = energy.clone();
        ct.to_representation(Representation::Coefficients);
        assert_eq!(ct.v[0] as u128, norm_sq % MOD_Q as u128);

        roundtrip(&w, |_| {
            vec![Claim::sums_to(witness() * witness().conjugate(), energy.clone())]
        });
    }

    #[test]
    fn test_scaled_difference_roundtrip() {
        init_common();
        let w = WitnessBuilder {
            height: 64,
            width: 4,
            data: short(256, 100),
            cursor: 256,
        }
        .finish();
        let a = short(256, 50);
        let b = short(256, 30);

        let expr = 7 * (table(&a) * witness()) - table(&b) * witness();
        let value = expr.sum(&w);
        roundtrip(&w, |_| {
            vec![Claim::sums_to(
                7 * (table(&a) * witness()) - table(&b) * witness(),
                value.clone(),
            )]
        });
    }
}
