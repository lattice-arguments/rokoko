use crate::{
    common::{
        hash::HashWrapper,
        matrix::VerticallyAlignedMatrix,
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        sumcheck_element::SumcheckElement,
        structured_row::{PreprocessedRow, StructuredRow},
    },
    protocol::{
        commitment::Prefix,
        open::{evaluation_point_to_structured_row, evaluation_point_to_structured_row_conjugate},
        sumcheck_utils::{
            combiner::{Combiner, CombinerEvaluation},
            common::{EvaluationSumcheckData, HighOrderSumcheckData, SumcheckBaseData},
            diff::{DiffSumcheck, DiffSumcheckEvaluation},
            elephant_cell::ElephantCell,
            linear::{BasicEvaluationLinearSumcheck, FakeEvaluationLinearSumcheck, LinearSumcheck},
            polynomial::Polynomial,
            product::{ProductSumcheck, ProductSumcheckEvaluation},
            ring_to_field_combiner::{RingToFieldCombiner, RingToFieldCombinerEvaluation},
            selector_eq::{SelectorEq, SelectorEqEvaluation},
            sum::{SumSumcheck, SumSumcheckEvaluation},
        },
    },
};

use std::sync::Arc;

/// A public weight vector. `weights` is its shape/representation;
/// `prefix_len`/`suffix_len` make it constant on the top/bottom variables and
/// vary over the middle (`(0, 0)` is the full cube).
#[derive(Clone)]
pub struct PublicFactor {
    pub prefix_len: usize,
    pub suffix_len: usize,
    pub weights: Weights,
}

/// `Tensor` reads these as per-variable layers (MSB-first); `Dense` as the
/// full middle table. `Field` evaluates faster on the verifier than `Ring`.
#[derive(Clone)]
pub enum Coeffs {
    Ring(Arc<Vec<RingElement>>),
    Field(Arc<Vec<QuadraticExtension>>),
}

#[derive(Clone)]
pub enum Weights {
    /// Product eq-tensor with layers `[1-a, a]`, MSB-first; verifier `O(layers)`.
    Tensor(Coeffs),
    /// Arbitrary table; verifier linear in its length.
    Dense(Coeffs),
    /// `eq(bits, .)` over the leading `length` variables; zero-cost gadget that
    /// every `WitnessSegment` lowers to.
    Selector { bits: usize, length: usize },
}

pub fn qe_one_minus(a: &QuadraticExtension) -> QuadraticExtension {
    let mut r = QuadraticExtension::one();
    r -= a;
    r
}

pub fn expand_field_tensor(layers: &[QuadraticExtension]) -> Vec<RingElement> {
    use crate::common::arithmetic::field_to_ring_element_into;
    let mut vals = vec![QuadraticExtension::one()];
    for a in layers.iter().rev() {
        let one_minus = qe_one_minus(a);
        let mut next = Vec::with_capacity(vals.len() * 2);
        let mut t = QuadraticExtension::zero();
        for v in &vals {
            t *= (v, &one_minus);
            next.push(t);
        }
        for v in &vals {
            t *= (v, a);
            next.push(t);
        }
        vals = next;
    }
    vals.iter()
        .map(|v| {
            let mut r = RingElement::zero(Representation::IncompleteNTT);
            field_to_ring_element_into(&mut r, v);
            r.from_homogenized_field_extensions_to_incomplete_ntt();
            r
        })
        .collect()
}

/// Transcript challenges for tensor layers, MSB-first.
pub fn sample_qe_layers(hw: &mut HashWrapper, n: usize) -> Vec<QuadraticExtension> {
    (0..n)
        .map(|_| {
            let mut f = QuadraticExtension::zero();
            hw.sample_field_element_into(&mut f);
            f
        })
        .collect()
}

/// The field scalar as a ring element, for use in term coefficients and
/// public weights.
pub fn embed_qe(v: &QuadraticExtension) -> RingElement {
    use crate::common::arithmetic::field_to_ring_element_into;
    let mut r = RingElement::zero(Representation::IncompleteNTT);
    field_to_ring_element_into(&mut r, v);
    r.from_homogenized_field_extensions_to_incomplete_ntt();
    r
}

pub fn qe_mul(a: &QuadraticExtension, b: &QuadraticExtension) -> QuadraticExtension {
    let mut r = QuadraticExtension::zero();
    r *= (a, b);
    r
}

/// Entry `index` of the eq-tensor with the given layers (equivalently,
/// `eq(layers, bits(index))`).
pub fn tensor_at(layers_msb: &[QuadraticExtension], index: usize) -> QuadraticExtension {
    let mut r = QuadraticExtension::one();
    for (j, a) in layers_msb.iter().enumerate() {
        let bit = (index >> (layers_msb.len() - 1 - j)) & 1;
        let f = if bit == 1 { a.clone() } else { qe_one_minus(a) };
        r = qe_mul(&r, &f);
    }
    r
}

/// `eq(a, z)` over matching layer/point slices, MSB-first.
pub fn eq_layers_qe(a: &[QuadraticExtension], z: &[QuadraticExtension]) -> QuadraticExtension {
    let mut r = QuadraticExtension::one();
    for (x, y) in a.iter().zip(z.iter()) {
        let mut t = qe_mul(x, y);
        t += &qe_mul(&qe_one_minus(x), &qe_one_minus(y));
        r = qe_mul(&r, &t);
    }
    r
}

/// The weight pair `(1, w)` as an eq layer: `(1 + w) * (1 - a, a)`. Returns
/// the layer value and the scale to fold into the term coefficient.
pub fn weighted_layer(w: u64) -> (QuadraticExtension, u64) {
    use crate::common::arithmetic::inv_mod;
    use crate::common::config::MOD_Q;
    let scale = (1 + w as u128 % MOD_Q as u128) as u64 % MOD_Q;
    assert_ne!(scale, 0, "weighted_layer is undefined for w = -1 mod q");
    let mut a = QuadraticExtension::zero();
    a.coeffs[0] = (w as u128 * inv_mod(scale) as u128 % MOD_Q as u128) as u64;
    (a, scale)
}

/// One factor of a claim term: an oracle evaluated at the common cube point.
pub enum ClaimFactor {
    /// MLE of the full committed vector; its evaluation is the standard
    /// opening z_0.
    Witness,
    /// MLE of the conjugated vector (X -> X^{-1} per element); the standard
    /// opening z_1.
    ConjWitness,
    /// The witness slice under a binary prefix: a term holding this factor
    /// sums over the segment's block only. Lowered internally to
    /// `eq(prefix, .)` times the full-vector oracle, so it costs one extra
    /// factor of term degree and no opening of its own.
    WitnessSegment(Prefix),
    /// Conjugate of a witness segment; lowered like [`Self::WitnessSegment`]
    /// against the conjugated vector.
    ConjWitnessSegment(Prefix),
    Public(PublicFactor),
}

/// A claim expression: a high-order combination of public and private
/// (witness) leaves, closed under product, sum, difference and scaling. It
/// lowers 1:1 to the sumcheck combinators - `Product` to `ProductSumcheck`,
/// `Sum` to `SumSumcheck`, `Diff` to `DiffSumcheck`. Build it from the leaf
/// constructors with the `+ - *` operators or the matching `add`/`sub`/`mul`
/// methods; `scale` multiplies by a ring scalar and `neg` flips the sign.
pub enum ClaimExpr {
    /// A single oracle or public-weight leaf.
    Factor(ClaimFactor),
    /// A ring scalar, constant over the cube.
    Constant(RingElement),
    /// Pointwise product of the two sub-expressions.
    Product(Box<ClaimExpr>, Box<ClaimExpr>),
    /// Sum of the two sub-expressions.
    Sum(Box<ClaimExpr>, Box<ClaimExpr>),
    /// Difference `lhs - rhs`.
    Diff(Box<ClaimExpr>, Box<ClaimExpr>),
    /// Scale a sub-expression by a ring scalar.
    Scale(RingElement, Box<ClaimExpr>),
}

impl ClaimExpr {
    pub(crate) fn witness() -> ClaimExpr {
        ClaimExpr::Factor(ClaimFactor::Witness)
    }
    pub(crate) fn conj_witness() -> ClaimExpr {
        ClaimExpr::Factor(ClaimFactor::ConjWitness)
    }
    pub(crate) fn segment(prefix: Prefix) -> ClaimExpr {
        ClaimExpr::Factor(ClaimFactor::WitnessSegment(prefix))
    }
    pub(crate) fn conj_segment(prefix: Prefix) -> ClaimExpr {
        ClaimExpr::Factor(ClaimFactor::ConjWitnessSegment(prefix))
    }
    pub(crate) fn public(factor: PublicFactor) -> ClaimExpr {
        ClaimExpr::Factor(ClaimFactor::Public(factor))
    }
    pub(crate) fn constant(value: RingElement) -> ClaimExpr {
        ClaimExpr::Constant(value)
    }

    pub fn mul(self, rhs: ClaimExpr) -> ClaimExpr {
        ClaimExpr::Product(Box::new(self), Box::new(rhs))
    }
    pub fn add(self, rhs: ClaimExpr) -> ClaimExpr {
        ClaimExpr::Sum(Box::new(self), Box::new(rhs))
    }
    pub fn sub(self, rhs: ClaimExpr) -> ClaimExpr {
        ClaimExpr::Diff(Box::new(self), Box::new(rhs))
    }
    pub fn scale(self, scale: &RingElement) -> ClaimExpr {
        ClaimExpr::Scale(scale.clone(), Box::new(self))
    }
    pub fn neg(self) -> ClaimExpr {
        self.scale(&RingElement::constant(
            crate::common::config::MOD_Q - 1,
            Representation::IncompleteNTT,
        ))
    }
}

impl std::ops::Mul for ClaimExpr {
    type Output = ClaimExpr;
    fn mul(self, rhs: ClaimExpr) -> ClaimExpr {
        ClaimExpr::mul(self, rhs)
    }
}
impl std::ops::Add for ClaimExpr {
    type Output = ClaimExpr;
    fn add(self, rhs: ClaimExpr) -> ClaimExpr {
        ClaimExpr::add(self, rhs)
    }
}
impl std::ops::Sub for ClaimExpr {
    type Output = ClaimExpr;
    fn sub(self, rhs: ClaimExpr) -> ClaimExpr {
        ClaimExpr::sub(self, rhs)
    }
}
impl std::ops::Neg for ClaimExpr {
    type Output = ClaimExpr;
    fn neg(self) -> ClaimExpr {
        ClaimExpr::neg(self)
    }
}

/// One functional-sumcheck claim: `sum_z expr(z) = value`.
pub struct SnarkClaim {
    pub(crate) expr: ClaimExpr,
    pub(crate) value: RingElement,
}

impl SnarkClaim {
    /// The claimed sum.
    pub fn value(&self) -> &RingElement {
        &self.value
    }
}

pub struct InitialSumcheckProof {
    pub polys: Vec<Polynomial<QuadraticExtension>>,
    /// `z_0 = MLE[vec(W)](c)`
    pub witness_eval: RingElement,
    /// `z_1 = MLE[conj(vec(W))](c)`
    pub conj_witness_eval: RingElement,
}

impl crate::protocol::config::SizeableProof for InitialSumcheckProof {
    fn size_in_bits(&self) -> usize {
        let mut size = self.witness_eval.size_in_bits() + self.conj_witness_eval.size_in_bits();
        for p in &self.polys {
            for c in &p.coefficients[..p.num_coefficients] {
                size += c.size_in_bits();
            }
        }
        size
    }
}

/// What the PCS chain consumes as its initial statement: evaluation rows and
/// outer claims (paper: l_j = tensor(c_1), r_j = tensor(c_0), t_j = z_j).
pub struct ChainInputs {
    pub evaluation_points_inner: Vec<StructuredRow>,
    pub evaluation_points_outer: Vec<StructuredRow>,
    pub claims: Vec<RingElement>,
}

fn is_unit(e: &RingElement) -> bool {
    e == &RingElement::constant(1, Representation::IncompleteNTT)
}

enum LeafCell {
    Linear(ElephantCell<LinearSumcheck<RingElement>>),
    Selector(ElephantCell<SelectorEq<RingElement>>),
}

impl LeafCell {
    fn partial_evaluate(&self, r: &RingElement) {
        match self {
            LeafCell::Linear(c) => c.borrow_mut().partial_evaluate(r),
            LeafCell::Selector(c) => c.borrow_mut().partial_evaluate(r),
        }
    }
}

/// Oracle pool keyed by leaf identity: distinct cell per use within one product
/// region (a cell cannot be aliased inside one product), reused across additive
/// branches and claims so a shared oracle folds once.
struct OraclePool {
    pools: std::collections::HashMap<(usize, usize, usize, bool), (Vec<ElephantCell<LinearSumcheck<RingElement>>>, usize)>,
}

const FULL_WITNESS_KEY: (usize, usize, usize, bool) = (usize::MAX, usize::MAX, 0, false);

impl OraclePool {
    fn new() -> Self {
        OraclePool {
            pools: std::collections::HashMap::new(),
        }
    }

    fn next(
        &mut self,
        key: (usize, usize, usize, bool),
        make: impl Fn() -> LinearSumcheck<RingElement>,
    ) -> ElephantCell<LinearSumcheck<RingElement>> {
        let entry = self.pools.entry(key).or_insert_with(|| (vec![], 0));
        if entry.1 == entry.0.len() {
            entry.0.push(ElephantCell::new(make()));
        }
        let cell = entry.0[entry.1].clone();
        entry.1 += 1;
        cell
    }

    fn reset_term(&mut self) {
        for entry in self.pools.values_mut() {
            entry.1 = 0;
        }
    }

    fn first_cell(&self, key: &(usize, usize, usize, bool)) -> Option<&ElephantCell<LinearSumcheck<RingElement>>> {
        self.pools.get(key).and_then(|(cells, _)| cells.first())
    }

    fn all_cells(&self) -> impl Iterator<Item = &ElephantCell<LinearSumcheck<RingElement>>> {
        self.pools.values().flat_map(|(cells, _)| cells.iter())
    }
}

type HighOrderCell = ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>;
type EvalCell = ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>;

/// Prover-side lowering of a canonical [`ClaimExpr`] to the gadget tree. A
/// product region is entered from an additive context (`in_product == false`),
/// where the pools reset so a shared oracle folds once and is reused across
/// additive branches; inside one region the pools keep advancing so each leaf
/// occurrence gets its own cell.
struct ProverAssembler<'a> {
    witness: &'a [RingElement],
    conjugated: &'a [RingElement],
    n: usize,
    total_vars: usize,
    witness_pool: OraclePool,
    conj_pool: OraclePool,
    public_pool: OraclePool,
    leaves: Vec<LeafCell>,
}

impl<'a> ProverAssembler<'a> {
    fn reset(&mut self) {
        self.witness_pool.reset_term();
        self.conj_pool.reset_term();
        self.public_pool.reset_term();
    }

    fn build(&mut self, expr: &ClaimExpr, in_product: bool) -> HighOrderCell {
        match expr {
            ClaimExpr::Sum(a, b) => {
                let l = self.build(a, in_product);
                let r = self.build(b, in_product);
                ElephantCell::new(SumSumcheck::new(l, r)) as _
            }
            ClaimExpr::Diff(a, b) => {
                let l = self.build(a, in_product);
                let r = self.build(b, in_product);
                ElephantCell::new(DiffSumcheck::new(l, r)) as _
            }
            _ => {
                if !in_product {
                    self.reset();
                }
                self.build_mult(expr)
            }
        }
    }

    fn build_mult(&mut self, expr: &ClaimExpr) -> HighOrderCell {
        match expr {
            ClaimExpr::Product(a, b) => {
                let l = self.build_mult(a);
                let r = self.build_mult(b);
                ElephantCell::new(ProductSumcheck::new(l, r)) as _
            }
            ClaimExpr::Sum(_, _) | ClaimExpr::Diff(_, _) => self.build(expr, true),
            ClaimExpr::Factor(f) => self.leaf(f),
            ClaimExpr::Constant(c) => self.constant_leaf(c),
            ClaimExpr::Scale(_, _) => unreachable!("scale is folded into a constant by canon"),
        }
    }

    fn constant_leaf(&mut self, value: &RingElement) -> HighOrderCell {
        let mut ls = LinearSumcheck::new_with_prefixed_sufixed_data(1, self.total_vars, 0);
        ls.load_from(std::slice::from_ref(value));
        let cell = ElephantCell::new(ls);
        self.leaves.push(LeafCell::Linear(cell.clone()));
        cell as _
    }

    fn full_ring_leaf(&mut self, data: Vec<RingElement>) -> HighOrderCell {
        let mut ls = LinearSumcheck::new(self.n);
        ls.load_from(&data);
        let cell = ElephantCell::new(ls);
        self.leaves.push(LeafCell::Linear(cell.clone()));
        cell as _
    }

    fn pooled_leaf(
        &mut self,
        ptr: usize,
        prefix_len: usize,
        suffix_len: usize,
        data: Vec<RingElement>,
    ) -> HighOrderCell {
        let cell = self.public_pool.next((ptr, prefix_len, suffix_len, false), move || {
            let mut ls =
                LinearSumcheck::new_with_prefixed_sufixed_data(data.len(), prefix_len, suffix_len);
            ls.load_from(&data);
            ls
        });
        cell as _
    }

    fn leaf(&mut self, factor: &ClaimFactor) -> HighOrderCell {
        match factor {
            ClaimFactor::Witness => {
                let data = self.witness;
                let cell = self.witness_pool.next(FULL_WITNESS_KEY, move || {
                    let mut ls = LinearSumcheck::new(data.len());
                    ls.load_from(data);
                    ls
                });
                cell as _
            }
            ClaimFactor::ConjWitness => {
                let data = self.conjugated;
                let cell = self.conj_pool.next(FULL_WITNESS_KEY, move || {
                    let mut ls = LinearSumcheck::new(data.len());
                    ls.load_from(data);
                    ls
                });
                cell as _
            }
            ClaimFactor::WitnessSegment(_) | ClaimFactor::ConjWitnessSegment(_) => {
                unreachable!("segments are lowered by canon")
            }
            ClaimFactor::Public(public) => {
                let PublicFactor {
                    prefix_len,
                    suffix_len,
                    weights,
                } = public;
                let prefix_len = *prefix_len;
                let suffix_len = *suffix_len;
                let placed = prefix_len != 0 || suffix_len != 0;
                let middle_len = self.n >> (prefix_len + suffix_len);
                match weights {
                    Weights::Selector { bits, length } => {
                        let cell =
                            ElephantCell::new(SelectorEq::new(*bits, *length, self.total_vars));
                        self.leaves.push(LeafCell::Selector(cell.clone()));
                        cell as _
                    }
                    Weights::Tensor(Coeffs::Ring(layers)) if !placed => {
                        assert_eq!(layers.len(), self.total_vars);
                        self.full_ring_leaf(
                            PreprocessedRow::from_layers(&layers[..]).preprocessed_row,
                        )
                    }
                    Weights::Dense(Coeffs::Ring(v)) if !placed => {
                        assert_eq!(v.len(), self.n);
                        self.full_ring_leaf((**v).clone())
                    }
                    Weights::Tensor(Coeffs::Ring(layers)) => {
                        assert_eq!(1usize << layers.len(), middle_len);
                        self.pooled_leaf(
                            Arc::as_ptr(layers) as *const () as usize,
                            prefix_len,
                            suffix_len,
                            PreprocessedRow::from_layers(&layers[..]).preprocessed_row,
                        )
                    }
                    Weights::Tensor(Coeffs::Field(layers)) => {
                        assert_eq!(1usize << layers.len(), middle_len);
                        self.pooled_leaf(
                            Arc::as_ptr(layers) as *const () as usize,
                            prefix_len,
                            suffix_len,
                            expand_field_tensor(&layers[..]),
                        )
                    }
                    Weights::Dense(Coeffs::Ring(v)) => {
                        assert_eq!(v.len(), middle_len);
                        self.pooled_leaf(
                            Arc::as_ptr(v) as *const () as usize,
                            prefix_len,
                            suffix_len,
                            (**v).clone(),
                        )
                    }
                    Weights::Dense(Coeffs::Field(v)) => {
                        assert_eq!(v.len(), middle_len);
                        self.pooled_leaf(
                            Arc::as_ptr(v) as *const () as usize,
                            prefix_len,
                            suffix_len,
                            v.iter().map(embed_qe).collect::<Vec<_>>(),
                        )
                    }
                }
            }
        }
    }
}

/// Verifier-side mirror: lowers the same canonical [`ClaimExpr`] to evaluation
/// gadgets at the final point. The witness and conjugate evaluations are shared
/// cells; public factors and constants build fresh.
struct VerifierAssembler {
    witness_eval: EvalCell,
    conj_eval: EvalCell,
    n: usize,
    total_vars: usize,
}

impl VerifierAssembler {
    fn build(&self, expr: &ClaimExpr) -> EvalCell {
        match expr {
            ClaimExpr::Sum(a, b) => {
                ElephantCell::new(SumSumcheckEvaluation::new(self.build(a), self.build(b))) as _
            }
            ClaimExpr::Diff(a, b) => {
                ElephantCell::new(DiffSumcheckEvaluation::new(self.build(a), self.build(b))) as _
            }
            ClaimExpr::Product(a, b) => {
                ElephantCell::new(ProductSumcheckEvaluation::new(self.build(a), self.build(b))) as _
            }
            ClaimExpr::Factor(ClaimFactor::Witness) => self.witness_eval.clone(),
            ClaimExpr::Factor(ClaimFactor::ConjWitness) => self.conj_eval.clone(),
            ClaimExpr::Factor(ClaimFactor::WitnessSegment(_))
            | ClaimExpr::Factor(ClaimFactor::ConjWitnessSegment(_)) => {
                unreachable!("segments are lowered by canon")
            }
            ClaimExpr::Factor(ClaimFactor::Public(pf)) => self.public(pf),
            ClaimExpr::Constant(value) => {
                let cell = ElephantCell::new(FakeEvaluationLinearSumcheck::new());
                cell.borrow_mut().set_result(value.clone());
                cell as _
            }
            ClaimExpr::Scale(_, _) => unreachable!("scale is folded into a constant by canon"),
        }
    }

    fn public(&self, pf: &PublicFactor) -> EvalCell {
        use crate::protocol::sumcheck_utils::linear::{
            RingToFieldWrapperEvaluation, StructuredRowEvaluationLinearSumcheck,
        };
        let PublicFactor {
            prefix_len,
            suffix_len,
            weights,
        } = pf;
        let prefix_len = *prefix_len;
        let suffix_len = *suffix_len;
        let placed = prefix_len != 0 || suffix_len != 0;
        match weights {
            Weights::Selector { bits, length } => {
                ElephantCell::new(SelectorEqEvaluation::new(*bits, *length, self.total_vars)) as _
            }
            Weights::Tensor(Coeffs::Ring(layers)) if !placed => {
                let mut ev = StructuredRowEvaluationLinearSumcheck::new(self.n);
                ev.load_from(StructuredRow {
                    tensor_layers: (**layers).clone(),
                });
                ElephantCell::new(ev) as _
            }
            Weights::Dense(Coeffs::Ring(v)) if !placed => {
                let mut ev = BasicEvaluationLinearSumcheck::new(self.n);
                ev.load_from(&v[..]);
                ElephantCell::new(ev) as _
            }
            Weights::Tensor(Coeffs::Ring(layers)) => {
                let mut ev = StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
                    1usize << layers.len(),
                    prefix_len,
                    suffix_len,
                );
                ev.load_from(StructuredRow {
                    tensor_layers: (**layers).clone(),
                });
                ElephantCell::new(ev) as _
            }
            Weights::Tensor(Coeffs::Field(layers)) => {
                let mut ev = StructuredRowEvaluationLinearSumcheck::<QuadraticExtension>::new_with_prefixed_sufixed_data(
                    1usize << layers.len(),
                    prefix_len,
                    suffix_len,
                );
                ev.load_from(StructuredRow {
                    tensor_layers: (**layers).clone(),
                });
                ElephantCell::new(RingToFieldWrapperEvaluation::new(ElephantCell::new(ev) as _)) as _
            }
            Weights::Dense(Coeffs::Ring(v)) => {
                let mut ev = BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
                    v.len(),
                    prefix_len,
                    suffix_len,
                );
                ev.load_from(&v[..]);
                ElephantCell::new(ev) as _
            }
            Weights::Dense(Coeffs::Field(v)) => {
                let mut ev = BasicEvaluationLinearSumcheck::<QuadraticExtension>::new_with_prefixed_sufixed_data(
                    v.len(),
                    prefix_len,
                    suffix_len,
                );
                ev.load_from(&v[..]);
                ElephantCell::new(RingToFieldWrapperEvaluation::new(ElephantCell::new(ev) as _)) as _
            }
        }
    }
}

fn chain_inputs(
    evaluation_points: &[RingElement],
    witness_width: usize,
    witness_eval: &RingElement,
    conj_witness_eval: &RingElement,
) -> ChainInputs {
    let width_bits = witness_width.ilog2() as usize;
    let (points_outer, points_inner) = evaluation_points.split_at(width_bits);
    ChainInputs {
        evaluation_points_inner: vec![
            evaluation_point_to_structured_row(points_inner),
            evaluation_point_to_structured_row_conjugate(points_inner),
        ],
        evaluation_points_outer: vec![
            evaluation_point_to_structured_row(points_outer),
            evaluation_point_to_structured_row_conjugate(points_outer),
        ],
        claims: vec![witness_eval.clone(), conj_witness_eval.conjugate()],
    }
}
/// Canonicalize a claim: lower segments to `selector x witness`, fold scales
/// into a single constant per product, and collapse duplicate selectors within
/// a product (eq is 0/1 on the cube, so eq^2 = eq and the sum is unchanged).
/// The result's leaves are `Witness`/`ConjWitness`/`Public`/`Constant` only;
/// both prover and verifier consume it, so their trees match exactly.
fn canonicalize(claim: &SnarkClaim) -> SnarkClaim {
    SnarkClaim {
        expr: canon(&claim.expr),
        value: claim.value.clone(),
    }
}

fn canon(expr: &ClaimExpr) -> ClaimExpr {
    match expr {
        ClaimExpr::Sum(a, b) => ClaimExpr::Sum(Box::new(canon(a)), Box::new(canon(b))),
        ClaimExpr::Diff(a, b) => ClaimExpr::Diff(Box::new(canon(a)), Box::new(canon(b))),
        _ => canon_product(expr),
    }
}

fn canon_product(expr: &ClaimExpr) -> ClaimExpr {
    let mut const_acc = RingElement::constant(1, Representation::IncompleteNTT);
    let mut have_const = false;
    let mut selectors: Vec<(usize, usize)> = vec![];
    let mut members: Vec<ClaimExpr> = vec![];
    gather_product(
        expr,
        &mut const_acc,
        &mut have_const,
        &mut selectors,
        &mut members,
    );

    let mut parts: Vec<ClaimExpr> = vec![];
    if have_const && !is_unit(&const_acc) {
        parts.push(ClaimExpr::Constant(const_acc.clone()));
    }
    for (bits, length) in selectors {
        parts.push(ClaimExpr::public(PublicFactor {
            prefix_len: 0,
            suffix_len: 0,
            weights: Weights::Selector { bits, length },
        }));
    }
    parts.extend(members);

    let mut parts = parts.into_iter();
    let mut acc = match parts.next() {
        Some(first) => first,
        None => return ClaimExpr::Constant(const_acc),
    };
    for p in parts {
        acc = ClaimExpr::Product(Box::new(acc), Box::new(p));
    }
    acc
}

fn push_selector(selectors: &mut Vec<(usize, usize)>, bits: usize, length: usize) {
    if !selectors.contains(&(bits, length)) {
        selectors.push((bits, length));
    }
}

fn gather_product(
    expr: &ClaimExpr,
    const_acc: &mut RingElement,
    have_const: &mut bool,
    selectors: &mut Vec<(usize, usize)>,
    members: &mut Vec<ClaimExpr>,
) {
    match expr {
        ClaimExpr::Scale(c, x) => {
            *const_acc *= c;
            *have_const = true;
            gather_product(x, const_acc, have_const, selectors, members);
        }
        ClaimExpr::Constant(c) => {
            *const_acc *= c;
            *have_const = true;
        }
        ClaimExpr::Product(a, b) => {
            gather_product(a, const_acc, have_const, selectors, members);
            gather_product(b, const_acc, have_const, selectors, members);
        }
        ClaimExpr::Sum(_, _) | ClaimExpr::Diff(_, _) => members.push(canon(expr)),
        ClaimExpr::Factor(ClaimFactor::WitnessSegment(p)) => {
            push_selector(selectors, p.prefix, p.length);
            members.push(ClaimExpr::witness());
        }
        ClaimExpr::Factor(ClaimFactor::ConjWitnessSegment(p)) => {
            push_selector(selectors, p.prefix, p.length);
            members.push(ClaimExpr::conj_witness());
        }
        ClaimExpr::Factor(ClaimFactor::Public(pf)) => {
            if let Weights::Selector { bits, length } = &pf.weights {
                push_selector(selectors, *bits, *length);
            } else {
                members.push(ClaimExpr::public(pf.clone()));
            }
        }
        ClaimExpr::Factor(ClaimFactor::Witness) => members.push(ClaimExpr::witness()),
        ClaimExpr::Factor(ClaimFactor::ConjWitness) => members.push(ClaimExpr::conj_witness()),
    }
}

/// Structural validation of a canonical claim. A round polynomial's degree at a
/// variable is the number of factors depending on it, so `Product` adds the
/// per-variable degrees and `Sum`/`Diff` take the max; the cap is per-variable
/// (factors on disjoint blocks multiply freely) and at most three.
fn validate_claims(claims: &[SnarkClaim], total_vars: usize) {
    assert!(!claims.is_empty(), "no claims");
    for claim in claims {
        let degree = expr_degree(&claim.expr, total_vars);
        assert!(
            degree.iter().all(|&d| d <= 3),
            "a term's round polynomials carry degree at most three; some variable \
             is multiplied by more than three factors"
        );
    }
}

fn expr_degree(expr: &ClaimExpr, total_vars: usize) -> Vec<usize> {
    match expr {
        ClaimExpr::Factor(ClaimFactor::Witness) | ClaimExpr::Factor(ClaimFactor::ConjWitness) => {
            vec![1; total_vars]
        }
        ClaimExpr::Factor(ClaimFactor::WitnessSegment(_))
        | ClaimExpr::Factor(ClaimFactor::ConjWitnessSegment(_)) => {
            unreachable!("segments are lowered by canon")
        }
        ClaimExpr::Factor(ClaimFactor::Public(pf)) => {
            let (lo, hi) = public_support(pf, total_vars);
            let mut d = vec![0usize; total_vars];
            for slot in &mut d[lo..hi] {
                *slot = 1;
            }
            d
        }
        ClaimExpr::Constant(_) => vec![0usize; total_vars],
        ClaimExpr::Product(a, b) => {
            let mut da = expr_degree(a, total_vars);
            for (x, y) in da.iter_mut().zip(expr_degree(b, total_vars)) {
                *x += y;
            }
            da
        }
        ClaimExpr::Sum(a, b) | ClaimExpr::Diff(a, b) => {
            let mut da = expr_degree(a, total_vars);
            for (x, y) in da.iter_mut().zip(expr_degree(b, total_vars)) {
                *x = (*x).max(y);
            }
            da
        }
        ClaimExpr::Scale(_, x) => expr_degree(x, total_vars),
    }
}

fn public_support(pf: &PublicFactor, total_vars: usize) -> (usize, usize) {
    match &pf.weights {
        Weights::Selector { bits, length } => {
            assert!(*length <= total_vars, "prefix length exceeds the cube");
            assert!(
                *length == 0 || *bits < (1usize << *length),
                "prefix value exceeds its declared length"
            );
            (0, *length)
        }
        _ => {
            assert!(
                pf.prefix_len + pf.suffix_len <= total_vars,
                "weight placement exceeds the cube"
            );
            (pf.prefix_len, total_vars - pf.suffix_len)
        }
    }
}

pub fn prove_claims(
    witness: &VerticallyAlignedMatrix<RingElement>,
    claims: &[SnarkClaim],
    hash_wrapper: &mut HashWrapper,
) -> (InitialSumcheckProof, ChainInputs) {
    let n = witness.data.len();
    assert!(n.is_power_of_two());
    let total_vars = n.ilog2() as usize;
    let canon: Vec<SnarkClaim> = claims.iter().map(canonicalize).collect();
    let claims = &canon[..];
    validate_claims(claims, total_vars);

    // z_1 (the conjugate opening) is always part of the chain inputs, so the
    // conjugated vector is needed whether or not any claim conjugates.
    let mut conjugated = vec![RingElement::zero(Representation::IncompleteNTT); n];
    witness
        .data
        .iter()
        .zip(conjugated.iter_mut())
        .for_each(|(orig, conj)| orig.conjugate_into(conj));

    let mut asm = ProverAssembler {
        witness: &witness.data,
        conjugated: &conjugated,
        n,
        total_vars,
        witness_pool: OraclePool::new(),
        conj_pool: OraclePool::new(),
        public_pool: OraclePool::new(),
        leaves: vec![],
    };

    let mut outputs: Vec<HighOrderCell> = vec![];
    for claim in claims {
        outputs.push(asm.build(&claim.expr, false));
    }

    // ensure the full-witness oracle exists: z_0 always seeds the chain
    if asm.witness_pool.first_cell(&FULL_WITNESS_KEY).is_none() {
        let data = asm.witness;
        asm.witness_pool.next(FULL_WITNESS_KEY, move || {
            let mut ls = LinearSumcheck::new(data.len());
            ls.load_from(data);
            ls
        });
    }
    let pooled: Vec<ElephantCell<LinearSumcheck<RingElement>>> = asm
        .witness_pool
        .all_cells()
        .chain(asm.conj_pool.all_cells())
        .chain(asm.public_pool.all_cells())
        .cloned()
        .collect();
    for cell in pooled {
        asm.leaves.push(LeafCell::Linear(cell));
    }

    // Bind the claim values, then sample batching challenges
    for claim in claims {
        hash_wrapper.update_with_ring_element(&claim.value);
    }
    let mut combination = vec![RingElement::zero(Representation::IncompleteNTT); outputs.len()];
    hash_wrapper.sample_ring_element_vec_into(&mut combination);

    let mut combination_to_field = RingElement::zero(Representation::IncompleteNTT);
    hash_wrapper.sample_ring_element_into(&mut combination_to_field);
    combination_to_field.from_incomplete_ntt_to_homogenized_field_extensions();
    let qe = combination_to_field.split_into_quadratic_extensions();

    let mut combiner = Combiner::new(outputs);
    combiner.load_challenges_from(&combination);
    let combiner_cell = ElephantCell::new(combiner);
    let mut field_combiner = RingToFieldCombiner::new(combiner_cell as _);
    field_combiner.load_challenges_from(qe.clone());

    let mut num_vars = total_vars;
    let mut polys: Vec<Polynomial<QuadraticExtension>> = vec![];
    let mut evaluation_points: Vec<RingElement> = vec![];

    use crate::common::arithmetic::field_to_ring_element_into;
    while num_vars > 0 {
        num_vars -= 1;

        let mut poly_over_field = Polynomial::<QuadraticExtension>::new(0);
        field_combiner.univariate_polynomial_into(&mut poly_over_field);
        hash_wrapper.update_with_quadratic_extension_slice(&poly_over_field.coefficients);

        let mut f = QuadraticExtension::zero();
        hash_wrapper.sample_field_element_into(&mut f);
        let mut r = RingElement::zero(Representation::IncompleteNTT);
        field_to_ring_element_into(&mut r, &f);
        r.from_homogenized_field_extensions_to_incomplete_ntt();

        for leaf in &asm.leaves {
            leaf.partial_evaluate(&r);
        }
        evaluation_points.push(r);
        polys.push(poly_over_field);
    }
    #[cfg(feature = "profile-sumcheck")]
    crate::protocol::sumcheck_utils::profile::print_and_reset("entry");

    let witness_eval = asm
        .witness_pool
        .first_cell(&FULL_WITNESS_KEY)
        .unwrap()
        .borrow()
        .final_evaluations()
        .clone();
    let conj_witness_eval = if asm.conj_pool.first_cell(&FULL_WITNESS_KEY).is_none() {
        // no conjugate oracle was used: evaluate one on demand at the final point
        let mut ls = LinearSumcheck::new(n);
        ls.load_from(asm.conjugated);
        for r in &evaluation_points {
            ls.partial_evaluate(r);
        }
        ls.final_evaluations().clone()
    } else {
        asm.conj_pool
            .first_cell(&FULL_WITNESS_KEY)
            .unwrap()
            .borrow()
            .final_evaluations()
            .clone()
    };

    hash_wrapper.update_with_ring_element(&witness_eval);
    hash_wrapper.update_with_ring_element(&conj_witness_eval);

    evaluation_points.reverse();

    let inputs = chain_inputs(
        &evaluation_points,
        witness.width,
        &witness_eval,
        &conj_witness_eval,
    );

    (
        InitialSumcheckProof {
            polys,
            witness_eval,
            conj_witness_eval,
        },
        inputs,
    )
}

/// Height and width of the committed witness matrix, as the verifier knows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WitnessShape {
    pub height: usize,
    pub width: usize,
}

impl WitnessShape {
    pub fn new(height: usize, width: usize) -> WitnessShape {
        WitnessShape { height, width }
    }
}

impl From<(usize, usize)> for WitnessShape {
    fn from((height, width): (usize, usize)) -> WitnessShape {
        WitnessShape { height, width }
    }
}

impl<T> From<&VerticallyAlignedMatrix<T>> for WitnessShape {
    fn from(m: &VerticallyAlignedMatrix<T>) -> WitnessShape {
        WitnessShape { height: m.height, width: m.width }
    }
}

/// The verifier's side of [`prove_claims`]: replays the batching, checks
/// every sumcheck round, evaluates all public factors at the final point,
/// and returns the evaluation claims the chain must prove. Claim values are
/// used as given: witness-dependent values travel in the caller's envelope,
/// and any structural check on them (a zero constant coefficient, say) is
/// the caller's, on this side. Panics on any mismatch; `claims` must be
/// rebuilt exactly as the prover built them (same transcript state).
pub fn verify_claims(
    shape: impl Into<WitnessShape>,
    claims: &[SnarkClaim],
    proof: &InitialSumcheckProof,
    hash_wrapper: &mut HashWrapper,
) -> ChainInputs {
    let shape = shape.into();
    let n = shape.height * shape.width;
    assert!(n.is_power_of_two());
    let total_vars = n.ilog2() as usize;
    let canon: Vec<SnarkClaim> = claims.iter().map(canonicalize).collect();
    let claims = &canon[..];
    validate_claims(claims, total_vars);
    assert_eq!(proof.polys.len(), total_vars);

    // Mirror of the prover's gadget tree over the claimed evaluations
    let witness_eval_cell = ElephantCell::new(FakeEvaluationLinearSumcheck::new());
    witness_eval_cell
        .borrow_mut()
        .set_result(proof.witness_eval.clone());
    let conj_eval_cell = ElephantCell::new(FakeEvaluationLinearSumcheck::new());
    conj_eval_cell
        .borrow_mut()
        .set_result(proof.conj_witness_eval.clone());

    let asm = VerifierAssembler {
        witness_eval: witness_eval_cell as _,
        conj_eval: conj_eval_cell as _,
        n,
        total_vars,
    };

    let mut outputs: Vec<EvalCell> = vec![];
    for claim in claims {
        outputs.push(asm.build(&claim.expr));
    }

    let effective_values: Vec<RingElement> =
        claims.iter().map(|claim| claim.value.clone()).collect();
    for value in &effective_values {
        hash_wrapper.update_with_ring_element(value);
    }
    let mut combination = vec![RingElement::zero(Representation::IncompleteNTT); outputs.len()];
    hash_wrapper.sample_ring_element_vec_into(&mut combination);

    let mut combination_to_field = RingElement::zero(Representation::IncompleteNTT);
    hash_wrapper.sample_ring_element_into(&mut combination_to_field);
    combination_to_field.from_incomplete_ntt_to_homogenized_field_extensions();
    let qe = combination_to_field.split_into_quadratic_extensions();

    // batched claim = sum_i gamma_i * value_i, mapped through Phi
    let mut batched_claim = RingElement::zero(Representation::IncompleteNTT);
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    for (value, gamma) in effective_values.iter().zip(combination.iter()) {
        temp *= (value, gamma);
        batched_claim += &temp;
    }

    let mut batched_claim_over_field = {
        let mut t = batched_claim.clone();
        t.from_incomplete_ntt_to_homogenized_field_extensions();
        let mut split = t.split_into_quadratic_extensions();
        let mut result = QuadraticExtension::zero();
        for i in 0..crate::common::config::HALF_DEGREE {
            split[i] *= &qe[i];
            result += &split[i];
        }
        result
    };

    let mut combiner_evaluation = CombinerEvaluation::new(outputs);
    combiner_evaluation.load_challenges_from(&combination);
    let mut field_combiner_evaluation =
        RingToFieldCombinerEvaluation::new(ElephantCell::new(combiner_evaluation) as _);
    field_combiner_evaluation.load_challenges_from(qe.clone());

    let mut evaluation_points: Vec<RingElement> = vec![];
    for (round, poly_over_field) in proof.polys.iter().enumerate() {
        hash_wrapper.update_with_quadratic_extension_slice(&poly_over_field.coefficients);

        // The transcript absorbs the full coefficient array; the unused tail
        // must be zero so the prover cannot vary it under one absorption.
        for c in &poly_over_field.coefficients[poly_over_field.num_coefficients..] {
            assert_eq!(c, &QuadraticExtension::zero(), "round polynomial tail nonzero in round {round}");
        }

        assert_eq!(
            poly_over_field.at_zero() + poly_over_field.at_one(),
            batched_claim_over_field,
            "round claim mismatch in sumcheck round {round}"
        );

        let mut f = QuadraticExtension::zero();
        hash_wrapper.sample_field_element_into(&mut f);
        batched_claim_over_field = poly_over_field.at(&f);

        use crate::common::arithmetic::field_to_ring_element_into;
        let mut r = RingElement::zero(Representation::IncompleteNTT);
        field_to_ring_element_into(&mut r, &f);
        r.from_homogenized_field_extensions_to_incomplete_ntt();
        evaluation_points.push(r);
    }

    assert_eq!(
        &batched_claim_over_field,
        field_combiner_evaluation.evaluate_at_ring_point(&evaluation_points),
        "Initial sumcheck final evaluation mismatch"
    );

    hash_wrapper.update_with_ring_element(&proof.witness_eval);
    hash_wrapper.update_with_ring_element(&proof.conj_witness_eval);

    evaluation_points.reverse();

    chain_inputs(
        &evaluation_points,
        shape.width,
        &proof.witness_eval,
        &proof.conj_witness_eval,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{init_common, sampling::sample_random_short_vector};
    use crate::protocol::snark::{eq, table, Region};

    fn inner_product_direct(a: &[RingElement], b: &[RingElement]) -> RingElement {
        let mut acc = RingElement::zero(Representation::IncompleteNTT);
        let mut temp = RingElement::zero(Representation::IncompleteNTT);
        for (x, y) in a.iter().zip(b.iter()) {
            temp *= (x, y);
            acc += &temp;
        }
        acc
    }

    fn toy_setup() -> (VerticallyAlignedMatrix<RingElement>, Vec<SnarkClaim>) {
        let height = 64;
        let width = 4;
        let n = height * width;
        let witness = VerticallyAlignedMatrix {
            height,
            width,
            used_cols: width,
            data: sample_random_short_vector(n, 100, Representation::IncompleteNTT),
        };

        // <a, w> = t
        let a = sample_random_short_vector(n, 50, Representation::IncompleteNTT);
        let t1 = inner_product_direct(&a, &witness.data);
        let claim1 = SnarkClaim {
            expr: table(a.clone()) * ClaimExpr::witness(),
            value: t1,
        };

        // sum_z b(z) * w(z)^2 = t (degree 2 in the witness oracle)
        let b = sample_random_short_vector(n, 10, Representation::IncompleteNTT);
        let mut sq = witness.data.clone();
        for (s, w) in sq.iter_mut().zip(witness.data.iter()) {
            let w2 = w.clone();
            *s *= (w, &w2);
        }
        let t2 = inner_product_direct(&b, &sq);
        let claim2 = SnarkClaim {
            expr: table(b) * ClaimExpr::witness() * ClaimExpr::witness(),
            value: t2,
        };

        // segment sum via selector: indices [n/4, n/2)
        let prefix = Prefix {
            prefix: 1,
            length: 2,
        };
        let mut t3 = RingElement::zero(Representation::IncompleteNTT);
        for w in &witness.data[n / 4..n / 2] {
            t3 += w;
        }
        let claim3 = SnarkClaim {
            expr: ClaimExpr::segment(prefix),
            value: t3,
        };

        // scaled difference of expressions: 7*<a, w> - <a, conj(w)> = t
        let seven = RingElement::constant(7, Representation::IncompleteNTT);
        let conj: Vec<RingElement> = witness.data.iter().map(|w| w.conjugate()).collect();
        let mut t4 = inner_product_direct(&a, &witness.data);
        t4 *= &seven;
        t4 -= &inner_product_direct(&a, &conj);
        let claim4 = SnarkClaim {
            expr: (table(a.clone()) * ClaimExpr::witness()).scale(&seven)
                - (table(a) * ClaimExpr::conj_witness()),
            value: t4,
        };

        (witness, vec![claim1, claim2, claim3, claim4])
    }

    #[test]
    fn test_initial_claims_roundtrip() {
        init_common();
        let (witness, claims) = toy_setup();

        let mut hw_prover = HashWrapper::new();
        let (proof, chain_prover) = prove_claims(&witness, &claims, &mut hw_prover);

        let mut hw_verifier = HashWrapper::new();
        let chain_verifier = verify_claims((witness.height, witness.width),
            &claims,
            &proof,
            &mut hw_verifier,
        );

        assert_eq!(chain_prover.claims, chain_verifier.claims);

        // the emitted chain claims must match direct witness evaluation:
        // t_0 = <l_0^T W, r_0> for l_0 = tensor(c_inner), r_0 = tensor(c_outer)
        let direct = crate::protocol::open::claim(
            &witness,
            &chain_prover.evaluation_points_inner[0],
            &chain_prover.evaluation_points_outer[0],
        );
        assert_eq!(direct, chain_prover.claims[0]);

        let direct_conj = crate::protocol::open::claim(
            &witness,
            &chain_prover.evaluation_points_inner[1],
            &chain_prover.evaluation_points_outer[1],
        );
        assert_eq!(direct_conj, chain_prover.claims[1]);
    }

    #[test]
    fn test_field_tensor_roundtrip() {
        init_common();
        let height = 64;
        let width = 4;
        let n = height * width;
        let witness = VerticallyAlignedMatrix {
            height,
            width,
            used_cols: width,
            data: sample_random_short_vector(n, 100, Representation::IncompleteNTT),
        };
        let quarter = n / 4;

        let layers: Vec<QuadraticExtension> = (0..6)
            .map(|i| QuadraticExtension {
                coeffs: [7 + 3 * i as u64, 11 + 5 * i as u64],
            })
            .collect();
        let dense1 = expand_field_tensor(&layers);
        let value1 = inner_product_direct(&dense1, &witness.data[quarter..2 * quarter]);
        let make_claim1 = || SnarkClaim {
            expr: eq(layers.clone()).on(Region::new(n / 4, n / 4, n))
                * ClaimExpr::segment(Prefix {
                    prefix: 1,
                    length: 2,
                }),
            value: value1.clone(),
        };

        let mut hw_p = HashWrapper::new();
        let claims_p = vec![make_claim1()];
        let (proof, chain_p) = prove_claims(&witness, &claims_p, &mut hw_p);

        let mut hw_v = HashWrapper::new();
        let claims_v = vec![make_claim1()];
        let chain_v =
            verify_claims((witness.height, witness.width), &claims_v, &proof, &mut hw_v);
        assert_eq!(chain_p.claims, chain_v.claims);

        for j in 0..chain_p.claims.len() {
            let direct = crate::protocol::open::claim(
                &witness,
                &chain_p.evaluation_points_inner[j],
                &chain_p.evaluation_points_outer[j],
            );
            assert_eq!(direct, chain_p.claims[j], "opening {}", j);
        }
    }

    #[test]
    fn test_disjoint_public_factors_roundtrip() {
        // A localized linear claim whose weight is a node eq-tensor on one
        // block of variables times an arbitrary table on a disjoint block:
        // tensor * dense * segment lowers to four factors (tensor, dense,
        // selector, Witness) yet every variable is multiplied by at most two,
        // so the round polynomials stay degree two - the per-variable cap.
        init_common();
        let height = 64;
        let width = 4;
        let n = height * width; // total_vars = 8
        let witness = VerticallyAlignedMatrix {
            height,
            width,
            used_cols: width,
            data: sample_random_short_vector(n, 100, Representation::IncompleteNTT),
        };

        let prefix = Prefix {
            prefix: 2,
            length: 2,
        };
        let (mb, in_bits) = (2usize, 4usize); // mb + in_bits = 6 = free variables
        let start = prefix.prefix << (n.ilog2() as usize - prefix.length);

        let alpha: Vec<QuadraticExtension> = (0..mb)
            .map(|i| QuadraticExtension {
                coeffs: [7 + 3 * i as u64, 11 + 5 * i as u64],
            })
            .collect();
        let k = sample_random_short_vector(1 << in_bits, 50, Representation::IncompleteNTT);

        // value = sum_{m,i} eq(alpha, m) * k[i] * w[start + m*2^in_bits + i]
        let mut value = RingElement::zero(Representation::IncompleteNTT);
        let mut prod = RingElement::zero(Representation::IncompleteNTT);
        for m in 0..(1usize << mb) {
            let em = embed_qe(&tensor_at(&alpha, m));
            for i in 0..(1usize << in_bits) {
                prod *= (&k[i], &witness.data[start + (m << in_bits) + i]);
                let mut weighted = em.clone();
                weighted *= &prod;
                value += &weighted;
            }
        }

        let region = Region::new(start, n >> prefix.length, n);
        let (node, slot) = region.vars().split_at(mb);
        let make_claim = || SnarkClaim {
            expr: eq(alpha.clone()).on(node)
                * table(k.clone()).on(slot)
                * ClaimExpr::segment(prefix.clone()),
            value: value.clone(),
        };

        let mut hw_p = HashWrapper::new();
        let claims_p = vec![make_claim()];
        let (proof, chain_p) = prove_claims(&witness, &claims_p, &mut hw_p);

        let mut hw_v = HashWrapper::new();
        let claims_v = vec![make_claim()];
        let chain_v =
            verify_claims((witness.height, witness.width), &claims_v, &proof, &mut hw_v);
        assert_eq!(chain_p.claims, chain_v.claims);

        for j in 0..chain_p.claims.len() {
            let direct = crate::protocol::open::claim(
                &witness,
                &chain_p.evaluation_points_inner[j],
                &chain_p.evaluation_points_outer[j],
            );
            assert_eq!(direct, chain_p.claims[j], "opening {}", j);
        }
    }

    #[test]
    fn test_field_dense_and_placed_ring_tensor_roundtrip() {
        init_common();
        let height = 64;
        let width = 4;
        let n = height * width;
        let witness = VerticallyAlignedMatrix {
            height,
            width,
            used_cols: width,
            data: sample_random_short_vector(n, 100, Representation::IncompleteNTT),
        };
        let quarter = n / 4;

        // field dense table over block 1
        let tab: Vec<QuadraticExtension> = (0..quarter)
            .map(|i| QuadraticExtension {
                coeffs: [3 + i as u64, 5 + 2 * i as u64],
            })
            .collect();
        let table_ring: Vec<RingElement> = tab.iter().map(embed_qe).collect();
        let value_a = inner_product_direct(&table_ring, &witness.data[quarter..2 * quarter]);

        // ring eq-tensor over block 2
        let layers: Vec<RingElement> = (0..quarter.ilog2() as usize)
            .map(|i| RingElement::constant(2 + i as u64, Representation::IncompleteNTT))
            .collect();
        let expanded = PreprocessedRow::from_layers(&layers).preprocessed_row;
        let value_b = inner_product_direct(&expanded, &witness.data[2 * quarter..3 * quarter]);

        let make_claims = || {
            vec![
                SnarkClaim {
                    expr: table(tab.clone()).on(Region::new(quarter, quarter, n))
                        * ClaimExpr::segment(Prefix {
                            prefix: 1,
                            length: 2,
                        }),
                    value: value_a.clone(),
                },
                SnarkClaim {
                    expr: eq(layers.clone()).on(Region::new(2 * quarter, quarter, n))
                        * ClaimExpr::segment(Prefix {
                            prefix: 2,
                            length: 2,
                        }),
                    value: value_b.clone(),
                },
            ]
        };

        let mut hw_p = HashWrapper::new();
        let (proof, chain_p) = prove_claims(&witness, &make_claims(), &mut hw_p);
        let mut hw_v = HashWrapper::new();
        let chain_v =
            verify_claims((witness.height, witness.width), &make_claims(), &proof, &mut hw_v);
        assert_eq!(chain_p.claims, chain_v.claims);

        for j in 0..chain_p.claims.len() {
            let direct = crate::protocol::open::claim(
                &witness,
                &chain_p.evaluation_points_inner[j],
                &chain_p.evaluation_points_outer[j],
            );
            assert_eq!(direct, chain_p.claims[j], "opening {}", j);
        }
    }

    fn binariness_setup(tamper: bool) -> (VerticallyAlignedMatrix<RingElement>, Vec<SnarkClaim>) {
        let height = 64;
        let width = 4;
        let n = height * width;
        let mut data = sample_random_short_vector(n, 100, Representation::IncompleteNTT);
        let quarter = n / 4;
        for (i, w) in data[quarter..2 * quarter].iter_mut().enumerate() {
            let mut bits = RingElement::zero(Representation::EvenOddCoefficients);
            for (c, b) in bits.v.iter_mut().enumerate() {
                *b = ((i * 31 + c * 7 + 3) % 5 < 2) as u64;
            }
            if tamper && i == 5 {
                bits.v[17] = 2;
            }
            bits.from_even_odd_coefficients_to_incomplete_ntt_representation();
            *w = bits;
        }
        let witness = VerticallyAlignedMatrix {
            height,
            width,
            used_cols: width,
            data,
        };

        let ones = {
            let mut e = RingElement::zero(Representation::EvenOddCoefficients);
            for c in e.v.iter_mut() {
                *c = 1;
            }
            e.from_even_odd_coefficients_to_incomplete_ntt_representation();
            e
        };
        let p = Prefix {
            prefix: 1,
            length: 2,
        };
        let mut value = RingElement::zero(Representation::IncompleteNTT);
        let mut tmp = RingElement::zero(Representation::IncompleteNTT);
        for w in &witness.data[quarter..2 * quarter] {
            tmp *= (w, &w.conjugate());
            value += &tmp;
            tmp *= (w, &ones.conjugate());
            value -= &tmp;
        }
        // sum_seg w*conj(w) - ones_conj*w = 0 iff each coefficient is binary
        let claims = vec![SnarkClaim {
            expr: (ClaimExpr::segment(p.clone()) * ClaimExpr::conj_segment(p.clone()))
                - (table(vec![ones.conjugate(); quarter]).on(Region::new(quarter, quarter, n))
                    * ClaimExpr::segment(p)),
            value,
        }];
        (witness, claims)
    }

    #[test]
    fn test_conj_segment_ct_zero_roundtrip() {
        init_common();
        // Witness-dependent claim values travel in the caller's envelope:
        // here the verifier receives the prover's value, performs the
        // constant-coefficient check itself, and uses the value as is.
        let (witness, claims) = binariness_setup(false);
        let mut hw_p = HashWrapper::new();
        let (proof, chain_p) = prove_claims(&witness, &claims, &mut hw_p);

        let (witness_v, mut claims_v) = binariness_setup(false);
        let _ = witness_v;
        let shipped = claims[0].value.clone();
        let mut ct = shipped.clone();
        ct.to_representation(Representation::Coefficients);
        assert_eq!(ct.v[0], 0, "claim constant term nonzero");
        claims_v[0].value = shipped;
        let mut hw_v = HashWrapper::new();
        let chain_v =
            verify_claims((witness.height, witness.width), &claims_v, &proof, &mut hw_v);
        assert_eq!(chain_p.claims, chain_v.claims);
        for j in 0..chain_p.claims.len() {
            let direct = crate::protocol::open::claim(
                &witness,
                &chain_p.evaluation_points_inner[j],
                &chain_p.evaluation_points_outer[j],
            );
            assert_eq!(direct, chain_p.claims[j], "opening {}", j);
        }
    }

    #[test]
    #[should_panic(expected = "claim constant term nonzero")]
    fn test_conj_segment_nonbinary_rejected() {
        init_common();
        let (witness, claims) = binariness_setup(true);
        let mut hw_p = HashWrapper::new();
        let _ = prove_claims(&witness, &claims, &mut hw_p);

        let shipped = claims[0].value.clone();
        let mut ct = shipped;
        ct.to_representation(Representation::Coefficients);
        assert_eq!(ct.v[0], 0, "claim constant term nonzero");
    }

    #[test]
    #[should_panic(expected = "round claim mismatch")]
    fn test_initial_claims_wrong_value_rejected() {
        init_common();
        let (witness, mut claims) = toy_setup();

        let mut hw_prover = HashWrapper::new();
        let (proof, _) = prove_claims(&witness, &claims, &mut hw_prover);

        claims[0].value += &RingElement::constant(1, Representation::IncompleteNTT);
        let mut hw_verifier = HashWrapper::new();
        verify_claims((witness.height, witness.width),
            &claims,
            &proof,
            &mut hw_verifier,
        );
    }

    /// Prove and verify `make()` (rebuilt for each side), then check the
    /// emitted chain claims agree and match direct witness openings.
    fn roundtrip(
        witness: &VerticallyAlignedMatrix<RingElement>,
        make: impl Fn() -> Vec<SnarkClaim>,
    ) {
        let mut hw_p = HashWrapper::new();
        let (proof, chain_p) = prove_claims(witness, &make(), &mut hw_p);
        let mut hw_v = HashWrapper::new();
        let chain_v =
            verify_claims((witness.height, witness.width), &make(), &proof, &mut hw_v);
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

    /// 64x4 witness with two public weight vectors for `<a, w>` and `<b, w>`.
    fn combine_setup() -> (
        VerticallyAlignedMatrix<RingElement>,
        Vec<RingElement>,
        Vec<RingElement>,
    ) {
        let height = 64;
        let width = 4;
        let n = height * width;
        let witness = VerticallyAlignedMatrix {
            height,
            width,
            used_cols: width,
            data: sample_random_short_vector(n, 100, Representation::IncompleteNTT),
        };
        let a = sample_random_short_vector(n, 50, Representation::IncompleteNTT);
        let b = sample_random_short_vector(n, 30, Representation::IncompleteNTT);
        (witness, a, b)
    }

    fn dot(weights: &[RingElement]) -> ClaimExpr {
        table(weights.to_vec()) * ClaimExpr::witness()
    }

    #[test]
    fn test_add_exprs_roundtrip() {
        init_common();
        let (witness, a, b) = combine_setup();
        let t_a = inner_product_direct(&a, &witness.data);
        let t_b = inner_product_direct(&b, &witness.data);
        let mut value = t_a;
        value += &t_b;
        roundtrip(&witness, || {
            vec![SnarkClaim {
                expr: dot(&a) + dot(&b),
                value: value.clone(),
            }]
        });
    }

    #[test]
    fn test_sub_exprs_roundtrip() {
        init_common();
        let (witness, a, b) = combine_setup();
        let t_a = inner_product_direct(&a, &witness.data);
        let t_b = inner_product_direct(&b, &witness.data);
        let mut value = t_a;
        value -= &t_b;
        roundtrip(&witness, || {
            vec![SnarkClaim {
                expr: dot(&a) - dot(&b),
                value: value.clone(),
            }]
        });
    }

    #[test]
    fn test_scale_then_sub_exprs_roundtrip() {
        init_common();
        let (witness, a, b) = combine_setup();
        let t_a = inner_product_direct(&a, &witness.data);
        let t_b = inner_product_direct(&b, &witness.data);
        let seven = RingElement::constant(7, Representation::IncompleteNTT);
        let mut value = t_a;
        value *= &seven;
        value -= &t_b;
        roundtrip(&witness, || {
            vec![SnarkClaim {
                expr: dot(&a).scale(&seven) - dot(&b),
                value: value.clone(),
            }]
        });
    }

    #[test]
    fn test_sub_self_is_zero_expr() {
        init_common();
        let (witness, a, _b) = combine_setup();
        // <a, w> - <a, w> = 0: a Diff over the same expression
        roundtrip(&witness, || {
            vec![SnarkClaim {
                expr: dot(&a) - dot(&a),
                value: RingElement::zero(Representation::IncompleteNTT),
            }]
        });
    }

    #[test]
    #[should_panic(expected = "round claim mismatch")]
    fn test_add_exprs_wrong_value_rejected() {
        init_common();
        let (witness, a, b) = combine_setup();
        let t_a = inner_product_direct(&a, &witness.data);
        let t_b = inner_product_direct(&b, &witness.data);
        let mut value = t_a;
        value += &t_b;

        let mut hw_p = HashWrapper::new();
        let combined = vec![SnarkClaim {
            expr: dot(&a) + dot(&b),
            value: value.clone(),
        }];
        let (proof, _) = prove_claims(&witness, &combined, &mut hw_p);

        value += &RingElement::constant(1, Representation::IncompleteNTT);
        let claims_v = vec![SnarkClaim {
            expr: dot(&a) + dot(&b),
            value,
        }];
        let mut hw_v = HashWrapper::new();
        verify_claims((witness.height, witness.width), &claims_v, &proof, &mut hw_v);
    }
}
