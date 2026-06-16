use crate::{
    common::{
        hash::HashWrapper,
        matrix::{new_vec_zero_preallocated, VerticallyAlignedMatrix},
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

impl PublicFactor {
    fn full(weights: Weights) -> Self {
        PublicFactor {
            prefix_len: 0,
            suffix_len: 0,
            weights,
        }
    }
    /// eq-tensor with ring layers, over the full cube.
    pub fn tensor_ring(layers: impl Into<Arc<Vec<RingElement>>>) -> Self {
        Self::full(Weights::Tensor(Coeffs::Ring(layers.into())))
    }
    /// eq-tensor with field layers, over the full cube.
    pub fn tensor_field(layers: impl Into<Arc<Vec<QuadraticExtension>>>) -> Self {
        Self::full(Weights::Tensor(Coeffs::Field(layers.into())))
    }
    /// Arbitrary ring table over the full cube.
    pub fn dense_ring(table: impl Into<Arc<Vec<RingElement>>>) -> Self {
        Self::full(Weights::Dense(Coeffs::Ring(table.into())))
    }
    /// Arbitrary field table over the full cube.
    pub fn dense_field(table: impl Into<Arc<Vec<QuadraticExtension>>>) -> Self {
        Self::full(Weights::Dense(Coeffs::Field(table.into())))
    }
    /// `eq(bits, .)` over the leading `length` variables.
    pub fn selector(bits: usize, length: usize) -> Self {
        Self::full(Weights::Selector { bits, length })
    }
    /// Place the weights over the middle variables: constant on the top
    /// `prefix_len` and bottom `suffix_len` variables.
    pub fn over_middle(mut self, prefix_len: usize, suffix_len: usize) -> Self {
        self.prefix_len = prefix_len;
        self.suffix_len = suffix_len;
        self
    }
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

/// `coefficient * prod(factors)`, summed over the cube (a segment factor
/// restricts the term's sum to its block). The coefficient is a full ring
/// element: batching scalars and fixed public elements ride here.
pub struct ClaimTerm {
    pub coefficient: RingElement,
    pub factors: Vec<ClaimFactor>,
}

impl ClaimTerm {
    pub fn new(factors: Vec<ClaimFactor>) -> Self {
        ClaimTerm {
            coefficient: RingElement::constant(1, Representation::IncompleteNTT),
            factors,
        }
    }

    pub fn scaled(coefficient: RingElement, factors: Vec<ClaimFactor>) -> Self {
        ClaimTerm {
            coefficient,
            factors,
        }
    }
}

/// One functional-sumcheck claim (paper: f_sc with
/// `sum_z f_sc(MLE[w], MLE[conj w])(z) = value`).
pub struct SnarkClaim {
    pub terms: Vec<ClaimTerm>,
    pub value: RingElement,
}

pub struct InitialSumcheckProof {
    pub polys: Vec<Polynomial<QuadraticExtension>>,
    /// `z_0 = MLE[vec(W)](c)`
    pub witness_eval: RingElement,
    /// `z_1 = MLE[conj(vec(W))](c)`
    pub conj_witness_eval: RingElement,
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

/// Verifier-side wrapper scaling an inner evaluation by a public constant.
struct ScaledEvaluation {
    inner: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
    scale: RingElement,
    result: RingElement,
}

impl EvaluationSumcheckData for ScaledEvaluation {
    type Element = RingElement;

    fn evaluate(&mut self, point: &Vec<RingElement>) -> &RingElement {
        self.result.set_from(self.inner.borrow_mut().evaluate(point));
        self.result *= &self.scale;
        &self.result
    }
}

/// Oracle pool keyed by segment; distinct cell per use within one term (a
/// RefCell cannot appear twice in one product), reused across terms.
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

/// The order in which segment prefixes first appear in the claims; both sides
/// derive it from the public claims.
fn fold_product(
    mut factors: Vec<ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>>,
) -> ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>> {
    let mut acc = factors.pop().expect("term must have at least one factor");
    while let Some(f) = factors.pop() {
        acc = ElephantCell::new(ProductSumcheck::new(f, acc));
    }
    acc
}

fn fold_sum(
    mut outputs: Vec<ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>>,
) -> ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>> {
    let mut acc = outputs.pop().expect("claim must have at least one term");
    while let Some(o) = outputs.pop() {
        acc = ElephantCell::new(SumSumcheck::new(o, acc));
    }
    acc
}

fn fold_product_evaluation(
    mut factors: Vec<ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>>,
) -> ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>> {
    let mut acc = factors.pop().expect("term must have at least one factor");
    while let Some(f) = factors.pop() {
        acc = ElephantCell::new(ProductSumcheckEvaluation::new(f, acc));
    }
    acc
}

fn fold_sum_evaluation(
    mut outputs: Vec<ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>>,
) -> ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>> {
    let mut acc = outputs.pop().expect("claim must have at least one term");
    while let Some(o) = outputs.pop() {
        acc = ElephantCell::new(SumSumcheckEvaluation::new(o, acc));
    }
    acc
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
fn lower_claims(claims: &[SnarkClaim]) -> Vec<SnarkClaim> {
    claims
        .iter()
        .map(|claim| SnarkClaim {
            value: claim.value.clone(),
            terms: claim
                .terms
                .iter()
                .map(|term| {
                    let mut factors = Vec::with_capacity(term.factors.len() + 1);
                    for f in &term.factors {
                        match f {
                            ClaimFactor::WitnessSegment(p) => {
                                factors.push(ClaimFactor::Public(PublicFactor::selector(
                                    p.prefix, p.length,
                                )));
                                factors.push(ClaimFactor::Witness);
                            }
                            ClaimFactor::ConjWitnessSegment(p) => {
                                factors.push(ClaimFactor::Public(PublicFactor::selector(
                                    p.prefix, p.length,
                                )));
                                factors.push(ClaimFactor::ConjWitness);
                            }
                            ClaimFactor::Witness => factors.push(ClaimFactor::Witness),
                            ClaimFactor::ConjWitness => factors.push(ClaimFactor::ConjWitness),
                            ClaimFactor::Public(p) => {
                                factors.push(ClaimFactor::Public(p.clone()))
                            }
                        }
                    }
                    // two factors restricted to the same block need only one
                    // selector: on the cube eq is 0/1, so eq^2 = eq there,
                    // and the claim is a statement about the cube sum
                    let mut seen: Vec<(usize, usize)> = vec![];
                    factors.retain(|f| {
                        if let ClaimFactor::Public(PublicFactor {
                            weights: Weights::Selector { bits, length },
                            ..
                        }) = f
                        {
                            let key = (*bits, *length);
                            if seen.contains(&key) {
                                return false;
                            }
                            seen.push(key);
                        }
                        true
                    });
                    ClaimTerm {
                        coefficient: term.coefficient.clone(),
                        factors,
                    }
                })
                .collect(),
        })
        .collect()
}

/// Structural validation of the lowered claims: every prefix must fit the
/// cube, and a term may hold at most three non-constant factors (the round
/// polynomials carry degree three; a segment factor counts twice, for its
/// selector and its vector).
fn validate_claims(claims: &[SnarkClaim], total_vars: usize) {
    assert!(!claims.is_empty(), "no claims");
    for claim in claims {
        assert!(!claim.terms.is_empty(), "claim with no terms");
        for term in &claim.terms {
            assert!(!term.factors.is_empty(), "term with no factors");
            let mut oracles = 0usize;
            for f in &term.factors {
                oracles += 1;
                if let ClaimFactor::Public(PublicFactor {
                    weights: Weights::Selector { bits, length },
                    ..
                }) = f
                {
                    assert!(*length <= total_vars, "prefix length exceeds the cube");
                    assert!(
                        *length == 0 || *bits < (1usize << *length),
                        "prefix value exceeds its declared length"
                    );
                }
            }
            assert!(
                oracles <= 3,
                "a term holds at most three factors (round polynomials carry degree \
                 three; a segment counts as two, its selector and its vector)"
            );
        }
    }
}

pub fn prove_initial_claims(
    witness: &VerticallyAlignedMatrix<RingElement>,
    claims: &[SnarkClaim],
    hash_wrapper: &mut HashWrapper,
) -> (InitialSumcheckProof, ChainInputs) {
    let n = witness.data.len();
    assert!(n.is_power_of_two());
    let total_vars = n.ilog2() as usize;
    let lowered = lower_claims(claims);
    let claims = &lowered[..];
    validate_claims(claims, total_vars);

    let mut conjugated = new_vec_zero_preallocated(n);
    witness
        .data
        .iter()
        .zip(conjugated.iter_mut())
        .for_each(|(orig, conj)| orig.conjugate_into(conj));

    let mut witness_pool = OraclePool::new();
    let mut conj_pool = OraclePool::new();
    let mut public_pool = OraclePool::new();
    let mut leaves: Vec<LeafCell> = vec![];

    let make_full = |data: &[RingElement]| {
        let mut ls = LinearSumcheck::new(data.len());
        ls.load_from(data);
        ls
    };

    let mut outputs: Vec<ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>> = vec![];
    for claim in claims {
        let mut term_cells = vec![];
        for term in &claim.terms {
            witness_pool.reset_term();
            conj_pool.reset_term();
            public_pool.reset_term();
            let mut factors: Vec<ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>> =
                vec![];
            // the term coefficient is folded into the first scalable public factor
            let mut pending_scale = (!is_unit(&term.coefficient)).then(|| term.coefficient.clone());
            for factor in &term.factors {
                match factor {
                    ClaimFactor::Witness => {
                        let cell = witness_pool.next(FULL_WITNESS_KEY, || make_full(&witness.data));
                        factors.push(cell.clone() as _);
                    }
                    ClaimFactor::ConjWitness => {
                        let cell = conj_pool.next(FULL_WITNESS_KEY, || make_full(&conjugated));
                        factors.push(cell.clone() as _);
                    }
                    ClaimFactor::WitnessSegment(_) | ClaimFactor::ConjWitnessSegment(_) => {
                        unreachable!("segments are lowered before assembly")
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
                        let middle_len = n >> (prefix_len + suffix_len);
                        // Pool a ring middle-table leaf, keyed by its Arc identity.
                        macro_rules! pooled {
                            ($ptr:expr, $middle:expr) => {{
                                let middle = $middle;
                                let key = ($ptr, prefix_len, suffix_len, false);
                                let cell = public_pool.next(key, || {
                                    let mut ls = LinearSumcheck::new_with_prefixed_sufixed_data(
                                        middle.len(),
                                        prefix_len,
                                        suffix_len,
                                    );
                                    ls.load_from(&middle);
                                    ls
                                });
                                factors.push(cell.clone() as _);
                                continue;
                            }};
                        }
                        let mut data = match weights {
                            Weights::Selector { bits, length } => {
                                let cell =
                                    ElephantCell::new(SelectorEq::new(*bits, *length, total_vars));
                                leaves.push(LeafCell::Selector(cell.clone()));
                                factors.push(cell as _);
                                continue;
                            }
                            Weights::Tensor(Coeffs::Ring(layers)) if !placed => {
                                assert_eq!(layers.len(), total_vars);
                                PreprocessedRow::from_layers(&layers[..]).preprocessed_row
                            }
                            Weights::Dense(Coeffs::Ring(v)) if !placed => {
                                assert_eq!(v.len(), n);
                                (**v).clone()
                            }
                            Weights::Tensor(Coeffs::Ring(layers)) => {
                                assert_eq!(1usize << layers.len(), middle_len);
                                pooled!(
                                    Arc::as_ptr(layers) as *const () as usize,
                                    PreprocessedRow::from_layers(&layers[..]).preprocessed_row
                                );
                            }
                            Weights::Tensor(Coeffs::Field(layers)) => {
                                assert_eq!(1usize << layers.len(), middle_len);
                                pooled!(
                                    Arc::as_ptr(layers) as *const () as usize,
                                    expand_field_tensor(&layers[..])
                                );
                            }
                            Weights::Dense(Coeffs::Ring(v)) => {
                                assert_eq!(v.len(), middle_len);
                                pooled!(Arc::as_ptr(v) as *const () as usize, (**v).clone());
                            }
                            Weights::Dense(Coeffs::Field(v)) => {
                                assert_eq!(v.len(), middle_len);
                                pooled!(
                                    Arc::as_ptr(v) as *const () as usize,
                                    v.iter().map(embed_qe).collect::<Vec<_>>()
                                );
                            }
                        };
                        if let Some(scale) = pending_scale.take() {
                            for d in data.iter_mut() {
                                *d *= &scale;
                            }
                        }
                        let mut ls = LinearSumcheck::new(n);
                        ls.load_from(&data);
                        let cell = ElephantCell::new(ls);
                        leaves.push(LeafCell::Linear(cell.clone()));
                        factors.push(cell as _);
                    }
                }
            }
            if let Some(scale) = pending_scale.take() {
                // constant factor: one element repeated across all variables
                let mut ls = LinearSumcheck::new_with_prefixed_sufixed_data(1, total_vars, 0);
                ls.load_from(std::slice::from_ref(&scale));
                let cell = ElephantCell::new(ls);
                leaves.push(LeafCell::Linear(cell.clone()));
                factors.push(cell as _);
            }
            term_cells.push(fold_product(factors));
        }
        outputs.push(fold_sum(term_cells));
    }

    // ensure the full-witness oracle exists: z_0 always seeds the chain
    if witness_pool.first_cell(&FULL_WITNESS_KEY).is_none() {
        let _ = witness_pool.next(FULL_WITNESS_KEY, || make_full(&witness.data));
    }
    for cell in witness_pool.all_cells() {
        leaves.push(LeafCell::Linear(cell.clone()));
    }
    for cell in conj_pool.all_cells() {
        leaves.push(LeafCell::Linear(cell.clone()));
    }
    for cell in public_pool.all_cells() {
        leaves.push(LeafCell::Linear(cell.clone()));
    }

    // Bind the claim values, then sample batching challenges
    for claim in claims {
        hash_wrapper.update_with_ring_element(&claim.value);
    }
    let mut combination = new_vec_zero_preallocated(outputs.len());
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

        for leaf in &leaves {
            leaf.partial_evaluate(&r);
        }
        evaluation_points.push(r);
        polys.push(poly_over_field);
    }
    #[cfg(feature = "profile-sumcheck")]
    crate::protocol::sumcheck_utils::profile::print_and_reset("entry");


    let witness_eval = witness_pool
        .first_cell(&FULL_WITNESS_KEY)
        .unwrap()
        .borrow()
        .final_evaluations()
        .clone();
    let conj_witness_eval = if conj_pool.first_cell(&FULL_WITNESS_KEY).is_none() {
        // derive without a dedicated oracle: MLE[conj w](c) = conj(MLE[w](conj c)),
        // but we have no second run; instead evaluate by loading on demand.
        let mut ls = LinearSumcheck::new(n);
        ls.load_from(&conjugated);
        for r in &evaluation_points {
            ls.partial_evaluate(r);
        }
        ls.final_evaluations().clone()
    } else {
        conj_pool
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

/// The verifier's side of [`prove_initial_claims`]: replays the batching,
/// checks every sumcheck round, evaluates all public factors at the final
/// point, and returns the evaluation claims the chain must prove. Claim
/// values are used as given: witness-dependent values travel in the
/// caller's envelope, and any structural check on them (a zero constant
/// coefficient, say) is the caller's, on this side. Panics on any
/// mismatch; `claims` must be rebuilt exactly as the prover built them
/// (same transcript state).
pub fn verify_initial_claims(
    witness_height: usize,
    witness_width: usize,
    claims: &[SnarkClaim],
    proof: &InitialSumcheckProof,
    hash_wrapper: &mut HashWrapper,
) -> ChainInputs {
    let n = witness_height * witness_width;
    assert!(n.is_power_of_two());
    let total_vars = n.ilog2() as usize;
    let lowered = lower_claims(claims);
    let claims = &lowered[..];
    validate_claims(claims, total_vars);
    assert_eq!(proof.polys.len(), total_vars);

    // Mirror of the prover's gadget tree over claimed evaluations
    let witness_eval_cell = ElephantCell::new(FakeEvaluationLinearSumcheck::new());
    witness_eval_cell
        .borrow_mut()
        .set_result(proof.witness_eval.clone());
    let conj_eval_cell = ElephantCell::new(FakeEvaluationLinearSumcheck::new());
    conj_eval_cell
        .borrow_mut()
        .set_result(proof.conj_witness_eval.clone());

    let mut outputs: Vec<ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>> = vec![];
    for claim in claims {
        let mut term_cells = vec![];
        for term in &claim.terms {
            let mut factors: Vec<ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>> =
                vec![];
            let mut pending_scale = (!is_unit(&term.coefficient)).then(|| term.coefficient.clone());
            for factor in &term.factors {
                match factor {
                    ClaimFactor::Witness => factors.push(witness_eval_cell.clone() as _),
                    ClaimFactor::ConjWitness => factors.push(conj_eval_cell.clone() as _),
                    ClaimFactor::WitnessSegment(_) | ClaimFactor::ConjWitnessSegment(_) => {
                        unreachable!("segments are lowered before assembly")
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
                        use crate::protocol::sumcheck_utils::linear::{
                            RingToFieldWrapperEvaluation, StructuredRowEvaluationLinearSumcheck,
                        };
                        let inner: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>> =
                            match weights {
                                Weights::Selector { bits, length } => {
                                    factors.push(ElephantCell::new(SelectorEqEvaluation::new(
                                        *bits, *length, total_vars,
                                    )) as _);
                                    continue;
                                }
                                Weights::Tensor(Coeffs::Ring(layers)) if !placed => {
                                    let mut ev = StructuredRowEvaluationLinearSumcheck::new(n);
                                    ev.load_from(StructuredRow {
                                        tensor_layers: (**layers).clone(),
                                    });
                                    ElephantCell::new(ev) as _
                                }
                                Weights::Dense(Coeffs::Ring(v)) if !placed => {
                                    let mut ev = BasicEvaluationLinearSumcheck::new(n);
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
                                    factors.push(ElephantCell::new(ev) as _);
                                    continue;
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
                                    factors.push(ElephantCell::new(RingToFieldWrapperEvaluation::new(
                                        ElephantCell::new(ev) as _,
                                    )) as _);
                                    continue;
                                }
                                Weights::Dense(Coeffs::Ring(v)) => {
                                    let mut ev = BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
                                        v.len(),
                                        prefix_len,
                                        suffix_len,
                                    );
                                    ev.load_from(&v[..]);
                                    factors.push(ElephantCell::new(ev) as _);
                                    continue;
                                }
                                Weights::Dense(Coeffs::Field(v)) => {
                                    let mut ev = BasicEvaluationLinearSumcheck::<QuadraticExtension>::new_with_prefixed_sufixed_data(
                                        v.len(),
                                        prefix_len,
                                        suffix_len,
                                    );
                                    ev.load_from(&v[..]);
                                    factors.push(ElephantCell::new(RingToFieldWrapperEvaluation::new(
                                        ElephantCell::new(ev) as _,
                                    )) as _);
                                    continue;
                                }
                            };
                        if let Some(scale) = pending_scale.take() {
                            factors.push(ElephantCell::new(ScaledEvaluation {
                                inner,
                                scale,
                                result: RingElement::zero(Representation::IncompleteNTT),
                            }) as _);
                        } else {
                            factors.push(inner);
                        }
                    }
                }
            }
            if let Some(scale) = pending_scale.take() {
                let constant = ElephantCell::new(FakeEvaluationLinearSumcheck::new());
                constant.borrow_mut().set_result(scale);
                factors.push(constant as _);
            }
            term_cells.push(fold_product_evaluation(factors));
        }
        outputs.push(fold_sum_evaluation(term_cells));
    }

    let effective_values: Vec<RingElement> =
        claims.iter().map(|claim| claim.value.clone()).collect();
    for value in &effective_values {
        hash_wrapper.update_with_ring_element(value);
    }
    let mut combination = new_vec_zero_preallocated(outputs.len());
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
    for poly_over_field in proof.polys.iter() {
        hash_wrapper.update_with_quadratic_extension_slice(&poly_over_field.coefficients);

        // The transcript absorbs the full coefficient array; the unused tail
        // must be zero so the prover cannot vary it under one absorption.
        for c in &poly_over_field.coefficients[poly_over_field.num_coefficients..] {
            assert_eq!(c, &QuadraticExtension::zero(), "round polynomial tail nonzero");
        }

        assert_eq!(
            poly_over_field.at_zero() + poly_over_field.at_one(),
            batched_claim_over_field,
            "Initial sumcheck round claim mismatch"
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
        witness_width,
        &proof.witness_eval,
        &proof.conj_witness_eval,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{init_common, sampling::sample_random_short_vector};

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
            terms: vec![ClaimTerm::new(vec![
                ClaimFactor::Public(PublicFactor::dense_ring(a.clone())),
                ClaimFactor::Witness,
            ])],
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
            terms: vec![ClaimTerm::new(vec![
                ClaimFactor::Public(PublicFactor::dense_ring(b)),
                ClaimFactor::Witness,
                ClaimFactor::Witness,
            ])],
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
            terms: vec![ClaimTerm::new(vec![
                ClaimFactor::Public(PublicFactor::selector(prefix.prefix, prefix.length)),
                ClaimFactor::Witness,
            ])],
            value: t3,
        };

        // scaled two-term claim: 7*<a, w> - <a, conj(w)> = t
        let seven = RingElement::constant(7, Representation::IncompleteNTT);
        let minus_one = RingElement::constant(crate::common::config::MOD_Q - 1, Representation::IncompleteNTT);
        let conj: Vec<RingElement> = witness.data.iter().map(|w| w.conjugate()).collect();
        let mut t4 = inner_product_direct(&a, &witness.data);
        t4 *= &seven;
        let mut t4b = inner_product_direct(&a, &conj);
        t4b *= &minus_one;
        t4 += &t4b;
        let claim4 = SnarkClaim {
            terms: vec![
                ClaimTerm::scaled(
                    seven,
                    vec![
                        ClaimFactor::Public(PublicFactor::dense_ring(a.clone())),
                        ClaimFactor::Witness,
                    ],
                ),
                ClaimTerm::scaled(
                    minus_one,
                    vec![
                        ClaimFactor::Public(PublicFactor::dense_ring(a)),
                        ClaimFactor::ConjWitness,
                    ],
                ),
            ],
            value: t4,
        };

        (witness, vec![claim1, claim2, claim3, claim4])
    }

    #[test]
    fn test_initial_claims_roundtrip() {
        init_common();
        let (witness, claims) = toy_setup();

        let mut hw_prover = HashWrapper::new();
        let (proof, chain_prover) = prove_initial_claims(&witness, &claims, &mut hw_prover);

        let mut hw_verifier = HashWrapper::new();
        let chain_verifier = verify_initial_claims(
            witness.height,
            witness.width,
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
            terms: vec![ClaimTerm::new(
                vec![
                    ClaimFactor::Public(
                        PublicFactor::tensor_field(layers.clone()).over_middle(2, 0),
                    ),
                    ClaimFactor::WitnessSegment(Prefix {
                        prefix: 1,
                        length: 2,
                    }),
                ],
            )],
            value: value1.clone(),
        };

        let mut hw_p = HashWrapper::new();
        let claims_p = vec![make_claim1()];
        let (proof, chain_p) = prove_initial_claims(&witness, &claims_p, &mut hw_p);

        let mut hw_v = HashWrapper::new();
        let claims_v = vec![make_claim1()];
        let chain_v =
            verify_initial_claims(witness.height, witness.width, &claims_v, &proof, &mut hw_v);
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
        let table: Vec<QuadraticExtension> = (0..quarter)
            .map(|i| QuadraticExtension {
                coeffs: [3 + i as u64, 5 + 2 * i as u64],
            })
            .collect();
        let table_ring: Vec<RingElement> = table.iter().map(embed_qe).collect();
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
                    terms: vec![ClaimTerm::new(vec![
                        ClaimFactor::Public(
                            PublicFactor::dense_field(table.clone()).over_middle(2, 0),
                        ),
                        ClaimFactor::WitnessSegment(Prefix {
                            prefix: 1,
                            length: 2,
                        }),
                    ])],
                    value: value_a.clone(),
                },
                SnarkClaim {
                    terms: vec![ClaimTerm::new(vec![
                        ClaimFactor::Public(
                            PublicFactor::tensor_ring(layers.clone()).over_middle(2, 0),
                        ),
                        ClaimFactor::WitnessSegment(Prefix {
                            prefix: 2,
                            length: 2,
                        }),
                    ])],
                    value: value_b.clone(),
                },
            ]
        };

        let mut hw_p = HashWrapper::new();
        let (proof, chain_p) = prove_initial_claims(&witness, &make_claims(), &mut hw_p);
        let mut hw_v = HashWrapper::new();
        let chain_v =
            verify_initial_claims(witness.height, witness.width, &make_claims(), &proof, &mut hw_v);
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
        let minus_one = RingElement::constant(
            crate::common::config::MOD_Q - 1,
            Representation::IncompleteNTT,
        );
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
            let mut neg = tmp.clone();
            neg *= &RingElement::constant(crate::common::config::MOD_Q - 1, Representation::IncompleteNTT);
            value += &neg;
        }
        let claims = vec![SnarkClaim {
            terms: vec![
                ClaimTerm::new(vec![
                    ClaimFactor::WitnessSegment(p.clone()),
                    ClaimFactor::ConjWitnessSegment(p.clone()),
                ]),
                ClaimTerm::scaled(
                    minus_one,
                    vec![
                        ClaimFactor::Public(
                            PublicFactor::dense_ring(vec![ones.conjugate(); quarter])
                                .over_middle(2, 0),
                        ),
                        ClaimFactor::WitnessSegment(p),
                    ],
                ),
            ],
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
        let (proof, chain_p) = prove_initial_claims(&witness, &claims, &mut hw_p);

        let (witness_v, mut claims_v) = binariness_setup(false);
        let _ = witness_v;
        let shipped = claims[0].value.clone();
        let mut ct = shipped.clone();
        ct.to_representation(Representation::Coefficients);
        assert_eq!(ct.v[0], 0, "claim constant term nonzero");
        claims_v[0].value = shipped;
        let mut hw_v = HashWrapper::new();
        let chain_v =
            verify_initial_claims(witness.height, witness.width, &claims_v, &proof, &mut hw_v);
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
        let _ = prove_initial_claims(&witness, &claims, &mut hw_p);

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
        let (proof, _) = prove_initial_claims(&witness, &claims, &mut hw_prover);

        claims[0].value += &RingElement::constant(1, Representation::IncompleteNTT);
        let mut hw_verifier = HashWrapper::new();
        verify_initial_claims(
            witness.height,
            witness.width,
            &claims,
            &proof,
            &mut hw_verifier,
        );
    }
}
