//! SNARK entry round (paper: Pi^lin on Xi^sum_COM): batches user claims
//! sum_z sum_t coeff_t * prod_f factor_f(z) = value over MLE[vec(W)] and its
//! conjugate, reduces them to z_0, z_1 evaluation claims that seed the PCS
//! chain. Claims target the committed vector itself: nonlinear claims do not
//! commute with the PCS path's initial norm decomposition, so SNARK witnesses
//! are committed undecomposed.

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

pub enum PublicFactor {
    /// Tensor row over all sumcheck variables; succinct verifier evaluation.
    Structured(StructuredRow),
    /// eq(prefix, .) on the leading variables.
    Selector(Prefix),
    /// Arbitrary public vector of full hypercube length; verifier evaluation
    /// is linear in the length, intended for tests and small relations.
    Dense(Vec<RingElement>),
    /// Public vector over middle variables (constant in `prefix` high
    /// variables and `suffix` low variables), aligned with WitnessSegment /
    /// WitnessSegmentShifted oracles. Arc-shared: terms reusing one vector
    /// cost a single buffer in the prover.
    DensePrefixed(usize, usize, Arc<Vec<RingElement>>),
}

pub enum ClaimFactor {
    Witness,
    ConjWitness,
    /// MLE of the witness slice under a binary prefix, as its own oracle over
    /// the low variables (constant in the prefix variables). Its final
    /// evaluation becomes an extra chain opening at (bits(prefix), c-tail).
    WitnessSegment(Prefix),
    /// Like WitnessSegment, but the oracle's data variables sit above
    /// `suffix_dummies` low variables in which it is constant; pairs operands
    /// whose index spaces interleave (the opening point takes the matching
    /// middle slice of the challenges).
    WitnessSegmentShifted(Prefix, usize),
    /// Virtual oracle sum_i scale_i * segment_i, all segments sharing one
    /// (length, suffix). No opening of its own: the verifier derives its
    /// evaluation from the component openings, which are forced into the chain.
    WitnessSegmentsScaled(Vec<(Prefix, RingElement)>, usize),
    Public(PublicFactor),
}

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
/// sum_z f_sc(MLE[w], MLE[conj w])(z) = value).
pub struct SnarkClaim {
    pub terms: Vec<ClaimTerm>,
    pub value: RingElement,
}

pub struct InitialSumcheckProof {
    pub polys: Vec<Polynomial<QuadraticExtension>>,
    /// z_0 = MLE[vec(W)](c)
    pub witness_eval: RingElement,
    /// z_1 = MLE[conj(vec(W))](c)
    pub conj_witness_eval: RingElement,
    /// MLE[segment](c-tail) per distinct WitnessSegment prefix, in order of
    /// first appearance in the claims.
    pub segment_evals: Vec<RingElement>,
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
    pools: std::collections::HashMap<(usize, usize, usize), (Vec<ElephantCell<LinearSumcheck<RingElement>>>, usize)>,
}

const FULL_WITNESS_KEY: (usize, usize, usize) = (usize::MAX, usize::MAX, 0);

impl OraclePool {
    fn new() -> Self {
        OraclePool {
            pools: std::collections::HashMap::new(),
        }
    }

    fn next(
        &mut self,
        key: (usize, usize, usize),
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

    fn first_cell(&self, key: &(usize, usize, usize)) -> Option<&ElephantCell<LinearSumcheck<RingElement>>> {
        self.pools.get(key).and_then(|(cells, _)| cells.first())
    }

    fn all_cells(&self) -> impl Iterator<Item = &ElephantCell<LinearSumcheck<RingElement>>> {
        self.pools.values().flat_map(|(cells, _)| cells.iter())
    }
}

/// The order in which segment prefixes first appear in the claims; both sides
/// derive it from the public claims.
fn segment_order(claims: &[SnarkClaim]) -> Vec<(usize, usize, usize)> {
    let mut order = vec![];
    for claim in claims {
        for term in &claim.terms {
            for factor in &term.factors {
                let keys: Vec<(usize, usize, usize)> = match factor {
                    ClaimFactor::WitnessSegment(p) => vec![(p.prefix, p.length, 0)],
                    ClaimFactor::WitnessSegmentShifted(p, s) => vec![(p.prefix, p.length, *s)],
                    ClaimFactor::WitnessSegmentsScaled(parts, s) => {
                        parts.iter().map(|(p, _)| (p.prefix, p.length, *s)).collect()
                    }
                    _ => continue,
                };
                for key in keys {
                    if !order.contains(&key) {
                        order.push(key);
                    }
                }
            }
        }
    }
    order
}

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
    segments: &[((usize, usize, usize), RingElement)],
) -> ChainInputs {
    let total_vars = evaluation_points.len();
    let width_bits = witness_width.ilog2() as usize;
    let (points_outer, points_inner) = evaluation_points.split_at(width_bits);
    let mut inner = vec![
        evaluation_point_to_structured_row(points_inner),
        evaluation_point_to_structured_row_conjugate(points_inner),
    ];
    let mut outer = vec![
        evaluation_point_to_structured_row(points_outer),
        evaluation_point_to_structured_row_conjugate(points_outer),
    ];
    let mut claims = vec![witness_eval.clone(), conj_witness_eval.conjugate()];

    for ((prefix, length, suffix), eval) in segments {
        // full MSB-first point: prefix bits, then the data-variable challenge
        // slice; a suffix-shifted oracle folded vars [suffix, suffix+data),
        // i.e. the slice eps[length - suffix .. total - suffix]
        let mut point: Vec<RingElement> = (0..*length)
            .map(|i| {
                RingElement::constant(
                    ((prefix >> (length - 1 - i)) & 1) as u64,
                    Representation::IncompleteNTT,
                )
            })
            .collect();
        point.extend_from_slice(&evaluation_points[length - suffix..total_vars - suffix]);
        let (p_outer, p_inner) = point.split_at(width_bits);
        inner.push(evaluation_point_to_structured_row(p_inner));
        outer.push(evaluation_point_to_structured_row(p_outer));
        claims.push(eval.clone());
    }

    // InnerEvalFold prefix arithmetic requires a power-of-two opening count
    let target = claims.len().next_power_of_two();
    while claims.len() < target {
        inner.push(inner[0].clone());
        outer.push(outer[0].clone());
        claims.push(claims[0].clone());
    }

    ChainInputs {
        evaluation_points_inner: inner,
        evaluation_points_outer: outer,
        claims,
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
    let make_segment = |data: &[RingElement], prefix: usize, length: usize, suffix: usize| {
        let seg_len = data.len() >> length;
        let start = prefix << (total_vars - length);
        let mut ls = LinearSumcheck::new_with_prefixed_sufixed_data(seg_len, length - suffix, suffix);
        ls.load_from(&data[start..start + seg_len]);
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
                    ClaimFactor::WitnessSegment(p) => {
                        let (prefix, length) = (p.prefix, p.length);
                        let cell = witness_pool.next((prefix, length, 0), || {
                            make_segment(&witness.data, prefix, length, 0)
                        });
                        factors.push(cell.clone() as _);
                    }
                    ClaimFactor::WitnessSegmentShifted(p, s) => {
                        let (prefix, length, suffix) = (p.prefix, p.length, *s);
                        let cell = witness_pool.next((prefix, length, suffix), || {
                            make_segment(&witness.data, prefix, length, suffix)
                        });
                        factors.push(cell.clone() as _);
                    }
                    ClaimFactor::WitnessSegmentsScaled(parts, suffix) => {
                        let length = parts[0].0.length;
                        let seg_len = n >> length;
                        let mut combined =
                            vec![RingElement::zero(Representation::IncompleteNTT); seg_len];
                        let mut tmp = RingElement::zero(Representation::IncompleteNTT);
                        for (p, scale) in parts {
                            assert_eq!(p.length, length);
                            let start = p.prefix << (total_vars - length);
                            let mut scale = scale.clone();
                            if let Some(s) = &pending_scale {
                                scale *= s;
                            }
                            for (acc, w) in combined
                                .iter_mut()
                                .zip(&witness.data[start..start + seg_len])
                            {
                                tmp *= (w, &scale);
                                *acc += &tmp;
                            }
                        }
                        pending_scale = None;
                        let mut ls = LinearSumcheck::new_with_prefixed_sufixed_data(
                            seg_len,
                            length - suffix,
                            *suffix,
                        );
                        ls.load_from(&combined);
                        let cell = ElephantCell::new(ls);
                        leaves.push(LeafCell::Linear(cell.clone()));
                        factors.push(cell as _);
                    }
                    ClaimFactor::Public(public) => {
                        let mut data = match public {
                            PublicFactor::Structured(row) => {
                                assert_eq!(row.tensor_layers.len(), total_vars);
                                PreprocessedRow::from_structured_row(row).preprocessed_row
                            }
                            PublicFactor::Dense(v) => {
                                assert_eq!(v.len(), n);
                                v.clone()
                            }
                            PublicFactor::DensePrefixed(prefix_len, suffix_len, v) => {
                                assert_eq!(v.len(), n >> (prefix_len + suffix_len));
                                let key = (Arc::as_ptr(v) as usize, *prefix_len, *suffix_len);
                                let cell = public_pool.next(key, || {
                                    let mut ls = LinearSumcheck::new_with_prefixed_sufixed_data(
                                        v.len(),
                                        *prefix_len,
                                        *suffix_len,
                                    );
                                    ls.load_from(v);
                                    ls
                                });
                                factors.push(cell.clone() as _);
                                continue;
                            }
                            PublicFactor::Selector(prefix) => {
                                let cell = ElephantCell::new(SelectorEq::new(
                                    prefix.prefix,
                                    prefix.length,
                                    total_vars,
                                ));
                                leaves.push(LeafCell::Selector(cell.clone()));
                                factors.push(cell as _);
                                continue;
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

    for claim in claims {
        for term in &claim.terms {
            for factor in &term.factors {
                if let ClaimFactor::WitnessSegmentsScaled(parts, suffix) = factor {
                    witness_pool.reset_term();
                    for (p, _) in parts {
                        let (prefix, length) = (p.prefix, p.length);
                        let _ = witness_pool.next((prefix, length, *suffix), || {
                            make_segment(&witness.data, prefix, length, *suffix)
                        });
                    }
                }
            }
        }
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

    let segment_evals: Vec<RingElement> = segment_order(claims)
        .iter()
        .map(|key| {
            witness_pool
                .first_cell(key)
                .unwrap()
                .borrow()
                .final_evaluations()
                .clone()
        })
        .collect();

    hash_wrapper.update_with_ring_element(&witness_eval);
    hash_wrapper.update_with_ring_element(&conj_witness_eval);
    for eval in &segment_evals {
        hash_wrapper.update_with_ring_element(eval);
    }

    evaluation_points.reverse();

    let segments: Vec<((usize, usize, usize), RingElement)> = segment_order(claims)
        .into_iter()
        .zip(segment_evals.iter().cloned())
        .collect();
    let inputs = chain_inputs(
        &evaluation_points,
        witness.width,
        &witness_eval,
        &conj_witness_eval,
        &segments,
    );

    (
        InitialSumcheckProof {
            polys,
            witness_eval,
            conj_witness_eval,
            segment_evals,
        },
        inputs,
    )
}

pub fn verify_initial_claims(
    witness_height: usize,
    witness_width: usize,
    claims: &[SnarkClaim],
    proof: &InitialSumcheckProof,
    hash_wrapper: &mut HashWrapper,
) -> ChainInputs {
    let n = witness_height * witness_width;
    let total_vars = n.ilog2() as usize;
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

    let seg_order = segment_order(claims);
    assert_eq!(
        seg_order.len(),
        proof.segment_evals.len(),
        "segment evaluation count mismatch"
    );
    let segment_cells: std::collections::HashMap<
        (usize, usize, usize),
        ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    > = seg_order
        .iter()
        .zip(proof.segment_evals.iter())
        .map(|(key, eval)| {
            let cell = ElephantCell::new(FakeEvaluationLinearSumcheck::new());
            cell.borrow_mut().set_result(eval.clone());
            (*key, cell)
        })
        .collect();
    let eval_of: std::collections::HashMap<(usize, usize, usize), RingElement> = seg_order
        .iter()
        .zip(proof.segment_evals.iter())
        .map(|(key, eval)| (*key, eval.clone()))
        .collect();

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
                    ClaimFactor::WitnessSegment(p) => {
                        factors.push(segment_cells[&(p.prefix, p.length, 0)].clone() as _)
                    }
                    ClaimFactor::WitnessSegmentShifted(p, s) => {
                        factors.push(segment_cells[&(p.prefix, p.length, *s)].clone() as _)
                    }
                    ClaimFactor::WitnessSegmentsScaled(parts, suffix) => {
                        let mut value = RingElement::zero(Representation::IncompleteNTT);
                        let mut tmp = RingElement::zero(Representation::IncompleteNTT);
                        for (p, scale) in parts {
                            let mut scale = scale.clone();
                            if let Some(s) = &pending_scale {
                                scale *= s;
                            }
                            tmp *= (&eval_of[&(p.prefix, p.length, *suffix)], &scale);
                            value += &tmp;
                        }
                        pending_scale = None;
                        let cell = ElephantCell::new(FakeEvaluationLinearSumcheck::new());
                        cell.borrow_mut().set_result(value);
                        factors.push(cell as _);
                    }
                    ClaimFactor::Public(public) => {
                        let inner: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>> =
                            match public {
                                PublicFactor::Structured(row) => {
                                    let mut ev = crate::protocol::sumcheck_utils::linear::StructuredRowEvaluationLinearSumcheck::new(n);
                                    ev.load_from(row.clone());
                                    ElephantCell::new(ev) as _
                                }
                                PublicFactor::Dense(v) => {
                                    let mut ev = BasicEvaluationLinearSumcheck::new(n);
                                    ev.load_from(v);
                                    ElephantCell::new(ev) as _
                                }
                                PublicFactor::DensePrefixed(prefix_len, suffix_len, v) => {
                                    let mut ev =
                                        BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
                                            v.len(),
                                            *prefix_len,
                                            *suffix_len,
                                        );
                                    ev.load_from(v);
                                    factors.push(ElephantCell::new(ev) as _);
                                    continue;
                                }
                                PublicFactor::Selector(prefix) => {
                                    factors.push(ElephantCell::new(SelectorEqEvaluation::new(
                                        prefix.prefix,
                                        prefix.length,
                                        total_vars,
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

    for claim in claims {
        hash_wrapper.update_with_ring_element(&claim.value);
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
    for (claim, gamma) in claims.iter().zip(combination.iter()) {
        temp *= (&claim.value, gamma);
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
    for eval in &proof.segment_evals {
        hash_wrapper.update_with_ring_element(eval);
    }

    evaluation_points.reverse();

    let segments: Vec<((usize, usize, usize), RingElement)> = seg_order
        .into_iter()
        .zip(proof.segment_evals.iter().cloned())
        .collect();
    chain_inputs(
        &evaluation_points,
        witness_width,
        &proof.witness_eval,
        &proof.conj_witness_eval,
        &segments,
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
                ClaimFactor::Public(PublicFactor::Dense(a.clone())),
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
                ClaimFactor::Public(PublicFactor::Dense(b)),
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
                ClaimFactor::Public(PublicFactor::Selector(prefix)),
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
                        ClaimFactor::Public(PublicFactor::Dense(a.clone())),
                        ClaimFactor::Witness,
                    ],
                ),
                ClaimTerm::scaled(
                    minus_one,
                    vec![
                        ClaimFactor::Public(PublicFactor::Dense(a)),
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
    fn test_segment_product_claim_roundtrip() {
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
        let total_vars = n.ilog2() as usize;

        // two segments of length n/4: prefixes 1 and 2 (2 bits)
        let p_d = Prefix {
            prefix: 1,
            length: 2,
        };
        let p_g = Prefix {
            prefix: 2,
            length: 2,
        };
        let seg = n / 4;
        let d = &witness.data[seg..2 * seg];
        let g = &witness.data[2 * seg..3 * seg];

        // weight vector over the segment offsets, embedded at prefix 0 so the
        // full-hypercube sum picks each offset exactly once
        let w_seg = sample_random_short_vector(seg, 10, Representation::IncompleteNTT);
        let mut weight_full = vec![RingElement::zero(Representation::IncompleteNTT); n];
        weight_full[..seg].clone_from_slice(&w_seg);

        let mut value = RingElement::zero(Representation::IncompleteNTT);
        let mut tmp = RingElement::zero(Representation::IncompleteNTT);
        let mut tmp2 = RingElement::zero(Representation::IncompleteNTT);
        for i in 0..seg {
            tmp *= (&d[i], &g[i]);
            tmp2 *= (&tmp, &w_seg[i]);
            value += &tmp2;
        }

        let claims = vec![SnarkClaim {
            terms: vec![ClaimTerm::new(vec![
                ClaimFactor::Public(PublicFactor::Dense(weight_full)),
                ClaimFactor::WitnessSegment(p_d),
                ClaimFactor::WitnessSegment(p_g),
            ])],
            value,
        }];

        let mut hw_p = HashWrapper::new();
        let (proof, chain_p) = prove_initial_claims(&witness, &claims, &mut hw_p);
        assert_eq!(proof.segment_evals.len(), 2);

        let mut hw_v = HashWrapper::new();
        let chain_v = verify_initial_claims(height, width, &claims, &proof, &mut hw_v);
        assert_eq!(chain_p.claims, chain_v.claims);
        assert_eq!(chain_p.claims.len(), 4); // 2 + 2 segments, already pow2

        // every opening, including the segment ones, must match direct
        // witness evaluation under the chain's convention
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
