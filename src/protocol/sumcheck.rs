use core::hash;
use std::{cell::RefCell, rc::Rc};

use num::range;

use crate::{
    common::{
        arithmetic::inner_product,
        decomposition::{
            compose_from_decomposed, get_composer_offset, get_decomposed_offset_scaled,
        },
        hash::HashWrapper,
        projection_matrix::{self, ProjectionMatrix},
        ring_arithmetic::{Representation, RingElement},
    },
    protocol::{
        commitment::{self, Prefix}, config::{BASIC_COMMITMENT_RANK, Config, slice_by_prefix}, crs::{self, CRS}, open::Opening, sumcheck_utils::{
            common::{HighOrderSumcheckData, SumcheckBaseData},
            diff::DiffSumcheck,
            linear::LinearSumcheck,
            polynomial::Polynomial,
            product::ProductSumcheck,
            selector_eq::SelectorEq,
        }
    },
};

fn composition_sumcheck(
    base_log: u64,
    chunks: usize,
    total_vars: usize,
) -> (
    Rc<RefCell<LinearSumcheck<RingElement>>>,
    Rc<RefCell<LinearSumcheck<RingElement>>>,
) {
    let conmposition_basis = range(0, chunks)
        .map(|i| {
            // Basis element corresponding to 2^{base_log * i}
            RingElement::constant(
                1u64 << (base_log as u64 * i as u64),
                Representation::IncompleteNTT,
            )
        })
        .collect::<Vec<RingElement>>();
    let combiner_sumcheck = Rc::new(RefCell::new(
        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
            conmposition_basis.len(),
            total_vars - conmposition_basis.len().ilog2() as usize,
            0,
        ),
    ));

    combiner_sumcheck
        .borrow_mut()
        .load_from(&conmposition_basis);

    let mut witness_combiner_constant_sumcheck = Rc::new(RefCell::new(
        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(1, total_vars, 0),
    ));

    witness_combiner_constant_sumcheck
        .borrow_mut()
        .load_from(&vec![RingElement::all(
            get_decomposed_offset_scaled(base_log as u64, chunks),
            // 0,
            Representation::IncompleteNTT,
        )]);

    (combiner_sumcheck, witness_combiner_constant_sumcheck)
}

fn sumcheck_from_prefix(
    prefix: &Prefix,
    total_vars: usize,
) -> Rc<RefCell<SelectorEq<RingElement>>> {
    Rc::new(RefCell::new(SelectorEq::<RingElement>::new(
        prefix.prefix,
        prefix.length,
        total_vars,
    )))
}

fn ck_sumcheck(
    crs: &CRS,
    total_vars: usize,
    wit_dim: usize,
    i: usize,
    sufix: usize,
) -> Rc<RefCell<LinearSumcheck<RingElement>>> {
    let ck = crs.ck_for_wit_dim(wit_dim);

    let mut sumcheck = Rc::new(RefCell::new(
        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
            wit_dim,
            total_vars - wit_dim.ilog2() as usize - sufix,
            sufix,
        ),
    ));

    sumcheck.borrow_mut().load_from(&ck[i].preprocessed_row);

    sumcheck
}

// Computes the tensor product of two vectors of RingElements.
// e.g. (a0, a1) ⊗ (b0, b1) = (a0*b0, a0*b1, a1*b0, a1*b1)
fn tensor_product(a: &Vec<RingElement>, b: &Vec<RingElement>) -> Vec<RingElement> {
    let mut result = Vec::with_capacity(a.len() * b.len());
    for a_elem in a.iter() {
        for b_elem in b.iter() {
            result.push(a_elem * b_elem);
        }
    }
    result
}

// This is domain-specific sumcheck executor implementations
// for various sumcheck protocols used in the main protocol.

// SUMCHECK
// we want to check that
// ck \cdot folded_witness - commitment \cdot fold_challenge = 0
// outer_evaluation_points \cdot folded_witness - opening \cdot fold_challenge = 0
// <opening, inner_evaluation_points> - evaluations = 0
// I \otimes projection_matrix \cdot folded_witness - projection_image \cdot fold_challenge = 0
// rc_projection_image, rc_opening, rc_commitment are well-formed
// <w, conj(w)> + <y, conj(y)> - t = 0

pub struct Type0SumcheckContext {
    basic_commitment_row_sumcheck: Rc<RefCell<SelectorEq<RingElement>>>,
    pub output: Rc<RefCell<DiffSumcheck<RingElement>>>,
}

pub struct SumcheckContext {
    pub combined_witness_sumcheck: Rc<RefCell<LinearSumcheck<RingElement>>>,
    pub folded_witness_selector_sumcheck: Rc<RefCell<SelectorEq<RingElement>>>,
    pub folded_witness_combiner_sumcheck: Rc<RefCell<LinearSumcheck<RingElement>>>,
    pub witness_combiner_constant_sumcheck: Rc<RefCell<LinearSumcheck<RingElement>>>,
    pub folding_challenges_sumcheck: Rc<RefCell<LinearSumcheck<RingElement>>>,
    // basic_commitment_rows_sumcheck: Vec<RefCell<LinearSumcheck<RingElement>>>,
    pub basic_commitment_combiner_sumcheck: Rc<RefCell<LinearSumcheck<RingElement>>>,
    pub basic_commitment_combiner_constant_sumcheck: Rc<RefCell<LinearSumcheck<RingElement>>>,
    // commitment_key_row_sumcheck: RefCell<LinearSumcheck<RingElement>>,
    pub commitment_key_rows_sumcheck: Vec<Rc<RefCell<LinearSumcheck<RingElement>>>>,
    pub type0sumchecks: Vec<Type0SumcheckContext>,
    // put type1sumchecks
}

impl SumcheckContext {
    pub fn partial_evaluate_all(&mut self, r: &RingElement) {
        self.combined_witness_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.folded_witness_selector_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.folded_witness_combiner_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.witness_combiner_constant_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.folding_challenges_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.basic_commitment_combiner_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.basic_commitment_combiner_constant_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        for ck_row_sc in self.commitment_key_rows_sumcheck.iter() {
            ck_row_sc.borrow_mut().partial_evaluate(r);
        }
        for type0_sc in self.type0sumchecks.iter() {
            type0_sc
                .basic_commitment_row_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }
    }
}

// // Initialization of sumcheck protocols which happens before the rounds start
// // this function sets up the sumcheck data structures
// // and loads the available data into them
pub fn init_sumcheck(crs: &crs::CRS, config: &Config) -> SumcheckContext {
    let total_vars = config.composed_witness_length.ilog2() as usize;

    let mut combined_witness_sumcheck = Rc::new(RefCell::new(LinearSumcheck::<RingElement>::new(
        config.composed_witness_length,
    )));

    let folded_witness_selector_sumcheck =
        sumcheck_from_prefix(&config.folded_witness_prefix, total_vars);

    let commitment_key_rows_sumcheck = (0..config.basic_commitment_rank)
        .map(|i| {
            ck_sumcheck(
                crs,
                total_vars,
                config.witness_height,
                i,
                config.witness_decomposition_chunks.ilog2() as usize,
            )
        })
        .collect::<Vec<Rc<RefCell<LinearSumcheck<RingElement>>>>>();

    let (mut folded_witness_combiner_sumcheck, mut witness_combiner_constant_sumcheck) =
        composition_sumcheck(
            config.witness_decomposition_base_log as u64,
            config.witness_decomposition_chunks,
            config.composed_witness_length.ilog2() as usize,
        );

    let (mut basic_commitment_combiner_sumcheck, mut basic_commitment_combiner_constant_sumcheck) =
        composition_sumcheck(
            config.commitment_recursion.decomposition_base_log as u64,
            config.commitment_recursion.decomposition_chunks,
            config.composed_witness_length.ilog2() as usize,
        );

    let folding_challenges_sumcheck = Rc::new(RefCell::new(
        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
            config.witness_width,
            config.composed_witness_length.ilog2() as usize
                - config.witness_width.ilog2() as usize
                - config.commitment_recursion.decomposition_chunks.ilog2() as usize,
            config.commitment_recursion.decomposition_chunks.ilog2() as usize,
        ),
    ));

    // Type0 sumchecks
    // CK \cdot folded_witness - commitment \cdot fold_challenge = 0
    let type0sumchecks = (0..config.basic_commitment_rank)
        .map(|i| {
            let basic_commitment_row_sumcheck = sumcheck_from_prefix(
                &Prefix {
                    prefix: config.commitment_recursion.prefix.prefix
                        * config.basic_commitment_rank
                        + i,
                    length: config.commitment_recursion.prefix.length
                        + config.basic_commitment_rank.ilog2() as usize,
                },
                total_vars,
            );

            let ctxt = Type0SumcheckContext {
                basic_commitment_row_sumcheck: basic_commitment_row_sumcheck.clone(),
                output: Rc::new(RefCell::new(DiffSumcheck::new(
                    Rc::new(RefCell::new(ProductSumcheck::new(
                        folded_witness_selector_sumcheck.clone(),
                        Rc::new(RefCell::new(ProductSumcheck::new(
                            Rc::new(RefCell::new(DiffSumcheck::new(
                                Rc::new(RefCell::new(ProductSumcheck::new(
                                    combined_witness_sumcheck.clone(),
                                    folded_witness_combiner_sumcheck.clone(),
                                ))),
                                witness_combiner_constant_sumcheck.clone(),
                            ))),
                            commitment_key_rows_sumcheck[i].clone(),
                        ))),
                    ))),
                    Rc::new(RefCell::new(ProductSumcheck::new(
                        basic_commitment_row_sumcheck,
                        Rc::new(RefCell::new(ProductSumcheck::new(
                            Rc::new(RefCell::new(DiffSumcheck::new(
                                Rc::new(RefCell::new(ProductSumcheck::new(
                                    combined_witness_sumcheck.clone(),
                                    basic_commitment_combiner_sumcheck.clone(),
                                ))),
                                basic_commitment_combiner_constant_sumcheck.clone(),
                            ))),
                            folding_challenges_sumcheck.clone(),
                        ))),
                    ))),
                ))),
            };
            ctxt
        })
        .collect::<Vec<Type0SumcheckContext>>();

    // Type1 sumchecks 
    // inner_evaluation_points \cdot folded_witness - opening.rhs \cdot fold_challenge = 0

    // TODO

    // Type2 sumchecks
    // <opening.rhs[i], outer_evaluation_points> = evaluations[i] (public)

    // type3 sumchecks
    // I \otimes projection_matrix \cdot folded_witness - projection_image \cdot fold_challenge = 0

    // type4 sumchecks
    // rc_projection_image, rc_opening, rc_commitment are well-formed

    // type5 sumchecks
    // <combined_witness, conj(combined_witness)> = t (public)

    SumcheckContext {
        combined_witness_sumcheck,
        folded_witness_selector_sumcheck,
        folded_witness_combiner_sumcheck,
        witness_combiner_constant_sumcheck,
        commitment_key_rows_sumcheck,
        folding_challenges_sumcheck,
        basic_commitment_combiner_sumcheck,
        basic_commitment_combiner_constant_sumcheck,
        type0sumchecks,
    }
}
pub fn sumcheck(
    crs: &crs::CRS,
    config: &Config,
    combined_witness: &Vec<RingElement>,
    projection_matrix: &ProjectionMatrix,
    folding_challenges: &Vec<RingElement>,
    opening: &Opening,
    claims: &Vec<RingElement>,
    hash_wrapper: &mut HashWrapper,
) {
    let mut sumcheck_context = init_sumcheck(crs, config);

    sumcheck_context
        .combined_witness_sumcheck
        .borrow_mut()
        .load_from(combined_witness);
    sumcheck_context
        .folding_challenges_sumcheck
        .borrow_mut()
        .load_from(&folding_challenges);

    let mut poly = Polynomial::new(0);
    let i = 0;
    sumcheck_context.type0sumchecks[i]
        .output
        .borrow_mut()
        .univariate_polynomial_into(&mut poly);

    assert_eq!(
        &poly.at_zero() + &poly.at_one(),
        RingElement::zero(Representation::IncompleteNTT)
    );

    let r0 = RingElement::constant(7, Representation::IncompleteNTT);
    let claim_after_r0 = poly.at(&r0);

    sumcheck_context.partial_evaluate_all(&r0);
    sumcheck_context.type0sumchecks[i]
        .output
        .borrow_mut()
        .univariate_polynomial_into(&mut poly);
    assert_eq!(&poly.at_zero() + &poly.at_one(), claim_after_r0);
}
