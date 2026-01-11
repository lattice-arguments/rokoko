use core::hash;
use std::{cell::RefCell, rc::Rc};

use num::range;

use crate::{
    common::{
        arithmetic::inner_product, decomposition::{
            compose_from_decomposed, get_composer_offset, get_decomposed_offset_scaled,
        }, hash::HashWrapper, matrix::{new_vec_zero_preallocated, VerticallyAlignedMatrix}, projection_matrix::{self, ProjectionMatrix}, ring_arithmetic::{Representation, RingElement}, structured_row::PreprocessedRow
    },
    protocol::{
        commitment::{self, Prefix}, config::{BASIC_COMMITMENT_RANK, Config, slice_by_prefix}, crs::{self, CRS}, open::{Opening, evaluation_point_to_structured_row}, project::project, sumcheck_utils::{
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

fn projection_coefficients(
    projection_matrix: &ProjectionMatrix,
    projection_flatter: &PreprocessedRow,
    witness_height: usize,
    projection_ratio: usize,
) -> Vec<RingElement> {
    let mut result = new_vec_zero_preallocated(witness_height);
    let height = crate::common::config::PROJECTION_HEIGHT;
    let blocks = projection_flatter.preprocessed_row.len() / height;

    for block in 0..blocks {
        for inner_row in 0..height {
            let weight = &projection_flatter.preprocessed_row[block * height + inner_row];
            if weight == &RingElement::zero(Representation::IncompleteNTT) {
                continue;
            }

            for i in 0..projection_ratio * height {
                let (is_positive, is_non_zero) = projection_matrix[(inner_row, i)];
                if !is_non_zero {
                    continue;
                }
                let idx = block * projection_ratio * height + i;
                if is_positive {
                    result[idx] += weight;
                } else {
                    result[idx] -= weight;
                }
            }
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

pub struct Type1SumcheckContext {
    inner_evaluation_sumcheck: Rc<RefCell<LinearSumcheck<RingElement>>>,
    opening_selector_sumcheck: Rc<RefCell<SelectorEq<RingElement>>>,
    pub output: Rc<RefCell<DiffSumcheck<RingElement>>>,
}

pub struct Type2SumcheckContext {
    outer_evaluation_sumcheck: Rc<RefCell<LinearSumcheck<RingElement>>>,
    pub output: Rc<RefCell<ProductSumcheck<RingElement>>>,
}

pub struct Type3SumcheckContext {
    lhs_sumcheck: Rc<RefCell<LinearSumcheck<RingElement>>>,
    rhs_sumcheck: Rc<RefCell<LinearSumcheck<RingElement>>>,
    projection_selector_sumcheck: Rc<RefCell<SelectorEq<RingElement>>>,
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
    pub opening_combiner_sumcheck: Rc<RefCell<LinearSumcheck<RingElement>>>,
    pub opening_combiner_constant_sumcheck: Rc<RefCell<LinearSumcheck<RingElement>>>,
    pub projection_combiner_sumcheck: Rc<RefCell<LinearSumcheck<RingElement>>>,
    pub projection_combiner_constant_sumcheck: Rc<RefCell<LinearSumcheck<RingElement>>>,
    pub type0sumchecks: Vec<Type0SumcheckContext>,
    pub type1sumchecks: Vec<Type1SumcheckContext>,
    pub type2sumchecks: Vec<Type2SumcheckContext>,
    pub type3sumcheck: Type3SumcheckContext,
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
        self.opening_combiner_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.opening_combiner_constant_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.projection_combiner_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.projection_combiner_constant_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        for type1_sc in self.type1sumchecks.iter() {
            type1_sc
                .inner_evaluation_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            type1_sc
                .opening_selector_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }
        for type2_sc in self.type2sumchecks.iter() {
            type2_sc
                .outer_evaluation_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }
        self.type3sumcheck
            .lhs_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.type3sumcheck
                .rhs_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        self.type3sumcheck
                .projection_selector_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
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

    let (opening_combiner_sumcheck, opening_combiner_constant_sumcheck) = composition_sumcheck(
        config.opening_recursion.decomposition_base_log as u64,
        config.opening_recursion.decomposition_chunks,
        config.composed_witness_length.ilog2() as usize,
    );

    let (projection_combiner_sumcheck, projection_combiner_constant_sumcheck) =
        composition_sumcheck(
            config.projection_recursion.decomposition_base_log as u64,
            config.projection_recursion.decomposition_chunks,
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

    let recomposed_folded_witness = Rc::new(RefCell::new(DiffSumcheck::new(
        Rc::new(RefCell::new(ProductSumcheck::new(
            combined_witness_sumcheck.clone(),
            folded_witness_combiner_sumcheck.clone(),
        ))),
        witness_combiner_constant_sumcheck.clone(),
    )));

    let recomposed_opening = Rc::new(RefCell::new(DiffSumcheck::new(
        Rc::new(RefCell::new(ProductSumcheck::new(
            combined_witness_sumcheck.clone(),
            opening_combiner_sumcheck.clone(),
        ))),
        opening_combiner_constant_sumcheck.clone(),
    )));

    let type1sumchecks = (0..config.nof_openings)
        .map(|i| {
            let opening_selector_sumcheck = sumcheck_from_prefix(
                &Prefix {
                    prefix: config.opening_recursion.prefix.prefix * config.nof_openings + i,
                    length: config.opening_recursion.prefix.length
                        + config.nof_openings.ilog2() as usize,
                },
                total_vars,
            );

            let inner_evaluation_sumcheck = Rc::new(RefCell::new(
                LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                    config.witness_height,
                    total_vars - config.witness_height.ilog2() as usize
                        - config.witness_decomposition_chunks.ilog2() as usize,
                    config.witness_decomposition_chunks.ilog2() as usize,
                ),
            ));

            let lhs = Rc::new(RefCell::new(ProductSumcheck::new(
                folded_witness_selector_sumcheck.clone(),
                Rc::new(RefCell::new(ProductSumcheck::new(
                    recomposed_folded_witness.clone(),
                    inner_evaluation_sumcheck.clone(),
                ))),
            )));

            let rhs = Rc::new(RefCell::new(ProductSumcheck::new(
                opening_selector_sumcheck.clone(),
                Rc::new(RefCell::new(ProductSumcheck::new(
                    recomposed_opening.clone(),
                    folding_challenges_sumcheck.clone(),
                ))),
            )));

            let output = Rc::new(RefCell::new(DiffSumcheck::new(lhs, rhs)));

            Type1SumcheckContext {
                inner_evaluation_sumcheck,
                opening_selector_sumcheck,
                output,
            }
        })
        .collect::<Vec<Type1SumcheckContext>>();

    // Type2 sumchecks
    // <opening.rhs[i], outer_evaluation_points> = evaluations[i] (public)
    let type2sumchecks = type1sumchecks
        .iter()
        .map(|type1_sc| {
            let outer_evaluation_sumcheck = Rc::new(RefCell::new(
                LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                    config.witness_width,
                    total_vars - config.witness_width.ilog2() as usize
                        - config.opening_recursion.decomposition_chunks.ilog2() as usize,
                    config.opening_recursion.decomposition_chunks.ilog2() as usize,
                ),
            ));

            let output = Rc::new(RefCell::new(ProductSumcheck::new(
                type1_sc.opening_selector_sumcheck.clone(),
                Rc::new(RefCell::new(ProductSumcheck::new(
                    recomposed_opening.clone(),
                    outer_evaluation_sumcheck.clone(),
                ))),
            )));

            Type2SumcheckContext {
                outer_evaluation_sumcheck,
                output,
            }
        })
        .collect::<Vec<Type2SumcheckContext>>();

    // type3 sumchecks
    // projection_matrix_flatter \cdot (I \otimes projection_matrix) \cdot folded_witness - projection_matrix_flatter \cdot projection_image \cdot fold_challenge = 0
    let recomposed_projection = Rc::new(RefCell::new(DiffSumcheck::new(
        Rc::new(RefCell::new(ProductSumcheck::new(
            combined_witness_sumcheck.clone(),
            projection_combiner_sumcheck.clone(),
        ))),
        projection_combiner_constant_sumcheck.clone(),
    )));

    let projection_height_flat = config.witness_height / config.projection_ratio;
    let type3sumcheck = {
        let projection_selector_sumcheck =
            sumcheck_from_prefix(&config.projection_recursion.prefix, total_vars);

        let projection_coeff_sumcheck = Rc::new(RefCell::new(
            LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                config.witness_height,
                total_vars - config.witness_height.ilog2() as usize
                    - config.witness_decomposition_chunks.ilog2() as usize,
                config.witness_decomposition_chunks.ilog2() as usize,
            ),
        ));

        let fold_tensor_sumcheck = Rc::new(RefCell::new(
            LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                projection_height_flat * config.witness_width,
                total_vars
                    - (projection_height_flat * config.witness_width).ilog2() as usize
                    - config.projection_recursion.decomposition_chunks.ilog2() as usize,
                config.projection_recursion.decomposition_chunks.ilog2() as usize,
            ),
        ));

        let lhs = Rc::new(RefCell::new(ProductSumcheck::new(
            folded_witness_selector_sumcheck.clone(),
            Rc::new(RefCell::new(ProductSumcheck::new(
                recomposed_folded_witness.clone(),
                projection_coeff_sumcheck.clone(),
            ))),
        )));
        let rhs = Rc::new(RefCell::new(ProductSumcheck::new(
            projection_selector_sumcheck.clone(),
            Rc::new(RefCell::new(ProductSumcheck::new(
                recomposed_projection.clone(),
                fold_tensor_sumcheck.clone(),
            ))),
        )));
        let output = Rc::new(RefCell::new(DiffSumcheck::new(lhs, rhs)));

        Type3SumcheckContext {
            lhs_sumcheck: projection_coeff_sumcheck,
            rhs_sumcheck: fold_tensor_sumcheck,
            projection_selector_sumcheck,
            output,
        }
    };



    // type4 sumchecks
    // rc_projection_image, rc_opening, rc_commitment are well-formed

    // type5 sumchecks
    // <combined_witness, conj(combined_witness)> = t (public)

    SumcheckContext {
        combined_witness_sumcheck,
        folded_witness_selector_sumcheck,
        folded_witness_combiner_sumcheck,
        witness_combiner_constant_sumcheck,
        folding_challenges_sumcheck,
        basic_commitment_combiner_sumcheck,
        basic_commitment_combiner_constant_sumcheck,
        commitment_key_rows_sumcheck,
        opening_combiner_sumcheck,
        opening_combiner_constant_sumcheck,
        projection_combiner_sumcheck,
        projection_combiner_constant_sumcheck,
        type0sumchecks,
        type1sumchecks,
        type2sumchecks,
        type3sumcheck,
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

    let projection_height_flat = config.witness_height / config.projection_ratio;
    let mut projection_matrix_flatter_base = new_vec_zero_preallocated(projection_height_flat.ilog2() as usize);
    hash_wrapper.sample_ring_element_vec_into(&mut projection_matrix_flatter_base);

    let projection_matrix_flatter = PreprocessedRow::from_structured_row(
        &evaluation_point_to_structured_row(&projection_matrix_flatter_base),
    );


    
    sumcheck_context
        .combined_witness_sumcheck
        .borrow_mut()
        .load_from(combined_witness);
    sumcheck_context
        .folding_challenges_sumcheck
        .borrow_mut()
        .load_from(&folding_challenges);

    for (i, type1_sc) in sumcheck_context.type1sumchecks.iter().enumerate() {
        type1_sc
            .inner_evaluation_sumcheck
            .borrow_mut()
            .load_from(&opening.evaluation_points_inner[i].preprocessed_row);
    }
    for (i, type2_sc) in sumcheck_context.type2sumchecks.iter().enumerate() {
        type2_sc
            .outer_evaluation_sumcheck
            .borrow_mut()
            .load_from(&opening.evaluation_points_outer[i].preprocessed_row);
    }
    let type3_sc = &mut sumcheck_context.type3sumcheck;
    {
        let projection_coeffs = projection_coefficients(
            projection_matrix,
            &projection_matrix_flatter,
            config.witness_height,
            config.projection_ratio,
        );
        type3_sc
            .lhs_sumcheck
            .borrow_mut()
            .load_from(&projection_coeffs);

        let fold_tensor = tensor_product(
            folding_challenges,
            &projection_matrix_flatter.preprocessed_row,
        );
        type3_sc
            .rhs_sumcheck
            .borrow_mut()
            .load_from(&fold_tensor);
    }

    let mut poly = Polynomial::new(0);
    let i = 0; // we check only the first type0 sumcheck here for testing
    sumcheck_context.type0sumchecks[i]
        .output
        .borrow_mut()
        .univariate_polynomial_into(&mut poly);

    assert_eq!(
        &poly.at_zero() + &poly.at_one(),
        RingElement::zero(Representation::IncompleteNTT)
    );

    // check type1 sumcheck here

    let r0 = RingElement::constant(7, Representation::IncompleteNTT);
    let type0_claim_after_r0 = poly.at(&r0);

    let mut type1_claim_after_r0 = None;
    if let Some(type1_sc) = sumcheck_context.type1sumchecks.get(0) {
        type1_sc
            .output
            .borrow_mut()
            .univariate_polynomial_into(&mut poly);
        assert_eq!(
            &poly.at_zero() + &poly.at_one(),
            RingElement::zero(Representation::IncompleteNTT)
        );
        type1_claim_after_r0 = Some(poly.at(&r0));
    }

    let mut type2_claim_after_r0 = None;
    if let Some(type2_sc) = sumcheck_context.type2sumchecks.get(0) {
        type2_sc
            .output
            .borrow_mut()
            .univariate_polynomial_into(&mut poly);
        assert_eq!(&poly.at_zero() + &poly.at_one(), claims[0]);
        type2_claim_after_r0 = Some(poly.at(&r0));
    }

    let mut type3_claim_after_r0 = None;
    sumcheck_context.type3sumcheck
        .output
        .borrow_mut()
        .univariate_polynomial_into(&mut poly);
    assert_eq!(
        &poly.at_zero() + &poly.at_one(),
        RingElement::zero(Representation::IncompleteNTT)
    );
    type3_claim_after_r0 = Some(poly.at(&r0));


    sumcheck_context.partial_evaluate_all(&r0);
    sumcheck_context.type0sumchecks[i]
        .output
        .borrow_mut()
        .univariate_polynomial_into(&mut poly);
    assert_eq!(&poly.at_zero() + &poly.at_one(), type0_claim_after_r0);

    if let (Some(type1_sc), Some(type1_claim)) =
        (sumcheck_context.type1sumchecks.get(0), type1_claim_after_r0)
    {
        type1_sc
            .output
            .borrow_mut()
            .univariate_polynomial_into(&mut poly);
        assert_eq!(&poly.at_zero() + &poly.at_one(), type1_claim);
    }
    if let (Some(type2_sc), Some(type2_claim)) =
        (sumcheck_context.type2sumchecks.get(0), type2_claim_after_r0)
    {
        type2_sc
            .output
            .borrow_mut()
            .univariate_polynomial_into(&mut poly);
        assert_eq!(&poly.at_zero() + &poly.at_one(), type2_claim);
    }
    sumcheck_context.type3sumcheck
        .output
        .borrow_mut()
        .univariate_polynomial_into(&mut poly);
    assert_eq!(&poly.at_zero() + &poly.at_one(), type3_claim_after_r0.unwrap());
}
