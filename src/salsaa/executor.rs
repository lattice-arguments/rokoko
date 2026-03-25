use std::process::Output;

use num::range;
use rand::rand_core::le;

use crate::{
    common::{
        arithmetic::{ALL_ONE_COEFFS, ONE, ZERO, field_to_ring_element_into}, config::{self, HALF_DEGREE}, decomposition::{compose_from_decomposed, decompose_chunks_into}, hash::HashWrapper, matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix, new_vec_zero_preallocated}, projection_matrix::ProjectionMatrix, ring_arithmetic::{QuadraticExtension, Representation, RingElement}, sampling::sample_random_short_vector, structured_row::{PreprocessedRow, StructuredRow}, sumcheck_element::SumcheckElement
    },
    protocol::{
        commitment::{self, BasicCommitment, Prefix, commit_basic, commit_basic_internal}, config::paste_by_prefix, crs::{self, CRS}, fold::fold, open::{claim, evaluation_point_to_structured_row}, project::{self, prepare_i16_witness, project}, sumcheck_utils::{
            combiner::{Combiner, CombinerEvaluation},
            common::{EvaluationSumcheckData, HighOrderSumcheckData, SumcheckBaseData},
            diff::{DiffSumcheck, DiffSumcheckEvaluation},
            elephant_cell::ElephantCell,
            linear::{
                BasicEvaluationLinearSumcheck, FakeEvaluationLinearSumcheck, LinearSumcheck,
                StructuredRowEvaluationLinearSumcheck,
            },
            polynomial::Polynomial,
            product::{ProductSumcheck, ProductSumcheckEvaluation},
            ring_to_field_combiner::{RingToFieldCombiner, RingToFieldCombinerEvaluation},
            selector_eq::{SelectorEq, SelectorEqEvaluation},
            sum,
        }, sumchecks::helpers::{projection_flatter_1_times_matrix, sumcheck_from_prefix}
    },
};

const WITNESS_DIM: usize = 2usize.pow(16);
const WITNESS_WIDTH: usize = 2usize;
const RANK: usize = 8;

pub struct SalsaaProof {
    projection_commitment: BasicCommitment,
    sumcheck_transcript: Vec<Polynomial<QuadraticExtension>>,
    ip_l2_claim: Option<RingElement>,
    ip_linf_claim: Option<RingElement>,
    claims: HorizontallyAlignedMatrix<RingElement>,
    claim_over_projection: Vec<RingElement>,
    new_claims: HorizontallyAlignedMatrix<RingElement>,
    commitment_to_folded_witness: BasicCommitment, // TODO: this is unnecessary, remove later
    decomposed_split_commitment: BasicCommitment, // TODO: this should be optional as on the last round we just send the witness immediately post folding
    next: Option<Box<SalsaaProof>>,
}

pub struct RoundConfig {
    pub witness_length: usize,
    pub exact_binariness: bool, // whether the proof should be for exact binariness
    pub l2: bool,               // whether the proof should be for l2 norm of the witness
    pub projection_ratio: usize, // set 0 for no projection
    pub main_witness_columns: usize,
    pub projection_prefix: Prefix,
    pub main_witness_prefix: Prefix,    // shall be always 0
    pub inner_evaluation_claims: usize, // how many inner evaluation claims we want to make, this determines the number of type1 sumchecks we need
    pub decomposition_base_log: u64, // base log for decomposition, if 0 then no decomposition is done and the proof is for the folded witness
    pub next: Option<Box<RoundConfig>>,
}

const NUM_COLUMNS_INITIAL: usize = 2;

const PROJECTION_HEIGHT: usize = 256;

// All configs shall be auto-derived, but we keep this struct for clarity for now
const CONFIG: RoundConfig = RoundConfig {
    witness_length: WITNESS_DIM * WITNESS_WIDTH * 2, // we ``bloat up'' the witness times two to account to the projection
    exact_binariness: true,
    l2: true,
    projection_ratio: 2, // for the first round is 2, later shall be 8 (I think)
    main_witness_columns: NUM_COLUMNS_INITIAL,
    main_witness_prefix: Prefix {
        prefix: 0b0,
        length: 1, // main witness takes half, always
    }, // in the first round we start with the origicnal witness that is out in NUM_COLUMNS_INITIAL columns and the projection will be the third column. In leter rounds, it will be different.
    projection_prefix: Prefix {
        prefix: 0b10,
        length: NUM_COLUMNS_INITIAL.ilog2() as usize + 1, // if the witness is 2 colums, then length is 2, if the witness is 8 columns, then length is 4, etc.
    },
    next: None, // for now, we test only the first round, but we will need to fill this in for later rounds
    inner_evaluation_claims: 1, // for VDF one will be enough
    decomposition_base_log: 12 // for now, we test only the first round, but we will need to fill this in for later rounds
};

// ==== Prover Sumcheck context initialization ====

pub struct ProverSumcheckContext {
    pub witness_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub witness_conjugated_sumcheck: ElephantCell<LinearSumcheck<RingElement>>, // for verifying norms. Should be optional?
    pub main_witness_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub projection_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub type1sumcheck: Vec<Type1ProverSumcheckContext>, // for verifying inner evaluation points
    pub type3sumcheck: Option<Type3ProverSumcheckContext>, // for verifying the projection
    // pub type5sumcheck: Option<Type5ProverSumcheckContext>, // for verifying the l2 norm of the witness, only used when exact_binariness is false TODO
    pub l2sumcheck: Option<L2ProverSumcheckContext>,
    pub linfsumcheck: Option<LinfSumcheckContext>,
    // pub type6sumcheck: Option<Type6ProvserSumcheckContext>, // for verifying ecact binariness, only used when exact_binariness is true TODO
    pub combiner: ElephantCell<Combiner<RingElement>>,
    pub field_combiner: ElephantCell<RingToFieldCombiner>,
    pub next: Option<Box<ProverSumcheckContext>>,
}

pub struct L2ProverSumcheckContext {
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
}

pub struct LinfSumcheckContext {
    pub all_one_constant_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
}

pub struct Type1ProverSumcheckContext {
    pub inner_evaluation_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub outer_evaluation_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
}

// we want to check that
// (I \otimes J) · witness = projected_witness
// post batching, this can be written as
// c^T (I \otimes J) · witness c_2 = c^T projected_witness c_2
// To be more precise, the projected witness is vectorised (by stacking columns)
// witness itself is vertically aligned so it can be viewed as a single column so we write:
// (c_2 \otimes c)^T (I \otimes J) · witness = (c_2 \otimes c)^T projected_witness
// we keep c and c_2 separated as c_2 will be needed as ``outer evaluation point'' since the  prover will open to
// c^T (I \otimes J) · witness and (c_2 \otimes c)^T · projected_witness and verify consistency between the two using the outer evaluation point c_2.
// c = (c_0, c_1) so that c^T (I \otimes J) = c_0 \otimes c_1^T J
// c_1^T J is denoted as flattened_projection_matrix
// to sum up, the relation we prove via sumcheck
// (c_2 \otimes c_0 \otimes c_1^T J) · witness = (c_2 \otimes c_0 \otimes c_1)^T projected_witness
pub struct Type3ProverSumcheckContext {
    pub c2l_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub c0l_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub flattened_projection_matrix_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub c2r_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub c0r_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub c1r_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub lhs: ElephantCell<ProductSumcheck<RingElement>>,
    pub rhs: ElephantCell<ProductSumcheck<RingElement>>,
    pub output: ElephantCell<DiffSumcheck<RingElement>>,
}

fn init_prover_type_1_sumcheck(
    config: &RoundConfig,
    main_witness_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
) -> Type1ProverSumcheckContext {
    let single_col_height = config.witness_length / 2 / config.main_witness_columns;
    let total_vars = config.witness_length.ilog2() as usize;
    let inner_evaluation_sumcheck =
        ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
            single_col_height,
            total_vars - single_col_height.ilog2() as usize,
            0,
        ));

    let outer_evaluation_sumcheck =
        ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
            config.main_witness_columns,
            total_vars
                - config.main_witness_columns.ilog2() as usize
                - single_col_height.ilog2() as usize,
            single_col_height.ilog2() as usize,
        ));
    // let outer_evaluation_sumcheck = ElephantCell::new(LinearSumcheck::new(config.main / 2));
    // we view MLE[w](evaluation_points_inner) as a sumcheck
    let output = ElephantCell::new(ProductSumcheck::new(
        ElephantCell::new(ProductSumcheck::new(
            inner_evaluation_sumcheck.clone(),
            outer_evaluation_sumcheck.clone(),
        )),
        main_witness_sumcheck.clone(),
    ));

    Type1ProverSumcheckContext {
        inner_evaluation_sumcheck,
        outer_evaluation_sumcheck,
        output,
    }
}

fn init_prover_type_3_sumcheck(
    config: &RoundConfig,
    main_witness_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
    projection_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
) -> Type3ProverSumcheckContext {
    let c2_len = config.main_witness_columns;
    let c1_len = PROJECTION_HEIGHT;
    // (c_2 \otimes c_0 \otimes c_1^T J) · witness = (c_2 \otimes c_0 \otimes c_1)^T projected_witness
    let single_col_height = config.witness_length / 2 / config.main_witness_columns;
    let c0_len: usize = single_col_height / (PROJECTION_HEIGHT * config.projection_ratio);
    assert!(c0_len > 0, "c0_len must be greater than 0");

    let total_vars = config.witness_length.ilog2() as usize;

    assert_eq!(c0_len * c1_len * c2_len, config.witness_length / (2_usize.pow(config.projection_prefix.length as u32)), "c0_len * c1_len * c2_len must be equal to witness_length, got c0_len: {}, c1_len: {}, c2_len: {}, witness_length: {}", c0_len, c1_len, c2_len, config.witness_length);

    // We have the following variables structure:
    // LEFT
    // prefix
    // c_2.ilog2() variables for c_2
    // c_0.ilog2() variables for c_0
    // (c_1^T J).ilog2() variables for (c_1^T J)

    // RIGHT
    // prefix
    // c_2.ilog2() variables for c_2
    // c_0.ilog2() variables for c_0
    // c_1.ilog2() variables for c_1

    // left
    let fltr_len = (config.projection_ratio * PROJECTION_HEIGHT).ilog2() as usize;

    let flattened_projection_matrix_sumcheck =
        ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
            config.projection_ratio * PROJECTION_HEIGHT,
            total_vars - fltr_len,
            0,
        ));
    let c0l_sumcheck = ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
        c0_len,
        total_vars - fltr_len - c0_len.ilog2() as usize,
        fltr_len,
    ));

    let c2l_sumcheck = ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
        c2_len,
        total_vars - fltr_len - c0_len.ilog2() as usize - c2_len.ilog2() as usize,
        fltr_len + c0_len.ilog2() as usize,
    ));

    // right
    let c1r_sumcheck = ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
        c1_len,
        total_vars - c1_len.ilog2() as usize,
        0,
    ));

    let c0r_sumcheck = ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
        c0_len,
        total_vars - c1_len.ilog2() as usize - c0_len.ilog2() as usize,
        c1_len.ilog2() as usize,
    ));

    let c2r_sumcheck = ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
        c2_len,
        total_vars - c1_len.ilog2() as usize - c0_len.ilog2() as usize - c2_len.ilog2() as usize,
        c1_len.ilog2() as usize + c0_len.ilog2() as usize,
    ));

    let lhs = ElephantCell::new(ProductSumcheck::new(
        c2l_sumcheck.clone(),
        ElephantCell::new(ProductSumcheck::new(
            c0l_sumcheck.clone(),
            ElephantCell::new(ProductSumcheck::new(
                flattened_projection_matrix_sumcheck.clone(),
                main_witness_sumcheck.clone(),
            )),
        )),
    ));

    let rhs = ElephantCell::new(ProductSumcheck::new(
        c2r_sumcheck.clone(),
        ElephantCell::new(ProductSumcheck::new(
            c0r_sumcheck.clone(),
            ElephantCell::new(ProductSumcheck::new(
                c1r_sumcheck.clone(),
                projection_sumcheck.clone(),
            )),
        )),
    ));

    let output = ElephantCell::new(DiffSumcheck::new(lhs.clone(), rhs.clone()));

    Type3ProverSumcheckContext {
        flattened_projection_matrix_sumcheck,
        c0l_sumcheck,
        c2l_sumcheck,
        c1r_sumcheck,
        c0r_sumcheck,
        c2r_sumcheck,
        lhs,
        rhs,
        output,
    }
}

pub fn init_linf_sumcheck(
    witness_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
    main_witness_selector: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
    conjugated_witness_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
) -> LinfSumcheckContext {
    let all_one_constant_sumcheck =
        ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
            1,
            witness_sumcheck.borrow().variable_count(),
            0,
        ));

    all_one_constant_sumcheck
        .borrow_mut()
        .load_from(&vec![ALL_ONE_COEFFS.clone()]);

    let one_minus_wit_sumcheck = ElephantCell::new(DiffSumcheck::new(
        all_one_constant_sumcheck.clone(),
        witness_sumcheck.clone(),
    ));

    let one_minus_wit_selector_sumcheck = ElephantCell::new(ProductSumcheck::new(
        main_witness_selector.clone(),
        one_minus_wit_sumcheck.clone(),
    ));

    let output = ElephantCell::new(ProductSumcheck::new(
        conjugated_witness_sumcheck.clone(),
        one_minus_wit_selector_sumcheck.clone(),
    ));

    LinfSumcheckContext {
        all_one_constant_sumcheck,
        output,
    }
}

pub fn init_prover_sumcheck(crs: &CRS, config: &RoundConfig) -> ProverSumcheckContext {
    let witness_sumcheck = ElephantCell::new(LinearSumcheck::new(config.witness_length));
    let witness_conjugated_sumcheck =
        ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
            config.witness_length >> config.main_witness_prefix.length,
            config.main_witness_prefix.length as usize,
            0,
        ));

    let main_witness_selector_sumcheck = sumcheck_from_prefix(
        &config.main_witness_prefix,
        config.witness_length.ilog2() as usize,
    );
    let projection_selector_sumcheck = sumcheck_from_prefix(
        &config.projection_prefix,
        config.witness_length.ilog2() as usize,
    );

    let main_witness_sumcheck: ElephantCell<ProductSumcheck<_>> =
        ElephantCell::new(ProductSumcheck::new(
            witness_sumcheck.clone(),
            main_witness_selector_sumcheck.clone(),
        ));
    let projection_sumcheck: ElephantCell<ProductSumcheck<_>> =
        ElephantCell::new(ProductSumcheck::new(
            witness_sumcheck.clone(),
            projection_selector_sumcheck.clone(),
        ));

    let type1sumcheck = (0..config.inner_evaluation_claims)
        .map(|_| init_prover_type_1_sumcheck(config, main_witness_sumcheck.clone()))
        .collect::<Vec<_>>();

    let type3sumcheck = if config.projection_ratio > 0 {
        Some(init_prover_type_3_sumcheck(
            config,
            main_witness_sumcheck.clone(),
            projection_sumcheck.clone(),
        ))
    } else {
        None
    };

    let l2sumcheck = if config.l2 {
        Some(L2ProverSumcheckContext {
            output: ElephantCell::new(ProductSumcheck::new(
                witness_conjugated_sumcheck.clone(),
                main_witness_sumcheck.clone(),
            )),
        })
    } else {
        None
    };

    let linfsumcheck = if config.exact_binariness {
        Some(init_linf_sumcheck(
            witness_sumcheck.clone(),
            main_witness_selector_sumcheck.clone(),
            witness_conjugated_sumcheck.clone(),
        ))
    } else {
        None
    };

    let mut all_outputs: Vec<ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>> =
        vec![];
    for type1 in &type1sumcheck {
        all_outputs.push(type1.output.clone());
    }

    if let Some(type3) = &type3sumcheck {
        all_outputs.push(type3.output.clone());
    }
    if let Some(l2) = &l2sumcheck {
        all_outputs.push(l2.output.clone());
    }
    if let Some(linf) = &linfsumcheck {
        all_outputs.push(linf.output.clone());
    }

    let combiner = ElephantCell::new(Combiner::new(all_outputs));
    let field_combiner = ElephantCell::new(RingToFieldCombiner::new(combiner.clone()));

    ProverSumcheckContext {
        witness_sumcheck,
        witness_conjugated_sumcheck,
        main_witness_selector_sumcheck,
        projection_selector_sumcheck,
        type1sumcheck,
        type3sumcheck,
        combiner,
        field_combiner,
        l2sumcheck,
        linfsumcheck,
        next: config
            .next
            .as_ref()
            .map(|next_config| Box::new(init_prover_sumcheck(crs, next_config))),
    }
}

pub struct BatchingChallenges {
    // in succinct form
    pub c0: StructuredRow<RingElement>,
    pub c1: StructuredRow<RingElement>,
    pub c2: StructuredRow<RingElement>,
}

impl BatchingChallenges {
    pub fn sample(config: &RoundConfig, hash_wrapper: &mut HashWrapper) -> Self {
        let c2_len = config.main_witness_columns;
        let c1_len = PROJECTION_HEIGHT;
        let single_col_height = config.witness_length / 2 / config.main_witness_columns;
        let c0_len: usize = single_col_height / (PROJECTION_HEIGHT * config.projection_ratio);
        assert!(c0_len > 0, "c0_len must be greater than 0");
        let mut result = Self {
            c0: StructuredRow {
                tensor_layers: new_vec_zero_preallocated(c0_len.ilog2() as usize),
            },
            c1: StructuredRow {
                tensor_layers: new_vec_zero_preallocated(c1_len.ilog2() as usize),
            },
            c2: StructuredRow {
                tensor_layers: new_vec_zero_preallocated(c2_len.ilog2() as usize),
            },
        };

        hash_wrapper.sample_ring_element_ntt_slots_same_vec_into(&mut result.c0.tensor_layers);
        hash_wrapper.sample_ring_element_ntt_slots_same_vec_into(&mut result.c1.tensor_layers);
        hash_wrapper.sample_ring_element_ntt_slots_same_vec_into(&mut result.c2.tensor_layers);

        result
    }
}

impl ProverSumcheckContext {
    pub fn partial_evaluate_all(&mut self, r: &RingElement) {
        self.witness_sumcheck.borrow_mut().partial_evaluate(r);
        self.witness_conjugated_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.main_witness_selector_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.projection_selector_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        for type1 in &mut self.type1sumcheck {
            type1
                .inner_evaluation_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            type1
                .outer_evaluation_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }
        if let Some(type3) = &mut self.type3sumcheck {
            type3
                .flattened_projection_matrix_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            type3.c0r_sumcheck.borrow_mut().partial_evaluate(r);
            type3.c1r_sumcheck.borrow_mut().partial_evaluate(r);
            type3.c2r_sumcheck.borrow_mut().partial_evaluate(r);
            type3.c0l_sumcheck.borrow_mut().partial_evaluate(r);
            type3.c2l_sumcheck.borrow_mut().partial_evaluate(r);
        }

        if let Some(next) = &mut self.next {
            next.partial_evaluate_all(r);
        }

        // it's dumb, but it doesn't do anything except reducing the degree
        if let Some(linf) = &mut self.linfsumcheck {
            linf.all_one_constant_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }
    }

    pub fn load_data(
        &mut self,
        witness: &Vec<RingElement>,
        witness_conjugated: &Vec<RingElement>,
        evaluation_points_inner: &Vec<StructuredRow>,
        evaluation_points_outer: &Vec<RingElement>,
        projection_matrix: &ProjectionMatrix,
        projection_batching_challenges: &Option<BatchingChallenges>,
    ) {
        self.witness_sumcheck.borrow_mut().load_from(&witness);
        self.witness_conjugated_sumcheck
            .borrow_mut()
            .load_from(&witness_conjugated);
        if let Some(projection_challenges) = projection_batching_challenges {
            let c0_expanded = PreprocessedRow::from_structured_row(&projection_challenges.c0);
            let c1_expanded = PreprocessedRow::from_structured_row(&projection_challenges.c1);
            let c2_expanded = PreprocessedRow::from_structured_row(&projection_challenges.c2);
            let flattened_projection =
                projection_flatter_1_times_matrix(projection_matrix, &c1_expanded);
            let mut flattened_projection_ring =
                new_vec_zero_preallocated(flattened_projection.len());

            for (i, el) in flattened_projection.iter().enumerate() {
                field_to_ring_element_into(&mut flattened_projection_ring[i], el);
                // TODO: I Spent 1h debugging this and it turned out that I forgot to convert the flattened projection matrix from homogenized field extensions to incomplete NTT, which is what the sumcheck expects. Rethink the interfaces here to avoid such issues in the future, maybe by having a clear type for the per rep.
                flattened_projection_ring[i].from_homogenized_field_extensions_to_incomplete_ntt();
            }
            if let Some(type3) = &mut self.type3sumcheck {
                type3
                    .c0r_sumcheck
                    .borrow_mut()
                    .load_from(&c0_expanded.preprocessed_row);
                type3
                    .c1r_sumcheck
                    .borrow_mut()
                    .load_from(&c1_expanded.preprocessed_row);
                type3
                    .c2r_sumcheck
                    .borrow_mut()
                    .load_from(&c2_expanded.preprocessed_row);

                type3
                    .c0l_sumcheck
                    .borrow_mut()
                    .load_from(&c0_expanded.preprocessed_row);
                type3
                    .c2l_sumcheck
                    .borrow_mut()
                    .load_from(&c2_expanded.preprocessed_row);
                type3
                    .flattened_projection_matrix_sumcheck
                    .borrow_mut()
                    .load_from(&flattened_projection_ring);
            } else {
                panic!(
                    "Projection batching challenges provided but type3 sumcheck is not initialized"
                );
            }
        }
        for (i, type1) in self.type1sumcheck.iter_mut().enumerate() {
            let evaluation_points_inner_expanded =
                PreprocessedRow::from_structured_row(&evaluation_points_inner[i]);
            type1
                .inner_evaluation_sumcheck
                .borrow_mut()
                .load_from(&evaluation_points_inner_expanded.preprocessed_row);
            type1
                .outer_evaluation_sumcheck
                .borrow_mut()
                .load_from(&evaluation_points_outer);
        }
    }
}

pub fn prover_round(
    crs: &CRS,
    commitmens: &BasicCommitment,
    witness: &VerticallyAlignedMatrix<RingElement>,
    config: &RoundConfig,
    sumcheck_context: &mut ProverSumcheckContext,
    evaluation_points_inner: &Vec<StructuredRow>,
    claims: &HorizontallyAlignedMatrix<RingElement>,
    // evaluation_points_outer: &Vec<StructuredRow>,
    exact_binariness: bool, // whether the proof should be for exact binariness. If not l2 norm of the witness is given by the proof
    hash_wrapper: &mut HashWrapper,
) -> SalsaaProof {
    let witness_16 = prepare_i16_witness(witness);

    let mut projection_matrix = ProjectionMatrix::new(witness.width, 256);

    projection_matrix.sample(hash_wrapper);

    let mut projected_witness = project(&witness_16, &projection_matrix);

    // The projection procent r columns into r columns with less rows, so we rearrange the projected witness taking advantage of the vertical alignment
    projected_witness.width = 1;
    projected_witness.used_cols = 1;
    projected_witness.height = witness.height;

    let projection_commitment = commit_basic(crs, &projected_witness, RANK);

    let batching_challenges = BatchingChallenges::sample(&CONFIG, hash_wrapper);

    let mut extended_witness = new_vec_zero_preallocated(witness.data.len() * 2);

    let mut witness_conjugated = new_vec_zero_preallocated(witness.data.len());
    for (i, w) in witness.data.iter().enumerate() {
        w.conjugate_into(&mut witness_conjugated[i]);
    }

    let ip_l2_claim = if config.l2 {
        let mut temp = RingElement::zero(Representation::IncompleteNTT);
        let mut claim = RingElement::zero(Representation::IncompleteNTT);
        for (w, wc) in witness.data.iter().zip(witness_conjugated.iter()) {
            temp *= (w, wc);
            claim += &temp;
        }
        Some(claim)
    } else {
        None
    };

    let ip_linf_claim = if config.exact_binariness {
        let mut temp = RingElement::zero(Representation::IncompleteNTT);
        let mut claim = RingElement::zero(Representation::IncompleteNTT);
        for (w, wc) in witness.data.iter().zip(witness_conjugated.iter()) {
            temp -= (&*ALL_ONE_COEFFS, w);
            temp *= wc;
            claim += &temp;
        }
        Some(claim)
    } else {
        None
    };

    paste_by_prefix(
        &mut extended_witness,
        &witness.data,
        &config.main_witness_prefix,
    );
    paste_by_prefix(
        &mut extended_witness,
        &projected_witness.data,
        &config.projection_prefix,
    );

    let mut evaluation_points_outer = new_vec_zero_preallocated(config.main_witness_columns);
    hash_wrapper.sample_ring_element_vec_into(&mut evaluation_points_outer);

    sumcheck_context.load_data(
        &extended_witness,
        &witness_conjugated,
        evaluation_points_inner,
        &evaluation_points_outer,
        &projection_matrix,
        &Some(batching_challenges),
    );

    // Sample random batching coefficients from Fiat-Shamir
    let num_sumchecks = sumcheck_context.combiner.borrow().sumchecks_count();
    let mut combination = new_vec_zero_preallocated(num_sumchecks);
    hash_wrapper.sample_ring_element_vec_into(&mut combination);

    sumcheck_context
        .combiner
        .borrow_mut()
        .load_challenges_from(&combination);

    let mut combination_to_field = RingElement::zero(Representation::IncompleteNTT);
    hash_wrapper.sample_ring_element_into(&mut combination_to_field);
    combination_to_field.from_incomplete_ntt_to_homogenized_field_extensions();
    let qe = combination_to_field.split_into_quadratic_extensions();

    sumcheck_context
        .field_combiner
        .borrow_mut()
        .load_challenges_from(qe);

    // TEST ONLY
    let claim = sumcheck_context.type1sumcheck[0].output.borrow().claim();

    let mut expected_claim = ZERO.clone();
    for (c, r) in claims.row(0).iter().zip(evaluation_points_outer.iter()) {
        expected_claim += &(c * r);
    }
    assert_eq!(claim, expected_claim, "Claim from the sumcheck does not match the expected claim computed from the committed witness and the evaluation points");

    let projection_claim = sumcheck_context
        .type3sumcheck
        .as_ref()
        .unwrap()
        .output
        .borrow()
        .claim();
    let expected_projection_claim = ZERO.clone();
    assert_eq!(
        projection_claim, expected_projection_claim,
        "Projection claim from the sumcheck does not match the expected projection claim"
    );

    if config.l2 {
        let l2_claim = sumcheck_context
            .l2sumcheck
            .as_ref()
            .unwrap()
            .output
            .borrow()
            .claim();
        assert_eq!(
            l2_claim, ip_l2_claim.clone().unwrap(),
            "L2 claim from the projection sumcheck does not match the expected l2 claim computed from the witness"
        );
    }

    if config.exact_binariness {
        let linf_claim = sumcheck_context
            .linfsumcheck
            .as_ref()
            .unwrap()
            .output
            .borrow()
            .claim();
        let ct = linf_claim.constant_term_from_incomplete_ntt();
        assert_eq!(ct, 0, "Linf claim from the projection sumcheck is not zero, which means that the witness is not exactly binary as expected");

        assert_eq!(
            linf_claim, ip_linf_claim.clone().unwrap(),
            "Linf claim from the projection sumcheck does not match the expected linf claim computed from the witness"
        );
    }
    // END TEST ONLY

    let mut num_vars = sumcheck_context.combiner.borrow().variable_count();

    let mut time_poly = 0u128;
    let mut time_eval = 0u128;
    let mut evaluation_points = Vec::new();
    let mut polys = Vec::new();

    while num_vars > 0 {
        num_vars -= 1;

        let t1 = std::time::Instant::now();
        let mut poly_over_field = Polynomial::<QuadraticExtension>::new(0);

        sumcheck_context
            .field_combiner
            .borrow_mut()
            .univariate_polynomial_into(&mut poly_over_field);
        time_poly += t1.elapsed().as_millis();

        hash_wrapper.update_with_quadratic_extension_slice(&poly_over_field.coefficients);

        let mut r = RingElement::zero(Representation::IncompleteNTT);
        let mut f = QuadraticExtension::zero();

        hash_wrapper.sample_field_element_into(&mut f);

        field_to_ring_element_into(&mut r, &f);
        r.from_homogenized_field_extensions_to_incomplete_ntt();

        evaluation_points.push(r.clone());

        let t2 = std::time::Instant::now();
        sumcheck_context.partial_evaluate_all(&r);
        time_eval += t2.elapsed().as_millis();

        polys.push(poly_over_field);
    }

    println!(
        "Polynomial time: {:?} ms, Evaluation time: {:?} ms",
        time_poly, time_eval
    );

    let outer_points_len = config.main_witness_columns.ilog2() as usize + 1; // extended witness is two times the original witness, so we need one more bit for the prefix
    let evaluation_points_inner = evaluation_points
        .iter()
        .skip(outer_points_len)
        .cloned()
        .collect::<Vec<_>>();
    let mut preprocessed_evaluation_points_inner = PreprocessedRow::from_structured_row(
        &evaluation_point_to_structured_row(&evaluation_points_inner),
    );

    let mut temp = RingElement::zero(Representation::IncompleteNTT);

    let mut claims =
        HorizontallyAlignedMatrix::new_zero_preallocated(2, config.main_witness_columns);

    let mut claim_over_projection = new_vec_zero_preallocated(2);

    for i in 0..config.main_witness_columns {
        for (w, r) in witness
            .col(i)
            .iter()
            .zip(preprocessed_evaluation_points_inner.preprocessed_row.iter())
        {
            temp *= (w, r);
            claims[(0, i)] += &temp;
        }
    }

    for (c, r) in projected_witness
        .data
        .iter()
        .zip(preprocessed_evaluation_points_inner.preprocessed_row.iter())
    {
        temp *= (c, r);
        claim_over_projection[0] += &temp;
    }

    // now let's conjugate eval point in place and repeat the logic to get the claims for the conjugated witness, which will be used in the l2 and linf sumchecks
    for r in preprocessed_evaluation_points_inner
        .preprocessed_row
        .iter_mut()
    {
        r.conjugate_in_place();
    }

    for i in 0..witness.width {
        for (w, r) in witness
            .col(i)
            .iter()
            .zip(preprocessed_evaluation_points_inner.preprocessed_row.iter())
        {
            temp *= (w, r);
            claims[(1, i)] += &temp;
        }
    }

    for (c, r) in projected_witness
        .data
        .iter()
        .zip(preprocessed_evaluation_points_inner.preprocessed_row.iter())
    {
        temp *= (c, r);
        claim_over_projection[1] += &temp;
    }

    // for i in 0..config.main_witness_columns {
    //     claims[(1, i)].conjugate_in_place(); // we had evals over conjugated witness, now we have conjugated evals over a regular witness
    // }

    // // we have conjugated claims for completeness (TODO: do we really need them?)
    // for (c, r) in claim_over_projection.iter().zip(preprocessed_evaluation_points_inner.preprocessed_row.iter_mut()) {
    //     r.conjugate_in_place();
    //     temp *= (c, r);
    //     claim_over_projection[1] += &temp;
    // }

    let mut folding_challenges = new_vec_zero_preallocated(config.main_witness_columns);
    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut folding_challenges);

    let mut folded_witness = fold(&witness, &folding_challenges);

    // TODO: remove me later as we can recmpose it
    let commitment_to_folded_witness = commit_basic(crs, &folded_witness, RANK);

    let split_witness = VerticallyAlignedMatrix {
        height: folded_witness.height / 2,
        width: 2,
        data: folded_witness.data,
        used_cols: 2,
    };

    // TEST ONLU

    let commitment_to_split_witness = commit_basic(crs, &split_witness, RANK);

    let old_ck = crs.structured_ck_for_wit_dim(split_witness.height * 2);
    let composed = &(&(&*ONE - &old_ck[0].tensor_layers[0]) * &commitment_to_split_witness[(0, 0)]) + &(&old_ck[0].tensor_layers[0] * &commitment_to_split_witness[(0, 1)]);
    assert_eq!(composed, commitment_to_folded_witness[(0, 0)], "Composed commitment from the split witness does not match the commitment to the folded witness");

    // END TEST ONLY

    let mut decomposed_split_witness = VerticallyAlignedMatrix {
        height: split_witness.height,
        width: 8,
        data: new_vec_zero_preallocated(split_witness.height * 8),
        used_cols: 8,
    };

    decompose_chunks_into(&mut decomposed_split_witness.data[..split_witness.height * 2], &split_witness.data[..split_witness.height], config.decomposition_base_log, 2);

    decompose_chunks_into(&mut decomposed_split_witness.data[split_witness.height * 2..split_witness.height * 4], &split_witness.data[split_witness.height..], config.decomposition_base_log, 2);

    decompose_chunks_into(&mut decomposed_split_witness.data[split_witness.height * 4.. split_witness.height * 6], &projected_witness.data[..split_witness.height], config.decomposition_base_log, 2);

    decompose_chunks_into(&mut decomposed_split_witness.data[split_witness.height * 6..], &projected_witness.data[split_witness.height..], config.decomposition_base_log, 2);

    let decomposed_split_commitment = commit_basic(crs, &decomposed_split_witness, RANK);

    // TEST ONLY

    let composed = compose_from_decomposed(&vec![
        decomposed_split_commitment[(0, 0)].clone(),
        decomposed_split_commitment[(0, 1)].clone(),
        decomposed_split_commitment[(0, 2)].clone(),
        decomposed_split_commitment[(0, 3)].clone(),
    ], config.decomposition_base_log, 2);

    assert_eq!(composed[0], commitment_to_split_witness[(0, 0)], "Composed commitment from the decomposed split witness does not match the commitment to the split witness");

    assert_eq!(composed[1], commitment_to_split_witness[(0, 1)], "Composed commitment from the decomposed split projected witness does not match the commitment to the projected witness");

    let composed_projection = compose_from_decomposed(&vec![
        decomposed_split_commitment[(0, 4)].clone(),
        decomposed_split_commitment[(0, 5)].clone(),
        decomposed_split_commitment[(0, 6)].clone(),
        decomposed_split_commitment[(0, 7)].clone(),
    ], config.decomposition_base_log, 2);

    let unsplit_projection = &(&(&*ONE - &old_ck[0].tensor_layers[0]) * &composed_projection[0]) + &(&old_ck[0].tensor_layers[0] * &composed_projection[1]);

    assert_eq!(unsplit_projection, projection_commitment[(0, 0)], "Composed commitment from the decomposed split projected witness does not match the commitment to the projected witness");

    // END TEST ONLY

    let new_evaluation_points_inner = evaluation_points
        .iter()
        .skip(outer_points_len + 1)
        .cloned()
        .collect::<Vec<_>>();

    // let new_evaluation_points_inner = evaluation_points
    //     .iter()
    //     .skip(evaluation_points.len() - decomposed_split_commitment.height.ilog2() as usize)
    //     .cloned()
    //     .collect::<Vec<_>>();

    let new_evaluation_points_inner_expanded = PreprocessedRow::from_structured_row(&evaluation_point_to_structured_row(&new_evaluation_points_inner));

    let new_evaluation_points_inner_conjugated = new_evaluation_points_inner
        .iter()
        .map(RingElement::conjugate)
        .collect::<Vec<_>>();

    let new_evaluation_points_inner_conjugated_expanded = PreprocessedRow::from_structured_row(&evaluation_point_to_structured_row(&new_evaluation_points_inner_conjugated));

    let new_claims = commit_basic_internal(&vec![new_evaluation_points_inner_expanded, new_evaluation_points_inner_conjugated_expanded], &decomposed_split_witness, 1);



    SalsaaProof {
        projection_commitment,
        ip_l2_claim,
        ip_linf_claim,
        sumcheck_transcript: polys,
        claims, // NOT needed
        claim_over_projection, // not needed
        new_claims,
        commitment_to_folded_witness, // TODO: we can remove this as we can recompute it from the split witness, but for now let's keep it for simplicity
        decomposed_split_commitment,
        next: None,
    }

    // let claim_over_projection_matrix = commit_basic_internal(&vec![preprocessed_evaluation_points_inner], &projected_witness, 1);
    // let claim_over_projection = claim_over_projection_matrix.row(0).to_vec();

    // SalsaaProof {
    //     projection_commitment,
    //     sumcheck_transcript: polys,
    //     claims,
    //     claim_over_projection,
    //     next: None,
    // }
}

fn sample_random_binary_vector(len: usize) -> Vec<RingElement> {
    (0..len)
        .map(|_| RingElement::random_bounded_unsigned(Representation::IncompleteNTT, 2))
        .collect()
}

pub fn binary_witness_sampler() -> VerticallyAlignedMatrix<RingElement> {
    VerticallyAlignedMatrix {
        height: WITNESS_DIM,
        width: WITNESS_WIDTH,
        data: sample_random_binary_vector(WITNESS_DIM * WITNESS_WIDTH),
        // data: vec![RingElement::all(0, Representation::IncompleteNTT); WITNESS_DIM * WITNESS_WIDTH],
        used_cols: WITNESS_WIDTH,
    }
}

// ==========================
// Verifier Sumcheck Context
// ==========================

fn selector_evaluation_from_prefix(
    prefix: &Prefix,
    total_vars: usize,
) -> ElephantCell<SelectorEqEvaluation> {
    ElephantCell::new(SelectorEqEvaluation::new(
        prefix.prefix,
        prefix.length,
        total_vars,
    ))
}

pub struct VerifierSumcheckContext {
    pub witness_evaluation: ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    pub witness_conjugated_evaluation: ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    pub main_witness_selector_evaluation: ElephantCell<SelectorEqEvaluation>,
    pub projection_selector_evaluation: ElephantCell<SelectorEqEvaluation>,
    pub type1evaluations: Vec<Type1VerifierSumcheckContext>,
    pub type3evaluation: Option<Type3VerifierSumcheckContext>,
    pub l2evaluation: Option<L2VerifierSumcheckContext>,
    pub linfevaluation: Option<LinfVerifierSumcheckContext>,
    pub combiner_evaluation: ElephantCell<CombinerEvaluation<RingElement>>,
    pub field_combiner_evaluation: ElephantCell<RingToFieldCombinerEvaluation>,
    pub next: Option<Box<VerifierSumcheckContext>>,
}

pub struct L2VerifierSumcheckContext {
    pub output: ElephantCell<ProductSumcheckEvaluation>,
}

pub struct LinfVerifierSumcheckContext {
    pub all_one_constant_evaluation: ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheckEvaluation>,
    pub one_minus_wit_evaluation: ElephantCell<DiffSumcheckEvaluation>,
    pub one_minus_wit_selector_evaluation: ElephantCell<ProductSumcheckEvaluation>,
}

pub struct Type1VerifierSumcheckContext {
    pub inner_evaluation_sumcheck: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub outer_evaluation_sumcheck: ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheckEvaluation>,
}

pub struct Type3VerifierSumcheckContext {
    pub c2l_evaluation: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub c0l_evaluation: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    // TODO: this can be over fields, then then mapped to rings?. Actually, all of those can be over fields (I guess?).
    pub flattened_projection_matrix_evaluation:
        ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
    pub c2r_evaluation: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub c0r_evaluation: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub c1r_evaluation: ElephantCell<StructuredRowEvaluationLinearSumcheck<RingElement>>,
    pub lhs: ElephantCell<ProductSumcheckEvaluation>,
    pub rhs: ElephantCell<ProductSumcheckEvaluation>,
    pub output: ElephantCell<DiffSumcheckEvaluation>,
}

fn init_verifier_type_1_sumcheck(
    config: &RoundConfig,
    main_witness_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
) -> Type1VerifierSumcheckContext {
    let single_col_height = config.witness_length / 2 / config.main_witness_columns;
    let total_vars = config.witness_length.ilog2() as usize;

    let inner_evaluation_sumcheck = ElephantCell::new(
        StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            single_col_height,
            total_vars - single_col_height.ilog2() as usize,
            0,
        ),
    );

    let outer_evaluation_sumcheck = ElephantCell::new(
        BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            config.main_witness_columns,
            total_vars
                - config.main_witness_columns.ilog2() as usize
                - single_col_height.ilog2() as usize,
            single_col_height.ilog2() as usize,
        ),
    );

    let output = ElephantCell::new(ProductSumcheckEvaluation::new(
        ElephantCell::new(ProductSumcheckEvaluation::new(
            inner_evaluation_sumcheck.clone(),
            outer_evaluation_sumcheck.clone(),
        )),
        main_witness_evaluation.clone(),
    ));

    Type1VerifierSumcheckContext {
        inner_evaluation_sumcheck,
        outer_evaluation_sumcheck,
        output,
    }
}

fn init_verifier_type_3_sumcheck(
    config: &RoundConfig,
    main_witness_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
    projection_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
) -> Type3VerifierSumcheckContext {
    let c2_len = config.main_witness_columns;
    let c1_len = PROJECTION_HEIGHT;
    let single_col_height = config.witness_length / 2 / config.main_witness_columns;
    let c0_len: usize = single_col_height / (PROJECTION_HEIGHT * config.projection_ratio);
    let total_vars = config.witness_length.ilog2() as usize;

    // LEFT: prefix, c2, c0, flattened_projection_matrix (c1^T J)
    let fltr_len = (config.projection_ratio * PROJECTION_HEIGHT).ilog2() as usize;

    let flattened_projection_matrix_evaluation = ElephantCell::new(
        BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            config.projection_ratio * PROJECTION_HEIGHT,
            total_vars - fltr_len,
            0,
        ),
    );
    let c0l_evaluation = ElephantCell::new(
        StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            c0_len,
            total_vars - fltr_len - c0_len.ilog2() as usize,
            fltr_len,
        ),
    );
    let c2l_evaluation = ElephantCell::new(
        StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            c2_len,
            total_vars - fltr_len - c0_len.ilog2() as usize - c2_len.ilog2() as usize,
            fltr_len + c0_len.ilog2() as usize,
        ),
    );

    // RIGHT: prefix, c2, c0, c1
    let c1r_evaluation = ElephantCell::new(
        StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            c1_len,
            total_vars - c1_len.ilog2() as usize,
            0,
        ),
    );
    let c0r_evaluation = ElephantCell::new(
        StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            c0_len,
            total_vars - c1_len.ilog2() as usize - c0_len.ilog2() as usize,
            c1_len.ilog2() as usize,
        ),
    );
    let c2r_evaluation = ElephantCell::new(
        StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            c2_len,
            total_vars
                - c1_len.ilog2() as usize
                - c0_len.ilog2() as usize
                - c2_len.ilog2() as usize,
            c1_len.ilog2() as usize + c0_len.ilog2() as usize,
        ),
    );

    let lhs = ElephantCell::new(ProductSumcheckEvaluation::new(
        c2l_evaluation.clone(),
        ElephantCell::new(ProductSumcheckEvaluation::new(
            c0l_evaluation.clone(),
            ElephantCell::new(ProductSumcheckEvaluation::new(
                flattened_projection_matrix_evaluation.clone(),
                main_witness_evaluation.clone(),
            )),
        )),
    ));

    let rhs = ElephantCell::new(ProductSumcheckEvaluation::new(
        c2r_evaluation.clone(),
        ElephantCell::new(ProductSumcheckEvaluation::new(
            c0r_evaluation.clone(),
            ElephantCell::new(ProductSumcheckEvaluation::new(
                c1r_evaluation.clone(),
                projection_evaluation.clone(),
            )),
        )),
    ));

    let output = ElephantCell::new(DiffSumcheckEvaluation::new(lhs.clone(), rhs.clone()));

    Type3VerifierSumcheckContext {
        c2l_evaluation,
        c0l_evaluation,
        flattened_projection_matrix_evaluation,
        c2r_evaluation,
        c0r_evaluation,
        c1r_evaluation,
        lhs,
        rhs,
        output,
    }
}

fn init_verifier_l2_sumcheck(
    witness_conjugated_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
    main_witness_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
) -> L2VerifierSumcheckContext {
    L2VerifierSumcheckContext {
        output: ElephantCell::new(ProductSumcheckEvaluation::new(
            witness_conjugated_evaluation,
            main_witness_evaluation,
        )),
    }
}

fn init_verifier_linf_sumcheck(
    witness_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
    main_witness_selector_evaluation: ElephantCell<
        dyn EvaluationSumcheckData<Element = RingElement>,
    >,
    witness_conjugated_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
) -> LinfVerifierSumcheckContext {
    let all_one_constant_evaluation =
        ElephantCell::new(FakeEvaluationLinearSumcheck::<RingElement>::new());
    all_one_constant_evaluation
        .borrow_mut()
        .set_result(ALL_ONE_COEFFS.clone());

    let one_minus_wit_evaluation = ElephantCell::new(DiffSumcheckEvaluation::new(
        all_one_constant_evaluation.clone(),
        witness_evaluation,
    ));

    let one_minus_wit_selector_evaluation = ElephantCell::new(ProductSumcheckEvaluation::new(
        main_witness_selector_evaluation,
        one_minus_wit_evaluation.clone(),
    ));

    let output = ElephantCell::new(ProductSumcheckEvaluation::new(
        witness_conjugated_evaluation.clone(),
        one_minus_wit_selector_evaluation.clone(),
    ));

    LinfVerifierSumcheckContext {
        one_minus_wit_evaluation,
        one_minus_wit_selector_evaluation,
        all_one_constant_evaluation,
        output,
    }
}

pub fn init_verifier_sumcheck(config: &RoundConfig) -> VerifierSumcheckContext {
    let total_vars = config.witness_length.ilog2() as usize;

    let witness_evaluation = ElephantCell::new(FakeEvaluationLinearSumcheck::<RingElement>::new());
    let witness_conjugated_evaluation =
        ElephantCell::new(FakeEvaluationLinearSumcheck::<RingElement>::new());

    let main_witness_selector_evaluation =
        selector_evaluation_from_prefix(&config.main_witness_prefix, total_vars);
    let projection_selector_evaluation =
        selector_evaluation_from_prefix(&config.projection_prefix, total_vars);

    let main_witness_evaluation: ElephantCell<ProductSumcheckEvaluation> =
        ElephantCell::new(ProductSumcheckEvaluation::new(
            witness_evaluation.clone(),
            main_witness_selector_evaluation.clone(),
        ));
    let projection_eval: ElephantCell<ProductSumcheckEvaluation> =
        ElephantCell::new(ProductSumcheckEvaluation::new(
            witness_evaluation.clone(),
            projection_selector_evaluation.clone(),
        ));

    let type1evaluations = (0..config.inner_evaluation_claims)
        .map(|_| init_verifier_type_1_sumcheck(config, main_witness_evaluation.clone()))
        .collect::<Vec<_>>();

    let type3evaluation = if config.projection_ratio > 0 {
        Some(init_verifier_type_3_sumcheck(
            config,
            main_witness_evaluation.clone(),
            projection_eval.clone(),
        ))
    } else {
        None
    };

    let l2evaluation = if config.l2 {
        Some(init_verifier_l2_sumcheck(
            witness_conjugated_evaluation.clone(),
            main_witness_evaluation.clone(),
        ))
    } else {
        None
    };

    let linfevaluation = if config.exact_binariness {
        Some(init_verifier_linf_sumcheck(
            witness_evaluation.clone(),
            main_witness_selector_evaluation.clone(),
            witness_conjugated_evaluation.clone(),
        ))
    } else {
        None
    };

    let mut all_outputs: Vec<ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>> =
        vec![];
    for type1 in &type1evaluations {
        all_outputs.push(type1.output.clone());
    }
    if let Some(type3) = &type3evaluation {
        all_outputs.push(type3.output.clone());
    }
    if let Some(l2) = &l2evaluation {
        all_outputs.push(l2.output.clone());
    }
    if let Some(linf) = &linfevaluation {
        all_outputs.push(linf.output.clone());
    }

    let combiner_evaluation = ElephantCell::new(CombinerEvaluation::new(all_outputs));
    let field_combiner_evaluation = ElephantCell::new(RingToFieldCombinerEvaluation::new(
        combiner_evaluation.clone(),
    ));

    VerifierSumcheckContext {
        witness_evaluation,
        witness_conjugated_evaluation,
        main_witness_selector_evaluation,
        projection_selector_evaluation,
        type1evaluations,
        type3evaluation,
        l2evaluation,
        linfevaluation,
        combiner_evaluation,
        field_combiner_evaluation,
        next: config
            .next
            .as_ref()
            .map(|next_config| Box::new(init_verifier_sumcheck(next_config))),
    }
}

/// Computes the batched claim from individual sumcheck claims.
/// Type1 sumchecks are product sumchecks with claims = <evaluation_outer, column_claims>,
/// Type3 is a diff sumcheck with claim = 0.
fn batch_claims(
    config: &RoundConfig,
    claims: &HorizontallyAlignedMatrix<RingElement>,
    evaluation_points_outer: &[RingElement],
    ip_l2_claim: Option<&RingElement>,
    ip_linf_claim: Option<&RingElement>,
    combination: &[RingElement],
) -> RingElement {
    let mut batched_claim = RingElement::zero(Representation::IncompleteNTT);
    let mut idx = 0;

    // Type1 sumchecks: claim = <evaluation_outer, column_claims[i]>
    for i in 0..config.inner_evaluation_claims {
        let mut type1_claim = RingElement::zero(Representation::IncompleteNTT);
        for (c, r) in claims.row(i).iter().zip(evaluation_points_outer.iter()) {
            type1_claim += &(c * r);
        }
        let mut weighted = type1_claim;
        weighted *= &combination[idx];
        batched_claim += &weighted;
        idx += 1;
    }

    // Type3: diff sumcheck, claim = 0
    if config.projection_ratio > 0 {
        // zero claim, nothing to add
        idx += 1;
    }

    // L2: product sumcheck over conjugated witness and selected witness.
    if config.l2 {
        let mut weighted = ip_l2_claim
            .expect("Missing l2 claim in proof while l2 constraint is enabled")
            .clone();
        weighted *= &combination[idx];
        batched_claim += &weighted;
        idx += 1;
    }

    // Linf: exact-binariness sumcheck claim.
    if config.exact_binariness {
        let mut weighted = ip_linf_claim
            .expect("Missing linf claim in proof while exact_binariness is enabled")
            .clone();
        weighted *= &combination[idx];
        batched_claim += &weighted;
        idx += 1;
    }

    assert_eq!(
        idx,
        combination.len(),
        "batch_claims: index mismatch with combination length"
    );
    batched_claim
}

impl VerifierSumcheckContext {
    pub fn load_data(
        &mut self,
        config: &RoundConfig,
        proof: &SalsaaProof,
        evaluation_points_ring: &[RingElement],
        evaluation_points_inner: &[StructuredRow],
        evaluation_points_outer: &[RingElement],
        batching_challenges: &BatchingChallenges,
        projection_matrix: &ProjectionMatrix,
        combination: &[RingElement],
        qe: [QuadraticExtension; HALF_DEGREE],
    ) {
        let outer_points_len = config.main_witness_columns.ilog2() as usize + 1;
        let outer_points = &evaluation_points_ring[0..outer_points_len].to_vec();
        let outer_points_expanded =
            PreprocessedRow::from_structured_row(&evaluation_point_to_structured_row(outer_points))
                .preprocessed_row;

        let mut temp = ZERO.clone();

        let mut claim_over_witness = ZERO.clone();
        for (claim, outer) in proof.claims.row(0).iter().zip(outer_points_expanded.iter()) {
            temp *= (claim, outer);
            claim_over_witness += &temp;
        }

        temp *= (
            proof.claim_over_projection.get(0).unwrap(),
            &outer_points_expanded[config.main_witness_columns],
        );
        claim_over_witness += &temp;

        let mut main_cols_points = evaluation_points_ring[1..outer_points_len].to_vec();
        for r in main_cols_points.iter_mut() {
            r.conjugate_in_place();
        }
        let main_cols_points_expanded = PreprocessedRow::from_structured_row(
            &evaluation_point_to_structured_row(&main_cols_points),
        )
        .preprocessed_row;

        let mut claim_over_conjugated_witness = ZERO.clone();
        for (claim, outer) in proof
            .claims
            .row(1)
            .iter()
            .zip(main_cols_points_expanded.iter())
        {
            temp *= (claim, outer);
            claim_over_conjugated_witness += &temp;
        }
        claim_over_conjugated_witness.conjugate_in_place();

        self.witness_evaluation
            .borrow_mut()
            .set_result(claim_over_witness);
        self.witness_conjugated_evaluation
            .borrow_mut()
            .set_result(claim_over_conjugated_witness);

        for (i, type1_eval) in self.type1evaluations.iter().enumerate() {
            type1_eval
                .inner_evaluation_sumcheck
                .borrow_mut()
                .load_from(evaluation_points_inner[i].clone());
            type1_eval
                .outer_evaluation_sumcheck
                .borrow_mut()
                .load_from(evaluation_points_outer);
        }

        if let Some(type3_eval) = &mut self.type3evaluation {
            let c1_expanded = PreprocessedRow::from_structured_row(&batching_challenges.c1);

            let flattened_projection =
                projection_flatter_1_times_matrix(projection_matrix, &c1_expanded);
            let mut flattened_projection_ring =
                new_vec_zero_preallocated(flattened_projection.len());
            for (i, el) in flattened_projection.iter().enumerate() {
                field_to_ring_element_into(&mut flattened_projection_ring[i], el);
                flattened_projection_ring[i].from_homogenized_field_extensions_to_incomplete_ntt();
            }

            type3_eval
                .flattened_projection_matrix_evaluation
                .borrow_mut()
                .load_from(&flattened_projection_ring);
            type3_eval
                .c0l_evaluation
                .borrow_mut()
                .load_from(batching_challenges.c0.clone());
            type3_eval
                .c2l_evaluation
                .borrow_mut()
                .load_from(batching_challenges.c2.clone());
            type3_eval
                .c0r_evaluation
                .borrow_mut()
                .load_from(batching_challenges.c0.clone());
            type3_eval
                .c1r_evaluation
                .borrow_mut()
                .load_from(batching_challenges.c1.clone());
            type3_eval
                .c2r_evaluation
                .borrow_mut()
                .load_from(batching_challenges.c2.clone());
        }

        self.combiner_evaluation
            .borrow_mut()
            .load_challenges_from(combination);
        self.field_combiner_evaluation
            .borrow_mut()
            .load_challenges_from(qe);
    }
}

pub fn verifier_round(
    config: &RoundConfig,
    crs: &CRS,
    verifier_context: &mut VerifierSumcheckContext,
    proof: &SalsaaProof,
    evaluation_points_inner: &[StructuredRow],
    claims: &HorizontallyAlignedMatrix<RingElement>,
    hash_wrapper: &mut HashWrapper,
) {
    // Replay prover's Fiat-Shamir: sample projection matrix, batching challenges
    let mut projection_matrix =
        ProjectionMatrix::new(config.main_witness_columns, PROJECTION_HEIGHT);
    projection_matrix.sample(hash_wrapper);

    let batching_challenges = BatchingChallenges::sample(config, hash_wrapper);

    let mut evaluation_points_outer = new_vec_zero_preallocated(config.main_witness_columns);
    hash_wrapper.sample_ring_element_vec_into(&mut evaluation_points_outer);

    // Sample random batching coefficients (same Fiat-Shamir as prover)
    let num_sumchecks = verifier_context
        .combiner_evaluation
        .borrow()
        .sumchecks_count();
    let mut combination = new_vec_zero_preallocated(num_sumchecks);
    hash_wrapper.sample_ring_element_vec_into(&mut combination);

    let mut combination_to_field = RingElement::zero(Representation::IncompleteNTT);
    hash_wrapper.sample_ring_element_into(&mut combination_to_field);
    combination_to_field.from_incomplete_ntt_to_homogenized_field_extensions();
    let qe = combination_to_field.split_into_quadratic_extensions();

    // Compute expected batched claim over field
    let batched_claim = batch_claims(
        config,
        claims,
        &evaluation_points_outer,
        proof.ip_l2_claim.as_ref(),
        proof.ip_linf_claim.as_ref(),
        &combination,
    );

    let mut batched_claim_over_field = {
        let batched_claim_field = {
            let mut temp = batched_claim.clone();
            temp.from_incomplete_ntt_to_homogenized_field_extensions();
            temp
        };
        let mut temp = batched_claim_field.split_into_quadratic_extensions();
        let mut result = QuadraticExtension::zero();
        for i in 0..HALF_DEGREE {
            temp[i] *= &qe[i];
            result += &temp[i];
        }
        result
    };

    // Verify each sumcheck round: poly(0) + poly(1) == running_claim
    let mut num_vars = proof.sumcheck_transcript.len();
    let mut evaluation_points_field: Vec<QuadraticExtension> = Vec::new();
    let mut evaluation_points_ring: Vec<RingElement> = Vec::new();

    let mut round_idx = 0;
    while num_vars > 0 {
        num_vars -= 1;
        let poly_over_field = &proof.sumcheck_transcript[round_idx];

        if round_idx < 3 {}

        hash_wrapper.update_with_quadratic_extension_slice(&poly_over_field.coefficients);

        assert_eq!(
            poly_over_field.at_zero() + poly_over_field.at_one(),
            batched_claim_over_field,
            "Sumcheck round {}: poly(0) + poly(1) != running claim",
            round_idx,
        );

        let mut f = QuadraticExtension::zero();
        hash_wrapper.sample_field_element_into(&mut f);

        if round_idx < 3 {}

        batched_claim_over_field = poly_over_field.at(&f);

        evaluation_points_field.push(f);

        let mut r = RingElement::zero(Representation::IncompleteNTT);
        field_to_ring_element_into(&mut r, &f);
        r.from_homogenized_field_extensions_to_incomplete_ntt();
        evaluation_points_ring.push(r);

        round_idx += 1;
    }

    // let sck_for_current_witness = crs.structured_ck_for_wit_dim(config.witness_length);

    // verify evaluation claims (TODO: change to recompute them from the proof data)
    let recomposed_claims = HorizontallyAlignedMatrix {
        height: proof.claims.height,
        width: proof.claims.width / 2,
        data: compose_from_decomposed(&proof.claims.data, config.decomposition_base_log, 2),
    };

    // let new_witness_height = config.witness_length / 2;
    // let layer = evaluation_points_outer.last().unwrap().clone();
    //         let outer_points_len = config.main_witness_columns.ilog2() as usize + 1;
    let layer = &evaluation_points_ring[2];

    // assert_eq!(
    //     recomposed_claims,
    //     proof.claims.clone(),
    //     "Recomposed claims from the proof do not match the original claims"
    // );
   
   assert_eq!(
        proof.claims[(0,0)],
        &(&(&*ONE - layer) * &recomposed_claims[(0, 0)]) + &(layer * &recomposed_claims[(0, 1)]),
        "Recomposed claim for the witness does not match the original claim"
    );

    verifier_context.load_data(
        config,
        proof,
        &evaluation_points_ring,
        evaluation_points_inner,
        &evaluation_points_outer,
        &batching_challenges,
        &projection_matrix,
        &combination,
        qe,
    );

    // let eval_l2 = verifier_context.l2evaluation.as_ref().unwrap().output.eva

    // Final evaluation check: the tree evaluated at the random points must match
    // the last round's batched_claim_over_field
    let verifier_eval = verifier_context
        .field_combiner_evaluation
        .borrow_mut()
        .evaluate_at_ring_point(&evaluation_points_ring)
        .clone();

    assert_eq!(
        verifier_eval, batched_claim_over_field,
        "Verifier final check failed: tree evaluation does not match sumcheck claim"
    );
}

pub fn execute() {
    println!("Generating CRS...");

    let crs = CRS::gen_crs(WITNESS_DIM, 8);

    let mut sumcheck_context = init_prover_sumcheck(&crs, &CONFIG);

    let witness = binary_witness_sampler();

    println!("===== COMMITTING WITNESS =====");
    let start = std::time::Instant::now();

    let commitment = commit_basic(&crs, &witness, RANK);

    let commit_duration = start.elapsed().as_nanos();
    println!("TOTAL Commit time: {:?} ns", commit_duration);

    let evaluation_points_inner = vec![evaluation_point_to_structured_row(
        &range(0, WITNESS_DIM.ilog2() as usize)
            .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 2))
            .collect::<Vec<RingElement>>(),
    )];

    let preprocessed_row_inner =
        PreprocessedRow::from_structured_row(evaluation_points_inner.get(0).unwrap());
    // TODO rename commit_basic_internal as we abuse it sometimes
    let claims = commit_basic_internal(&vec![preprocessed_row_inner], &witness, 1);

    println!("===== STARTING PROVER =====");
    let start = std::time::Instant::now();
    let proof = prover_round(
        &crs,
        &commitment,
        &witness,
        &CONFIG,
        &mut sumcheck_context,
        &evaluation_points_inner,
        &claims,
        CONFIG.exact_binariness,
        &mut HashWrapper::new(),
    );
    let prove_duration = start.elapsed().as_millis();
    println!("TOTAL Prove time: {:?} ms", prove_duration);

    println!("===== STARTING VERIFIER =====");
    let start = std::time::Instant::now();
    let mut verifier_context = init_verifier_sumcheck(&CONFIG);
    verifier_round(
        &CONFIG,
        &crs,
        &mut verifier_context,
        &proof,
        &evaluation_points_inner,
        &claims,
        &mut HashWrapper::new(),
    );
    let verify_duration = start.elapsed().as_nanos();
    println!("TOTAL Verify time: {:?} ns", verify_duration);
    println!("===== VERIFICATION PASSED =====");
}
