use num::range;
use rand::rand_core::le;

use crate::{
    common::{
        arithmetic::field_to_ring_element_into,
        config,
        hash::HashWrapper,
        matrix::{VerticallyAlignedMatrix, new_vec_zero_preallocated},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        sampling::sample_random_short_vector,
        structured_row::{PreprocessedRow, StructuredRow}, sumcheck_element::SumcheckElement,
    },
    protocol::{
        commitment::{self, BasicCommitment, Prefix, commit_basic}, config::paste_by_prefix, crs::CRS, open::evaluation_point_to_structured_row, project::{self, prepare_i16_witness, project}, sumcheck_utils::{
            combiner::Combiner, common::{HighOrderSumcheckData, SumcheckBaseData}, diff::DiffSumcheck, elephant_cell::ElephantCell, linear::LinearSumcheck, polynomial::Polynomial, product::ProductSumcheck, ring_to_field_combiner::RingToFieldCombiner, selector_eq::SelectorEq, sum
        }, sumchecks::helpers::{projection_flatter_1_times_matrix, sumcheck_from_prefix}
    },
};

const WITNESS_DIM: usize = 2usize.pow(16);
const WITNESS_WIDTH: usize = 2usize;
const RANK: usize = 8;

pub struct SalsaaProof {
    projection_commitment: BasicCommitment,
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
    pub next: Option<Box<RoundConfig>>,
}

const NUM_COLUMNS_INITIAL: usize = 2;

const PROJECTION_HEIGHT: usize = 256;

// All configs shall be auto-derived, but we keep this struct for clarity for now
const CONFIG: RoundConfig = RoundConfig {
    witness_length: WITNESS_DIM * WITNESS_WIDTH * 2, // we ``bloat up'' the witness times two to account to the projection
    exact_binariness: false,
    l2: false,
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
    inner_evaluation_claims: 0, // for VDF one will be enough
};

// ==== Prover Sumcheck context initialization ====

pub struct ProverSumcheckContext {
    pub witness_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub main_witness_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub projection_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub type1sumcheck: Vec<Type1ProverSumcheckContext>, // for verifying inner evaluation points
    pub type3sumcheck: Option<Type3ProverSumcheckContext>, // for verifying the projection
    // pub type5sumcheck: Option<Type5ProverSumcheckContext>, // for verifying the l2 norm of the witness, only used when exact_binariness is false TODO
    // pub type6sumcheck: Option<Type6ProvserSumcheckContext>, // for verifying ecact binariness, only used when exact_binariness is true TODO
    pub combiner: ElephantCell<Combiner<RingElement>>,
    pub field_combiner: ElephantCell<RingToFieldCombiner>,
    pub next: Option<Box<ProverSumcheckContext>>,
}

pub struct Type1ProverSumcheckContext {
    pub inner_evaluation_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
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
    let inner_evaluation_sumcheck = ElephantCell::new(LinearSumcheck::new(single_col_height));
    // we view MLE[w](evaluation_points_inner) as a sumcheck
    let output = ElephantCell::new(ProductSumcheck::new(
        inner_evaluation_sumcheck.clone(),
        main_witness_sumcheck.clone(),
    ));

    Type1ProverSumcheckContext {
        inner_evaluation_sumcheck,
        output,
    }
}

fn init_prover_type_3_sumcheck(
    config: &RoundConfig,
    main_witness_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
    projection_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
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
                main_witness_sumcheck.clone()
            ))
        ))
    ));

    let rhs = ElephantCell::new(ProductSumcheck::new(
        c2r_sumcheck.clone(),
        ElephantCell::new(ProductSumcheck::new(
            c0r_sumcheck.clone(),
            ElephantCell::new(ProductSumcheck::new(
                c1r_sumcheck.clone(),
                projection_selector_sumcheck.clone(),
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

pub fn init_prover_sumcheck(crs: &CRS, config: &RoundConfig) -> ProverSumcheckContext {
    let witness_sumcheck = ElephantCell::new(LinearSumcheck::new(config.witness_length));

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

    let type1sumcheck = (0..config.inner_evaluation_claims)
        .map(|_| init_prover_type_1_sumcheck(config, main_witness_sumcheck.clone()))
        .collect::<Vec<_>>();

    let type3sumcheck = if config.projection_ratio > 0 {
        Some(init_prover_type_3_sumcheck(
            config,
            main_witness_sumcheck.clone(),
            projection_selector_sumcheck.clone(),
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

    let combiner = ElephantCell::new(Combiner::new(all_outputs.clone()));
    let field_combiner = ElephantCell::new(RingToFieldCombiner::new(combiner.clone()));

    ProverSumcheckContext {
        witness_sumcheck,
        main_witness_selector_sumcheck,
        projection_selector_sumcheck,
        type1sumcheck,
        type3sumcheck,
        combiner,
        field_combiner,
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
    }

    pub fn load_data(
        &mut self,
        witness: &Vec<RingElement>,
        evaluation_points_inner: &Vec<StructuredRow>,
        projection_matrix: &ProjectionMatrix,
        projection_batching_challenges: &Option<BatchingChallenges>,
    ) {
        self.witness_sumcheck.borrow_mut().load_from(&witness);
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
            println!("Loading data for type1 sumcheck {}", i);
            let evaluation_points_inner_expanded =
                PreprocessedRow::from_structured_row(&evaluation_points_inner[i]);
            type1
                .inner_evaluation_sumcheck
                .borrow_mut()
                .load_from(&evaluation_points_inner_expanded.preprocessed_row);
        }
        println!("Finished loading data for type1 sumchecks");
    }
}

pub fn prover_round(
    crs: &CRS,
    commitmens: &BasicCommitment,
    witness: &VerticallyAlignedMatrix<RingElement>,
    config: &RoundConfig,
    sumcheck_context: &mut ProverSumcheckContext,
    evaluation_points_inner: &Vec<StructuredRow>,
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

    let mut new_witness = new_vec_zero_preallocated(witness.data.len() * 2);

    paste_by_prefix(&mut new_witness, &witness.data, &config.main_witness_prefix);
    paste_by_prefix(
        &mut new_witness,
        &projected_witness.data,
        &config.projection_prefix,
    );

    sumcheck_context.load_data(
        &new_witness,
        evaluation_points_inner,
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


    // let c = sumcheck_context.type3sumcheck.as_ref().borrow_mut().claim();
    // println!("Combiner claim: {:?}", c);

    sumcheck_context.type3sumcheck.as_ref().map(|type3| {
        let claim = type3.output.borrow_mut().claim();
        let lhs = type3.lhs.borrow_mut().claim();
        let rhs = type3.rhs.borrow_mut().claim();

        println!("Type3 claim: {:?}", claim);
        println!("Type3 lhs claim: {:?}", lhs);
        println!("Type3 rhs claim: {:?}", rhs);
    });

    println!("===== STARTING SUMCHECK =====");

    let mut num_vars = sumcheck_context.combiner.borrow().variable_count();
    println!(
        "    sumcheck num_vars: {}, hypercube_size: {}",
        num_vars,
        1u64 << (num_vars - 1)
    ); 

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

    println!("Polynomial time: {:?} ms, Evaluation time: {:?} ms", time_poly, time_eval);



    // let sumcheck_transcript = 

    // SalsaaProof {
    //     projection_commitment,
    // };
    panic!("Not implemented yet");
}

pub fn binary_witness_sampler() -> VerticallyAlignedMatrix<RingElement> {
    VerticallyAlignedMatrix {
        height: WITNESS_DIM,
        width: WITNESS_WIDTH,
        data: sample_random_short_vector(
            WITNESS_DIM * WITNESS_WIDTH,
            2,
            Representation::IncompleteNTT,
        ),
        used_cols: WITNESS_WIDTH,
    }
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
    println!("===== STARTING PROVER =====");
    let start = std::time::Instant::now();
    let proof = prover_round(
        &crs,
        &commitment,
        &witness,
        &CONFIG,
        &mut sumcheck_context,
        &evaluation_points_inner,
        CONFIG.exact_binariness,
        &mut HashWrapper::new(),
    );
}
