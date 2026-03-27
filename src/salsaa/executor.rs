use std::process::Output;
use std::sync::LazyLock;

use num::range;
use rand::rand_core::le;

use crate::{
    common::{
        arithmetic::{field_to_ring_element_into, ALL_ONE_COEFFS, ONE, ZERO},
        config::{self, DEGREE, HALF_DEGREE, MOD_Q},
        decomposition::{compose_from_decomposed, decompose_chunks_into},
        hash::HashWrapper,
        matrix::{new_vec_zero_preallocated, HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        sampling::sample_random_short_vector,
        structured_row::{PreprocessedRow, StructuredRow},
        sumcheck_element::SumcheckElement,
    },
    protocol::{
        commitment::{self, commit_basic, commit_basic_internal, BasicCommitment, Prefix},
        config::{paste_by_prefix, to_kb, SizeableProof},
        crs::{self, CRS},
        fold::fold,
        open::{claim, evaluation_point_to_structured_row},
        project::{self, prepare_i16_witness, project},
        sumcheck_utils::{
            combiner::{Combiner, CombinerEvaluation},
            common::{EvaluationSumcheckData, HighOrderSumcheckData, SumcheckBaseData},
            diff::{DiffSumcheck, DiffSumcheckEvaluation},
            elephant_cell::ElephantCell,
            linear::{
                BasicEvaluationLinearSumcheck, FakeEvaluationLinearSumcheck, LinearSumcheck,
                RingToFieldWrapperEvaluation, StructuredRowEvaluationLinearSumcheck,
            },
            polynomial::Polynomial,
            product::{ProductSumcheck, ProductSumcheckEvaluation},
            ring_to_field_combiner::{RingToFieldCombiner, RingToFieldCombinerEvaluation},
            selector_eq::{SelectorEq, SelectorEqEvaluation},
            sum,
        },
        sumchecks::helpers::{projection_flatter_1_times_matrix, sumcheck_from_prefix},
    },
};

const DEBUG: bool = false;

// 2^7 (degree) * 2^{19} (witness height) * 2 (witness width) = 2^27
const WITNESS_DIM: usize = 2usize.pow(14);
const WITNESS_WIDTH: usize = 2usize;
const RANK: usize = 8;

pub struct SalsaaProofCommon {
    sumcheck_transcript: Vec<Polynomial<QuadraticExtension>>,
    ip_l2_claim: Option<RingElement>,
    ip_linf_claim: Option<RingElement>,
    claims: HorizontallyAlignedMatrix<RingElement>,
}

pub enum SalsaaProof {
    Intermediate {
        projection_commitment: BasicCommitment,
        common: SalsaaProofCommon,
        new_claims: HorizontallyAlignedMatrix<RingElement>,
        decomposed_split_commitment: BasicCommitment,
        next: Box<SalsaaProof>,
        claim_over_projection: Vec<RingElement>,
    },
    IntermediateUnstructured {
        common: SalsaaProofCommon,
        new_claims: Vec<RingElement>,
        decomposed_split_commitment: BasicCommitment,
        next: Box<SalsaaProof>,
        projection_image_ct: [u64; 256],
        projection_image_batched: [RingElement; 2],
    },
    Last {
        common: SalsaaProofCommon,
        folded_witness: Vec<RingElement>,
        projection_image_ct: [u64; 256],
        projection_image_batched: [RingElement; 2],
    },
}

impl std::ops::Deref for SalsaaProof {
    type Target = SalsaaProofCommon;
    fn deref(&self) -> &SalsaaProofCommon {
        match self {
            SalsaaProof::Intermediate { common, .. } => common,
            SalsaaProof::Last { common, .. } => common,
            SalsaaProof::IntermediateUnstructured { common, .. } => common,
        }
    }
}

fn ring_vec_size(v: &[RingElement]) -> usize {
    v.iter().map(|e| e.size_in_bits()).sum()
}

impl SizeableProof for SalsaaProof {
    fn size_in_bits(&self) -> usize {
        let common = &**self;

        // Sumcheck transcript (polynomials)
        let mut polys_size = 0;
        for poly in &common.sumcheck_transcript {
            for coeff in &poly.coefficients[0..poly.num_coefficients] {
                polys_size += coeff.size_in_bits();
            }
        }
        println!("  Polys: {:.2} KB", to_kb(polys_size));

        // Claims matrix
        let claims_size = ring_vec_size(&common.claims.data);
        println!("  Claims: {:.2} KB", to_kb(claims_size));

        // Claim over projection
        let proj_claim_size = match self {
            SalsaaProof::Intermediate {
                claim_over_projection,
                ..
            } => ring_vec_size(claim_over_projection),
            SalsaaProof::Last { .. } => 0,
            SalsaaProof::IntermediateUnstructured { .. } => 0,
        };

        println!("  Claim over projection: {:.2} KB", to_kb(proj_claim_size));

        // Projection commitment
        let proj_commit_size = match self {
            SalsaaProof::Intermediate {
                projection_commitment,
                ..
            } => ring_vec_size(&projection_commitment.data),
            SalsaaProof::Last { .. } => 0,
            SalsaaProof::IntermediateUnstructured { .. } => 0,
        };

        println!("  Projection commitment: {:.2} KB", to_kb(proj_commit_size));

        // Norm claims
        let l2_size = common.ip_l2_claim.as_ref().map_or(0, |c| c.size_in_bits());
        let linf_size = common
            .ip_linf_claim
            .as_ref()
            .map_or(0, |c| c.size_in_bits());
        println!(
            "  L2 claim: {:.2} KB, Linf claim: {:.2} KB",
            to_kb(l2_size),
            to_kb(linf_size)
        );

        let mut round_size =
            polys_size + claims_size + proj_claim_size + proj_commit_size + l2_size + linf_size;

        match self {
            SalsaaProof::Intermediate {
                new_claims,
                decomposed_split_commitment,
                next,
                ..
            } => {
                let new_claims_size = ring_vec_size(&new_claims.data);
                println!("  New claims: {:.2} KB", to_kb(new_claims_size));

                let decomp_commit_size = ring_vec_size(&decomposed_split_commitment.data);
                println!(
                    "  Decomposed split commitment: {:.2} KB",
                    to_kb(decomp_commit_size)
                );

                round_size += new_claims_size + decomp_commit_size;
                println!("  Round total: {:.2} KB", to_kb(round_size));

                round_size + next.size_in_bits()
            }
            SalsaaProof::Last {
                folded_witness,
                projection_image_batched,
                ..
            } => {
                let folded_size = ring_vec_size(folded_witness);
                println!("  Folded witness: {:.2} KB", to_kb(folded_size));

                let projection_image_size = 256 * 64; // over estimated
                println!("  Projection image: {:.2} KB", to_kb(projection_image_size));

                let batched_projection_size = ring_vec_size(projection_image_batched);
                println!(
                    "  Batched projection: {:.2} KB",
                    to_kb(batched_projection_size)
                );

                round_size += folded_size + batched_projection_size + projection_image_size;
                println!("  Round total (last): {:.2} KB", to_kb(round_size));

                round_size
            }
            SalsaaProof::IntermediateUnstructured {
                new_claims,
                decomposed_split_commitment,
                projection_image_batched,
                next,
                ..
            } => {
                let new_claims_size = ring_vec_size(new_claims);
                println!("  New claims: {:.2} KB", to_kb(new_claims_size));

                let decomp_commit_size = ring_vec_size(&decomposed_split_commitment.data);
                println!(
                    "  Decomposed split commitment: {:.2} KB",
                    to_kb(decomp_commit_size)
                );

                let projection_image_size = 256 * 64; // over estimated
                println!("  Projection image: {:.2} KB", to_kb(projection_image_size));

                let batched_projection_size = ring_vec_size(projection_image_batched);
                println!(
                    "  Batched projection: {:.2} KB",
                    to_kb(batched_projection_size)
                );

                round_size += new_claims_size
                    + decomp_commit_size
                    + projection_image_size
                    + batched_projection_size;
                println!("  Round total: {:.2} KB", to_kb(round_size));

                round_size + next.size_in_bits()
            }
        }
    }
}

#[derive(Clone)]
pub struct RoundConfigCommon {
    pub main_witness_prefix: Prefix,
    pub main_witness_columns: usize,
    pub witness_length: usize,
    pub exact_binariness: bool, // whether the proof should be for exact binariness
    pub vdf: bool,              // for the first round
    pub l2: bool,               // whether the proof should be for l2 norm of the witness
    pub inner_evaluation_claims: usize, // how many inner evaluation claims we want to make, this determines the number of type1 sumchecks we need
}

#[derive(Clone)]
pub enum RoundConfig {
    Intermediate {
        common: RoundConfigCommon,
        decomposition_base_log: u64,
        projection_ratio: usize, // set 0 for no projection
        projection_prefix: Prefix,
        next: Box<RoundConfig>,
    },
    IntermediateUnstructured {
        common: RoundConfigCommon,
        decomposition_base_log: u64,
        next: Box<RoundConfig>,
    },
    Last {
        common: RoundConfigCommon,
    },
}

impl std::ops::Deref for RoundConfig {
    type Target = RoundConfigCommon;
    fn deref(&self) -> &RoundConfigCommon {
        match self {
            RoundConfig::Intermediate { common, .. } => common,
            RoundConfig::Last { common, .. } => common,
            RoundConfig::IntermediateUnstructured { common, .. } => common,
        }
    }
}

impl RoundConfig {
    pub fn is_last(&self) -> bool {
        matches!(self, RoundConfig::Last { .. })
    }
}

const NUM_COLUMNS_INITIAL: usize = 2;

const PROJECTION_HEIGHT: usize = 256;

/// Recursively builds the round config chain.
/// - First round: uses NUM_COLUMNS_INITIAL columns, projection_ratio=2, VDF+exact_binariness enabled.
/// - Subsequent rounds: 8 columns, projection_ratio=8, L2 enabled.
/// - Recursion stops when the *next* round's single_col_height would be < PROJECTION_HEIGHT * projection_ratio
///   (i.e., the next round couldn't support projection).
fn build_round_config(witness_length: usize, is_first_round: bool) -> RoundConfig {
    let main_witness_columns = if is_first_round {
        NUM_COLUMNS_INITIAL
    } else {
        8
    };
    let projection_ratio = if is_first_round { 2 } else { 8 };

    let single_col_height = witness_length / 2 / main_witness_columns;
    // After fold+split+decompose, next round's column height = single_col_height / 2
    let next_single_col_height = single_col_height / 2;
    let next_main_witness_columns = 8usize;
    let next_projection_ratio = 8usize;
    let can_recurse = next_single_col_height >= PROJECTION_HEIGHT * next_projection_ratio;
    println!("Building round config: witness_length={}, single_col_height={}, next_single_col_height={}, can_recurse={}", witness_length, single_col_height, next_single_col_height, can_recurse);

    let inner_evaluation_claims = if is_first_round { 0 } else { 2 };

    let common = RoundConfigCommon {
        witness_length,
        exact_binariness: is_first_round,
        l2: !is_first_round,
        vdf: is_first_round,
        inner_evaluation_claims,
        main_witness_columns,
        main_witness_prefix: Prefix {
            prefix: 0b0,
            length: 1,
        },
    };

    if can_recurse {
        let next_witness_length = next_single_col_height * next_main_witness_columns * 2;
        RoundConfig::Intermediate {
            common,
            decomposition_base_log: 8,
            projection_ratio,
            projection_prefix: Prefix {
                prefix: main_witness_columns,
                length: main_witness_columns.ilog2() as usize + 1,
            },
            next: Box::new(build_round_config(next_witness_length, false)),
        }
    } else {
        // Transition to unstructured rounds (no projection).
        // The first unstructured round has 8 input columns (from
        // the Intermediate decomposition). With prefix=0, witness_length
        // does not include the factor-of-2 doubling.
        let unstructured_cols = 8usize; // first unstructured inherits 8 cols from Intermediate output
        let unstructured_witness_length = next_single_col_height * unstructured_cols;
        let unstructured_single_col_height = next_single_col_height;
        let next_unstructured_height = unstructured_single_col_height / 2;
        let next_unstructured_cols = 4usize;
        let next_unstructured_wl = next_unstructured_height * next_unstructured_cols;

        let unstructured_common = RoundConfigCommon {
            witness_length: unstructured_witness_length,
            exact_binariness: false,
            l2: true,
            vdf: false,
            inner_evaluation_claims: 2,
            main_witness_columns: unstructured_cols,
            main_witness_prefix: Prefix {
                prefix: 0,
                length: 0,
            },
        };

        println!(
            "Building unstructured round config: witness_length={}, single_col_height={}, next_height={}",
            unstructured_witness_length, unstructured_single_col_height, next_unstructured_height
        );

        let next_unstructured_config = if next_unstructured_height >= PROJECTION_HEIGHT {
            build_unstructured_round_config(next_unstructured_wl)
        } else {
            RoundConfig::Last {
                common: RoundConfigCommon {
                    witness_length: next_unstructured_wl,
                    exact_binariness: false,
                    l2: true,
                    vdf: false,
                    inner_evaluation_claims: 2,
                    main_witness_columns: next_unstructured_cols,
                    main_witness_prefix: Prefix {
                        prefix: 0,
                        length: 0,
                    },
                },
            }
        };

        let next_config = RoundConfig::IntermediateUnstructured {
            common: unstructured_common,
            decomposition_base_log: 8,
            next: Box::new(next_unstructured_config),
        };

        RoundConfig::Intermediate {
            common,
            decomposition_base_log: 8,
            projection_ratio,
            projection_prefix: Prefix {
                prefix: main_witness_columns,
                length: main_witness_columns.ilog2() as usize + 1,
            },
            next: Box::new(next_config),
        }
    }
}

const LAST_ROUND_THRESHOLD: usize = 256;
/// Builds unstructured round configs (4 columns, prefix 0, no projection).
/// Continues until single_col_height / 2 < PROJECTION_HEIGHT, then produces Last.
fn build_unstructured_round_config(witness_length: usize) -> RoundConfig {
    let main_witness_columns = 4usize;
    let single_col_height = witness_length / main_witness_columns;
    let next_single_col_height = single_col_height / 2;
    let next_cols = 4usize;
    let next_wl = next_single_col_height * next_cols;

    println!(
        "Building unstructured round config: witness_length={}, single_col_height={}, next_height={}",
        witness_length, single_col_height, next_single_col_height
    );

    let common = RoundConfigCommon {
        witness_length,
        exact_binariness: false,
        l2: true,
        vdf: false,
        inner_evaluation_claims: 2,
        main_witness_columns,
        main_witness_prefix: Prefix {
            prefix: 0,
            length: 0,
        },
    };

    let next_config = if next_single_col_height >= LAST_ROUND_THRESHOLD {
        build_unstructured_round_config(next_wl)
    } else {
        RoundConfig::Last {
            common: RoundConfigCommon {
                witness_length: next_wl,
                exact_binariness: false,
                l2: true,
                vdf: false,
                inner_evaluation_claims: 2,
                main_witness_columns: next_cols,
                main_witness_prefix: Prefix {
                    prefix: 0,
                    length: 0,
                },
            },
        }
    };

    RoundConfig::IntermediateUnstructured {
        common,
        decomposition_base_log: 8,
        next: Box::new(next_config),
    }
}

static CONFIG: LazyLock<RoundConfig> =
    LazyLock::new(|| build_round_config(WITNESS_DIM * WITNESS_WIDTH * 2, true));

// ==== Prover Sumcheck context initialization ====

pub struct ProverSumcheckContext {
    pub witness_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub witness_conjugated_sumcheck: ElephantCell<LinearSumcheck<RingElement>>, // for verifying norms. Should be optional?
    pub main_witness_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub projection_selector_sumcheck: Option<ElephantCell<SelectorEq<RingElement>>>,
    pub type1sumcheck: Vec<Type1ProverSumcheckContext>, // for verifying inner evaluation points
    pub type3sumcheck: Option<Type3ProverSumcheckContext>, // for verifying the projection
    pub l2sumcheck: Option<L2ProverSumcheckContext>,
    pub linfsumcheck: Option<LinfSumcheckContext>,
    pub vdfsumcheck: Option<VDFProverSumcheckContext>, // for verifying the VDF, only used in the first round
    pub combiner: ElephantCell<Combiner<RingElement>>,
    pub field_combiner: ElephantCell<RingToFieldCombiner>,
    pub next: Option<Box<ProverSumcheckContext>>,
}

// VDF sumcheck: we prove that M · w = b where M is the VDF matrix and b = (-y_0, 0, ..., 0, y_t).
//
// The VDF matrix has the structure:
// |-----------------|      | ------- |
// | g               |      |  -y_0   |
// | a g             |      |    0    |
// |   a g           |      |    0    |
// |     a g         |      |    0    |
// |       a g       |  w = |    0    |
// |         a g     |      |    0    |
// |           a g   |      |    0    |
// |             a g |      |    0    |
// |               a |      |   y_t   |
// |-----------------|      | ------- |
//
// where w = (w_0 // w_1) i.e. the columns are stacked (matching our vertical memory alignment).
//
// We batch the rows with challenge powers (vdf_step_powers)^T = (1, c, c^2, ..., c^{2K}):
//
//                |-----------------|                         | ------- |
//                | g               |                         |  -y_0   |
//                | a g             |                         |    0    |
//                |   a g           |                         |    0    |
//                |     a g         |                         |    0    |
//  step_powers^T |       a g       |  w = step_powers^T      |    0    |
//                |         a g     |                         |    0    |
//                |           a g   |                         |    0    |
//                |             a g |                         |    0    |
//                |               a |                         |   y_t   |
//                |-----------------|                         | ------- |
//
// We factor this into two vectors:
//   vdf_batched_row^T = (1, c) ⊗ (g // a)
//     = [2^0 + c·a_0, 2^1 + c·a_1, ..., 2^63 + c·a_63]  (batched matrix column)
//   vdf_step_powers^T = (1, c, c^2, ..., c^{2K-1})  (challenge powers weighting each step)
//
// So the batched relation becomes:
//   (vdf_step_powers^T ⊗ vdf_batched_row^T) · w = -y_0 + c^{2K} · y_t
//
// All of the logic generalises to any t that is a power of two (i.e. WITNESS_DIM * 2).
// The full product (vdf_step_powers ⊗ vdf_batched_row) is one sumcheck for the prover.
// For the verifier:
//   - MLE[vdf_batched_row] evaluation is one small sumcheck (64 elements)
//   - MLE[vdf_step_powers] evaluation is efficient via the tensor structure:
//     vdf_step_powers = (1, c^{t/2}) ⊗ (1, c^{t/4}) ⊗ ... ⊗ (1, c)
//     MLE[vdf_step_powers](x) = prod_i ((1-x_i) + x_i · c^{t/2^{i+1}})

pub struct VDFProverSumcheckContext {
    pub vdf_step_powers_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub vdf_batched_row_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
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
    let single_col_height = (config.witness_length >> config.main_witness_prefix.length) / config.main_witness_columns;
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

fn init_prover_vdf_sumcheck(
    config: &RoundConfig,
    main_witness_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
) -> VDFProverSumcheckContext {
    let total_vars = config.witness_length.ilog2() as usize;
    let two_k = config.witness_length / 2 / VDF_MATRIX_WIDTH; // 2K = total VDF steps across both columns

    // vdf_step_powers: varies over log2(2K) middle variables (one per VDF step)
    // prefix = 1 (main_witness_selector bit), suffix = 6 (element-within-block bits)
    let vdf_step_powers_sumcheck = ElephantCell::new(
        LinearSumcheck::new_with_prefixed_sufixed_data(two_k, 1, VDF_MATRIX_WIDTH.ilog2() as usize),
    );

    // vdf_batched_row: varies over 6 LSB variables (element within 64-element block)
    // prefix = total_vars - 6 (all higher bits)
    let vdf_batched_row_sumcheck =
        ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
            VDF_MATRIX_WIDTH,
            total_vars - VDF_MATRIX_WIDTH.ilog2() as usize,
            0,
        ));

    let output = ElephantCell::new(ProductSumcheck::new(
        ElephantCell::new(ProductSumcheck::new(
            vdf_step_powers_sumcheck.clone(),
            vdf_batched_row_sumcheck.clone(),
        )),
        main_witness_sumcheck.clone(),
    ));

    VDFProverSumcheckContext {
        vdf_step_powers_sumcheck,
        vdf_batched_row_sumcheck,
        output,
    }
}

fn init_prover_type_3_sumcheck(
    config: &RoundConfig,
    main_witness_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
    projection_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>,
) -> Type3ProverSumcheckContext {
    match config {
        RoundConfig::Intermediate {
            projection_ratio,
            projection_prefix,
            ..
        } => {
            let c2_len = config.main_witness_columns;
            let c1_len = PROJECTION_HEIGHT;
            // (c_2 \otimes c_0 \otimes c_1^T J) · witness = (c_2 \otimes c_0 \otimes c_1)^T projected_witness
            let single_col_height = config.witness_length / 2 / config.main_witness_columns;

            let c0_len: usize = single_col_height / (PROJECTION_HEIGHT * projection_ratio);
            assert!(c0_len > 0, "c0_len must be greater than 0");

            let total_vars = config.witness_length.ilog2() as usize;

            assert_eq!(c0_len * c1_len * c2_len, config.witness_length / (2_usize.pow(projection_prefix.length as u32)), "c0_len * c1_len * c2_len must be equal to witness_length, got c0_len: {}, c1_len: {}, c2_len: {}, witness_length: {}", c0_len, c1_len, c2_len, config.witness_length);

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
            let fltr_len = (projection_ratio * PROJECTION_HEIGHT).ilog2() as usize;

            let flattened_projection_matrix_sumcheck =
                ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
                    projection_ratio * PROJECTION_HEIGHT,
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
                total_vars
                    - c1_len.ilog2() as usize
                    - c0_len.ilog2() as usize
                    - c2_len.ilog2() as usize,
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
        _ => panic!("type 3 sumcheck should only be initialized for rounds with projection"),
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

    let main_witness_sumcheck: ElephantCell<ProductSumcheck<_>> =
        ElephantCell::new(ProductSumcheck::new(
            witness_sumcheck.clone(),
            main_witness_selector_sumcheck.clone(),
        ));

    let projection_selector_sumcheck = match config {
        RoundConfig::Intermediate {
            projection_prefix, ..
        } => Some(sumcheck_from_prefix(
            &projection_prefix,
            config.witness_length.ilog2() as usize,
        )),
        _ => None,
    };

    let projection_sumcheck = match config {
        RoundConfig::Intermediate { .. } => Some(ElephantCell::new(ProductSumcheck::new(
            witness_sumcheck.clone(),
            projection_selector_sumcheck.as_ref().unwrap().clone(),
        ))),
        _ => None,
    };

    let type1sumcheck = (0..config.inner_evaluation_claims)
        .map(|_| init_prover_type_1_sumcheck(config, main_witness_sumcheck.clone()))
        .collect::<Vec<_>>();

    let type3sumcheck = match config {
        RoundConfig::Intermediate { .. } => Some(init_prover_type_3_sumcheck(
            config,
            main_witness_sumcheck.clone(),
            projection_sumcheck.clone().unwrap(),
        )),
        _ => None,
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

    let vdfsumcheck = if config.vdf {
        Some(init_prover_vdf_sumcheck(
            config,
            main_witness_sumcheck.clone(),
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
    if let Some(vdf) = &vdfsumcheck {
        all_outputs.push(vdf.output.clone());
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
        vdfsumcheck,
        next: match config {
            RoundConfig::Intermediate { next, .. } => {
                Some(Box::new(init_prover_sumcheck(crs, next)))
            }
            RoundConfig::IntermediateUnstructured { next, .. } => {
                Some(Box::new(init_prover_sumcheck(crs, next)))
            }
            RoundConfig::Last { .. } => None,
        },
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
        match config {
            RoundConfig::Intermediate {
                projection_ratio, ..
            } => {
                let c2_len = config.main_witness_columns;
                let c1_len = PROJECTION_HEIGHT;
                let single_col_height = config.witness_length / 2 / config.main_witness_columns;
                let c0_len: usize = single_col_height / (PROJECTION_HEIGHT * projection_ratio);
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

                hash_wrapper
                    .sample_ring_element_ntt_slots_same_vec_into(&mut result.c0.tensor_layers);
                hash_wrapper
                    .sample_ring_element_ntt_slots_same_vec_into(&mut result.c1.tensor_layers);
                hash_wrapper
                    .sample_ring_element_ntt_slots_same_vec_into(&mut result.c2.tensor_layers);

                result
            }
            _ => panic!("Batching challenges should only be sampled for rounds with projection"),
        }
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
            .as_ref()
            .map(|sumcheck| sumcheck.borrow_mut().partial_evaluate(r));
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

        // it's dumb, but it doesn't do anything except reducing the degree
        if let Some(linf) = &mut self.linfsumcheck {
            linf.all_one_constant_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }

        if let Some(vdf) = &mut self.vdfsumcheck {
            vdf.vdf_step_powers_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            vdf.vdf_batched_row_sumcheck
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
        projection_matrix: &Option<ProjectionMatrix>,
        projection_batching_challenges: &Option<BatchingChallenges>,
        vdf_challenge: Option<&RingElement>,
        vdf_crs_param: Option<&vdf_crs>,
    ) {
        self.witness_sumcheck.borrow_mut().load_from(&witness);
        self.witness_conjugated_sumcheck
            .borrow_mut()
            .load_from(&witness_conjugated);
        if let Some(projection_challenges) = projection_batching_challenges {
            let c0_expanded = PreprocessedRow::from_structured_row(&projection_challenges.c0);
            let c1_expanded = PreprocessedRow::from_structured_row(&projection_challenges.c1);
            let c2_expanded = PreprocessedRow::from_structured_row(&projection_challenges.c2);
            let flattened_projection = projection_flatter_1_times_matrix(
                projection_matrix.as_ref().unwrap(),
                &c1_expanded,
            );
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

        if let Some(vdf) = &mut self.vdfsumcheck {
            let c = vdf_challenge.expect("VDF sumcheck enabled but no vdf_challenge provided");
            let vdf_crs_ref = vdf_crs_param.expect("VDF sumcheck enabled but no vdf_crs provided");

            // Compute vdf_batched_row[j] = 2^j + c * a_j for j = 0..63
            // (2^j reduced mod q since RingElement::constant doesn't reduce)
            let mut batched_row: Vec<RingElement> = Vec::with_capacity(VDF_MATRIX_WIDTH);
            let mut ca_j = RingElement::zero(Representation::IncompleteNTT);
            for j in 0..VDF_MATRIX_WIDTH {
                let mut row_j =
                    RingElement::constant((1u64 << j) % MOD_Q, Representation::IncompleteNTT);
                ca_j *= (c, &vdf_crs_ref.A[(0, j)]);
                row_j += &ca_j;
                batched_row.push(row_j);
            }
            vdf.vdf_batched_row_sumcheck
                .borrow_mut()
                .load_from(&batched_row);

            // Compute vdf_step_powers[i] = c^i for i = 0..2K
            let two_k = witness.len() / 2 / VDF_MATRIX_WIDTH;
            let mut step_powers: Vec<RingElement> = Vec::with_capacity(two_k);
            let mut c_power = RingElement::constant(1, Representation::IncompleteNTT);
            for _ in 0..two_k {
                step_powers.push(c_power.clone());
                c_power *= c;
            }
            vdf.vdf_step_powers_sumcheck
                .borrow_mut()
                .load_from(&step_powers);
        }
    }
}

pub fn prover_round(
    crs: &CRS,
    witness: &VerticallyAlignedMatrix<RingElement>,
    config: &RoundConfig,
    sumcheck_context: &mut ProverSumcheckContext,
    evaluation_points_inner: &Vec<StructuredRow>,
    claims: &HorizontallyAlignedMatrix<RingElement>,
    // evaluation_points_outer: &Vec<StructuredRow>,
    hash_wrapper: &mut HashWrapper,
    vdf_params: Option<(&RingElement, &RingElement, &vdf_crs)>, // (y_0, y_t, crs) - only for first round
) -> SalsaaProof {
    let (projection_matrix, projection_commitment, projected_witness, batching_challenges) =
        match config {
            RoundConfig::Intermediate {
                projection_ratio, ..
            } => {
                let witness_16 = prepare_i16_witness(witness);

                let mut projection_matrix = ProjectionMatrix::new(witness.width, 256);

                projection_matrix.sample(hash_wrapper);

                let mut projected_witness = project(&witness_16, &projection_matrix);

                // The projection procent r columns into r columns with less rows, so we rearrange the projected witness taking advantage of the vertical alignment
                projected_witness.width = 1;
                projected_witness.used_cols = 1;
                projected_witness.height = witness.height;

                let projection_commitment = commit_basic(crs, &projected_witness, RANK);

                let batching_challenges = BatchingChallenges::sample(config, hash_wrapper);

                (
                    Some(projection_matrix),
                    Some(projection_commitment),
                    Some(projected_witness),
                    Some(batching_challenges),
                )
            }
            _ => (None, None, None, None),
        };


    let vdf_challenge = if config.vdf {
        let mut challenge = RingElement::zero(Representation::IncompleteNTT);
        hash_wrapper.sample_ring_element_ntt_slots_into(&mut challenge);
        Some(challenge)
    } else {
        None
    };

    if DEBUG {
        println!("witness.data.len {:?}", witness.data.len());
    }
    let mut extended_witness = new_vec_zero_preallocated(witness.data.len() << config.main_witness_prefix.length);

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

    match config {
        RoundConfig::Intermediate {
            projection_prefix, ..
        } => {
            paste_by_prefix(
                &mut extended_witness,
                &projected_witness.as_ref().unwrap().data,
                projection_prefix,
            );
        }
        _ => {}
    }

    let mut evaluation_points_outer = new_vec_zero_preallocated(config.main_witness_columns);
    hash_wrapper.sample_ring_element_vec_into(&mut evaluation_points_outer);

    sumcheck_context.load_data(
        &extended_witness,
        &witness_conjugated,
        evaluation_points_inner,
        &evaluation_points_outer,
        &projection_matrix,
        &batching_challenges,
        vdf_challenge.as_ref(),
        vdf_params.map(|(_, _, crs)| crs),
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

    if DEBUG {
        let ip_vdf_claim = compute_ip_vdf_claim(config, vdf_challenge.as_ref(), vdf_params);

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

        if config.vdf {
            let vdf_claim = sumcheck_context
                .vdfsumcheck
                .as_ref()
                .unwrap()
                .output
                .borrow()
                .claim();

            assert_eq!(
                vdf_claim,
                ip_vdf_claim.clone().unwrap(),
                "VDF claim from the sumcheck does not match the expected VDF claim"
            );
        }
    }

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

    if DEBUG {
        println!(
            "Polynomial time: {:?} ms, Evaluation time: {:?} ms",
            time_poly, time_eval
        );
    }

    let outer_points_len = config.main_witness_columns.ilog2() as usize + config.main_witness_prefix.length;
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

    let mut claim_over_projection = match config {
        RoundConfig::Intermediate { .. } => Some(new_vec_zero_preallocated(2)),
        _ => None,
    };

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

    match config {
        RoundConfig::Intermediate { .. } => {
            for (c, r) in projected_witness
                .as_ref()
                .unwrap()
                .data
                .iter()
                .zip(preprocessed_evaluation_points_inner.preprocessed_row.iter())
            {
                temp *= (c, r);
                claim_over_projection.as_mut().unwrap()[0] += &temp;
            }
        }
        _ => {}
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

    match config {
        RoundConfig::Intermediate { .. } => {
            for (c, r) in projected_witness
                .as_ref()
                .unwrap()
                .data
                .iter()
                .zip(preprocessed_evaluation_points_inner.preprocessed_row.iter())
            {
                temp *= (c, r);
                claim_over_projection.as_mut().unwrap()[1] += &temp;
            }
        }
        _ => {}
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

    let folded_witness = fold(&witness, &folding_challenges);

    let common = SalsaaProofCommon {
        // projection_commitment,
        ip_l2_claim,
        ip_linf_claim,
        sumcheck_transcript: polys,
        claims,
        // claim_over_projection,
    };

    match config {
        RoundConfig::Intermediate {
            decomposition_base_log,
            next,
            ..
        } => {
            if DEBUG {
                let commitment_to_folded_witness = commit_basic(crs, &folded_witness, RANK);
                let split_ref = VerticallyAlignedMatrix {
                    height: folded_witness.height / 2,
                    width: 2,
                    data: folded_witness.data.clone(),
                    used_cols: 2,
                };
                let commitment_to_split_witness = commit_basic(crs, &split_ref, RANK);
                let old_ck = crs.structured_ck_for_wit_dim(split_ref.height * 2);
                let composed = &(&(&*ONE - &old_ck[0].tensor_layers[0])
                    * &commitment_to_split_witness[(0, 0)])
                    + &(&old_ck[0].tensor_layers[0] * &commitment_to_split_witness[(0, 1)]);
                assert_eq!(composed, commitment_to_folded_witness[(0, 0)], "Composed commitment from the split witness does not match the commitment to the folded witness");
            }
            let split_witness = VerticallyAlignedMatrix {
                height: folded_witness.height / 2,
                width: 2,
                data: folded_witness.data,
                used_cols: 2,
            };

            let mut decomposed_split_witness = VerticallyAlignedMatrix {
                height: split_witness.height,
                width: 8,
                data: new_vec_zero_preallocated(split_witness.height * 8),
                used_cols: 8,
            };

            decompose_chunks_into(
                &mut decomposed_split_witness.data[..split_witness.height * 2],
                &split_witness.data[..split_witness.height],
                *decomposition_base_log,
                2,
            );

            decompose_chunks_into(
                &mut decomposed_split_witness.data
                    [split_witness.height * 2..split_witness.height * 4],
                &split_witness.data[split_witness.height..],
                *decomposition_base_log,
                2,
            );

            decompose_chunks_into(
                &mut decomposed_split_witness.data
                    [split_witness.height * 4..split_witness.height * 6],
                &projected_witness.as_ref().unwrap().data[..split_witness.height],
                *decomposition_base_log,
                2,
            );

            decompose_chunks_into(
                &mut decomposed_split_witness.data[split_witness.height * 6..],
                &projected_witness.as_ref().unwrap().data[split_witness.height..],
                *decomposition_base_log,
                2,
            );

            let decomposed_split_commitment = commit_basic(crs, &decomposed_split_witness, RANK);

            if DEBUG {
                let commitment_to_split_witness = commit_basic(crs, &split_witness, RANK);
                let old_ck = crs.structured_ck_for_wit_dim(split_witness.height * 2);

                let composed = compose_from_decomposed(
                    &vec![
                        decomposed_split_commitment[(0, 0)].clone(),
                        decomposed_split_commitment[(0, 1)].clone(),
                        decomposed_split_commitment[(0, 2)].clone(),
                        decomposed_split_commitment[(0, 3)].clone(),
                    ],
                    *decomposition_base_log,
                    2,
                );

                assert_eq!(composed[0], commitment_to_split_witness[(0, 0)], "Composed commitment from the decomposed split witness does not match the commitment to the split witness");

                assert_eq!(composed[1], commitment_to_split_witness[(0, 1)], "Composed commitment from the decomposed split projected witness does not match the commitment to the projected witness");

                let composed_projection = compose_from_decomposed(
                    &vec![
                        decomposed_split_commitment[(0, 4)].clone(),
                        decomposed_split_commitment[(0, 5)].clone(),
                        decomposed_split_commitment[(0, 6)].clone(),
                        decomposed_split_commitment[(0, 7)].clone(),
                    ],
                    *decomposition_base_log,
                    2,
                );

                let unsplit_projection = &(&(&*ONE - &old_ck[0].tensor_layers[0])
                    * &composed_projection[0])
                    + &(&old_ck[0].tensor_layers[0] * &composed_projection[1]);

                assert_eq!(unsplit_projection, projection_commitment.as_ref().unwrap()[(0, 0)], "Composed commitment from the decomposed split projected witness does not match the commitment to the projected witness");
            }

            let new_evaluation_points_inner = evaluation_points
                .iter()
                .skip(outer_points_len + 1)
                .cloned()
                .collect::<Vec<_>>();

            let new_evaluation_points_inner_expanded = PreprocessedRow::from_structured_row(
                &evaluation_point_to_structured_row(&new_evaluation_points_inner),
            );

            let new_evaluation_points_inner_conjugated = new_evaluation_points_inner
                .iter()
                .map(RingElement::conjugate)
                .collect::<Vec<_>>();

            let new_evaluation_points_inner_conjugated_expanded =
                PreprocessedRow::from_structured_row(&evaluation_point_to_structured_row(
                    &new_evaluation_points_inner_conjugated,
                ));

            let new_claims = commit_basic_internal(
                &vec![
                    new_evaluation_points_inner_expanded,
                    new_evaluation_points_inner_conjugated_expanded,
                ],
                &decomposed_split_witness,
                2,
            );

            let next_level_eval_points = vec![
                evaluation_point_to_structured_row(&new_evaluation_points_inner),
                evaluation_point_to_structured_row(&new_evaluation_points_inner_conjugated),
            ];
            let next_level_proof = prover_round(
                crs,
                &decomposed_split_witness,
                next,
                sumcheck_context.next.as_mut().unwrap(),
                &next_level_eval_points,
                &new_claims,
                hash_wrapper,
                None, // VDF only in first round
            );

            SalsaaProof::Intermediate {
                common,
                new_claims,
                decomposed_split_commitment,
                projection_commitment: projection_commitment.unwrap(),
                claim_over_projection: claim_over_projection.unwrap(),
                next: Box::new(next_level_proof),
            }
        }

        RoundConfig::IntermediateUnstructured {
            decomposition_base_log,
            next,
            ..
        } => {
            // Same as Intermediate but without projection columns:
            // fold → split → decompose → 4 columns (2 split × 2 decomp chunks)
            let split_witness = VerticallyAlignedMatrix {
                height: folded_witness.height / 2,
                width: 2,
                data: folded_witness.data,
                used_cols: 2,
            };

            let mut decomposed_split_witness = VerticallyAlignedMatrix {
                height: split_witness.height,
                width: 4,
                data: new_vec_zero_preallocated(split_witness.height * 4),
                used_cols: 4,
            };

            decompose_chunks_into(
                &mut decomposed_split_witness.data[..split_witness.height * 2],
                &split_witness.data[..split_witness.height],
                *decomposition_base_log,
                2,
            );

            decompose_chunks_into(
                &mut decomposed_split_witness.data[split_witness.height * 2..],
                &split_witness.data[split_witness.height..],
                *decomposition_base_log,
                2,
            );

            let decomposed_split_commitment = commit_basic(crs, &decomposed_split_witness, RANK);

            let new_evaluation_points_inner = evaluation_points
                .iter()
                .skip(outer_points_len + 1)
                .cloned()
                .collect::<Vec<_>>();

            let new_evaluation_points_inner_expanded = PreprocessedRow::from_structured_row(
                &evaluation_point_to_structured_row(&new_evaluation_points_inner),
            );

            let new_evaluation_points_inner_conjugated = new_evaluation_points_inner
                .iter()
                .map(RingElement::conjugate)
                .collect::<Vec<_>>();

            let new_evaluation_points_inner_conjugated_expanded =
                PreprocessedRow::from_structured_row(&evaluation_point_to_structured_row(
                    &new_evaluation_points_inner_conjugated,
                ));

            let new_claims = commit_basic_internal(
                &vec![
                    new_evaluation_points_inner_expanded,
                    new_evaluation_points_inner_conjugated_expanded,
                ],
                &decomposed_split_witness,
                2,
            );

            let next_level_eval_points = vec![
                evaluation_point_to_structured_row(&new_evaluation_points_inner),
                evaluation_point_to_structured_row(&new_evaluation_points_inner_conjugated),
            ];
            let next_level_proof = prover_round(
                crs,
                &decomposed_split_witness,
                next,
                sumcheck_context.next.as_mut().unwrap(),
                &next_level_eval_points,
                &new_claims,
                hash_wrapper,
                None,
            );

            SalsaaProof::IntermediateUnstructured {
                common,
                new_claims: new_claims.data,
                decomposed_split_commitment,
                next: Box::new(next_level_proof),
                projection_image_ct: [0u64; 256],
                projection_image_batched: [
                    RingElement::zero(Representation::IncompleteNTT),
                    RingElement::zero(Representation::IncompleteNTT),
                ],
            }
        }

        RoundConfig::Last { .. } => {
            // Last round: send the folded witness and projected witness directly, no decomposition
            SalsaaProof::Last {
                common,
                folded_witness: folded_witness.data,
                projection_image_ct: [0u64; 256],
                projection_image_batched: [
                    RingElement::zero(Representation::IncompleteNTT),
                    RingElement::zero(Representation::IncompleteNTT),
                ], // TODO change me to the actual evaluation of the projected witness at the evaluation points, but for now we just want to test the verifier logic with dummy values
            }
        }
    }
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
    pub projection_selector_evaluation: Option<ElephantCell<SelectorEqEvaluation>>,
    pub type1evaluations: Vec<Type1VerifierSumcheckContext>,
    pub type3evaluation: Option<Type3VerifierSumcheckContext>,
    pub l2evaluation: Option<L2VerifierSumcheckContext>,
    pub linfevaluation: Option<LinfVerifierSumcheckContext>,
    pub vdfevaluation: Option<VDFVerifierSumcheckContext>,
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

pub struct VDFVerifierSumcheckContext {
    pub vdf_step_powers_evaluation: ElephantCell<FakeEvaluationLinearSumcheck<RingElement>>,
    pub vdf_batched_row_evaluation: ElephantCell<BasicEvaluationLinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheckEvaluation>,
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
        ElephantCell<BasicEvaluationLinearSumcheck<QuadraticExtension>>,
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
    let single_col_height = (config.witness_length >> config.main_witness_prefix.length) / config.main_witness_columns;
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
    match config {
        RoundConfig::Intermediate {
            projection_ratio, ..
        } => {
            let c2_len = config.main_witness_columns;
            let c1_len = PROJECTION_HEIGHT;
            let single_col_height = config.witness_length / 2 / config.main_witness_columns;
            let c0_len: usize = single_col_height / (PROJECTION_HEIGHT * projection_ratio);
            let total_vars = config.witness_length.ilog2() as usize;

            // LEFT: prefix, c2, c0, flattened_projection_matrix (c1^T J)
            let fltr_len = (projection_ratio * PROJECTION_HEIGHT).ilog2() as usize;

            let flattened_projection_matrix_evaluation = ElephantCell::new(
                BasicEvaluationLinearSumcheck::<QuadraticExtension>::new_with_prefixed_sufixed_data(
                    projection_ratio * PROJECTION_HEIGHT,
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
                        ElephantCell::new(RingToFieldWrapperEvaluation::new(
                            flattened_projection_matrix_evaluation.clone(),
                        )),
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
        _ => panic!(
            "Type 3 sumcheck should only be initialized for intermediate rounds with projection"
        ),
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

fn init_verifier_vdf_sumcheck(
    config: &RoundConfig,
    main_witness_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
) -> VDFVerifierSumcheckContext {
    let total_vars = config.witness_length.ilog2() as usize;

    let vdf_step_powers_evaluation =
        ElephantCell::new(FakeEvaluationLinearSumcheck::<RingElement>::new());

    let vdf_batched_row_evaluation = ElephantCell::new(
        BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(
            VDF_MATRIX_WIDTH,
            total_vars - VDF_MATRIX_WIDTH.ilog2() as usize,
            0,
        ),
    );

    let output = ElephantCell::new(ProductSumcheckEvaluation::new(
        ElephantCell::new(ProductSumcheckEvaluation::new(
            vdf_step_powers_evaluation.clone(),
            vdf_batched_row_evaluation.clone(),
        )),
        main_witness_evaluation.clone(),
    ));

    VDFVerifierSumcheckContext {
        vdf_step_powers_evaluation,
        vdf_batched_row_evaluation,
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
    let projection_selector_evaluation = match config {
        RoundConfig::Intermediate {
            projection_prefix, ..
        } => Some(selector_evaluation_from_prefix(
            projection_prefix,
            total_vars,
        )),
        _ => None,
    };

    let main_witness_evaluation: ElephantCell<ProductSumcheckEvaluation> =
        ElephantCell::new(ProductSumcheckEvaluation::new(
            witness_evaluation.clone(),
            main_witness_selector_evaluation.clone(),
        ));

    let projection_eval = match config {
        RoundConfig::Intermediate {
            projection_prefix, ..
        } => Some(ElephantCell::new(ProductSumcheckEvaluation::new(
            witness_evaluation.clone(),
            selector_evaluation_from_prefix(projection_prefix, total_vars),
        ))),
        _ => None,
    };

    let type1evaluations = (0..config.inner_evaluation_claims)
        .map(|_| init_verifier_type_1_sumcheck(config, main_witness_evaluation.clone()))
        .collect::<Vec<_>>();

    let type3evaluation = match config {
        RoundConfig::Intermediate { projection_ratio, .. } => Some(init_verifier_type_3_sumcheck(
            config,
            main_witness_evaluation.clone(),
            projection_eval.expect("Projection evaluation should be initialized for intermediate rounds with projection"),
        )),
        _ => None,
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

    let vdfevaluation = if config.vdf {
        Some(init_verifier_vdf_sumcheck(
            config,
            main_witness_evaluation.clone(),
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
    if let Some(vdf) = &vdfevaluation {
        all_outputs.push(vdf.output.clone());
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
        vdfevaluation,
        combiner_evaluation,
        field_combiner_evaluation,
        next: match config {
            RoundConfig::Intermediate { next, .. } => Some(Box::new(init_verifier_sumcheck(next))),
            RoundConfig::IntermediateUnstructured { next, .. } => {
                Some(Box::new(init_verifier_sumcheck(next)))
            }
            RoundConfig::Last { .. } => None,
        },
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
    ip_vdf_claim: Option<&RingElement>,
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

    match config {
        RoundConfig::Intermediate { .. } => {
            // zero claim, nothing to add
            idx += 1;
        }
        _ => {}
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

    // VDF: product sumcheck claim = -y_0 + c^{2K} · y_t
    if config.vdf {
        let mut weighted = ip_vdf_claim
            .expect("Missing vdf claim in proof while vdf is enabled")
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
        batching_challenges: &Option<BatchingChallenges>,
        projection_matrix: &Option<ProjectionMatrix>,
        combination: &[RingElement],
        qe: [QuadraticExtension; HALF_DEGREE],
        vdf_challenge: Option<&RingElement>,
        vdf_crs_param: Option<&vdf_crs>,
    ) {
        let outer_points_len = config.main_witness_columns.ilog2() as usize + config.main_witness_prefix.length;
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

        match proof {
            SalsaaProof::Intermediate {
                claim_over_projection,
                ..
            } => {
                temp *= (
                    claim_over_projection.get(0).unwrap(),
                    &outer_points_expanded[config.main_witness_columns],
                );
                claim_over_witness += &temp;
            }
            _ => {}
        }

        let mut main_cols_points = evaluation_points_ring[config.main_witness_prefix.length..outer_points_len].to_vec();
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
            let c1_expanded =
                PreprocessedRow::from_structured_row(&batching_challenges.as_ref().unwrap().c1);

            let flattened_projection =
                projection_flatter_1_times_matrix(projection_matrix.as_ref().unwrap(), &c1_expanded);

            type3_eval
                .flattened_projection_matrix_evaluation
                .borrow_mut()
                .load_from(&flattened_projection);
            type3_eval
                .c0l_evaluation
                .borrow_mut()
                .load_from(batching_challenges.as_ref().unwrap().c0.clone());
            type3_eval
                .c2l_evaluation
                .borrow_mut()
                .load_from(batching_challenges.as_ref().unwrap().c2.clone());
            type3_eval
                .c0r_evaluation
                .borrow_mut()
                .load_from(batching_challenges.as_ref().unwrap().c0.clone());
            type3_eval
                .c1r_evaluation
                .borrow_mut()
                .load_from(batching_challenges.as_ref().unwrap().c1.clone());
            type3_eval
                .c2r_evaluation
                .borrow_mut()
                .load_from(batching_challenges.as_ref().unwrap().c2.clone());
        }

        if let Some(vdf_eval) = &mut self.vdfevaluation {
            let c = vdf_challenge.expect("VDF evaluation enabled but no vdf_challenge provided");
            let vdf_crs_ref =
                vdf_crs_param.expect("VDF evaluation enabled but no vdf_crs provided");

            // Compute vdf_batched_row[j] = 2^j + c * a_j for j = 0..63
            // (2^j reduced mod q since RingElement::constant doesn't reduce)
            let mut batched_row: Vec<RingElement> = Vec::with_capacity(VDF_MATRIX_WIDTH);
            let mut ca_j = RingElement::zero(Representation::IncompleteNTT);
            for j in 0..VDF_MATRIX_WIDTH {
                let mut row_j =
                    RingElement::constant((1u64 << j) % MOD_Q, Representation::IncompleteNTT);
                ca_j *= (c, &vdf_crs_ref.A[(0, j)]);
                row_j += &ca_j;
                batched_row.push(row_j);
            }
            vdf_eval
                .vdf_batched_row_evaluation
                .borrow_mut()
                .load_from(&batched_row);

            // Compute MLE[vdf_step_powers](x) = prod_i ((1-x_i) + x_i * c^{2^i})
            // step_powers variables: skip prefix=1 (MSB column selector), take log2(2K) vars
            let two_k = config.witness_length / 2 / VDF_MATRIX_WIDTH;
            let step_powers_num_vars = two_k.ilog2() as usize;
            let prefix = 1usize; // MSB selector bit (column selector)
            let step_powers_vars = &evaluation_points_ring[prefix..prefix + step_powers_num_vars];

            // Variables are MSB-first: step_powers_vars[0] = MSB of step index.
            // MLE[vdf_step_powers](x) = prod_i [(1-x_i) + x_i * c^{2^{n-1-i}}]
            // Iterate in reverse so c_power starts at c^{2^0} and pairs with LSB.
            let mut mle_step_powers = RingElement::constant(1, Representation::IncompleteNTT);
            let mut c_power = c.clone(); // c^{2^0} = c
            let mut temp_sq = RingElement::zero(Representation::IncompleteNTT);
            let mut term = RingElement::zero(Representation::IncompleteNTT);
            for x_i in step_powers_vars.iter().rev() {
                // factor = (1 - x_i) + x_i * c^{2^k}
                let mut factor = &*ONE - x_i;
                term *= (x_i, &c_power);
                factor += &term;
                mle_step_powers *= &factor;
                // c_power = c_power^2 for next iteration
                temp_sq *= (&c_power, &c_power);
                std::mem::swap(&mut c_power, &mut temp_sq);
            }
            vdf_eval
                .vdf_step_powers_evaluation
                .borrow_mut()
                .set_result(mle_step_powers);
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
    commitment: &BasicCommitment,
    proof: &SalsaaProof,
    evaluation_points_inner: &[StructuredRow],
    claims: &HorizontallyAlignedMatrix<RingElement>,
    hash_wrapper: &mut HashWrapper,
    vdf_crs_param: Option<&vdf_crs>,
    vdf_outputs: Option<(&RingElement, &RingElement)>, // (y_0, y_t) - only for first round
    round_index: usize,
) {
    let round_start = std::time::Instant::now();
    // TODO: check linf, l2 cts
    // Replay prover's Fiat-Shamir: sample projection matrix, batching challenges
    let mut projection_matrix = match config {
        RoundConfig::Intermediate { .. } => {
            let mut pm = ProjectionMatrix::new(config.main_witness_columns, PROJECTION_HEIGHT);
            pm.sample(hash_wrapper);
            Some(pm)
        }
        _ => None,
    };

    let batching_challenges = match config {
        RoundConfig::Intermediate { .. } => Some(BatchingChallenges::sample(config, hash_wrapper)),
        _ => None,
    };

    let vdf_challenge = if config.vdf {
        let mut challenge = RingElement::zero(Representation::IncompleteNTT);
        hash_wrapper.sample_ring_element_ntt_slots_into(&mut challenge);
        Some(challenge)
    } else {
        None
    };

    if config.l2 {
        let claim: &RingElement = proof
            .ip_l2_claim
            .as_ref()
            .expect("Missing l2 claim in proof while l2 constraint is enabled");
        let ct = claim.constant_term_from_incomplete_ntt();
        println!("asserted norm is sqrt({})", ct);
    }

    if config.exact_binariness {
        let claim: &RingElement = proof
            .ip_linf_claim
            .as_ref()
            .expect("Missing linf claim in proof while exact_binariness is enabled");
        let ct = claim.constant_term_from_incomplete_ntt();
        if ct != 0 {
            println!(
                "Binariness verification failed: constant term is not zero, got {}",
                ct
            );
        } else {
            println!("Binariness verification passed: constant term is zero");
        }
    }

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
        compute_ip_vdf_claim(
            config,
            vdf_challenge.as_ref(),
            vdf_outputs.map(|(y_0, y_t)| (y_0, y_t, vdf_crs_param.unwrap())),
        )
        .as_ref(),
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

    // verify evaluation claims (TODO: change to recompute them from the proof data)

    // Replay Fiat-Shamir: sample folding challenges (same as prover does post-sumcheck)
    let mut folding_challenges = new_vec_zero_preallocated(config.main_witness_columns);
    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut folding_challenges);

    let outer_points_len = config.main_witness_columns.ilog2() as usize + config.main_witness_prefix.length;
    let layer = &evaluation_points_ring[outer_points_len];
    let conj_layer = layer.conjugate();

    // Compute the folded claim: sum_i folding_challenges[i] * claims[(0, i)]
    let mut folded_claim = RingElement::zero(Representation::IncompleteNTT);
    for i in 0..config.main_witness_columns {
        let mut term = folding_challenges[i].clone();
        term *= &proof.claims[(0, i)];
        folded_claim += &term;
    }

    let mut folded_conj_claim = RingElement::zero(Representation::IncompleteNTT);
    for i in 0..config.main_witness_columns {
        let mut term = folding_challenges[i].clone();
        term *= &proof.claims[(1, i)];
        folded_conj_claim += &term;
    }

    match (config, proof) {
        (
            RoundConfig::Intermediate {
                decomposition_base_log,
                next,
                ..
            },
            SalsaaProof::Intermediate {
                new_claims,
                decomposed_split_commitment,
                claim_over_projection,
                projection_commitment,
                next: next_proof,
                ..
            },
        ) => {
            let recomposed_claims = HorizontallyAlignedMatrix {
                height: 2,
                width: 4,
                data: compose_from_decomposed(&new_claims.data, *decomposition_base_log, 2),
            };

            assert_eq!(
                folded_claim,
                &(&(&*ONE - layer) * &recomposed_claims[(0, 0)])
                    + &(layer * &recomposed_claims[(0, 1)]),
                "Recomposed claim for the witness does not match the original claim"
            );

            assert_eq!(
                folded_conj_claim,
                &(&(&*ONE - &conj_layer) * &recomposed_claims[(1, 0)])
                    + &(&conj_layer * &recomposed_claims[(1, 1)]),
                "Recomposed conjugate claim for the witness does not match the original claim"
            );

            // Check claims over the projection
            assert_eq!(
                claim_over_projection[0],
                &(&(&*ONE - layer) * &recomposed_claims[(0, 2)])
                    + &(layer * &recomposed_claims[(0, 3)]),
                "Recomposed claim for the projection does not match the original claim"
            );

            assert_eq!(
                claim_over_projection[1],
                &(&(&*ONE - &conj_layer) * &recomposed_claims[(1, 2)])
                    + &(&conj_layer * &recomposed_claims[(1, 3)]),
                "Recomposed conjugate claim for the projection does not match the original claim"
            );

            let recomposed_commitments = HorizontallyAlignedMatrix {
                height: RANK,
                width: 4,
                data: compose_from_decomposed(
                    &decomposed_split_commitment.data,
                    *decomposition_base_log,
                    2,
                ),
            };

            let mut temp = RingElement::zero(Representation::IncompleteNTT);
            for r in 0..RANK {
                let layer = crs.structured_ck_for_wit_dim(
                    config.witness_length / 2 / config.main_witness_columns,
                )[r]
                    .tensor_layers
                    .get(0)
                    .unwrap();

                let mut folded_commitment_r = RingElement::zero(Representation::IncompleteNTT);
                for i in 0..config.main_witness_columns {
                    temp *= (&folding_challenges[i], &commitment[(r, i)]);
                    folded_commitment_r += &temp;
                }

                assert_eq!(
                    folded_commitment_r,
                    &(&(&*ONE - layer) * &recomposed_commitments[(r, 0)])
                        + &(layer * &recomposed_commitments[(r, 1)]),
                    "Recomposed commitment for the witness does not match the folded commitment"
                );

                assert_eq!(
                    projection_commitment[(r, 0)],
                    &(&(&*ONE - layer) * &recomposed_commitments[(r, 2)])
                        + &(layer * &recomposed_commitments[(r, 3)]),
                    "Recomposed commitment for the projection does not match"
                );
            }

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
                vdf_challenge.as_ref(),
                vdf_crs_param,
            );

            let verifier_eval = verifier_context
                .field_combiner_evaluation
                .borrow_mut()
                .evaluate_at_ring_point(&evaluation_points_ring)
                .clone();

            assert_eq!(
                verifier_eval, batched_claim_over_field,
                "Verifier final check failed: tree evaluation does not match sumcheck claim"
            );

            // Recurse into the next round
            let new_evaluation_points_inner = evaluation_points_ring
                .iter()
                .skip(outer_points_len + 1)
                .cloned()
                .collect::<Vec<_>>();

            let new_evaluation_points_inner_conjugated = new_evaluation_points_inner
                .iter()
                .map(RingElement::conjugate)
                .collect::<Vec<_>>();

            let next_level_eval_points = vec![
                evaluation_point_to_structured_row(&new_evaluation_points_inner),
                evaluation_point_to_structured_row(&new_evaluation_points_inner_conjugated),
            ];

            println!(
                "Verifier round {} took {:?}",
                round_index,
                round_start.elapsed()
            );

            verifier_round(
                next,
                crs,
                verifier_context.next.as_mut().unwrap(),
                decomposed_split_commitment,
                next_proof,
                &next_level_eval_points,
                new_claims,
                hash_wrapper,
                None, // VDF only in first round
                None, // no VDF outputs in recursive rounds
                round_index + 1,
            );
        }

        (
            RoundConfig::IntermediateUnstructured {
                decomposition_base_log,
                next,
                ..
            },
            SalsaaProof::IntermediateUnstructured {
                new_claims,
                decomposed_split_commitment,
                next: next_proof,
                ..
            },
        ) => {
            // Recompose claims: width=2 (no projection columns)
            let recomposed_claims = HorizontallyAlignedMatrix {
                height: 2,
                width: 2,
                data: compose_from_decomposed(new_claims, *decomposition_base_log, 2),
            };

            assert_eq!(
                folded_claim,
                &(&(&*ONE - layer) * &recomposed_claims[(0, 0)])
                    + &(layer * &recomposed_claims[(0, 1)]),
                "IntermediateUnstructured: recomposed claim does not match the folded claim"
            );

            assert_eq!(
                folded_conj_claim,
                &(&(&*ONE - &conj_layer) * &recomposed_claims[(1, 0)])
                    + &(&conj_layer * &recomposed_claims[(1, 1)]),
                "IntermediateUnstructured: recomposed conjugate claim does not match"
            );

            // Recompose commitments: width=2 (no projection)
            let recomposed_commitments = HorizontallyAlignedMatrix {
                height: RANK,
                width: 2,
                data: compose_from_decomposed(
                    &decomposed_split_commitment.data,
                    *decomposition_base_log,
                    2,
                ),
            };

            let mut temp = RingElement::zero(Representation::IncompleteNTT);
            for r in 0..RANK {
                let layer = crs.structured_ck_for_wit_dim(
                    (config.witness_length >> config.main_witness_prefix.length)
                        / config.main_witness_columns,
                )[r]
                    .tensor_layers
                    .get(0)
                    .unwrap();

                let mut folded_commitment_r = RingElement::zero(Representation::IncompleteNTT);
                for i in 0..config.main_witness_columns {
                    temp *= (&folding_challenges[i], &commitment[(r, i)]);
                    folded_commitment_r += &temp;
                }

                assert_eq!(
                    folded_commitment_r,
                    &(&(&*ONE - layer) * &recomposed_commitments[(r, 0)])
                        + &(layer * &recomposed_commitments[(r, 1)]),
                    "IntermediateUnstructured: recomposed commitment does not match"
                );
            }

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
                vdf_challenge.as_ref(),
                vdf_crs_param,
            );

            let verifier_eval = verifier_context
                .field_combiner_evaluation
                .borrow_mut()
                .evaluate_at_ring_point(&evaluation_points_ring)
                .clone();

            assert_eq!(
                verifier_eval, batched_claim_over_field,
                "IntermediateUnstructured: tree evaluation does not match sumcheck claim"
            );

            // Recurse into the next round
            let new_evaluation_points_inner = evaluation_points_ring
                .iter()
                .skip(outer_points_len + 1)
                .cloned()
                .collect::<Vec<_>>();

            let new_evaluation_points_inner_conjugated = new_evaluation_points_inner
                .iter()
                .map(RingElement::conjugate)
                .collect::<Vec<_>>();

            let next_level_eval_points = vec![
                evaluation_point_to_structured_row(&new_evaluation_points_inner),
                evaluation_point_to_structured_row(&new_evaluation_points_inner_conjugated),
            ];

            let recomposed_new_claims = HorizontallyAlignedMatrix {
                height: 2,
                width: next.main_witness_columns,
                data: new_claims.clone(),
            };

            println!(
                "Verifier round {} (unstructured) took {:?}",
                round_index,
                round_start.elapsed()
            );

            verifier_round(
                next,
                crs,
                verifier_context.next.as_mut().unwrap(),
                decomposed_split_commitment,
                next_proof,
                &next_level_eval_points,
                &recomposed_new_claims,
                hash_wrapper,
                None,
                None,
                round_index + 1,
            );
        }

        (RoundConfig::Last { .. }, SalsaaProof::Last { folded_witness, .. }) => {
            // Last round: verify claims directly from the witness data

            // Reconstruct the folded witness as a VerticallyAlignedMatrix (1 column)
            let folded_witness_matrix = VerticallyAlignedMatrix {
                height: folded_witness.len(),
                width: 1,
                data: folded_witness.clone(),
                used_cols: 1,
            };

            // Reconstruct projected witness (1 column, same height as folded)
            // let projected_witness_matrix = VerticallyAlignedMatrix {
            //     height: projected_witness.len(),
            //     width: 1,
            //     data: projected_witness.clone(),
            //     used_cols: 1,
            // };

            // Use the current round's sumcheck evaluation points, including the "layer" variable
            // (no +1 skip since there's no split at the last round).
            // The prover computes claims using evaluation_points[outer_points_len..] from THIS round's sumcheck.
            let current_inner_points: Vec<_> = evaluation_points_ring
                .iter()
                .skip(outer_points_len)
                .cloned()
                .collect();

            let eval_points_inner_expanded = PreprocessedRow::from_structured_row(
                &evaluation_point_to_structured_row(&current_inner_points),
            );

            let current_inner_points_conjugated: Vec<_> = current_inner_points
                .iter()
                .map(RingElement::conjugate)
                .collect();

            let eval_points_inner_conj_expanded = PreprocessedRow::from_structured_row(
                &evaluation_point_to_structured_row(&current_inner_points_conjugated),
            );

            // Compute expected claim over folded witness: <eval_points, folded_witness>
            let mut temp = RingElement::zero(Representation::IncompleteNTT);
            let mut expected_folded_claim = RingElement::zero(Representation::IncompleteNTT);
            for (w, r) in folded_witness
                .iter()
                .zip(eval_points_inner_expanded.preprocessed_row.iter())
            {
                temp *= (w, r);
                expected_folded_claim += &temp;
            }

            assert_eq!(
                folded_claim, expected_folded_claim,
                "Last round: folded claim does not match evaluation of the folded witness"
            );

            // Compute expected conjugate claim over folded witness
            let mut expected_folded_conj_claim = RingElement::zero(Representation::IncompleteNTT);
            for (w, r) in folded_witness
                .iter()
                .zip(eval_points_inner_conj_expanded.preprocessed_row.iter())
            {
                temp *= (w, r);
                expected_folded_conj_claim += &temp;
            }

            assert_eq!(
                folded_conj_claim, expected_folded_conj_claim,
                "Last round: folded conjugate claim does not match evaluation of the folded witness"
            );

            // // Compute expected claim over projected witness
            // let mut expected_projection_claim = RingElement::zero(Representation::IncompleteNTT);
            // for (w, r) in projected_witness.iter().zip(eval_points_inner_expanded.preprocessed_row.iter()) {
            //     temp *= (w, r);
            //     expected_projection_claim += &temp;
            // }

            // assert_eq!(
            //     proof.claim_over_projection[0], expected_projection_claim,
            //     "Last round: projection claim does not match evaluation of the projected witness"
            // );

            // // Compute expected conjugate claim over projected witness
            // let mut expected_projection_conj_claim = RingElement::zero(Representation::IncompleteNTT);
            // for (w, r) in projected_witness.iter().zip(eval_points_inner_conj_expanded.preprocessed_row.iter()) {
            //     temp *= (w, r);
            //     expected_projection_conj_claim += &temp;
            // }

            // assert_eq!(
            //     proof.claim_over_projection[1], expected_projection_conj_claim,
            //     "Last round: conjugate projection claim does not match evaluation of the projected witness"
            // );

            let comm_time = std::time::Instant::now();

            // Verify commitment: commit(folded_witness) should match folded commitments
            let folded_witness_commitment = commit_basic(crs, &folded_witness_matrix, RANK);
            // let projected_witness_commitment = commit_basic(crs, &projected_witness_matrix, RANK);

            let elapsed = comm_time.elapsed();
            println!(
                "Verifier commitment recomputation took {} µs",
                elapsed.as_micros()
            );

            for r in 0..RANK {
                let mut folded_commitment_r = RingElement::zero(Representation::IncompleteNTT);
                for i in 0..config.main_witness_columns {
                    temp *= (&folding_challenges[i], &commitment[(r, i)]);
                    folded_commitment_r += &temp;
                }

                assert_eq!(
                    folded_commitment_r,
                    folded_witness_commitment[(r, 0)],
                    "Last round: folded witness commitment does not match"
                );

                // assert_eq!(
                //     proof.projection_commitment[(r, 0)], projected_witness_commitment[(r, 0)],
                //     "Last round: projected witness commitment does not match"
                // );
            }

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
                vdf_challenge.as_ref(),
                vdf_crs_param,
            );

            let verifier_eval = verifier_context
                .field_combiner_evaluation
                .borrow_mut()
                .evaluate_at_ring_point(&evaluation_points_ring)
                .clone();

            assert_eq!(
                verifier_eval, batched_claim_over_field,
                "Verifier final check failed: tree evaluation does not match sumcheck claim"
            );

            println!(
                "Verifier round {} (last) took {:?}",
                round_index,
                round_start.elapsed()
            );
            // No recursion at the last round
        }

        _ => panic!("Config and proof variant mismatch"),
    }
}

/// Computes ip_vdf_claim = -y_0 + c^{2K} * y_t from the VDF challenge and outputs.
fn compute_ip_vdf_claim(
    config: &RoundConfig,
    vdf_challenge: Option<&RingElement>,
    vdf_params: Option<(&RingElement, &RingElement, &vdf_crs)>,
) -> Option<RingElement> {
    if !config.vdf {
        return None;
    }
    let c = vdf_challenge.expect("VDF enabled but no challenge");
    let (y_0, y_t, _) = vdf_params.expect("VDF enabled but no params");
    let two_k = config.witness_length / 2 / VDF_MATRIX_WIDTH;
    let mut c_power = RingElement::constant(1, Representation::IncompleteNTT);
    for _ in 0..two_k {
        c_power *= c;
    }
    let mut claim = y_0.negate();
    c_power *= y_t;
    claim += &c_power;
    Some(claim)
}

const VDF_MATRIX_WIDTH: usize = 64;
const VDF_MATRIX_HEIGHT: usize = 1;
pub struct vdf_crs {
    A: HorizontallyAlignedMatrix<RingElement>,
}
pub fn vdf_init() -> vdf_crs {
    println!("Initializing VDF CRS...");
    let A = HorizontallyAlignedMatrix {
        height: VDF_MATRIX_HEIGHT,
        width: VDF_MATRIX_WIDTH,
        data: (0..VDF_MATRIX_HEIGHT * VDF_MATRIX_WIDTH)
            .map(|_| RingElement::random(Representation::IncompleteNTT))
            .collect(),
    };
    vdf_crs { A }
}

/// Decomposes a RingElement into 64 bit-plane RingElements, writing into `target`.
/// target\[b\].v\[j\] = (element.v\[j\] >> b) & 1 for each coefficient j and bit b.
/// The input is assumed to be in IncompleteNTT; we convert to EvenOddCoefficients
/// to access raw coefficients, decompose, then convert each result back.
pub fn decompose_binary_into(element: &RingElement, target: &mut [RingElement]) {
    assert!(
        target.len() >= 64,
        "target slice must have at least 64 elements"
    );

    let mut tmp = element.clone();
    tmp.from_incomplete_ntt_to_even_odd_coefficients();

    for bit_elem in target[..64].iter_mut() {
        *bit_elem = RingElement::zero(Representation::EvenOddCoefficients);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        use std::arch::x86_64::*;
        unsafe {
            let one = _mm512_set1_epi64(1);
            // Process 8 coefficients at a time
            for chunk_start in (0..DEGREE).step_by(8) {
                let coeffs = _mm512_loadu_epi64(tmp.v[chunk_start..].as_ptr() as *const i64);
                for b in 0..64u64 {
                    let shift_amt = _mm512_set1_epi64(b as i64);
                    let shifted = _mm512_srlv_epi64(coeffs, shift_amt);
                    let masked = _mm512_and_epi64(shifted, one);
                    _mm512_storeu_epi64(
                        target[b as usize].v[chunk_start..].as_mut_ptr() as *mut i64,
                        masked,
                    );
                }
            }
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        for j in 0..DEGREE {
            let val = tmp.v[j];
            for b in 0..64usize {
                target[b].v[j] = (val >> b) & 1;
            }
        }
    }

    for bit_elem in target[..64].iter_mut() {
        bit_elem.from_even_odd_coefficients_to_incomplete_ntt_representation();
    }
}

pub struct VDFOutput {
    y_int: RingElement, // TODO: this y_int is not needed but let's keep it for now
    y_t: RingElement,
    trace_witness: VerticallyAlignedMatrix<RingElement>,
}
pub fn execute_vdf(y_0: &RingElement, dim: usize, vdf_crs: &vdf_crs) -> VDFOutput {
    let vdf_crs_ref = vdf_crs;

    // g = (1, 2, 4, .., 2^63)
    // we want to obtain the following form
    //
    // |-----------------|      | ------- |
    // | g               |      |  -y_0   |
    // | a g             |      |    0    |
    // |   a g           |      |    0    |
    // |     a g         |      |    0    |
    // |       a g       |  w = |    0    |
    // |         a g     |      |    0    |
    // |           a g   |      |    0    |
    // |             a g |      |    0    |
    // |               a |      |   y_t   |
    // |-----------------|      | ------- |
    // t = 8
    // and in VDF we compute it like
    // w_0 = g^{-1} (-y_0)
    // (a g) * (w_0 || w_1) = 0
    // a w_0 = - g w_1
    // w_1 = g^{-1} (- a w_0) and we call y_1 = a w_0
    // w_2 = g^{-1} (- a w_1) and we call y_2 = a w_1
    // w_3 = g^{-1} (- a w_2) and we call y_3 = a w_2
    // w_4 = g^{-1} (- a w_3) and we call y_4 = a w_3
    // w_5 = g^{-1} (- a w_4) and we call y_5 = a w_4
    // w_6 = g^{-1} (- a w_5) and we call y_6 = a w_5
    // w_7 = g^{-1} (- a w_6) and we call y_7 = a w_6
    // y_8 = a w_7

    // |---------|    |---------|     |--------------|
    // | g       |    | w_0 w_4 |     | -y_0   -y_4  |
    // | a g     |    | w_1 w_5 |     |   0     0    |
    // |   a g   |  * | w_2 w_6 |   = |   0     0    |
    // |     a g |    | w_3 w_7 |     |   0     0    |
    // |       a |    |---------|     |  y_4   y_8   |
    // |---------|                    |--------------|

    // we call y_int = y_4
    // then we can split the wintess witness into two cols
    //
    // This intuition naturally generalises to any t = 2^k,
    // i.e. t = WITNESS_DIM/64*2 (2 columns, 64 for decomposition)
    //

    let mut trace_witness = VerticallyAlignedMatrix {
        height: dim,
        width: 2,
        data: new_vec_zero_preallocated(dim * 2),
        used_cols: 2,
    };

    let steps_per_col = dim / VDF_MATRIX_WIDTH;
    let total_steps = steps_per_col * 2;

    let mut neg_y = y_0.negate();
    let mut y_int = RingElement::zero(Representation::IncompleteNTT);
    let mut temp = RingElement::zero(Representation::IncompleteNTT);

    println!("Executing VDF with {} steps", total_steps);
    for step in 0..total_steps {
        let col = step / steps_per_col;
        let row_in_col = step % steps_per_col;
        let base_row = row_in_col * VDF_MATRIX_WIDTH;

        // w_step = g^{-1}(-y_step) = decompose_binary(-y_step)
        // Write directly into the trace_witness column-major slice
        let data_offset = col * dim + base_row;
        decompose_binary_into(
            &neg_y,
            &mut trace_witness.data[data_offset..data_offset + VDF_MATRIX_WIDTH],
        );

        // y_{step+1} = <a, w_step> = sum_j a_j * bits[j]
        let mut y_next = RingElement::zero(Representation::IncompleteNTT);
        for j in 0..VDF_MATRIX_WIDTH {
            temp *= (&vdf_crs_ref.A[(0, j)], &trace_witness.data[data_offset + j]);
            y_next += &temp;
        }

        if step == steps_per_col - 1 {
            y_int = y_next.clone();
        }

        neg_y = y_next.negate();
    }

    let y_t = neg_y.negate();

    VDFOutput {
        y_int,
        y_t,
        trace_witness,
    }
}

pub fn execute() {
    println!("Generating CRS...");

    let crs = CRS::gen_crs(WITNESS_DIM, 8);
    let vdf_crs = vdf_init();

    println!("CRS generated. Starting execution...");
    let vdf_start = std::time::Instant::now();
    let y_0: RingElement = RingElement::random(Representation::IncompleteNTT); // TODO: from hash
    let vdf_output = execute_vdf(&y_0, WITNESS_DIM, &vdf_crs);
    let vdf_duration = vdf_start.elapsed().as_millis();
    println!("VDF executed in {:?} ms", vdf_duration);

    let mut sumcheck_context = init_prover_sumcheck(&crs, &CONFIG);

    println!("===== COMMITTING WITNESS =====");
    let start = std::time::Instant::now();

    let commitment = commit_basic(&crs, &vdf_output.trace_witness, RANK);

    let commit_duration = start.elapsed().as_nanos();
    println!("TOTAL Commit time: {:?} ns", commit_duration);

    let no_claims = HorizontallyAlignedMatrix {
        height: 0,
        width: 2,
        data: vec![],
    };

    println!("===== STARTING PROVER =====");
    let start = std::time::Instant::now();
    let proof = prover_round(
        &crs,
        &vdf_output.trace_witness,
        &CONFIG,
        &mut sumcheck_context,
        &vec![], // no evaluation points for first round
        &no_claims,
        &mut HashWrapper::new(),
        Some((&y_0, &vdf_output.y_t, &vdf_crs)),
    );
    let prove_duration = start.elapsed().as_millis();
    println!("TOTAL Prove time: {:?} ms", prove_duration);

    println!("===== PROOF SIZE =====");
    let proof_size_bits = proof.size_in_bits();
    println!("Total proof size: {:.2} KB", to_kb(proof_size_bits));

    println!("===== STARTING VERIFIER =====");
    let start = std::time::Instant::now();
    let mut verifier_context = init_verifier_sumcheck(&CONFIG);
    verifier_round(
        &CONFIG,
        &crs,
        &mut verifier_context,
        &commitment,
        &proof,
        &vec![],    // no evaluation points for first round
        &no_claims, // no claims for first round
        &mut HashWrapper::new(),
        Some(&vdf_crs),
        Some((&y_0, &vdf_output.y_t)),
        0,
    );
    let verify_duration = start.elapsed().as_nanos();
    println!("TOTAL Verify time: {:?} ns", verify_duration);
    println!("===== VERIFICATION PASSED =====");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::config::MOD_Q;

    #[test]
    fn test_decompose_binary_roundtrip() {
        let elem = RingElement::random(Representation::IncompleteNTT);
        let mut bits: Vec<RingElement> = (0..64)
            .map(|_| RingElement::zero(Representation::IncompleteNTT))
            .collect();
        decompose_binary_into(&elem, &mut bits);

        assert_eq!(bits.len(), 64);

        // Recompose: sum_b bits[b] * 2^b  (in EvenOdd space, then convert back)
        let mut recomposed = RingElement::zero(Representation::IncompleteNTT);
        recomposed.from_incomplete_ntt_to_even_odd_coefficients();
        for (b, bit_elem) in bits.iter().enumerate() {
            let mut bit_copy = bit_elem.clone();
            bit_copy.from_incomplete_ntt_to_even_odd_coefficients();
            let shift = 1u64 << b;
            for j in 0..DEGREE {
                recomposed.v[j] = (recomposed.v[j] + bit_copy.v[j] * shift) % MOD_Q;
            }
        }
        recomposed.from_even_odd_coefficients_to_incomplete_ntt_representation();

        assert_eq!(recomposed, elem, "Binary decomposition roundtrip failed");
    }

    #[test]
    fn test_decompose_binary_bits_are_binary() {
        let elem = RingElement::random(Representation::IncompleteNTT);
        let mut bits: Vec<RingElement> = (0..64)
            .map(|_| RingElement::zero(Representation::IncompleteNTT))
            .collect();
        decompose_binary_into(&elem, &mut bits);

        for (b, bit_elem) in bits.iter().enumerate() {
            let mut bit_copy = bit_elem.clone();
            bit_copy.from_incomplete_ntt_to_even_odd_coefficients();
            for j in 0..DEGREE {
                assert!(
                    bit_copy.v[j] == 0 || bit_copy.v[j] == 1,
                    "Bit plane {} coeff {} is {}, expected 0 or 1",
                    b,
                    j,
                    bit_copy.v[j]
                );
            }
        }
    }

    #[test]
    fn test_decompose_binary_high_bits_zero() {
        // MOD_Q < 2^51, so bits 51..63 should be all zero
        let elem = RingElement::random(Representation::IncompleteNTT);
        let mut bits: Vec<RingElement> = (0..64)
            .map(|_| RingElement::zero(Representation::IncompleteNTT))
            .collect();
        decompose_binary_into(&elem, &mut bits);

        for b in 51..64 {
            let mut bit_copy = bits[b].clone();
            bit_copy.from_incomplete_ntt_to_even_odd_coefficients();
            for j in 0..DEGREE {
                assert_eq!(
                    bit_copy.v[j], 0,
                    "Bit plane {} coeff {} should be 0 (above modulus bit-width)",
                    b, j
                );
            }
        }
    }

    /// Verify the matrix equation from execute_vdf:
    ///
    /// | g       |    | w_0 w_K |     | -y_0   -y_int |
    /// | a g     |    | w_1 ... |     |   0      0    |
    /// |   a g   |  * | ...     |  =  |   0      0    |
    /// |     a g |    | ...     |     |   0      0    |
    /// |       a |    |---------|     |  y_int  y_t   |
    ///
    /// where K = steps_per_col and g recomposes binary (sum_j 2^j * bit_j).
    #[test]
    fn test_vdf_matrix_equation() {
        let test_dim: usize = 1 << 12; // 4096, giving 2^6 = 64 steps per column
        let y_0 = RingElement::random(Representation::IncompleteNTT);
        let vdf_crs = vdf_init();
        let vdf_output = execute_vdf(&y_0, test_dim, &vdf_crs);

        let steps_per_col = test_dim / VDF_MATRIX_WIDTH;
        let w = &vdf_output.trace_witness;

        // Helper: compute g * w_block = recompose binary = sum_j 2^j * w[(base+j, col)]
        // We work in EvenOdd to do the weighted sum, then convert back.
        let recompose = |base_row: usize, col: usize| -> RingElement {
            let mut result = RingElement::zero(Representation::IncompleteNTT);
            result.from_incomplete_ntt_to_even_odd_coefficients();
            for j in 0..VDF_MATRIX_WIDTH {
                let mut bit_copy = w[(base_row + j, col)].clone();
                bit_copy.from_incomplete_ntt_to_even_odd_coefficients();
                let shift = 1u64 << j;
                for k in 0..DEGREE {
                    result.v[k] = (result.v[k] + bit_copy.v[k] * shift) % MOD_Q;
                }
            }
            result.from_even_odd_coefficients_to_incomplete_ntt_representation();
            result
        };

        // Helper: compute a * w_block = <A[0], w_block> = sum_j A[(0,j)] * w[(base+j, col)]
        let inner_product_a = |base_row: usize, col: usize| -> RingElement {
            let mut result = RingElement::zero(Representation::IncompleteNTT);
            let mut temp = RingElement::zero(Representation::IncompleteNTT);
            for j in 0..VDF_MATRIX_WIDTH {
                temp *= (&vdf_crs.A[(0, j)], &w[(base_row + j, col)]);
                result += &temp;
            }
            result
        };

        let zero = RingElement::zero(Representation::IncompleteNTT);

        // Check both columns
        let y_starts = [&y_0, &vdf_output.y_int];
        let y_ends = [&vdf_output.y_int, &vdf_output.y_t];

        for col in 0..2 {
            // First row: g * w_0 = -y_start
            let gw0 = recompose(0, col);
            assert_eq!(
                gw0,
                y_starts[col].negate(),
                "Column {}: g * w_0 != -y_start",
                col
            );

            // Middle rows: a * w_i + g * w_{i+1} = 0
            for i in 0..steps_per_col - 1 {
                let aw_i = inner_product_a(i * VDF_MATRIX_WIDTH, col);
                let gw_next = recompose((i + 1) * VDF_MATRIX_WIDTH, col);
                let sum = &aw_i + &gw_next;
                assert_eq!(
                    sum,
                    zero,
                    "Column {}, row {}: a*w_{} + g*w_{} != 0",
                    col,
                    i + 1,
                    i,
                    i + 1
                );
            }

            // Last row: a * w_{last} = y_end
            let aw_last = inner_product_a((steps_per_col - 1) * VDF_MATRIX_WIDTH, col);
            assert_eq!(aw_last, *y_ends[col], "Column {}: a * w_last != y_end", col);
        }
    }
}
