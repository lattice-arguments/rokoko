use std::{any::Any, sync::LazyLock};

use crate::{
    common::{
        config::*,
        matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        ring_arithmetic::{QuadraticExtension, RingElement},
    },
    protocol::{
        commitment::{BasicCommitment, Prefix, RecursionConfig},
        sumcheck_utils::polynomial::Polynomial,
    },
};

// 2^26 = 2^7 (DEGREE) * 2^19
// 2^19 = 2^5 * 2^14

#[derive(Clone)]
pub enum ProjectionType {
    Type0, // full ring elements + no batching
    Type1, // coefficient-wise + batching + commit to constant terms + commit to batched projection
}

#[derive(Clone)]
pub struct Type1ProjectionConfig {
    pub nof_batches: usize,
    pub recursion_constant_term: RecursionConfig, // here we check the norm
    pub recursion_batched_projection: RecursionConfig, // this one is for the actual projection and we need to show consistency
}

pub type Type0ProjectionConfig = RecursionConfig;

#[derive(Clone)]
pub enum Projection {
    Type0(Type0ProjectionConfig),
    Type1(Type1ProjectionConfig),
    Skip,
}

pub trait ConfigBase: Any {
    fn witness_height(&self) -> usize;
    fn witness_width(&self) -> usize;
    fn projection_ratio(&self) -> usize;
    fn projection_height(&self) -> usize;
    fn basic_commitment_rank(&self) -> usize;
}

pub trait SizeableProof {
    fn size_in_bits(&self) -> usize;
}

pub fn to_kb(size_in_bits: usize) -> f64 {
    size_in_bits as f64 / 8.0 / 1024.0
}

pub struct SalsaaProofCommon {
    pub sumcheck_transcript: Vec<Polynomial<QuadraticExtension>>,
    pub ip_l2_claim: Option<RingElement>,
    pub ip_linf_claim: Option<RingElement>,
    pub(crate) claims: HorizontallyAlignedMatrix<RingElement>,
}

#[derive(Clone)]
pub struct RoundConfigCommon {
    pub main_witness_prefix: Prefix,
    pub main_witness_columns: usize,
    pub extended_witness_length: usize,
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
        projection_ratio: usize,
        common: RoundConfigCommon,
        decomposition_base_log: u64,
        next: Box<RoundConfig>,
    },
    Last {
        common: RoundConfigCommon,
        projection_ratio: usize,
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

fn ring_vec_size(v: &[RingElement]) -> usize {
    v.iter().map(|e| e.size_in_bits()).sum()
}

/// Recursively builds the round config chain.
/// - First round: uses NUM_COLUMNS_INITIAL columns, projection_ratio=2, VDF+exact_binariness enabled.
/// - Subsequent rounds: 8 columns, projection_ratio=8, L2 enabled.
/// - Recursion stops when the *next* round's single_col_height would be < PROJECTION_HEIGHT * projection_ratio
///   (i.e., the next round couldn't support projection).
fn build_round_config(extended_witness_length: usize, is_first_round: bool) -> RoundConfig {
    let main_witness_columns = if is_first_round {
        NUM_COLUMNS_INITIAL
    } else {
        8
    };

    // only for structured case
    let projection_ratio = if is_first_round { 2 } else { 8 };

    let single_col_height = extended_witness_length / 2 / main_witness_columns;
    // After fold+split+decompose, next round's column height = single_col_height / 2
    let next_single_col_height = single_col_height / 2;
    let next_main_witness_columns = 8usize;
    let next_projection_ratio = 8usize;
    let can_recurse = next_single_col_height >= PROJECTION_HEIGHT * next_projection_ratio;
    println!("Building round config: extended_witness_length={}, single_col_height={}, next_single_col_height={}, can_recurse={}", extended_witness_length, single_col_height, next_single_col_height, can_recurse);

    let inner_evaluation_claims = if is_first_round { 0 } else { 2 };

    let common = RoundConfigCommon {
        extended_witness_length,
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
        let next_extended_witness_length = next_single_col_height * next_main_witness_columns * 2;
        RoundConfig::Intermediate {
            common,
            decomposition_base_log: 8,
            projection_ratio,
            projection_prefix: Prefix {
                prefix: main_witness_columns,
                length: main_witness_columns.ilog2() as usize + 1,
            },
            next: Box::new(build_round_config(next_extended_witness_length, false)),
        }
    } else {
        // Transition to unstructured rounds (no projection).
        // The first unstructured round has 8 input columns (from
        // the Intermediate decomposition). With prefix=0, extended_witness_length
        // does not include the factor-of-2 doubling.
        let unstructured_cols = 8usize; // first unstructured inherits 8 cols from Intermediate output
        let unstructured_extended_witness_length = next_single_col_height * unstructured_cols;
        let unstructured_single_col_height = next_single_col_height;
        let next_unstructured_height = unstructured_single_col_height / 2;
        let next_unstructured_cols = 4usize;
        let next_unstructured_wl = next_unstructured_height * next_unstructured_cols;

        let unstructured_common = RoundConfigCommon {
            extended_witness_length: unstructured_extended_witness_length,
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
            "Building unstructured round config: extended_witness_length={}, single_col_height={}, next_height={}",
            unstructured_extended_witness_length, unstructured_single_col_height, next_unstructured_height
        );

        let next_unstructured_config = if next_unstructured_height >= PROJECTION_HEIGHT {
            build_unstructured_round_config(next_unstructured_wl)
        } else {
            RoundConfig::Last {
                common: RoundConfigCommon {
                    extended_witness_length: next_unstructured_wl,
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
                projection_ratio: std::cmp::min(
                    DEGREE * next_unstructured_height / PROJECTION_HEIGHT,
                    MAX_UNSTRUCT_PROJ_RATIO,
                ),
            }
        };

        let next_config = RoundConfig::IntermediateUnstructured {
            common: unstructured_common,
            decomposition_base_log: 8,
            projection_ratio: std::cmp::min(
                DEGREE * unstructured_single_col_height / PROJECTION_HEIGHT,
                MAX_UNSTRUCT_PROJ_RATIO,
            ),
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

impl ConfigBase for RoundConfig {
    fn witness_height(&self) -> usize {
        self.extended_witness_length
            / (self.main_witness_columns * 2usize.pow(self.main_witness_prefix.length as u32))
    }

    fn witness_width(&self) -> usize {
        self.main_witness_columns * 2usize.pow(self.main_witness_prefix.length as u32)
    }

    fn projection_ratio(&self) -> usize {
        match self {
            RoundConfig::Intermediate {
                projection_ratio, ..
            } => *projection_ratio,
            RoundConfig::IntermediateUnstructured {
                projection_ratio, ..
            } => *projection_ratio,
            RoundConfig::Last {
                projection_ratio, ..
            } => *projection_ratio,
        }
    }

    fn projection_height(&self) -> usize {
        PROJECTION_HEIGHT
    }

    fn basic_commitment_rank(&self) -> usize {
        panic!("basic_commitment_rank is not defined for RoundConfig");
    }
}

/// Builds unstructured round configs (4 columns, prefix 0, unstructured projection).
/// Continues until single_col_height / 2 < PROJECTION_HEIGHT, then produces Last.
fn build_unstructured_round_config(extended_witness_length: usize) -> RoundConfig {
    let main_witness_columns = 4usize;
    let single_col_height = extended_witness_length / main_witness_columns;
    let next_single_col_height = single_col_height / 2;
    let next_cols = 4usize;
    let next_wl = next_single_col_height * next_cols;

    println!(
        "Building unstructured round config: extended_witness_length={}, single_col_height={}, next_height={}",
        extended_witness_length, single_col_height, next_single_col_height
    );

    let common = RoundConfigCommon {
        extended_witness_length,
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
                extended_witness_length: next_wl,
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
            projection_ratio: std::cmp::min(
                DEGREE * next_single_col_height / PROJECTION_HEIGHT,
                MAX_UNSTRUCT_PROJ_RATIO,
            ),
        }
    };

    RoundConfig::IntermediateUnstructured {
        common,
        decomposition_base_log: 8,
        projection_ratio: std::cmp::min(
            DEGREE * single_col_height / PROJECTION_HEIGHT,
            MAX_UNSTRUCT_PROJ_RATIO,
        ), // for now, we assume that each column is projected to PROJECTION_HEIGHT Zq elements.
        next: Box::new(next_config),
    }
}

pub static CONFIG: LazyLock<RoundConfig> =
    LazyLock::new(|| build_round_config(WITNESS_DIM * WITNESS_WIDTH * 2, true));

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
        projection_image_ct: VerticallyAlignedMatrix<RingElement>,
        projection_image_batched: HorizontallyAlignedMatrix<RingElement>,
    },
    Last {
        common: SalsaaProofCommon,
        folded_witness: Vec<RingElement>,
        projection_image_ct: VerticallyAlignedMatrix<RingElement>,
        projection_image_batched: HorizontallyAlignedMatrix<RingElement>,
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
                projection_image_ct,
                projection_image_batched,
                ..
            } => {
                let folded_size = ring_vec_size(folded_witness);
                println!("  Folded witness: {:.2} KB", to_kb(folded_size));

                let projection_image_size = ring_vec_size(&projection_image_ct.data);
                println!("  Projection image: {:.2} KB", to_kb(projection_image_size));

                let batched_projection_size = ring_vec_size(&projection_image_batched.data);
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

                let batched_projection_size = ring_vec_size(&projection_image_batched.data);
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

#[inline]
pub fn paste_by_prefix(dest: &mut Vec<RingElement>, src: &Vec<RingElement>, prefix: &Prefix) {
    debug_assert_eq!(
        src.len().next_power_of_two(),
        1 << dest.len().ilog2() as usize - prefix.length,
        "Pasting failed. Source length does not match prefix length."
    );
    // e.g. if dest.len() = 2048, prefix.length = 4, prefix.prefix = 9 (0b1001)
    // then start = 9 << (11 - 4) = 9 << 7 = 1152 = 10010000000 index to start pasting
    let start = prefix.prefix << (dest.len().ilog2() as usize - prefix.length);
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dest.as_mut_ptr().add(start), src.len());
    }
}

#[inline]
pub fn slice_by_prefix(src: &Vec<RingElement>, prefix: &Prefix) -> Vec<RingElement> {
    let start = prefix.prefix << (src.len().ilog2() as usize - prefix.length);
    let length = 1 << (src.len().ilog2() as usize - prefix.length);
    src[start..start + length].to_vec()
}
