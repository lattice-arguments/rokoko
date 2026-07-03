use std::{any::Any, sync::LazyLock};

use crate::{
    common::{
        matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        ring_arithmetic::{QuadraticExtension, RingElement},
    },
    protocol::{
        commitment::{Prefix, RecursionConfig, RecursiveCommitment, RecursiveCommitmentWithAux},
        config_generator::{AuxConfig, AuxProjection, AuxRecursionConfig, AuxSumcheckConfig},
        params::P,
        sumcheck_utils::polynomial::Polynomial,
    },
};

#[derive(Clone)]
pub struct FineProjectionConfig {
    pub nof_batches: usize,
    pub recursion_constant_term: RecursionConfig, // carries the norm claim
    pub recursion_batched_projection: RecursionConfig, // carries the consistency checks
}

pub type CoarseProjectionConfig = RecursionConfig;

/// Paper: Π^proj-c (ring elements) / Π^proj-f (coefficients); `Skip` in the
/// first round, where extraction slack is tolerable.
#[derive(Clone)]
pub enum Projection {
    Coarse(CoarseProjectionConfig),
    Fine(FineProjectionConfig),
    Skip,
}

pub static SOMEWHAT_REAL_CONFIG: LazyLock<Config> = LazyLock::new(|| {
    AuxSumcheckConfig {
        witness_height: 2usize.pow(15),   // 2^15
        witness_width: 2usize.pow(6),     // 2^6
        projection_ratio: 2usize.pow(6),  // 2^6
        projection_height: 2usize.pow(8), // 2^8
        basic_commitment_rank: 4,
        nof_openings: 1,
        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 15, // 2^5 (witness_width) * 2^2 (rank) * 2^2 (decomp) = 2^9
            decomposition_chunks: 4,
            rank: 1,
            next: Some(Box::new(AuxRecursionConfig {
                decomposition_base_log: 7,
                decomposition_chunks: 8, // 1 (rank) * 8 (decomp) = 2^3
                rank: 1,
                next: None,
            })),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 15, // 2^5 (witness_width) * 2^0 (nof openings) * 2^2 (decomp) = 2^7
            decomposition_chunks: 4, // for now, there's no reason why decomposition_chunks here shall be different from commitment_recursion.decomposition_chunks. I will use that assumption in sumcheck.
            rank: 1,
            next: None,
        },
        projection_recursion: AuxProjection::Coarse(AuxRecursionConfig {
            // 2^14 (witness_height) * 2^5 (witness_width) / 2^5 (projection_ratio) * 2^0 (decomp) = 2^14
            decomposition_base_log: 20, // no decomposition
            decomposition_chunks: 1,
            rank: 1,
            next: None,
        }),

        witness_decomposition_chunks: 2,
        witness_decomposition_base_log: 10, // no decomposition

        next: Some(Box::new(AuxConfig::Sumcheck(AuxSumcheckConfig {
            witness_height: 2usize.pow(10),
            witness_width: 2usize.pow(7),
            projection_ratio: 2usize.pow(7),
            projection_height: 2usize.pow(8),
            basic_commitment_rank: 2,
            nof_openings: 2,
            commitment_recursion: AuxRecursionConfig {
                decomposition_base_log: 15, // 2^5 (witness_width) * 2^2 (rank) * 2^2 (decomp) = 2^9
                decomposition_chunks: 4,
                rank: 1,
                next: Some(Box::new(AuxRecursionConfig {
                    decomposition_base_log: 7,
                    decomposition_chunks: 8, // 1 (rank) * 8 (decomp) = 2^3
                    rank: 1,
                    next: None,
                })),
            },
            opening_recursion: AuxRecursionConfig {
                decomposition_base_log: 15, // 2^5 (witness_width) * 2^0 (nof openings) * 2^2 (decomp) = 2^7
                decomposition_chunks: 4, // for now, there's no reason why decomposition_chunks here shall be different from commitment_recursion.decomposition_chunks. I will use that assumption in sumcheck.
                rank: 1,
                next: None,
            },
            projection_recursion: AuxProjection::Fine {
                nof_batches: 2,
                recursion_constant_term: AuxRecursionConfig {
                    decomposition_base_log: 15,
                    decomposition_chunks: 4,
                    rank: 1,
                    next: None,
                },
                recursion_batched_projection: AuxRecursionConfig {
                    decomposition_base_log: 15,
                    decomposition_chunks: 4,
                    rank: 1,
                    next: None,
                },
            },

            witness_decomposition_chunks: 2,
            witness_decomposition_base_log: 10, // no decomposition

            next: None,
        }))),
    }
    .generate_config()
});

pub static TOY_CONFIG: LazyLock<Config> = LazyLock::new(|| {
    AuxSumcheckConfig {
        witness_height: 512,
        witness_width: 16,
        projection_ratio: 32,
        projection_height: 8, // small for testing
        basic_commitment_rank: 2,
        nof_openings: 1,

        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 15,
            decomposition_chunks: 4,
            rank: 1,
            next: Some(Box::new(AuxRecursionConfig {
                decomposition_base_log: 7,
                decomposition_chunks: 8,
                rank: 1,
                next: None,
            })),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 15,
            decomposition_chunks: 4,
            rank: 1,
            next: None,
        },
        projection_recursion: AuxProjection::Coarse(AuxRecursionConfig {
            decomposition_base_log: 15,
            decomposition_chunks: 2,
            rank: 1,
            next: None,
        }),

        witness_decomposition_chunks: 2,
        witness_decomposition_base_log: 15,
        next: None,
    }
    .generate_config()
});

pub static TOY_CONFIG_II: LazyLock<Config> = LazyLock::new(|| {
    AuxSumcheckConfig {
        witness_height: 1024,
        witness_width: 16,
        projection_ratio: 32,
        projection_height: 256,
        basic_commitment_rank: 2,
        nof_openings: 1,

        commitment_recursion: AuxRecursionConfig {
            decomposition_base_log: 15,
            decomposition_chunks: 4,
            rank: 1,
            next: Some(Box::new(AuxRecursionConfig {
                decomposition_base_log: 7,
                decomposition_chunks: 8,
                rank: 1,
                next: None,
            })),
        },
        opening_recursion: AuxRecursionConfig {
            decomposition_base_log: 15,
            decomposition_chunks: 4,
            rank: 1,
            next: None,
        },
        projection_recursion: AuxProjection::Fine {
            nof_batches: 2,
            recursion_constant_term: AuxRecursionConfig {
                decomposition_base_log: 10,
                decomposition_chunks: 2,
                rank: 1,
                next: None,
            },
            recursion_batched_projection: AuxRecursionConfig {
                decomposition_base_log: 15,
                decomposition_chunks: 4,
                rank: 1,
                next: None,
            },
        },

        witness_decomposition_chunks: 2,
        witness_decomposition_base_log: 15,
        next: Some(Box::new(AuxConfig::Simple(SimpleConfig {
            witness_height: 256,
            witness_width: 16,
            projection_ratio: 128,
            projection_height: 256,
            projection_nof_batches: 2,
            basic_commitment_rank: 2,
            witness_norm_bound: f64::INFINITY,
            projection_norm_bound: f64::INFINITY,
        }))),
    }
    .generate_config()
});

pub static CONFIG: LazyLock<Config> = LazyLock::new(|| P.clone());

#[derive(Clone)]
pub enum Config {
    Sumcheck(SumcheckConfig),
    Intermediate(IntermediateConfig),
    Simple(SimpleConfig),
}

pub trait ConfigBase: Any {
    fn witness_height(&self) -> usize;
    fn witness_width(&self) -> usize;
    fn projection_ratio(&self) -> usize;
    fn projection_height(&self) -> usize;
    fn basic_commitment_rank(&self) -> usize;
}

pub fn config_base_from_config(config: &Config) -> &dyn ConfigBase {
    match config {
        Config::Sumcheck(sumcheck_config) => sumcheck_config,
        Config::Simple(simple_config) => simple_config,
        Config::Intermediate(intermediate_config) => intermediate_config,
    }
}

#[derive(Clone)]
pub struct SumcheckConfig {
    pub witness_height: usize,
    pub witness_width: usize,
    pub projection_ratio: usize,  // shall be likely the witness_height
    pub projection_height: usize, // likely 256 unless for testing
    pub commitment_recursion: RecursionConfig,
    pub next_level_usage_ratio: f64, // we always assume that width is a power of two, but next_level_usage_ratio can be less than 1. I.e. for width = 16, and next_level_usage_ratio = 0.51, we only use 9 cols in the next level.
    pub opening_recursion: RecursionConfig,
    pub projection_recursion: Projection,
    pub nof_openings: usize,

    pub witness_decomposition_base_log: usize,
    pub witness_decomposition_chunks: usize,
    pub folded_witness_prefix: Prefix,

    pub basic_commitment_rank: usize,
    pub composed_witness_length: usize,

    pub norm_bound: f64,
    pub most_inner_norm_bound: f64,

    pub next: Option<Box<Config>>, // for multiple rounds
}

impl ConfigBase for SumcheckConfig {
    fn witness_height(&self) -> usize {
        self.witness_height
    }

    fn witness_width(&self) -> usize {
        self.witness_width
    }

    fn projection_ratio(&self) -> usize {
        self.projection_ratio
    }

    fn projection_height(&self) -> usize {
        self.projection_height
    }
    fn basic_commitment_rank(&self) -> usize {
        self.basic_commitment_rank
    }
}

#[derive(Clone)]
pub struct IntermediateConfig {
    pub witness_height: usize,
    pub witness_width: usize,
    pub projection_ratio: usize,
    pub projection_height: usize,
    pub nof_openings: usize,
    pub projection_nof_batches: usize,
    pub basic_commitment_rank: usize,

    pub witness_decomposition_base_log: usize,
    pub witness_decomposition_chunks: usize,

    pub norm_bound: f64,
    pub projection_norm_bound: f64,

    pub next: Option<Box<Config>>,
}

impl ConfigBase for IntermediateConfig {
    fn witness_height(&self) -> usize {
        self.witness_height
    }

    fn witness_width(&self) -> usize {
        self.witness_width
    }

    fn projection_ratio(&self) -> usize {
        self.projection_ratio
    }

    fn projection_height(&self) -> usize {
        self.projection_height
    }
    fn basic_commitment_rank(&self) -> usize {
        self.basic_commitment_rank
    }
}

#[derive(Clone)]
pub struct SimpleConfig {
    pub witness_height: usize,
    pub witness_width: usize,
    pub projection_ratio: usize,  // shall be likely the witness_height
    pub projection_height: usize, // likely 256 unless for testing
    pub projection_nof_batches: usize,
    pub basic_commitment_rank: usize,
    pub witness_norm_bound: f64,
    pub projection_norm_bound: f64,
    // pub next: Option<Box<SimpleConfig>>, // for multiple rounds
}

impl ConfigBase for SimpleConfig {
    fn witness_height(&self) -> usize {
        self.witness_height
    }

    fn witness_width(&self) -> usize {
        self.witness_width
    }

    fn projection_ratio(&self) -> usize {
        self.projection_ratio
    }

    fn projection_height(&self) -> usize {
        self.projection_height
    }
    fn basic_commitment_rank(&self) -> usize {
        self.basic_commitment_rank
    }
}

pub enum RoundProof {
    Sumcheck(SumcheckRoundProof),
    Simple(SimpleRoundProof),
    Intermediate(IntermediateRoundProof),
}

pub enum NextRoundCommitment {
    Recursive(RecursiveCommitment), // if the next round is sumcheck
    Simple(HorizontallyAlignedMatrix<RingElement>), // if the next round is simple or intermediate
}

pub trait SizeableProof {
    fn size_in_bits(&self) -> usize;
}

impl SizeableProof for RoundProof {
    fn size_in_bits(&self) -> usize {
        match self {
            RoundProof::Sumcheck(p) => p.size_in_bits(),
            RoundProof::Simple(p) => p.size_in_bits(),
            RoundProof::Intermediate(p) => p.size_in_bits(),
        }
    }
}

pub struct SumcheckRoundProof {
    pub polys: Vec<Polynomial<QuadraticExtension>>,
    pub claim_over_witness: RingElement,
    pub claim_over_witness_conjugate: RingElement,
    pub norm_claim: RingElement,
    pub most_inner_norm_claim: RingElement,
    pub rc_opening_inner: Vec<RingElement>,
    pub rc_coarse_projection_inner: Option<Vec<RingElement>>,
    pub rc_fine_projection_inner: Option<(Vec<RingElement>, Vec<RingElement>)>,
    pub constant_term_claims: Option<Vec<RingElement>>,
    pub next_round_commitment: Option<NextRoundCommitment>,
    pub next: Option<Box<RoundProof>>,
    /// Outer length of next round's commitment, recorded for the size-table header.
    /// `next_round_commitment` only carries the innermost layer, so the outer length
    /// would otherwise be unrecoverable when printing the breakdown.
    pub next_commitment_outer_len: Option<usize>,
}

pub fn to_kb(size_in_bits: usize) -> f64 {
    size_in_bits as f64 / 8.0 / 1024.0
}

fn emit_size_table(name: &str, params: &[(&str, String)], rows: &[(&str, usize)], total: usize) {
    const LABEL_W: usize = 32;
    let param_str = params
        .iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join(" ");
    if param_str.is_empty() {
        tracing::debug!("\n=== {name} ===");
    } else {
        tracing::debug!("\n=== {name} === {param_str}");
    }
    for (label, bits) in rows {
        tracing::debug!("  {:<LABEL_W$}  {:>8.2} KB", label, to_kb(*bits));
    }
    tracing::debug!("  {:<LABEL_W$}  {:>8.2} KB", "TOTAL", to_kb(total));
}

impl SizeableProof for SumcheckRoundProof {
    fn size_in_bits(&self) -> usize {
        let mut rows: Vec<(&str, usize)> = Vec::new();

        let polys_size: usize = self
            .polys
            .iter()
            .flat_map(|p| &p.coefficients[0..p.num_coefficients])
            .map(|c| c.size_in_bits())
            .sum();
        rows.push(("Polys", polys_size));

        let claims_size: usize = [
            &self.claim_over_witness,
            &self.claim_over_witness_conjugate,
            &self.norm_claim,
            &self.most_inner_norm_claim,
        ]
        .iter()
        .map(|c| c.size_in_bits())
        .sum();
        rows.push(("Claims", claims_size));

        rows.push((
            "RC opening inner",
            self.rc_opening_inner
                .iter()
                .map(|el| el.size_in_bits())
                .sum(),
        ));

        if let Some(rc_coarse) = &self.rc_coarse_projection_inner {
            rows.push((
                "RC coarse projection inner",
                rc_coarse.iter().map(|el| el.size_in_bits()).sum(),
            ));
        }

        if let Some((p0, p1)) = &self.rc_fine_projection_inner {
            rows.push((
                "RC fine projection inner",
                p0.iter().chain(p1).map(|el| el.size_in_bits()).sum(),
            ));
        }

        if let Some(ct) = &self.constant_term_claims {
            rows.push((
                "Constant term claims",
                ct.iter().map(|el| el.size_in_bits()).sum(),
            ));
        }

        let next_round_size: usize = match &self.next_round_commitment {
            Some(NextRoundCommitment::Recursive(rc)) => rc.iter().map(|el| el.size_in_bits()).sum(),
            Some(NextRoundCommitment::Simple(mat)) => {
                mat.data.iter().map(|el| el.size_in_bits()).sum()
            }
            None => 0,
        };
        rows.push(("Next round commitment", next_round_size));

        let local: usize = rows.iter().map(|(_, s)| s).sum();

        let vars = self.polys.len();
        let hypercube = if vars > 0 { 1u64 << (vars - 1) } else { 0 };
        let mut params: Vec<(&str, String)> = vec![
            ("vars", vars.to_string()),
            ("hypercube", hypercube.to_string()),
        ];
        if let Some(outer_len) = self.next_commitment_outer_len {
            params.push(("next_commitment", outer_len.to_string()));
        }
        emit_size_table("Sumcheck round", &params, &rows, local);

        local + self.next.as_deref().map_or(0, SizeableProof::size_in_bits)
    }
}

pub struct SimpleRoundProof {
    pub folded_witness: VerticallyAlignedMatrix<RingElement>,
    pub projection_image_ct: VerticallyAlignedMatrix<RingElement>, // cosntant term projection image embedded
    pub batched_projection_image: HorizontallyAlignedMatrix<RingElement>,
    pub opening_rhs: HorizontallyAlignedMatrix<RingElement>,
}

impl SizeableProof for SimpleRoundProof {
    fn size_in_bits(&self) -> usize {
        let rows: Vec<(&str, usize)> = vec![
            (
                "Folded witness",
                self.folded_witness
                    .data
                    .iter()
                    .map(|el| el.size_in_bits())
                    .sum(),
            ),
            (
                "Projection image ct",
                self.projection_image_ct
                    .data
                    .iter()
                    .map(|el| el.size_in_bits())
                    .sum(),
            ),
            (
                "Batched projection image",
                self.batched_projection_image
                    .data
                    .iter()
                    .map(|el| el.size_in_bits())
                    .sum(),
            ),
            (
                "Opening RHS",
                self.opening_rhs
                    .data
                    .iter()
                    .map(|el| el.size_in_bits())
                    .sum(),
            ),
        ];
        let total: usize = rows.iter().map(|(_, s)| s).sum();
        let params = [(
            "opening",
            format!("{}x{}", self.opening_rhs.height, self.opening_rhs.width),
        )];
        emit_size_table("Simple round", &params, &rows, total);
        total
    }
}

pub struct IntermediateRoundProof {
    pub opening_rhs: HorizontallyAlignedMatrix<RingElement>,
    pub polys: Vec<Polynomial<QuadraticExtension>>,
    pub claim_over_witness: RingElement,
    pub claim_over_witness_conjugate: RingElement,
    pub norm_claim: RingElement,
    pub next_round_commitment: Option<NextRoundCommitment>,
    pub projection_image_ct: VerticallyAlignedMatrix<RingElement>,
    pub batched_projection_image: HorizontallyAlignedMatrix<RingElement>,
    pub next: Option<Box<RoundProof>>,
}

impl SizeableProof for IntermediateRoundProof {
    fn size_in_bits(&self) -> usize {
        let polys_size: usize = self
            .polys
            .iter()
            .flat_map(|p| &p.coefficients[0..p.num_coefficients])
            .map(|c| c.size_in_bits())
            .sum();

        let claims_size: usize = [
            &self.claim_over_witness,
            &self.claim_over_witness_conjugate,
            &self.norm_claim,
        ]
        .iter()
        .map(|c| c.size_in_bits())
        .sum();

        // Intermediate's `next_round_commitment` is always `Simple(mat)`, and `mat.data.len()`
        // IS the outer length — so we can derive `next_commitment` for the header directly.
        let (next_round_size, next_commitment_outer_len): (usize, Option<usize>) =
            match &self.next_round_commitment {
                Some(NextRoundCommitment::Recursive(_)) => unreachable!(
                    "Intermediate round should not have recursive commitment for next round."
                ),
                Some(NextRoundCommitment::Simple(mat)) => (
                    mat.data.iter().map(|el| el.size_in_bits()).sum(),
                    Some(mat.data.len()),
                ),
                None => (0, None),
            };

        let rows: Vec<(&str, usize)> = vec![
            ("Polys", polys_size),
            ("Claims", claims_size),
            (
                "Projection image ct",
                self.projection_image_ct
                    .data
                    .iter()
                    .map(|el| el.size_in_bits())
                    .sum(),
            ),
            (
                "Batched projection image",
                self.batched_projection_image
                    .data
                    .iter()
                    .map(|el| el.size_in_bits())
                    .sum(),
            ),
            (
                "Opening RHS",
                self.opening_rhs
                    .data
                    .iter()
                    .map(|el| el.size_in_bits())
                    .sum(),
            ),
            ("Next round commitment", next_round_size),
        ];
        let local: usize = rows.iter().map(|(_, s)| s).sum();

        let mut params: Vec<(&str, String)> = vec![
            ("sumcheck_vars", self.polys.len().to_string()),
            (
                "opening",
                format!("{}x{}", self.opening_rhs.height, self.opening_rhs.width),
            ),
        ];
        if let Some(outer_len) = next_commitment_outer_len {
            params.push(("next_commitment", outer_len.to_string()));
        }
        emit_size_table("Intermediate round", &params, &rows, local);

        local + self.next.as_deref().map_or(0, SizeableProof::size_in_bits)
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

pub fn paste_recursive_commitment(
    dest: &mut Vec<RingElement>,
    commitment: &RecursiveCommitmentWithAux,
    config: &RecursionConfig,
) {
    paste_by_prefix(dest, &commitment.committed_data, &config.prefix);

    if let (Some(next_commitment), Some(next_config)) = (&commitment.next, &config.next) {
        paste_recursive_commitment(dest, next_commitment, next_config);
    }
}

#[inline]
pub fn slice_by_prefix(src: &Vec<RingElement>, prefix: &Prefix) -> Vec<RingElement> {
    let start = prefix.prefix << (src.len().ilog2() as usize - prefix.length);
    let length = 1 << (src.len().ilog2() as usize - prefix.length);
    src[start..start + length].to_vec()
}
