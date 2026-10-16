use crate::common::{
    matrix::HorizontallyAlignedMatrix,
    ring_arithmetic::{Representation, RingElement},
    sampling::{sample_public_vector_from_seed, PUBLIC_CRS_SEED},
    structured_row::{PreprocessedRow, StructuredRow},
};
use crate::protocol::commitment::RecursionConfig;
use crate::protocol::config::{Config, Projection, SimpleConfig, SumcheckConfig};

pub type CK = Vec<PreprocessedRow>;
pub type SCK = Vec<StructuredRow>;

/// Struct representing the Common Reference String (CRS).
pub struct CRS {
    pub cks: Vec<CK>,             // Commitment keys for each witness length
    pub structured_cks: Vec<SCK>, // Structured commitment keys for each witness length
}

/// Only the structured keys; the expanded rows are prover-side preprocessing.
#[derive(Debug)]
pub struct VerifierCRS {
    pub structured_cks: Vec<SCK>,
    /// The simple round's key rows, expanded at setup.
    pub simple_ck: CK,
}

impl VerifierCRS {
    pub fn structured_ck_for_wit_dim(&self, wit_dim: usize) -> &Vec<StructuredRow> {
        let index = wit_dim.ilog2() as usize - 1;
        &self.structured_cks[index]
    }
}

fn simple_round(config: &SumcheckConfig) -> Option<&SimpleConfig> {
    let mut next = config.next.as_deref();
    while let Some(round) = next {
        next = match round {
            Config::Sumcheck(c) => c.next.as_deref(),
            Config::Intermediate(c) => c.next.as_deref(),
            Config::Simple(c) => return Some(c),
        };
    }
    None
}

impl CRS {
    // Returns the commitment key for a given witness dimension.
    pub fn ck_for_wit_dim(&self, wit_dim: usize) -> &Vec<PreprocessedRow> {
        let index = wit_dim.ilog2() as usize - 1;
        &self.cks[index]
    }

    // Returns the structured commitment key for a given witness dimension.
    pub fn structured_ck_for_wit_dim(&self, wit_dim: usize) -> &Vec<StructuredRow> {
        let index = wit_dim.ilog2() as usize - 1;
        &self.structured_cks[index]
    }
}

fn key_rows(rows: &mut [usize], wit_dim: usize, rank: usize) {
    let dim = &mut rows[wit_dim.ilog2() as usize - 1];
    *dim = (*dim).max(rank);
}

fn rows_of_tree(rows: &mut [usize], tree: &RecursionConfig) {
    let mut level = Some(tree);
    while let Some(t) = level {
        key_rows(rows, t.committed_len(), t.rank);
        level = t.next.as_deref();
    }
}

fn rows_of_round(rows: &mut [usize], config: &SumcheckConfig) {
    key_rows(rows, config.witness_height, config.basic_commitment_rank);
    rows_of_tree(rows, &config.commitment_recursion);
    rows_of_tree(rows, &config.opening_recursion);
    match &config.projection_recursion {
        Projection::Coarse(tree) => rows_of_tree(rows, tree),
        Projection::Fine(fine) => {
            rows_of_tree(rows, &fine.recursion_constant_term);
            rows_of_tree(rows, &fine.recursion_batched_projection);
        }
        Projection::Skip => {}
    }
    let mut next = config.next.as_deref();
    while let Some(round) = next {
        next = match round {
            Config::Sumcheck(c) => return rows_of_round(rows, c),
            Config::Intermediate(c) => {
                key_rows(rows, c.witness_height, c.basic_commitment_rank);
                c.next.as_deref()
            }
            Config::Simple(c) => {
                key_rows(rows, c.witness_height, c.basic_commitment_rank);
                None
            }
        };
    }
}

fn key_rows_of(config: &SumcheckConfig) -> Vec<usize> {
    let mut rows = vec![0; config.composed_witness_length.ilog2() as usize];
    rows_of_round(&mut rows, config);
    rows
}

fn gen_structured_cks(max_wit_dim: usize, max_module_size: usize) -> Vec<SCK> {
    debug_assert!(max_wit_dim.is_power_of_two());

    let shared_v_module = HorizontallyAlignedMatrix::<RingElement> {
        data: sample_public_vector_from_seed(
            PUBLIC_CRS_SEED,
            max_wit_dim.ilog2() as usize * max_module_size,
            Representation::IncompleteNTT,
        ),
        width: max_wit_dim.ilog2() as usize,
        height: max_module_size,
    };

    (1..=max_wit_dim.ilog2() as usize)
        .map(|i| {
            (0..max_module_size)
                .map(|j| StructuredRow {
                    tensor_layers: shared_v_module
                        .row(j)
                        .iter()
                        .skip(max_wit_dim.ilog2() as usize - i)
                        .cloned()
                        .collect(),
                })
                .collect()
        })
        .collect()
}

/// Generates a Common Reference String (CRS).
impl CRS {
    pub fn gen_crs(max_wit_dim: usize, max_module_size: usize) -> CRS {
        let structured_cks = gen_structured_cks(max_wit_dim, max_module_size);
        let cks = structured_cks
            .iter()
            .map(|sck| {
                sck.iter()
                    .map(PreprocessedRow::from_structured_row)
                    .collect()
            })
            .collect();

        CRS {
            cks,
            structured_cks,
        }
    }

    pub fn gen_prover_crs(config: &SumcheckConfig) -> CRS {
        let rows = key_rows_of(config);
        let structured_cks = gen_structured_cks(
            config.composed_witness_length,
            rows.iter().copied().max().expect("a chain reads at least one key row"),
        );
        let cks = structured_cks
            .iter()
            .zip(&rows)
            .map(|(sck, &n)| {
                sck[..n]
                    .iter()
                    .map(PreprocessedRow::from_structured_row)
                    .collect()
            })
            .collect();
        CRS {
            cks,
            structured_cks,
        }
    }

    pub fn gen_verifier_crs(config: &SumcheckConfig) -> VerifierCRS {
        let mut crs = VerifierCRS {
            structured_cks: gen_structured_cks(
                config.composed_witness_length,
                key_rows_of(config).iter().copied().max().expect("a chain reads at least one key row"),
            ),
            simple_ck: Vec::new(),
        };
        if let Some(simple) = simple_round(config) {
            crs.simple_ck = crs
                .structured_ck_for_wit_dim(simple.witness_height)
                .iter()
                .take(simple.basic_commitment_rank)
                .map(PreprocessedRow::from_structured_row)
                .collect();
        }
        crs
    }
}
