use crate::common::{
    ring_arithmetic::{Representation, RingElement},
    sampling::{AesCtrPublicSampler, PUBLIC_CRS_SEED},
    structured_row::PreprocessedRow,
};
use crate::protocol::commitment::RecursionConfig;
use crate::protocol::config::{Config, Projection, SimpleConfig, SumcheckConfig};

pub type CK = Vec<PreprocessedRow>;

/// Struct representing the Common Reference String (CRS).
pub struct CRS {
    /// Keys by witness length, empty at the lengths the chain never commits over.
    pub cks: Vec<CK>,
    /// The root key in the CRT basis, planned from the witness schedule rather than from a
    /// witness, so that the transform of the key never sits on the commitment's critical path.
    #[cfg(feature = "crt-commitment")]
    pub crt_root: Option<(
        crate::protocol::commitment_crt::Plan,
        crate::protocol::commitment_crt::CrtKey,
    )>,
}

#[derive(Debug)]
pub struct VerifierCRS {
    pub cks: Vec<CK>,
    /// The simple round's rows, which the verifier recommits with rather than evaluates.
    pub simple_ck: CK,
}

impl VerifierCRS {
    pub fn ck_for_wit_dim(&self, wit_dim: usize) -> &CK {
        &self.cks[key_index(wit_dim)]
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
    pub fn ck_for_wit_dim(&self, wit_dim: usize) -> &CK {
        &self.cks[key_index(wit_dim)]
    }
}

fn key_index(wit_dim: usize) -> usize {
    debug_assert!(wit_dim.is_power_of_two(), "key lengths are dyadic");
    wit_dim.ilog2() as usize - 1
}

/// The stream is keyed by the length alone and read row by row, so row `i` is the same vector
/// however many rows the caller asks for.
fn dense_ck(len: usize, rows: usize) -> CK {
    let mut seed = PUBLIC_CRS_SEED.to_vec();
    seed.extend_from_slice(b"/dense/");
    seed.extend_from_slice(&(len as u64).to_le_bytes());
    let mut sampler = AesCtrPublicSampler::from_seed(&seed);

    (0..rows)
        .map(|_| {
            let mut row = Vec::with_capacity(len);
            for _ in 0..len {
                let mut element = RingElement::new(Representation::IncompleteNTT);
                sampler.fill_ring_element(&mut element, Representation::IncompleteNTT);
                row.push(element);
            }
            PreprocessedRow {
                preprocessed_row: row,
            }
        })
        .collect()
}

fn recursion_shapes(config: &RecursionConfig, shapes: &mut Vec<(usize, usize)>) {
    shapes.push((config.block_len(), config.blockwise_rank()));
    if let Some(next) = config.next.as_deref() {
        recursion_shapes(next, shapes);
    }
}

fn sumcheck_shapes(config: &SumcheckConfig, shapes: &mut Vec<(usize, usize)>) {
    shapes.push((
        config.witness_height / config.basic_commitment_diag_blocks,
        config.basic_commitment_rank / config.basic_commitment_diag_blocks,
    ));
    recursion_shapes(&config.commitment_recursion, shapes);
    recursion_shapes(&config.opening_recursion, shapes);
    match &config.projection_recursion {
        Projection::Coarse(recursion) => recursion_shapes(recursion, shapes),
        Projection::Fine(recursion) => {
            recursion_shapes(&recursion.recursion_constant_term, shapes);
            recursion_shapes(&recursion.recursion_batched_projection, shapes);
        }
        Projection::Skip => {}
    }
    if let Some(next) = config.next.as_deref() {
        key_shapes(next, shapes);
    }
}

/// Every `(length, rows)` the chain commits over, narrowed to one block. A dense key costs its
/// whole shape, so keying every dyadic length up to the composed witness runs to gigabytes.
fn key_shapes(config: &Config, shapes: &mut Vec<(usize, usize)>) {
    match config {
        Config::Sumcheck(c) => sumcheck_shapes(c, shapes),
        Config::Intermediate(c) => {
            shapes.push((c.witness_height, c.basic_commitment_rank));
            if let Some(next) = c.next.as_deref() {
                key_shapes(next, shapes);
            }
        }
        Config::Simple(c) => shapes.push((c.witness_height, c.basic_commitment_rank)),
    }
}

fn gen_cks(shapes: &[(usize, usize)]) -> Vec<CK> {
    let mut rows = Vec::new();
    for (len, wanted) in shapes {
        let index = key_index(*len);
        if rows.len() <= index {
            rows.resize(index + 1, 0);
        }
        rows[index] = rows[index].max(*wanted);
    }

    rows.into_iter()
        .enumerate()
        .map(|(index, rows)| {
            if rows == 0 {
                Vec::new()
            } else {
                dense_ck(1 << (index + 1), rows)
            }
        })
        .collect()
}

fn chain_shapes(config: &SumcheckConfig) -> Vec<(usize, usize)> {
    let mut shapes = Vec::new();
    sumcheck_shapes(config, &mut shapes);
    shapes
}

/// Generates a Common Reference String (CRS).
impl CRS {
    pub fn gen_crs(max_wit_dim: usize, max_module_size: usize) -> CRS {
        CRS {
            #[cfg(feature = "crt-commitment")]
            crt_root: None,
            cks: gen_cks(&[(max_wit_dim, max_module_size)]),
        }
    }

    pub fn gen_prover_crs(config: &SumcheckConfig) -> CRS {
        #[allow(unused_mut)]
        let mut crs = CRS {
            #[cfg(feature = "crt-commitment")]
            crt_root: None,
            cks: gen_cks(&chain_shapes(config)),
        };
        #[cfg(feature = "crt-commitment")]
        {
            use crate::protocol::commitment_crt::{CrtKey, Plan};
            use crate::protocol::params::WITNESS_CONFIG;
            let blocks = config.basic_commitment_diag_blocks;
            let rank = config.basic_commitment_rank / blocks;
            let rows = config.witness_height / blocks;
            let bound = 1u64 << (WITNESS_CONFIG.decomposition_base_log - 1);
            let plan = Plan::for_shape(rows, bound, rank);
            let key = CrtKey::preprocess(crs.ck_for_wit_dim(rows), rank, &plan);
            crs.crt_root = Some((plan, key));
        }
        crs
    }

    pub fn gen_verifier_crs(config: &SumcheckConfig) -> VerifierCRS {
        let mut crs = VerifierCRS {
            cks: gen_cks(&chain_shapes(config)),
            simple_ck: Vec::new(),
        };
        if let Some(simple) = simple_round(config) {
            crs.simple_ck = crs
                .ck_for_wit_dim(simple.witness_height)
                .iter()
                .take(simple.basic_commitment_rank)
                .cloned()
                .collect();
        }
        crs
    }
}
