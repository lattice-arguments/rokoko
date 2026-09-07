use crate::common::{
    matrix::HorizontallyAlignedMatrix,
    ring_arithmetic::{Representation, RingElement},
    sampling::{sample_public_vector_from_seed, PUBLIC_CRS_SEED},
    structured_row::{PreprocessedRow, StructuredRow},
};
use crate::protocol::config::SumcheckConfig;

pub type CK = Vec<PreprocessedRow>;
pub type SCK = Vec<StructuredRow>;

/// Struct representing the Common Reference String (CRS).
pub struct CRS {
    pub cks: Vec<CK>,             // Commitment keys for each witness length
    pub structured_cks: Vec<SCK>, // Structured commitment keys for each witness length
    /// The root key in the CRT basis, planned from the witness schedule rather than from a
    /// witness, so that the transform of the key never sits on the commitment's critical path.
    #[cfg(feature = "crt-commitment")]
    pub crt_root: Option<(
        crate::protocol::commitment_crt::Plan,
        crate::protocol::commitment_crt::CrtKey,
    )>,
}

/// Only the structured keys; the expanded rows are prover-side preprocessing.
#[derive(Debug)]
pub struct VerifierCRS {
    pub structured_cks: Vec<SCK>,
}

impl VerifierCRS {
    pub fn structured_ck_for_wit_dim(&self, wit_dim: usize) -> &Vec<StructuredRow> {
        let index = wit_dim.ilog2() as usize - 1;
        &self.structured_cks[index]
    }
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
            #[cfg(feature = "crt-commitment")]
            crt_root: None,
            cks,
            structured_cks,
        }
    }

    /// Two rows of headroom over the basic rank cover the inner rounds.
    pub fn gen_prover_crs(config: &SumcheckConfig) -> CRS {
        #[allow(unused_mut)]
        let mut crs = CRS::gen_crs(
            config.composed_witness_length,
            config.basic_commitment_rank + 2,
        );
        #[cfg(feature = "crt-commitment")]
        {
            use crate::protocol::commitment_crt::{CrtKey, Plan};
            use crate::protocol::params::WITNESS_CONFIG;
            let rows = config.witness_height;
            let bound = 1u64 << (WITNESS_CONFIG.decomposition_base_log - 1);
            let plan = Plan::for_shape(rows, bound, config.basic_commitment_rank);
            let key = CrtKey::preprocess(
                crs.ck_for_wit_dim(rows),
                config.basic_commitment_rank,
                &plan,
            );
            crs.crt_root = Some((plan, key));
        }
        crs
    }

    pub fn gen_verifier_crs(config: &SumcheckConfig) -> VerifierCRS {
        VerifierCRS {
            structured_cks: gen_structured_cks(
                config.composed_witness_length,
                config.basic_commitment_rank + 2,
            ),
        }
    }
}
