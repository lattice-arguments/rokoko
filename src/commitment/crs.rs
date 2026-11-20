//! The common reference string of the commitment scheme: commitment keys as structured (tensor)
//! rows drawn from a public seed, one key per power-of-two input length (`CRS`), and the
//! verifier's structured-only `VerifierCRS`. The CRS a round chain needs is built in
//! `rok::config::crs`.

use crate::{
    common::{
        matrix::HorizontallyAlignedMatrix,
        sampling::{sample_public_vector_from_seed, PUBLIC_CRS_SEED},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    ring::ring_arithmetic::{Representation, RingElement},
};

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

pub(crate) fn gen_structured_cks(max_wit_dim: usize, max_module_size: usize) -> Vec<SCK> {
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
}
