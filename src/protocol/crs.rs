use crate::common::{
    ring_arithmetic::{Representation, RingElement},
    sampling::sample_random_vector,
    structured_row::{PreprocessedRow, StructuredRow},
};
pub type CK = Vec<PreprocessedRow>;

/// Struct representing the Common Reference String (CRS).
pub struct CRS {
    pub cks: Vec<CK>, // Commitment keys for each witness length
}

impl CRS {
    // Returns the commitment key for a given witness dimension.
    pub fn ck_for_wit_dim(&self, wit_dim: usize) -> &Vec<PreprocessedRow> {
        let index = wit_dim.ilog2() as usize - 1;
        &self.cks[index]
    }
}

/// Generates a Common Reference String (CRS).
impl CRS {
    pub fn gen_crs(max_wit_dim: usize, max_module_size: usize) -> CRS {
        assert!(max_wit_dim.is_power_of_two());

        let cks = (1..=max_wit_dim.ilog2() as usize)
            .map(|nof_tensor_layers| {
                let mut ck = Vec::with_capacity(max_module_size);

                for _ in 0..max_module_size {
                    let v_module =
                        sample_random_vector(nof_tensor_layers, Representation::IncompleteNTT);

                    let structured_row = StructuredRow {
                        tensor_layers: v_module,
                    };
                    let preprocessed_row = PreprocessedRow::from_structured_row(&structured_row);
                    ck.push(preprocessed_row);
                }
                ck
            })
            .collect::<Vec<Vec<PreprocessedRow>>>();
        CRS { cks }
    }
}
