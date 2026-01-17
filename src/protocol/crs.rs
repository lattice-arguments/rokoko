use crate::common::{
    ring_arithmetic::Representation,
    sampling::sample_random_vector,
    structured_row::{PreprocessedRow, StructuredRow},
};
pub type CK = Vec<PreprocessedRow>;
pub type SCK = Vec<StructuredRow>;

/// Struct representing the Common Reference String (CRS).
pub struct CRS {
    pub cks: Vec<CK>,             // Commitment keys for each witness length
    pub structured_cks: Vec<SCK>, // Structured commitment keys for each witness length
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

/// Generates a Common Reference String (CRS).
impl CRS {
    pub fn gen_crs(max_wit_dim: usize, max_module_size: usize) -> CRS {
        assert!(max_wit_dim.is_power_of_two());

        let (cks, structured_cks): (Vec<_>, Vec<_>) = (1..=max_wit_dim.ilog2() as usize)
            .map(|nof_tensor_layers| {
                let mut ck = Vec::with_capacity(max_module_size);
                let mut sck = Vec::with_capacity(max_module_size);

                for _ in 0..max_module_size {
                    let v_module =
                        sample_random_vector(nof_tensor_layers, Representation::IncompleteNTT);

                    let structured_row = StructuredRow {
                        tensor_layers: v_module,
                    };
                    let preprocessed_row = PreprocessedRow::from_structured_row(&structured_row);
                    ck.push(preprocessed_row);
                    sck.push(structured_row);
                }
                (ck, sck)
            })
            .unzip();

        CRS {
            cks,
            structured_cks,
        }
    }
}
