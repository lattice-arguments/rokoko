use crate::common::{
    matrix::HorizontallyAlignedMatrix,
    ring_arithmetic::{Representation, RingElement},
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
        debug_assert!(max_wit_dim.is_power_of_two());

        let shared_v_module = HorizontallyAlignedMatrix::<RingElement> {
            data: sample_random_vector(
                max_wit_dim.ilog2() as usize * max_module_size,
                Representation::IncompleteNTT,
            ),
            width: max_wit_dim.ilog2() as usize,
            height: max_module_size,
        };

        let (cks, structured_cks): (Vec<_>, Vec<_>) = (1..=max_wit_dim.ilog2() as usize)
            .map(|i| {
                let mut ck = Vec::with_capacity(max_module_size);
                let mut sck = Vec::with_capacity(max_module_size);

                for j in 0..max_module_size {
                    let v_module = shared_v_module
                        .row(j)
                        .iter()
                        .skip(max_wit_dim.ilog2() as usize - i)
                        .cloned()
                        .collect();

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
