use crate::common::{
    ring_arithmetic::Representation,
    sampling::sample_random_vector,
    structured_row::{PreprocessedRow, StructuredRow},
};
pub type CK = Vec<PreprocessedRow>;
pub type SCK = Vec<StructuredRow>;

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
        debug_assert!(max_wit_dim.is_power_of_two());

        let cks: Vec<_> = (1..=max_wit_dim.ilog2() as usize)
            .map(|i| {
                let mut ck = Vec::with_capacity(max_module_size);

                for _ in 0..max_module_size {
                    let row = sample_random_vector(
                        2u32.pow(i as u32) as usize,
                        Representation::IncompleteNTT,
                    );
                    let preprocessed_row = PreprocessedRow {
                        preprocessed_row: row,
                    };
                    ck.push(preprocessed_row);
                }
                ck
            })
            .collect();

        CRS { cks }
    }
}
