use crate::common::{
    ring_arithmetic::{Representation, RingElement},
    sampling::sample_random_vector,
    structured_row::{PreprocessedRow, StructuredRow},
};

/// Struct representing the Common Reference String (CRS) for cryptographic operations.
pub struct CRS {
    pub(crate) ck: Vec<PreprocessedRow>,
}

/// Generates a Common Reference String (CRS).
///
/// # Returns
///
/// A `CRS` containing commitment keys (`ck`) a randomly sampled vector (`a`), and a challenge set.
impl CRS {
    pub fn gen_crs(wit_dim: usize, module_size: usize) -> CRS {
        let nof_tensor_layers = wit_dim.ilog2() as usize;
        let mut ck = Vec::with_capacity(module_size);

        for _ in 0..module_size {
            let v_module = sample_random_vector(nof_tensor_layers, Representation::IncompleteNTT);

            let structured_row = StructuredRow {
                tensor_layers: v_module,
            };
            let preprocessed_row = PreprocessedRow::from_structured_row(structured_row);
            ck.push(preprocessed_row);
        }

        CRS { ck }
    }
}
