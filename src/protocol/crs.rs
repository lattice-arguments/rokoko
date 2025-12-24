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
        let v_module = sample_random_vector(module_size, Representation::IncompleteNTT);

        let nof_tensor_layers = wit_dim.ilog2() as usize;

        let ck = v_module
            .iter()
            .map(|elem| {
                let mut tensor_layers = Vec::with_capacity(nof_tensor_layers);
                let mut elem_power = elem.clone();
                for i in 0..nof_tensor_layers {
                    let layer = [
                        RingElement::one(Representation::IncompleteNTT),
                        elem_power.clone(),
                    ];
                    // TODO how to optimize this?
                    // This is not very important as CRS generation is done only once.
                    elem_power *= &elem_power.clone();
                    tensor_layers.push(layer);
                }
                let row = StructuredRow { tensor_layers };
                PreprocessedRow::from_structured_row(row)
            })
            .collect();

        CRS { ck }
    }
}
