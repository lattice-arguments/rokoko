use std::ops::Index;

use crate::common::{
    config::MOD_Q,
    ring_arithmetic::{Representation, RingElement},
};

#[derive(Debug, Clone)]
pub struct StructuredRow {
    // Each layer corresponds to one dimension in the tensor product structure.
    // Each layer has two correlated elements, corresponding to the two choices (0 or 1) at that dimension.
    // Each layer is of the form [1 - a, a] for some a which is similar to the evaluation point.
    // For example, for a 3-layer structured row, the tensor_layers might look like:
    // [
    //   [1 - a, a],  // Layer 0
    //   [1 - b, b],  // Layer 1
    //   [1 - c, c],  // Layer 2
    // ]
    // Then, the entry at position 5 (binary 100) would be computed as:
    // (1 - a) * (1 - b) * c.
    // Notably, the order of layers corresponds to the inverse order of bits in the index,
    // (i.e. with the first layer corresponding to the least significant bit.)
    pub tensor_layers: Vec<RingElement>,
}

impl StructuredRow {
    pub fn at(&self, pos: usize) -> RingElement {
        let mut result = RingElement::one(Representation::IncompleteNTT);
        let mut index = pos;
        for layer in &self.tensor_layers {
            let bit = index & 1;
            index >>= 1;
            if bit == 0 {
                result *= &(&RingElement::one(Representation::IncompleteNTT) - layer);
            } else {
                result *= layer;
            }
        }
        result
    }
}

// impl Index<usize> for StructuredRow {
//     type Output = RingElement;

//     fn index(&self, index: usize) -> &Self::Output {
//         panic!("Indexing StructuredRow directly is not supported. Use the `at` method instead.");
//     }
// }

pub struct PreprocessedRow {
    pub structured_row: StructuredRow,
    pub preprocessed_row: Vec<RingElement>,
}

impl PreprocessedRow {
    pub fn from_structured_row(structured_row: StructuredRow) -> Self {
        let mut result = Vec::with_capacity(2usize.pow(structured_row.tensor_layers.len() as u32));
        result.push(RingElement::one(Representation::IncompleteNTT));

        for layer in &structured_row.tensor_layers {
            let mut new_entries: Vec<RingElement> = Vec::with_capacity(result.len());
            for r in &mut result {
                let r1 = &*r * &layer;
                new_entries.push(r1);
                *r *= &(&RingElement::one(Representation::IncompleteNTT) - layer);
            }
            for e in new_entries {
                result.push(e);
            }
        }
        PreprocessedRow {
            structured_row,
            preprocessed_row: result,
        }
    }
}

#[test]
fn test_structured_row() {
    let tensor_layers = vec![
        RingElement::constant(2, Representation::IncompleteNTT),
        RingElement::constant(3, Representation::IncompleteNTT),
        RingElement::constant(5, Representation::IncompleteNTT),
    ];
    let structured_row = StructuredRow { tensor_layers };

    assert_eq!(
        structured_row.at(0),
        RingElement::constant(
            (MOD_Q as i64 + (1 - 2) * (1 - 3) * (1 - 5)) as u64,
            Representation::IncompleteNTT
        )
    );
    assert_eq!(
        structured_row.at(1),
        RingElement::constant(
            (MOD_Q as i64 + 2 * (1 - 3) * (1 - 5)) as u64,
            Representation::IncompleteNTT
        )
    );
    assert_eq!(
        structured_row.at(2),
        RingElement::constant(
            (MOD_Q as i64 + (1 - 2) * 3 * (1 - 5)) as u64,
            Representation::IncompleteNTT
        )
    );
    assert_eq!(
        structured_row.at(3),
        RingElement::constant(
            (MOD_Q as i64 + 2 * 3 * (1 - 5)) as u64,
            Representation::IncompleteNTT
        )
    );
    assert_eq!(
        structured_row.at(4),
        RingElement::constant(
            (MOD_Q as i64 + (1 - 2) * (1 - 3) * 5) as u64,
            Representation::IncompleteNTT
        )
    );
    assert_eq!(
        structured_row.at(5),
        RingElement::constant(
            (MOD_Q as i64 + 2 * (1 - 3) * 5) as u64,
            Representation::IncompleteNTT
        )
    );
    assert_eq!(
        structured_row.at(6),
        RingElement::constant(
            (MOD_Q as i64 + (1 - 2) * 3 * 5) as u64,
            Representation::IncompleteNTT
        )
    );
    assert_eq!(
        structured_row.at(7),
        RingElement::constant(
            (MOD_Q as i64 + 2 * 3 * 5) as u64,
            Representation::IncompleteNTT
        )
    );
}

#[test]
fn test_preprocessed_row() {
    let tensor_layers = vec![
        RingElement::constant(1, Representation::IncompleteNTT),
        RingElement::constant(3, Representation::IncompleteNTT),
        RingElement::constant(5, Representation::IncompleteNTT),
    ];
    let structured_row = StructuredRow { tensor_layers };

    let structured_row_unused = structured_row.clone();

    let preprocessed_row = PreprocessedRow::from_structured_row(structured_row_unused);

    for i in 0..preprocessed_row.preprocessed_row.len() {
        assert_eq!(preprocessed_row.preprocessed_row[i], structured_row.at(i));
    }
}
