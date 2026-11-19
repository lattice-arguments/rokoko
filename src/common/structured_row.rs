//! Rows with tensor-product structure, generic over `FiniteRing`: `StructuredRow` holds
//! layers `a_i` and evaluates entry `pos` as the product of `a_i` or `1 − a_i` over the bits
//! of `pos`, the last layer taking the lowest bit; `PreprocessedRow` expands all `2^k` entries.

use crate::ring::{finite_ring::FiniteRing, ring_arithmetic::RingElement};

#[cfg(test)]
use crate::ring::{config::MOD_Q, ring_arithmetic::Representation};

#[derive(Debug, Clone)]
pub struct StructuredRow<E: FiniteRing = RingElement> {
    // Each layer corresponds to one dimension in the tensor product structure.
    // Each layer has two correlated elements, corresponding to the two choices (0 or 1) at that dimension.
    // Each layer is of the form [1 - a, a] for some a which is similar to the evaluation point.
    // For example, for a 3-layer structured row, the tensor_layers might look like:
    // [
    //   [1 - a, a],  // Layer 0
    //   [1 - b, b],  // Layer 1
    //   [1 - c, c],  // Layer 2
    // ]
    // Then, the entry at position 5 (binary 101) would be computed as:
    // a * (1 - b) * c.
    // The first layer corresponds to the most significant bit of the index and the last layer
    // to the least significant bit.
    pub tensor_layers: Vec<E>,
}

impl<E: FiniteRing> StructuredRow<E> {
    pub fn at(&self, pos: usize) -> E {
        let mut result = E::one();
        let mut index = pos;
        for layer in self.tensor_layers.iter().rev() {
            let bit = index & 1;
            index >>= 1;
            if bit == 0 {
                let mut one_minus_layer = E::one();
                one_minus_layer -= layer;
                result *= &one_minus_layer;
            } else {
                result *= layer;
            }
        }
        result
    }
}

// impl<E: FiniteRing> Index<usize> for StructuredRow<E> {
//     type Output = E;
//
//     fn index(&self, index: usize) -> &Self::Output {
//         panic!("Indexing StructuredRow directly is not supported. Use the `at` method instead.");
//     }
// }

#[derive(Debug, Clone)]
pub struct PreprocessedRow<E: FiniteRing = RingElement> {
    // pub structured_row: StructuredRow<E>,
    pub preprocessed_row: Vec<E>,
}

impl<E: FiniteRing> PreprocessedRow<E> {
    pub fn from_structured_row(structured_row: &StructuredRow<E>) -> Self {
        Self::from_layers(&structured_row.tensor_layers)
    }

    pub fn from_layers(tensor_layers: &[E]) -> Self {
        let mut result = Vec::with_capacity(2usize.pow(tensor_layers.len() as u32));
        result.push(E::one());

        for layer in tensor_layers.iter().rev() {
            let mut new_entries: Vec<E> = Vec::with_capacity(result.len());
            for r in &mut result {
                let mut product = E::zero();
                product *= (&*r, layer);
                // r (1 - layer)
                *r -= &product;
                new_entries.push(product);
            }
            for e in new_entries {
                result.push(e);
            }
        }
        PreprocessedRow {
            // structured_row,
            preprocessed_row: result,
        }
    }
}

#[test]
fn test_structured_row() {
    use crate::ring::ring_arithmetic::RingElement;

    let tensor_layers = vec![
        RingElement::constant(5, Representation::IncompleteNTT),
        RingElement::constant(3, Representation::IncompleteNTT),
        RingElement::constant(2, Representation::IncompleteNTT),
    ];
    let structured_row = StructuredRow { tensor_layers };

    debug_assert_eq!(
        structured_row.at(0),
        RingElement::constant(
            (MOD_Q as i64 + (1 - 2) * (1 - 3) * (1 - 5)) as u64, // 0 0 0
            Representation::IncompleteNTT
        )
    );
    debug_assert_eq!(
        structured_row.at(1),
        RingElement::constant(
            (MOD_Q as i64 + 2 * (1 - 3) * (1 - 5)) as u64, // 0 0 1
            Representation::IncompleteNTT
        )
    );
    debug_assert_eq!(
        structured_row.at(2),
        RingElement::constant(
            (MOD_Q as i64 + (1 - 2) * 3 * (1 - 5)) as u64, // 0 1 0
            Representation::IncompleteNTT
        )
    );
    debug_assert_eq!(
        structured_row.at(3),
        RingElement::constant(
            (MOD_Q as i64 + 2 * 3 * (1 - 5)) as u64, // 0 1 1
            Representation::IncompleteNTT
        )
    );
    debug_assert_eq!(
        structured_row.at(4),
        RingElement::constant(
            (MOD_Q as i64 + (1 - 2) * (1 - 3) * 5) as u64, // 1 0 0
            Representation::IncompleteNTT
        )
    );
    debug_assert_eq!(
        structured_row.at(5),
        RingElement::constant(
            (MOD_Q as i64 + 2 * (1 - 3) * 5) as u64, // 1 0 1
            Representation::IncompleteNTT
        )
    );
    debug_assert_eq!(
        structured_row.at(6),
        RingElement::constant(
            (MOD_Q as i64 + (1 - 2) * 3 * 5) as u64, // 1 1 0
            Representation::IncompleteNTT
        )
    );
    debug_assert_eq!(
        structured_row.at(7),
        RingElement::constant(
            (MOD_Q as i64 + 2 * 3 * 5) as u64, // 1 1 1
            Representation::IncompleteNTT
        )
    );
}

#[test]
fn test_preprocessed_row() {
    use crate::ring::ring_arithmetic::RingElement;

    let tensor_layers = vec![
        RingElement::constant(5, Representation::IncompleteNTT),
        RingElement::constant(3, Representation::IncompleteNTT),
        RingElement::constant(1, Representation::IncompleteNTT),
    ];
    let structured_row = StructuredRow { tensor_layers };

    let structured_row_unused = structured_row.clone();

    let preprocessed_row = PreprocessedRow::from_structured_row(&structured_row_unused);

    for i in 0..preprocessed_row.preprocessed_row.len() {
        debug_assert_eq!(preprocessed_row.preprocessed_row[i], structured_row.at(i));
    }
}

#[test]
fn test_at_matches_preprocessed_row_random() {
    use crate::ring::ring_arithmetic::RingElement;

    // Validate that StructuredRow::at stays consistent with the preprocessing logic
    // for a handful of random evaluation points.
    for _ in 0..5 {
        let tensor_layers = (0..4)
            .map(|_| RingElement::random(Representation::IncompleteNTT))
            .collect::<Vec<_>>();
        let structured_row = StructuredRow {
            tensor_layers: tensor_layers.clone(),
        };
        let preprocessed_row = PreprocessedRow::from_structured_row(&structured_row);

        for idx in 0..preprocessed_row.preprocessed_row.len() {
            let from_at = structured_row.at(idx);
            let from_pre = &preprocessed_row.preprocessed_row[idx];
            debug_assert_eq!(
                &from_at, from_pre,
                "Mismatch at idx {} for layers {:?}",
                idx, tensor_layers
            );
        }
    }
}
