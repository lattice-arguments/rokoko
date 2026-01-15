use std::sync::LazyLock;

use crate::{
    common::{
        config::{HALF_DEGREE, MOD_Q},
        ring_arithmetic::{
            incomplete_ntt_multiplication, QuadraticExtension, Representation, RingElement,
            SHIFT_FACTORS,
        },
        structured_row::{PreprocessedRow, StructuredRow},
        sumcheck_element::SumcheckElement,
    },
    hexl::bindings::{multiply_mod, sub_mod},
};

#[inline]
pub fn inner_product(a: &Vec<RingElement>, b: &Vec<RingElement>) -> RingElement {
    assert_eq!(a.len(), b.len());
    let mut result = RingElement::zero(Representation::IncompleteNTT);
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    for (x, y) in a.iter().zip(b.iter()) {
        incomplete_ntt_multiplication(&mut temp, x, y);
        result += &temp;
    }
    result
}

#[inline]
pub fn inner_product_into(mut r: &mut RingElement, a: &Vec<RingElement>, b: &Vec<RingElement>) {
    assert_eq!(a.len(), b.len());
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    for (x, y) in a.iter().zip(b.iter()) {
        incomplete_ntt_multiplication(&mut temp, x, y);
        *r += &temp;
    }
}

#[inline]
pub fn field_to_ring_element(fe: &QuadraticExtension) -> RingElement {
    let mut result = RingElement::zero(Representation::HomogenizedFieldExtensions);
    for i in 0..2 {
        for j in 0..HALF_DEGREE {
            result.v[j + i * HALF_DEGREE] += fe.coeffs[i];
        }
    }
    result
}

#[inline]
pub fn field_to_ring_element_into(mut r: &mut RingElement, fe: &QuadraticExtension) {
    for i in 0..2 {
        for j in 0..HALF_DEGREE {
            r.v[j + i * HALF_DEGREE] += fe.coeffs[i];
        }
    }
    r.representation = Representation::HomogenizedFieldExtensions;
}

pub static ONE: LazyLock<RingElement> =
    LazyLock::new(|| RingElement::one(Representation::IncompleteNTT));

pub static TWO: LazyLock<RingElement> =
    LazyLock::new(|| RingElement::constant(2, Representation::IncompleteNTT));

pub static ZERO: LazyLock<RingElement> =
    LazyLock::new(|| RingElement::zero(Representation::IncompleteNTT));

pub static ONE_QUAD: LazyLock<QuadraticExtension> = LazyLock::new(|| QuadraticExtension {
    coeffs: [1, 0],
});
pub static TWO_QUAD: LazyLock<QuadraticExtension> = LazyLock::new(|| QuadraticExtension {
    coeffs: [2, 0],
});
pub static ZERO_QUAD: LazyLock<QuadraticExtension> = LazyLock::new(|| QuadraticExtension {
    coeffs: [0, 0],
});

#[test]
fn test_field_to_ring_roundtrip() {
    let fe = QuadraticExtension {
        coeffs: [123456789, 987654321],
    };
    let re = field_to_ring_element(&fe);
    let fes = re.split_into_quadratic_extensions();
    for f in fes {
        assert_eq!(f, fe);
    }
}

// this is only for u64
pub fn precompute_structured_values(layers: &[u64]) -> Vec<u64> {
    let size = 1 << layers.len();
    let mut values = vec![1u64; size];

    for (layer_idx, &layer) in layers.iter().rev().enumerate() {
        let layer_complement = unsafe { sub_mod(1, layer, MOD_Q) };

        for i in 0..size {
            if (i >> layer_idx) & 1 == 1 {
                unsafe {
                    values[i] = multiply_mod(values[i], layer, MOD_Q);
                }
            } else {
                unsafe {
                    values[i] = multiply_mod(values[i], layer_complement, MOD_Q);
                }
            }
        }
    }

    values
}

// Vectorized version using eltwise_mult_mod for better performance
pub fn precompute_structured_values_fast(layers: &[u64]) -> Vec<u64> {
    let size = 1 << layers.len();
    // TODO: can we use preallocated pool here? Does it make sense?
    let mut values = vec![1u64; size];

    for (layer_idx, &layer) in layers.iter().rev().enumerate() {
        let layer_complement = unsafe { sub_mod(1, layer, MOD_Q) };
        let chunk_size = 1 << (layer_idx + 1);
        let half_chunk = 1 << layer_idx;

        // Process in chunks where bit pattern is uniform
        for chunk_start in (0..size).step_by(chunk_size) {
            // First half of chunk (bit layer_idx = 0): multiply by layer_complement
            let start_0 = chunk_start;
            let end_0 = chunk_start + half_chunk;

            // Second half of chunk (bit layer_idx = 1): multiply by layer
            let start_1 = chunk_start + half_chunk;
            let end_1 = chunk_start + chunk_size;

            // Multiply in-place by scalar
            for i in start_0..end_0 {
                unsafe {
                    // TODO: use vectorisation
                    values[i] = multiply_mod(values[i], layer_complement, MOD_Q);
                }
            }

            for i in start_1..end_1 {
                unsafe {
                    values[i] = multiply_mod(values[i], layer, MOD_Q);
                }
            }
        }
    }

    values
}

#[test]
fn test_precompute_structured_values() {
    use crate::common::hash::HashWrapper;

    // Test with different layer sizes
    for num_layers in 1..=10 {
        let mut hash = HashWrapper::new();
        let layers: Vec<u64> = (0..num_layers).map(|_| hash.sample_u64_mod_q()).collect();

        let result_slow = precompute_structured_values(&layers);
        let result_fast = precompute_structured_values_fast(&layers);

        assert_eq!(
            result_slow.len(),
            result_fast.len(),
            "Length mismatch for {} layers",
            num_layers
        );

        for (i, (slow, fast)) in result_slow.iter().zip(result_fast.iter()).enumerate() {
            assert_eq!(
                slow, fast,
                "Mismatch at index {} for {} layers: slow={}, fast={}",
                i, num_layers, slow, fast
            );
        }
    }
}

#[test]
fn test_precompute_structured_values_properties() {
    use crate::common::hash::HashWrapper;

    let mut hash = HashWrapper::new();
    let layers: Vec<u64> = (0..5).map(|_| hash.sample_u64_mod_q()).collect();
    let values = precompute_structured_values_fast(&layers);

    // Size should be 2^k for k layers
    assert_eq!(values.len(), 1 << layers.len());

    // Test specific properties: values[i] should match the tensor product computation
    // For index i with binary representation b_k...b_1b_0:
    // values[i] = product of (layer[j] if b_j=1, else (1-layer[j]))

    let manual_compute = |index: usize| -> u64 {
        let mut result = 1u64;
        for (bit_pos, &layer) in layers.iter().rev().enumerate() {
            if (index >> bit_pos) & 1 == 1 {
                unsafe {
                    result = multiply_mod(result, layer, MOD_Q);
                }
            } else {
                unsafe {
                    result = multiply_mod(result, sub_mod(1, layer, MOD_Q), MOD_Q);
                }
            }
        }
        result
    };

    for i in 0..values.len() {
        assert_eq!(
            values[i],
            manual_compute(i),
            "Value mismatch at index {} (binary: {:05b})",
            i,
            i
        );
    }
}

#[test]
fn test_precompute_structured_values_mathces_preprocessed_row() {
    let layers = vec![2u64, 3u64, 5u64];
    let layers_ring = layers
        .iter()
        .map(|&l| RingElement::constant(l, Representation::IncompleteNTT))
        .collect::<Vec<RingElement>>();

    let structure_row = StructuredRow {
        tensor_layers: layers_ring,
    };
    let preprocessed_row = PreprocessedRow::from_structured_row(&structure_row);

    let precomputed_values = precompute_structured_values_fast(&layers);
    let precomputed_values_ring = precomputed_values
        .iter()
        .map(|&v| RingElement::constant(v, Representation::IncompleteNTT))
        .collect::<Vec<RingElement>>();

    assert_eq!(
        preprocessed_row.preprocessed_row.len(),
        precomputed_values_ring.len()
    );
    for i in 0..preprocessed_row.preprocessed_row.len() {
        assert_eq!(
            preprocessed_row.preprocessed_row[i],
            precomputed_values_ring[i],
        );
    }
}

pub static ADDITION_SUBTRACTION_BUDGET: LazyLock<u64> = LazyLock::new(|| u64::MAX / (MOD_Q * 2)); // if we start from number HALF_WAY_MOD_Q how many additions/subtractions (with elements in [0,MOD_Q)) can we do without overflowing u64?

pub static HALF_WAY_MOD_Q: LazyLock<u64> = LazyLock::new(|| {
    let budget = u64::MAX / (MOD_Q * 2);
    budget * MOD_Q
});

pub static HALF_WAY_MOD_Q_RING_CF: LazyLock<RingElement> =
    LazyLock::new(|| RingElement::all(*HALF_WAY_MOD_Q, Representation::Coefficients));
