use std::sync::LazyLock;

use crate::common::{
    config::HALF_DEGREE,
    ring_arithmetic::{
        incomplete_ntt_multiplication, QuadraticExtension, Representation, RingElement,
        SHIFT_FACTORS,
    },
    structured_row::StructuredRow,
    sumcheck_element::SumcheckElement,
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

#[test]
fn test_field_to_ring_roundtrip() {
    let fe = QuadraticExtension {
        coeffs: [123456789, 987654321],
        shift: SHIFT_FACTORS[0],
    };
    let re = field_to_ring_element(&fe);
    let fes = re.split_into_quadratic_extensions();
    for f in fes {
        assert_eq!(f, fe);
    }
}
