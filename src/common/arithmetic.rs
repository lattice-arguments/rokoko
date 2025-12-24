use crate::common::ring_arithmetic::{incomplete_ntt_multiplication, Representation, RingElement};

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
