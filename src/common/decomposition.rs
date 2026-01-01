use crate::common::{
    config::MOD_Q,
    matrix::new_vec_zero_preallocated,
    ring_arithmetic::{Representation, RingElement},
};

impl RingElement {
    pub fn bits_into(&mut self, target: &mut RingElement, from: u64, to: u64) {
        assert!(from < to);

        let mask: u64 = (1u64 << (to - from)) - 1;
        for i in 0..self.v.len() {
            target.v[i] = (self.v[i] >> from) & mask;
        }
    }
}

pub fn decompose(input: &mut Vec<RingElement>, base_log: u64, radix: usize) -> Vec<RingElement> {
    let mut decomposed = new_vec_zero_preallocated(input.len() * radix);

    let big_shift = RingElement::all(
        1u64 << (base_log * radix as u64 - 1),
        Representation::EvenOddCoefficients,
    );

    let small_shift = RingElement::all(1u64 << (base_log - 1), Representation::EvenOddCoefficients);

    for (index, el) in input.iter_mut().enumerate() {
        el.to_representation(Representation::EvenOddCoefficients);
        *el += &big_shift;
        for i in 0..radix {
            decomposed[index * radix + i].to_representation(Representation::EvenOddCoefficients);
            el.bits_into(
                &mut decomposed[index * radix + i],
                i as u64 * base_log,
                (i as u64 + 1) * base_log,
            );
            decomposed[index * radix + i] -= &small_shift;
            decomposed[index * radix + i].to_representation(Representation::IncompleteNTT);
        }
        *el -= &big_shift;
        el.to_representation(Representation::IncompleteNTT);
    }

    decomposed
}

#[test]
fn test_decompose() {
    let mut input = vec![RingElement::all(37, Representation::IncompleteNTT)];
    let base_log = 3; // base 8
    let radix = 4;
    let decomposed = decompose(&mut input, base_log, radix);
    assert_eq!(
        input[0],
        RingElement::all(37, Representation::IncompleteNTT)
    );
    assert_eq!(decomposed.len(), radix * 1);
    // 37 is shifted to 37 + (8^4) / 2 = 2085
    // base 8 representation of 2085 = 4 * 8^3 + 0 * 8^2 + 4 * 8^1 + 5 * 8^0
    // so the decomposed elements should be [5, 4, 0, 4]
    // after removing the shift, they should be [5 - 4, 4 - 4, 0 - 4, 4 - 4] = [1, 0, -4, 0]
    assert_eq!(
        decomposed[0],
        RingElement::all(1, Representation::IncompleteNTT)
    );
    assert_eq!(
        decomposed[1],
        RingElement::all(0, Representation::IncompleteNTT)
    );
    assert_eq!(
        decomposed[2],
        RingElement::all(MOD_Q - 4, Representation::IncompleteNTT)
    );
    assert_eq!(
        decomposed[3],
        RingElement::all(0, Representation::IncompleteNTT)
    );
}
