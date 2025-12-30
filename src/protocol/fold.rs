use crate::common::{
    matrix::{VerticallyAlignedMatrix, ZeroNew},
    ring_arithmetic::{Representation, RingElement},
};

pub fn fold(
    witness: &VerticallyAlignedMatrix<RingElement>,
    fold_challenge: &[RingElement],
) -> VerticallyAlignedMatrix<RingElement> {
    let mut folded_witness = VerticallyAlignedMatrix::new_zero_preallocated(witness.height, 1);

    assert_eq!(witness.width, fold_challenge.len());

    for col in 0..witness.width {
        for row in 0..folded_witness.height {
            let w_el = &witness[(row, col)];
            let challenge = &fold_challenge[col];
            folded_witness[(row, 0)] += &(challenge * w_el);
        }
    }
    folded_witness
}

#[test]
fn test_fold() {
    let mut witness = VerticallyAlignedMatrix {
        data: vec![
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(2, Representation::IncompleteNTT),
            RingElement::constant(3, Representation::IncompleteNTT),
            RingElement::constant(4, Representation::IncompleteNTT),
        ],
        width: 2,
        height: 2,
    };

    let fold_challenge = vec![
        RingElement::constant(2, Representation::IncompleteNTT),
        RingElement::constant(3, Representation::IncompleteNTT),
    ];

    let mut folded_witness = VerticallyAlignedMatrix {
        data: vec![RingElement::zero(Representation::IncompleteNTT); 2 * 2],
        width: 2,
        height: 2,
    };

    let folded_witness = fold(&witness, &fold_challenge);

    assert_eq!(
        folded_witness[(0, 0)],
        RingElement::constant(1 * 2 + 3 * 3, Representation::IncompleteNTT)
    );
    assert_eq!(
        folded_witness[(1, 0)],
        RingElement::constant(2 * 2 + 4 * 3, Representation::IncompleteNTT)
    );
}
