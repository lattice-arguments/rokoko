use crate::common::{
    matrix::VerticallyAlignedMatrix,
    ring_arithmetic::{Representation, RingElement},
};

pub fn fold(
    folded_witness: &mut VerticallyAlignedMatrix<RingElement>,
    witness: &VerticallyAlignedMatrix<RingElement>,
    fold_challenge: &[RingElement],
) {
    assert_eq!(witness.width, fold_challenge.len());
    assert_eq!(folded_witness.width, witness.width);
    assert_eq!(folded_witness.height, witness.height);

    for col in 0..witness.width {
        for row in 0..folded_witness.height {
            let w_el = &witness[(row, col)];
            let challenge = &fold_challenge[col];
            folded_witness[(row, 0)] += &(challenge * w_el);
        }
    }
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

    fold(&mut folded_witness, &witness, &fold_challenge);

    assert_eq!(
        folded_witness[(0, 0)],
        RingElement::constant(1 * 2 + 3 * 3, Representation::IncompleteNTT)
    );
    assert_eq!(
        folded_witness[(1, 0)],
        RingElement::constant(2 * 2 + 4 * 3, Representation::IncompleteNTT)
    );
}
