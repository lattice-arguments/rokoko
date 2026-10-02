use crate::common::{
    matrix::VerticallyAlignedMatrix,
    ring_arithmetic::{incomplete_ntt_dot_many_into, Representation, RingElement},
};

pub fn fold(
    witness: &VerticallyAlignedMatrix<RingElement>,
    fold_challenge: &[RingElement],
) -> VerticallyAlignedMatrix<RingElement> {
    let mut folded_witness = VerticallyAlignedMatrix {
        data: vec![RingElement::zero(Representation::IncompleteNTT); witness.height * 1],
        width: 1,
        height: witness.height,
        used_cols: 1,
    };

    debug_assert_eq!(witness.width, fold_challenge.len());

    incomplete_ntt_dot_many_into(
        &mut folded_witness.data,
        fold_challenge,
        1,
        &witness.data,
        witness.height,
        witness.used_cols,
    );
    folded_witness
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fold() {
        let witness = VerticallyAlignedMatrix {
            data: vec![
                RingElement::constant(1, Representation::IncompleteNTT),
                RingElement::constant(2, Representation::IncompleteNTT),
                RingElement::constant(3, Representation::IncompleteNTT),
                RingElement::constant(4, Representation::IncompleteNTT),
            ],
            width: 2,
            height: 2,
            used_cols: 2,
        };

        let fold_challenge = vec![
            RingElement::constant(2, Representation::IncompleteNTT),
            RingElement::constant(3, Representation::IncompleteNTT),
        ];

        let folded_witness = fold(&witness, &fold_challenge);

        debug_assert_eq!(
            folded_witness[(0, 0)],
            RingElement::constant(1 * 2 + 3 * 3, Representation::IncompleteNTT)
        );
        debug_assert_eq!(
            folded_witness[(1, 0)],
            RingElement::constant(2 * 2 + 4 * 3, Representation::IncompleteNTT)
        );
    }
}
