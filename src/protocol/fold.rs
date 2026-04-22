use crate::common::{
    matrix::VerticallyAlignedMatrix,
    parallel::chunk_size_for_par,
    ring_arithmetic::{Representation, RingElement},
};
use crate::par_chunks_mut;

#[cfg(feature = "parallel")]
use crate::common::parallel::*;

pub fn fold(
    witness: &VerticallyAlignedMatrix<RingElement>,
    fold_challenge: &[RingElement],
) -> VerticallyAlignedMatrix<RingElement> {
    let mut folded_witness = VerticallyAlignedMatrix::new_zero_preallocated(witness.height, 1);

    debug_assert_eq!(witness.width, fold_challenge.len());

    let used_cols = witness.used_cols;
    let height = folded_witness.height;

    // Split the output row range into chunks. Each chunk is handled by one
    // worker with its own scratch `temp`. Within a chunk we keep the
    // original col-outer/row-inner order so each column read is a
    // contiguous stride (column-major witness storage). In serial mode the
    // chunk size equals `height`, recovering the original single-loop
    // behavior exactly.
    let chunk_size = chunk_size_for_par(height, 4);
    par_chunks_mut!(folded_witness.data.as_mut_slice(), chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, chunk_out)| {
            let row_start = chunk_idx * chunk_size;
            let mut temp = RingElement::zero(Representation::IncompleteNTT);
            for col in 0..used_cols {
                let challenge = &fold_challenge[col];
                let w_col = witness.col(col);
                for (local_row, out) in chunk_out.iter_mut().enumerate() {
                    let w_el = &w_col[row_start + local_row];
                    temp *= (challenge, w_el);
                    *out += &temp;
                }
            }
        });

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
