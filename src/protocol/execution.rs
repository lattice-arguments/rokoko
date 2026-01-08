use std::process::exit;

use num::range;

use crate::{
    common::{
        hash::HashWrapper,
        matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix, new_vec_zero_preallocated},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{Representation, RingElement},
        sampling::sample_random_short_vector,
        structured_row::{self, PreprocessedRow, StructuredRow},
    },
    protocol::{
        commitment::{Commitment, commit, commit_basic},
        crs::{CK, CRS},
        fold::fold,
        open::{Opening, evaluation_point_to_structured_row, open_at},
        project::project,
    },
};

pub struct RoundOutput {
    folded_witness: VerticallyAlignedMatrix<RingElement>,
    projection_image: VerticallyAlignedMatrix<RingElement>,
    opening: Opening,
}

pub fn prover_simple_round(
    commitment: &Commitment,
    witness: &VerticallyAlignedMatrix<RingElement>,
    evaluation_points_inner: &Vec<Vec<RingElement>>,
    evaluation_points_outer: &Vec<Vec<RingElement>>,
) -> RoundOutput {
    let mut hash_wrapper = HashWrapper::new();

    hash_wrapper.update_with_ring_element_slice(&commitment.commitment.data);

    let opening = open_at(&witness, &evaluation_points_inner, &evaluation_points_outer);

    hash_wrapper.update_with_ring_element_slice(&opening.rhs.commitment.data);

    let mut projection_matrix = ProjectionMatrix::new(8);

    projection_matrix.sample(&mut hash_wrapper);

    let projection_image = project(&witness, &projection_matrix);

    hash_wrapper.update_with_ring_element_slice(&projection_image.data);

    let mut fold_challenge = vec![RingElement::zero(Representation::IncompleteNTT); witness.width];

    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut fold_challenge);

    let folded_witness = fold(&witness, &fold_challenge);

    RoundOutput {
        folded_witness,
        projection_image,
        opening,
    }
}

pub fn verifier_simple_round(
    ck: &CK,
    commitment: &Commitment,
    round_output: &RoundOutput,
    evaluation_points_inner: &Vec<Vec<RingElement>>,
    evaluation_points_outer: &Vec<Vec<RingElement>>,
) {
    // We check if:
    // folded commitment == commit(folded witness)
    // projection_image is correct projection of witness
    // opening is valid opening of witness at evaluation points

    let mut hash_wrapper = HashWrapper::new();
    hash_wrapper.update_with_ring_element_slice(&commitment.commitment.data);
    hash_wrapper.update_with_ring_element_slice(&round_output.opening.rhs.commitment.data);

    let mut projection_matrix = ProjectionMatrix::new(8);
    projection_matrix.sample(&mut hash_wrapper);
    hash_wrapper.update_with_ring_element_slice(&round_output.projection_image.data);
    let mut fold_challenge =
        vec![RingElement::zero(Representation::IncompleteNTT); commitment.commitment.width];
    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut fold_challenge);

    let commitment_of_folded_witness = commit(ck, &round_output.folded_witness, &[]);
    let mut folded_commitment = VerticallyAlignedMatrix::new_zero_preallocated(ck.len(), 1);

    for i in 0..ck.len() {
        for col in 0..commitment.commitment.width {
            let mut temp = RingElement::zero(Representation::IncompleteNTT);
            temp *= (&commitment.commitment[(i, col)], &fold_challenge[col]);
            folded_commitment[(i, 0)] += &temp;
        }
    }

    assert_eq!(commitment_of_folded_witness.commitment, folded_commitment);

    let opening_to_folded_witness = open_at(
        &round_output.folded_witness,
        evaluation_points_inner,
        &vec![],
    );

    let mut folded_opening = VerticallyAlignedMatrix::new_zero_preallocated(
        round_output.opening.rhs.commitment.height,
        1,
    );

    for i in 0..round_output.opening.rhs.commitment.height {
        let mut temp = RingElement::zero(Representation::IncompleteNTT);
        for col in 0..commitment.commitment.width {
            temp *= (
                &round_output.opening.rhs.commitment[(i, col)],
                &fold_challenge[col],
            );
            folded_opening[(i, 0)] += &temp;
        }
    }

    assert_eq!(opening_to_folded_witness.rhs.commitment, folded_opening);

    let projection_of_folded_witness = project(&round_output.folded_witness, &projection_matrix);

    let mut folded_projection_image =
        VerticallyAlignedMatrix::new_zero_preallocated(round_output.projection_image.height, 1);

    for i in 0..folded_projection_image.height {
        let mut temp = RingElement::zero(Representation::IncompleteNTT);
        for col in 0..round_output.projection_image.width {
            temp *= (
                &round_output.projection_image[(i, col)],
                &fold_challenge[col],
            );
            folded_projection_image[(i, 0)] += &temp;
        }
    }

    assert_eq!(projection_of_folded_witness, folded_projection_image);

    let mut evaluation = new_vec_zero_preallocated(round_output.opening.evaluations.len());

    for (i, preprocessed_row_outer) in round_output
        .opening
        .evaluation_points_outer
        .iter()
        .enumerate()
    {
        let mut temp = RingElement::zero(Representation::IncompleteNTT);
        for col in 0..round_output.opening.rhs.commitment.width {
            temp *= (
                &round_output.opening.rhs.commitment[(i, col)],
                &preprocessed_row_outer.preprocessed_row[col],
            );
            evaluation[i] += &temp;
        }
    }

    assert_eq!(round_output.opening.evaluations, evaluation);
}

pub fn execute() {
    let crs = CRS::gen_crs(256, 2);

    let witness = VerticallyAlignedMatrix {
        height: 256,
        width: 16,
        data: sample_random_short_vector(256 * 16, 10, Representation::IncompleteNTT),
    };

    let ck = &crs.ck_for_wit_dim(witness.height);

    let commitment = commit_basic(&crs, &witness);

    let evaluation_points_inner = vec![range(0, witness.height.ilog2() as usize)
        .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 2))
        .collect::<Vec<RingElement>>()];

    let evaluation_points_outer = vec![range(0, witness.width.ilog2() as usize)
        .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 2))
        .collect::<Vec<RingElement>>()];

    let round_output = prover_simple_round(
        &commitment,
        &witness,
        &evaluation_points_inner,
        &evaluation_points_outer,
    );

    verifier_simple_round(
        &ck,
        &commitment,
        &round_output,
        &evaluation_points_inner,
        &evaluation_points_outer,
    );
}
