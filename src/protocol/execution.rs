use std::{process::exit, sync::LazyLock};

use num::range;

use crate::{
    common::{
        arithmetic::inner_product,
        decomposition::{compose_from_decomposed, decompose},
        hash::HashWrapper,
        matrix::{new_vec_zero_preallocated, HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{Representation, RingElement},
        sampling::sample_random_short_vector,
        structured_row::{self, PreprocessedRow, StructuredRow},
    },
    protocol::{
        commitment::{commit_basic, recursive_commit, RecursiveCommitment},
        config::{paste_by_prefix, paste_recursive_commitment, CONFIG},
        crs::{CK, CRS},
        fold::fold,
        open::{claim, evaluation_point_to_structured_row, open_at, Opening},
        prefix::check_prefixing_correctness,
        project::project,
        sumcheck::sumcheck,
        // sumcheck::sumcheck,
    },
};

pub fn prover_round(
    crs: &CRS,
    rc_commitment: &RecursiveCommitment,
    witness: &VerticallyAlignedMatrix<RingElement>,
    evaluation_points_inner: &Vec<StructuredRow>,
    evaluation_points_outer: &Vec<StructuredRow>,
    claims: &Vec<RingElement>,
) {
    let mut hash_wrapper = HashWrapper::new();

    hash_wrapper.update_with_ring_element_slice(&rc_commitment.most_inner_commitment());

    let opening = open_at(&witness, &evaluation_points_inner, &evaluation_points_outer);

    let rc_opening = recursive_commit(crs, &CONFIG.opening_recursion, &opening.rhs.data);

    hash_wrapper.update_with_ring_element_slice(&opening.rhs.data);

    let mut projection_matrix = ProjectionMatrix::new(CONFIG.projection_ratio);

    projection_matrix.sample(&mut hash_wrapper);

    let projection_image = project(&witness, &projection_matrix);

    let rc_projection_image =
        recursive_commit(&crs, &CONFIG.projection_recursion, &projection_image.data);

    hash_wrapper.update_with_ring_element_slice(&rc_projection_image.most_inner_commitment());

    let mut fold_challenge = vec![RingElement::zero(Representation::IncompleteNTT); witness.width];

    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut fold_challenge);

    let folded_witness = fold(&witness, &fold_challenge);

    let mut next_round_data = new_vec_zero_preallocated(CONFIG.composed_witness_length);

    let folded_witness_decomposed = decompose(
        &folded_witness.data,
        CONFIG.witness_decomposition_base_log as u64,
        CONFIG.witness_decomposition_chunks,
    );

    // TODO: can we avoid those copies?
    paste_by_prefix(
        &mut next_round_data,
        &folded_witness_decomposed,
        &CONFIG.folded_witness_prefix,
    );

    paste_recursive_commitment(
        &mut next_round_data,
        &rc_projection_image,
        &CONFIG.projection_recursion,
    );

    paste_recursive_commitment(&mut next_round_data, &rc_opening, &CONFIG.opening_recursion);

    paste_recursive_commitment(
        &mut next_round_data,
        &rc_commitment,
        &CONFIG.commitment_recursion,
    );

    sumcheck(
        crs,
        &CONFIG,
        &next_round_data,
        &projection_matrix,
        &fold_challenge,
        &opening,
        &claims,
        &mut hash_wrapper,
    );
    // RoundOutput {
    //     folded_witness,
    //     projection_image,
    //     opening,
    // }
}

pub fn execute() {
    check_prefixing_correctness(&CONFIG);
    let crs = CRS::gen_crs(CONFIG.witness_height * 2, 2);

    let witness = VerticallyAlignedMatrix {
        height: CONFIG.witness_height,
        width: CONFIG.witness_width,
        data: sample_random_short_vector(
            CONFIG.witness_height * CONFIG.witness_width,
            10,
            Representation::IncompleteNTT,
        ),
    };

    let ck = &crs.ck_for_wit_dim(witness.height);

    let basic_commitment = commit_basic(&crs, &witness, CONFIG.basic_commitment_rank);

    let rc_commitment =
        recursive_commit(&crs, &CONFIG.commitment_recursion, &basic_commitment.data);

    let evaluation_points_inner = vec![evaluation_point_to_structured_row(
        &range(0, witness.height.ilog2() as usize)
            .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 2))
            .collect::<Vec<RingElement>>(),
    )];

    let evaluation_points_outer = vec![evaluation_point_to_structured_row(
        &range(0, witness.width.ilog2() as usize)
            .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 2))
            .collect::<Vec<RingElement>>(),
    )];

    let claims = vec![claim(
        &witness,
        &evaluation_points_inner[0],
        &evaluation_points_outer[0],
    )];
    let round_output = prover_round(
        &crs,
        &rc_commitment,
        &witness,
        &evaluation_points_inner,
        &evaluation_points_outer,
        &claims,
    );
}
