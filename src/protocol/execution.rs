use std::{process::exit, sync::LazyLock};

use num::range;

use crate::{
    common::{
        hash::HashWrapper,
        matrix::{new_vec_zero_preallocated, HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{Representation, RingElement},
        sampling::sample_random_short_vector,
        structured_row::{self, PreprocessedRow, StructuredRow},
    },
    protocol::{
        commitment::{
            commit_basic, commit_basic_internal, recursive_commit, BasicCommitment,
            RecursionConfig, RecursiveCommitment,
        },
        crs::{CK, CRS},
        fold::fold,
        open::{evaluation_point_to_structured_row, open_at, Opening},
        project::project,
    },
};

pub struct RoundOutput {
    folded_witness: VerticallyAlignedMatrix<RingElement>,
    projection_image: VerticallyAlignedMatrix<RingElement>,
    opening: Opening,
}

pub static CONFIG: LazyLock<Config> = LazyLock::new(|| Config {
    witness_height: 256,
    witness_width: 16,
    challenge_width: 16,  // shall be likely the witness width
    projection_ratio: 16, // shall be likely the witness_height
    commitment_recursion: RecursionConfig {
        decomposition_radix_log: 15,
        decomposition_chunks: 4,
        rank: 1,
        next: None,
    },
    projection_opening_recursion: RecursionConfig {
        decomposition_radix_log: 15,
        decomposition_chunks: 2,
        rank: 1,
        next: None,
    },
});

pub struct Config {
    witness_height: usize,
    witness_width: usize,
    challenge_width: usize,  // shall be likely the witness width
    projection_ratio: usize, // shall be likely the witness_height
    commitment_recursion: RecursionConfig,
    projection_opening_recursion: RecursionConfig,
    // next: Option<Box<Config>>, // for multiple rounds
}

// pub fn init_empty_recursive_commitment(config: &Vec<RecursionConfig>) -> RecursiveCommitment {

// }

pub fn prover_round(
    crs: &CRS,
    commitment: &RecursiveCommitment,
    witness: &VerticallyAlignedMatrix<RingElement>,
    evaluation_points_inner: &Vec<Vec<RingElement>>,
    evaluation_points_outer: &Vec<Vec<RingElement>>,
) -> RoundOutput {
    let mut hash_wrapper = HashWrapper::new();

    hash_wrapper.update_with_ring_element_slice(&commitment.most_inner_commitment());

    let opening = open_at(&witness, &evaluation_points_inner, &evaluation_points_outer);

    let rc_opening = recursive_commit(crs, &CONFIG.projection_opening_recursion, &opening.rhs.data);

    hash_wrapper.update_with_ring_element_slice(&opening.rhs.data);

    let mut projection_matrix = ProjectionMatrix::new(CONFIG.projection_ratio);

    projection_matrix.sample(&mut hash_wrapper);

    let projection_image = project(&witness, &projection_matrix);

    print!("Projection image sampled.\n");
    println!("{:?}", projection_image.height);
    println!("{:?}", projection_image.width);

    let rc_projection_image = recursive_commit(
        &crs,
        &CONFIG.projection_opening_recursion,
        &projection_image.data,
    );

    hash_wrapper.update_with_ring_element_slice(&rc_projection_image.most_inner_commitment());

    let mut fold_challenge = vec![RingElement::zero(Representation::IncompleteNTT); witness.width];

    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut fold_challenge);

    let folded_witness = fold(&witness, &fold_challenge);

    // SUMCHECK
    // we want to check that
    // ck \cdot folded_witness - commitment \cdot fold_challenge = 0
    // outer_evaluation_points \cdot folded_witness - opening \cdot fold_challenge = 0
    // <opening, inner_evaluation_points> - evaluations = 0
    // I \otimes projection_matrix \cdot folded_witness - projection_image \cdot fold_challenge = 0
    // rc_projection_image, rc_opening, rc_commitment are well-formed
    // <w, conj(w)> + <y, conj(y)> - t = 0

    RoundOutput {
        folded_witness,
        projection_image,
        opening,
    }
}

pub fn execute() {
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

    let basic_commitment = commit_basic(&crs, &witness);

    let rc_commitment =
        recursive_commit(&crs, &CONFIG.commitment_recursion, &basic_commitment.data);

    let evaluation_points_inner = vec![range(0, witness.height.ilog2() as usize)
        .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 2))
        .collect::<Vec<RingElement>>()];

    let evaluation_points_outer = vec![range(0, witness.width.ilog2() as usize)
        .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 2))
        .collect::<Vec<RingElement>>()];

    let round_output = prover_round(
        &crs,
        &rc_commitment,
        &witness,
        &evaluation_points_inner,
        &evaluation_points_outer,
    );
}
