use std::{cell::RefCell, rc::Rc};

use num::range;

use crate::{
    common::{
        decomposition::get_decomposed_offset_scaled,
        matrix::new_vec_zero_preallocated,
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    protocol::{
        commitment::Prefix,
        crs::CRS,
        sumcheck_utils::{
            linear::LinearSumcheck, product::ProductSumcheck, selector_eq::SelectorEq,
        },
    },
};

/// Builds the pair of sumchecks that recompose a base-`2^{base_log}` decomposition.
/// In our protocol, many objects are stored in a digit-decomposed form to keep
/// coefficients small. When we want to prove claims about the recomposed values,
/// we need a gadget that: (1) folds the decomposed digits with the appropriate
/// radix weights, and (2) accounts for the constant offset introduced by the
/// signed-digit representation. The first sumcheck returned here carries the
/// radix weights (`combiner_sumcheck`), while the second holds the constant
/// offset term that is subtracted from the folded witness. Keeping the offset
/// in a dedicated linear sumcheck lets us reuse the same folding machinery for
/// many decompositions without duplicating arithmetic. The variable arity is
/// expanded with prefix padding so these sumchecks can be plugged into larger
/// products without re-indexing the hypercube variables.
pub(crate) fn composition_sumcheck(
    base_log: u64,
    chunks: usize,
    total_vars: usize,
) -> (
    Rc<RefCell<LinearSumcheck<RingElement>>>,
    Rc<RefCell<LinearSumcheck<RingElement>>>,
) {
    let conmposition_basis = range(0, chunks)
        .map(|i| {
            // Basis element corresponding to 2^{base_log * i}
            RingElement::constant(
                1u64 << (base_log as u64 * i as u64),
                Representation::IncompleteNTT,
            )
        })
        .collect::<Vec<RingElement>>();
    let combiner_sumcheck = Rc::new(RefCell::new(
        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
            conmposition_basis.len(),
            total_vars - conmposition_basis.len().ilog2() as usize,
            0,
        ),
    ));

    combiner_sumcheck
        .borrow_mut()
        .load_from(&conmposition_basis);

    let mut witness_combiner_constant_sumcheck = Rc::new(RefCell::new(
        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(1, total_vars, 0),
    ));

    witness_combiner_constant_sumcheck
        .borrow_mut()
        .load_from(&vec![RingElement::all(
            get_decomposed_offset_scaled(base_log as u64, chunks),
            Representation::IncompleteNTT,
        )]);

    (combiner_sumcheck, witness_combiner_constant_sumcheck)
}

/// Creates a selector sumcheck that picks out a specific slice from the hypercube.
/// 
/// This function constructs a selector equality (SelectorEq) sumcheck that evaluates to 1
/// on all points in the hypercube whose first `prefix.length` bits match `prefix.prefix`,
/// and 0 everywhere else. This is a fundamental building block in our protocol because:
/// 
/// 1. **Selective Commitment Enforcement**: We use selectors to enforce that different
///    commitments are correctly formed only on their designated slices of the witness vector.
///    For example, if we have multiple recursive commitment layers at different prefix
///    positions, each layer's CK·witness check only needs to hold on that layer's slice.
/// 
/// 2. **Memory Layout Optimization**: By organizing the combined witness as a flat vector
///    where different objects (folded witness, recursive commitments, opening RHS values, etc.)
///    occupy disjoint prefix regions, we can reuse the same sumcheck machinery across all
///    constraints without copying data or maintaining separate witnesses.
/// 
/// 3. **Verifier Efficiency**: Instead of proving separate sumchecks for each constraint,
///    we multiply each constraint by its selector and sum them all together. The verifier
///    then only needs to fold a single set of challenges across the entire hypercube, which
///    dramatically reduces round complexity.
/// 
/// The `total_vars` parameter ensures the selector hypercube matches the global witness size,
/// so prefix padding is inserted automatically when `prefix.length < total_vars`.
pub(crate) fn sumcheck_from_prefix(
    prefix: &Prefix,
    total_vars: usize,
) -> Rc<RefCell<SelectorEq<RingElement>>> {
    Rc::new(RefCell::new(SelectorEq::<RingElement>::new(
        prefix.prefix,
        prefix.length,
        total_vars,
    )))
}

/// Loads a single row from the commitment key (CK) into a linear sumcheck gadget.
/// 
/// The commitment key is a matrix where each row represents one constraint in the linear
/// commitment scheme. This function packages the i-th row into a sumcheck-friendly format
/// with the correct variable padding. The design choices here reflect several protocol needs:
/// 
/// 1. **Dimension Flexibility**: The `wit_dim` parameter specifies how many elements this
///    CK row operates on. In recursive commitments, inner layers may work on smaller slices
///    of the witness, so we can't hardcode a single dimension.
/// 
/// 2. **Suffix Padding for Decomposition**: The `sufix` parameter reserves trailing variables
///    for the decomposition chunks. When we prove that CK·folded_witness matches a commitment,
///    the folded witness is itself stored in a decomposed form (multiple chunks per entry to
///    keep coefficients small). By padding with `sufix` trailing variables, the hypercube
///    layout aligns decomposition indices with the actual memory layout, avoiding expensive
///    reshuffling during the sumcheck rounds.
/// 
/// 3. **Prefix Padding for Global Context**: The `total_vars - wit_dim.ilog2() - sufix`
///    computation determines how many leading variables to pad. This ensures the CK row
///    sumcheck lives in the same hypercube as all other sumchecks (which operate on the
///    full combined witness), so a single verifier random challenge can fold everything.
/// 
/// 4. **Preprocessing Reuse**: The CK matrix is part of the public CRS and is preprocessed
///    once. By directly loading `ck[i].preprocessed_row`, we avoid recomputing tensor
///    structure or evaluation tables during every sumcheck invocation, which speeds up
///    the prover significantly.
/// 
/// The returned `Rc<RefCell<...>>` allows multiple sumchecks (like different type0 checks
/// for different commitment rows) to share references to the same CK data.
pub(crate) fn ck_sumcheck(
    crs: &CRS,
    total_vars: usize,
    wit_dim: usize,
    i: usize,
    sufix: usize,
) -> Rc<RefCell<LinearSumcheck<RingElement>>> {
    let ck = crs.ck_for_wit_dim(wit_dim);

    let mut sumcheck = Rc::new(RefCell::new(
        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
            wit_dim,
            total_vars - wit_dim.ilog2() as usize - sufix,
            sufix,
        ),
    ));

    sumcheck.borrow_mut().load_from(&ck[i].preprocessed_row);

    sumcheck
}

/// Computes the tensor (Kronecker) product of two vectors of ring elements.
/// 
/// Given vectors a = [a0, a1, ..., a_{m-1}] and b = [b0, b1, ..., b_{n-1}], this function
/// computes their tensor product a ⊗ b, which is the mn-dimensional vector:
///   [a0·b0, a0·b1, ..., a0·b_{n-1}, a1·b0, a1·b1, ..., a_{m-1}·b_{n-1}]
/// 
/// **Why Tensor Products in Sumcheck?**
/// 
/// In our protocol, tensor products arise naturally when we need to prove properties
/// about structured matrices or multi-dimensional data:
/// 
/// 1. **Projection Matrix Structure**: The projection matrix has a block structure where
///    each block is a copy of a smaller projection matrix. To prove that the witness
///    projects correctly, we need to compute (folding_challenges ⊗ projection_flatter),
///    which gives us the effective "selector" for which projected elements contribute
///    to the final folded projection image.
/// 
/// 2. **Hypercube Factorization**: The tensor product corresponds to the multiplication
///    of multilinear extensions over independent sets of variables. When we evaluate
///    MLE(x) ⊗ MLE(y) at point (x, y), it factors as MLE_x(x) · MLE_y(y), which is
///    exactly how the sumcheck verifier combines challenges across different dimensions.
/// 
/// 3. **Fold-Then-Project Commutativity**: By expressing the projection constraint as
///    an inner product with a tensor, we can prove that folding the witness first and
///    then projecting gives the same result as projecting each chunk separately and
///    then folding the images. This is crucial for the recursive structure of the protocol.
/// 
/// The ordering (outer loop over `a`, inner loop over `b`) matches the standard Kronecker
/// product definition and ensures the indices align with how we lay out the hypercube.
pub(crate) fn tensor_product(a: &Vec<RingElement>, b: &Vec<RingElement>) -> Vec<RingElement> {
    let mut result: Vec<RingElement> = new_vec_zero_preallocated(a.len() * b.len());
    let mut idx = 0;
    for a_elem in a.iter() {
        for b_elem in b.iter() {
            result[idx] *= (a_elem, b_elem);
            idx += 1;
        }
    }
    result
}

/// Computes the projection coefficients for proving the projection image consistency.
/// 
/// This is one of the most intricate helper functions in the protocol because it bridges
/// the gap between the structured projection matrix and the flat witness representation
/// that the sumcheck operates on. Here's the full story:
/// 
/// **High-Level Goal:**
/// We need to prove that:
///   (I ⊗ ProjectionMatrix) · folded_witness = projection_image · fold_challenge
/// 
/// where I is a block identity matrix, ProjectionMatrix is a small structured matrix,
/// and the tensor product arranges copies of ProjectionMatrix along the diagonal blocks.
/// 
/// **Why This Constraint Matters:**
/// The projection mechanism is how we compress the witness for efficiency. The prover
/// computes a projected image that's smaller than the original witness, and the verifier
/// needs to check this projection was done correctly. However, we can't just multiply
/// the matrix directly—we need to fold it with a random verifier challenge (fold_challenge)
/// to keep the protocol non-interactive and succinct.
/// 
/// **The Flattening Trick:**
/// Instead of proving the projection row-by-row (which would require many constraints),
/// we sample a random linear combination (projection_flatter) of the rows. This gives us
/// a single inner product constraint:
///   <projection_flatter, (I ⊗ ProjectionMatrix) · folded_witness> 
///      = <projection_flatter, projection_image · fold_challenge>
/// 
/// **Block Structure Exploitation:**
/// The witness is organized into `blocks` many blocks of size `inner_width`, where
/// inner_width = projection_ratio * height. Each block gets multiplied by its own
/// copy of the projection matrix. By splitting `projection_flatter` into:
///   - projection_flatter_0: weights for which block (length = blocks)
///   - projection_flatter_1: weights for positions within each block (length = height)
/// we can compute the effective coefficients for the full witness vector by combining
/// these two layers.
/// 
/// **Sparsity Optimization:**
/// ProjectionMatrix is typically sparse (many entries are zero or ±1), and we expect
/// projection_flatter to have relatively few non-zero entries after the MLE evaluation.
/// By tracking `non_zero_inner_indices`, we avoid iterating over zero contributions,
/// which gives a significant speedup in practice.
/// 
/// **Return Value:**
/// The returned vector has length `witness_height` and contains the effective linear
/// combination weights. Specifically, result[block * inner_width + i] is the coefficient
/// that the i-th element of block `block` contributes to the final inner product. These
/// coefficients are then loaded into a linear sumcheck that gets multiplied with the
/// folded witness to produce the LHS of the projection constraint.
pub(crate) fn projection_coefficients(
    projection_matrix: &ProjectionMatrix,
    projection_flatter: &StructuredRow,
    witness_height: usize,
    projection_ratio: usize,
) -> Vec<RingElement> {
    let height = crate::common::config::PROJECTION_HEIGHT;
    let height_log = height.ilog2() as usize;
    let tensor_layers = &projection_flatter.tensor_layers;
    debug_assert!(tensor_layers.len() >= height_log);
    debug_assert_eq!(1usize << height_log, height);
    let block_layers = tensor_layers.len() - height_log;
    let blocks = witness_height / (projection_ratio * height);
    debug_assert_eq!(blocks, 1usize << block_layers);
    debug_assert_eq!(
        witness_height,
        projection_ratio * (1usize << tensor_layers.len())
    );

    let projection_flatter_0 = PreprocessedRow::from_structured_row(&StructuredRow {
        tensor_layers: tensor_layers[..block_layers].to_vec(),
    });
    let projection_flatter_1 = PreprocessedRow::from_structured_row(&StructuredRow {
        tensor_layers: tensor_layers[block_layers..].to_vec(),
    });

    let inner_width = projection_ratio * height;
    let zero = RingElement::zero(Representation::IncompleteNTT);

    let mut projection_flatter_1_projection = new_vec_zero_preallocated(inner_width);
    for inner_row in 0..height {
        let weight = &projection_flatter_1.preprocessed_row[inner_row];
        if weight == &zero {
            continue;
        }

        for i in 0..inner_width {
            let (is_positive, is_non_zero) = projection_matrix[(inner_row, i)];
            if !is_non_zero {
                continue;
            }
            if is_positive {
                projection_flatter_1_projection[i] += weight;
            } else {
                projection_flatter_1_projection[i] -= weight;
            }
        }
    }

    let non_zero_inner_indices = projection_flatter_1_projection
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| (value != &zero).then_some(idx))
        .collect::<Vec<_>>();

    let mut result = new_vec_zero_preallocated(witness_height);
    for block in 0..blocks {
        let coeff = &projection_flatter_0.preprocessed_row[block];
        if coeff == &zero {
            continue;
        }

        let offset = block * inner_width;
        for &i in non_zero_inner_indices.iter() {
            let mut contribution = projection_flatter_1_projection[i].clone();
            contribution *= coeff;
            result[offset + i] = contribution;
        }
    }

    result
}
