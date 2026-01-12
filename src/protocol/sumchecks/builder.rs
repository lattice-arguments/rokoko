use std::{
    cell::{Ref, RefCell},
    rc::Rc,
};

use crate::{
    common::ring_arithmetic::RingElement,
    protocol::{
        commitment::{self, Prefix},
        config::Config,
        crs::{self, CRS},
        sumcheck_utils::{
            combiner::Combiner, common::HighOrderSumcheckData, diff::DiffSumcheck,
            linear::LinearSumcheck, product::ProductSumcheck,
            ring_to_field_combiner::RingToFieldCombiner,
        },
        sumchecks::context::Type5SumcheckContext,
    },
};

use super::{
    context::{
        SumcheckContext, Type0SumcheckContext, Type1SumcheckContext, Type2SumcheckContext,
        Type3SumcheckContext, Type4LayerSumcheckContext, Type4OutputLayerSumcheckContext,
        Type4SumcheckContext,
    },
    helpers::{ck_sumcheck, composition_sumcheck, sumcheck_from_prefix},
};

/// Constructs the recursive sumcheck structure for verifying commitment well-formedness.
///
/// This function builds the sumcheck gadgets needed to prove that a recursive commitment
/// is correctly constructed—that is, each layer is properly decomposed from its parent.
/// The construction follows a tree-like structure where:
///
/// **Recursive Commitment Overview:**
/// In our protocol, commitments are organized hierarchically to keep proof sizes manageable.
/// Instead of committing to the full witness at once (which would require a huge CK matrix),
/// we decompose the witness into chunks, commit to each chunk separately, and then recursively
/// commit to those commitments. This gives us a tree where:
///   - Leaves: chunks of the original witness
///   - Internal nodes: commitments to their children's commitments
///   - Root: the final public commitment
///
/// **What This Function Proves:**
/// For each internal layer i in the recursion tree, we need to prove:
///   CK_i · selected_witness_slice_i = compose(child_commitment_{i+1})
///
/// where:
///   - CK_i is the commitment key for layer i (with dimension matching that layer's slice size)
///   - selected_witness_slice_i is the portion of the combined witness that holds layer i's data
///   - compose(...) reconstructs the parent value from decomposed child chunks
///   - child_commitment_{i+1} is the commitment at the next layer down
///
/// **Layer-by-Layer Construction:**
/// The function walks through the recursion config from root to leaves, building a
/// `Type4LayerSumcheckContext` for each non-leaf layer. Each layer context contains:
///
/// 1. **Selectors**: Two selectors that identify which slice of the combined witness belongs
///    to the current layer and which belongs to the child layer. These ensure we're checking
///    the right portions of the witness without interference from unrelated data.
///
/// 2. **Combiner Sumchecks**: A pair (combiner, constant) that reconstructs the composed
///    value from decomposed chunks. The combiner holds the radix weights (1, base, base²,
///    ...) and the constant holds the signed-digit offset that needs to be subtracted.
///
/// 3. **CK Sumchecks**: One sumcheck per rank, each loaded with a different row of the
///    commitment key. The rank determines how many constraints we're batching together
///    (higher rank = more security but larger proofs).
///
/// 4. **Output Constraints**: For each CK row, we create a DiffSumcheck that enforces:
///       selector_i · (CK_row_i · witness_i) = selector_{i+1} · compose(witness_{i+1})
///    This is the core recursive constraint. The selectors ensure both sides are only
///    evaluated on their respective slices, and the compose operation on the RHS mirrors
///    how the child commitment is stored in decomposed form.
///
/// **Why Stop at Non-Leaf Layers:**
/// The function intentionally does NOT build a sumcheck for the leaf layer (the layer with
/// `current.next == None`). This is because the leaf layer's commitment is assumed to be
/// public and trusted as input to the protocol. We're only proving that the recursive
/// structure above it is correct, not that the leaf itself satisfies any constraint. This
/// design choice reduces proof size and verification time.
///
/// **Shared References and RefCell:**
/// All the sumcheck gadgets are wrapped in `Rc<RefCell<...>>` because:
///   - Multiple outputs may share the same underlying data (e.g., all rows use the same
///     selector sumcheck)
///   - The sumcheck protocol mutates the gadgets as it folds with verifier challenges
///   - We want to avoid copying large preprocessed structures
///
/// The RefCell provides interior mutability so we can call `partial_evaluate` on shared
/// references during the sumcheck rounds.
///
/// **Returned Structure:**
/// The function returns a `Type4SumcheckContext` containing:
///   - `layers`: A vector of `Type4LayerSumcheckContext`, one per internal node level
///
/// This nested structure mirrors the tree topology and allows the prover/verifier to fold
/// constraints level-by-level in a systematic way.
fn build_type4_sumcheck_context(
    crs: &CRS,
    total_vars: usize,
    combined_witness_sumcheck: Rc<RefCell<LinearSumcheck<RingElement>>>,
    config: &commitment::RecursionConfig,
) -> Type4SumcheckContext {
    let mut layers = Vec::new();
    let mut current = config;
    while let Some(next) = current.next.as_deref() {
        let selector_sumcheck = sumcheck_from_prefix(&current.prefix, total_vars);
        let child_selector_sumcheck = sumcheck_from_prefix(&next.prefix, total_vars);

        let data_len = 1 << (total_vars - current.prefix.length);

        let data_selected_sumcheck = Rc::new(RefCell::new(ProductSumcheck::new(
            selector_sumcheck.clone(),
            combined_witness_sumcheck.clone(),
        )));

        let (combiner_sumcheck, combiner_constant_sumcheck) = composition_sumcheck(
            next.decomposition_base_log as u64,
            next.decomposition_chunks,
            total_vars,
        );

        let recomposed_child_raw = Rc::new(RefCell::new(DiffSumcheck::new(
            Rc::new(RefCell::new(ProductSumcheck::new(
                combined_witness_sumcheck.clone(),
                combiner_sumcheck.clone(),
            ))),
            combiner_constant_sumcheck.clone(),
        )));

        let recomposed_child_sumcheck = Rc::new(RefCell::new(ProductSumcheck::new(
            child_selector_sumcheck.clone(),
            recomposed_child_raw,
        )));

        let mut ck_sumchecks = Vec::with_capacity(current.rank);
        for i in 0..current.rank {
            ck_sumchecks.push(ck_sumcheck(crs, total_vars, data_len, i, 0));
        }

        let outputs = ck_sumchecks
            .iter()
            .map(|ck_row| {
                let lhs = Rc::new(RefCell::new(ProductSumcheck::new(
                    ck_row.clone(),
                    data_selected_sumcheck.clone(),
                )));
                let rhs = recomposed_child_sumcheck.clone();
                Rc::new(RefCell::new(DiffSumcheck::new(lhs, rhs)))
            })
            .collect::<Vec<_>>();

        layers.push(Type4LayerSumcheckContext {
            selector_sumcheck,
            child_selector_sumcheck: Some(child_selector_sumcheck),
            combiner_sumcheck: Some(combiner_sumcheck),
            combiner_constant_sumcheck: Some(combiner_constant_sumcheck),
            data_selected_sumcheck,
            rhs_sumcheck: recomposed_child_sumcheck,
            commitment_sumcheck: None,
            ck_sumchecks,
            outputs,
        });

        current = next;
    }

    // Build the output (leaf) layer
    // This is the base case that checks against the public commitment value
    let selector_sumcheck = sumcheck_from_prefix(&current.prefix, total_vars);

    let mut ck_sumchecks = Vec::with_capacity(current.rank);
    for i in 0..current.rank {
        ck_sumchecks.push(ck_sumcheck(
            crs,
            total_vars,
            1 << (total_vars - current.prefix.length),
            i,
            0,
        ));
    }

    let outputs = ck_sumchecks
        .iter()
        .map(|ck_row| {
            let output = Rc::new(RefCell::new(ProductSumcheck::new(
                selector_sumcheck.clone(),
                Rc::new(RefCell::new(ProductSumcheck::new(
                    combined_witness_sumcheck.clone(),
                    ck_row.clone(),
                ))),
            )));
            output
        })
        .collect::<Vec<_>>();

    Type4SumcheckContext {
        layers,
        output_layer: Type4OutputLayerSumcheckContext {
            selector_sumcheck,
            ck_sumchecks,
            outputs,
        },
    }
}

/// Constructs all sumcheck gadgets used across the protocol for a single round.
/// The function wires every semantic constraint into a dedicated sumcheck:
///   - commitment key rows against the folded witness (type0)
///   - inner and outer evaluation consistency for openings (type1/type2)
///   - projection image consistency (type3)
///   - recursive commitment well-formedness for every recursion tree (type4)
///   - inner product of combined witness with its conjugate for norm checking (type5)
/// It also prepares the folding combiners/constants so that later folds only
/// require calling `partial_evaluate_all`. The assembled context is reused for
/// both prover-side simulation (the asserts in `sumcheck`) and as the live
/// state during interactive folding. Prefix padding is chosen so every helper
/// sumcheck can be embedded into larger products without reindexing, and the
/// decomposition offsets are preloaded so the recomposition gadgets mirror the
/// commitment arithmetic exactly. When the folding schedule changes, this is the
/// single place to update the plumbing.
pub fn init_sumcheck(crs: &crs::CRS, config: &Config) -> SumcheckContext {
    let total_vars = config.composed_witness_length.ilog2() as usize;

    let mut combined_witness_sumcheck = Rc::new(RefCell::new(LinearSumcheck::<RingElement>::new(
        config.composed_witness_length,
    )));

    let folded_witness_selector_sumcheck =
        sumcheck_from_prefix(&config.folded_witness_prefix, total_vars);

    let commitment_key_rows_sumcheck = (0..config.basic_commitment_rank)
        .map(|i| {
            ck_sumcheck(
                crs,
                total_vars,
                config.witness_height,
                i,
                config.witness_decomposition_chunks.ilog2() as usize,
            )
        })
        .collect::<Vec<Rc<RefCell<LinearSumcheck<RingElement>>>>>();

    let (mut folded_witness_combiner_sumcheck, mut witness_combiner_constant_sumcheck) =
        composition_sumcheck(
            config.witness_decomposition_base_log as u64,
            config.witness_decomposition_chunks,
            config.composed_witness_length.ilog2() as usize,
        );

    let (mut basic_commitment_combiner_sumcheck, mut basic_commitment_combiner_constant_sumcheck) =
        composition_sumcheck(
            config.commitment_recursion.decomposition_base_log as u64,
            config.commitment_recursion.decomposition_chunks,
            config.composed_witness_length.ilog2() as usize,
        );

    let (opening_combiner_sumcheck, opening_combiner_constant_sumcheck) = composition_sumcheck(
        config.opening_recursion.decomposition_base_log as u64,
        config.opening_recursion.decomposition_chunks,
        config.composed_witness_length.ilog2() as usize,
    );
    let (projection_combiner_sumcheck, projection_combiner_constant_sumcheck) =
        composition_sumcheck(
            config.projection_recursion.decomposition_base_log as u64,
            config.projection_recursion.decomposition_chunks,
            config.composed_witness_length.ilog2() as usize,
        );

    let folding_challenges_sumcheck = Rc::new(RefCell::new(
        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
            config.witness_width,
            config.composed_witness_length.ilog2() as usize
                - config.witness_width.ilog2() as usize
                - config.commitment_recursion.decomposition_chunks.ilog2() as usize,
            config.commitment_recursion.decomposition_chunks.ilog2() as usize,
        ),
    ));

    // Type0 sumchecks
    // CK \cdot folded_witness - commitment \cdot fold_challenge = 0
    let type0sumchecks = (0..config.basic_commitment_rank)
        .map(|i| {
            let basic_commitment_row_sumcheck = sumcheck_from_prefix(
                &Prefix {
                    prefix: config.commitment_recursion.prefix.prefix
                        * config.basic_commitment_rank
                        + i,
                    length: config.commitment_recursion.prefix.length
                        + config.basic_commitment_rank.ilog2() as usize,
                },
                total_vars,
            );

            let ctxt = Type0SumcheckContext {
                basic_commitment_row_sumcheck: basic_commitment_row_sumcheck.clone(),
                output: Rc::new(RefCell::new(DiffSumcheck::new(
                    Rc::new(RefCell::new(ProductSumcheck::new(
                        folded_witness_selector_sumcheck.clone(),
                        Rc::new(RefCell::new(ProductSumcheck::new(
                            Rc::new(RefCell::new(DiffSumcheck::new(
                                Rc::new(RefCell::new(ProductSumcheck::new(
                                    combined_witness_sumcheck.clone(),
                                    folded_witness_combiner_sumcheck.clone(),
                                ))),
                                witness_combiner_constant_sumcheck.clone(),
                            ))),
                            commitment_key_rows_sumcheck[i].clone(),
                        ))),
                    ))),
                    Rc::new(RefCell::new(ProductSumcheck::new(
                        basic_commitment_row_sumcheck,
                        Rc::new(RefCell::new(ProductSumcheck::new(
                            Rc::new(RefCell::new(DiffSumcheck::new(
                                Rc::new(RefCell::new(ProductSumcheck::new(
                                    combined_witness_sumcheck.clone(),
                                    basic_commitment_combiner_sumcheck.clone(),
                                ))),
                                basic_commitment_combiner_constant_sumcheck.clone(),
                            ))),
                            folding_challenges_sumcheck.clone(),
                        ))),
                    ))),
                ))),
            };
            ctxt
        })
        .collect::<Vec<Type0SumcheckContext>>();

    // Type1 sumchecks
    // inner_evaluation_points \cdot folded_witness - opening.rhs \cdot fold_challenge = 0

    let recomposed_folded_witness = Rc::new(RefCell::new(DiffSumcheck::new(
        Rc::new(RefCell::new(ProductSumcheck::new(
            combined_witness_sumcheck.clone(),
            folded_witness_combiner_sumcheck.clone(),
        ))),
        witness_combiner_constant_sumcheck.clone(),
    )));

    let recomposed_opening = Rc::new(RefCell::new(DiffSumcheck::new(
        Rc::new(RefCell::new(ProductSumcheck::new(
            combined_witness_sumcheck.clone(),
            opening_combiner_sumcheck.clone(),
        ))),
        opening_combiner_constant_sumcheck.clone(),
    )));

    let type1sumchecks = (0..config.nof_openings)
        .map(|i| {
            let opening_selector_sumcheck = sumcheck_from_prefix(
                &Prefix {
                    prefix: config.opening_recursion.prefix.prefix * config.nof_openings + i,
                    length: config.opening_recursion.prefix.length
                        + config.nof_openings.ilog2() as usize,
                },
                total_vars,
            );

            let inner_evaluation_sumcheck = Rc::new(RefCell::new(
                LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                    config.witness_height,
                    total_vars
                        - config.witness_height.ilog2() as usize
                        - config.witness_decomposition_chunks.ilog2() as usize,
                    config.witness_decomposition_chunks.ilog2() as usize,
                ),
            ));

            let lhs = Rc::new(RefCell::new(ProductSumcheck::new(
                folded_witness_selector_sumcheck.clone(),
                Rc::new(RefCell::new(ProductSumcheck::new(
                    recomposed_folded_witness.clone(),
                    inner_evaluation_sumcheck.clone(),
                ))),
            )));

            let rhs = Rc::new(RefCell::new(ProductSumcheck::new(
                opening_selector_sumcheck.clone(),
                Rc::new(RefCell::new(ProductSumcheck::new(
                    recomposed_opening.clone(),
                    folding_challenges_sumcheck.clone(),
                ))),
            )));

            let output = Rc::new(RefCell::new(DiffSumcheck::new(lhs, rhs)));

            Type1SumcheckContext {
                inner_evaluation_sumcheck,
                opening_selector_sumcheck,
                output,
            }
        })
        .collect::<Vec<Type1SumcheckContext>>();

    // Type2 sumchecks
    // <opening.rhs[i], outer_evaluation_points> = evaluations[i] (public)
    let type2sumchecks = type1sumchecks
        .iter()
        .map(|type1_sc| {
            let outer_evaluation_sumcheck = Rc::new(RefCell::new(
                LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                    config.witness_width,
                    total_vars
                        - config.witness_width.ilog2() as usize
                        - config.opening_recursion.decomposition_chunks.ilog2() as usize,
                    config.opening_recursion.decomposition_chunks.ilog2() as usize,
                ),
            ));

            let output = Rc::new(RefCell::new(ProductSumcheck::new(
                type1_sc.opening_selector_sumcheck.clone(),
                Rc::new(RefCell::new(ProductSumcheck::new(
                    recomposed_opening.clone(),
                    outer_evaluation_sumcheck.clone(),
                ))),
            )));

            Type2SumcheckContext {
                outer_evaluation_sumcheck,
                output,
            }
        })
        .collect::<Vec<Type2SumcheckContext>>();

    // type3 sumchecks
    // projection_matrix_flatter \cdot (I \otimes projection_matrix) \cdot folded_witness - projection_matrix_flatter \cdot projection_image \cdot fold_challenge = 0
    let recomposed_projection = Rc::new(RefCell::new(DiffSumcheck::new(
        Rc::new(RefCell::new(ProductSumcheck::new(
            combined_witness_sumcheck.clone(),
            projection_combiner_sumcheck.clone(),
        ))),
        projection_combiner_constant_sumcheck.clone(),
    )));

    let projection_height_flat = config.witness_height / config.projection_ratio;
    let type3sumcheck = {
        let projection_selector_sumcheck =
            sumcheck_from_prefix(&config.projection_recursion.prefix, total_vars);

        let projection_coeff_sumcheck = Rc::new(RefCell::new(
            LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                config.witness_height,
                total_vars
                    - config.witness_height.ilog2() as usize
                    - config.witness_decomposition_chunks.ilog2() as usize,
                config.witness_decomposition_chunks.ilog2() as usize,
            ),
        ));

        let fold_tensor_sumcheck = Rc::new(RefCell::new(
            LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
                projection_height_flat * config.witness_width,
                total_vars
                    - (projection_height_flat * config.witness_width).ilog2() as usize
                    - config.projection_recursion.decomposition_chunks.ilog2() as usize,
                config.projection_recursion.decomposition_chunks.ilog2() as usize,
            ),
        ));

        let lhs = Rc::new(RefCell::new(ProductSumcheck::new(
            folded_witness_selector_sumcheck.clone(),
            Rc::new(RefCell::new(ProductSumcheck::new(
                recomposed_folded_witness.clone(),
                projection_coeff_sumcheck.clone(),
            ))),
        )));
        let rhs = Rc::new(RefCell::new(ProductSumcheck::new(
            projection_selector_sumcheck.clone(),
            Rc::new(RefCell::new(ProductSumcheck::new(
                recomposed_projection.clone(),
                fold_tensor_sumcheck.clone(),
            ))),
        )));
        let output = Rc::new(RefCell::new(DiffSumcheck::new(lhs, rhs)));

        Type3SumcheckContext {
            lhs_sumcheck: projection_coeff_sumcheck,
            rhs_sumcheck: fold_tensor_sumcheck,
            projection_selector_sumcheck,
            output,
        }
    };

    let conjugated_combined_witness_sumcheck = Rc::new(RefCell::new(
        LinearSumcheck::<RingElement>::new(config.composed_witness_length),
    ));

    let type5sumcheck = Type5SumcheckContext {
        conjugated_combined_witness: conjugated_combined_witness_sumcheck.clone(),
        output: Rc::new(RefCell::new(ProductSumcheck::new(
            combined_witness_sumcheck.clone(),
            conjugated_combined_witness_sumcheck.clone(),
        ))),
    };

    // Type4 sumchecks: Three separate recursive commitment trees
    // 1. Commitment recursion: verifies the basic witness commitments are well-formed
    // 2. Opening recursion: verifies the opening proofs are correctly committed
    // 3. Projection recursion: verifies the projection images are correctly committed
    // Each tree has its own depth, rank, and decomposition parameters defined in config.
    let type4sumchecks = [
        build_type4_sumcheck_context(
            crs,
            total_vars,
            combined_witness_sumcheck.clone(),
            &config.commitment_recursion,
        ),
        build_type4_sumcheck_context(
            crs,
            total_vars,
            combined_witness_sumcheck.clone(),
            &config.opening_recursion,
        ),
        build_type4_sumcheck_context(
            crs,
            total_vars,
            combined_witness_sumcheck.clone(),
            &config.projection_recursion,
        ),
    ];

    let mut all_outputs: Vec<Rc<RefCell<dyn HighOrderSumcheckData<Element = RingElement>>>> =
        vec![];
    for type0 in &type0sumchecks {
        all_outputs.push(type0.output.clone());
    }
    for type1 in &type1sumchecks {
        all_outputs.push(type1.output.clone());
    }
    for type2 in &type2sumchecks {
        all_outputs.push(type2.output.clone());
    }
    all_outputs.push(type3sumcheck.output.clone());

    for type4 in &type4sumchecks {
        for layer in &type4.layers {
            for output in &layer.outputs {
                all_outputs.push(output.clone());
            }
        }
        for output in &type4.output_layer.outputs {
            all_outputs.push(output.clone());
        }
    }

    all_outputs.push(type5sumcheck.output.clone());

    // TODO: do something smart here
    let combiner = Rc::new(RefCell::new(Combiner::new(all_outputs)));

    let field_combiner = Rc::new(RefCell::new(RingToFieldCombiner::new(combiner.clone())));

    SumcheckContext {
        combined_witness_sumcheck: combined_witness_sumcheck.clone(),
        folded_witness_selector_sumcheck,
        folded_witness_combiner_sumcheck,
        witness_combiner_constant_sumcheck,
        folding_challenges_sumcheck,
        basic_commitment_combiner_sumcheck,
        basic_commitment_combiner_constant_sumcheck,
        commitment_key_rows_sumcheck,
        opening_combiner_sumcheck,
        opening_combiner_constant_sumcheck,
        projection_combiner_sumcheck,
        projection_combiner_constant_sumcheck,
        type0sumchecks,
        type1sumchecks,
        type2sumchecks,
        type3sumcheck,
        type4sumchecks,
        type5sumcheck,
        combiner,
        field_combiner,
    }
}
