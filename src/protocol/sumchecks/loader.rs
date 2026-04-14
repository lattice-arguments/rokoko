use crate::protocol::parties::prover::vdf_crs;
use crate::{
    common::{
        arithmetic::field_to_ring_element_into,
        config::*,
        matrix::new_vec_zero_preallocated,
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    protocol::{
        project::BatchingChallenges,
        project_2::BatchedProjectionChallenges,
        sumchecks::{context::ProverSumcheckContext, helpers::projection_flatter_1_times_matrix},
    },
};

impl ProverSumcheckContext {
    pub fn load_data(
        &mut self,
        witness: &Vec<RingElement>,
        witness_conjugated: &Vec<RingElement>,
        evaluation_points_inner: &Vec<StructuredRow>,
        evaluation_points_outer: &Vec<RingElement>,
        projection_matrix: &Option<ProjectionMatrix>,
        projection_batching_challenges: &Option<BatchingChallenges>,
        unstructured_projection_batching_challenges: &Option<
            [BatchedProjectionChallenges; NOF_BATCHES],
        >,
        vdf_challenge: Option<&RingElement>,
        vdf_crs_param: Option<&vdf_crs>,
    ) {
        self.witness_sumcheck.borrow_mut().load_from(&witness);
        self.witness_conjugated_sumcheck
            .borrow_mut()
            .load_from(&witness_conjugated);
        if let Some(projection_challenges) = projection_batching_challenges {
            let c0_expanded = PreprocessedRow::from_structured_row(&projection_challenges.c0);
            let c1_expanded = PreprocessedRow::from_structured_row(&projection_challenges.c1);
            let c2_expanded = PreprocessedRow::from_structured_row(&projection_challenges.c2);
            let flattened_projection = projection_flatter_1_times_matrix(
                projection_matrix.as_ref().unwrap(),
                &c1_expanded,
            );
            let mut flattened_projection_ring =
                new_vec_zero_preallocated(flattened_projection.len());

            for (i, el) in flattened_projection.iter().enumerate() {
                field_to_ring_element_into(&mut flattened_projection_ring[i], el);
                // TODO: I Spent 1h debugging this and it turned out that I forgot to convert the flattened projection matrix from homogenized field extensions to incomplete NTT, which is what the sumcheck expects. Rethink the interfaces here to avoid such issues in the future, maybe by having a clear type for the per rep.
                flattened_projection_ring[i].from_homogenized_field_extensions_to_incomplete_ntt();
            }
            if let Some(type3) = &mut self.type3sumcheck {
                type3
                    .c0r_sumcheck
                    .borrow_mut()
                    .load_from(&c0_expanded.preprocessed_row);
                type3
                    .c1r_sumcheck
                    .borrow_mut()
                    .load_from(&c1_expanded.preprocessed_row);
                type3
                    .c2r_sumcheck
                    .borrow_mut()
                    .load_from(&c2_expanded.preprocessed_row);

                type3
                    .c0l_sumcheck
                    .borrow_mut()
                    .load_from(&c0_expanded.preprocessed_row);
                type3
                    .c2l_sumcheck
                    .borrow_mut()
                    .load_from(&c2_expanded.preprocessed_row);
                type3
                    .flattened_projection_matrix_sumcheck
                    .borrow_mut()
                    .load_from(&flattened_projection_ring);
            } else {
                panic!(
                    "Projection batching challenges provided but type3 sumcheck is not initialized"
                );
            }
        }

        if let Some(unstructured_projection_challenges) =
            unstructured_projection_batching_challenges
        {
            for (batch_idx, batch_challenges) in
                unstructured_projection_challenges.iter().enumerate()
            {
                // Lift c_0_values from u64 to RingElement and load into lhs_flatter_0
                let c_0_ring: Vec<RingElement> = batch_challenges
                    .c_0_values
                    .iter()
                    .map(|&val| RingElement::constant(val, Representation::IncompleteNTT))
                    .collect();

                let c_2_ring: Vec<RingElement> = batch_challenges
                    .c_2_values
                    .iter()
                    .map(|&val| RingElement::constant(val, Representation::IncompleteNTT))
                    .collect();

                if let Some(type31sumchecks) = &mut self.type31sumchecks {
                    type31sumchecks[batch_idx]
                        .c_0_sumcheck
                        .borrow_mut()
                        .load_from(&c_0_ring);
                    type31sumchecks[batch_idx]
                        .c_2_sumcheck
                        .borrow_mut()
                        .load_from(&c_2_ring);
                    type31sumchecks[batch_idx]
                        .j_batched_sumcheck
                        .borrow_mut()
                        .load_from(&batch_challenges.j_batched);
                } else {
                    panic!(
                        "Unstructured projection batching challenges provided but type 3.1 sumcheck is not initialized"
                    );
                }
            }
        }
        for (i, type1) in self.type1sumcheck.iter_mut().enumerate() {
            let evaluation_points_inner_expanded =
                PreprocessedRow::from_structured_row(&evaluation_points_inner[i]);
            type1
                .inner_evaluation_sumcheck
                .borrow_mut()
                .load_from(&evaluation_points_inner_expanded.preprocessed_row);
            type1
                .outer_evaluation_sumcheck
                .borrow_mut()
                .load_from(&evaluation_points_outer);
        }

        if let Some(vdf) = &mut self.vdfsumcheck {
            let c = vdf_challenge.expect("VDF sumcheck enabled but no vdf_challenge provided");
            let vdf_crs_ref = vdf_crs_param.expect("VDF sumcheck enabled but no vdf_crs provided");

            // Compute vdf_batched_row[j] for j = 0..VDF_MATRIX_WIDTH-1:
            //   vdf_batched_row[j] = c^{j/VDF_BITS} · 2^{j%VDF_BITS} + sum_{r=0}^{HEIGHT-1} c^{HEIGHT+r} · A[r,j]
            // The G contribution: block row (j/VDF_BITS) contributes 2^{j%VDF_BITS} with weight c^{j/VDF_BITS}.
            // The A contribution: each row r contributes A[r,j] with weight c^{HEIGHT+r}.
            let mut batched_row: Vec<RingElement> = Vec::with_capacity(VDF_MATRIX_WIDTH);
            // Precompute c powers: c^0, c^1, ..., c^{2*HEIGHT-1}
            // (G uses c^0..c^{HEIGHT-1}, A uses c^{HEIGHT}..c^{2*HEIGHT-1})
            let num_local_powers = 2 * VDF_MATRIX_HEIGHT;
            let mut c_powers: Vec<RingElement> = Vec::with_capacity(num_local_powers);
            c_powers.push(RingElement::constant(1, Representation::IncompleteNTT));
            for _ in 1..num_local_powers {
                let prev = c_powers.last().unwrap().clone();
                c_powers.push(&prev * c);
            }
            let mut temp_a = RingElement::zero(Representation::IncompleteNTT);
            for j in 0..VDF_MATRIX_WIDTH {
                let block = j / VDF_BITS;
                let bit = j % VDF_BITS;
                // G contribution: c^{block} · 2^{bit}
                let mut row_j =
                    RingElement::constant((1u64 << bit) % MOD_Q, Representation::IncompleteNTT);
                row_j *= &c_powers[block];
                // A contributions: sum_{r=0}^{HEIGHT-1} c^{HEIGHT+r} · A[r,j]
                for r in 0..VDF_MATRIX_HEIGHT {
                    temp_a *= (&c_powers[VDF_MATRIX_HEIGHT + r], &vdf_crs_ref.A[(r, j)]);
                    row_j += &temp_a;
                }
                batched_row.push(row_j);
            }
            vdf.vdf_batched_row_sumcheck
                .borrow_mut()
                .load_from(&batched_row);

            // Compute vdf_step_powers[i] = c^{VDF_STRIDE * i} for i = 0..2K-1
            let two_k = witness.len() / 2 / VDF_MATRIX_WIDTH;
            let mut c_stride = RingElement::constant(1, Representation::IncompleteNTT);
            for _ in 0..VDF_STRIDE {
                c_stride *= c;
            }
            let mut step_powers: Vec<RingElement> = Vec::with_capacity(two_k);
            let mut c_power = RingElement::constant(1, Representation::IncompleteNTT);
            for _ in 0..two_k {
                step_powers.push(c_power.clone());
                c_power *= &c_stride;
            }
            vdf.vdf_step_powers_sumcheck
                .borrow_mut()
                .load_from(&step_powers);
        }
    }
}
