use crate::common::config::*;
use crate::protocol::project::BatchingChallenges;
use crate::{
    common::{
        arithmetic::{ONE, ZERO},
        config::{HALF_DEGREE, NOF_BATCHES},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    protocol::{
        config::{RoundConfig, SalsaaProof},
        open::evaluation_point_to_structured_row,
        parties::prover::vdf_crs,
        project_2::BatchedProjectionChallengesSuccinct,
        sumchecks::{
            context_verifier::VerifierSumcheckContext, helpers::projection_flatter_1_times_matrix,
        },
    },
};

impl VerifierSumcheckContext {
    pub fn load_data(
        &mut self,
        config: &RoundConfig,
        proof: &SalsaaProof,
        evaluation_points_ring: &[RingElement],
        evaluation_points_inner: &[StructuredRow],
        evaluation_points_outer: &[RingElement],
        batching_challenges: &Option<BatchingChallenges>,
        projection_matrix: &Option<ProjectionMatrix>,
        projection_challenges_unstructured: &Option<
            [BatchedProjectionChallengesSuccinct; NOF_BATCHES],
        >,
        combination: &[RingElement],
        qe: [QuadraticExtension; HALF_DEGREE],
        vdf_challenge: Option<&RingElement>,
        vdf_crs_param: Option<&vdf_crs>,
    ) {
        let outer_points_len =
            config.main_witness_columns.ilog2() as usize + config.main_witness_prefix.length;
        let outer_points = &evaluation_points_ring[0..outer_points_len].to_vec();
        let outer_points_expanded =
            PreprocessedRow::from_structured_row(&evaluation_point_to_structured_row(outer_points))
                .preprocessed_row;

        let mut temp = ZERO.clone();

        let mut claim_over_witness = ZERO.clone();
        for (claim, outer) in proof.claims.row(0).iter().zip(outer_points_expanded.iter()) {
            temp *= (claim, outer);
            claim_over_witness += &temp;
        }

        match proof {
            SalsaaProof::Intermediate {
                claim_over_projection,
                ..
            } => {
                temp *= (
                    claim_over_projection.get(0).unwrap(),
                    &outer_points_expanded[config.main_witness_columns],
                );
                claim_over_witness += &temp;
            }
            _ => {}
        }

        let mut main_cols_points =
            evaluation_points_ring[config.main_witness_prefix.length..outer_points_len].to_vec();
        for r in main_cols_points.iter_mut() {
            r.conjugate_in_place();
        }
        let main_cols_points_expanded = PreprocessedRow::from_structured_row(
            &evaluation_point_to_structured_row(&main_cols_points),
        )
        .preprocessed_row;

        let mut claim_over_conjugated_witness = ZERO.clone();
        for (claim, outer) in proof
            .claims
            .row(1)
            .iter()
            .zip(main_cols_points_expanded.iter())
        {
            temp *= (claim, outer);
            claim_over_conjugated_witness += &temp;
        }
        claim_over_conjugated_witness.conjugate_in_place();

        self.witness_evaluation
            .borrow_mut()
            .set_result(claim_over_witness);
        self.witness_conjugated_evaluation
            .borrow_mut()
            .set_result(claim_over_conjugated_witness);

        for (i, type1_eval) in self.type1evaluations.iter().enumerate() {
            type1_eval
                .inner_evaluation_sumcheck
                .borrow_mut()
                .load_from(evaluation_points_inner[i].clone());
            type1_eval
                .outer_evaluation_sumcheck
                .borrow_mut()
                .load_from(evaluation_points_outer);
        }

        if let Some(type3_eval) = &mut self.type3evaluation {
            let c1_expanded =
                PreprocessedRow::from_structured_row(&batching_challenges.as_ref().unwrap().c1);

            let flattened_projection = projection_flatter_1_times_matrix(
                projection_matrix.as_ref().unwrap(),
                &c1_expanded,
            );

            type3_eval
                .flattened_projection_matrix_evaluation
                .borrow_mut()
                .load_from(&flattened_projection);
            type3_eval
                .c0l_evaluation
                .borrow_mut()
                .load_from(batching_challenges.as_ref().unwrap().c0.clone());
            type3_eval
                .c2l_evaluation
                .borrow_mut()
                .load_from(batching_challenges.as_ref().unwrap().c2.clone());
            type3_eval
                .c0r_evaluation
                .borrow_mut()
                .load_from(batching_challenges.as_ref().unwrap().c0.clone());
            type3_eval
                .c1r_evaluation
                .borrow_mut()
                .load_from(batching_challenges.as_ref().unwrap().c1.clone());
            type3_eval
                .c2r_evaluation
                .borrow_mut()
                .load_from(batching_challenges.as_ref().unwrap().c2.clone());
        }

        if let Some(type31_evals) = &mut self.type31evaluations {
            let challenges = projection_challenges_unstructured
                .as_ref()
                .expect("Missing projection challenges for type 3.1 verifier sumcheck");
            for (batch_idx, type31_eval) in type31_evals.iter_mut().enumerate() {
                let c_0_structured = StructuredRow {
                    tensor_layers: challenges[batch_idx]
                        .c_0_layers
                        .iter()
                        .map(|&val| RingElement::constant(val, Representation::IncompleteNTT))
                        .collect(),
                };
                let c_2_structured = StructuredRow {
                    tensor_layers: challenges[batch_idx]
                        .c_2_layers
                        .iter()
                        .map(|&val| RingElement::constant(val, Representation::IncompleteNTT))
                        .collect(),
                };
                type31_eval
                    .c_0_evaluation
                    .borrow_mut()
                    .load_from(c_0_structured);
                type31_eval
                    .c_2_evaluation
                    .borrow_mut()
                    .load_from(c_2_structured);
                type31_eval
                    .j_batched_evaluation
                    .borrow_mut()
                    .load_from(&challenges[batch_idx].j_batched);
            }
        }

        if let Some(vdf_eval) = &mut self.vdfevaluation {
            let c = vdf_challenge.expect("VDF evaluation enabled but no vdf_challenge provided");
            let vdf_crs_ref =
                vdf_crs_param.expect("VDF evaluation enabled but no vdf_crs provided");

            // Compute vdf_batched_row[j] for j = 0..VDF_MATRIX_WIDTH-1:
            //   vdf_batched_row[j] = c^{j/VDF_BITS} · 2^{j%VDF_BITS} + Σ_{r} c^{HEIGHT+r} · A[r,j]
            let mut batched_row: Vec<RingElement> = Vec::with_capacity(VDF_MATRIX_WIDTH);
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
                let mut row_j =
                    RingElement::constant((1u64 << bit) % MOD_Q, Representation::IncompleteNTT);
                row_j *= &c_powers[block];
                for r in 0..VDF_MATRIX_HEIGHT {
                    temp_a *= (&c_powers[VDF_MATRIX_HEIGHT + r], &vdf_crs_ref.A[(r, j)]);
                    row_j += &temp_a;
                }
                batched_row.push(row_j);
            }
            vdf_eval
                .vdf_batched_row_evaluation
                .borrow_mut()
                .load_from(&batched_row);

            // Compute MLE[vdf_step_powers](x) where step_powers[i] = c^{VDF_STRIDE * i}
            // MLE = prod_k ((1-x_k) + x_k · (c^{VDF_STRIDE})^{2^{n-1-k}})
            // We iterate in reverse with c_power starting at c^{VDF_STRIDE} and squaring.
            let two_k = config.extended_witness_length / 2 / VDF_MATRIX_WIDTH;
            let step_powers_num_vars = two_k.ilog2() as usize;
            let prefix = 1usize; // MSB selector bit (column selector)
            let step_powers_vars = &evaluation_points_ring[prefix..prefix + step_powers_num_vars];

            let mut mle_step_powers = RingElement::constant(1, Representation::IncompleteNTT);
            // c_power starts at c^{VDF_STRIDE} (not c^1)
            let mut c_power = RingElement::constant(1, Representation::IncompleteNTT);
            for _ in 0..VDF_STRIDE {
                c_power *= c;
            }
            let mut temp_sq = RingElement::zero(Representation::IncompleteNTT);
            let mut term = RingElement::zero(Representation::IncompleteNTT);
            for x_i in step_powers_vars.iter().rev() {
                // factor = (1 - x_i) + x_i * c_power
                let mut factor = &*ONE - x_i;
                term *= (x_i, &c_power);
                factor += &term;
                mle_step_powers *= &factor;
                // c_power = c_power^2 for next iteration
                temp_sq *= (&c_power, &c_power);
                std::mem::swap(&mut c_power, &mut temp_sq);
            }
            vdf_eval
                .vdf_step_powers_evaluation
                .borrow_mut()
                .set_result(mle_step_powers);
        }

        self.combiner_evaluation
            .borrow_mut()
            .load_challenges_from(combination);
        self.field_combiner_evaluation
            .borrow_mut()
            .load_challenges_from(qe);
    }
}
