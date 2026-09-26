use crate::{
    common::{
        matrix::VerticallyAlignedMatrix,
        ring_arithmetic::{Representation, RingElement},
        sampling::sample_random_short_vector,
    },
    protocol::config::{Config, SumcheckConfig},
};

pub const NORM_MARGIN: f64 = 1.85; // verifier accepts norms up to this factor times the expected bound

pub struct Instantiation {
    pub config: Config,
    pub witness: InitialWitnessParams,
}

impl Instantiation {
    pub fn root(&self) -> &SumcheckConfig {
        match &self.config {
            Config::Sumcheck(config) => config,
            _ => panic!("Expected sumcheck config at the top level."),
        }
    }
}

pub fn assign_norm_bounds(config: &mut Config, bounds: &[[f64; 3]]) {
    fn rec(config: &mut Config, bounds: &[[f64; 3]], i: &mut usize) {
        match config {
            Config::Sumcheck(c) => {
                c.norm_bound = bounds[*i][0] * NORM_MARGIN;
                c.most_inner_norm_bound = bounds[*i][1] * NORM_MARGIN;
                c.projection_norm_bound = bounds[*i][2] * NORM_MARGIN;
                *i += 1;
                if let Some(next) = c.next.as_deref_mut() {
                    rec(next, bounds, i);
                }
            }
            Config::Intermediate(c) => {
                c.norm_bound = bounds[*i][0] * NORM_MARGIN;
                c.projection_norm_bound = bounds[*i][1] * NORM_MARGIN;
                *i += 1;
                if let Some(next) = c.next.as_deref_mut() {
                    rec(next, bounds, i);
                }
            }
            Config::Simple(c) => {
                c.witness_norm_bound = bounds[*i][0] * NORM_MARGIN;
                c.projection_norm_bound = bounds[*i][1] * NORM_MARGIN;
                *i += 1;
            }
        }
    }
    let mut i = 0;
    rec(config, bounds, &mut i);
    assert!(
        i <= bounds.len(),
        "norm-bound array length be at least the number of configs in the chain"
    );
}

pub struct InitialWitnessParams {
    pub height: usize,
    pub width: usize,
    pub decomposition_base_log: usize,
    pub decomposition_chunks: usize,
    pub initial_norm_log: usize,
}

impl InitialWitnessParams {
    pub fn digit_bound(&self) -> u64 {
        1u64 << (self.decomposition_base_log - 1)
    }
}

pub fn witness_sampler(params: &InitialWitnessParams) -> VerticallyAlignedMatrix<RingElement> {
    VerticallyAlignedMatrix {
        height: params.height,
        width: params.width,
        data: sample_random_short_vector(
            params.height * params.width,
            2u64.pow(params.initial_norm_log as u32 - 1),
            Representation::IncompleteNTT,
        ),
        used_cols: params.width,
    }
}

#[tracing::instrument(skip_all, name = "commit::decompose_witness")]
pub fn decompose_witness(
    witness: &VerticallyAlignedMatrix<RingElement>,
    params: &InitialWitnessParams,
) -> VerticallyAlignedMatrix<RingElement> {
    decompose_witness_with_digits(witness, params, None).0
}

/// The decomposition and, when asked, the same digits narrowed to `i16` in the coefficient
/// domain, which is the form the CRT commitment consumes.
#[tracing::instrument(skip_all, name = "commit::decompose_witness")]
pub fn decompose_witness_with_digits(
    witness: &VerticallyAlignedMatrix<RingElement>,
    params: &InitialWitnessParams,
    digits: Option<&mut Vec<crate::protocol::project_coarse::Signed16RingElement>>,
) -> (
    VerticallyAlignedMatrix<RingElement>,
    Option<VerticallyAlignedMatrix<crate::protocol::project_coarse::Signed16RingElement>>,
) {
    let height = witness.height * params.decomposition_chunks;
    let wanted = digits.is_some();
    let mut sink = digits;
    if let Some(sink) = sink.as_deref_mut() {
        sink.clear();
        sink.reserve(height * witness.width);
    }
    let decomposed_data = crate::common::decomposition::decompose_into(
        &witness.data,
        params.decomposition_base_log as u64,
        params.decomposition_chunks,
        sink.as_deref_mut(),
    );
    let decomposed = VerticallyAlignedMatrix {
        height,
        width: witness.width,
        data: decomposed_data,
        used_cols: witness.width,
    };
    let narrowed = wanted.then(|| VerticallyAlignedMatrix {
        height,
        width: witness.width,
        data: std::mem::take(sink.unwrap()),
        used_cols: witness.width,
    });
    (decomposed, narrowed)
}

/// Sizing rule for targets between parameter sets: keep the set's height and drop column bits
/// (p27 = p28 with one column-bit fewer). Returns the number of witness columns to use;
/// remaining columns stay zero (`used_cols` on the witness matrix).
pub fn witness_cols_for_target(
    witness_height: usize,
    witness_width: usize,
    target_log2_zq_coeffs: usize,
) -> usize {
    use crate::common::config::DEGREE;
    let full_log2 = (witness_height * witness_width * DEGREE).ilog2() as usize;
    assert!(
        target_log2_zq_coeffs <= full_log2,
        "target 2^{} exceeds the parameter set's capacity 2^{}",
        target_log2_zq_coeffs,
        full_log2
    );
    let drop = full_log2 - target_log2_zq_coeffs;
    assert!(
        drop < witness_width.ilog2() as usize,
        "target 2^{} too small for this parameter set; pick a smaller one",
        target_log2_zq_coeffs
    );
    witness_width >> drop
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_witness_cols_for_target() {
        // p-28-shaped set: 2^13 x 2^8 ring elements = 2^28 Zq coefficients
        assert_eq!(super::witness_cols_for_target(1 << 13, 1 << 8, 28), 1 << 8);
        // p27 rule: one column-bit fewer
        assert_eq!(super::witness_cols_for_target(1 << 13, 1 << 8, 27), 1 << 7);
        assert_eq!(super::witness_cols_for_target(1 << 13, 1 << 8, 25), 1 << 5);
    }
}
