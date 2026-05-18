use crate::{
    common::{
        config::MOD_Q,
        ring_arithmetic::{Representation, RingElement},
        structured_row::StructuredRow,
    },
    hexl::bindings::inv_mod,
    protocol::open::evaluation_point_to_structured_row,
};

pub struct InitialEvaluationPoints {
    pub inner: Vec<StructuredRow>,
    pub outer: Vec<StructuredRow>,
    pub witness_inner: Vec<StructuredRow>,
    pub witness_claim_scale: RingElement,
}

pub fn sample_initial_evaluation_points(
    witness_height: usize,
    witness_width: usize,
    decomposition_base_log: usize,
    decomposition_chunks: usize,
) -> InitialEvaluationPoints {
    let (inner, witness_inner, witness_claim_scale) = sample_inner_evaluation_points(
        witness_height,
        decomposition_base_log,
        decomposition_chunks,
    );

    InitialEvaluationPoints {
        inner,
        outer: sample_outer_evaluation_points(witness_width),
        witness_inner,
        witness_claim_scale,
    }
}

fn sample_inner_evaluation_points(
    witness_height: usize,
    decomposition_base_log: usize,
    decomposition_chunks: usize,
) -> (Vec<StructuredRow>, Vec<StructuredRow>, RingElement) {
    debug_assert!(witness_height.is_power_of_two());
    debug_assert!(decomposition_chunks.is_power_of_two());

    let mut witness_evaluation_point = Vec::with_capacity(witness_height.ilog2() as usize);
    witness_evaluation_point.extend(
        (0..witness_height.ilog2() as usize)
            .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 2)),
    );

    let (decomposition_layers, witness_claim_scale) =
        decomposition_selector_layers_and_scale(decomposition_base_log, decomposition_chunks);

    let mut evaluation_point =
        Vec::with_capacity(witness_evaluation_point.len() + decomposition_layers.len());
    evaluation_point.extend_from_slice(&witness_evaluation_point);
    evaluation_point.extend(decomposition_layers);

    (
        vec![evaluation_point_to_structured_row(&evaluation_point)],
        vec![evaluation_point_to_structured_row(
            &witness_evaluation_point,
        )],
        witness_claim_scale,
    )
}

fn decomposition_selector_layers_and_scale(
    decomposition_base_log: usize,
    decomposition_chunks: usize,
) -> (Vec<RingElement>, RingElement) {
    let base = mod_pow(2, decomposition_base_log as u64);
    let chunk_bits = decomposition_chunks.ilog2() as usize;
    let mut layers = Vec::with_capacity(chunk_bits);
    let mut scale = 1u64;

    for bit in (0..chunk_bits).rev() {
        let basis_power = mod_pow(base, 1u64 << bit);
        let inverse_scale = unsafe { inv_mod((basis_power + 1) % MOD_Q, MOD_Q) };
        let selector = ((basis_power as u128 * inverse_scale as u128) % MOD_Q as u128) as u64;
        scale = ((scale as u128 * inverse_scale as u128) % MOD_Q as u128) as u64;
        layers.push(RingElement::constant(
            selector,
            Representation::IncompleteNTT,
        ));
    }

    (
        layers,
        RingElement::constant(scale, Representation::IncompleteNTT),
    )
}

fn mod_pow(mut base: u64, mut exponent: u64) -> u64 {
    let mut result = 1u64;
    base %= MOD_Q;

    while exponent > 0 {
        if exponent & 1 == 1 {
            result = ((result as u128 * base as u128) % MOD_Q as u128) as u64;
        }
        base = ((base as u128 * base as u128) % MOD_Q as u128) as u64;
        exponent >>= 1;
    }

    result
}

fn sample_outer_evaluation_points(witness_width: usize) -> Vec<StructuredRow> {
    debug_assert!(witness_width.is_power_of_two());

    vec![evaluation_point_to_structured_row(
        &(0..witness_width.ilog2() as usize)
            .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 2))
            .collect::<Vec<RingElement>>(),
    )]
}
