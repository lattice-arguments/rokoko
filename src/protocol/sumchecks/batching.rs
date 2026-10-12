//! A term claimed only through `Σ_s w_s · slot_s(y)` gets `slot_s(γ) = ρ · w_s / fe_s`, so it
//! enters the batched claim as `ρ · Σ_s w_s · slot_s(y)`.

use std::sync::LazyLock;

use crate::{
    common::{
        config::{NOF_BATCHES, NUM_SLOTS},
        hash::HashWrapper,
        ring_arithmetic::{FieldExtension, Representation, RingElement, CONSTANT_TERM_FACTORS},
        sumcheck_element::SumcheckElement,
    },
    protocol::config::{Projection, SumcheckConfig},
};

pub type SlotWeights = [FieldExtension; NUM_SLOTS];

/// Part 0 of `Σ_s CONSTANT_TERM_FACTORS[s] · slot_s(a)` is `ct(a)`.
pub static TRACE_WEIGHTS: LazyLock<SlotWeights> =
    LazyLock::new(|| std::array::from_fn(|s| FieldExtension::from_base(CONSTANT_TERM_FACTORS[s])));

pub fn slot_form(weights: &SlotWeights, a: &RingElement) -> FieldExtension {
    let mut homogenized = a.clone();
    homogenized.from_incomplete_ntt_to_homogenized_field_extensions();
    let mut result = FieldExtension::zero();
    for (w, slot) in weights
        .iter()
        .zip(homogenized.split_into_field_extensions().iter())
    {
        result += &(*w * *slot);
    }
    result
}

/// `fe_s · slot_s(a)`.
pub fn slot_weights(fe: &SlotWeights, a: &RingElement) -> SlotWeights {
    let mut homogenized = a.clone();
    homogenized.from_incomplete_ntt_to_homogenized_field_extensions();
    let slots = homogenized.split_into_field_extensions();
    std::array::from_fn(|s| fe[s] * slots[s])
}

/// `Σ_s w_s · slot_s(y) = value`.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldClaim {
    pub weights: SlotWeights,
    pub value: FieldExtension,
}

pub enum OuterClaim {
    Ring(RingElement),
    Field(FieldClaim),
}

/// The outer evaluation claims given as `(opening, weights)`, then the trace terms.
pub fn field_terms<'a>(
    config: &SumcheckConfig,
    outputs: usize,
    outer: impl IntoIterator<Item = (usize, &'a SlotWeights)>,
) -> Vec<(usize, &'a SlotWeights)> {
    let first_outer = config.basic_commitment_rank + config.nof_openings;
    outer
        .into_iter()
        .map(|(opening, weights)| (first_outer + opening, weights))
        .chain(
            TraceTerms::new(config, outputs)
                .indices()
                .into_iter()
                .map(|index| (index, &*TRACE_WEIGHTS)),
        )
        .collect()
}

/// The combiner outputs claimed through `TRACE_WEIGHTS`.
pub struct TraceTerms {
    pub constant_terms: Option<[usize; NOF_BATCHES]>,
    pub norm: usize,
    pub most_inner_norm: usize,
    pub projection_norm: Option<usize>,
    pub binariness: Option<usize>,
}

impl TraceTerms {
    pub fn new(config: &SumcheckConfig, outputs: usize) -> TraceTerms {
        let projection = config.basic_commitment_rank + 2 * config.nof_openings;
        let constant_terms = matches!(config.projection_recursion, Projection::Fine(_))
            .then(|| std::array::from_fn(|i| projection + 2 * i + 1));
        let projection_norm = config.projection_norm_scope().is_some();
        let binariness = !config.binary_leaves().is_empty();
        let norm = outputs - 2 - projection_norm as usize - binariness as usize;
        TraceTerms {
            constant_terms,
            norm,
            most_inner_norm: norm + 1,
            projection_norm: projection_norm.then_some(norm + 2),
            binariness: binariness.then_some(norm + 2 + projection_norm as usize),
        }
    }

    pub fn indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = self.constant_terms.iter().flatten().copied().collect();
        indices.extend([self.norm, self.most_inner_norm]);
        indices.extend(self.projection_norm);
        indices.extend(self.binariness);
        indices
    }
}

pub struct Batching {
    pub combination: Vec<RingElement>,
    pub fe: SlotWeights,
    pub rho: Vec<FieldExtension>,
}

/// `fe` is resampled until every slot is nonzero.
pub fn sample_batching(
    hash_wrapper: &mut HashWrapper,
    outputs: usize,
    field_terms: &[(usize, &SlotWeights)],
) -> Batching {
    let mut combination = vec![RingElement::zero(Representation::IncompleteNTT); outputs];
    hash_wrapper.sample_ring_element_vec_into(&mut combination);

    let (fe, fe_inverse) = loop {
        let mut combination_to_field = RingElement::zero(Representation::IncompleteNTT);
        hash_wrapper.sample_ring_element_into(&mut combination_to_field);
        combination_to_field.from_incomplete_ntt_to_homogenized_field_extensions();
        let fe = combination_to_field.split_into_field_extensions();
        if fe.iter().all(|f| *f != FieldExtension::zero()) {
            break (
                fe,
                combination_to_field.inverse().split_into_field_extensions(),
            );
        }
    };

    let rho = field_terms
        .iter()
        .map(|&(index, weights)| {
            let mut rho = FieldExtension::zero();
            hash_wrapper.sample_field_element_into(&mut rho);
            let slots: SlotWeights = std::array::from_fn(|s| rho * weights[s] * fe_inverse[s]);
            let coefficient = &mut combination[index];
            coefficient.representation = Representation::HomogenizedFieldExtensions;
            coefficient.combine_from_field_extensions(&slots);
            coefficient.from_homogenized_field_extensions_to_incomplete_ntt();
            rho
        })
        .collect();

    Batching {
        combination,
        fe,
        rho,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::init_common;

    #[test]
    fn field_claimed_terms_enter_as_rho_times_their_slot_form() {
        init_common();
        let claim = RingElement::random(Representation::IncompleteNTT);
        let other = RingElement::random(Representation::IncompleteNTT);
        let weights: SlotWeights = std::array::from_fn(|s| {
            let mut w = FieldExtension::zero();
            w.coeffs[0] = 3 + s as u64;
            w.coeffs[1] = 7 * s as u64;
            w
        });
        let mut hash_wrapper = HashWrapper::new();
        let batching = sample_batching(&mut hash_wrapper, 3, &[(1, &TRACE_WEIGHTS), (2, &weights)]);

        let batched = |coefficient: &RingElement, y: &RingElement| {
            let mut product = coefficient.clone();
            product *= y;
            slot_form(&batching.fe, &product)
        };
        assert_eq!(
            batched(&batching.combination[1], &claim),
            batching.rho[0] * slot_form(&TRACE_WEIGHTS, &claim)
        );
        assert_eq!(
            batched(&batching.combination[2], &other),
            batching.rho[1] * slot_form(&weights, &other)
        );
        assert_eq!(
            slot_form(&TRACE_WEIGHTS, &claim).coeffs[0],
            claim.constant_term_from_incomplete_ntt()
        );
    }
}
