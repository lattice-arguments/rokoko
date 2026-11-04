//! Statement encoding for strong Fiat–Shamir.
//!
//! Only public inputs are encoded here. Proof messages keep their existing absorption order.
//! The protocol identifier fixes the built-in deterministic CRS construction.
use super::{
    commitment::{Placement, RecursionConfig},
    config::{Config, IntermediateConfig, Projection, SimpleConfig, SumcheckConfig},
};
use crate::common::{
    config::{DEGREE, MOD_Q},
    hash::HashWrapper,
    norms::NORMS_ENFORCED,
    ring_arithmetic::{FieldExtension, Representation, RingElement},
    structured_row::StructuredRow,
};

/// Fixed-width integers, counted sequences, and explicit enum tags make this encoding
/// unambiguous. Ring values use one representation regardless of their in-memory label.
pub(crate) struct Statement<'a>(pub &'a mut HashWrapper);

impl Statement<'_> {
    pub fn domain(&mut self, tag: &[u8]) {
        self.usize(tag.len());
        self.0.update_with_bytes(tag);
        self.usize(DEGREE);
        self.u64(MOD_Q);
    }
    pub fn u64(&mut self, n: u64) {
        self.0.update_with_u64(n);
    }
    pub fn usize(&mut self, n: usize) {
        self.u64(n as u64);
    }
    pub fn ring(&mut self, r: &RingElement) {
        let mut r = r.clone();
        r.to_representation(Representation::IncompleteNTT);
        for c in &r.v {
            self.u64(*c);
        }
    }
    pub fn field(&mut self, f: &FieldExtension) {
        for c in &f.coeffs {
            self.u64(*c);
        }
    }
    pub fn rings(&mut self, values: &[RingElement]) {
        self.usize(values.len());
        for r in values {
            self.ring(r);
        }
    }
    fn rows(&mut self, rows: &[StructuredRow]) {
        self.usize(rows.len());
        for row in rows {
            self.rings(&row.tensor_layers);
        }
    }
    pub fn openings(
        &mut self,
        inner: &[StructuredRow],
        outer: &[StructuredRow],
        claims: &[RingElement],
    ) {
        self.rows(inner);
        self.rows(outer);
        self.rings(claims);
    }
    fn placement(&mut self, p: &Placement) {
        self.usize(p.size);
        self.usize(p.blocks.len());
        for block in &p.blocks {
            self.usize(block.prefix);
            self.usize(block.length);
        }
    }
    fn recursion(&mut self, c: &RecursionConfig) {
        self.usize(c.decomposition_base_log);
        self.usize(c.decomposition_chunks);
        self.usize(c.rank);
        self.usize(c.placements.len());
        for p in &c.placements {
            self.placement(p);
        }
        self.u64(c.next.is_some() as u64);
        if let Some(next) = &c.next {
            self.recursion(next);
        }
    }
    fn next(&mut self, c: Option<&Config>) {
        match c {
            None => self.u64(0),
            Some(Config::Sumcheck(c)) => self.sumcheck(c),
            Some(Config::Intermediate(c)) => self.intermediate(c),
            Some(Config::Simple(c)) => self.simple(c),
        }
    }
    fn base(
        &mut self,
        height: usize,
        width: usize,
        ratio: usize,
        projection_height: usize,
        rank: usize,
    ) {
        for n in [height, width, ratio, projection_height, rank] {
            self.usize(n);
        }
    }
    // These describe the verifier's relation, excluding prover-only allocation hints
    // (next_level_usage_ratio) and expanded/cached key data.
    pub fn sumcheck(&mut self, c: &SumcheckConfig) {
        self.u64(1);
        self.base(
            c.witness_height,
            c.witness_width,
            c.projection_ratio,
            c.projection_height,
            c.basic_commitment_rank,
        );
        self.usize(c.nof_openings);
        self.usize(c.witness_decomposition_base_log);
        self.usize(c.witness_decomposition_chunks);
        self.usize(c.composed_witness_length);
        self.placement(&c.folded_witness_placement);
        self.recursion(&c.commitment_recursion);
        self.recursion(&c.opening_recursion);
        match &c.projection_recursion {
            Projection::Skip => self.u64(0),
            Projection::Coarse(r) => {
                self.u64(1);
                self.recursion(r);
            }
            Projection::Fine(r) => {
                self.u64(2);
                self.recursion(&r.recursion_constant_term);
                self.recursion(&r.recursion_batched_projection);
            }
        }
        self.u64(c.norm_bound.to_bits());
        self.u64(c.most_inner_norm_bound.to_bits());
        self.u64(c.exact_projection_norm as u64);
        if c.projection_norm_scope().is_some() {
            self.u64(c.projection_norm_bound.to_bits());
        }
        self.next(c.next.as_deref());
    }
    pub fn intermediate(&mut self, c: &IntermediateConfig) {
        self.u64(2);
        self.base(
            c.witness_height,
            c.witness_width,
            c.projection_ratio,
            c.projection_height,
            c.basic_commitment_rank,
        );
        self.usize(c.nof_openings);
        self.usize(c.witness_decomposition_base_log);
        self.usize(c.witness_decomposition_chunks);
        self.u64(c.norm_bound.to_bits());
        self.u64(c.projection_norm_bound.to_bits());
        self.next(c.next.as_deref());
    }
    pub fn simple(&mut self, c: &SimpleConfig) {
        self.u64(3);
        self.base(
            c.witness_height,
            c.witness_width,
            c.projection_ratio,
            c.projection_height,
            c.basic_commitment_rank,
        );
        self.u64(c.witness_norm_bound.to_bits());
        self.u64(c.projection_norm_bound.to_bits());
    }
    pub fn pcs_domain(&mut self) {
        self.domain(b"rokoko/pcs/strong-fs/v1");
        self.u64(NORMS_ENFORCED as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::init_common;

    #[test]
    fn opening_statement_binds_points_and_values() {
        init_common();
        let zero = RingElement::zero(Representation::IncompleteNTT);
        let one = RingElement::one(Representation::IncompleteNTT);
        let row = |value| StructuredRow {
            tensor_layers: vec![value],
        };
        let digest = |inner, outer, value| {
            let mut hash = HashWrapper::new();
            Statement(&mut hash).openings(&[row(inner)], &[row(outer)], &[value]);
            hash.sample_bytes(32)
        };
        let original = digest(zero.clone(), zero.clone(), zero.clone());
        assert_ne!(original, digest(one.clone(), zero.clone(), zero.clone()));
        assert_ne!(original, digest(zero.clone(), one.clone(), zero.clone()));
        assert_ne!(original, digest(zero.clone(), zero.clone(), one));
    }

}
