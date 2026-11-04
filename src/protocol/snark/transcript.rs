//! Bind the compact, canonical claim language, without expanding tensor weights or
//! reading witness data. Sequence counts and node tags preserve the expression structure.
use super::lowering::{
    ClaimExpr, ClaimFactor, Coeffs, PublicFactor, SnarkClaim, Weights, WitnessShape,
};
use crate::{common::hash::HashWrapper, protocol::fiat_shamir::Statement};

pub(super) fn bind_claims(hash: &mut HashWrapper, shape: WitnessShape, claims: &[SnarkClaim]) {
    let mut s = Statement(hash);
    s.domain(b"rokoko/claims/strong-fs/v1");
    s.usize(shape.height);
    s.usize(shape.width);
    s.usize(claims.len());
    for claim in claims {
        expression(&mut s, &claim.expr);
        s.ring(&claim.value);
    }
}

fn coefficients(s: &mut Statement<'_>, c: &Coeffs) {
    match c {
        Coeffs::Ring(values) => {
            s.u64(0);
            s.rings(values);
        }
        Coeffs::Field(values) => {
            s.u64(1);
            s.usize(values.len());
            for value in values.iter() {
                s.field(value);
            }
        }
    }
}

fn public(s: &mut Statement<'_>, p: &PublicFactor) {
    s.usize(p.prefix_len);
    s.usize(p.suffix_len);
    match &p.weights {
        Weights::Tensor(c) => {
            s.u64(0);
            coefficients(s, c);
        }
        Weights::Dense(c) => {
            s.u64(1);
            coefficients(s, c);
        }
        Weights::Selector { bits, length } => {
            s.u64(2);
            s.usize(*bits);
            s.usize(*length);
        }
        Weights::Combination(parts) => {
            s.u64(3);
            s.usize(parts.len());
            for (scalar, factors) in parts.iter() {
                s.ring(scalar);
                s.usize(factors.len());
                for factor in factors {
                    public(s, factor);
                }
            }
        }
    }
}

fn expression(s: &mut Statement<'_>, expr: &ClaimExpr) {
    match expr {
        ClaimExpr::Factor(f) => {
            s.u64(0);
            match f {
                ClaimFactor::Witness => s.u64(0),
                ClaimFactor::ConjWitness => s.u64(1),
                ClaimFactor::WitnessSegment(p) | ClaimFactor::ConjWitnessSegment(p) => {
                    s.u64(if matches!(f, ClaimFactor::WitnessSegment(_)) {
                        2
                    } else {
                        3
                    });
                    s.usize(p.prefix);
                    s.usize(p.length);
                }
                ClaimFactor::Public(p) => {
                    s.u64(4);
                    public(s, p);
                }
            }
        }
        ClaimExpr::Constant(c) => {
            s.u64(1);
            s.ring(c);
        }
        ClaimExpr::Product(a, b) => {
            s.u64(2);
            expression(s, a);
            expression(s, b);
        }
        ClaimExpr::Sum(a, b) => {
            s.u64(3);
            expression(s, a);
            expression(s, b);
        }
        ClaimExpr::Diff(a, b) => {
            s.u64(4);
            expression(s, a);
            expression(s, b);
        }
        ClaimExpr::Scale(c, x) => {
            s.u64(5);
            s.ring(c);
            expression(s, x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{
        init_common,
        ring_arithmetic::{FieldExtension, Representation, RingElement},
    };
    use std::sync::Arc;

    fn scalar(n: u64) -> RingElement {
        RingElement::constant(n, Representation::IncompleteNTT)
    }

    fn digest(expr: ClaimExpr, shape: WitnessShape) -> Vec<u8> {
        let mut hash = HashWrapper::new();
        bind_claims(
            &mut hash,
            shape,
            &[SnarkClaim {
                expr,
                value: scalar(0),
            }],
        );
        hash.sample_bytes(32)
    }

    #[test]
    fn equal_values_do_not_hide_different_claim_expressions() {
        init_common();
        let shape = WitnessShape::new(2, 2);
        let witness = || ClaimExpr::Factor(ClaimFactor::Witness);
        let baseline = digest(witness(), shape);
        for expr in [
            ClaimExpr::Factor(ClaimFactor::ConjWitness),
            ClaimExpr::Constant(scalar(0)),
            ClaimExpr::Sum(Box::new(witness()), Box::new(witness())),
            ClaimExpr::Diff(Box::new(witness()), Box::new(witness())),
            ClaimExpr::Product(Box::new(witness()), Box::new(witness())),
            ClaimExpr::Scale(scalar(2), Box::new(witness())),
        ] {
            assert_ne!(baseline, digest(expr, shape));
        }
        assert_ne!(baseline, digest(witness(), WitnessShape::new(1, 4)));
    }

    #[test]
    fn public_weights_bind_values_placement_and_kind() {
        init_common();
        let shape = WitnessShape::new(2, 2);
        let public_digest = |prefix_len, suffix_len, weights| {
            digest(
                ClaimExpr::Factor(ClaimFactor::Public(PublicFactor {
                    prefix_len,
                    suffix_len,
                    weights,
                })),
                shape,
            )
        };
        let tensor =
            |n| Weights::Tensor(Coeffs::Field(Arc::new(vec![FieldExtension::from_base(n)])));
        let baseline = public_digest(1, 0, tensor(0));
        assert_ne!(baseline, public_digest(1, 0, tensor(1)));
        assert_ne!(baseline, public_digest(0, 1, tensor(0)));
        assert_ne!(
            baseline,
            public_digest(
                1,
                0,
                Weights::Dense(Coeffs::Field(Arc::new(vec![FieldExtension::from_base(0)])))
            )
        );
        assert_ne!(
            public_digest(0, 0, Weights::Selector { bits: 0, length: 1 }),
            public_digest(0, 0, Weights::Selector { bits: 1, length: 1 }),
        );
    }
}
