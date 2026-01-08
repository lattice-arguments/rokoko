use std::cell::RefCell;

use crate::{
    common::{
        ring_arithmetic::{Representation, RingElement},
        sumcheck_element::SumcheckElement,
    },
    protocol::sumcheck::{
        common::HighOrderSumcheckData,
        hypercube_point::HypercubePoint,
        polynomial::{mul_poly_into, Polynomial},
    },
};

#[cfg(test)]
use crate::{
    common::config::MOD_Q,
    protocol::sumcheck::{common::SumcheckBaseData, linear::LinearSumcheck},
};

/// Sumcheck data that represents a pointwise product of two other sumcheck polynomials.
/// Each inner sumcheck is evaluated at the same hypercube point and the resulting
/// univariate polynomials are multiplied together.
pub struct ProductSumcheck<'a, E: SumcheckElement = RingElement> {
    pub lhs_sumcheck: &'a RefCell<dyn HighOrderSumcheckData<Element = E> + 'a>, // interior mutability to share between protocols
    pub rhs_sumcheck: &'a RefCell<dyn HighOrderSumcheckData<Element = E> + 'a>,

    lhs_eval_poly: RefCell<Polynomial<E>>,
    rhs_eval_poly: RefCell<Polynomial<E>>,
    scratch_poly: RefCell<Polynomial<E>>,
}

impl<'a, E: SumcheckElement> ProductSumcheck<'a, E> {
    pub fn new(
        lhs_sumcheck: &'a RefCell<dyn HighOrderSumcheckData<Element = E> + 'a>,
        rhs_sumcheck: &'a RefCell<dyn HighOrderSumcheckData<Element = E> + 'a>,
    ) -> ProductSumcheck<'a, E> {
        assert_eq!(
            lhs_sumcheck.borrow().variable_count(),
            rhs_sumcheck.borrow().variable_count(),
            "Inner product sumcheck: both sumchecks must have the same data length"
        );

        ProductSumcheck {
            lhs_sumcheck,
            rhs_sumcheck,
            lhs_eval_poly: RefCell::new(Polynomial::new(0)),
            rhs_eval_poly: RefCell::new(Polynomial::new(0)),
            scratch_poly: RefCell::new(Polynomial::new(0)),
        }
    }
}

impl<E: SumcheckElement> HighOrderSumcheckData for ProductSumcheck<'_, E> {
    type Element = E;

    fn get_scratch_poly(&self) -> &RefCell<Polynomial<E>> {
        &self.scratch_poly
    }
    fn max_num_polynomial_coefficients(&self) -> usize {
        self.lhs_sumcheck.borrow().max_num_polynomial_coefficients()
            + self.rhs_sumcheck.borrow().max_num_polynomial_coefficients()
            - 1
    }

    fn variable_count(&self) -> usize {
        self.lhs_sumcheck.borrow().variable_count()
    }

    fn is_univariate_polynomial_zero_at_point(&self, point: HypercubePoint) -> bool {
        self.lhs_sumcheck
            .borrow()
            .is_univariate_polynomial_zero_at_point(point)
            || self
                .rhs_sumcheck
                .borrow()
                .is_univariate_polynomial_zero_at_point(point)
    }

    fn univariate_polynomial_at_point_into(
        &self,
        point: HypercubePoint,
        polynomial: &mut Polynomial<E>,
    ) {
        // Reset accumulator for this point and build g(x) = g_lhs(x) * g_rhs(x).
        polynomial.set_zero();
        polynomial.num_coefficients = 0;

        let mut lhs_eval_poly = self.lhs_eval_poly.borrow_mut();
        let mut rhs_eval_poly = self.rhs_eval_poly.borrow_mut();

        self.lhs_sumcheck
            .borrow()
            .univariate_polynomial_at_point_into(point, &mut lhs_eval_poly);

        self.rhs_sumcheck
            .borrow()
            .univariate_polynomial_at_point_into(point, &mut rhs_eval_poly);

        mul_poly_into(polynomial, &lhs_eval_poly, &rhs_eval_poly);
    }
}

#[test]
fn test_inner_product_sumcheck() {
    let lhs_data = vec![
        RingElement::constant(1, Representation::IncompleteNTT),
        RingElement::constant(2, Representation::IncompleteNTT),
        RingElement::constant(3, Representation::IncompleteNTT),
        RingElement::constant(4, Representation::IncompleteNTT),
        RingElement::constant(5, Representation::IncompleteNTT),
        RingElement::constant(6, Representation::IncompleteNTT),
        RingElement::constant(7, Representation::IncompleteNTT),
        RingElement::constant(8, Representation::IncompleteNTT),
    ];

    let rhs_data = vec![
        RingElement::constant(9, Representation::IncompleteNTT),
        RingElement::constant(10, Representation::IncompleteNTT),
        RingElement::constant(11, Representation::IncompleteNTT),
        RingElement::constant(12, Representation::IncompleteNTT),
        RingElement::constant(13, Representation::IncompleteNTT),
        RingElement::constant(14, Representation::IncompleteNTT),
        RingElement::constant(15, Representation::IncompleteNTT),
        RingElement::constant(16, Representation::IncompleteNTT),
    ];

    let sumcheck_0 = RefCell::new(LinearSumcheck::new(lhs_data.len()));
    sumcheck_0.borrow_mut().load_from(&lhs_data);
    let sumcheck_1 = RefCell::new(LinearSumcheck::new(rhs_data.len()));
    sumcheck_1.borrow_mut().load_from(&rhs_data);

    // Build a product sumcheck that should track the inner product of lhs_data and rhs_data.
    let inner_product_sumcheck = ProductSumcheck::new(&sumcheck_0, &sumcheck_1);

    let mut univariate_poly = Polynomial::new(0);

    inner_product_sumcheck.univariate_polynomial_into(&mut univariate_poly);

    // The polynomial evaluated at 0 and 1 should sum to the true inner product.

    assert_eq!(
        &univariate_poly.at_zero() + &univariate_poly.at_one(),
        RingElement::constant(
            1 * 9 + 2 * 10 + 3 * 11 + 4 * 12 + 5 * 13 + 6 * 14 + 7 * 15 + 8 * 16,
            Representation::IncompleteNTT
        )
    );

    let r0 = RingElement::constant(524, Representation::IncompleteNTT);

    let claim = univariate_poly.at(&r0);

    // Fold both underlying multilinear extensions by r0 and ensure the verifier
    // still sees the same claim when re-running round 0 of the protocol.
    sumcheck_0.borrow_mut().partial_evaluate(&r0);
    sumcheck_1.borrow_mut().partial_evaluate(&r0);

    inner_product_sumcheck.univariate_polynomial_into(&mut univariate_poly);

    assert_eq!(
        &univariate_poly.at_zero() + &univariate_poly.at_one(),
        claim
    );

    let r1 = RingElement::constant(1337, Representation::IncompleteNTT);

    let claim = univariate_poly.at(&r1);

    // Same invariance check after the second round challenge.
    sumcheck_0.borrow_mut().partial_evaluate(&r1);
    sumcheck_1.borrow_mut().partial_evaluate(&r1);

    inner_product_sumcheck.univariate_polynomial_into(&mut univariate_poly);

    assert_eq!(
        &univariate_poly.at_zero() + &univariate_poly.at_one(),
        claim
    );

    let r2 = RingElement::constant(42, Representation::IncompleteNTT);

    let claim = univariate_poly.at(&r2);

    // After the final fold, the product of the two fully evaluated claims
    // should equal the verifier's accumulated claim.
    sumcheck_0.borrow_mut().partial_evaluate(&r2);
    sumcheck_1.borrow_mut().partial_evaluate(&r2);

    assert_eq!(
        sumcheck_0.borrow().final_evaluations() * sumcheck_1.borrow().final_evaluations(),
        claim,
    );

    // Explicit multilinear evaluations of each folded claim for documentation.
    assert_eq!(
        sumcheck_0.borrow().final_evaluations(),
        &RingElement::constant(
            (MOD_Q as i64
                + (1 - 524) * (1 - 1337) * (1 - 42) * 1
                + (1 - 524) * (1 - 1337) * (42) * 2
                + (1 - 524) * (1337) * (1 - 42) * 3
                + (1 - 524) * (1337) * (42) * 4
                + (524) * (1 - 1337) * (1 - 42) * 5
                + (524) * (1 - 1337) * (42) * 6
                + (524) * (1337) * (1 - 42) * 7
                + (524) * (1337) * (42) * 8) as u64,
            Representation::IncompleteNTT,
        )
    );

    assert_eq!(
        sumcheck_1.borrow().final_evaluations(),
        &RingElement::constant(
            (MOD_Q as i64
                + (1 - 524) * (1 - 1337) * (1 - 42) * 9
                + (1 - 524) * (1 - 1337) * (42) * 10
                + (1 - 524) * (1337) * (1 - 42) * 11
                + (1 - 524) * (1337) * (42) * 12
                + (524) * (1 - 1337) * (1 - 42) * 13
                + (524) * (1 - 1337) * (42) * 14
                + (524) * (1337) * (1 - 42) * 15
                + (524) * (1337) * (42) * 16) as u64,
            Representation::IncompleteNTT,
        )
    );

    // Final consistency: inner product claim equals product of individual folded claims.
    assert_eq!(
        claim,
        RingElement::constant(
            (MOD_Q as i64
                + ((1 - 524) * (1 - 1337) * (1 - 42) * 1
                    + (1 - 524) * (1 - 1337) * (42) * 2
                    + (1 - 524) * (1337) * (1 - 42) * 3
                    + (1 - 524) * (1337) * (42) * 4
                    + (524) * (1 - 1337) * (1 - 42) * 5
                    + (524) * (1 - 1337) * (42) * 6
                    + (524) * (1337) * (1 - 42) * 7
                    + (524) * (1337) * (42) * 8)
                    * ((1 - 524) * (1 - 1337) * (1 - 42) * 9
                        + (1 - 524) * (1 - 1337) * (42) * 10
                        + (1 - 524) * (1337) * (1 - 42) * 11
                        + (1 - 524) * (1337) * (42) * 12
                        + (524) * (1 - 1337) * (1 - 42) * 13
                        + (524) * (1 - 1337) * (42) * 14
                        + (524) * (1337) * (1 - 42) * 15
                        + (524) * (1337) * (42) * 16)) as u64,
            Representation::IncompleteNTT,
        )
    );
}

#[test]
fn test_self_inner_product_sumcheck() {
    let data = vec![
        RingElement::constant(1, Representation::IncompleteNTT),
        RingElement::constant(2, Representation::IncompleteNTT),
        RingElement::constant(3, Representation::IncompleteNTT),
        RingElement::constant(4, Representation::IncompleteNTT),
        RingElement::constant(5, Representation::IncompleteNTT),
        RingElement::constant(6, Representation::IncompleteNTT),
        RingElement::constant(7, Representation::IncompleteNTT),
        RingElement::constant(8, Representation::IncompleteNTT),
    ];

    let sumcheck = RefCell::new(LinearSumcheck::new(data.len()));
    sumcheck.borrow_mut().load_from(&data);

    let inner_product_sumcheck = ProductSumcheck::new(&sumcheck, &sumcheck);

    let mut univariate_poly = Polynomial::new(0);

    inner_product_sumcheck.univariate_polynomial_into(&mut univariate_poly);

    // When both inputs are identical, the inner product collapses to a sum of squares.
    assert_eq!(
        &univariate_poly.at_zero() + &univariate_poly.at_one(),
        RingElement::constant(
            1 * 1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5 + 6 * 6 + 7 * 7 + 8 * 8,
            Representation::IncompleteNTT
        )
    );
}

#[test]
fn test_three_way_sumcheck() {
    let data0 = vec![
        RingElement::constant(1, Representation::IncompleteNTT),
        RingElement::constant(2, Representation::IncompleteNTT),
        RingElement::constant(3, Representation::IncompleteNTT),
        RingElement::constant(4, Representation::IncompleteNTT),
    ];

    let data1 = vec![
        RingElement::constant(5, Representation::IncompleteNTT),
        RingElement::constant(6, Representation::IncompleteNTT),
        RingElement::constant(7, Representation::IncompleteNTT),
        RingElement::constant(8, Representation::IncompleteNTT),
    ];

    let data2 = vec![
        RingElement::constant(9, Representation::IncompleteNTT),
        RingElement::constant(10, Representation::IncompleteNTT),
        RingElement::constant(11, Representation::IncompleteNTT),
        RingElement::constant(12, Representation::IncompleteNTT),
    ];

    let mut sumcheck_0 = LinearSumcheck::new(data0.len());
    sumcheck_0.load_from(&data0);
    let mut sumcheck_1 = LinearSumcheck::new(data1.len());
    sumcheck_1.load_from(&data1);
    let mut sumcheck_2 = LinearSumcheck::new(data2.len());
    sumcheck_2.load_from(&data2);

    let sumcheck_0_ref = RefCell::new(sumcheck_0);
    let sumcheck_1_ref = RefCell::new(sumcheck_1);
    let sumcheck_2_ref = RefCell::new(sumcheck_2);

    // Compose two nested product sumchecks to validate a three-way product.

    let inner_product_sumcheck_01 = ProductSumcheck::new(&sumcheck_0_ref, &sumcheck_1_ref);

    let inner_product_sumcheck_01_ref = RefCell::new(inner_product_sumcheck_01);

    let inner_product_sumcheck_012 =
        ProductSumcheck::new(&inner_product_sumcheck_01_ref, &sumcheck_2_ref);

    let mut univariate_poly = Polynomial::new(0);
    inner_product_sumcheck_012.univariate_polynomial_into(&mut univariate_poly);

    // Evaluating the first-round polynomial at 0 and 1 should give the full triple product sum.
    assert_eq!(
        &univariate_poly.at_zero() + &univariate_poly.at_one(),
        RingElement::constant(
            1 * 5 * 9 + 2 * 6 * 10 + 3 * 7 * 11 + 4 * 8 * 12,
            Representation::IncompleteNTT
        )
    );

    let r0 = RingElement::constant(42, Representation::IncompleteNTT);

    let claim = univariate_poly.at(&r0);

    sumcheck_0_ref.borrow_mut().partial_evaluate(&r0);
    sumcheck_1_ref.borrow_mut().partial_evaluate(&r0);
    sumcheck_2_ref.borrow_mut().partial_evaluate(&r0);

    inner_product_sumcheck_012.univariate_polynomial_into(&mut univariate_poly);

    // The verifier's running claim should stay consistent after folding all three vectors.
    assert_eq!(
        &univariate_poly.at_zero() + &univariate_poly.at_one(),
        claim
    );
}

#[test]
fn test_product_of_linear_sumchecks_over_disjoint_variables() {
    // Test the product sumcheck over three linear sumchecks, each
    // defined over disjoint sets of variables.
    // The first sumcheck is defined over the highest-order variables,
    // the second over the middle variables, and the third over the
    // lowest-order variables.
    // we make use that the univariate polynomial of a linear sumcheck is just a degree 1 polynomial
    // since it should be a produce of two constant polys and one degree 1 poly.

    use crate::common::ring_arithmetic::RingElement;
    use crate::protocol::sumcheck::product::ProductSumcheck;

    let data1 = vec![
        RingElement::constant(1, Representation::IncompleteNTT),
        RingElement::constant(2, Representation::IncompleteNTT),
        RingElement::constant(3, Representation::IncompleteNTT),
        RingElement::constant(4, Representation::IncompleteNTT),
    ];

    let data2 = vec![
        RingElement::constant(5, Representation::IncompleteNTT),
        RingElement::constant(6, Representation::IncompleteNTT),
        RingElement::constant(7, Representation::IncompleteNTT),
        RingElement::constant(8, Representation::IncompleteNTT),
    ];

    let data3 = vec![
        RingElement::constant(9, Representation::IncompleteNTT),
        RingElement::constant(10, Representation::IncompleteNTT),
        RingElement::constant(11, Representation::IncompleteNTT),
        RingElement::constant(12, Representation::IncompleteNTT),
    ];

    let sumcheck_1 = RefCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
        data1.len(),
        0,
        4,
    ));
    sumcheck_1.borrow_mut().load_from(&data1);

    // 1 x 16, 2 x 16, 3 x 16, 4 x 16, 1 x 16, 2 x 16, 3 x 16, 4 x 16,

    let sumcheck_2 = RefCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
        data2.len(),
        2,
        2,
    ));

    // (5 x 4, 6 x 4, 7 x 4, 8 x 4, 5 x 4, 6 x 4, 7 x 4, 8 x 4) x 4

    sumcheck_2.borrow_mut().load_from(&data2);
    let sumcheck_3 = RefCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
        data3.len(),
        4,
        0,
    ));
    sumcheck_3.borrow_mut().load_from(&data3);

    let product_12 = RefCell::new(ProductSumcheck::new(&sumcheck_1, &sumcheck_2));
    let product_123 = RefCell::new(ProductSumcheck::new(&product_12, &sumcheck_3));

    let mut univariate_poly = Polynomial::new(0);

    let mut claim = RingElement::constant(
        (1 + 2 + 3 + 4) * (5 + 6 + 7 + 8) * (9 + 10 + 11 + 12),
        Representation::IncompleteNTT,
    );

    for _ in 0..5 {
        product_123
            .borrow()
            .univariate_polynomial_into(&mut univariate_poly);
        assert_eq!(
            &univariate_poly.at_zero() + &univariate_poly.at_one(),
            claim
        );

        assert_eq!(univariate_poly.num_coefficients, 2); // degree 1 polynomial as expected

        let r = RingElement::constant(7, Representation::IncompleteNTT);

        claim = univariate_poly.at(&r);

        sumcheck_1.borrow_mut().partial_evaluate(&r);
        sumcheck_2.borrow_mut().partial_evaluate(&r);
        sumcheck_3.borrow_mut().partial_evaluate(&r);
    }

    product_123
        .borrow()
        .univariate_polynomial_into(&mut univariate_poly);
    assert_eq!(
        &univariate_poly.at_zero() + &univariate_poly.at_one(),
        claim
    );

    assert_eq!(univariate_poly.num_coefficients, 2); // degree 1 polynomial as expected
}
