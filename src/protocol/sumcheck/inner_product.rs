use std::cell::RefCell;

use crate::{
    common::{
        config::MOD_Q,
        ring_arithmetic::{Representation, RingElement},
    },
    protocol::sumcheck::{
        common::{HighOrderSumcheckData, Polynomial, SumcheckBaseData},
        linear::LinearSumcheck,
    },
};

pub struct InnerProductSumcheck<'a> {
    pub sumcheck_0: &'a RefCell<LinearSumcheck>, // interior mutability to share between protocols
    pub sumcheck_1: &'a RefCell<LinearSumcheck>,
}

impl InnerProductSumcheck<'_> {
    pub fn new<'a>(
        sumcheck_0: &'a RefCell<LinearSumcheck>,
        sumcheck_1: &'a RefCell<LinearSumcheck>,
    ) -> InnerProductSumcheck<'a> {
        assert_eq!(
            sumcheck_0.borrow().data.len(),
            sumcheck_1.borrow().data.len(),
            "Inner product sumcheck: both sumchecks must have the same data length"
        );

        assert_eq!(
            sumcheck_0.borrow().data[0].representation,
            sumcheck_1.borrow().data[0].representation,
            "Inner product sumcheck: both sumchecks must have the same representation"
        );

        assert_eq!(
            sumcheck_0.borrow().get_variable_count(),
            sumcheck_1.borrow().get_variable_count(),
            "Inner product sumcheck: both sumchecks must have the same variable count"
        );

        let rep = sumcheck_0.borrow().data[0].representation;

        InnerProductSumcheck {
            sumcheck_0,
            sumcheck_1,
        }
    }
}

struct QuadraticPolynomial {
    pub coefficients: [RingElement; 3],
}

impl QuadraticPolynomial {
    fn new(representation: Representation) -> Self {
        QuadraticPolynomial {
            coefficients: [
                RingElement::zero(representation),
                RingElement::zero(representation),
                RingElement::zero(representation),
            ],
        }
    }
}

impl Polynomial for QuadraticPolynomial {
    fn at_zero(&self) -> RingElement {
        self.coefficients[0].clone()
    }

    fn at_one(&self) -> RingElement {
        // TODO: optimize memory allocations here
        &(&self.coefficients[0] + &self.coefficients[1]) + &self.coefficients[2]
    }

    fn at(&self, x: &RingElement) -> RingElement {
        // TODO: optimize memory allocations here
        let x_squared = x * x;
        &(&self.coefficients[0] + &(x * &self.coefficients[1]))
            + &(&x_squared * &self.coefficients[2])
    }
}

impl HighOrderSumcheckData<QuadraticPolynomial, (RingElement, RingElement)>
    for InnerProductSumcheck<'_>
{
    fn univariate_polynomial_into(&self, polynomial: &mut QuadraticPolynomial) {
        polynomial.coefficients[0].set_zero();
        polynomial.coefficients[1].set_zero();
        polynomial.coefficients[2].set_zero();

        // TODO: optimize memory allocations here
        let mut a_diff = RingElement::zero(self.sumcheck_0.borrow().data[0].representation);
        let mut b_diff = RingElement::zero(self.sumcheck_0.borrow().data[0].representation);
        let mut prod = RingElement::zero(self.sumcheck_0.borrow().data[0].representation);

        let sc0 = self.sumcheck_0.borrow();
        let sc1 = self.sumcheck_1.borrow();
        let half = sc0.data.len() / 2;
        for i in 0..half {
            let a0 = &sc0.data[i];
            let a1 = &sc0.data[i + half];
            let b0 = &sc1.data[i];
            let b1 = &sc1.data[i + half];

            // a_diff = A(1) - A(0); b_diff = B(1) - B(0)
            a_diff.set_from(a1);
            a_diff -= a0;
            b_diff.set_from(b1);
            b_diff -= b0;

            // x^2 term: a_diff * b_diff
            prod.set_from(&a_diff);
            prod *= &b_diff;
            polynomial.coefficients[2] += &prod;

            // x term: a_diff * b0 + b_diff * a0
            prod.set_from(&a_diff);
            prod *= b0;
            polynomial.coefficients[1] += &prod;
            prod.set_from(&b_diff);
            prod *= a0;
            polynomial.coefficients[1] += &prod;

            // constant term: a0 * b0
            prod.set_from(a0);
            prod *= b0;
            polynomial.coefficients[0] += &prod;
        }
    }
}

#[test]
fn test_inner_product_sumcheck() {
    let data_0 = vec![
        RingElement::constant(1, Representation::IncompleteNTT),
        RingElement::constant(2, Representation::IncompleteNTT),
        RingElement::constant(3, Representation::IncompleteNTT),
        RingElement::constant(4, Representation::IncompleteNTT),
        RingElement::constant(5, Representation::IncompleteNTT),
        RingElement::constant(6, Representation::IncompleteNTT),
        RingElement::constant(7, Representation::IncompleteNTT),
        RingElement::constant(8, Representation::IncompleteNTT),
    ];

    let data_1 = vec![
        RingElement::constant(9, Representation::IncompleteNTT),
        RingElement::constant(10, Representation::IncompleteNTT),
        RingElement::constant(11, Representation::IncompleteNTT),
        RingElement::constant(12, Representation::IncompleteNTT),
        RingElement::constant(13, Representation::IncompleteNTT),
        RingElement::constant(14, Representation::IncompleteNTT),
        RingElement::constant(15, Representation::IncompleteNTT),
        RingElement::constant(16, Representation::IncompleteNTT),
    ];

    let sumcheck_0 = RefCell::new(LinearSumcheck::new(data_0.len(), data_0[0].representation));
    sumcheck_0.borrow_mut().from(&data_0);
    let sumcheck_1 = RefCell::new(LinearSumcheck::new(data_1.len(), data_1[0].representation));
    sumcheck_1.borrow_mut().from(&data_1);

    let inner_product_sumcheck = InnerProductSumcheck::new(&sumcheck_0, &sumcheck_1);

    let mut univariate_poly = QuadraticPolynomial::new(Representation::IncompleteNTT);

    inner_product_sumcheck.univariate_polynomial_into(&mut univariate_poly);

    assert_eq!(
        &univariate_poly.at_zero() + &univariate_poly.at_one(),
        RingElement::constant(
            1 * 9 + 2 * 10 + 3 * 11 + 4 * 12 + 5 * 13 + 6 * 14 + 7 * 15 + 8 * 16,
            Representation::IncompleteNTT
        )
    );

    let r0 = RingElement::constant(524, Representation::IncompleteNTT);

    let claim = univariate_poly.at(&r0);

    sumcheck_0.borrow_mut().partial_evaluate(&r0);
    sumcheck_1.borrow_mut().partial_evaluate(&r0);

    inner_product_sumcheck.univariate_polynomial_into(&mut univariate_poly);

    assert_eq!(
        &univariate_poly.at_zero() + &univariate_poly.at_one(),
        claim
    );

    let r1 = RingElement::constant(1337, Representation::IncompleteNTT);

    let claim = univariate_poly.at(&r1);

    sumcheck_0.borrow_mut().partial_evaluate(&r1);
    sumcheck_1.borrow_mut().partial_evaluate(&r1);

    inner_product_sumcheck.univariate_polynomial_into(&mut univariate_poly);

    assert_eq!(
        &univariate_poly.at_zero() + &univariate_poly.at_one(),
        claim
    );

    let r2 = RingElement::constant(42, Representation::IncompleteNTT);

    let claim = univariate_poly.at(&r2);

    sumcheck_0.borrow_mut().partial_evaluate(&r2);
    sumcheck_1.borrow_mut().partial_evaluate(&r2);

    assert_eq!(
        sumcheck_0.borrow().final_evaluations() * sumcheck_1.borrow().final_evaluations(),
        claim,
    );

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
