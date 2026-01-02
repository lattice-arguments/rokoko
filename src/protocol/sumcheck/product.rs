use std::cell::RefCell;

use crate::{
    common::{
        config::MOD_Q,
        ring_arithmetic::{Representation, RingElement},
    },
    protocol::sumcheck::{
        self,
        common::{HighOrderSumcheckData, SumcheckBaseData},
        hypercube_point::HypercubePoint,
        linear::LinearSumcheck,
        polynomial::{add_poly_in_place, add_poly_into, mul_poly_into, Polynomial},
    },
};

pub struct ProductSumcheck<'a> {
    pub sumcheck_0: &'a RefCell<dyn HighOrderSumcheckData + 'a>, // interior mutability to share between protocols
    pub sumcheck_1: &'a RefCell<dyn HighOrderSumcheckData + 'a>,
}

impl ProductSumcheck<'_> {
    pub fn new<'a>(
        sumcheck_0: &'a RefCell<dyn HighOrderSumcheckData + 'a>,
        sumcheck_1: &'a RefCell<dyn HighOrderSumcheckData + 'a>,
    ) -> ProductSumcheck<'a> {
        assert_eq!(
            sumcheck_0.borrow().get_variable_count(),
            sumcheck_1.borrow().get_variable_count(),
            "Inner product sumcheck: both sumchecks must have the same data length"
        );

        ProductSumcheck {
            sumcheck_0,
            sumcheck_1,
        }
    }
}

impl HighOrderSumcheckData for ProductSumcheck<'_> {
    fn get_variable_count(&self) -> usize {
        self.sumcheck_0.borrow().get_variable_count()
    }
    fn univariate_polynomial_into(&self, polynomial: &mut Polynomial) {
        polynomial.set_zero();
        let mut temp_poly_0 = Polynomial::new(0, Representation::IncompleteNTT);

        let n = self.get_variable_count();
        let len = 1 << n;
        let half = len / 2;
        for i in 0..half {
            let point = HypercubePoint::new(i);
            self.univariate_polynomial_at_point_into(&point, &mut temp_poly_0);
            add_poly_in_place(polynomial, &temp_poly_0);
        }
    }

    fn univariate_polynomial_at_point_into(
        &self,
        point: &HypercubePoint,
        polynomial: &mut Polynomial,
    ) {
        // reset accumulator for this point
        polynomial.set_zero();
        polynomial.nof_coefficients = 0;

        let mut temp_poly_0 = Polynomial::new(0, Representation::IncompleteNTT);
        let mut temp_poly_1 = Polynomial::new(0, Representation::IncompleteNTT);

        self.sumcheck_0
            .borrow()
            .univariate_polynomial_at_point_into(point, &mut temp_poly_0);

        self.sumcheck_1
            .borrow()
            .univariate_polynomial_at_point_into(point, &mut temp_poly_1);

        mul_poly_into(polynomial, &temp_poly_0, &temp_poly_1);
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

    let inner_product_sumcheck = ProductSumcheck::new(&sumcheck_0, &sumcheck_1);

    let mut univariate_poly = Polynomial::new(0, data_0[0].representation);

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

    let sumcheck = RefCell::new(LinearSumcheck::new(data.len(), data[0].representation));
    sumcheck.borrow_mut().from(&data);

    let inner_product_sumcheck = ProductSumcheck::new(&sumcheck, &sumcheck);

    let mut univariate_poly = Polynomial::new(0, data[0].representation);

    inner_product_sumcheck.univariate_polynomial_into(&mut univariate_poly);

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

    let mut sumcheck_0 = LinearSumcheck::new(data0.len(), data0[0].representation);
    sumcheck_0.from(&data0);
    let mut sumcheck_1 = LinearSumcheck::new(data1.len(), data1[0].representation);
    sumcheck_1.from(&data1);
    let mut sumcheck_2 = LinearSumcheck::new(data2.len(), data2[0].representation);
    sumcheck_2.from(&data2);

    let sumcheck_0_ref = RefCell::new(sumcheck_0);
    let sumcheck_1_ref = RefCell::new(sumcheck_1);
    let sumcheck_2_ref = RefCell::new(sumcheck_2);

    // let sumcheck0 = RefCell::new(LinearSumcheck::new(data0.len(), data0[0].representation));
    // sumcheck0.borrow_mut().from(&data0);
    // let sumcheck1 = RefCell::new(LinearSumcheck::new(data1.len(), data1[0].representation));
    // sumcheck1.borrow_mut().from(&data1);
    // let sumcheck2 = RefCell::new(LinearSumcheck::new(data2.len(), data2[0].representation));
    // sumcheck2.borrow_mut().from(&data2);

    let inner_product_sumcheck_01 = ProductSumcheck::new(&sumcheck_0_ref, &sumcheck_1_ref);

    let inner_product_sumcheck_01_ref = RefCell::new(inner_product_sumcheck_01);

    let inner_product_sumcheck_012 =
        ProductSumcheck::new(&inner_product_sumcheck_01_ref, &sumcheck_2_ref);

    let mut univariate_poly = Polynomial::new(0, data0[0].representation);
    inner_product_sumcheck_012.univariate_polynomial_into(&mut univariate_poly);

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

    assert_eq!(
        &univariate_poly.at_zero() + &univariate_poly.at_one(),
        claim
    );
}
