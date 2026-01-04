use std::{cell::RefCell, cmp::max};

use crate::{
    common::ring_arithmetic::{Representation, RingElement},
    protocol::sumcheck::{
        common::{HighOrderSumcheckData, SumcheckBaseData},
        hypercube_point::HypercubePoint,
        linear::LinearSumcheck,
        polynomial::{add_poly_in_place, Polynomial},
    },
};

pub struct DiffSumcheck<'a> {
    pub sumcheck_0: &'a RefCell<dyn HighOrderSumcheckData + 'a>,
    pub sumcheck_1: &'a RefCell<dyn HighOrderSumcheckData + 'a>,
}

impl DiffSumcheck<'_> {
    pub fn new<'a>(
        sumcheck_0: &'a RefCell<dyn HighOrderSumcheckData + 'a>,
        sumcheck_1: &'a RefCell<dyn HighOrderSumcheckData + 'a>,
    ) -> DiffSumcheck<'a> {
        assert_eq!(
            sumcheck_0.borrow().variable_count(),
            sumcheck_1.borrow().variable_count(),
            "Diff sumcheck: both sumchecks must have the same variable count"
        );

        DiffSumcheck {
            sumcheck_0,
            sumcheck_1,
        }
    }
}

impl HighOrderSumcheckData for DiffSumcheck<'_> {
    fn nof_polynomial_coefficients(&self) -> usize {
        max(
            self.sumcheck_0.borrow().nof_polynomial_coefficients(),
            self.sumcheck_1.borrow().nof_polynomial_coefficients(),
        )
    }
    fn variable_count(&self) -> usize {
        self.sumcheck_0.borrow().variable_count()
    }

    fn univariate_polynomial_at_point_into(
        &self,
        point: HypercubePoint,
        polynomial: &mut Polynomial,
    ) -> bool {
        polynomial.set_zero();

        // TODO: optimize allocations
        let mut temp_poly_0 = Polynomial::new(0, Representation::IncompleteNTT);
        let mut temp_poly_1 = Polynomial::new(0, Representation::IncompleteNTT);

        self.sumcheck_0
            .borrow()
            .univariate_polynomial_at_point_into(point, &mut temp_poly_0);

        self.sumcheck_1
            .borrow()
            .univariate_polynomial_at_point_into(point, &mut temp_poly_1);

        let max_deg = std::cmp::max(temp_poly_0.nof_coefficients, temp_poly_1.nof_coefficients);
        for i in 0..max_deg {
            polynomial.coefficients[i] =
                &temp_poly_0.coefficients[i] - &temp_poly_1.coefficients[i];
        }
        polynomial.nof_coefficients = max_deg;

        true
    }
}

#[test]
fn test_diff_sumcheck_basic() {
    let data_0 = vec![
        RingElement::constant(8, Representation::IncompleteNTT),
        RingElement::constant(7, Representation::IncompleteNTT),
        RingElement::constant(6, Representation::IncompleteNTT),
        RingElement::constant(5, Representation::IncompleteNTT),
    ];

    let data_1 = vec![
        RingElement::constant(1, Representation::IncompleteNTT),
        RingElement::constant(2, Representation::IncompleteNTT),
        RingElement::constant(3, Representation::IncompleteNTT),
        RingElement::constant(4, Representation::IncompleteNTT),
    ];

    let sumcheck_0 = RefCell::new(LinearSumcheck::new(data_0.len(), data_0[0].representation));
    sumcheck_0.borrow_mut().from(&data_0);
    let sumcheck_1 = RefCell::new(LinearSumcheck::new(data_1.len(), data_1[0].representation));
    sumcheck_1.borrow_mut().from(&data_1);

    let diff_sumcheck = DiffSumcheck::new(&sumcheck_0, &sumcheck_1);

    let mut poly = Polynomial::new(0, data_0[0].representation);
    diff_sumcheck.univariate_polynomial_into(&mut poly);

    // Sum(data_0) - Sum(data_1) = 26 - 10 = 16
    let claim = RingElement::constant(16, Representation::IncompleteNTT);
    assert_eq!(&poly.at_zero() + &poly.at_one(), claim);

    // Check evaluation at a random point stays consistent.
    let r0 = RingElement::constant(5, Representation::IncompleteNTT);
    let claim_r0 = poly.at(&r0);
    sumcheck_0.borrow_mut().partial_evaluate(&r0);
    sumcheck_1.borrow_mut().partial_evaluate(&r0);
    diff_sumcheck.univariate_polynomial_into(&mut poly);
    assert_eq!(&poly.at_zero() + &poly.at_one(), claim_r0);
}
