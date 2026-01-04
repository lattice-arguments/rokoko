use std::{cell::RefCell, cmp::max};

use crate::{
    common::ring_arithmetic::Representation,
    protocol::sumcheck::{
        common::HighOrderSumcheckData,
        hypercube_point::HypercubePoint,
        polynomial::{add_poly_in_place, sub_poly_in_place, Polynomial},
    },
};

#[cfg(test)]
use crate::{
    common::ring_arithmetic::RingElement,
    protocol::sumcheck::{common::SumcheckBaseData, linear::LinearSumcheck},
};

/// Sumcheck data that represents the difference between two other sumchecks.
/// Useful for enforcing equality constraints between two multilinear extensions.
pub struct DiffSumcheck<'a> {
    pub lhs_sumcheck: &'a RefCell<dyn HighOrderSumcheckData + 'a>,
    pub rhs_sumcheck: &'a RefCell<dyn HighOrderSumcheckData + 'a>,

    lhs_eval_poly: RefCell<Polynomial>,
    rhs_eval_poly: RefCell<Polynomial>,
    scratch_poly: RefCell<Polynomial>,
}

impl DiffSumcheck<'_> {
    pub fn new<'a>(
        lhs_sumcheck: &'a RefCell<dyn HighOrderSumcheckData + 'a>,
        rhs_sumcheck: &'a RefCell<dyn HighOrderSumcheckData + 'a>,
    ) -> DiffSumcheck<'a> {
        assert_eq!(
            lhs_sumcheck.borrow().variable_count(),
            rhs_sumcheck.borrow().variable_count(),
            "Diff sumcheck: both sumchecks must have the same variable count"
        );

        DiffSumcheck {
            lhs_sumcheck,
            rhs_sumcheck,
            lhs_eval_poly: RefCell::new(Polynomial::new(0, Representation::IncompleteNTT)),
            rhs_eval_poly: RefCell::new(Polynomial::new(0, Representation::IncompleteNTT)),
            scratch_poly: RefCell::new(Polynomial::new(0, Representation::IncompleteNTT)),
        }
    }
}

impl HighOrderSumcheckData for DiffSumcheck<'_> {
    fn get_scratch_poly(&self) -> &RefCell<Polynomial> {
        &self.scratch_poly
    }
    fn num_polynomial_coefficients(&self) -> usize {
        max(
            self.lhs_sumcheck.borrow().num_polynomial_coefficients(),
            self.rhs_sumcheck.borrow().num_polynomial_coefficients(),
        )
    }
    fn variable_count(&self) -> usize {
        self.lhs_sumcheck.borrow().variable_count()
    }

    fn univariate_polynomial_at_point_into(
        &self,
        point: HypercubePoint,
        polynomial: &mut Polynomial,
    ) -> bool {
        // Compute the per-round polynomial as the difference of the two inputs.
        polynomial.set_zero();

        let mut lhs_eval_poly = self.lhs_eval_poly.borrow_mut();
        let mut rhs_eval_poly = self.rhs_eval_poly.borrow_mut();

        self.lhs_sumcheck
            .borrow()
            .univariate_polynomial_at_point_into(point, &mut lhs_eval_poly);

        self.rhs_sumcheck
            .borrow()
            .univariate_polynomial_at_point_into(point, &mut rhs_eval_poly);

        add_poly_in_place(polynomial, &lhs_eval_poly);
        sub_poly_in_place(polynomial, &rhs_eval_poly);

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

    let sumcheck_0 = RefCell::new(LinearSumcheck::new(data_0.len()));
    sumcheck_0.borrow_mut().load_from(&data_0);
    let sumcheck_1 = RefCell::new(LinearSumcheck::new(data_1.len()));
    sumcheck_1.borrow_mut().load_from(&data_1);

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
