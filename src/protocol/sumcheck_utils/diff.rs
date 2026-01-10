use std::{
    cell::RefCell,
    cmp::max,
    rc::{Rc, Weak},
};

use crate::{
    common::{
        ring_arithmetic::{Representation, RingElement},
        sumcheck_element::SumcheckElement,
    },
    protocol::sumcheck_utils::{
        common::HighOrderSumcheckData,
        hypercube_point::HypercubePoint,
        polynomial::{add_poly_in_place, sub_poly_in_place, Polynomial},
        selector_eq::SelectorEq,
    },
};

#[cfg(test)]
use crate::protocol::sumcheck_utils::{common::SumcheckBaseData, linear::LinearSumcheck};

/// Sumcheck data that represents the difference between two other sumchecks.
/// Useful for enforcing equality constraints between two multilinear extensions.
pub struct DiffSumcheck<E: SumcheckElement = RingElement> {
    pub lhs_sumcheck: Rc<RefCell<dyn HighOrderSumcheckData<Element = E>>>,
    pub rhs_sumcheck: Rc<RefCell<dyn HighOrderSumcheckData<Element = E>>>,

    lhs_eval_poly: RefCell<Polynomial<E>>,
    rhs_eval_poly: RefCell<Polynomial<E>>,
    scratch_poly: RefCell<Polynomial<E>>,
}

impl<E: SumcheckElement> DiffSumcheck<E> {
    pub fn new(
        lhs_sumcheck: Rc<RefCell<dyn HighOrderSumcheckData<Element = E>>>,
        rhs_sumcheck: Rc<RefCell<dyn HighOrderSumcheckData<Element = E>>>,
    ) -> DiffSumcheck<E> {
        assert_eq!(
            lhs_sumcheck.borrow().variable_count(),
            rhs_sumcheck.borrow().variable_count(),
            "Diff sumcheck: both sumchecks must have the same variable count"
        );

        DiffSumcheck {
            lhs_sumcheck,
            rhs_sumcheck,
            lhs_eval_poly: RefCell::new(Polynomial::new(0)),
            rhs_eval_poly: RefCell::new(Polynomial::new(0)),
            scratch_poly: RefCell::new(Polynomial::new(0)),
        }
    }
}

impl<E: SumcheckElement> HighOrderSumcheckData for DiffSumcheck<E> {
    type Element = E;

    fn get_scratch_poly(&self) -> &RefCell<Polynomial<E>> {
        &self.scratch_poly
    }
    fn max_num_polynomial_coefficients(&self) -> usize {
        max(
            self.lhs_sumcheck.borrow().max_num_polynomial_coefficients(),
            self.rhs_sumcheck.borrow().max_num_polynomial_coefficients(),
        )
    }
    fn variable_count(&self) -> usize {
        self.lhs_sumcheck.borrow().variable_count()
    }

    fn is_univariate_polynomial_zero_at_point(&self, point: HypercubePoint) -> bool {
        self.lhs_sumcheck
            .borrow()
            .is_univariate_polynomial_zero_at_point(point)
            && self
                .rhs_sumcheck
                .borrow()
                .is_univariate_polynomial_zero_at_point(point)
    }

    fn univariate_polynomial_at_point_into(
        &self,
        point: HypercubePoint,
        polynomial: &mut Polynomial<E>,
    ) {
        // Compute the per-round polynomial as the difference of the two inputs.
        polynomial.set_zero();

        let mut lhs_eval_poly = self.lhs_eval_poly.borrow_mut();
        let lhs_sumcheck = &self.lhs_sumcheck;
        if !lhs_sumcheck
            .borrow()
            .is_univariate_polynomial_zero_at_point(point)
        {
            lhs_sumcheck
                .borrow()
                .univariate_polynomial_at_point_into(point, &mut lhs_eval_poly);
            add_poly_in_place(polynomial, &lhs_eval_poly);
        }

        let mut rhs_eval_poly = self.rhs_eval_poly.borrow_mut();
        let rhs_sumcheck = &self.rhs_sumcheck;
        if !rhs_sumcheck
            .borrow()
            .is_univariate_polynomial_zero_at_point(point)
        {
            rhs_sumcheck
                .borrow()
                .univariate_polynomial_at_point_into(point, &mut rhs_eval_poly);
            sub_poly_in_place(polynomial, &rhs_eval_poly);
        }
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

    let mut sumcheck_0 = Rc::new(RefCell::new(LinearSumcheck::new(data_0.len())));
    let mut sumcheck_1 = Rc::new(RefCell::new(LinearSumcheck::new(data_1.len())));

    sumcheck_0.borrow_mut().load_from(&data_0);
    sumcheck_1.borrow_mut().load_from(&data_1);

    let diff_sumcheck = DiffSumcheck::new(sumcheck_0.clone(), sumcheck_1.clone());

    let mut poly = Polynomial::new(0);
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

#[test]
fn diff_with_eqs() {
    let lhs = Rc::new(RefCell::new(SelectorEq::<RingElement>::new(0b101, 3, 5)));
    let rhs = Rc::new(RefCell::new(SelectorEq::<RingElement>::new(0b011, 3, 5)));

    let diff = DiffSumcheck::new(lhs.clone(), rhs.clone());
    let claim = RingElement::constant(0, Representation::IncompleteNTT);

    // Initial claim: the difference of the two selectors over the full hypercube is zero.
    // Both selectors are 1 at exactly 4 points, so their difference sums to zero.

    let mut poly = Polynomial::new(0);
    diff.univariate_polynomial_into(&mut poly);

    assert_eq!(&poly.at_zero() + &poly.at_one(), claim);
}
