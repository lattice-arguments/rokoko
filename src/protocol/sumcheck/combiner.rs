use std::cell::RefCell;

use crate::{
    common::{
        config::MOD_Q,
        ring_arithmetic::{Representation, RingElement},
        structured_row::PreprocessedRow,
    },
    protocol::{
        open::evaluation_point_to_structured_row,
        sumcheck::{
            common::{HighOrderSumcheckData, SumcheckBaseData},
            hypercube_point::HypercubePoint,
            linear::LinearSumcheck,
            polynomial::Polynomial,
        },
    },
};

pub struct Combiner<'a> {
    sumchecks: Vec<&'a RefCell<dyn HighOrderSumcheckData + 'a>>,
    challenges: PreprocessedRow, // for now let's batch in ring elements
    temp_poly: RefCell<Polynomial>,
    scratch_poly: RefCell<Polynomial>,
}

impl<'a> Combiner<'a> {
    pub fn new(
        sumchecks: Vec<&'a RefCell<dyn HighOrderSumcheckData + 'a>>,
        challenges: PreprocessedRow,
    ) -> Self {
        assert_eq!(
            sumchecks.len().next_power_of_two(),
            challenges.preprocessed_row.len()
        );
        // assert all variables counts are the same
        let var_count = sumchecks[0].borrow().variable_count();
        for sumcheck in &sumchecks {
            assert_eq!(sumcheck.borrow().variable_count(), var_count);
        }
        Self {
            sumchecks,
            challenges,
            scratch_poly: RefCell::new(super::polynomial::Polynomial::new(
                0,
                Representation::IncompleteNTT,
            )),
            temp_poly: RefCell::new(super::polynomial::Polynomial::new(
                0,
                Representation::IncompleteNTT,
            )),
        }
    }
}

impl HighOrderSumcheckData for Combiner<'_> {
    fn max_num_polynomial_coefficients(&self) -> usize {
        let mut max_coeffs = 0;
        for sumcheck in &self.sumchecks {
            let sumcheck_borrowed = sumcheck.borrow();
            let coeffs = sumcheck_borrowed.max_num_polynomial_coefficients();
            if coeffs > max_coeffs {
                max_coeffs = coeffs;
            }
        }
        max_coeffs
    }

    fn variable_count(&self) -> usize {
        self.sumchecks[0].borrow().variable_count()
    }

    fn get_scratch_poly(&self) -> &RefCell<super::polynomial::Polynomial> {
        &self.scratch_poly
    }

    fn univariate_polynomial_at_point_into(
        &self,
        point: HypercubePoint, // this is just the usize so we pass it by value
        polynomial: &mut Polynomial,
    ) {
        polynomial.set_zero();
        polynomial.num_coefficients = 0; // will be updated as we add terms

        let mut temp_poly = self.temp_poly.borrow_mut();
        let nof_sumchecks = self.sumchecks.len();
        for i in 0..nof_sumchecks {
            let sumcheck = &self.sumchecks[i];
            let challenge = &self.challenges.preprocessed_row[i];
            sumcheck
                .borrow()
                .univariate_polynomial_at_point_into(point, &mut temp_poly);
            // multiply temp_poly by challenge
            for j in 0..temp_poly.num_coefficients {
                temp_poly.coefficients[j] *= challenge;
            }
            // add to polynomial
            super::polynomial::add_poly_in_place(polynomial, &temp_poly);
        }
    }

    fn is_univariate_polynomial_zero_at_point(
        &self,
        point: super::hypercube_point::HypercubePoint,
    ) -> bool {
        false
    }
}

#[test]
fn test_combiner() {
    let data0 = (0..8)
        .map(|i| RingElement::constant(i + 1 as u64, Representation::IncompleteNTT))
        .collect::<Vec<RingElement>>();

    let data1 = (8..16)
        .map(|i| RingElement::constant((i + 1) as u64, Representation::IncompleteNTT))
        .collect::<Vec<RingElement>>();

    let data2 = (16..24)
        .map(|i| RingElement::constant((i + 1) as u64, Representation::IncompleteNTT))
        .collect::<Vec<RingElement>>();

    let data3 = (24..32)
        .map(|i| RingElement::constant((i + 1) as u64, Representation::IncompleteNTT))
        .collect::<Vec<RingElement>>();

    let combiner = evaluation_point_to_structured_row(&vec![
        RingElement::constant(32, Representation::IncompleteNTT),
        RingElement::constant(33, Representation::IncompleteNTT),
    ]);

    // vector is (1 - 32) * (1 - 33), (1 - 32) * 33, 32 * (1 - 33), 32 * 33

    let preprocessed_challenges = PreprocessedRow::from_structured_row(combiner);

    let mut sumcheck0 = LinearSumcheck::new(data0.len());

    sumcheck0.load_from(&data0);

    let mut sumcheck1 = LinearSumcheck::new(data1.len());
    sumcheck1.load_from(&data1);

    let mut sumcheck2 = LinearSumcheck::new(data2.len());

    sumcheck2.load_from(&data2);

    let mut sumcheck3 = LinearSumcheck::new(data3.len());
    sumcheck3.load_from(&data3);

    let sumcheck0_ref = RefCell::new(sumcheck0);
    let sumcheck1_ref = RefCell::new(sumcheck1);
    let sumcheck2_ref = RefCell::new(sumcheck2);
    let sumcheck3_ref = RefCell::new(sumcheck3);

    let combiner = Combiner::new(
        vec![
            &sumcheck0_ref,
            &sumcheck1_ref,
            &sumcheck2_ref,
            &sumcheck3_ref,
        ],
        preprocessed_challenges,
    );

    let mut poly = Polynomial::new(0, Representation::IncompleteNTT);

    combiner.univariate_polynomial_into(&mut poly);

    // manually compute expected coefficients
    // 1 + 2 + ... + 8 = 36
    // 9 + 10 + ... + 16 = 100
    // 17 + ... + 24 = 164
    // 25 + ... + 32 = 228

    let claim = RingElement::constant(
        (MOD_Q as i64
            + 36 * (1 - 32) * (1 - 33)
            + 100 * (1 - 32) * 33
            + 164 * 32 * (1 - 33)
            + 228 * 32 * 33) as u64,
        Representation::IncompleteNTT,
    );

    assert_eq!(&poly.at_zero() + &poly.at_one(), claim);

    let r0 = RingElement::constant(2, Representation::IncompleteNTT);

    let claim = poly.at(&r0);

    sumcheck0_ref.borrow_mut().partial_evaluate(&r0);
    sumcheck1_ref.borrow_mut().partial_evaluate(&r0);
    sumcheck2_ref.borrow_mut().partial_evaluate(&r0);
    sumcheck3_ref.borrow_mut().partial_evaluate(&r0);

    combiner.univariate_polynomial_into(&mut poly);
    assert_eq!(&poly.at_zero() + &poly.at_one(), claim);
}
