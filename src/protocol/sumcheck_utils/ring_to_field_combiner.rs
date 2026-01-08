use std::cell::RefCell;

use crate::{
    common::{
        config::HALF_DEGREE,
        ring_arithmetic::{QuadraticExtension, Representation, RingElement, SHIFT_FACTORS},
        sumcheck_element::SumcheckElement,
    },
    protocol::sumcheck::{
        common::{HighOrderSumcheckData, SumcheckBaseData},
        linear::LinearSumcheck,
        polynomial::Polynomial,
    },
};

pub struct RingToFieldCombiner<'a> {
    sumcheck: &'a RefCell<dyn HighOrderSumcheckData<Element = RingElement> + 'a>,
    challenge_vec: [QuadraticExtension; HALF_DEGREE],
    temp_poly: RefCell<Polynomial<RingElement>>,
    scratch_poly: RefCell<Polynomial<QuadraticExtension>>,
}

impl<'a> RingToFieldCombiner<'a> {
    pub fn new(
        sumcheck: &'a RefCell<dyn HighOrderSumcheckData<Element = RingElement> + 'a>,
    ) -> Self {
        Self {
            sumcheck,
            challenge_vec: [QuadraticExtension::zero(); HALF_DEGREE],
            scratch_poly: RefCell::new(Polynomial::new(0)),
            temp_poly: RefCell::new(Polynomial::new(0)),
        }
    }

    fn load_challenges(&mut self, challenge: [QuadraticExtension; HALF_DEGREE]) {
        self.challenge_vec = challenge;
    }
}

impl<'a> HighOrderSumcheckData for RingToFieldCombiner<'a> {
    type Element = QuadraticExtension;

    fn max_num_polynomial_coefficients(&self) -> usize {
        self.sumcheck.borrow().max_num_polynomial_coefficients()
    }

    fn variable_count(&self) -> usize {
        self.sumcheck.borrow().variable_count()
    }

    fn get_scratch_poly(&self) -> &RefCell<Polynomial<Self::Element>> {
        &self.scratch_poly
    }

    fn univariate_polynomial_at_point_into(
        &self,
        point: super::hypercube_point::HypercubePoint, // this is just the usize so we pass it by value
        polynomial: &mut Polynomial<Self::Element>,
    ) {
        let temp = &mut self.temp_poly.borrow_mut();
        self.sumcheck
            .borrow()
            .univariate_polynomial_at_point_into(point, temp);

        polynomial.set_zero();
        for i in 0..temp.num_coefficients {
            temp.coefficients[i].from_incomplete_ntt_to_homogenized_field_extensions();
            let mut coeff = temp.coefficients[i].split_into_quadratic_extensions();
            for j in 0..HALF_DEGREE {
                coeff[j] *= &self.challenge_vec[j];
                polynomial.coefficients[i] += &coeff[j];
            }
            // this will be zeroed anyway so no need to keep it in the final representation
            temp.coefficients[i].representation = Representation::IncompleteNTT;
        }
        polynomial.num_coefficients = temp.num_coefficients;
    }

    fn is_univariate_polynomial_zero_at_point(
        &self,
        point: super::hypercube_point::HypercubePoint,
    ) -> bool {
        false
    }
}

#[test]
fn test_ring_to_field_combiner() {
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

    let sumcheck = RefCell::new(LinearSumcheck::<RingElement>::new(data.len()));
    sumcheck.borrow_mut().load_from(&data);

    let mut challenge_qe = vec![];
    for i in 0..HALF_DEGREE {
        challenge_qe.push(QuadraticExtension {
            coeffs: [i as u64 + 1, 0],
            shift: SHIFT_FACTORS[0],
        });
    }

    let mut combiner = RingToFieldCombiner::new(&sumcheck);

    combiner.load_challenges(challenge_qe.try_into().unwrap());

    let claim = (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) * (HALF_DEGREE + 1) * (HALF_DEGREE) / 2;

    let mut poly = Polynomial::<QuadraticExtension>::new(0);

    combiner.univariate_polynomial_into(&mut poly);

    assert_eq!(
        poly.at_zero() + poly.at_one(),
        QuadraticExtension {
            coeffs: [claim as u64, 0],
            shift: SHIFT_FACTORS[0],
        }
    );

    let r0qe = QuadraticExtension {
        coeffs: [7, 3],
        shift: SHIFT_FACTORS[0],
    };

    let mut r0 = RingElement::constant(0, Representation::HomogenizedFieldExtensions);

    r0.combine_from_quadratic_extensions(&[r0qe; HALF_DEGREE]);

    r0.from_homogenized_field_extensions_to_incomplete_ntt();

    let claim_after_r0 = poly.at(&r0qe);

    sumcheck.borrow_mut().partial_evaluate(&r0);
    combiner.univariate_polynomial_into(&mut poly);

    assert_eq!(poly.at_zero() + poly.at_one(), claim_after_r0);

    let r1qe = QuadraticExtension {
        coeffs: [21, 37],
        shift: SHIFT_FACTORS[0],
    };

    let mut r1 = RingElement::constant(0, Representation::HomogenizedFieldExtensions);

    r1.combine_from_quadratic_extensions(&[r1qe; HALF_DEGREE]);

    r1.from_homogenized_field_extensions_to_incomplete_ntt();

    let claim_after_r1 = poly.at(&r1qe);

    sumcheck.borrow_mut().partial_evaluate(&r1);
    combiner.univariate_polynomial_into(&mut poly);

    assert_eq!(poly.at_zero() + poly.at_one(), claim_after_r1);

    let r2qe = QuadraticExtension {
        coeffs: [53, 89],
        shift: SHIFT_FACTORS[0],
    };

    let mut r2 = RingElement::constant(0, Representation::HomogenizedFieldExtensions);

    r2.combine_from_quadratic_extensions(&[r2qe; HALF_DEGREE]);

    r2.from_homogenized_field_extensions_to_incomplete_ntt();

    let final_claim = poly.at(&r2qe);

    sumcheck.borrow_mut().partial_evaluate(&r2);

    // this API is bit awkward

    let mut final_qe = sumcheck.borrow().final_evaluations().clone();

    final_qe.from_incomplete_ntt_to_homogenized_field_extensions();

    let mut final_qes = final_qe.split_into_quadratic_extensions();
    let mut final_eval = QuadraticExtension::zero();
    for i in 0..HALF_DEGREE {
        final_qes[i] *= &combiner.challenge_vec[i];
        final_eval += &final_qes[i];
    }
    assert_eq!(final_eval, final_claim);
}
