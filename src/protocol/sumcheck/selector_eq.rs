use std::cell::RefCell;

use crate::{
    common::ring_arithmetic::{Representation, RingElement},
    protocol::sumcheck::{
        common::{HighOrderSumcheckData, SumcheckBaseData},
        hypercube_point::HypercubePoint,
        polynomial::Polynomial,
    },
};

#[cfg(test)]
use crate::common::config::MOD_Q;

/// Sumcheck for the multilinear equality check `eq(x, selector)`.
/// It evaluates to 1 exactly when the first `selector_variable_count` bits of
/// the query match `selector`, while ignoring additional trailing variables.
pub struct SelectorEq {
    selector: usize,
    selector_variable_count: usize,
    total_variable_count: usize,

    // if we do partial evaluation, we have to store the current claim
    current_claim: RingElement,
    temp_product: RefCell<RingElement>,
    scratch_poly: RefCell<Polynomial>,
}

// Selector is a sumcheck that of eq(x, s) over all x in {0,1}^n
// where s is the selector

// we allow that nof_variables can be larger than the number of bits in selector. Selector specifies only the higher order bits.

// E.g. selector = 0b10, nof_variables = 4 means that
// f(x_0, ... x_3) = eq(x0 x1, 1 0)
// Alternatively, this SelectorEq is functionally equivalent to linear sumcheck over vector (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0)
// However, this implementation is more efficient in space and time.
// as univariate_polynomial_at_point_into returns `false` unless the point matches the selector in the higher order bits.
// This is useful for implementing sumcheck over sparse vectors.
impl SelectorEq {
    pub fn new(
        selector: usize,
        selector_variable_count: usize,
        total_variable_count: usize,
    ) -> Self {
        SelectorEq {
            selector,
            selector_variable_count,
            total_variable_count,
            current_claim: RingElement::one(Representation::IncompleteNTT),
            temp_product: RefCell::new(RingElement::zero(Representation::IncompleteNTT)),
            scratch_poly: RefCell::new(Polynomial::new(2, Representation::IncompleteNTT)),
        }
    }
}

impl HighOrderSumcheckData for SelectorEq {
    fn get_scratch_poly(&self) -> &RefCell<Polynomial> {
        &self.scratch_poly
    }
    fn num_polynomial_coefficients(&self) -> usize {
        2
    }
    fn variable_count(&self) -> usize {
        self.total_variable_count
    }

    fn univariate_polynomial_at_point_into(
        &self,
        point: HypercubePoint,
        polynomial: &mut Polynomial,
    ) -> bool {
        polynomial.set_zero();
        polynomial.num_coefficients = 2;

        if self.selector_variable_count == 0 {
            // now we have a constrant function which is always 1
            polynomial.coefficients[0] += &self.current_claim;
            return true;
        }

        // We evaluate from the highest order bit to the lowest order bit
        // Therefore, we check if the higher order bits of point match the selector (expect for the current variable being evaluated being the highest order bit)

        let point_higher_bits =
            point.shifted(self.total_variable_count - self.selector_variable_count);

        let selector_bits = self.selector & ((1 << self.selector_variable_count - 1) - 1); // mask to get only the relevant bits

        if point_higher_bits.coordinates != selector_bits {
            // the polynomial is identically zero
            return false;
        }

        let current_bit = (self.selector >> (self.selector_variable_count - 1)) & 1;

        if current_bit == 1 {
            // then we have a function which is 1 when the variable is 1, and 0 when the variable is 0. So this is an identity
            polynomial.coefficients[1] += &self.current_claim;
        } else {
            // then we have a function which is 1 when the variable is 0, and 0 when the variable is 1. So this is (1 - x)
            polynomial.coefficients[0] += &self.current_claim;
            polynomial.coefficients[1] -= &self.current_claim;
        }

        true
    }
}

impl SumcheckBaseData for SelectorEq {
    fn partial_evaluate(&mut self, value: &RingElement) {
        // e.g. if selector = 0b110, selector_variable_count = 3, total_variable_count = 5
        // then bit_index = 0b1 (we look at the highest order bit of the selector)
        if self.selector_variable_count > 0 {
            let bit_index = self.selector >> (self.selector_variable_count - 1);

            if bit_index & 1 == 1 {
                // then we have a function which is 1 when the variable is 1, and 0 when the variable is 0. So this is an identity
                self.current_claim *= value;
            } else {
                // then we have a function which is 1 when the variable is 0, and 0 when the variable is 1. So this is (1 - x)

                let mut temp = self.temp_product.borrow_mut();
                *temp *= (value, &self.current_claim); // temp = claim_factor * value
                self.current_claim -= &*temp;
            }
            self.selector &= (1 << (self.selector_variable_count - 1)) - 1;
            self.selector_variable_count -= 1;

            // } else {
            // now we have a constrant function which is always 1, so the claim remains the same
        }
        self.total_variable_count -= 1;
    }

    fn final_evaluations(&self) -> &RingElement {
        if self.total_variable_count != 0 {
            panic!("final_evaluations called before all variables were evaluated");
        }
        &self.current_claim
    }
}

#[test]
fn test_selector_eq_basic() {
    let selector = 0b10;
    let selector_variable_count = 2;
    let total_variable_count = 4;

    // this can be viewed as a sumcheck over the vector (0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0)
    let mut sumcheck = SelectorEq::new(selector, selector_variable_count, total_variable_count);
    let mut polynomial = Polynomial::new(2, Representation::IncompleteNTT);

    // Irrelevant points should produce an identically-zero polynomial.
    let result =
        sumcheck.univariate_polynomial_at_point_into(HypercubePoint::new(0b100), &mut polynomial);

    // 0b0100 and 0b1100 do not match the selector in the higher order bits, so the polynomial is identically zero
    assert_eq!(result, false);

    let result =
        sumcheck.univariate_polynomial_at_point_into(HypercubePoint::new(0b101), &mut polynomial);

    // 0b0101 and 0b1101 do not match the selector in the higher order bits, so the polynomial is identically zero
    assert_eq!(result, false);

    let result =
        sumcheck.univariate_polynomial_at_point_into(HypercubePoint::new(0b010), &mut polynomial);

    // 0b1010 matches the selector in the higher order bits
    assert_eq!(result, true);

    // as selector = 0b10, the polynomial should be x as it's 1 when the variable is 1, and 0 when the variable is 0
    assert_eq!(
        polynomial.coefficients[0],
        RingElement::zero(Representation::IncompleteNTT)
    );
    assert_eq!(
        polynomial.coefficients[1],
        RingElement::one(Representation::IncompleteNTT)
    );

    // There are exactly four satisfying assignments for the selector bits.
    let mut claim = RingElement::constant(4, Representation::IncompleteNTT);

    sumcheck.univariate_polynomial_into(&mut polynomial);

    assert_eq!(&polynomial.at_zero() + &polynomial.at_one(), claim);

    let r0 = RingElement::constant(53, Representation::IncompleteNTT);

    // Folding the highest bit turns the indicator vector into four copies of r0.
    sumcheck.partial_evaluate(&r0);

    claim = polynomial.at(&r0);

    // (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0)
    // changes into (r0, r0, r0, r0, 0, 0, 0, 0) after partial evaluation at r0
    assert_eq!(
        claim,
        RingElement::constant(4 * 53, Representation::IncompleteNTT)
    );

    sumcheck.univariate_polynomial_into(&mut polynomial);

    assert_eq!(&polynomial.at_zero() + &polynomial.at_one(), claim);

    let r1 = RingElement::constant(73, Representation::IncompleteNTT);

    sumcheck.partial_evaluate(&r1);

    claim = polynomial.at(&r1);

    // (r0, r0, r0, r0, 0, 0, 0, 0) // at 0 it's r0, at 1 it's 0 so the funtion is r0 * (1 - x)
    // changes into (r0 * (1 - r1), r0 * (1 - r1), r0 * (1 - r1), r0 * (1 - r1)) after partial evaluation at r1

    assert_eq!(
        claim,
        RingElement::constant(
            4 * 53 * (MOD_Q as i64 + 1 - 73) as u64,
            Representation::IncompleteNTT
        )
    );

    sumcheck.univariate_polynomial_into(&mut polynomial);

    assert_eq!(&polynomial.at_zero() + &polynomial.at_one(), claim);

    assert_eq!(
        polynomial.coefficients[1],
        RingElement::zero(Representation::IncompleteNTT)
    );
    // after partial evaluation at r1, the function is constant, so the coeff of x is 0

    let r2 = RingElement::constant(19, Representation::IncompleteNTT);

    sumcheck.partial_evaluate(&r2);

    claim = polynomial.at(&r2);

    // (r0 * (1 - r1), r0 * (1 - r1), r0 * (1 - r1), r0 * (1 - r1))
    // changes into (r0 * (1 - r1), r0 * (1 - r1)) after partial evaluation at r2 (as the function is constant and the variable is ignored)

    assert_eq!(
        claim,
        RingElement::constant(
            2 * 53 * (MOD_Q as i64 + 1 - 73) as u64,
            Representation::IncompleteNTT
        )
    );

    sumcheck.univariate_polynomial_into(&mut polynomial);

    assert_eq!(&polynomial.at_zero() + &polynomial.at_one(), claim);

    let r3 = RingElement::constant(743, Representation::IncompleteNTT);

    sumcheck.partial_evaluate(&r3);

    claim = polynomial.at(&r3);

    // (r0 * (1 - r1), r0 * (1 - r1))
    // changes into (r0 * (1 - r1)) after partial evaluation at r3 (as the function is constant and the variable is ignored)

    assert_eq!(
        claim,
        RingElement::constant(
            53 * (MOD_Q as i64 + 1 - 73) as u64,
            Representation::IncompleteNTT
        )
    );

    // After exhausting all variables, the stored claim should match the verifier's expectation.
    assert_eq!(sumcheck.final_evaluations(), &claim);
}
