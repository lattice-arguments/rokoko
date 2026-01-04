use std::ops::Index;

use crate::{
    common::{
        config::MOD_Q,
        ring_arithmetic::{Representation, RingElement},
    },
    protocol::sumcheck::{
        common::{HighOrderSumcheckData, SumcheckBaseData},
        hypercube_point::HypercubePoint,
        polynomial::Polynomial,
    },
};

pub struct SelectorEq {
    selector: usize,
    nof_variables_for_selector: usize,
    nof_variables: usize,

    // if we do partial evaluation, we have to store the current claim
    claim_factor: RingElement,
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
    pub fn new(selector: usize, nof_variables_for_selector: usize, nof_variables: usize) -> Self {
        SelectorEq {
            selector,
            nof_variables_for_selector,
            nof_variables,
            claim_factor: RingElement::one(Representation::IncompleteNTT),
        }
    }
}

impl HighOrderSumcheckData for SelectorEq {
    fn nof_polynomial_coefficients(&self) -> usize {
        2
    }
    fn variable_count(&self) -> usize {
        self.nof_variables
    }

    fn univariate_polynomial_at_point_into(
        &self,
        point: HypercubePoint,
        polynomial: &mut Polynomial,
    ) -> bool {
        polynomial.set_zero();
        polynomial.nof_coefficients = 2;

        if self.nof_variables_for_selector == 0 {
            // now we have a constrant function which is always 1
            polynomial.coefficients[0] += &self.claim_factor;
            return true;
        }

        // We evaluate from the highest order bit to the lowest order bit
        // Therefore, we check if the higher order bits of point match the selector (expect for the current variable being evaluated being the highest order bit)

        let point_higher_bits = point.shifted(self.nof_variables - self.nof_variables_for_selector);

        println!(
            "point_higher_bits: {:b}, selector: {:b}",
            point_higher_bits.coordinates, self.selector
        );

        let selector_bits = self.selector & ((1 << self.nof_variables_for_selector - 1) - 1); // mask to get only the relevant bits

        println!("selector_bits: {:b}", selector_bits);

        if point_higher_bits.coordinates != selector_bits {
            // the polynomial is identically zero
            return false;
        }

        let current_bit = (self.selector >> (self.nof_variables_for_selector - 1)) & 1;

        println!("current_bit: {}", current_bit);
        if current_bit == 1 {
            // then we have a function which is 1 when the variable is 1, and 0 when the variable is 0. So this is an identity
            polynomial.coefficients[1] += &self.claim_factor;
        } else {
            // then we have a function which is 1 when the variable is 0, and 0 when the variable is 1. So this is (1 - x)
            polynomial.coefficients[0] += &self.claim_factor;
            polynomial.coefficients[1] -= &self.claim_factor;
        }

        true
    }
}

impl SumcheckBaseData for SelectorEq {
    fn partial_evaluate(&mut self, value: &RingElement) {
        // e.g. if selector = 0b110, nof_variables_for_selector = 3, nof_variables = 5
        // then bit_index = 0b1 (we look at the highest order bit of the selector)
        if self.nof_variables_for_selector > 0 {
            let bit_index = self.selector >> (self.nof_variables_for_selector - 1);

            if bit_index & 1 == 1 {
                // then we have a function which is 1 when the variable is 1, and 0 when the variable is 0. So this is an identity
                self.claim_factor *= value;
            } else {
                // then we have a function which is 1 when the variable is 0, and 0 when the variable is 1. So this is (1 - x)

                // TODO: optimize memory allocations
                let mut one = RingElement::one(Representation::IncompleteNTT);
                one -= value;
                self.claim_factor *= &one;
            }
            self.selector &= (1 << (self.nof_variables_for_selector - 1)) - 1;
            self.nof_variables_for_selector -= 1;

            // } else {
            // now we have a constrant function which is always 1, so the claim remains the same
        }
        self.nof_variables -= 1;
    }

    fn final_evaluations(&self) -> &RingElement {
        if self.nof_variables != 0 {
            panic!("final_evaluations called before all variables were evaluated");
        }
        &self.claim_factor
    }
}

#[test]
fn test_selector_eq_basic() {
    let selector = 0b10;
    let nof_variables_for_selector = 2;
    let nof_variables = 4;

    // this can be viewed as a sumcheck over the vector (0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0)
    let mut sumcheck = SelectorEq::new(selector, nof_variables_for_selector, nof_variables);
    let mut polynomial = Polynomial::new(2, Representation::IncompleteNTT);

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

    let mut claim = RingElement::constant(4, Representation::IncompleteNTT);

    sumcheck.univariate_polynomial_into(&mut polynomial);

    assert_eq!(&polynomial.at_zero() + &polynomial.at_one(), claim);

    let r0 = RingElement::constant(53, Representation::IncompleteNTT);

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
            (4 * 53 * (MOD_Q as i64 + 1 - 73) as u64),
            Representation::IncompleteNTT
        )
    );

    sumcheck.univariate_polynomial_into(&mut polynomial);

    assert_eq!(&polynomial.at_zero() + &polynomial.at_one(), claim);


    assert_eq!(polynomial.coefficients[1], RingElement::zero(Representation::IncompleteNTT));
    // after partial evaluation at r1, the function is constant, so the coeff of x is 0

    let r2 = RingElement::constant(19, Representation::IncompleteNTT);

    sumcheck.partial_evaluate(&r2);

    claim = polynomial.at(&r2);

    // (r0 * (1 - r1), r0 * (1 - r1), r0 * (1 - r1), r0 * (1 - r1))
    // changes into (r0 * (1 - r1), r0 * (1 - r1)) after partial evaluation at r2 (as the function is constant and the variable is ignored)

    assert_eq!(
        claim,
        RingElement::constant(
            (2 * 53 * (MOD_Q as i64 + 1 - 73) as u64),
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
            (53 * (MOD_Q as i64 + 1 - 73) as u64),
            Representation::IncompleteNTT
        )
    );

    assert_eq!(sumcheck.final_evaluations(), &claim);

}
