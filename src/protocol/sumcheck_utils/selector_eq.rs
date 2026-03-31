use std::cell::RefCell;

use crate::{
    common::{
        arithmetic::ONE,
        ring_arithmetic::{Representation, RingElement},
        sumcheck_element::SumcheckElement,
    },
    protocol::sumcheck_utils::{
        common::{EvaluationSumcheckData, HighOrderSumcheckData, SumcheckBaseData},
        hypercube_point::HypercubePoint,
        polynomial::Polynomial,
    },
};

#[cfg(test)]
use crate::common::config::MOD_Q;

/// Sumcheck for the multilinear equality check `eq(x, selector)`.
/// It evaluates to 1 exactly when the first `selector_variable_count` bits of
/// the query match `selector`, while ignoring additional trailing variables.
pub struct SelectorEq<E: SumcheckElement = RingElement> {
    selector: usize,
    selector_variable_count: usize,
    total_variable_count: usize,

    // if we do partial evaluation, we have to store the current claim
    current_claim: E,
    temp_product: RefCell<E>,
    scratch_poly: RefCell<Polynomial<E>>,
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
impl<E: SumcheckElement> SelectorEq<E> {
    pub fn new(
        selector: usize,
        selector_variable_count: usize,
        total_variable_count: usize,
    ) -> Self {
        SelectorEq {
            selector,
            selector_variable_count,
            total_variable_count,
            current_claim: E::one(),
            temp_product: RefCell::new(E::zero()),
            scratch_poly: RefCell::new(Polynomial::new(2)),
        }
    }
}

impl<E: SumcheckElement> HighOrderSumcheckData for SelectorEq<E> {
    type Element = E;

    fn get_scratch_poly(&self) -> &RefCell<Polynomial<E>> {
        &self.scratch_poly
    }
    fn max_num_polynomial_coefficients(&self) -> usize {
        2
    }
    fn variable_count(&self) -> usize {
        self.total_variable_count
    }

    #[inline]
    fn constant_univariate_polynomial_at_point_available_by_ref(
        &self,
        _point: HypercubePoint,
    ) -> Option<&Self::Element> {
        if self.selector_variable_count == 0 {
            // All selector bits consumed; the function is a constant equal to
            // the accumulated claim for every point in the remaining hypercube.
            return Some(&self.current_claim);
        }
        None
    }

    fn is_univariate_polynomial_zero_at_point(&self, point: HypercubePoint) -> bool {
        if self.selector_variable_count == 0 {
            return false; // now we have a constant function
        }

        let point_higher_bits =
            point.shifted(self.total_variable_count - self.selector_variable_count);

        let selector_bits = self.selector & ((1 << self.selector_variable_count - 1) - 1); // mask to get only the relevant bits
        point_higher_bits.coordinates != selector_bits
    }

    fn non_zero_range(&self) -> Option<(usize, usize)> {
        if self.selector_variable_count == 0 {
            return None; // constant → non-zero everywhere
        }
        // Non-zero points: those where the upper bits (below the current
        // variable) match `selector_bits`.
        let selector_bits = self.selector & ((1 << (self.selector_variable_count - 1)) - 1);
        let shift = self.total_variable_count - self.selector_variable_count;
        let start = selector_bits << shift;
        let end = (selector_bits + 1) << shift;
        Some((start, end))
    }

    fn univariate_polynomial_at_point_into(
        &self,
        point: HypercubePoint,
        polynomial: &mut Polynomial<E>,
    ) {
        polynomial.set_zero();
        polynomial.num_coefficients = 2;

        if self.selector_variable_count == 0 {
            // now we have a constrant function which is always 1
            polynomial.coefficients[0] += &self.current_claim;
            polynomial.num_coefficients = 1;
            return;
        }

        // We evaluate from the highest order bit to the lowest order bit
        // Therefore, we check if the higher order bits of point match the selector (expect for the current variable being evaluated being the highest order bit)

        let point_higher_bits =
            point.shifted(self.total_variable_count - self.selector_variable_count);

        let selector_bits = self.selector & ((1 << self.selector_variable_count - 1) - 1); // mask to get only the relevant bits

        debug_assert_eq!(
            point_higher_bits.coordinates, selector_bits,
            "the polynomial is identically zero at this point. Eval should not be called."
        );

        let current_bit = (self.selector >> (self.selector_variable_count - 1)) & 1;

        if current_bit == 1 {
            // then we have a function which is 1 when the variable is 1, and 0 when the variable is 0. So this is an identity
            polynomial.coefficients[1] += &self.current_claim;
            polynomial.num_coefficients = 2;
        } else {
            // then we have a function which is 1 when the variable is 0, and 0 when the variable is 1. So this is (1 - x)
            polynomial.coefficients[0] += &self.current_claim;
            polynomial.coefficients[1] -= &self.current_claim;
            polynomial.num_coefficients = 2;
        }
    }

    fn final_evaluations_test_only(&self) -> E {
        if self.total_variable_count != 0 {
            panic!("final_evaluations called before all variables were evaluated");
        }
        self.current_claim.clone()
    }
}

impl<E: SumcheckElement> SumcheckBaseData for SelectorEq<E> {
    fn partial_evaluate(&mut self, value: &E) {
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

    fn final_evaluations(&self) -> &E {
        if self.total_variable_count != 0 {
            panic!("final_evaluations called before all variables were evaluated");
        }
        &self.current_claim
    }
}

pub struct SelectorEqEvaluation {
    selector: usize,
    selector_variable_count: usize,
    total_variable_count: usize,
    result: RingElement,
    scratch: RingElement,
    evaluated: bool,
}

impl SelectorEqEvaluation {
    pub fn new(
        selector: usize,
        selector_variable_count: usize,
        total_variable_count: usize,
    ) -> Self {
        SelectorEqEvaluation {
            selector,
            selector_variable_count,
            total_variable_count,
            result: RingElement::constant(1, Representation::IncompleteNTT),
            scratch: RingElement::zero(Representation::IncompleteNTT),
            evaluated: false,
        }
    }
}

impl EvaluationSumcheckData for SelectorEqEvaluation {
    type Element = RingElement;

    fn evaluate(&mut self, point: &Vec<Self::Element>) -> &Self::Element {
        if self.evaluated {
            return &self.result;
        }

        if point.len() != self.total_variable_count {
            panic!("Point has incorrect number of variables");
        }

        self.result.set_from(&*ONE);

        // For the selector variables, compute the eq polynomial value
        // eq(x, selector) = product over i of: (x_i * selector_i + (1-x_i) * (1-selector_i))
        // = product over i of: (x_i if selector_i=1, else (1-x_i))

        for i in 0..self.selector_variable_count {
            let selector_bit = (self.selector >> (self.selector_variable_count - 1 - i)) & 1;
            let r = &point[i];

            if selector_bit == 1 {
                // Multiply by r
                self.result *= r;
            } else {
                // Multiply by (1 - r)
                // Compute: result = result - result * r = result * (1 - r)
                self.scratch = &self.result * r;
                self.result -= &self.scratch;
            }
        }

        // For the remaining variables (beyond selector_variable_count), they don't affect
        // the selector equality - it's constant across those dimensions.
        // So we don't need to do anything with them.

        self.evaluated = true;
        &self.result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selector_eq_basic() {
        let selector = 0b10;
        let selector_variable_count = 2;
        let total_variable_count = 4;

        // this can be viewed as a sumcheck over the vector (0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0)
        let mut sumcheck =
            SelectorEq::<RingElement>::new(selector, selector_variable_count, total_variable_count);
        let mut polynomial = Polynomial::new(2);

        // Irrelevant points should produce an identically-zero polynomial.
        // let result =
        //     sumcheck.univariate_polynomial_at_point_into(HypercubePoint::new(0b100), &mut polynomial);

        // 0b0100 and 0b1100 do not match the selector in the higher order bits, so the polynomial is identically zero
        debug_assert_eq!(
            sumcheck.is_univariate_polynomial_zero_at_point(HypercubePoint::new(0b100)),
            true
        );

        // let result =
        // sumcheck.univariate_polynomial_at_point_into(HypercubePoint::new(0b101), &mut polynomial);

        // 0b0101 and 0b1101 do not match the selector in the higher order bits, so the polynomial is identically zero
        debug_assert_eq!(
            sumcheck.is_univariate_polynomial_zero_at_point(HypercubePoint::new(0b101)),
            true
        );

        let _result = sumcheck
            .univariate_polynomial_at_point_into(HypercubePoint::new(0b010), &mut polynomial);

        // 0b1010 matches the selector in the higher order bits
        debug_assert_eq!(
            sumcheck.is_univariate_polynomial_zero_at_point(HypercubePoint::new(0b010)),
            false
        );

        // as selector = 0b10, the polynomial should be x as it's 1 when the variable is 1, and 0 when the variable is 0
        debug_assert_eq!(
            polynomial.coefficients[0],
            RingElement::zero(Representation::IncompleteNTT)
        );
        debug_assert_eq!(
            polynomial.coefficients[1],
            RingElement::one(Representation::IncompleteNTT)
        );

        // There are exactly four satisfying assignments for the selector bits.
        let mut claim = RingElement::constant(4, Representation::IncompleteNTT);

        sumcheck.univariate_polynomial_into(&mut polynomial);

        debug_assert_eq!(&polynomial.at_zero() + &polynomial.at_one(), claim);

        let r0 = RingElement::constant(53, Representation::IncompleteNTT);

        // Folding the highest bit turns the indicator vector into four copies of r0.
        sumcheck.partial_evaluate(&r0);

        claim = polynomial.at(&r0);

        // (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0)
        // changes into (r0, r0, r0, r0, 0, 0, 0, 0) after partial evaluation at r0
        debug_assert_eq!(
            claim,
            RingElement::constant(4 * 53, Representation::IncompleteNTT)
        );

        sumcheck.univariate_polynomial_into(&mut polynomial);

        debug_assert_eq!(&polynomial.at_zero() + &polynomial.at_one(), claim);

        let r1 = RingElement::constant(73, Representation::IncompleteNTT);

        sumcheck.partial_evaluate(&r1);

        claim = polynomial.at(&r1);

        // (r0, r0, r0, r0, 0, 0, 0, 0) // at 0 it's r0, at 1 it's 0 so the funtion is r0 * (1 - x)
        // changes into (r0 * (1 - r1), r0 * (1 - r1), r0 * (1 - r1), r0 * (1 - r1)) after partial evaluation at r1

        debug_assert_eq!(
            claim,
            RingElement::constant(
                ((4 * 53 * (MOD_Q as i64 + 1 - 73)) as u64) % MOD_Q,
                Representation::IncompleteNTT
            )
        );

        sumcheck.univariate_polynomial_into(&mut polynomial);

        debug_assert_eq!(&polynomial.at_zero() + &polynomial.at_one(), claim);

        debug_assert_eq!(
            polynomial.coefficients[1],
            RingElement::zero(Representation::IncompleteNTT)
        );
        // after partial evaluation at r1, the function is constant, so the coeff of x is 0

        let r2 = RingElement::constant(19, Representation::IncompleteNTT);

        sumcheck.partial_evaluate(&r2);

        claim = polynomial.at(&r2);

        // (r0 * (1 - r1), r0 * (1 - r1), r0 * (1 - r1), r0 * (1 - r1))
        // changes into (r0 * (1 - r1), r0 * (1 - r1)) after partial evaluation at r2 (as the function is constant and the variable is ignored)

        debug_assert_eq!(
            claim,
            RingElement::constant(
                ((2 * 53 * (MOD_Q as i64 + 1 - 73)) as u64) % MOD_Q,
                Representation::IncompleteNTT
            )
        );

        sumcheck.univariate_polynomial_into(&mut polynomial);

        debug_assert_eq!(&polynomial.at_zero() + &polynomial.at_one(), claim);

        let r3 = RingElement::constant(743, Representation::IncompleteNTT);

        sumcheck.partial_evaluate(&r3);

        claim = polynomial.at(&r3);

        // (r0 * (1 - r1), r0 * (1 - r1))
        // changes into (r0 * (1 - r1)) after partial evaluation at r3 (as the function is constant and the variable is ignored)

        debug_assert_eq!(
            claim,
            RingElement::constant(
                ((53 * (MOD_Q as i64 + 1 - 73)) as u64) % MOD_Q,
                Representation::IncompleteNTT
            )
        );

        // After exhausting all variables, the stored claim should match the verifier's expectation.
        debug_assert_eq!(sumcheck.final_evaluations(), &claim);
    }

    #[test]
    fn test_selector_eq_evaluation() {
        use crate::common::ring_arithmetic::RingElement;

        let selector = 0b10;
        let selector_variable_count = 2;
        let total_variable_count = 4;

        let mut evaluation_sumcheck =
            SelectorEqEvaluation::new(selector, selector_variable_count, total_variable_count);

        let point = vec![
            RingElement::constant(53, Representation::IncompleteNTT), // var 0
            RingElement::constant(73, Representation::IncompleteNTT), // var 1
            RingElement::constant(19, Representation::IncompleteNTT), // var 2
            RingElement::constant(743, Representation::IncompleteNTT), // var 3
        ];

        let mut ref_sumcheck =
            SelectorEq::<RingElement>::new(selector, selector_variable_count, total_variable_count);

        for r in point.iter() {
            ref_sumcheck.partial_evaluate(r);
        }
        let expected_evaluation = ref_sumcheck.final_evaluations();

        debug_assert_eq!(evaluation_sumcheck.evaluate(&point), expected_evaluation);
    }
}
