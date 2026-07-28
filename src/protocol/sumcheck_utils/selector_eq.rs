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
        point: HypercubePoint,
    ) -> Option<&Self::Element> {
        if self.selector_variable_count == 0 {
            // All selector bits consumed; constant equal to accumulated claim.
            return Some(&self.current_claim);
        }
        // LS-first: while non-selector LS vars remain, the polynomial
        // is constant at each non-zero point (= current_claim).
        // We must verify the point is actually non-zero (matches selector).
        if self.total_variable_count > self.selector_variable_count {
            if !self.is_univariate_polynomial_zero_at_point(point) {
                return Some(&self.current_claim);
            }
            // Point is in the zero region — polynomial is logically zero (degree 0),
            // but we can't return a pointer to zero here. Callers must first check
            // is_univariate_polynomial_zero_at_point; if it returns true, they
            // should treat the polynomial as identically zero rather than relying
            // on the general evaluation path.
            return None;
        }
        None
    }

    fn is_univariate_polynomial_zero_at_point(&self, point: HypercubePoint) -> bool {
        if self.selector_variable_count == 0 {
            return false; // constant function, never zero
        }

        // LS-first: selector bits constrain the MS bits of the point.
        // During non-selector rounds: all selector_variable_count bits
        // must match the full selector in the upper bits of point.
        // During selector rounds: the remaining (selector_variable_count - 1)
        // upper bits must match selector >> 1.
        if self.total_variable_count > self.selector_variable_count {
            // Non-selector round: check all selector bits against upper bits of point.
            let shift = self.total_variable_count - 1 - self.selector_variable_count;
            let upper_bits = point.shifted(shift);
            upper_bits.coordinates != self.selector
        } else {
            // Selector round: the current bit (LSB) is the one being folded.
            // Check the remaining upper bits.
            let remaining_bits = self.selector >> 1;
            if self.selector_variable_count <= 1 {
                return false; // only one selector bit left, no upper bits to mismatch
            }
            // point has total_variable_count - 1 bits. Upper selector_variable_count - 1 bits
            // must match remaining_bits.
            let shift = self.total_variable_count - self.selector_variable_count;
            let upper_bits = point.shifted(shift);
            upper_bits.coordinates != remaining_bits
        }
    }

    fn non_zero_range(&self) -> Option<(usize, usize)> {
        if self.selector_variable_count == 0 {
            return None; // constant → non-zero everywhere
        }

        if self.total_variable_count > self.selector_variable_count {
            // Non-selector round: all selector bits constrain upper bits.
            // Half-hypercube has (total_variable_count - 1) bits.
            // Upper selector_variable_count bits must equal selector.
            let shift = self.total_variable_count - 1 - self.selector_variable_count;
            let start = self.selector << shift;
            let end = (self.selector + 1) << shift;
            Some((start, end))
        } else {
            // Selector round: remaining upper bits = selector >> 1.
            let remaining_bits = self.selector >> 1;
            if self.selector_variable_count <= 1 {
                return None; // only current bit, whole half-hypercube is active
            }
            let shift = self.total_variable_count - self.selector_variable_count;
            let start = remaining_bits << shift;
            let end = (remaining_bits + 1) << shift;
            Some((start, end))
        }
    }

    fn univariate_polynomial_at_point_into(
        &self,
        _point: HypercubePoint,
        polynomial: &mut Polynomial<E>,
    ) {
        polynomial.set_zero();
        polynomial.num_coefficients = 2;

        if self.selector_variable_count == 0 {
            // All selector bits consumed; constant function.
            polynomial.coefficients[0] += &self.current_claim;
            polynomial.num_coefficients = 1;
            return;
        }

        // LS-first: non-selector round → polynomial is constant.
        if self.total_variable_count > self.selector_variable_count {
            polynomial.coefficients[0] += &self.current_claim;
            polynomial.num_coefficients = 1;
            return;
        }

        // Selector round: consume the LS bit of the selector.
        let current_bit = self.selector & 1;

        if current_bit == 1 {
            // f = current_claim * x
            polynomial.coefficients[1] += &self.current_claim;
            polynomial.num_coefficients = 2;
        } else {
            // f = current_claim * (1 - x)
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
        if self.selector_variable_count > 0
            && self.total_variable_count <= self.selector_variable_count
        {
            // Selector round: consume the LS bit.
            let current_bit = self.selector & 1;

            if current_bit == 1 {
                self.current_claim *= value;
            } else {
                let mut temp = self.temp_product.borrow_mut();
                *temp *= (value, &self.current_claim);
                self.current_claim -= &*temp;
            }
            self.selector >>= 1;
            self.selector_variable_count -= 1;
        }
        // Non-selector round (or selector_variable_count == 0): just decrement.
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

        // LS-first: the selector variables are folded LAST.
        // point layout: [non-selector LS vars..., selector vars from LSB to MSB...]
        // Selector challenges start at index (total_variable_count - selector_variable_count).
        let selector_start = self.total_variable_count - self.selector_variable_count;

        for i in 0..self.selector_variable_count {
            // Under LS-first, selector challenges are consumed LSB first.
            // point[selector_start + i] corresponds to selector bit i (from LSB).
            let selector_bit = (self.selector >> i) & 1;
            let r = &point[selector_start + i];

            if selector_bit == 1 {
                self.result *= r;
            } else {
                self.scratch = &self.result * r;
                self.result -= &self.scratch;
            }
        }

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

        // Equivalent vector: (0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0)
        // selector=0b10 constrains the 2 MS bits: bit3=1, bit2=0.
        let mut sumcheck =
            SelectorEq::<RingElement>::new(selector, selector_variable_count, total_variable_count);
        let mut polynomial = Polynomial::new(2);

        // LS-first: round 0 folds b0, round 1 folds b1 (non-selector rounds),
        // round 2 folds b2 (selector LSB=0), round 3 folds b3 (selector MSB=1).

        // Under LS-first non-selector round, non-zero points have upper 2 bits of
        // the half-hypercube point matching selector=0b10. Half-cube has 2^3=8 points.
        // shift = 4-1-2=1, non-zero range = [4,6). Points 4,5 are non-zero.
        debug_assert_eq!(
            sumcheck.is_univariate_polynomial_zero_at_point(HypercubePoint::new(0)),
            true
        );
        debug_assert_eq!(
            sumcheck.is_univariate_polynomial_zero_at_point(HypercubePoint::new(4)),
            false
        );
        debug_assert_eq!(
            sumcheck.is_univariate_polynomial_zero_at_point(HypercubePoint::new(5)),
            false
        );
        debug_assert_eq!(
            sumcheck.is_univariate_polynomial_zero_at_point(HypercubePoint::new(6)),
            true
        );

        // non_zero_range check
        debug_assert_eq!(sumcheck.non_zero_range(), Some((4, 6)));

        // Full claim = 4 assignments (indices 8,9,10,11 each have value 1).
        let claim = RingElement::constant(4, Representation::IncompleteNTT);

        sumcheck.univariate_polynomial_into(&mut polynomial);

        debug_assert_eq!(&polynomial.at_zero() + &polynomial.at_one(), claim);

        // Round 0: non-selector. Polynomial at point 4 is constant = 1.
        // The univariate polynomial from summing is constant = 2 (points 4 and 5 contribute 1 each).
        debug_assert_eq!(polynomial.num_coefficients, 1);

        let r0 = RingElement::constant(53, Representation::IncompleteNTT);

        sumcheck.partial_evaluate(&r0);
        let claim_after_r0 = polynomial.at(&r0);

        // After round 0 (non-selector), claim unchanged but half-cube shrank.
        // at_zero + at_one should still equal the claim.
        // claim_after_r0 = constant term * 1 (since poly is constant) = 2
        debug_assert_eq!(
            claim_after_r0,
            RingElement::constant(2, Representation::IncompleteNTT)
        );

        sumcheck.univariate_polynomial_into(&mut polynomial);

        debug_assert_eq!(&polynomial.at_zero() + &polynomial.at_one(), claim_after_r0);

        // Round 1: non-selector. Polynomial is constant.
        debug_assert_eq!(polynomial.num_coefficients, 1);

        let r1 = RingElement::constant(73, Representation::IncompleteNTT);

        sumcheck.partial_evaluate(&r1);
        let claim_after_r1 = polynomial.at(&r1);

        // claim_after_r1 = 1 (one matching point in half-cube of size 2)
        debug_assert_eq!(
            claim_after_r1,
            RingElement::constant(1, Representation::IncompleteNTT)
        );

        sumcheck.univariate_polynomial_into(&mut polynomial);

        debug_assert_eq!(&polynomial.at_zero() + &polynomial.at_one(), claim_after_r1);

        // Round 2: selector round, consuming bit 0 of selector (=0).
        // f = current_claim * (1-x) since bit=0.
        debug_assert_eq!(polynomial.num_coefficients, 2);

        let r2 = RingElement::constant(19, Representation::IncompleteNTT);

        sumcheck.partial_evaluate(&r2);
        let claim_after_r2 = polynomial.at(&r2);

        // current_claim was 1, now becomes 1*(1-r2) = 1-19
        debug_assert_eq!(
            claim_after_r2,
            RingElement::constant(
                (MOD_Q as i64 + 1 - 19) as u64 % MOD_Q,
                Representation::IncompleteNTT
            )
        );

        sumcheck.univariate_polynomial_into(&mut polynomial);

        debug_assert_eq!(&polynomial.at_zero() + &polynomial.at_one(), claim_after_r2);

        // Round 3: selector round, consuming bit 1 of selector (=1).
        // f = current_claim * x since bit=1.
        debug_assert_eq!(polynomial.num_coefficients, 2);

        let r3 = RingElement::constant(743, Representation::IncompleteNTT);

        sumcheck.partial_evaluate(&r3);
        let claim_after_r3 = polynomial.at(&r3);

        // current_claim was (1-r2), now becomes (1-r2)*r3 = (1-19)*743
        debug_assert_eq!(
            claim_after_r3,
            RingElement::constant(
                (((MOD_Q as i64 + 1 - 19) * 743) as u64) % MOD_Q,
                Representation::IncompleteNTT
            )
        );

        debug_assert_eq!(sumcheck.final_evaluations(), &claim_after_r3);
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
            RingElement::constant(53, Representation::IncompleteNTT), // non-selector var 0
            RingElement::constant(73, Representation::IncompleteNTT), // non-selector var 1
            RingElement::constant(19, Representation::IncompleteNTT), // selector bit 0 (=0)
            RingElement::constant(743, Representation::IncompleteNTT), // selector bit 1 (=1)
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
