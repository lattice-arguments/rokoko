use std::cell::{Cell, RefCell};

use crate::{
    common::{
        ring_arithmetic::{Representation, RingElement},
        sumcheck_element::SumcheckElement,
    },
    protocol::sumcheck_utils::{
        common::{EvaluationSumcheckData, HighOrderSumcheckData},
        elephant_cell::ElephantCell,
        hypercube_point::HypercubePoint,
        polynomial::{add_poly_in_place, mul_poly_into, Polynomial},
    },
};

#[cfg(test)]
use crate::{
    common::config::MOD_Q,
    protocol::sumcheck_utils::{common::SumcheckBaseData, linear::LinearSumcheck},
};

/// Sumcheck data that represents a pointwise product of two other sumcheck polynomials.
/// Each inner sumcheck is evaluated at the same hypercube point and the resulting
/// univariate polynomials are multiplied together.
pub struct ProductSumcheck<E: SumcheckElement = RingElement> {
    pub lhs_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = E>>,
    pub rhs_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = E>>,

    lhs_eval_poly: RefCell<Polynomial<E>>,
    rhs_eval_poly: RefCell<Polynomial<E>>,
    scratch_poly: RefCell<Polynomial<E>>,

    // Per-point cache: shared ProductSumcheck nodes in the DAG get
    // `univariate_polynomial_at_point_into` called multiple times for the
    // same point (by different parent outputs in the Combiner).  Caching the
    // last result avoids redundant polynomial multiplications.
    cache_poly: RefCell<Polynomial<E>>,
    cache_point: Cell<usize>,
    cache_vc: Cell<usize>,

    // Cached constant product when both children are constant at a point.
    const_cache: RefCell<E>,

    // Per-round cached flags: whether each child is "constant" (degree-0
    // polynomial) at every point in this round.
    lhs_const_flag: Cell<bool>,
    rhs_const_flag: Cell<bool>,
    const_flag_vc: Cell<usize>,

    // When true, skip per-point cache stores (set during univariate_polynomial_into
    // sweeps where each point is visited exactly once and the cache is never hit).
    sweeping: Cell<bool>,
}

impl<E: SumcheckElement> ProductSumcheck<E> {
    pub fn new(
        lhs_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = E>>,
        rhs_sumcheck: ElephantCell<dyn HighOrderSumcheckData<Element = E>>,
    ) -> ProductSumcheck<E> {
        debug_assert_eq!(
            lhs_sumcheck.get_ref().variable_count(),
            rhs_sumcheck.get_ref().variable_count(),
            "Inner product sumcheck: both sumchecks must have the same data length"
        );

        ProductSumcheck {
            lhs_sumcheck,
            rhs_sumcheck,
            lhs_eval_poly: RefCell::new(Polynomial::new(0)),
            rhs_eval_poly: RefCell::new(Polynomial::new(0)),
            scratch_poly: RefCell::new(Polynomial::new(0)),
            cache_poly: RefCell::new(Polynomial::new(0)),
            cache_point: Cell::new(0),
            cache_vc: Cell::new(usize::MAX),
            const_cache: RefCell::new(E::zero()),
            lhs_const_flag: Cell::new(false),
            rhs_const_flag: Cell::new(false),
            const_flag_vc: Cell::new(usize::MAX),
            sweeping: Cell::new(false),
        }
    }

    /// Lazily compute and cache per-round constant flags.
    #[inline]
    fn ensure_const_flags(&self) {
        let vc = self.lhs_sumcheck.get_ref().variable_count();
        if self.const_flag_vc.get() != vc {
            let lhs_c = self
                .lhs_sumcheck
                .get_ref()
                .constant_univariate_polynomial_at_point_available_by_ref(HypercubePoint::new(0))
                .is_some();
            let rhs_c = self
                .rhs_sumcheck
                .get_ref()
                .constant_univariate_polynomial_at_point_available_by_ref(HypercubePoint::new(0))
                .is_some();
            self.lhs_const_flag.set(lhs_c);
            self.rhs_const_flag.set(rhs_c);
            self.const_flag_vc.set(vc);
        }
    }
}

impl<E: SumcheckElement> HighOrderSumcheckData for ProductSumcheck<E> {
    type Element = E;

    fn get_scratch_poly(&self) -> &RefCell<Polynomial<E>> {
        &self.scratch_poly
    }
    fn max_num_polynomial_coefficients(&self) -> usize {
        self.lhs_sumcheck
            .get_ref()
            .max_num_polynomial_coefficients()
            + self
                .rhs_sumcheck
                .get_ref()
                .max_num_polynomial_coefficients()
            - 1
    }

    fn variable_count(&self) -> usize {
        self.lhs_sumcheck.get_ref().variable_count()
    }

    #[inline]
    fn is_univariate_polynomial_zero_at_point(&self, point: HypercubePoint) -> bool {
        self.lhs_sumcheck
            .get_ref()
            .is_univariate_polynomial_zero_at_point(point)
            || self
                .rhs_sumcheck
                .get_ref()
                .is_univariate_polynomial_zero_at_point(point)
    }

    fn non_zero_range(&self) -> Option<(usize, usize)> {
        let lr = self.lhs_sumcheck.get_ref().non_zero_range();
        let rr = self.rhs_sumcheck.get_ref().non_zero_range();
        match (lr, rr) {
            (Some((ls, le)), Some((rs, re))) => {
                // Intersection (both must be non-zero)
                let start = ls.max(rs);
                let end = le.min(re);
                if start < end {
                    Some((start, end))
                } else {
                    Some((0, 0)) // empty intersection
                }
            }
            (Some(r), None) | (None, Some(r)) => Some(r),
            (None, None) => None,
        }
    }

    #[inline]
    fn constant_univariate_polynomial_at_point_available_by_ref(
        &self,
        point: HypercubePoint,
    ) -> Option<&Self::Element> {
        let lhs_const = self
            .lhs_sumcheck
            .get_ref()
            .constant_univariate_polynomial_at_point_available_by_ref(point);
        let rhs_const = self
            .rhs_sumcheck
            .get_ref()
            .constant_univariate_polynomial_at_point_available_by_ref(point);
        match (lhs_const, rhs_const) {
            (Some(lc), Some(rc)) => {
                // Both children return constants – compute the product and cache it.
                // SAFETY: single-threaded access; the returned reference is consumed
                // before the next call to this method overwrites the cache.
                let cache = unsafe { &mut *self.const_cache.as_ptr() };
                *cache *= (lc, rc);
                Some(cache)
            }
            _ => None,
        }
    }

    /// Override: skip the expensive per-point `constant_at_point` tree traversal
    /// in the default loop.  When both children expose raw data slices
    /// (both are non-prefixed `LinearSumcheck`), compute the summed polynomial
    /// via batched inner products (3 dot products), eliminating all per-point
    /// vtable dispatch.
    fn univariate_polynomial_into(&self, polynomial: &mut Polynomial<Self::Element>) {
        // Case 1: both children expose raw data → batched Karatsuba inner product
        let lhs_data = self.lhs_sumcheck.get_ref().as_data_slices();
        let rhs_data = self.rhs_sumcheck.get_ref().as_data_slices();

        if let (Some((a_lo, a_hi)), Some((b_lo, b_hi))) = (lhs_data, rhs_data) {
            // Σ_p poly_a(p) * poly_b(p)  where poly_x(p) = [x_lo[p], x_hi[p]-x_lo[p]]
            //
            // Using Karatsuba on the sums:
            //   C0   = Σ a_lo[p] * b_lo[p]
            //   C_mid = Σ a_hi[p] * b_hi[p]
            //   C2   = Σ (a_hi[p]-a_lo[p]) * (b_hi[p]-b_lo[p])
            //   C1   = C_mid - C0 - C2
            polynomial.set_zero();
            polynomial.num_coefficients = 3;

            let mut temp = Self::Element::zero();
            let mut temp_a = Self::Element::zero();
            let mut temp_b = Self::Element::zero();

            let n = a_lo.len();
            for p in 0..n {
                // C0 += a_lo[p] * b_lo[p]
                temp *= (&a_lo[p], &b_lo[p]);
                polynomial.coefficients[0] += &temp;

                // C_mid (accumulated into coeff[1]) += a_hi[p] * b_hi[p]
                temp *= (&a_hi[p], &b_hi[p]);
                polynomial.coefficients[1] += &temp;

                // C2 += (a_hi[p]-a_lo[p]) * (b_hi[p]-b_lo[p])
                temp_a.set_from(&a_hi[p]);
                temp_a -= &a_lo[p]; // temp_a = a_hi - a_lo
                temp_b.set_from(&b_hi[p]);
                temp_b -= &b_lo[p]; // temp_b = b_hi - b_lo
                temp *= (&temp_a, &temp_b);
                polynomial.coefficients[2] += &temp;
            }

            // C1 = C_mid - C0 - C2 (C_mid is currently in coeff[1])
            let (first, rest) = polynomial.coefficients.split_at_mut(1);
            let (second, third) = rest.split_at_mut(1);
            second[0] -= &first[0];
            second[0] -= &third[0];

            return;
        }

        // Case 2: fall through to point-by-point, but skip the expensive
        // `constant_at_point` tree traversal that the default impl performs.
        // Use non_zero_range to iterate only the relevant points when a
        // sparse selector is present.
        polynomial.set_zero();
        polynomial.num_coefficients = 1;

        let half_hypercube = 1 << (self.variable_count() - 1);
        let (range_start, range_end) = self.non_zero_range().unwrap_or((0, half_hypercube));

        // Disable per-point cache stores during the sweep: each point is
        // visited exactly once, so the cache is never hit and the stores
        // are pure overhead (~50ns per store × 3 levels × thousands of points).
        self.set_cache_bypass(true);
        let mut scratch = self.get_scratch_poly().borrow_mut();
        for i in range_start..range_end {
            let point = HypercubePoint::new(i);
            if self.is_univariate_polynomial_zero_at_point(point) {
                continue;
            }
            self.univariate_polynomial_at_point_into(point, &mut scratch);
            add_poly_in_place(polynomial, &scratch);
        }
        self.set_cache_bypass(false);
    }

    #[inline]
    fn univariate_polynomial_at_point_into(
        &self,
        point: HypercubePoint,
        polynomial: &mut Polynomial<E>,
    ) {
        // --- shared-node cache: same point + same round → reuse prior result ---
        let vc = self.variable_count();
        if self.cache_point.get() == point.coordinates && self.cache_vc.get() == vc {
            polynomial.copy_from(&self.cache_poly.borrow());
            return;
        }

        // --- Use cached per-round constant flags instead of per-point tree
        //     traversal.  The flags are computed once per round by
        //     ensure_const_flags(). ---
        self.ensure_const_flags();
        let lhs_is_const = self.lhs_const_flag.get();
        let rhs_is_const = self.rhs_const_flag.get();

        match (lhs_is_const, rhs_is_const) {
            (true, true) => {
                // Both children constant at every point this round.
                let lc = self
                    .lhs_sumcheck
                    .get_ref()
                    .constant_univariate_polynomial_at_point_available_by_ref(point)
                    .unwrap();
                let rc = self
                    .rhs_sumcheck
                    .get_ref()
                    .constant_univariate_polynomial_at_point_available_by_ref(point)
                    .unwrap();
                polynomial.coefficients[0] *= (lc, rc);
                polynomial.num_coefficients = 1;
            }
            (true, false) => {
                // LHS constant → scale RHS polynomial
                let lc = self
                    .lhs_sumcheck
                    .get_ref()
                    .constant_univariate_polynomial_at_point_available_by_ref(point)
                    .unwrap();
                self.rhs_sumcheck
                    .get_ref()
                    .univariate_polynomial_at_point_into(point, polynomial);
                for i in 0..polynomial.num_coefficients {
                    polynomial.coefficients[i] *= lc;
                }
            }
            (false, true) => {
                // RHS constant → scale LHS polynomial
                let rc = self
                    .rhs_sumcheck
                    .get_ref()
                    .constant_univariate_polynomial_at_point_available_by_ref(point)
                    .unwrap();
                self.lhs_sumcheck
                    .get_ref()
                    .univariate_polynomial_at_point_into(point, polynomial);
                for i in 0..polynomial.num_coefficients {
                    polynomial.coefficients[i] *= rc;
                }
            }
            (false, false) => {
                // General case: full polynomial multiplication
                let mut lhs_eval_poly = self.lhs_eval_poly.borrow_mut();
                let mut rhs_eval_poly = self.rhs_eval_poly.borrow_mut();

                self.lhs_sumcheck
                    .get_ref()
                    .univariate_polynomial_at_point_into(point, &mut lhs_eval_poly);

                self.rhs_sumcheck
                    .get_ref()
                    .univariate_polynomial_at_point_into(point, &mut rhs_eval_poly);

                mul_poly_into(polynomial, &lhs_eval_poly, &rhs_eval_poly);
            }
        }

        if !self.sweeping.get() {
            self.cache_poly.borrow_mut().copy_from(polynomial);
            self.cache_point.set(point.coordinates);
            self.cache_vc.set(vc);
        }
    }

    fn set_cache_bypass(&self, bypass: bool) {
        self.sweeping.set(bypass);
        self.lhs_sumcheck.get_ref().set_cache_bypass(bypass);
        self.rhs_sumcheck.get_ref().set_cache_bypass(bypass);
    }

    fn final_evaluations_test_only(&self) -> Self::Element {
        let mut result = self
            .lhs_sumcheck
            .get_ref()
            .final_evaluations_test_only()
            .clone();
        result *= &self.rhs_sumcheck.get_ref().final_evaluations_test_only();
        result
    }
}

/// Evaluation-only version of ProductSumcheck that evaluates the product of two sumchecks at a point.
pub struct ProductSumcheckEvaluation {
    lhs_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
    rhs_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
    result: RingElement,
}

impl ProductSumcheckEvaluation {
    pub fn new(
        lhs_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
        rhs_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>>,
    ) -> Self {
        ProductSumcheckEvaluation {
            lhs_evaluation,
            rhs_evaluation,
            result: RingElement::constant(0, Representation::IncompleteNTT),
        }
    }
}

impl EvaluationSumcheckData for ProductSumcheckEvaluation {
    type Element = RingElement;

    fn evaluate(&mut self, point: &Vec<Self::Element>) -> &Self::Element {
        self.result *= (
            self.lhs_evaluation.borrow_mut().evaluate(&point),
            self.rhs_evaluation.borrow_mut().evaluate(&point),
        );
        &self.result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_inner_product_sumcheck() {
        let lhs_data = vec![
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(2, Representation::IncompleteNTT),
            RingElement::constant(3, Representation::IncompleteNTT),
            RingElement::constant(4, Representation::IncompleteNTT),
            RingElement::constant(5, Representation::IncompleteNTT),
            RingElement::constant(6, Representation::IncompleteNTT),
            RingElement::constant(7, Representation::IncompleteNTT),
            RingElement::constant(8, Representation::IncompleteNTT),
        ];

        let rhs_data = vec![
            RingElement::constant(9, Representation::IncompleteNTT),
            RingElement::constant(10, Representation::IncompleteNTT),
            RingElement::constant(11, Representation::IncompleteNTT),
            RingElement::constant(12, Representation::IncompleteNTT),
            RingElement::constant(13, Representation::IncompleteNTT),
            RingElement::constant(14, Representation::IncompleteNTT),
            RingElement::constant(15, Representation::IncompleteNTT),
            RingElement::constant(16, Representation::IncompleteNTT),
        ];

        let sumcheck_0 = ElephantCell::new(LinearSumcheck::new(lhs_data.len()));
        sumcheck_0.borrow_mut().load_from(&lhs_data);
        let sumcheck_1 = ElephantCell::new(LinearSumcheck::new(rhs_data.len()));
        sumcheck_1.borrow_mut().load_from(&rhs_data);

        // Build a product sumcheck that should track the inner product of lhs_data and rhs_data.
        let inner_product_sumcheck = ProductSumcheck::new(sumcheck_0.clone(), sumcheck_1.clone());

        let mut univariate_poly = Polynomial::new(0);

        inner_product_sumcheck.univariate_polynomial_into(&mut univariate_poly);

        // The polynomial evaluated at 0 and 1 should sum to the true inner product.

        debug_assert_eq!(
            &univariate_poly.at_zero() + &univariate_poly.at_one(),
            RingElement::constant(
                1 * 9 + 2 * 10 + 3 * 11 + 4 * 12 + 5 * 13 + 6 * 14 + 7 * 15 + 8 * 16,
                Representation::IncompleteNTT
            )
        );

        let r0 = RingElement::constant(524, Representation::IncompleteNTT);

        let claim = univariate_poly.at(&r0);

        // Fold both underlying multilinear extensions by r0 and ensure the verifier
        // still sees the same claim when re-running round 0 of the protocol.
        sumcheck_0.borrow_mut().partial_evaluate(&r0);
        sumcheck_1.borrow_mut().partial_evaluate(&r0);

        inner_product_sumcheck.univariate_polynomial_into(&mut univariate_poly);

        debug_assert_eq!(
            &univariate_poly.at_zero() + &univariate_poly.at_one(),
            claim
        );

        let r1 = RingElement::constant(1337, Representation::IncompleteNTT);

        let claim = univariate_poly.at(&r1);

        // Same invariance check after the second round challenge.
        sumcheck_0.borrow_mut().partial_evaluate(&r1);
        sumcheck_1.borrow_mut().partial_evaluate(&r1);

        inner_product_sumcheck.univariate_polynomial_into(&mut univariate_poly);

        debug_assert_eq!(
            &univariate_poly.at_zero() + &univariate_poly.at_one(),
            claim
        );

        let r2 = RingElement::constant(42, Representation::IncompleteNTT);

        let claim = univariate_poly.at(&r2);

        // After the final fold, the product of the two fully evaluated claims
        // should equal the verifier's accumulated claim.
        sumcheck_0.borrow_mut().partial_evaluate(&r2);
        sumcheck_1.borrow_mut().partial_evaluate(&r2);

        debug_assert_eq!(
            sumcheck_0.get_ref().final_evaluations() * sumcheck_1.get_ref().final_evaluations(),
            claim,
        );

        // Explicit multilinear evaluations of each folded claim for documentation.
        debug_assert_eq!(
            sumcheck_0.get_ref().final_evaluations(),
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

        debug_assert_eq!(
            sumcheck_1.get_ref().final_evaluations(),
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

        // Final consistency: inner product claim equals product of individual folded claims.
        debug_assert_eq!(
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

        let sumcheck = ElephantCell::new(LinearSumcheck::new(data.len()));
        sumcheck.borrow_mut().load_from(&data);

        let inner_product_sumcheck = ProductSumcheck::new(sumcheck.clone(), sumcheck.clone());

        let mut univariate_poly = Polynomial::new(0);

        inner_product_sumcheck.univariate_polynomial_into(&mut univariate_poly);

        // When both inputs are identical, the inner product collapses to a sum of squares.
        debug_assert_eq!(
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

        let mut sumcheck_0 = LinearSumcheck::new(data0.len());
        sumcheck_0.load_from(&data0);
        let mut sumcheck_1 = LinearSumcheck::new(data1.len());
        sumcheck_1.load_from(&data1);
        let mut sumcheck_2 = LinearSumcheck::new(data2.len());
        sumcheck_2.load_from(&data2);

        let sumcheck_0_ref = ElephantCell::new(sumcheck_0);
        let sumcheck_1_ref = ElephantCell::new(sumcheck_1);
        let sumcheck_2_ref = ElephantCell::new(sumcheck_2);

        // Compose two nested product sumchecks to validate a three-way product.

        let inner_product_sumcheck_01 =
            ProductSumcheck::new(sumcheck_0_ref.clone(), sumcheck_1_ref.clone());

        let inner_product_sumcheck_01_ref = ElephantCell::new(inner_product_sumcheck_01);

        let inner_product_sumcheck_012 = ProductSumcheck::new(
            sumcheck_2_ref.clone(),
            inner_product_sumcheck_01_ref.clone(),
        );

        let mut univariate_poly = Polynomial::new(0);
        inner_product_sumcheck_012.univariate_polynomial_into(&mut univariate_poly);

        // Evaluating the first-round polynomial at 0 and 1 should give the full triple product sum.
        debug_assert_eq!(
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

        // The verifier's running claim should stay consistent after folding all three vectors.
        debug_assert_eq!(
            &univariate_poly.at_zero() + &univariate_poly.at_one(),
            claim
        );
    }

    #[test]
    fn test_product_of_linear_sumchecks_over_disjoint_variables() {
        // Test the product sumcheck over three linear sumchecks, each
        // defined over disjoint sets of variables.
        // The first sumcheck is defined over the highest-order variables,
        // the second over the middle variables, and the third over the
        // lowest-order variables.
        // we make use that the univariate polynomial of a linear sumcheck is just a degree 1 polynomial
        // since it should be a produce of two constant polys and one degree 1 poly.

        use crate::common::ring_arithmetic::RingElement;
        use crate::protocol::sumcheck_utils::product::ProductSumcheck;

        let data1 = vec![
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(2, Representation::IncompleteNTT),
            RingElement::constant(3, Representation::IncompleteNTT),
            RingElement::constant(4, Representation::IncompleteNTT),
        ];

        let data2 = vec![
            RingElement::constant(5, Representation::IncompleteNTT),
            RingElement::constant(6, Representation::IncompleteNTT),
            RingElement::constant(7, Representation::IncompleteNTT),
            RingElement::constant(8, Representation::IncompleteNTT),
        ];

        let data3 = vec![
            RingElement::constant(9, Representation::IncompleteNTT),
            RingElement::constant(10, Representation::IncompleteNTT),
            RingElement::constant(11, Representation::IncompleteNTT),
            RingElement::constant(12, Representation::IncompleteNTT),
        ];

        let sumcheck_1 = ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
            data1.len(),
            0,
            4,
        ));
        sumcheck_1.borrow_mut().load_from(&data1);

        // 1 x 16, 2 x 16, 3 x 16, 4 x 16, 1 x 16, 2 x 16, 3 x 16, 4 x 16,

        let sumcheck_2 = ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
            data2.len(),
            2,
            2,
        ));

        // (5 x 4, 6 x 4, 7 x 4, 8 x 4, 5 x 4, 6 x 4, 7 x 4, 8 x 4) x 4

        sumcheck_2.borrow_mut().load_from(&data2);
        let sumcheck_3 = ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
            data3.len(),
            4,
            0,
        ));
        sumcheck_3.borrow_mut().load_from(&data3);

        let product_12 =
            ElephantCell::new(ProductSumcheck::new(sumcheck_1.clone(), sumcheck_2.clone()));
        let product_123 =
            ElephantCell::new(ProductSumcheck::new(product_12.clone(), sumcheck_3.clone()));
        let mut univariate_poly = Polynomial::new(0);

        let mut claim = RingElement::constant(
            (1 + 2 + 3 + 4) * (5 + 6 + 7 + 8) * (9 + 10 + 11 + 12),
            Representation::IncompleteNTT,
        );

        for _ in 0..5 {
            product_123
                .get_ref()
                .univariate_polynomial_into(&mut univariate_poly);
            debug_assert_eq!(
                &univariate_poly.at_zero() + &univariate_poly.at_one(),
                claim
            );

            debug_assert_eq!(univariate_poly.num_coefficients, 2); // degree 1 polynomial as expected

            let r = RingElement::constant(7, Representation::IncompleteNTT);

            claim = univariate_poly.at(&r);

            sumcheck_1.borrow_mut().partial_evaluate(&r);
            sumcheck_2.borrow_mut().partial_evaluate(&r);
            sumcheck_3.borrow_mut().partial_evaluate(&r);
        }

        product_123
            .get_ref()
            .univariate_polynomial_into(&mut univariate_poly);
        debug_assert_eq!(
            &univariate_poly.at_zero() + &univariate_poly.at_one(),
            claim
        );

        debug_assert_eq!(univariate_poly.num_coefficients, 2); // degree 1 polynomial as expected
    }

    #[test]
    fn test_product_evaluation() {
        use crate::protocol::sumcheck_utils::linear::BasicEvaluationLinearSumcheck;

        let lhs_data = vec![
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(2, Representation::IncompleteNTT),
            RingElement::constant(3, Representation::IncompleteNTT),
            RingElement::constant(4, Representation::IncompleteNTT),
        ];

        let rhs_data = vec![
            RingElement::constant(5, Representation::IncompleteNTT),
            RingElement::constant(6, Representation::IncompleteNTT),
            RingElement::constant(7, Representation::IncompleteNTT),
            RingElement::constant(8, Representation::IncompleteNTT),
        ];

        let mut lhs_eval_impl = BasicEvaluationLinearSumcheck::new(lhs_data.len());
        lhs_eval_impl.load_from(&lhs_data);
        let lhs_eval: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>> =
            ElephantCell::new(lhs_eval_impl);

        let mut rhs_eval_impl = BasicEvaluationLinearSumcheck::new(rhs_data.len());
        rhs_eval_impl.load_from(&rhs_data);
        let rhs_eval: ElephantCell<dyn EvaluationSumcheckData<Element = RingElement>> =
            ElephantCell::new(rhs_eval_impl);

        let mut product_eval = ProductSumcheckEvaluation::new(lhs_eval, rhs_eval);

        let point = vec![
            RingElement::constant(11, Representation::IncompleteNTT),
            RingElement::constant(13, Representation::IncompleteNTT),
        ];

        // Create reference using the folding implementation
        let ref_lhs_data = vec![
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(2, Representation::IncompleteNTT),
            RingElement::constant(3, Representation::IncompleteNTT),
            RingElement::constant(4, Representation::IncompleteNTT),
        ];
        let ref_rhs_data = vec![
            RingElement::constant(5, Representation::IncompleteNTT),
            RingElement::constant(6, Representation::IncompleteNTT),
            RingElement::constant(7, Representation::IncompleteNTT),
            RingElement::constant(8, Representation::IncompleteNTT),
        ];

        let sumcheck_0 = ElephantCell::new(LinearSumcheck::new(ref_lhs_data.len()));
        sumcheck_0.borrow_mut().load_from(&ref_lhs_data);
        let sumcheck_1 = ElephantCell::new(LinearSumcheck::new(ref_rhs_data.len()));
        sumcheck_1.borrow_mut().load_from(&ref_rhs_data);

        for r in &point {
            sumcheck_0.borrow_mut().partial_evaluate(r);
            sumcheck_1.borrow_mut().partial_evaluate(r);
        }

        let expected =
            sumcheck_0.get_ref().final_evaluations() * sumcheck_1.get_ref().final_evaluations();

        debug_assert_eq!(product_eval.evaluate(&point), &expected);
    }
}
