pub mod combiner;
pub mod common;
pub mod diff;
pub mod elephant_cell;
pub mod hypercube_point;
pub mod linear;
pub mod polynomial;
pub mod product;
#[cfg(feature = "profile-sumcheck")]
pub mod profile;
pub mod ring_to_field_combiner;
pub mod selector_eq;
pub mod sum;

#[cfg(test)]
mod tests {
    use super::elephant_cell::ElephantCell;

    use crate::common::ring_arithmetic::{Representation, RingElement};

    use super::{
        diff::DiffSumcheck, linear::LinearSumcheck, polynomial::Polynomial,
        product::ProductSumcheck, selector_eq::SelectorEq,
    };
    use crate::protocol::sumcheck_utils::common::{HighOrderSumcheckData, SumcheckBaseData};

    #[test]
    fn test_subvector_inner_product_difference_zero() {
        let repr = Representation::IncompleteNTT;

        // Witness split into two halves with identical values; coefficients are
        // non-trivial but shared across halves so both inner products evaluate to 51.
        let witness = vec![
            RingElement::constant(1, repr),
            RingElement::constant(2, repr),
            RingElement::constant(3, repr),
            RingElement::constant(4, repr),
            RingElement::constant(2, repr),
            RingElement::constant(1, repr),
            RingElement::constant(3, repr),
            RingElement::constant(4, repr),
        ];

        // Coefficients applied to each half of the witness.
        let lhs_coeffs = vec![
            RingElement::constant(2, repr),
            RingElement::constant(3, repr),
            RingElement::constant(5, repr),
            RingElement::constant(7, repr),
        ];
        let rhs_coeffs = vec![
            RingElement::constant(3, repr),
            RingElement::constant(2, repr),
            RingElement::constant(5, repr),
            RingElement::constant(7, repr),
        ];

        // we are interested in proving that
        // <lhs_coeffs, witness[0..4]> - <rhs_coeffs, witness[4..8]> == 0
        // we will express this as a diff sumcheck
        // \sum_{z \in HC} }MLE[lhs_coeffs](z) * MLE[witness](z) * eq(z, 0b0)
        //     - MLE[rhs_coeffs](z) * MLE[witness](z) * eq(z, 0b0) == 0

        // Base sumchecks for each half of the witness and the two coefficient vectors.
        // Each is prefixed with one dummy variable so they share the same 3-variable
        // domain and can be multiplied together. Because the two witness halves and
        // the coefficient vectors are identical, the target inner products are equal.
        let witness_sc = ElephantCell::new(LinearSumcheck::new(witness.len()));
        witness_sc.borrow_mut().load_from(&witness);

        let lhs_coeff_sumcheck = ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
            lhs_coeffs.len(),
            1,
            0,
        ));
        lhs_coeff_sumcheck.borrow_mut().load_from(&lhs_coeffs);

        let rhs_coeff_sumcheck = ElephantCell::new(LinearSumcheck::new_with_prefixed_sufixed_data(
            rhs_coeffs.len(),
            1,
            0,
        ));
        rhs_coeff_sumcheck.borrow_mut().load_from(&rhs_coeffs);

        // Selectors act as neutral scalars here (selector_variable_count = 0),
        // but are wired in to demonstrate composition with equality gadgets.
        let lhs_selector = ElephantCell::new(SelectorEq::<RingElement>::new(0b0, 0, 3));
        let rhs_selector = ElephantCell::new(SelectorEq::<RingElement>::new(0b0, 0, 3));

        // Build product sumchecks for each half: <coeffs, witness_subset>.
        let lhs_inner = ElephantCell::new(ProductSumcheck::new(
            witness_sc.clone(),
            lhs_coeff_sumcheck.clone(),
        ));
        let lhs_masked = ElephantCell::new(ProductSumcheck::new(
            lhs_inner.clone(),
            lhs_selector.clone(),
        ));

        let rhs_inner = ElephantCell::new(ProductSumcheck::new(
            witness_sc.clone(),
            rhs_coeff_sumcheck.clone(),
        ));
        let rhs_masked = ElephantCell::new(ProductSumcheck::new(
            rhs_inner.clone(),
            rhs_selector.clone(),
        ));
        let diff_sumcheck = DiffSumcheck::new(lhs_masked.clone(), rhs_masked.clone());
        let mut poly = Polynomial::new(0);

        // Initial claim: the difference of inner products over the full hypercube is zero.
        diff_sumcheck.univariate_polynomial_into(&mut poly);
        debug_assert_eq!(&poly.at_zero() + &poly.at_one(), RingElement::zero(repr));

        // Round 1: fold highest-order variable, preserving claim consistency.
        let r0 = RingElement::constant(7, repr);
        let claim_after_r0 = poly.at(&r0);

        witness_sc.borrow_mut().partial_evaluate(&r0);
        lhs_coeff_sumcheck.borrow_mut().partial_evaluate(&r0);
        rhs_coeff_sumcheck.borrow_mut().partial_evaluate(&r0);
        lhs_selector.borrow_mut().partial_evaluate(&r0);
        rhs_selector.borrow_mut().partial_evaluate(&r0);

        diff_sumcheck.univariate_polynomial_into(&mut poly);
        debug_assert_eq!(&poly.at_zero() + &poly.at_one(), claim_after_r0);

        // Round 2: fold next variable.
        let r1 = RingElement::constant(11, repr);
        let claim_after_r1 = poly.at(&r1);

        witness_sc.borrow_mut().partial_evaluate(&r1);
        lhs_coeff_sumcheck.borrow_mut().partial_evaluate(&r1);
        rhs_coeff_sumcheck.borrow_mut().partial_evaluate(&r1);
        lhs_selector.borrow_mut().partial_evaluate(&r1);
        rhs_selector.borrow_mut().partial_evaluate(&r1);

        diff_sumcheck.univariate_polynomial_into(&mut poly);
        debug_assert_eq!(&poly.at_zero() + &poly.at_one(), claim_after_r1);

        // Round 3 (final): fold last variable.
        let r2 = RingElement::constant(13, repr);
        let final_claim = poly.at(&r2);

        witness_sc.borrow_mut().partial_evaluate(&r2);
        lhs_coeff_sumcheck.borrow_mut().partial_evaluate(&r2);
        rhs_coeff_sumcheck.borrow_mut().partial_evaluate(&r2);
        lhs_selector.borrow_mut().partial_evaluate(&r2);
        rhs_selector.borrow_mut().partial_evaluate(&r2);

        // we check if MLE[lhs_coeffs](r) * MLE[witness](r) * eq(r, 0b0)
        //     - MLE[rhs_coeffs](r) * MLE[witness](r) * eq(r, 0b0) == final_claim

        let lhs_product = &(lhs_coeff_sumcheck.borrow().final_evaluations()
            * witness_sc.borrow().final_evaluations())
            * lhs_selector.borrow().final_evaluations();
        let rhs_product = &(rhs_coeff_sumcheck.borrow().final_evaluations()
            * witness_sc.borrow().final_evaluations())
            * rhs_selector.borrow().final_evaluations();

        debug_assert_eq!(&lhs_product - &rhs_product, final_claim);
    }
}
