use std::{cell::RefCell, ops::Index};

use crate::{
    common::{
        ring_arithmetic::{Representation, RingElement},
        sumcheck_element::SumcheckElement,
    },
    protocol::sumcheck::{
        common::{HighOrderSumcheckData, SumcheckBaseData},
        hypercube_point::HypercubePoint,
        polynomial::Polynomial,
    },
};

#[cfg(test)]
use crate::common::config::MOD_Q;

/// Standard linear sumcheck over a vector that represents a multilinear extension.
pub struct LinearSumcheck<E: SumcheckElement = RingElement> {
    pub data: Vec<E>,
    variable_count: usize,
    index_mask: usize,
    suffix: usize,
    poly_scratch: RefCell<Polynomial<E>>,
}

impl<E: SumcheckElement> LinearSumcheck<E> {
    pub fn new(count: usize) -> Self {
        Self::new_with_prefixed_sufixed_data(count, 0, 0)
    }

    // here we want to handle the case when
    // we have f(x_0, x_1, ..., x_{n-1}, y_0, y_1, ..., y_{m-1}) = MLE[data](y_0, y_1, ..., y_{m-1})
    // so we need to prefix the variable count with n extra variables
    // This will be used if we have a large data, but our claim is only on a subset of the variables
    // for example, if we have a data of size 2^{10} but our claim is only on 2^{6} points,
    // Then, imagine we have a claim <s, subdata> = v, where subdata is of size 2^{6}
    // Since we want to run larger sumchecks on the full data, we need to embed subdata into data
    // so we have \sum_{z \in HC^10} MLE[data](z) * eq(x_0, ... x_3, SEL) * MLE[s](x_4 ... x_9) = v
    // where SEL is the selection vector that selects the subdata from data.
    // to conveniently operate we will assume that MLE[s](x_4 ... x_9) is prefixed with 4 dummy variables x_0 ... x_3
    pub fn new_with_prefixed_sufixed_data(
        count: usize,
        prefix_size: usize, // number of extra most significant dummy variables
        suffix_size: usize, // number of extra least significant dummy variables
    ) -> Self {
        LinearSumcheck {
            data: E::allocate_zero_vec(count),
            variable_count: count.ilog2() as usize + prefix_size + suffix_size,
            index_mask: count - 1, // this mask will be used to ignore prefixed variables
            poly_scratch: RefCell::new(Polynomial::new(2)),
            suffix: suffix_size,
        }
    }
    /// Populate the internal buffer with the provided values.
    pub fn load_from(&mut self, src: &[E]) {
        self.data.clone_from_slice(src);
    }
}

impl<E: SumcheckElement> Index<HypercubePoint> for LinearSumcheck<E> {
    type Output = E;

    fn index(&self, index: HypercubePoint) -> &Self::Output {
        let index_masked = index.shifted(self.suffix).masked(self.index_mask);
        &self.data[index_masked.coordinates] // masking to ignore prefixed variables
    }
}

impl<E: SumcheckElement> HighOrderSumcheckData for LinearSumcheck<E> {
    type Element = E;

    fn get_scratch_poly(&self) -> &RefCell<Polynomial<E>> {
        &self.poly_scratch
    }
    fn max_num_polynomial_coefficients(&self) -> usize {
        2
    }

    fn univariate_polynomial_at_point_into(
        &self,
        point: HypercubePoint,
        polynomial: &mut Polynomial<E>,
    ) {
        // Current round splits the hypercube into two halves depending on the
        // value of the highest-order variable.
        let len = 1 << self.variable_count;
        let half = len / 2;
        polynomial.coefficients[0].set_zero();
        polynomial.coefficients[0] += &self[point]; // constant term

        if self.variable_count > self.data.len().trailing_zeros() as usize + self.suffix {
            // we have some prefixed variables
            polynomial.num_coefficients = 1;
            return;
        }

        if self.data.len() == 1 && self.suffix > 0 {
            // only suffix variables remain
            polynomial.num_coefficients = 1;
            return;
        }

        polynomial.coefficients[1].set_zero();
        polynomial.coefficients[1] += &self[point.moved(half)]; // coeff of x
        polynomial.coefficients[1] -= &self[point]; // coeff of x
        polynomial.num_coefficients = 2;
    }

    fn is_univariate_polynomial_zero_at_point(&self, _point: HypercubePoint) -> bool {
        false // even if the polynomial is zero, we still need to perform the folding as normal
    }

    fn variable_count(&self) -> usize {
        self.variable_count
    }
}

impl<E: SumcheckElement> SumcheckBaseData for LinearSumcheck<E> {
    fn partial_evaluate(&mut self, value: &E) {
        // Fold the highest-order variable using the provided random challenge.
        // When there are prefixed variables, they are ignored and the claim is
        // scaled accordingly.
        if self.variable_count > self.data.len().trailing_zeros() as usize + self.suffix {
            // we have some prefixed variables
            self.variable_count -= 1;
            return;
        }

        if self.data.len() == 1 && self.suffix > 0 {
            // only suffix variables remain
            self.variable_count -= 1;
            self.suffix -= 1;
            return;
        }

        let n = self.data.len();
        if n % 2 != 0 {
            panic!("Sumcheck data length must be a power of 2");
        }
        let (left_half, right_half) = self.data.split_at_mut(n / 2);
        // For each pair (a, b) corresponding to variable values 0 and 1,
        // compute a + (b - a) * r, overwriting the left half in place.
        for i in 0..(n / 2) {
            right_half[i] -= &left_half[i];
            right_half[i] *= value;
            left_half[i] += &right_half[i];
        }
        self.data.truncate(n / 2);
        self.variable_count -= 1;
    }

    fn final_evaluations(&self) -> &E {
        if self.data.len() != 1 {
            panic!("Sumcheck is not fully evaluated yet");
        }
        &self.data[0]
    }
}

#[test]
fn test_linear_sumcheck() {
    use crate::common::ring_arithmetic::RingElement;

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

    let mut sumcheck = LinearSumcheck::new(data.len());
    sumcheck.load_from(&data);

    // Simulate three verifier challenges and ensure the running claim matches
    // the multilinear extension evaluated at the folded point.

    let r0 = RingElement::constant(524, Representation::IncompleteNTT);

    sumcheck.partial_evaluate(&r0);

    let r1 = RingElement::constant(1337, Representation::IncompleteNTT);

    sumcheck.partial_evaluate(&r1);

    let r2 = RingElement::constant(42, Representation::IncompleteNTT);

    sumcheck.partial_evaluate(&r2);

    assert!(sumcheck.data.len() == 1);

    assert_eq!(
        sumcheck.data[0],
        RingElement::constant(
            (MOD_Q as i64
                + 1 * (1 - 42) * (1 - 1337) * (1 - 524)
                + 2 * 42 * (1 - 1337) * (1 - 524)
                + 3 * (1 - 42) * 1337 * (1 - 524)
                + 4 * 42 * 1337 * (1 - 524)
                + 5 * (1 - 42) * (1 - 1337) * 524
                + 6 * 42 * (1 - 1337) * 524
                + 7 * (1 - 42) * 1337 * 524
                + 8 * 42 * 1337 * 524) as u64,
            Representation::IncompleteNTT
        )
    )
}

#[test]
fn test_linear_sumcheck_univariate_polynomial() {
    use crate::common::ring_arithmetic::RingElement;

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

    let mut sumcheck = LinearSumcheck::new(data.len());
    sumcheck.load_from(&data);

    let mut poly = Polynomial::new(2);

    // First round polynomial should encode how the highest-order variable
    // toggles between the left and right halves of the data vector.
    sumcheck.univariate_polynomial_into(&mut poly);

    // poly 1 + (5 - 1) * x + 2 + (6 - 2) * x + 3 + (7 - 3) * x + 4 + (8 - 4) * x

    assert_eq!(
        poly.coefficients[0],
        RingElement::constant(1 + 2 + 3 + 4, Representation::IncompleteNTT)
    ); // sum of all elements

    assert_eq!(
        poly.coefficients[1],
        RingElement::constant(
            (5 - 1) + (6 - 2) + (7 - 3) + (8 - 4),
            Representation::IncompleteNTT
        )
    ); // computed manually
}

#[test]
fn test_masked_sumcheck_indexing() {
    use crate::common::ring_arithmetic::RingElement;

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

    let mut sumcheck = LinearSumcheck::new_with_prefixed_sufixed_data(data.len(), 2, 0);
    sumcheck.load_from(&data);

    // Now, the sumcheck has 2 prefixed variables, so when we index with HypercubePoint.

    let mut poly = Polynomial::new(0);

    sumcheck.univariate_polynomial_into(&mut poly);

    // the first polynomial is over x_0 which is prefixed and should be ignored.
    // Therefore, the polynomial should be a constant equal to the sum of all data points times 2.
    // The factor of 2 comes from the fact that the sumcheck claim is has to account for dummy variables.
    assert_eq!(
        poly.coefficients[0],
        RingElement::constant(
            (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) * 2,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        poly.coefficients[1],
        RingElement::constant(0, Representation::IncompleteNTT)
    );

    assert_eq!(poly.num_coefficients, 1);

    let mut claim = poly.at_zero();
    claim += &poly.at_one();

    assert_eq!(
        claim,
        RingElement::constant(
            (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) * 4,
            Representation::IncompleteNTT,
        )
    );

    let r0 = RingElement::constant(524, Representation::IncompleteNTT);

    let new_claim = poly.at(&r0);

    assert_eq!(
        new_claim,
        RingElement::constant(
            (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) * 2,
            Representation::IncompleteNTT
        )
    );

    sumcheck.partial_evaluate(&r0);

    sumcheck.univariate_polynomial_into(&mut poly);

    assert_eq!(
        poly.coefficients[0],
        RingElement::constant(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8, Representation::IncompleteNTT)
    );

    assert_eq!(
        poly.coefficients[1],
        RingElement::constant(0, Representation::IncompleteNTT)
    );

    assert_eq!(poly.num_coefficients, 1);

    assert_eq!(&poly.at_zero() + &poly.at_one(), new_claim);

    let r1 = RingElement::constant(1337, Representation::IncompleteNTT);

    let new_claim = poly.at(&r1);

    assert_eq!(
        new_claim,
        RingElement::constant(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8, Representation::IncompleteNTT)
    );

    sumcheck.partial_evaluate(&r1);

    sumcheck.univariate_polynomial_into(&mut poly);

    assert_eq!(
        poly.coefficients[0],
        RingElement::constant(1 + 2 + 3 + 4, Representation::IncompleteNTT)
    );

    assert_eq!(
        poly.coefficients[1],
        RingElement::constant(
            (5 - 1) + (6 - 2) + (7 - 3) + (8 - 4),
            Representation::IncompleteNTT
        )
    );

    assert_eq!(poly.num_coefficients, 2);

    assert_eq!(&poly.at_zero() + &poly.at_one(), new_claim);

    let r2 = RingElement::constant(42, Representation::IncompleteNTT);

    let new_claim = poly.at(&r2);

    sumcheck.partial_evaluate(&r2);

    sumcheck.univariate_polynomial_into(&mut poly);

    assert_eq!(&poly.at_zero() + &poly.at_one(), new_claim);

    let r3 = RingElement::constant(7, Representation::IncompleteNTT);

    let new_claim = poly.at(&r3);

    sumcheck.partial_evaluate(&r3);

    sumcheck.univariate_polynomial_into(&mut poly);

    assert_eq!(&poly.at_zero() + &poly.at_one(), new_claim);

    let r4 = RingElement::constant(19, Representation::IncompleteNTT);

    let new_claim = poly.at(&r4);

    sumcheck.partial_evaluate(&r4);

    assert!(sumcheck.data.len() == 1);

    // now, we make a final check using r2, r3, r4
    assert_eq!(
        sumcheck.final_evaluations(),
        &RingElement::constant(
            (MOD_Q as i64
                + 1 * (1 - 19) * (1 - 7) * (1 - 42)
                + 2 * 19 * (1 - 7) * (1 - 42)
                + 3 * (1 - 19) * 7 * (1 - 42)
                + 4 * 19 * 7 * (1 - 42)
                + 5 * (1 - 19) * (1 - 7) * 42
                + 6 * 19 * (1 - 7) * 42
                + 7 * (1 - 19) * 7 * 42
                + 8 * 19 * 7 * 42) as u64,
            Representation::IncompleteNTT
        )
    );

    assert_eq!(&new_claim, sumcheck.final_evaluations());
}

#[test]
fn test_linear_sumcheck_with_suffixed_data() {
    use crate::common::ring_arithmetic::RingElement;

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

    // we head with a vector that has 2 suffixed variables, i.e. (1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7, 8,8,8,8)

    let mut sumcheck = LinearSumcheck::new_with_prefixed_sufixed_data(data.len(), 0, 2);
    sumcheck.load_from(&data);

    // Now, the sumcheck has 2 suffixed variables, so when we index with HypercubePoint.

    let mut poly = Polynomial::new(0);

    sumcheck.univariate_polynomial_into(&mut poly);

    let claim = RingElement::constant(
        (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) * 4,
        Representation::IncompleteNTT,
    );

    assert_eq!(&poly.at_zero() + &poly.at_one(), claim);
}
