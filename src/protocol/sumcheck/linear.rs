use std::ops::Index;

use crate::{
    common::{
        config::MOD_Q,
        matrix::new_vec_zero_preallocated,
        ring_arithmetic::{Representation, RingElement},
    },
    protocol::sumcheck::{
        common::{HighOrderSumcheckData, SumcheckBaseData},
        hypercube_point::HypercubePoint,
        polynomial::Polynomial,
    },
};

pub struct LinearSumcheck {
    pub data: Vec<RingElement>,
    variable_count: usize,
    mask: usize,
}

impl LinearSumcheck {
    // TODO: think if the pattern is right here
    // The idea is that we first create an empty sumcheck object and then fill it from a source vector
    pub fn new(count: usize, representation: Representation) -> Self {
        LinearSumcheck {
            data: new_vec_zero_preallocated(count),
            variable_count: count.ilog2() as usize,
            mask: count - 1, // this mask does nothing here
        }
    }

    // here we want to handle the case when
    // we have f(x_0, x_1, ..., x_{n-1}, y_0, y_1, ..., y_{m-1}) = MLE[data](y_0, y_1, ..., y_{m-1})
    // so we need to preffix the variable count with n extra variables
    // This will be used if we have a large data, but our claim is only on a subset of the variables
    // for example, if we have a data of size 2^{10} but our claim is only on 2^{6} points,
    // Then, imagine we have a claim <s, subdata> = v, where subdata is of size 2^{6}
    // Since we want to run larger sumchecks on the full data, we need to embed subdata into data
    // so we have \sum_{z \in HC^10} MLE[data](z) * eq(x_0, ... x_3, SEL) * MLE[s](x_4 ... x_9) = v
    // where SEL is the selection vector that selects the subdata from data.
    // to convenietly operate we will assume that MLE[s](x_4 ... x_9) is preffixed with 4 dummy variables x_0 ... x_3
    pub fn new_with_preffixed_data(
        count: usize,
        representation: Representation,
        preffix: usize, // number of extra dummy variables
    ) -> Self {
        LinearSumcheck {
            data: new_vec_zero_preallocated(count),
            variable_count: count.ilog2() as usize + preffix,
            mask: count - 1, // this mask will be used to ignore preffixed variables
        }
    }
    pub fn from(&mut self, src: &Vec<RingElement>) {
        self.data.clone_from_slice(src);
    }
}

impl Index<HypercubePoint> for LinearSumcheck {
    type Output = RingElement;

    fn index(&self, index: HypercubePoint) -> &Self::Output {
        let index_masked = index.masked(self.mask);
        &self.data[index_masked.coordinates] // masking to ignore preffixed variables
    }
}

impl HighOrderSumcheckData for LinearSumcheck {
    fn nof_polynomial_coefficients(&self) -> usize {
        2
    }

    fn univariate_polynomial_at_point_into(
        &self,
        point: HypercubePoint,
        polynomial: &mut Polynomial,
    ) -> bool {
        let len = 1 << self.variable_count;
        let half = len / 2;
        polynomial.coefficients[0].set_zero();
        polynomial.coefficients[0] += &self[point]; // constant term
        polynomial.coefficients[1].set_zero();
        polynomial.coefficients[1] += &self[point.moved(half)]; // coeff of x
        polynomial.coefficients[1] -= &self[point]; // coeff of x
        polynomial.nof_coefficients = 2;

        true
    }

    fn variable_count(&self) -> usize {
        self.variable_count
    }
}

impl SumcheckBaseData for LinearSumcheck {
    fn partial_evaluate(&mut self, value: &RingElement) {
        if self.variable_count > self.data.len().trailing_zeros() as usize {
            // we have some preffixed variables
            self.variable_count -= 1;
            return;
        }

        let n = self.data.len();
        if n % 2 != 0 {
            panic!("Sumcheck data length must be a power of 2");
        }
        let (left_half, right_half) = self.data.split_at_mut(n / 2);
        for i in 0..(n / 2) {
            right_half[i] -= &left_half[i];
            right_half[i] *= value;
            left_half[i] += &right_half[i];
        }
        self.data.truncate(n / 2);
        self.variable_count -= 1;
    }

    fn final_evaluations(&self) -> &RingElement {
        if self.data.len() != 1 {
            panic!("Sumcheck is not fully evaluated yet");
        }
        &self.data[0]
    }
}

#[test]
fn test_linear_sumcheck() {
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

    let mut sc = LinearSumcheck::new(data.len(), data[0].representation);
    sc.from(&data);

    // sumcheck execution

    let r0 = RingElement::constant(524, Representation::IncompleteNTT);

    sc.partial_evaluate(&r0);

    let r1 = RingElement::constant(1337, Representation::IncompleteNTT);

    sc.partial_evaluate(&r1);

    let r2 = RingElement::constant(42, Representation::IncompleteNTT);

    sc.partial_evaluate(&r2);

    assert!(sc.data.len() == 1);

    assert_eq!(
        sc.data[0],
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

    let mut sc = LinearSumcheck::new(data.len(), data[0].representation);
    sc.from(&data);

    let mut poly = Polynomial::new(2, data[0].representation);

    sc.univariate_polynomial_into(&mut poly);

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

    let mut sc = LinearSumcheck::new_with_preffixed_data(data.len(), data[0].representation, 2);
    sc.from(&data);

    // Now, the sumcheck has 2 preffixed variables, so when we index with HypercubePoint.

    let mut poly = Polynomial::new(0, data[0].representation);

    sc.univariate_polynomial_into(&mut poly);

    // the first polynomial is over x_0 which is preffixed and should be ignored.
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

    let claim = RingElement::constant(
        (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) * 4,
        Representation::IncompleteNTT,
    );

    assert_eq!(&poly.at_zero() + &poly.at_one(), claim);

    let r0 = RingElement::constant(524, Representation::IncompleteNTT);

    let new_claim = poly.at(&r0);

    assert_eq!(
        new_claim,
        RingElement::constant(
            (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) * 2,
            Representation::IncompleteNTT
        )
    );

    sc.partial_evaluate(&r0);

    sc.univariate_polynomial_into(&mut poly);

    assert_eq!(
        poly.coefficients[0],
        RingElement::constant(
            (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8),
            Representation::IncompleteNTT
        )
    );

    assert_eq!(
        poly.coefficients[1],
        RingElement::constant(0, Representation::IncompleteNTT)
    );

    assert_eq!(&poly.at_zero() + &poly.at_one(), new_claim);

    let r1 = RingElement::constant(1337, Representation::IncompleteNTT);

    let new_claim = poly.at(&r1);

    assert_eq!(
        new_claim,
        RingElement::constant(
            (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8),
            Representation::IncompleteNTT
        )
    );

    sc.partial_evaluate(&r1);

    sc.univariate_polynomial_into(&mut poly);

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

    // and let's run it until the end

    assert_eq!(&poly.at_zero() + &poly.at_one(), new_claim);

    let r2 = RingElement::constant(42, Representation::IncompleteNTT);

    let new_claim = poly.at(&r2);

    sc.partial_evaluate(&r2);

    sc.univariate_polynomial_into(&mut poly);

    assert_eq!(&poly.at_zero() + &poly.at_one(), new_claim);

    let r3 = RingElement::constant(7, Representation::IncompleteNTT);

    let new_claim = poly.at(&r3);

    sc.partial_evaluate(&r3);

    sc.univariate_polynomial_into(&mut poly);

    assert_eq!(&poly.at_zero() + &poly.at_one(), new_claim);

    let r4 = RingElement::constant(19, Representation::IncompleteNTT);

    let new_claim = poly.at(&r4);

    sc.partial_evaluate(&r4);

    assert!(sc.data.len() == 1);

    // now, we make a final check using r2, r3, r4
    assert_eq!(
        sc.final_evaluations(),
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

    assert_eq!(&new_claim, sc.final_evaluations());
}
