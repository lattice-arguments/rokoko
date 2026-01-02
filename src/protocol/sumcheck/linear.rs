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
}

impl LinearSumcheck {
    // TODO: think if the pattern is right here
    // The idea is that we first create an empty sumcheck object and then fill it from a source vector
    pub fn new(count: usize, representation: Representation) -> Self {
        LinearSumcheck {
            data: new_vec_zero_preallocated(count),
            variable_count: count.ilog2() as usize,
        }
    }
    pub fn from(&mut self, src: &Vec<RingElement>) {
        self.data.clone_from_slice(src);
    }
}

impl Index<HypercubePoint> for LinearSumcheck {
    type Output = RingElement;

    fn index(&self, index: HypercubePoint) -> &Self::Output {
        &self.data[index.coordinates]
    }
}

impl HighOrderSumcheckData for LinearSumcheck {
    fn univariate_polynomial_into(&self, polynomial: &mut Polynomial) {
        // TODO: optimize this to avoid allocating a temp polynomial each time
        let mut temp = Polynomial::new(2, self.data[0].representation);

        polynomial.coefficients[0].set_zero();
        polynomial.coefficients[1].set_zero();

        for i in 0..self.data.len() / 2 {
            self.univariate_polynomial_at_point_into(&HypercubePoint::new(i), &mut temp);
            polynomial.coefficients[0] += &temp.coefficients[0];
            polynomial.coefficients[1] += &temp.coefficients[1];
        }
    }

    fn univariate_polynomial_at_point_into(
        &self,
        point: &HypercubePoint,
        polynomial: &mut Polynomial,
    ) {
        let half = self.data.len() / 2;
        polynomial.coefficients[0].set_zero();
        polynomial.coefficients[0] += &self.data[point.coordinates]; // constant term
        polynomial.coefficients[1].set_zero();
        polynomial.coefficients[1] += &self.data[point.coordinates + half]; // coeff of x
        polynomial.coefficients[1] -= &self.data[point.coordinates]; // coeff of x
        polynomial.nof_coefficients = 2;
    }

    fn get_variable_count(&self) -> usize {
        self.variable_count
    }
}

impl SumcheckBaseData for LinearSumcheck {
    fn partial_evaluate(&mut self, value: &RingElement) {
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
