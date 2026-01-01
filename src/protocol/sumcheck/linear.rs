use crate::{
    common::{
        config::MOD_Q,
        matrix::new_vec_zero_preallocated,
        ring_arithmetic::{Representation, RingElement},
    },
    protocol::sumcheck::common::{HypercubePoint, Polynomial, Sumcheck},
};

pub struct LinearPolynomial {
    // TODO: maybe we should present this in eval domain instead
    pub coefficients: [RingElement; 2],
}

impl Polynomial for LinearPolynomial {
    fn at_zero(&self) -> RingElement {
        self.coefficients[0].clone()
    }

    fn at_one(&self) -> RingElement {
        &self.coefficients[0] + &self.coefficients[1]
    }

    fn at(&self, x: &RingElement) -> RingElement {
        &self.coefficients[0] + &(&self.coefficients[1] * x)
    }
}

pub struct LinearSumcheck {
    pub data: Vec<RingElement>,
    // this polynomial is stored here to avoid multiple allocations
    univariate_polynomial: LinearPolynomial,
    // sum claim at the current round
    //  claim: RingElement,
    variable_count: usize,
}

impl LinearSumcheck {
    // TODO: think if the pattern is right here
    // The idea is that we first create an empty sumcheck object and then fill it from a source vector
    pub fn new(count: usize, representation: Representation) -> Self {
        LinearSumcheck {
            data: new_vec_zero_preallocated(count),
            univariate_polynomial: LinearPolynomial {
                coefficients: [
                    RingElement::zero(representation),
                    RingElement::zero(representation),
                ],
            },
            // claim: RingElement::zero(representation),
            variable_count: count.ilog2() as usize,
        }
    }
    pub fn from(&mut self, src: &Vec<RingElement>) {
        self.data.clone_from_slice(src);
    }

    // this return univariate so that the most significant bit is a variable of the polynomial
    // fn update_univariate_polynomial_internal(&mut self) {

    // // update the claim
    // if self.variable_count == 0 {
    //     self.claim.set_from(&self.data[0]);
    // } else {
    //     self.claim += (
    //         &self.univariate_polynomial.coefficients[0],
    //         &self.univariate_polynomial.coefficients[1],
    //     );
    // }

    // we have that polynomial(x) = coeffs[0] * (1 - x) + coeffs[1] * x
    // we can rewrite this as polynomial(x) = (coeffs[1] - coeffs[0]) * x + coeffs[0]

    //     let (coeff0, coeff1) = self.univariate_polynomial.coefficients.split_at_mut(1);
    //     coeff1[0] -= &coeff0[0];
    // }
}

impl Sumcheck<LinearPolynomial> for LinearSumcheck {
    fn update_univariate_polynomial(&mut self) {
        let n = self.data.len();

        self.univariate_polynomial.coefficients[0].set_zero();
        self.univariate_polynomial.coefficients[1].set_zero();

        for i in 0..(n / 2) {
            self.univariate_polynomial.coefficients[0] += &self.data[i]; // coefficient for (1 - x)
            self.univariate_polynomial.coefficients[1] += &self.data[i + (n / 2)];
        }

        // we have that polynomial(x) = coeffs[0] * (1 - x) + coeffs[1] * x
        // we can rewrite this as polynomial(x) = (coeffs[1] - coeffs[0]) * x + coeffs[0]

        let (coeff0, coeff1) = self.univariate_polynomial.coefficients.split_at_mut(1);
        coeff1[0] -= &coeff0[0];
    }

    fn get_univariate_polynomial(&self) -> &LinearPolynomial {
        &self.univariate_polynomial
    }

    fn get_variable_count(&self) -> usize {
        self.variable_count
    }

    fn at_hypercube_point(&mut self, point: &HypercubePoint) -> &RingElement {
        &self.data[point.coordinates]
    }

    // we evaluate from the variable at the most significant bit to the least significant bit
    // this is done so that we can truncate the data vector in place
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

    let mut claim = RingElement::constant(36, Representation::IncompleteNTT); // sum of 1 to 8

    // assert_eq!(claim, sc.claim);

    sc.update_univariate_polynomial();

    assert_eq!(
        &sc.univariate_polynomial.at_zero() + &sc.univariate_polynomial.at_one(),
        claim
    );

    let r0 = RingElement::constant(524, Representation::IncompleteNTT);

    claim = sc.univariate_polynomial.at(&r0);

    sc.partial_evaluate(&r0);
    // assert_eq!(claim, sc.claim);

    sc.update_univariate_polynomial();

    assert_eq!(
        &sc.univariate_polynomial.at_zero() + &sc.univariate_polynomial.at_one(),
        claim
    );

    let r1 = RingElement::constant(1337, Representation::IncompleteNTT);

    claim = sc.univariate_polynomial.at(&r1);

    sc.partial_evaluate(&r1);

    sc.update_univariate_polynomial();

    // assert_eq!(claim, sc.claim);

    assert_eq!(
        &sc.univariate_polynomial.at_zero() + &sc.univariate_polynomial.at_one(),
        claim
    );

    let r2 = RingElement::constant(42, Representation::IncompleteNTT);

    claim = sc.univariate_polynomial.at(&r2);

    sc.partial_evaluate(&r2);

    sc.update_univariate_polynomial();

    // assert_eq!(claim, sc.claim);

    assert!(sc.data.len() == 1);

    assert_eq!(&sc.data[0], &claim);

    assert_eq!(
        claim,
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
