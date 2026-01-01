use crate::{
    common::{
        config::MOD_Q,
        ring_arithmetic::{Representation, RingElement},
    },
    protocol::sumcheck::{
        common::{Polynomial, Sumcheck},
        linear::LinearSumcheck,
    },
};

pub struct InnerProductSumcheck<'a> {
    pub sumcheck_0: &'a mut LinearSumcheck, // we don't want ownership here
    pub sumcheck_1: &'a mut LinearSumcheck,
    pub univariate_polynomial: QuadraticPolynomial,
    // sum claim at the current round
    pub claim: RingElement,
    // pub variable_count: usize,
    __hypercube_point: RingElement, // to store the evaluation point temporarily // TODO: maybe we can avoid this? This smells bad
    __a_diff: RingElement,
    __b_diff: RingElement,
    __a0b0: RingElement,
    __diff_prod: RingElement,
}

impl InnerProductSumcheck<'_> {
    pub fn new<'a>(
        sumcheck_0: &'a mut LinearSumcheck,
        sumcheck_1: &'a mut LinearSumcheck,
    ) -> InnerProductSumcheck<'a> {
        assert_eq!(
            sumcheck_0.data.len(),
            sumcheck_1.data.len(),
            "Inner product sumcheck: both sumchecks must have the same data length"
        );

        assert_eq!(
            sumcheck_0.data[0].representation, sumcheck_1.data[0].representation,
            "Inner product sumcheck: both sumchecks must have the same representation"
        );

        assert_eq!(
            sumcheck_0.get_variable_count(),
            sumcheck_1.get_variable_count(),
            "Inner product sumcheck: both sumchecks must have the same variable count"
        );

        let rep = sumcheck_0.data[0].representation;

        InnerProductSumcheck {
            sumcheck_0,
            sumcheck_1,
            univariate_polynomial: QuadraticPolynomial {
                coefficients: [
                    RingElement::zero(rep),
                    RingElement::zero(rep),
                    RingElement::zero(rep),
                ],
            },
            claim: RingElement::zero(rep),
            __hypercube_point: RingElement::zero(rep),
            __a_diff: RingElement::zero(rep),
            __b_diff: RingElement::zero(rep),
            __a0b0: RingElement::zero(rep),
            __diff_prod: RingElement::zero(rep),
        }
    }
}

struct QuadraticPolynomial {
    pub coefficients: [RingElement; 3],
}

impl Polynomial for QuadraticPolynomial {
    fn at_zero(&self) -> RingElement {
        self.coefficients[0].clone()
    }

    fn at_one(&self) -> RingElement {
        // TODO: optimize memory allocations here
        &(&self.coefficients[0] + &self.coefficients[1]) + &self.coefficients[2]
    }

    fn at(&self, x: &RingElement) -> RingElement {
        // TODO: optimize memory allocations here
        let x_squared = x * x;
        &(&self.coefficients[0] + &(x * &self.coefficients[1]))
            + &(&x_squared * &self.coefficients[2])
    }
}

impl Sumcheck<QuadraticPolynomial> for InnerProductSumcheck<'_> {
    fn get_univariate_polynomial(&self) -> &QuadraticPolynomial {
        &self.univariate_polynomial
    }

    fn update_univariate_polynomial(&mut self) {
        // this is a bit messy but we want to avoid allocations as much as possible here
        self.univariate_polynomial.coefficients[0].set_zero();
        self.univariate_polynomial.coefficients[1].set_zero();
        self.univariate_polynomial.coefficients[2].set_zero();

        let n = self.sumcheck_0.data.len();
        let half = n / 2;

        for i in 0..half {
            let a0 = &self.sumcheck_0.data[i];
            let a1 = &self.sumcheck_0.data[i + half];
            let b0 = &self.sumcheck_1.data[i];
            let b1 = &self.sumcheck_1.data[i + half];
            self.__a_diff -= (a1, a0); // A(1) - A(0)
            self.__b_diff -= (b1, b0); // B(1) - B(0)
                                       // P(x) = (a0 + a_diff * x) * (b0 + b_diff * x)
            self.__a0b0 *= (a0, b0);
            self.__diff_prod *= (&self.__a_diff, &self.__b_diff);
            self.univariate_polynomial.coefficients[0] += &self.__a0b0;
            self.univariate_polynomial.coefficients[2] += &self.__diff_prod;
            self.__a_diff *= b0;
            self.__b_diff *= a0;
            self.univariate_polynomial.coefficients[1] += &self.__a_diff;
            self.univariate_polynomial.coefficients[1] += &self.__b_diff;
        }
    }

    fn get_variable_count(&self) -> usize {
        self.sumcheck_0.get_variable_count()
    }

    fn at_hypercube_point(
        &mut self,
        point: &crate::protocol::sumcheck::common::HypercubePoint,
    ) -> &RingElement {
        let a = self.sumcheck_0.at_hypercube_point(point);
        let b = self.sumcheck_1.at_hypercube_point(point);
        self.__hypercube_point *= (a, b);
        &self.__hypercube_point // better use it fast before next calls idk
    }

    fn partial_evaluate(&mut self, value: &RingElement) {
        self.sumcheck_0.partial_evaluate(value); // TODO: I think we need to introduce some "semaphore" here to avoid partial evaluations being called from different sumchecks.
        self.sumcheck_1.partial_evaluate(value);

        // update the polynomial with the new folded data
        self.update_univariate_polynomial();

        // update the claim
        self.claim = self.univariate_polynomial.at(value);
    }
}

#[test]
fn test_inner_product_sumcheck() {
    let data_0 = vec![
        RingElement::constant(1, Representation::IncompleteNTT),
        RingElement::constant(2, Representation::IncompleteNTT),
        RingElement::constant(3, Representation::IncompleteNTT),
        RingElement::constant(4, Representation::IncompleteNTT),
        RingElement::constant(5, Representation::IncompleteNTT),
        RingElement::constant(6, Representation::IncompleteNTT),
        RingElement::constant(7, Representation::IncompleteNTT),
        RingElement::constant(8, Representation::IncompleteNTT),
    ];

    let data_1 = vec![
        RingElement::constant(9, Representation::IncompleteNTT),
        RingElement::constant(10, Representation::IncompleteNTT),
        RingElement::constant(11, Representation::IncompleteNTT),
        RingElement::constant(12, Representation::IncompleteNTT),
        RingElement::constant(13, Representation::IncompleteNTT),
        RingElement::constant(14, Representation::IncompleteNTT),
        RingElement::constant(15, Representation::IncompleteNTT),
        RingElement::constant(16, Representation::IncompleteNTT),
    ];

    let mut sumcheck_0 = LinearSumcheck::new(data_0.len(), data_0[0].representation);
    sumcheck_0.from(&data_0);
    let mut sumcheck_1 = LinearSumcheck::new(data_1.len(), data_1[0].representation);
    sumcheck_1.from(&data_1);

    let mut inner_product_sumcheck = InnerProductSumcheck::new(&mut sumcheck_0, &mut sumcheck_1);

    // sumcheck execution

    let mut claim = RingElement::constant(
        1 * 9 + 2 * 10 + 3 * 11 + 4 * 12 + 5 * 13 + 6 * 14 + 7 * 15 + 8 * 16,
        Representation::IncompleteNTT,
    );

    inner_product_sumcheck.update_univariate_polynomial();

    assert_eq!(
        &inner_product_sumcheck.univariate_polynomial.at_zero()
            + &inner_product_sumcheck.univariate_polynomial.at_one(),
        claim
    );

    let r0 = RingElement::constant(524, Representation::IncompleteNTT);

    claim = inner_product_sumcheck.univariate_polynomial.at(&r0);

    inner_product_sumcheck.partial_evaluate(&r0);

    inner_product_sumcheck.update_univariate_polynomial();

    assert_eq!(
        &inner_product_sumcheck.univariate_polynomial.at_zero()
            + &inner_product_sumcheck.univariate_polynomial.at_one(),
        claim
    );

    let r1 = RingElement::constant(1337, Representation::IncompleteNTT);
    claim = inner_product_sumcheck.univariate_polynomial.at(&r1);
    inner_product_sumcheck.partial_evaluate(&r1);
    inner_product_sumcheck.update_univariate_polynomial();
    assert_eq!(
        &inner_product_sumcheck.univariate_polynomial.at_zero()
            + &inner_product_sumcheck.univariate_polynomial.at_one(),
        claim
    );

    let r2 = RingElement::constant(42, Representation::IncompleteNTT);
    claim = inner_product_sumcheck.univariate_polynomial.at(&r2);
    inner_product_sumcheck.partial_evaluate(&r2);
    inner_product_sumcheck.update_univariate_polynomial();

    assert!(inner_product_sumcheck.get_variable_count() == 0);

    // assert_eq!(&inner_product_sumcheck.data[0], &claim);

    // We started from \sum_{z \in hypercube} A(z) * B(z)
    // After 3 rounds of partial evaluations, we should have claim = A(r) * B(r)
    // where r = (r0, r1, r2)
    assert_eq!(
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
