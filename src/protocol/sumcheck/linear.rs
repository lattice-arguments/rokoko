use crate::{
    common::{
        config::MOD_Q,
        matrix::new_vec_zero_preallocated,
        ring_arithmetic::{Representation, RingElement},
    },
    protocol::sumcheck::common::{HypercubePoint, Polynomial, SumcheckBaseData},
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

impl SumcheckBaseData for LinearSumcheck {
    fn get_variable_count(&self) -> usize {
        self.variable_count
    }

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
