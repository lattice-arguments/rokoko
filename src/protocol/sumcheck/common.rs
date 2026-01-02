use std::ops::Index;

use crate::{
    common::ring_arithmetic::RingElement,
    protocol::sumcheck::{hypercube_point::HypercubePoint, polynomial::Polynomial},
};

pub trait SumcheckBaseData:
    Index<HypercubePoint> + HighOrderSumcheckData + Index<HypercubePoint>
{
    fn partial_evaluate(&mut self, value: &RingElement);
    fn final_evaluations(&self) -> &RingElement;
}

pub trait HighOrderSumcheckData {
    fn get_variable_count(&self) -> usize;
    // this is the univariate polynomial for the current variable with the other variables summed out
    // i.e. let a = f(x_0, x_1, ..., x_{n-1}) then this function returns g(x) = sum_{x_1, ..., x_{n-1}} f(x, x_1, ..., x_{n-1})
    fn univariate_polynomial_into(&self, polynomial: &mut Polynomial);

    // this is similar to univariate_polynomial_into but evaluates the polynomial at a given point
    fn univariate_polynomial_at_point_into(
        &self,
        point: &HypercubePoint,
        polynomial: &mut Polynomial,
    );
}
