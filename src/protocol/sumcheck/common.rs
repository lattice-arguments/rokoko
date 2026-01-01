use crate::common::ring_arithmetic::RingElement;

pub struct HypercubePoint {
    // We can represent a point in the hypercube as an integer where each bit represents a coordinate
    pub coordinates: usize,
    // TODO: maybe we need some more methods here??
}

pub trait Polynomial {
    fn at_zero(&self) -> RingElement; // at_zero is done separately for efficiency // TODO: maybe we can return by reference??
    fn at_one(&self) -> RingElement; // at_one is done separately for efficiency // TODO: maybe we can return by reference??
    fn at(&self, x: &RingElement) -> RingElement;
}

pub trait Sumcheck<T: Polynomial> {
    fn update_univariate_polynomial(&mut self);
    fn get_univariate_polynomial(&self) -> &T;
    // fn get_claim(&self) -> &RingElement;
    fn get_variable_count(&self) -> usize;
    fn at_hypercube_point(&mut self, point: &HypercubePoint) -> &RingElement;
    fn partial_evaluate(&mut self, value: &RingElement);
}
