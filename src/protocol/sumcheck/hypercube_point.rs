pub struct HypercubePoint {
    // We can represent a point in the hypercube as an integer where each bit represents a coordinate
    pub coordinates: usize,
    // TODO: maybe we need some more methods here??
}

impl HypercubePoint {
    pub fn new(coordinates: usize) -> Self {
        HypercubePoint { coordinates }
    }
}
