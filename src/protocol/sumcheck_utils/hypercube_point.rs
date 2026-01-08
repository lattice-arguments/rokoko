/// Compact representation of a vertex in {0,1}^n using an integer bitmask.
#[derive(Clone, Copy, Debug)]
pub struct HypercubePoint {
    // We can represent a point in the hypercube as an integer where each bit represents a coordinate
    pub coordinates: usize,
}

impl HypercubePoint {
    pub fn new(coordinates: usize) -> Self {
        HypercubePoint { coordinates }
    }

    pub fn moved(&self, shift: usize) -> Self {
        HypercubePoint {
            coordinates: self.coordinates + shift,
        }
    }

    pub fn shifted(&self, shift: usize) -> Self {
        HypercubePoint {
            coordinates: self.coordinates >> shift,
        }
    }

    pub fn masked(&self, mask: usize) -> Self {
        HypercubePoint {
            coordinates: self.coordinates & mask,
        }
    }
}
