/// Compact representation of a vertex in {0,1}^n using an integer bitmask.
#[derive(Clone, Copy, Debug)]
pub struct HypercubePoint {
    // We can represent a point in the hypercube as an integer where each bit represents a coordinate
    pub coordinates: usize,
}

impl HypercubePoint {
    #[inline(always)]
    pub fn new(coordinates: usize) -> Self {
        HypercubePoint { coordinates }
    }

    #[inline(always)]
    pub fn moved(&self, shift: usize) -> Self {
        HypercubePoint {
            coordinates: self.coordinates + shift,
        }
    }

    #[inline(always)]
    pub fn shifted(&self, shift: usize) -> Self {
        HypercubePoint {
            coordinates: self.coordinates >> shift,
        }
    }

    #[inline(always)]
    pub fn masked(&self, mask: usize) -> Self {
        HypercubePoint {
            coordinates: self.coordinates & mask,
        }
    }
}
