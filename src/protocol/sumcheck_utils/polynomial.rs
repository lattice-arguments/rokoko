use std::cmp::max;

use crate::common::{ring_arithmetic::RingElement, sumcheck_element::SumcheckElement};

/// Univariate polynomial in EVALUATION form: `coefficients[i]` is `P(i)`, NOT a
/// monomial coefficient. `num_coefficients` is the evaluation count (degree + 1).
#[derive(Clone, Debug)]
pub struct Polynomial<E: SumcheckElement = RingElement> {
    pub coefficients: [E; 4],
    pub num_coefficients: usize,
}

/// Extend `evals[0..k]` (the evaluations of a degree-`k-1` polynomial) to
/// `evals[0..to]` by forward differences: the `k`-th difference vanishes, so
/// `f(n) = sum_{j=1}^{k} (-1)^{j+1} C(k,j) f(n-j)`.
#[inline]
fn extrapolate<E: SumcheckElement>(evals: &mut [E; 4], k: usize, to: usize) {
    for n in k..to {
        match k {
            0 => {}
            1 => evals[n] = evals[n - 1].clone(),
            2 => {
                let mut t = evals[n - 1].clone();
                t += &evals[n - 1];
                t -= &evals[n - 2];
                evals[n] = t;
            }
            3 => {
                let mut t = evals[n - 1].clone();
                t += &evals[n - 1];
                t += &evals[n - 1];
                let mut s = evals[n - 2].clone();
                s += &evals[n - 2];
                s += &evals[n - 2];
                t -= &s;
                t += &evals[n - 3];
                evals[n] = t;
            }
            _ => unreachable!("degree above cubic is unsupported"),
        }
    }
}

impl<E: SumcheckElement> Polynomial<E> {
    pub fn new(num_coefficients: usize) -> Self {
        debug_assert!(num_coefficients <= 4);
        Polynomial {
            coefficients: std::array::from_fn(|_| E::zero()),
            num_coefficients,
        }
    }

    #[inline]
    pub fn at_zero(&self) -> E {
        self.coefficients[0].clone()
    }

    #[inline]
    pub fn at_one(&self) -> E {
        if self.num_coefficients >= 2 {
            self.coefficients[1].clone()
        } else {
            self.coefficients[0].clone()
        }
    }

    /// Lagrange interpolation over nodes `0..d` (only general point needs it).
    pub fn at(&self, point: &E) -> E {
        match self.num_coefficients {
            0 => E::zero(),
            1 => self.coefficients[0].clone(),
            2 => {
                let mut t = self.coefficients[1].clone();
                t -= &self.coefficients[0];
                t *= point;
                t += &self.coefficients[0];
                t
            }
            3 => {
                let inv2 = E::inv_two_ref();
                let mut rm1 = point.clone();
                rm1 -= E::one_ref();
                let mut rm2 = point.clone();
                rm2 -= E::two_ref();
                let mut t0 = rm1.clone();
                t0 *= &rm2;
                t0 *= &self.coefficients[0];
                t0 *= inv2;
                let mut t1 = point.clone();
                t1 *= &rm2;
                t1 *= &self.coefficients[1];
                let mut t2 = point.clone();
                t2 *= &rm1;
                t2 *= &self.coefficients[2];
                t2 *= inv2;
                t0 -= &t1;
                t0 += &t2;
                t0
            }
            _ => {
                let inv2 = E::inv_two_ref();
                let inv6 = E::inv_six_ref();
                let mut three = E::two_ref().clone();
                three += E::one_ref();
                let mut rm1 = point.clone();
                rm1 -= E::one_ref();
                let mut rm2 = point.clone();
                rm2 -= E::two_ref();
                let mut rm3 = point.clone();
                rm3 -= &three;
                let mut t0 = rm1.clone();
                t0 *= &rm2;
                t0 *= &rm3;
                t0 *= &self.coefficients[0];
                t0 *= inv6;
                let mut t1 = point.clone();
                t1 *= &rm2;
                t1 *= &rm3;
                t1 *= &self.coefficients[1];
                t1 *= inv2;
                let mut t2 = point.clone();
                t2 *= &rm1;
                t2 *= &rm3;
                t2 *= &self.coefficients[2];
                t2 *= inv2;
                let mut t3 = point.clone();
                t3 *= &rm1;
                t3 *= &rm2;
                t3 *= &self.coefficients[3];
                t3 *= inv6;
                // P(r) = -t0 + t1 - t2 + t3
                t1 += &t3;
                t1 -= &t0;
                t1 -= &t2;
                t1
            }
        }
    }

    pub fn set_zero(&mut self) {
        for coeff in self.coefficients.iter_mut() {
            coeff.set_zero();
        }
        self.num_coefficients = 0;
    }

    /// Copy the contents of `other` into `self`.
    #[inline]
    pub fn copy_from(&mut self, other: &Polynomial<E>) {
        self.num_coefficients = other.num_coefficients;
        for i in 0..other.num_coefficients {
            self.coefficients[i].set_from(&other.coefficients[i]);
        }
    }
}

/// Multiply two polynomials and store the result in `result`. In evaluation
/// form: extrapolate both factors to the joint degree, then multiply pointwise.
#[inline]
pub fn mul_poly_into<E: SumcheckElement>(
    result: &mut Polynomial<E>,
    poly_0: &Polynomial<E>,
    poly_1: &Polynomial<E>,
) {
    let d0 = poly_0.num_coefficients;
    let d1 = poly_1.num_coefficients;
    let m = d0 + d1 - 1;
    debug_assert!(m <= 4, "resulting polynomial degree exceeds supported maximum");

    let mut e0: [E; 4] =
        std::array::from_fn(|i| if i < d0 { poly_0.coefficients[i].clone() } else { E::zero() });
    extrapolate(&mut e0, d0, m);
    let mut e1: [E; 4] =
        std::array::from_fn(|i| if i < d1 { poly_1.coefficients[i].clone() } else { E::zero() });
    extrapolate(&mut e1, d1, m);

    for i in 0..m {
        result.coefficients[i] *= (&e0[i], &e1[i]);
    }
    result.num_coefficients = m;
}

/// Add two polynomials and store the sum in `result`.
pub fn add_poly_into(result: &mut Polynomial, poly_0: &Polynomial, poly_1: &Polynomial) {
    result.copy_from(poly_0);
    add_poly_in_place(result, poly_1);
}

#[inline]
/// Add `poly` into `result` in place.
pub fn add_poly_in_place<E: SumcheckElement>(result: &mut Polynomial<E>, poly: &Polynomial<E>) {
    let m = max(result.num_coefficients, poly.num_coefficients);
    extrapolate(&mut result.coefficients, result.num_coefficients, m);
    if poly.num_coefficients == m {
        for i in 0..m {
            result.coefficients[i] += &poly.coefficients[i];
        }
    } else {
        let mut e: [E; 4] = std::array::from_fn(|i| {
            if i < poly.num_coefficients {
                poly.coefficients[i].clone()
            } else {
                E::zero()
            }
        });
        extrapolate(&mut e, poly.num_coefficients, m);
        for i in 0..m {
            result.coefficients[i] += &e[i];
        }
    }
    result.num_coefficients = m;
}

#[inline]
/// Subtract `poly` from `result` in place.
pub fn sub_poly_in_place<E: SumcheckElement>(result: &mut Polynomial<E>, poly: &Polynomial<E>) {
    let m = max(result.num_coefficients, poly.num_coefficients);
    extrapolate(&mut result.coefficients, result.num_coefficients, m);
    if poly.num_coefficients == m {
        for i in 0..m {
            result.coefficients[i] -= &poly.coefficients[i];
        }
    } else {
        let mut e: [E; 4] = std::array::from_fn(|i| {
            if i < poly.num_coefficients {
                poly.coefficients[i].clone()
            } else {
                E::zero()
            }
        });
        extrapolate(&mut e, poly.num_coefficients, m);
        for i in 0..m {
            result.coefficients[i] -= &e[i];
        }
    }
    result.num_coefficients = m;
}
