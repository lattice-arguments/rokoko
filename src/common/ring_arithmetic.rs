
use std::ops::{Add, Mul, Sub, AddAssign, MulAssign, SubAssign};
use crate::common::config::*;
use crate::hexl::bindings::*;
use std::sync::LazyLock;
use rand::Rng;

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum Representation {
    Coefficients, // This should not be used almost ever. Use only for printing or debugging.
    EvenOddCoefficients, // In this representation, coefficients are stored as even part followed by odd part. This is so that NTT can be applied more easily.
    IncompleteNTT, // Incomplete NTT representation, where even and odd parts are separately transformed.
}


// DO NOT derive Copy here, as RingElement is large.
#[derive(PartialEq, Clone, Debug)]
pub struct RingElement {
    pub v: [u64; DEGREE],
    representation: Representation,
}

pub static mut rng: LazyLock<rand::rngs::ThreadRng> = LazyLock::new(|| {
    rand::thread_rng()
});


impl RingElement {
    pub const fn new(representation: Representation) -> Self {
        Self { v: [0; DEGREE], representation }
    }

    pub fn new_random(representation: Representation) -> Self {
        let mut element = Self { v: [0; DEGREE], representation };

        unsafe {
            for i in 0..DEGREE {
                element.v[i] = rng.gen_range(0..MOD_Q);
            }
        }

        element
    }

    pub fn from_even_odd_coefficients_to_incomplete_ntt_representation(&mut self) {
        debug_assert!(self.representation == Representation::EvenOddCoefficients, "Already in Incomplete NTT representation");

        unsafe {
            ntt_forward_in_place(self.v.as_mut_ptr(), HALF_DEGREE, MOD_Q);
            ntt_forward_in_place(self.v.as_mut_ptr().add(HALF_DEGREE), HALF_DEGREE, MOD_Q);
        }

        self.representation = Representation::IncompleteNTT;
    }

    pub fn from_incomplete_ntt_to_even_odd_coefficients(&mut self) {
        debug_assert!(self.representation == Representation::IncompleteNTT, "Not in Incomplete NTT representation");

        unsafe {
            ntt_inverse_in_place(self.v.as_mut_ptr(), HALF_DEGREE, MOD_Q);
            ntt_inverse_in_place(self.v.as_mut_ptr().add(HALF_DEGREE), HALF_DEGREE, MOD_Q);
        }

        self.representation = Representation::EvenOddCoefficients;
    }

    pub fn from_coefficients_to_even_odd_coefficients(&mut self) {
        debug_assert!(self.representation == Representation::Coefficients, "Not in Coefficients representation");

        let mut temp = [0u64; DEGREE];

        for i in 0..(DEGREE / 2) {
            temp[i] = self.v[2 * i];
            temp[i + (DEGREE / 2)] = self.v[2 * i + 1];
        }

        self.v = temp;
        self.representation = Representation::EvenOddCoefficients;
    }

    pub fn from_even_odd_coefficients_to_coefficients(&mut self) {
        debug_assert!(self.representation == Representation::EvenOddCoefficients, "Not in Even-Odd Coefficients representation");

        let mut temp = [0u64; DEGREE];

        for i in 0..(DEGREE / 2) {
            temp[2 * i] = self.v[i];
            temp[2 * i + 1] = self.v[i + (DEGREE / 2)];
        }

        self.v = temp;
        self.representation = Representation::Coefficients;
    }

}


pub static shift_factors: LazyLock<[u64; HALF_DEGREE]> = LazyLock::new(|| {
    let mut factors = [0u64; HALF_DEGREE];
    factors[1] = 1;
    unsafe { ntt_forward_in_place(factors.as_mut_ptr(), factors.len(), MOD_Q) };
    factors
});

pub static mut temp_buffer: LazyLock<[u64; DEGREE]> = LazyLock::new(|| {
    [0u64; DEGREE]
});

fn get_temp_buffer() -> &'static mut [u64; DEGREE] {
    unsafe {
        &mut temp_buffer
    }
}

///// Helpers
#[inline]
pub fn incomplete_ntt_multiplication(
    result: &mut RingElement,
    operand1: &RingElement,
    operand2: &RingElement,
) {

    debug_assert!(operand1.representation == Representation::IncompleteNTT, "Operand1 not in Incomplete NTT representation");
    debug_assert!(operand2.representation == Representation::IncompleteNTT, "Operand2 not in Incomplete NTT representation");
    debug_assert!(result.representation == Representation::IncompleteNTT, "Result not in Incomplete NTT representation");

    let mut temp = get_temp_buffer();

    let op1_data = &operand1.v;
    let op2_data = &operand2.v;

    unsafe {
        // result_even = op1_even * op2_even
        eltwise_mult_mod(
            result.v.as_mut_ptr(),
            op1_data.as_ptr(),
            op2_data.as_ptr(),
            HALF_DEGREE as u64,
            MOD_Q,
        );

        // result_odd = op1_odd * op2_even
        eltwise_mult_mod(
            result.v.as_mut_ptr().add(HALF_DEGREE),
            op1_data.as_ptr().add(HALF_DEGREE),
            op2_data.as_ptr(),
            HALF_DEGREE as u64,
            MOD_Q,
        );

        // temp = op1_odd * op2_odd
        eltwise_mult_mod(
            temp.as_mut_ptr(),
            op1_data.as_ptr().add(HALF_DEGREE),
            op2_data.as_ptr().add(HALF_DEGREE),
            HALF_DEGREE as u64,
            MOD_Q,
        );

        // Apply shift factors
        eltwise_mult_mod(
                temp.as_mut_ptr(),
                temp.as_ptr(),
                shift_factors.as_ptr(),
                HALF_DEGREE as u64,
                MOD_Q,
        );

        // result_even += temp
        eltwise_add_mod(
            result.v.as_mut_ptr(),
            result.v.as_ptr(),
            temp.as_ptr(),
            HALF_DEGREE as u64,
            MOD_Q,
        );

        // Reuse temp for op1_even * op2_odd
        eltwise_mult_mod(
            temp.as_mut_ptr(),
            op1_data.as_ptr(),
            op2_data.as_ptr().add(HALF_DEGREE),
            HALF_DEGREE as u64,
            MOD_Q,
        );

        // result_odd += temp
        eltwise_add_mod(
            result.v.as_mut_ptr().add(HALF_DEGREE),
            result.v.as_ptr().add(HALF_DEGREE),
            temp.as_ptr(),
            HALF_DEGREE as u64,
            MOD_Q,
        );
    }
}

pub fn naive_polynomial_multiplication(
    result: &mut RingElement,
    operand1: &RingElement,
    operand2: &RingElement,
) {
    debug_assert!(operand1.representation == Representation::Coefficients, "Operand1 not in Coefficients representation");
    debug_assert!(operand2.representation == Representation::Coefficients, "Operand2 not in Coefficients representation");
    debug_assert!(result.representation == Representation::Coefficients, "Result not in Coefficients representation");

    for i in 0..DEGREE {
        result.v[i] = 0;
    }

    for i in 0..DEGREE {
        for j in 0..DEGREE {
            let index = (i + j) % DEGREE;
            let prod = (operand1.v[i] as u128 * operand2.v[j] as u128) % (MOD_Q as u128);
            if i + j >= DEGREE {
                // Negation in modular arithmetic: MOD_Q - value
                let neg_prod = (MOD_Q as u128 - prod) % (MOD_Q as u128);
                result.v[index] = (result.v[index] + neg_prod as u64) % MOD_Q;
            } else {
                result.v[index] = (result.v[index] + prod as u64) % MOD_Q;
            }
        }
    }
}