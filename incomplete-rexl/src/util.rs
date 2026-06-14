#![allow(dead_code)]
use core::cmp::Ordering;

#[inline(always)]
pub fn hexl_unused<T>(_value: T) {}

#[inline(always)]
pub fn multiply_u64(x: u64, y: u64) -> u128 {
    (x as u128) * (y as u128)
}

#[inline(always)]
pub fn multiply_u64_full(x: u64, y: u64) -> (u64, u64) {
    let prod = multiply_u64(x, y);
    ((prod >> 64) as u64, prod as u64)
}

#[inline(always)]
pub fn multiply_u64_hi<const BITSHIFT: usize>(x: u64, y: u64) -> u64 {
    let prod = multiply_u64(x, y);
    (prod >> BITSHIFT) as u64
}

#[inline(always)]
pub fn divide_u128_u64_lo(x1: u64, x0: u64, y: u64) -> u64 {
    let n = ((x1 as u128) << 64) | (x0 as u128);
    (n / y as u128) as u64
}

#[inline(always)]
pub fn msb(input: u64) -> u64 {
    if input == 0 {
        return 0;
    }
    63 - input.leading_zeros() as u64
}

#[inline(always)]
pub fn log2(x: u64) -> u64 {
    msb(x)
}

#[inline(always)]
pub fn is_power_of_two(num: u64) -> bool {
    num != 0 && (num & (num - 1)) == 0
}

#[inline(always)]
pub fn is_power_of_four(num: u64) -> bool {
    is_power_of_two(num) && (log2(num) % 2 == 0)
}

#[inline(always)]
pub fn maximum_value(bits: u64) -> u64 {
    if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

pub fn reverse_bits(mut x: u64, bit_width: u64) -> u64 {
    if bit_width == 0 {
        return 0;
    }
    let mut rev = 0u64;
    for i in (1..=bit_width).rev() {
        rev |= (x & 1) << (i - 1);
        x >>= 1;
    }
    rev
}

#[inline(always)]
pub fn add_u64(operand1: u64, operand2: u64) -> (u64, u8) {
    let result = operand1.wrapping_add(operand2);
    let carry = if result < operand1 { 1 } else { 0 };
    (result, carry)
}

#[derive(Copy, Clone, Debug)]
pub enum CmpInt {
    Eq,
    Lt,
    Le,
    False,
    Ne,
    Nlt,
    Nle,
    True,
}

impl CmpInt {
    pub fn not(self) -> Self {
        match self {
            CmpInt::Eq => CmpInt::Ne,
            CmpInt::Lt => CmpInt::Nlt,
            CmpInt::Le => CmpInt::Nle,
            CmpInt::False => CmpInt::True,
            CmpInt::Ne => CmpInt::Eq,
            CmpInt::Nlt => CmpInt::Lt,
            CmpInt::Nle => CmpInt::Le,
            CmpInt::True => CmpInt::False,
        }
    }
}


pub fn cmp_u64(cmp: CmpInt, lhs: u64, rhs: u64) -> bool {
    match cmp {
        CmpInt::Eq => lhs == rhs,
        CmpInt::Lt => lhs < rhs,
        CmpInt::Le => lhs <= rhs,
        CmpInt::False => false,
        CmpInt::Ne => lhs != rhs,
        CmpInt::Nlt => lhs >= rhs,
        CmpInt::Nle => lhs > rhs,
        CmpInt::True => true,
    }
}

pub fn compare(cmp: CmpInt, lhs: u64, rhs: u64) -> bool {
    match cmp {
        CmpInt::Eq => lhs == rhs,
        CmpInt::Lt => lhs < rhs,
        CmpInt::Le => lhs <= rhs,
        CmpInt::False => false,
        CmpInt::Ne => lhs != rhs,
        CmpInt::Nlt => lhs >= rhs,
        CmpInt::Nle => lhs > rhs,
        CmpInt::True => true,
    }
}

pub fn ordering_to_cmp(ordering: Ordering) -> CmpInt {
    match ordering {
        Ordering::Less => CmpInt::Lt,
        Ordering::Equal => CmpInt::Eq,
        Ordering::Greater => CmpInt::Nlt,
    }
}
