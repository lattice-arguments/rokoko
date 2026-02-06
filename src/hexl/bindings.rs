use hexl_rust as hexl;

#[inline(always)]
unsafe fn slice_from_raw<'a>(ptr: *const u64, n: u64) -> &'a [u64] {
    std::slice::from_raw_parts(ptr, n as usize)
}

#[inline(always)]
unsafe fn slice_from_raw_mut<'a>(ptr: *mut u64, n: u64) -> &'a mut [u64] {
    std::slice::from_raw_parts_mut(ptr, n as usize)
}

#[inline(always)]
unsafe fn slice_from_raw_mut_usize<'a>(ptr: *mut u64, n: usize) -> &'a mut [u64] {
    std::slice::from_raw_parts_mut(ptr, n)
}

#[inline(always)]
pub unsafe fn multiply_mod(a: u64, b: u64, modulus: u64) -> u64 {
    hexl::multiply_mod(a, b, modulus)
}

#[inline(always)]
pub unsafe fn power_mod(a: u64, b: u64, modulus: u64) -> u64 {
    hexl::power_mod(a, b, modulus)
}

#[inline(always)]
pub unsafe fn add_mod(a: u64, b: u64, modulus: u64) -> u64 {
    hexl::add_mod(a, b, modulus)
}

#[inline(always)]
pub unsafe fn sub_mod(a: u64, b: u64, modulus: u64) -> u64 {
    hexl::sub_mod(a, b, modulus)
}

#[inline(always)]
pub unsafe fn eltwise_mult_mod(
    result: *mut u64,
    operand1: *const u64,
    operand2: *const u64,
    n: u64,
    modulus: u64,
) {
    let result = slice_from_raw_mut(result, n);
    let operand1 = slice_from_raw(operand1, n);
    let operand2 = slice_from_raw(operand2, n);
    hexl::eltwise_mult_mod(result, operand1, operand2, modulus);
}

#[inline(always)]
pub unsafe fn get_roots(n: u64, modulus: u64) -> *const u64 {
    hexl::get_roots(n as usize, modulus)
}

#[inline(always)]
pub unsafe fn inv_mod(a: u64, modulus: u64) -> u64 {
    hexl::inv_mod(a, modulus)
}

#[inline(always)]
pub unsafe fn get_inv_roots(n: u64, modulus: u64) -> *const u64 {
    hexl::get_inv_roots(n as usize, modulus)
}

#[inline(always)]
pub unsafe fn eltwise_add_mod(
    result: *mut u64,
    operand1: *const u64,
    operand2: *const u64,
    n: u64,
    modulus: u64,
) {
    let result = slice_from_raw_mut(result, n);
    let operand1 = slice_from_raw(operand1, n);
    let operand2 = slice_from_raw(operand2, n);
    hexl::eltwise_add_mod(result, operand1, operand2, modulus);
}

#[inline(always)]
pub unsafe fn eltwise_sub_mod(
    result: *mut u64,
    operand1: *const u64,
    operand2: *const u64,
    n: u64,
    modulus: u64,
) {
    let result = slice_from_raw_mut(result, n);
    let operand1 = slice_from_raw(operand1, n);
    let operand2 = slice_from_raw(operand2, n);
    hexl::eltwise_sub_mod(result, operand1, operand2, modulus);
}

#[inline(always)]
pub unsafe fn eltwise_reduce_mod(result: *mut u64, operand: *const u64, n: u64, modulus: u64) {
    let result = slice_from_raw_mut(result, n);
    let operand = slice_from_raw(operand, n);
    hexl::eltwise_reduce_mod(result, operand, modulus);
}

#[inline(always)]
pub unsafe fn ntt_forward_in_place(operand: *mut u64, n: usize, modulus: u64) {
    let operand = slice_from_raw_mut_usize(operand, n);
    hexl::ntt_forward_in_place(operand, n, modulus);
}

#[inline(always)]
pub unsafe fn ntt_inverse_in_place(operand: *mut u64, n: usize, modulus: u64) {
    let operand = slice_from_raw_mut_usize(operand, n);
    hexl::ntt_inverse_in_place(operand, n, modulus);
}

#[inline(always)]
pub unsafe fn eltwise_fma_mod(
    result: *mut u64,
    operand1: *const u64,
    operand2: u64,
    operand3: *const u64,
    n: u64,
    modulus: u64,
) {
    let result = slice_from_raw_mut(result, n);
    let operand1 = slice_from_raw(operand1, n);
    let operand3 = slice_from_raw(operand3, n);
    hexl::eltwise_fma_mod(result, operand1, operand2, operand3, modulus);
}

/// Fused incomplete NTT ring multiplication.
///
/// Computes for each i in 0..n:
///   result[i]   = op1[i]*op2[i] + shift[i]*(op1[n+i]*op2[n+i])  (mod modulus)
///   result[n+i] = op1[n+i]*op2[i] + op1[i]*op2[n+i]             (mod modulus)
#[inline(always)]
pub unsafe fn fused_incomplete_ntt_mult(
    result: *mut u64,
    operand1: *const u64,
    operand2: *const u64,
    shift_factors: *const u64,
    shift_factors_f64: *const f64,
    n: usize,
    modulus: u64,
) {
    let result = std::slice::from_raw_parts_mut(result, 2 * n);
    let operand1 = std::slice::from_raw_parts(operand1, 2 * n);
    let operand2 = std::slice::from_raw_parts(operand2, 2 * n);
    let shift_factors = std::slice::from_raw_parts(shift_factors, n);
    let shift_factors_f64 = std::slice::from_raw_parts(shift_factors_f64, n);
    hexl::fused_incomplete_ntt_mult(result, operand1, operand2, shift_factors, shift_factors_f64, n, modulus);
}
