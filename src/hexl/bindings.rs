use incomplete_rexl as hexl;
pub use incomplete_rexl::SlotRing;

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
pub unsafe fn ntt_inverse(result: *mut u64, operand: *const u64, n: usize, modulus: u64) {
    let result = slice_from_raw_mut_usize(result, n);
    let operand = std::slice::from_raw_parts(operand, n);
    hexl::ntt_inverse(result, operand, n, modulus);
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

/// Fused incomplete NTT ring multiplication (Karatsuba + AVX512 float).
/// Shift factors are cached internally by incomplete-rexl.
#[inline(always)]
pub unsafe fn fused_incomplete_ntt_mult(
    result: *mut u64,
    operand1: *const u64,
    operand2: *const u64,
    _shift_factors: *const u64,
    n: usize,
    modulus: u64,
) {
    hexl::fused_incomplete_ntt_mult_ptr(result, operand1, operand2, n, modulus);
}

/// Slot-wise ring multiplication for slots of any supported degree; `result` may alias
/// `operand1`, as in [`fused_incomplete_ntt_mult`].
#[inline(always)]
pub unsafe fn fused_slot_mult(
    result: *mut u64,
    operand1: *const u64,
    operand2: *const u64,
    ring_degree: usize,
    modulus: u64,
) {
    hexl::fused_slot_mult_ptr(result, operand1, operand2, ring_degree, modulus);
}
