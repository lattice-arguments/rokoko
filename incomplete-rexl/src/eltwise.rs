use crate::cpu_features::{HAS_AVX512DQ, HAS_AVX512IFMA};
use crate::util::{log2, maximum_value, multiply_u64_full};
use crate::number_theory::{
    add_uint_mod, barrett_reduce64, multiply_mod_precon, reduce_mod, MultiplyFactor,
};

#[cfg(target_arch = "x86_64")]
use crate::avx512_util::{
    mm512_hexl_barrett_reduce64, mm512_hexl_mulhi_approx_epi, mm512_hexl_mulhi_epi,
    mm512_hexl_mullo_add_lo_epi, mm512_hexl_mullo_epi, mm512_hexl_shrdi_epi64,
    mm512_hexl_shrdi_epi64_runtime, mm512_hexl_small_add_mod_epi64, mm512_hexl_small_mod_epu64,
    mm512_hexl_small_sub_mod_epi64,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use crate::cpu_features::HAS_NEON;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

fn reduce_mod_input<const INPUT_MOD_FACTOR: i32>(
    x: u64,
    modulus: u64,
    twice_modulus: Option<&u64>,
    four_times_modulus: Option<&u64>,
) -> u64 {
    match INPUT_MOD_FACTOR {
        1 => reduce_mod::<1>(x, modulus, None, None),
        2 => reduce_mod::<2>(x, modulus, None, None),
        4 => reduce_mod::<4>(x, modulus, twice_modulus, None),
        8 => reduce_mod::<8>(x, modulus, twice_modulus, four_times_modulus),
        _ => x,
    }
}

fn eltwise_mult_mod_native_dispatch<const INPUT_MOD_FACTOR: i32>(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    modulus: u64,
) {
    match INPUT_MOD_FACTOR {
        1 => eltwise_mult_mod_native::<1>(result, operand1, operand2, modulus),
        2 => eltwise_mult_mod_native::<2>(result, operand1, operand2, modulus),
        4 => eltwise_mult_mod_native::<4>(result, operand1, operand2, modulus),
        _ => {}
    }
}

fn eltwise_fma_mod_native_dispatch<const INPUT_MOD_FACTOR: i32>(
    result: &mut [u64],
    arg1: &[u64],
    arg2: u64,
    arg3: Option<&[u64]>,
    modulus: u64,
) {
    match INPUT_MOD_FACTOR {
        1 => eltwise_fma_mod_native::<1>(result, arg1, arg2, arg3, modulus),
        2 => eltwise_fma_mod_native::<2>(result, arg1, arg2, arg3, modulus),
        4 => eltwise_fma_mod_native::<4>(result, arg1, arg2, arg3, modulus),
        8 => eltwise_fma_mod_native::<8>(result, arg1, arg2, arg3, modulus),
        _ => {}
    }
}

pub fn eltwise_add_mod(result: &mut [u64], operand1: &[u64], operand2: &[u64], modulus: u64) {
    let n = result.len();
    if n == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX512DQ {
            unsafe {
                eltwise_add_mod_avx512(result, operand1, operand2, modulus);
                return;
            }
        }
    }

    eltwise_add_mod_native(result, operand1, operand2, modulus);
}

fn eltwise_add_mod_native(result: &mut [u64], operand1: &[u64], operand2: &[u64], modulus: u64) {
    for i in 0..result.len() {
        let sum = operand1[i] + operand2[i];
        result[i] = if sum >= modulus { sum - modulus } else { sum };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
#[inline]
unsafe fn eltwise_add_mod_avx512(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    modulus: u64,
) {
    let mut n = result.len();
    let mut idx = 0usize;
    let n_mod_8 = n % 8;
    if n_mod_8 != 0 {
        eltwise_add_mod_native(
            &mut result[..n_mod_8],
            &operand1[..n_mod_8],
            &operand2[..n_mod_8],
            modulus,
        );
        idx += n_mod_8;
        n -= n_mod_8;
    }

    let v_modulus = _mm512_set1_epi64(modulus as i64);
    let mut vp_result = result[idx..].as_mut_ptr() as *mut __m512i;
    let mut vp_operand1 = operand1[idx..].as_ptr() as *const __m512i;
    let mut vp_operand2 = operand2[idx..].as_ptr() as *const __m512i;

    for _ in (0..n).step_by(8) {
        let v_operand1 = _mm512_loadu_si512(vp_operand1);
        let v_operand2 = _mm512_loadu_si512(vp_operand2);
        let v_result = mm512_hexl_small_add_mod_epi64(v_operand1, v_operand2, v_modulus);
        _mm512_storeu_si512(vp_result, v_result);
        vp_result = vp_result.add(1);
        vp_operand1 = vp_operand1.add(1);
        vp_operand2 = vp_operand2.add(1);
    }
}

pub fn eltwise_sub_mod(result: &mut [u64], operand1: &[u64], operand2: &[u64], modulus: u64) {
    let n = result.len();
    if n == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX512DQ {
            unsafe {
                eltwise_sub_mod_avx512(result, operand1, operand2, modulus);
                return;
            }
        }
    }

    eltwise_sub_mod_native(result, operand1, operand2, modulus);
}

fn eltwise_sub_mod_native(result: &mut [u64], operand1: &[u64], operand2: &[u64], modulus: u64) {
    for i in 0..result.len() {
        if operand1[i] >= operand2[i] {
            result[i] = operand1[i] - operand2[i];
        } else {
            result[i] = operand1[i] + modulus - operand2[i];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
#[inline]
unsafe fn eltwise_sub_mod_avx512(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    modulus: u64,
) {
    let mut n = result.len();
    let mut idx = 0usize;
    let n_mod_8 = n % 8;
    if n_mod_8 != 0 {
        eltwise_sub_mod_native(
            &mut result[..n_mod_8],
            &operand1[..n_mod_8],
            &operand2[..n_mod_8],
            modulus,
        );
        idx += n_mod_8;
        n -= n_mod_8;
    }

    let v_modulus = _mm512_set1_epi64(modulus as i64);
    let mut vp_result = result[idx..].as_mut_ptr() as *mut __m512i;
    let mut vp_operand1 = operand1[idx..].as_ptr() as *const __m512i;
    let mut vp_operand2 = operand2[idx..].as_ptr() as *const __m512i;

    for _ in (0..n).step_by(8) {
        let v_operand1 = _mm512_loadu_si512(vp_operand1);
        let v_operand2 = _mm512_loadu_si512(vp_operand2);
        let v_result = mm512_hexl_small_sub_mod_epi64(v_operand1, v_operand2, v_modulus);
        _mm512_storeu_si512(vp_result, v_result);
        vp_result = vp_result.add(1);
        vp_operand1 = vp_operand1.add(1);
        vp_operand2 = vp_operand2.add(1);
    }
}

pub fn eltwise_reduce_mod(result: &mut [u64], operand: &[u64], modulus: u64) {
    let n = result.len();
    if n == 0 {
        return;
    }

    let input_mod_factor = modulus;
    let output_mod_factor = 1u64;

    if input_mod_factor == output_mod_factor && !core::ptr::eq(result.as_ptr(), operand.as_ptr()) {
        result.copy_from_slice(operand);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX512IFMA
            && ((modulus < (1u64 << 51)) || (modulus < (1u64 << 52) && input_mod_factor <= 4))
        {
            unsafe {
                eltwise_reduce_mod_avx512::<52>(
                    result,
                    operand,
                    modulus,
                    input_mod_factor,
                    output_mod_factor,
                );
                return;
            }
        }

        if *HAS_AVX512DQ {
            unsafe {
                eltwise_reduce_mod_avx512::<64>(
                    result,
                    operand,
                    modulus,
                    input_mod_factor,
                    output_mod_factor,
                );
                return;
            }
        }
    }

    eltwise_reduce_mod_native(
        result,
        operand,
        modulus,
        input_mod_factor,
        output_mod_factor,
    );
}

fn eltwise_reduce_mod_native(
    result: &mut [u64],
    operand: &[u64],
    modulus: u64,
    input_mod_factor: u64,
    output_mod_factor: u64,
) {
    debug_assert!(input_mod_factor == modulus || input_mod_factor == 2 || input_mod_factor == 4);
    debug_assert!(output_mod_factor == 1 || output_mod_factor == 2);
    debug_assert!(input_mod_factor != output_mod_factor);

    let barrett_factor = MultiplyFactor::new(1, 64, modulus).barrett_factor();
    let twice_modulus = modulus << 1;

    if input_mod_factor == modulus {
        if output_mod_factor == 2 {
            for i in 0..result.len() {
                result[i] = if operand[i] >= twice_modulus {
                    barrett_reduce64::<2>(operand[i], modulus, barrett_factor)
                } else {
                    operand[i]
                };
            }
        } else {
            for i in 0..result.len() {
                result[i] = if operand[i] >= modulus {
                    barrett_reduce64::<1>(operand[i], modulus, barrett_factor)
                } else {
                    operand[i]
                };
            }
        }
        return;
    }

    if input_mod_factor == 2 {
        for i in 0..result.len() {
            result[i] = reduce_mod::<2>(operand[i], modulus, None, None);
        }
        return;
    }

    if input_mod_factor == 4 {
        if output_mod_factor == 1 {
            for i in 0..result.len() {
                result[i] = reduce_mod::<4>(operand[i], modulus, Some(&twice_modulus), None);
            }
        } else {
            for i in 0..result.len() {
                result[i] = reduce_mod::<2>(operand[i], twice_modulus, None, None);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
#[inline]
unsafe fn eltwise_reduce_mod_avx512<const BITSHIFT: i32>(
    result: &mut [u64],
    operand: &[u64],
    modulus: u64,
    input_mod_factor: u64,
    output_mod_factor: u64,
) {
    debug_assert!(input_mod_factor == modulus || input_mod_factor == 2 || input_mod_factor == 4);
    debug_assert!(output_mod_factor == 1 || output_mod_factor == 2);
    debug_assert!(input_mod_factor != output_mod_factor);

    let mut n_tmp = result.len();

    let alpha = BITSHIFT as i64 - 2;
    let beta = -2i64;
    let ceil_log_mod = log2(modulus) + 1;
    let prod_right_shift = (ceil_log_mod as i64 + beta) as u64;
    let v_neg_mod = _mm512_set1_epi64(-(modulus as i64));

    let mut barrett_factor = MultiplyFactor::new(
        1u64 << (ceil_log_mod + alpha as u64 - BITSHIFT as u64),
        BITSHIFT as u64,
        modulus,
    )
    .barrett_factor();
    let barrett_factor_52 = MultiplyFactor::new(1, 52, modulus).barrett_factor();

    if BITSHIFT == 64 {
        barrett_factor = MultiplyFactor::new(1, 64, modulus).barrett_factor();
    }

    let v_bf = _mm512_set1_epi64(barrett_factor as i64);
    let v_bf_52 = _mm512_set1_epi64(barrett_factor_52 as i64);

    let n_mod_8 = n_tmp % 8;
    let mut op_ptr = operand;
    let mut res_ptr = result;
    if n_mod_8 != 0 {
        eltwise_reduce_mod_native(
            &mut res_ptr[..n_mod_8],
            &op_ptr[..n_mod_8],
            modulus,
            input_mod_factor,
            output_mod_factor,
        );
        op_ptr = &op_ptr[n_mod_8..];
        res_ptr = &mut res_ptr[n_mod_8..];
        n_tmp -= n_mod_8;
    }

    let twice_mod = modulus << 1;
    let mut v_operand = op_ptr.as_ptr() as *const __m512i;
    let mut v_result = res_ptr.as_mut_ptr() as *mut __m512i;
    let v_modulus = _mm512_set1_epi64(modulus as i64);
    let v_twice_mod = _mm512_set1_epi64(twice_mod as i64);

    if input_mod_factor == modulus {
        if output_mod_factor == 2 {
            for _ in (0..n_tmp).step_by(8) {
                let mut v_op = _mm512_loadu_si512(v_operand);
                v_op = mm512_hexl_barrett_reduce64::<BITSHIFT, 2>(
                    v_op,
                    v_modulus,
                    v_bf,
                    v_bf_52,
                    prod_right_shift,
                    v_neg_mod,
                );
                _mm512_storeu_si512(v_result, v_op);
                v_operand = v_operand.add(1);
                v_result = v_result.add(1);
            }
        } else {
            for _ in (0..n_tmp).step_by(8) {
                let mut v_op = _mm512_loadu_si512(v_operand);
                v_op = mm512_hexl_barrett_reduce64::<BITSHIFT, 1>(
                    v_op,
                    v_modulus,
                    v_bf,
                    v_bf_52,
                    prod_right_shift,
                    v_neg_mod,
                );
                _mm512_storeu_si512(v_result, v_op);
                v_operand = v_operand.add(1);
                v_result = v_result.add(1);
            }
        }
    }

    if input_mod_factor == 2 {
        for _ in (0..n_tmp).step_by(8) {
            let mut v_op = _mm512_loadu_si512(v_operand);
            v_op = mm512_hexl_small_mod_epu64::<2>(v_op, v_modulus, None, None);
            _mm512_storeu_si512(v_result, v_op);
            v_operand = v_operand.add(1);
            v_result = v_result.add(1);
        }
    }

    if input_mod_factor == 4 {
        if output_mod_factor == 1 {
            for _ in (0..n_tmp).step_by(8) {
                let mut v_op = _mm512_loadu_si512(v_operand);
                v_op = mm512_hexl_small_mod_epu64::<2>(v_op, v_twice_mod, None, None);
                v_op = mm512_hexl_small_mod_epu64::<2>(v_op, v_modulus, None, None);
                _mm512_storeu_si512(v_result, v_op);
                v_operand = v_operand.add(1);
                v_result = v_result.add(1);
            }
        }
        if output_mod_factor == 2 {
            for _ in (0..n_tmp).step_by(8) {
                let mut v_op = _mm512_loadu_si512(v_operand);
                v_op = mm512_hexl_small_mod_epu64::<2>(v_op, v_twice_mod, None, None);
                _mm512_storeu_si512(v_result, v_op);
                v_operand = v_operand.add(1);
                v_result = v_result.add(1);
            }
        }
    }
}

pub fn eltwise_mult_mod(result: &mut [u64], operand1: &[u64], operand2: &[u64], modulus: u64) {
    let n = result.len();
    if n == 0 {
        return;
    }
    let input_mod_factor = 1u64;

    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX512DQ {
            if modulus < (1u64 << 50) {
                match input_mod_factor {
                    1 => unsafe {
                        eltwise_mult_mod_avx512_float::<1>(result, operand1, operand2, modulus)
                    },
                    2 => unsafe {
                        eltwise_mult_mod_avx512_float::<2>(result, operand1, operand2, modulus)
                    },
                    4 => unsafe {
                        eltwise_mult_mod_avx512_float::<4>(result, operand1, operand2, modulus)
                    },
                    _ => {}
                }
            } else {
                match input_mod_factor {
                    1 => unsafe {
                        eltwise_mult_mod_avx512_dq_int::<1>(result, operand1, operand2, modulus)
                    },
                    2 => unsafe {
                        eltwise_mult_mod_avx512_dq_int::<2>(result, operand1, operand2, modulus)
                    },
                    4 => unsafe {
                        eltwise_mult_mod_avx512_dq_int::<4>(result, operand1, operand2, modulus)
                    },
                    _ => {}
                }
            }
            return;
        }
    }

    match input_mod_factor {
        1 => eltwise_mult_mod_native::<1>(result, operand1, operand2, modulus),
        2 => eltwise_mult_mod_native::<2>(result, operand1, operand2, modulus),
        4 => eltwise_mult_mod_native::<4>(result, operand1, operand2, modulus),
        _ => {}
    }
}

fn eltwise_mult_mod_native<const INPUT_MOD_FACTOR: u64>(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    modulus: u64,
) {
    debug_assert!(INPUT_MOD_FACTOR == 1 || INPUT_MOD_FACTOR == 2 || INPUT_MOD_FACTOR == 4);
    debug_assert!(modulus < (1u64 << 62));

    let beta = -2i64;
    let alpha = 62i64;
    let ceil_log_mod = log2(modulus) + 1;
    let prod_right_shift = (ceil_log_mod as i64 + beta) as u64;

    let barr_lo = MultiplyFactor::new(1u64 << (ceil_log_mod + alpha as u64 - 64), 64, modulus)
        .barrett_factor();

    let twice_modulus = 2 * modulus;

    for i in 0..result.len() {
        let x = reduce_mod::<INPUT_MOD_FACTOR>(operand1[i], modulus, Some(&twice_modulus), None);
        let y = reduce_mod::<INPUT_MOD_FACTOR>(operand2[i], modulus, Some(&twice_modulus), None);

        let (prod_hi, prod_lo) = multiply_u64_full(x, y);
        let c1 = (prod_lo >> prod_right_shift) + (prod_hi << (64 - prod_right_shift));
        let (c2_hi, _c2_lo) = multiply_u64_full(c1, barr_lo);
        let q_hat = c2_hi;
        let z = prod_lo.wrapping_sub(q_hat.wrapping_mul(modulus));
        result[i] = if z >= modulus { z - modulus } else { z };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
#[inline]
unsafe fn eltwise_mult_mod_avx512_dq_int_loop_const<
    const PROD_RIGHT_SHIFT: i32,
    const INPUT_MOD_FACTOR: i32,
>(
    vp_result: *mut __m512i,
    vp_operand1: *const __m512i,
    vp_operand2: *const __m512i,
    v_barr_lo: __m512i,
    v_modulus: __m512i,
    v_twice_mod: __m512i,
    n: u64,
) {
    let mut res_ptr = vp_result;
    let mut op1_ptr = vp_operand1;
    let mut op2_ptr = vp_operand2;

    for _ in (0..n).step_by(8) {
        let mut v_op1 = _mm512_loadu_si512(op1_ptr);
        let mut v_op2 = _mm512_loadu_si512(op2_ptr);

        v_op1 = mm512_hexl_small_mod_epu64::<INPUT_MOD_FACTOR>(
            v_op1,
            v_modulus,
            Some(&v_twice_mod),
            None,
        );
        v_op2 = mm512_hexl_small_mod_epu64::<INPUT_MOD_FACTOR>(
            v_op2,
            v_modulus,
            Some(&v_twice_mod),
            None,
        );

        let v_prod_hi = mm512_hexl_mulhi_epi::<64>(v_op1, v_op2);
        let v_prod_lo = mm512_hexl_mullo_epi::<64>(v_op1, v_op2);
        let c1 = mm512_hexl_shrdi_epi64::<PROD_RIGHT_SHIFT>(v_prod_lo, v_prod_hi);
        let q_hat = mm512_hexl_mulhi_approx_epi::<64>(c1, v_barr_lo);
        let mut v_result = mm512_hexl_mullo_epi::<64>(q_hat, v_modulus);
        v_result = _mm512_sub_epi64(v_prod_lo, v_result);
        v_result = mm512_hexl_small_mod_epu64::<4>(v_result, v_modulus, Some(&v_twice_mod), None);
        _mm512_storeu_si512(res_ptr, v_result);

        res_ptr = res_ptr.add(1);
        op1_ptr = op1_ptr.add(1);
        op2_ptr = op2_ptr.add(1);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
#[inline]
unsafe fn eltwise_mult_mod_avx512_dq_int_loop_runtime<const INPUT_MOD_FACTOR: i32>(
    vp_result: *mut __m512i,
    vp_operand1: *const __m512i,
    vp_operand2: *const __m512i,
    v_barr_lo: __m512i,
    v_modulus: __m512i,
    v_twice_mod: __m512i,
    n: u64,
    prod_right_shift: u64,
) {
    let mut res_ptr = vp_result;
    let mut op1_ptr = vp_operand1;
    let mut op2_ptr = vp_operand2;

    for _ in (0..n).step_by(8) {
        let mut v_op1 = _mm512_loadu_si512(op1_ptr);
        let mut v_op2 = _mm512_loadu_si512(op2_ptr);

        v_op1 = mm512_hexl_small_mod_epu64::<INPUT_MOD_FACTOR>(
            v_op1,
            v_modulus,
            Some(&v_twice_mod),
            None,
        );
        v_op2 = mm512_hexl_small_mod_epu64::<INPUT_MOD_FACTOR>(
            v_op2,
            v_modulus,
            Some(&v_twice_mod),
            None,
        );

        let v_prod_hi = mm512_hexl_mulhi_epi::<64>(v_op1, v_op2);
        let v_prod_lo = mm512_hexl_mullo_epi::<64>(v_op1, v_op2);
        let c1 = mm512_hexl_shrdi_epi64_runtime(v_prod_lo, v_prod_hi, prod_right_shift as u32);
        let q_hat = mm512_hexl_mulhi_approx_epi::<64>(c1, v_barr_lo);
        let mut v_result = mm512_hexl_mullo_epi::<64>(q_hat, v_modulus);
        v_result = _mm512_sub_epi64(v_prod_lo, v_result);
        v_result = mm512_hexl_small_mod_epu64::<4>(v_result, v_modulus, Some(&v_twice_mod), None);
        _mm512_storeu_si512(res_ptr, v_result);

        res_ptr = res_ptr.add(1);
        op1_ptr = op1_ptr.add(1);
        op2_ptr = op2_ptr.add(1);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
#[inline]
unsafe fn eltwise_mult_mod_avx512_dq_int<const INPUT_MOD_FACTOR: i32>(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    modulus: u64,
) {
    debug_assert!(INPUT_MOD_FACTOR == 1 || INPUT_MOD_FACTOR == 2 || INPUT_MOD_FACTOR == 4);
    debug_assert!(INPUT_MOD_FACTOR as u64 * modulus > (1u64 << 50));
    debug_assert!(INPUT_MOD_FACTOR as u64 * modulus < (1u64 << 63));
    debug_assert!(modulus < (1u64 << 62));

    let mut n = result.len() as u64;
    let mut op1 = operand1;
    let mut op2 = operand2;
    let mut res = result;

    let n_mod_8 = n % 8;
    if n_mod_8 != 0 {
        eltwise_mult_mod_native_dispatch::<INPUT_MOD_FACTOR>(
            &mut res[..n_mod_8 as usize],
            &op1[..n_mod_8 as usize],
            &op2[..n_mod_8 as usize],
            modulus,
        );
        op1 = &op1[n_mod_8 as usize..];
        op2 = &op2[n_mod_8 as usize..];
        res = &mut res[n_mod_8 as usize..];
        n -= n_mod_8;
    }

    let beta = -2i64;
    let alpha = 62i64;
    let ceil_log_mod = log2(modulus) + 1;
    let prod_right_shift = (ceil_log_mod as i64 + beta) as u64;

    let barr_lo = MultiplyFactor::new(1u64 << (ceil_log_mod + alpha as u64 - 64), 64, modulus)
        .barrett_factor();

    let v_barr_lo = _mm512_set1_epi64(barr_lo as i64);
    let v_modulus = _mm512_set1_epi64(modulus as i64);
    let v_twice_mod = _mm512_set1_epi64((2 * modulus) as i64);

    let vp_operand1 = op1.as_ptr() as *const __m512i;
    let vp_operand2 = op2.as_ptr() as *const __m512i;
    let vp_result = res.as_mut_ptr() as *mut __m512i;

    let reduce_mod = 2 * log2(INPUT_MOD_FACTOR as u64) + prod_right_shift + (-beta as u64) >= 63;

    if reduce_mod {
        match prod_right_shift {
            57 => eltwise_mult_mod_avx512_dq_int_loop_const::<57, INPUT_MOD_FACTOR>(
                vp_result,
                vp_operand1,
                vp_operand2,
                v_barr_lo,
                v_modulus,
                v_twice_mod,
                n,
            ),
            58 => eltwise_mult_mod_avx512_dq_int_loop_const::<58, INPUT_MOD_FACTOR>(
                vp_result,
                vp_operand1,
                vp_operand2,
                v_barr_lo,
                v_modulus,
                v_twice_mod,
                n,
            ),
            59 => eltwise_mult_mod_avx512_dq_int_loop_const::<59, INPUT_MOD_FACTOR>(
                vp_result,
                vp_operand1,
                vp_operand2,
                v_barr_lo,
                v_modulus,
                v_twice_mod,
                n,
            ),
            60 => eltwise_mult_mod_avx512_dq_int_loop_const::<60, INPUT_MOD_FACTOR>(
                vp_result,
                vp_operand1,
                vp_operand2,
                v_barr_lo,
                v_modulus,
                v_twice_mod,
                n,
            ),
            61 => eltwise_mult_mod_avx512_dq_int_loop_const::<61, INPUT_MOD_FACTOR>(
                vp_result,
                vp_operand1,
                vp_operand2,
                v_barr_lo,
                v_modulus,
                v_twice_mod,
                n,
            ),
            _ => eltwise_mult_mod_avx512_dq_int_loop_runtime::<INPUT_MOD_FACTOR>(
                vp_result,
                vp_operand1,
                vp_operand2,
                v_barr_lo,
                v_modulus,
                v_twice_mod,
                n,
                prod_right_shift,
            ),
        }
    } else {
        match prod_right_shift {
            50 => eltwise_mult_mod_avx512_dq_int_loop_const::<50, 1>(
                vp_result,
                vp_operand1,
                vp_operand2,
                v_barr_lo,
                v_modulus,
                v_twice_mod,
                n,
            ),
            51 => eltwise_mult_mod_avx512_dq_int_loop_const::<51, 1>(
                vp_result,
                vp_operand1,
                vp_operand2,
                v_barr_lo,
                v_modulus,
                v_twice_mod,
                n,
            ),
            52 => eltwise_mult_mod_avx512_dq_int_loop_const::<52, 1>(
                vp_result,
                vp_operand1,
                vp_operand2,
                v_barr_lo,
                v_modulus,
                v_twice_mod,
                n,
            ),
            53 => eltwise_mult_mod_avx512_dq_int_loop_const::<53, 1>(
                vp_result,
                vp_operand1,
                vp_operand2,
                v_barr_lo,
                v_modulus,
                v_twice_mod,
                n,
            ),
            54 => eltwise_mult_mod_avx512_dq_int_loop_const::<54, 1>(
                vp_result,
                vp_operand1,
                vp_operand2,
                v_barr_lo,
                v_modulus,
                v_twice_mod,
                n,
            ),
            55 => eltwise_mult_mod_avx512_dq_int_loop_const::<55, 1>(
                vp_result,
                vp_operand1,
                vp_operand2,
                v_barr_lo,
                v_modulus,
                v_twice_mod,
                n,
            ),
            56 => eltwise_mult_mod_avx512_dq_int_loop_const::<56, 1>(
                vp_result,
                vp_operand1,
                vp_operand2,
                v_barr_lo,
                v_modulus,
                v_twice_mod,
                n,
            ),
            57 => eltwise_mult_mod_avx512_dq_int_loop_const::<57, 1>(
                vp_result,
                vp_operand1,
                vp_operand2,
                v_barr_lo,
                v_modulus,
                v_twice_mod,
                n,
            ),
            58 => eltwise_mult_mod_avx512_dq_int_loop_const::<58, 1>(
                vp_result,
                vp_operand1,
                vp_operand2,
                v_barr_lo,
                v_modulus,
                v_twice_mod,
                n,
            ),
            59 => eltwise_mult_mod_avx512_dq_int_loop_const::<59, 1>(
                vp_result,
                vp_operand1,
                vp_operand2,
                v_barr_lo,
                v_modulus,
                v_twice_mod,
                n,
            ),
            60 => eltwise_mult_mod_avx512_dq_int_loop_const::<60, 1>(
                vp_result,
                vp_operand1,
                vp_operand2,
                v_barr_lo,
                v_modulus,
                v_twice_mod,
                n,
            ),
            61 => eltwise_mult_mod_avx512_dq_int_loop_const::<61, 1>(
                vp_result,
                vp_operand1,
                vp_operand2,
                v_barr_lo,
                v_modulus,
                v_twice_mod,
                n,
            ),
            _ => eltwise_mult_mod_avx512_dq_int_loop_runtime::<1>(
                vp_result,
                vp_operand1,
                vp_operand2,
                v_barr_lo,
                v_modulus,
                v_twice_mod,
                n,
                prod_right_shift,
            ),
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
#[inline]
unsafe fn eltwise_mult_mod_avx512_float_loop<const INPUT_MOD_FACTOR: i32>(
    vp_result: *mut __m512i,
    vp_operand1: *const __m512i,
    vp_operand2: *const __m512i,
    v_u: __m512d,
    v_p: __m512d,
    v_modulus: __m512i,
    v_twice_mod: __m512i,
    n: u64,
) {
    const ROUND_MODE: i32 = _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC;

    let mut res_ptr = vp_result;
    let mut op1_ptr = vp_operand1;
    let mut op2_ptr = vp_operand2;

    // Process 2 vectors per iteration for better ILP
    let n_unroll = (n / 16) * 16;
    let mut i = 0;

    while i < n_unroll {
        // Load 2 vectors from each operand
        let mut v_op1_a = _mm512_loadu_si512(op1_ptr);
        let mut v_op1_b = _mm512_loadu_si512(op1_ptr.add(1));
        let mut v_op2_a = _mm512_loadu_si512(op2_ptr);
        let mut v_op2_b = _mm512_loadu_si512(op2_ptr.add(1));

        // Reduce both pairs (independent operations)
        v_op1_a = mm512_hexl_small_mod_epu64::<INPUT_MOD_FACTOR>(
            v_op1_a,
            v_modulus,
            Some(&v_twice_mod),
            None,
        );
        v_op1_b = mm512_hexl_small_mod_epu64::<INPUT_MOD_FACTOR>(
            v_op1_b,
            v_modulus,
            Some(&v_twice_mod),
            None,
        );
        v_op2_a = mm512_hexl_small_mod_epu64::<INPUT_MOD_FACTOR>(
            v_op2_a,
            v_modulus,
            Some(&v_twice_mod),
            None,
        );
        v_op2_b = mm512_hexl_small_mod_epu64::<INPUT_MOD_FACTOR>(
            v_op2_b,
            v_modulus,
            Some(&v_twice_mod),
            None,
        );

        // Convert to double (independent)
        let v_x_a = _mm512_cvt_roundepu64_pd(v_op1_a, ROUND_MODE);
        let v_x_b = _mm512_cvt_roundepu64_pd(v_op1_b, ROUND_MODE);
        let v_y_a = _mm512_cvt_roundepu64_pd(v_op2_a, ROUND_MODE);
        let v_y_b = _mm512_cvt_roundepu64_pd(v_op2_b, ROUND_MODE);

        // Multiply high parts (independent chains)
        let v_h_a = _mm512_mul_pd(v_x_a, v_y_a);
        let v_h_b = _mm512_mul_pd(v_x_b, v_y_b);

        // Compute low parts (FMA ports)
        let v_l_a = _mm512_fmsub_pd(v_x_a, v_y_a, v_h_a);
        let v_l_b = _mm512_fmsub_pd(v_x_b, v_y_b, v_h_b);

        // Barrett reduction prep
        let v_b_a = _mm512_mul_pd(v_h_a, v_u);
        let v_b_b = _mm512_mul_pd(v_h_b, v_u);

        let v_c_a = _mm512_roundscale_pd(v_b_a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        let v_c_b = _mm512_roundscale_pd(v_b_b, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

        let v_d_a = _mm512_fnmadd_pd(v_c_a, v_p, v_h_a);
        let v_d_b = _mm512_fnmadd_pd(v_c_b, v_p, v_h_b);

        let mut v_g_a = _mm512_add_pd(v_d_a, v_l_a);
        let mut v_g_b = _mm512_add_pd(v_d_b, v_l_b);

        // Conditional add
        let m_a = _mm512_cmp_pd_mask(v_g_a, _mm512_setzero_pd(), _CMP_LT_OQ);
        let m_b = _mm512_cmp_pd_mask(v_g_b, _mm512_setzero_pd(), _CMP_LT_OQ);
        v_g_a = _mm512_mask_add_pd(v_g_a, m_a, v_g_a, v_p);
        v_g_b = _mm512_mask_add_pd(v_g_b, m_b, v_g_b, v_p);

        // Convert back and store
        let v_result_a = _mm512_cvt_roundpd_epu64(v_g_a, ROUND_MODE);
        let v_result_b = _mm512_cvt_roundpd_epu64(v_g_b, ROUND_MODE);

        _mm512_storeu_si512(res_ptr, v_result_a);
        _mm512_storeu_si512(res_ptr.add(1), v_result_b);

        res_ptr = res_ptr.add(2);
        op1_ptr = op1_ptr.add(2);
        op2_ptr = op2_ptr.add(2);
        i += 16;
    }

    // Handle remaining 8-element tail (when n is not a multiple of 16)
    if i < n {
        let v_op1 = _mm512_loadu_si512(op1_ptr);
        let v_op2 = _mm512_loadu_si512(op2_ptr);

        let v_op1 = mm512_hexl_small_mod_epu64::<INPUT_MOD_FACTOR>(
            v_op1,
            v_modulus,
            Some(&v_twice_mod),
            None,
        );
        let v_op2 = mm512_hexl_small_mod_epu64::<INPUT_MOD_FACTOR>(
            v_op2,
            v_modulus,
            Some(&v_twice_mod),
            None,
        );

        let v_x = _mm512_cvt_roundepu64_pd(v_op1, ROUND_MODE);
        let v_y = _mm512_cvt_roundepu64_pd(v_op2, ROUND_MODE);

        let v_h = _mm512_mul_pd(v_x, v_y);
        let v_l = _mm512_fmsub_pd(v_x, v_y, v_h);
        let v_b = _mm512_mul_pd(v_h, v_u);
        let v_c = _mm512_roundscale_pd(v_b, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        let v_d = _mm512_fnmadd_pd(v_c, v_p, v_h);
        let mut v_g = _mm512_add_pd(v_d, v_l);

        let m = _mm512_cmp_pd_mask(v_g, _mm512_setzero_pd(), _CMP_LT_OQ);
        v_g = _mm512_mask_add_pd(v_g, m, v_g, v_p);

        let v_result = _mm512_cvt_roundpd_epu64(v_g, ROUND_MODE);
        _mm512_storeu_si512(res_ptr, v_result);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
#[inline]
unsafe fn eltwise_mult_mod_avx512_float<const INPUT_MOD_FACTOR: i32>(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    modulus: u64,
) {
    debug_assert!(modulus < maximum_value(50));
    debug_assert!(modulus > 1);

    let mut n = result.len() as u64;
    let mut op1 = operand1;
    let mut op2 = operand2;
    let mut res = result;

    let n_mod_8 = n % 8;
    if n_mod_8 != 0 {
        eltwise_mult_mod_native_dispatch::<INPUT_MOD_FACTOR>(
            &mut res[..n_mod_8 as usize],
            &op1[..n_mod_8 as usize],
            &op2[..n_mod_8 as usize],
            modulus,
        );
        op1 = &op1[n_mod_8 as usize..];
        op2 = &op2[n_mod_8 as usize..];
        res = &mut res[n_mod_8 as usize..];
        n -= n_mod_8;
    }

    let v_p = _mm512_set1_pd(modulus as f64);
    let v_modulus = _mm512_set1_epi64(modulus as i64);
    let v_twice_mod = _mm512_set1_epi64((modulus * 2) as i64);

    let u_bar = (1.0 + f64::EPSILON) / modulus as f64;
    let v_u = _mm512_set1_pd(u_bar);

    let vp_operand1 = op1.as_ptr() as *const __m512i;
    let vp_operand2 = op2.as_ptr() as *const __m512i;
    let vp_result = res.as_mut_ptr() as *mut __m512i;

    let no_input_reduce_mod =
        (INPUT_MOD_FACTOR as u64 * INPUT_MOD_FACTOR as u64 * modulus) < (1u64 << 50);
    if no_input_reduce_mod {
        eltwise_mult_mod_avx512_float_loop::<1>(
            vp_result,
            vp_operand1,
            vp_operand2,
            v_u,
            v_p,
            v_modulus,
            v_twice_mod,
            n,
        );
    } else {
        eltwise_mult_mod_avx512_float_loop::<INPUT_MOD_FACTOR>(
            vp_result,
            vp_operand1,
            vp_operand2,
            v_u,
            v_p,
            v_modulus,
            v_twice_mod,
            n,
        );
    }
}

pub fn eltwise_fma_mod(result: &mut [u64], arg1: &[u64], arg2: u64, arg3: &[u64], modulus: u64) {
    let input_mod_factor = 1u64;
    eltwise_fma_mod_internal(result, arg1, arg2, Some(arg3), modulus, input_mod_factor);
}

fn eltwise_fma_mod_internal(
    result: &mut [u64],
    arg1: &[u64],
    arg2: u64,
    arg3: Option<&[u64]>,
    modulus: u64,
    input_mod_factor: u64,
) {
    debug_assert!(
        input_mod_factor == 1
            || input_mod_factor == 2
            || input_mod_factor == 4
            || input_mod_factor == 8
    );

    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX512IFMA && input_mod_factor * modulus < (1u64 << 51) {
            unsafe {
                match input_mod_factor {
                    1 => eltwise_fma_mod_avx512::<52, 1>(result, arg1, arg2, arg3, modulus),
                    2 => eltwise_fma_mod_avx512::<52, 2>(result, arg1, arg2, arg3, modulus),
                    4 => eltwise_fma_mod_avx512::<52, 4>(result, arg1, arg2, arg3, modulus),
                    8 => eltwise_fma_mod_avx512::<52, 8>(result, arg1, arg2, arg3, modulus),
                    _ => {}
                }
                return;
            }
        }

        if *HAS_AVX512DQ {
            unsafe {
                match input_mod_factor {
                    1 => eltwise_fma_mod_avx512::<64, 1>(result, arg1, arg2, arg3, modulus),
                    2 => eltwise_fma_mod_avx512::<64, 2>(result, arg1, arg2, arg3, modulus),
                    4 => eltwise_fma_mod_avx512::<64, 4>(result, arg1, arg2, arg3, modulus),
                    8 => eltwise_fma_mod_avx512::<64, 8>(result, arg1, arg2, arg3, modulus),
                    _ => {}
                }
                return;
            }
        }
    }

    match input_mod_factor {
        1 => eltwise_fma_mod_native::<1>(result, arg1, arg2, arg3, modulus),
        2 => eltwise_fma_mod_native::<2>(result, arg1, arg2, arg3, modulus),
        4 => eltwise_fma_mod_native::<4>(result, arg1, arg2, arg3, modulus),
        8 => eltwise_fma_mod_native::<8>(result, arg1, arg2, arg3, modulus),
        _ => {}
    }
}

fn eltwise_fma_mod_native<const INPUT_MOD_FACTOR: u64>(
    result: &mut [u64],
    arg1: &[u64],
    arg2: u64,
    arg3: Option<&[u64]>,
    modulus: u64,
) {
    let twice_modulus = 2 * modulus;
    let four_times_modulus = 4 * modulus;
    let arg2 = reduce_mod::<INPUT_MOD_FACTOR>(
        arg2,
        modulus,
        Some(&twice_modulus),
        Some(&four_times_modulus),
    );

    let mf = MultiplyFactor::new(arg2, 64, modulus);
    let arg2_precon = mf.barrett_factor();

    if let Some(arg3) = arg3 {
        for i in 0..result.len() {
            let arg1_val = reduce_mod::<INPUT_MOD_FACTOR>(
                arg1[i],
                modulus,
                Some(&twice_modulus),
                Some(&four_times_modulus),
            );
            let arg3_val = reduce_mod::<INPUT_MOD_FACTOR>(
                arg3[i],
                modulus,
                Some(&twice_modulus),
                Some(&four_times_modulus),
            );
            let result_val = multiply_mod_precon(arg1_val, arg2, arg2_precon, modulus);
            result[i] = add_uint_mod(result_val, arg3_val, modulus);
        }
    } else {
        for i in 0..result.len() {
            let arg1_val = reduce_mod::<INPUT_MOD_FACTOR>(
                arg1[i],
                modulus,
                Some(&twice_modulus),
                Some(&four_times_modulus),
            );
            result[i] = multiply_mod_precon(arg1_val, arg2, arg2_precon, modulus);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
#[inline]
unsafe fn eltwise_fma_mod_avx512<const BITSHIFT: i32, const INPUT_MOD_FACTOR: i32>(
    result: &mut [u64],
    arg1: &[u64],
    arg2: u64,
    arg3: Option<&[u64]>,
    modulus: u64,
) {
    debug_assert!(modulus < maximum_value(BITSHIFT as u64));

    let mut n = result.len();
    let mut a1 = arg1;
    let mut a3 = arg3;
    let mut res = result;

    let n_mod_8 = n % 8;
    if n_mod_8 != 0 {
        eltwise_fma_mod_native_dispatch::<INPUT_MOD_FACTOR>(
            &mut res[..n_mod_8],
            &a1[..n_mod_8],
            arg2,
            a3.map(|v| &v[..n_mod_8]),
            modulus,
        );
        a1 = &a1[n_mod_8..];
        if let Some(v) = a3 {
            a3 = Some(&v[n_mod_8..]);
        }
        res = &mut res[n_mod_8..];
        n -= n_mod_8;
    }

    let twice_modulus = 2 * modulus;
    let four_times_modulus = 4 * modulus;
    let arg2 = reduce_mod_input::<INPUT_MOD_FACTOR>(
        arg2,
        modulus,
        Some(&twice_modulus),
        Some(&four_times_modulus),
    );
    let arg2_barr = MultiplyFactor::new(arg2, BITSHIFT as u64, modulus).barrett_factor();

    let varg2_barr = _mm512_set1_epi64(arg2_barr as i64);
    let v_modulus = _mm512_set1_epi64(modulus as i64);
    let v_neg_modulus = _mm512_set1_epi64(-(modulus as i64));
    let v2_modulus = _mm512_set1_epi64((2 * modulus) as i64);
    let v4_modulus = _mm512_set1_epi64((4 * modulus) as i64);

    let mut varg2 = _mm512_set1_epi64(arg2 as i64);
    varg2 = mm512_hexl_small_mod_epu64::<INPUT_MOD_FACTOR>(
        varg2,
        v_modulus,
        Some(&v2_modulus),
        Some(&v4_modulus),
    );

    let mut vp_arg1 = a1.as_ptr() as *const __m512i;
    let mut vp_result = res.as_mut_ptr() as *mut __m512i;

    if let Some(arg3) = a3 {
        let mut vp_arg3 = arg3.as_ptr() as *const __m512i;
        for _ in (0..n).step_by(8) {
            let mut v_arg1 = _mm512_loadu_si512(vp_arg1);
            let mut v_arg3 = _mm512_loadu_si512(vp_arg3);

            v_arg1 = mm512_hexl_small_mod_epu64::<INPUT_MOD_FACTOR>(
                v_arg1,
                v_modulus,
                Some(&v2_modulus),
                Some(&v4_modulus),
            );
            v_arg3 = mm512_hexl_small_mod_epu64::<INPUT_MOD_FACTOR>(
                v_arg3,
                v_modulus,
                Some(&v2_modulus),
                Some(&v4_modulus),
            );

            let va_times_b = mm512_hexl_mullo_epi::<BITSHIFT>(v_arg1, varg2);
            let vq = mm512_hexl_mulhi_epi::<BITSHIFT>(v_arg1, varg2_barr);
            let mut vq = mm512_hexl_mullo_add_lo_epi::<BITSHIFT>(va_times_b, vq, v_neg_modulus);
            vq = _mm512_add_epi64(vq, v_arg3);
            vq = mm512_hexl_small_mod_epu64::<4>(vq, v_modulus, Some(&v2_modulus), None);

            _mm512_storeu_si512(vp_result, vq);
            vp_arg1 = vp_arg1.add(1);
            vp_result = vp_result.add(1);
            vp_arg3 = vp_arg3.add(1);
        }
    } else {
        for _ in (0..n).step_by(8) {
            let mut v_arg1 = _mm512_loadu_si512(vp_arg1);
            v_arg1 = mm512_hexl_small_mod_epu64::<INPUT_MOD_FACTOR>(
                v_arg1,
                v_modulus,
                Some(&v2_modulus),
                Some(&v4_modulus),
            );

            let va_times_b = mm512_hexl_mullo_epi::<BITSHIFT>(v_arg1, varg2);
            let vq = mm512_hexl_mulhi_epi::<BITSHIFT>(v_arg1, varg2_barr);
            let mut vq = mm512_hexl_mullo_add_lo_epi::<BITSHIFT>(va_times_b, vq, v_neg_modulus);
            vq = mm512_hexl_small_mod_epu64::<2>(vq, v_modulus, None, None);
            _mm512_storeu_si512(vp_result, vq);
            vp_arg1 = vp_arg1.add(1);
            vp_result = vp_result.add(1);
        }
    }
}

/// Fused incomplete NTT ring multiplication (Karatsuba variant).
///
/// For each i in 0..n, computes:
///   result[i]   = op1[i]*op2[i] + shift[i]*(op1[n+i]*op2[n+i])   (mod modulus)
///   result[n+i] = op1[n+i]*op2[i] + op1[i]*op2[n+i]              (mod modulus)
///
/// Uses the Karatsuba identity  b·c + a·d = (a+b)(c+d) − a·c − b·d  to reduce
/// from 5 modular multiplications to 4, cutting FMA-port pressure by 20%.
///
/// This is the low-level inner routine that accepts explicit shift-factor slices.
/// Prefer the higher-level `fused_incomplete_ntt_mult` in `lib.rs` which caches
/// the shift factors automatically.
#[inline]
pub fn fused_incomplete_ntt_mult_inner(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    shift_factors: &[u64],
    shift_factors_f64: &[f64],
    n: usize,
    modulus: u64,
) {
    debug_assert!(n % 8 == 0);
    debug_assert!(result.len() >= 2 * n);
    debug_assert!(operand1.len() >= 2 * n);
    debug_assert!(operand2.len() >= 2 * n);
    debug_assert!(shift_factors.len() >= n);
    debug_assert!(shift_factors_f64.len() >= n);

    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX512DQ && modulus < (1u64 << 50) {
            unsafe {
                fused_incomplete_ntt_mult_avx512_float(
                    result,
                    operand1,
                    operand2,
                    shift_factors_f64,
                    n,
                    modulus,
                );
                return;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if HAS_NEON && modulus < (1u64 << 50) {
            unsafe {
                fused_incomplete_ntt_mult_neon(
                    result,
                    operand1,
                    operand2,
                    shift_factors_f64,
                    n,
                    modulus,
                );
                return;
            }
        }
    }

    fused_incomplete_ntt_mult_native(result, operand1, operand2, shift_factors, n, modulus);
}

fn fused_incomplete_ntt_mult_native(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    shift_factors: &[u64],
    n: usize,
    modulus: u64,
) {
    use crate::number_theory::{add_uint_mod, multiply_mod, sub_uint_mod};

    for i in 0..n {
        let a = operand1[i];
        let b = operand1[n + i];
        let c = operand2[i];
        let d = operand2[n + i];
        let s = shift_factors[i];

        // Shared products
        let ac = multiply_mod(a, c, modulus);
        let bd = multiply_mod(b, d, modulus);

        // result_even = ac + s*bd
        let sbd = multiply_mod(s, bd, modulus);
        result[i] = add_uint_mod(ac, sbd, modulus);

        // Karatsuba: b*c + a*d = (a+b)*(c+d) - ac - bd
        let ab = add_uint_mod(a, b, modulus);
        let cd = add_uint_mod(c, d, modulus);
        let abcd = multiply_mod(ab, cd, modulus);
        result[n + i] = sub_uint_mod(sub_uint_mod(abcd, ac, modulus), bd, modulus);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
#[inline]
unsafe fn fused_incomplete_ntt_mult_avx512_float(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    shift_factors_f64: &[f64],
    n: usize,
    modulus: u64,
) {
    let v_p = _mm512_set1_pd(modulus as f64);
    let u_bar = (1.0 + f64::EPSILON) / modulus as f64;
    let v_u = _mm512_set1_pd(u_bar);
    let v_zero = _mm512_setzero_pd();

    const ROUND_MODE: i32 = _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC;
    const FLOOR: i32 = _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC;

    let op1_e = operand1.as_ptr();
    let op1_o = operand1.as_ptr().add(n);
    let op2_e = operand2.as_ptr();
    let op2_o = operand2.as_ptr().add(n);
    let shift_f = shift_factors_f64.as_ptr();
    let res_e = result.as_mut_ptr();
    let res_o = result.as_mut_ptr().add(n);

    // Float modular multiply: (x * y) mod p  via Dekker error-free product.
    // Both x, y must be exact float representations of integers in [0, p).
    // Result is an exact float representation of an integer in [0, p).
    macro_rules! fmul_mod {
        ($x:expr, $y:expr) => {{
            let h = _mm512_mul_pd($x, $y);
            let l = _mm512_fmsub_pd($x, $y, h);
            let b = _mm512_mul_pd(h, v_u);
            let c = _mm512_roundscale_pd(b, FLOOR);
            let d = _mm512_fnmadd_pd(c, v_p, h);
            let g = _mm512_add_pd(d, l);
            let m = _mm512_cmp_pd_mask(g, v_zero, _CMP_LT_OQ);
            _mm512_mask_add_pd(g, m, g, v_p)
        }};
    }

    // Float modular add: (a + b) mod p.  a, b in [0, p) ⇒ sum in [0, 2p).
    // Since p < 2^50, sum < 2^51, exactly representable in f64.
    macro_rules! fadd_mod {
        ($a:expr, $b:expr) => {{
            let sum = _mm512_add_pd($a, $b);
            let m = _mm512_cmp_pd_mask(v_p, sum, _CMP_LE_OQ);
            _mm512_mask_sub_pd(sum, m, sum, v_p)
        }};
    }

    // Float modular sub: (a - b) mod p.  a, b in [0, p) ⇒ diff in (-p, p).
    // Exactly representable in f64. Conditionally add p if negative.
    macro_rules! fsub_mod {
        ($a:expr, $b:expr) => {{
            let diff = _mm512_sub_pd($a, $b);
            let m = _mm512_cmp_pd_mask(diff, v_zero, _CMP_LT_OQ);
            _mm512_mask_add_pd(diff, m, diff, v_p)
        }};
    }

    let mut i = 0usize;
    while i < n {
        // Load 4 operand vectors as u64, 1 shift vector as precomputed f64.
        // Using unaligned loads for compatibility with arbitrary allocators;
        // on modern x86 the penalty is zero when data happens to be aligned.
        let v_a = _mm512_loadu_si512(op1_e.add(i) as *const __m512i);
        let v_b = _mm512_loadu_si512(op1_o.add(i) as *const __m512i);
        let v_c = _mm512_loadu_si512(op2_e.add(i) as *const __m512i);
        let v_d = _mm512_loadu_si512(op2_o.add(i) as *const __m512i);

        // Convert operands to double (all values < 2^50, exactly representable)
        let f_a = _mm512_cvt_roundepu64_pd(v_a, ROUND_MODE);
        let f_b = _mm512_cvt_roundepu64_pd(v_b, ROUND_MODE);
        let f_c = _mm512_cvt_roundepu64_pd(v_c, ROUND_MODE);
        let f_d = _mm512_cvt_roundepu64_pd(v_d, ROUND_MODE);
        // Shift factors already f64 — use unaligned load for portability.
        // The internal AlignedVecF64 guarantees alignment, so no perf penalty.
        let f_s = _mm512_loadu_pd(shift_f.add(i));

        // Karatsuba prep: (a+b) mod q, (c+d) mod q — cheap float adds
        let f_ab = fadd_mod!(f_a, f_b);
        let f_cd = fadd_mod!(f_c, f_d);

        // 3 independent modular multiplies (maximal ILP)
        let ac = fmul_mod!(f_a, f_c);
        let bd = fmul_mod!(f_b, f_d);
        let abcd = fmul_mod!(f_ab, f_cd);

        // 1 dependent multiply: s * bd (needs bd result)
        let sbd = fmul_mod!(f_s, bd);

        // result_even = ac + s*bd
        let f_re = fadd_mod!(ac, sbd);

        // result_odd = (a+b)(c+d) - ac - bd   [Karatsuba]
        let tmp = fsub_mod!(abcd, ac);
        let f_ro = fsub_mod!(tmp, bd);

        // Convert back to integers and store
        let v_re = _mm512_cvt_roundpd_epu64(f_re, ROUND_MODE);
        let v_ro = _mm512_cvt_roundpd_epu64(f_ro, ROUND_MODE);
        _mm512_storeu_si512(res_e.add(i) as *mut __m512i, v_re);
        _mm512_storeu_si512(res_o.add(i) as *mut __m512i, v_ro);

        i += 8;
    }
}

/// Product over degree-4 slots: slot `i` is `(op[i], op[n+i], op[2n+i], op[3n+i])` modulo
/// `Y^4 - zeta[i]`, so `c_k = sum_{j+l=k} a_j b_l + zeta * sum_{j+l=k+4} a_j b_l`.
pub(crate) fn fused_slot4_mult_inner(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    zetas: &[u64],
    zetas_f64: &[f64],
    n: usize,
    modulus: u64,
) {
    debug_assert!(n % 8 == 0);
    debug_assert!(result.len() >= 4 * n);
    debug_assert!(operand1.len() >= 4 * n);
    debug_assert!(operand2.len() >= 4 * n);
    debug_assert!(zetas.len() >= n);
    debug_assert!(zetas_f64.len() >= n);

    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX512DQ && modulus < (1u64 << 50) {
            unsafe {
                fused_slot4_mult_avx512_float(result, operand1, operand2, zetas_f64, n, modulus);
                return;
            }
        }
    }

    fused_slot4_mult_native(result, operand1, operand2, zetas, n, modulus);
}

// Karatsuba over `a = (a0 + a1 Y) + Y^2 (a2 + a3 Y)`: products `l*`, `h*`, `s*` of the low
// halves, high halves and half sums; `Y^4..Y^6` fold back through `zeta`.
pub(crate) fn fused_slot4_mult_native(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    zetas: &[u64],
    n: usize,
    modulus: u64,
) {
    use crate::number_theory::{multiply_mod, sub_uint_mod};

    let add = |x, y| add_uint_mod(x, y, modulus);
    let sub = |x, y| sub_uint_mod(x, y, modulus);
    let mul = |x, y| multiply_mod(x, y, modulus);

    for i in 0..n {
        let (a0, a1, a2, a3) = (
            operand1[i],
            operand1[n + i],
            operand1[2 * n + i],
            operand1[3 * n + i],
        );
        let (b0, b1, b2, b3) = (
            operand2[i],
            operand2[n + i],
            operand2[2 * n + i],
            operand2[3 * n + i],
        );
        let (a01, a23, a02, a13) = (add(a0, a1), add(a2, a3), add(a0, a2), add(a1, a3));
        let (b01, b23, b02, b13) = (add(b0, b1), add(b2, b3), add(b0, b2), add(b1, b3));

        let (l0, l2, lm) = (mul(a0, b0), mul(a1, b1), mul(a01, b01));
        let (h0, h2, hm) = (mul(a2, b2), mul(a3, b3), mul(a23, b23));
        let (s0, s2, sm) = (
            mul(a02, b02),
            mul(a13, b13),
            mul(add(a02, a13), add(b02, b13)),
        );

        let p1 = sub(sub(lm, l0), l2);
        let p4 = sub(sub(add(s2, h0), l2), h2);
        let p5 = sub(sub(hm, h0), h2);
        let z = zetas[i];

        result[i] = add(l0, mul(z, p4));
        result[n + i] = add(p1, mul(z, p5));
        result[2 * n + i] = add(sub(sub(add(l2, s0), l0), h0), mul(z, h2));
        result[3 * n + i] = sub(sub(sub(sub(sm, s0), s2), p1), p5);
    }
}

// Lazy float reduction, p < 2^50: `fmul!` on |x| <= X p, |y| <= Y p returns x y mod p within
// (1/2 + 0.376 X Y) p of zero. With centred operand sums the products stay below 0.876 p
// (`sm` below 2.01 p) and, with `p5` reduced, every later sum below 7.4 p < 2^53, so all
// float sums are exact; `reduce!` maps |t| < 8p to [0, p).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
#[inline]
pub(crate) unsafe fn fused_slot4_mult_avx512_float(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    zetas_f64: &[f64],
    n: usize,
    modulus: u64,
) {
    let v_p = _mm512_set1_pd(modulus as f64);
    let v_u = _mm512_set1_pd(1.0 / modulus as f64);
    let v_zero = _mm512_setzero_pd();

    const ROUND_MODE: i32 = _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC;
    const NEAREST: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

    let op1 = operand1.as_ptr();
    let op2 = operand2.as_ptr();
    let zeta = zetas_f64.as_ptr();
    let res = result.as_mut_ptr();

    macro_rules! fmul {
        ($x:expr, $y:expr) => {{
            let h = _mm512_mul_pd($x, $y);
            let l = _mm512_fmsub_pd($x, $y, h);
            let c = _mm512_roundscale_pd(_mm512_mul_pd(h, v_u), NEAREST);
            _mm512_add_pd(_mm512_fnmadd_pd(c, v_p, h), l)
        }};
    }

    macro_rules! reduce {
        ($t:expr) => {{
            let t = $t;
            let c = _mm512_roundscale_pd(_mm512_mul_pd(t, v_u), NEAREST);
            let r = _mm512_fnmadd_pd(c, v_p, t);
            let m = _mm512_cmp_pd_mask(r, v_zero, _CMP_LT_OQ);
            _mm512_mask_add_pd(r, m, r, v_p)
        }};
    }

    macro_rules! load {
        ($ptr:expr, $offset:expr) => {
            _mm512_cvt_roundepu64_pd(
                _mm512_loadu_si512($ptr.add($offset) as *const __m512i),
                ROUND_MODE,
            )
        };
    }

    macro_rules! store {
        ($offset:expr, $x:expr) => {
            _mm512_storeu_si512(
                res.add($offset) as *mut __m512i,
                _mm512_cvt_roundpd_epu64($x, ROUND_MODE),
            )
        };
    }

    macro_rules! centred_sums {
        ($x0:expr, $x1:expr, $x2:expr, $x3:expr) => {{
            let x0 = _mm512_sub_pd($x0, v_p);
            let x3 = _mm512_sub_pd($x3, v_p);
            let x01 = _mm512_add_pd(x0, $x1);
            let x23 = _mm512_add_pd($x2, x3);
            (
                x01,
                x23,
                _mm512_add_pd(x0, $x2),
                _mm512_add_pd($x1, x3),
                _mm512_add_pd(x01, x23),
            )
        }};
    }

    let mut i = 0usize;
    while i < n {
        let (a0, a1, a2, a3) = (
            load!(op1, i),
            load!(op1, n + i),
            load!(op1, 2 * n + i),
            load!(op1, 3 * n + i),
        );
        let (b0, b1, b2, b3) = (
            load!(op2, i),
            load!(op2, n + i),
            load!(op2, 2 * n + i),
            load!(op2, 3 * n + i),
        );
        let z = _mm512_loadu_pd(zeta.add(i));
        let (a01, a23, a02, a13, a0123) = centred_sums!(a0, a1, a2, a3);
        let (b01, b23, b02, b13, b0123) = centred_sums!(b0, b1, b2, b3);

        let (l0, l2, lm) = (fmul!(a0, b0), fmul!(a1, b1), fmul!(a01, b01));
        let (h0, h2, hm) = (fmul!(a2, b2), fmul!(a3, b3), fmul!(a23, b23));
        let (s0, s2, sm) = (fmul!(a02, b02), fmul!(a13, b13), fmul!(a0123, b0123));

        let p1 = _mm512_sub_pd(_mm512_sub_pd(lm, l0), l2);
        let p4 = _mm512_sub_pd(_mm512_sub_pd(_mm512_add_pd(s2, h0), l2), h2);
        let p5 = reduce!(_mm512_sub_pd(_mm512_sub_pd(hm, h0), h2));

        let c0 = _mm512_add_pd(l0, fmul!(z, p4));
        let c1 = _mm512_add_pd(p1, fmul!(z, p5));
        let l2s0 = _mm512_add_pd(l2, s0);
        let c2 = _mm512_add_pd(_mm512_sub_pd(l2s0, _mm512_add_pd(l0, h0)), fmul!(z, h2));
        let c3 = _mm512_sub_pd(
            _mm512_sub_pd(_mm512_sub_pd(sm, s0), s2),
            _mm512_add_pd(p1, p5),
        );

        store!(i, reduce!(c0));
        store!(n + i, reduce!(c1));
        store!(2 * n + i, reduce!(c2));
        store!(3 * n + i, reduce!(c3));

        i += 8;
    }
}

// NEON 2-lane sibling of `fused_incomplete_ntt_mult_avx512_float`: same Karatsuba
// structure and float-Barrett reduction. NEON has no high-half or widening 64-bit
// integer multiply, so we reduce via the f64 53-bit mantissa with an FMA-based
// Dekker product, as the AVX-512 float path does. Modulus must fit (< 2^50, gated
// by the dispatcher above).

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn fmul_mod(
    x: float64x2_t,
    y: float64x2_t,
    p: float64x2_t,
    u: float64x2_t,
    zero: float64x2_t,
) -> float64x2_t {
    // Dekker error-free product: h + l = x*y exactly. NEON's vfmsq_f64(c, a, b)
    // computes c - a*b, so l_neg = h - x*y carries the negation of l. We carry
    // l_neg and subtract in the final combine instead of negating once.
    let h = vmulq_f64(x, y);
    let l_neg = vfmsq_f64(h, x, y);

    // Floating Barrett quotient: c = floor(h * mu), mu = (1 + eps) / p.
    let b_quot = vmulq_f64(h, u);
    let c = vrndmq_f64(b_quot);

    // d = h - c*p (single-rounded FMA).
    let d = vfmsq_f64(h, c, p);

    // g = (h - c*p) - l_neg = (h - c*p) + l = x*y - c*p, in (-p, p).
    let g = vsubq_f64(d, l_neg);

    // Conditional add p if g < 0.
    let mask = vcltq_f64(g, zero);
    vbslq_f64(mask, vaddq_f64(g, p), g)
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn fadd_mod(a: float64x2_t, b: float64x2_t, p: float64x2_t) -> float64x2_t {
    let sum = vaddq_f64(a, b);
    let mask = vcleq_f64(p, sum); // p <= sum
    vbslq_f64(mask, vsubq_f64(sum, p), sum)
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn fsub_mod(
    a: float64x2_t,
    b: float64x2_t,
    p: float64x2_t,
    zero: float64x2_t,
) -> float64x2_t {
    let diff = vsubq_f64(a, b);
    let mask = vcltq_f64(diff, zero);
    vbslq_f64(mask, vaddq_f64(diff, p), diff)
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn fused_incomplete_ntt_mult_neon(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    shift_factors_f64: &[f64],
    n: usize,
    modulus: u64,
) {
    let v_p = vdupq_n_f64(modulus as f64);
    let v_u = vdupq_n_f64((1.0 + f64::EPSILON) / modulus as f64);
    let v_zero = vdupq_n_f64(0.0);

    let op1_e = operand1.as_ptr();
    let op1_o = operand1.as_ptr().add(n);
    let op2_e = operand2.as_ptr();
    let op2_o = operand2.as_ptr().add(n);
    let shift = shift_factors_f64.as_ptr();
    let res_e = result.as_mut_ptr();
    let res_o = result.as_mut_ptr().add(n);

    let mut i = 0usize;
    while i < n {
        let f_a = vcvtq_f64_u64(vld1q_u64(op1_e.add(i)));
        let f_b = vcvtq_f64_u64(vld1q_u64(op1_o.add(i)));
        let f_c = vcvtq_f64_u64(vld1q_u64(op2_e.add(i)));
        let f_d = vcvtq_f64_u64(vld1q_u64(op2_o.add(i)));
        let f_s = vld1q_f64(shift.add(i));

        // Karatsuba prep: (a+b) mod p, (c+d) mod p
        let f_ab = fadd_mod(f_a, f_b, v_p);
        let f_cd = fadd_mod(f_c, f_d, v_p);

        // 3 independent modular multiplies (maximal ILP)
        let ac = fmul_mod(f_a, f_c, v_p, v_u, v_zero);
        let bd = fmul_mod(f_b, f_d, v_p, v_u, v_zero);
        let abcd = fmul_mod(f_ab, f_cd, v_p, v_u, v_zero);

        // 1 dependent multiply: s * bd
        let sbd = fmul_mod(f_s, bd, v_p, v_u, v_zero);

        // result_even = ac + s*bd
        let f_re = fadd_mod(ac, sbd, v_p);

        // result_odd = (a+b)(c+d) - ac - bd  [Karatsuba]
        let tmp = fsub_mod(abcd, ac, v_p, v_zero);
        let f_ro = fsub_mod(tmp, bd, v_p, v_zero);

        vst1q_u64(res_e.add(i), vcvtq_u64_f64(f_re));
        vst1q_u64(res_o.add(i), vcvtq_u64_f64(f_ro));

        i += 2;
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod neon_tests {
    use super::*;

    /// Both moduli we exercise are < 2^50 (the dispatch gate). The first is the
    /// production modulus rokoko passes in (`MOD_Q`); the second is an
    /// independent value. The float-Barrett constant `mu = (1 + eps) / p`
    /// depends on `p`, so a kernel can be bit-exact for one modulus and misround
    /// for another — we check both.
    const MODULI: [u64; 2] = [1125899906839937, 1125899906826241];

    /// Deterministic xorshift64 data in [0, modulus). Avoids dragging in `rand`.
    fn deterministic_vec(len: usize, modulus: u64, seed: u64) -> Vec<u64> {
        let mut state = seed | 1; // xorshift requires non-zero state
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            v.push(state % modulus);
        }
        v
    }

    /// Run the scalar oracle and the NEON kernel over identical inputs and
    /// assert bit-exactness. `n` must be even: the NEON loop strides by 2 over
    /// each half, so an odd tail would read across the even/odd boundary.
    fn assert_neon_matches_scalar(
        op1: &[u64],
        op2: &[u64],
        shift: &[u64],
        n: usize,
        modulus: u64,
        ctx: &str,
    ) {
        assert_eq!(n % 2, 0, "kernel requires even n ({ctx})");
        let shift_f64: Vec<f64> = shift.iter().map(|&x| x as f64).collect();

        let mut scalar_result = vec![0u64; 2 * n];
        fused_incomplete_ntt_mult_native(&mut scalar_result, op1, op2, shift, n, modulus);

        let mut neon_result = vec![0u64; 2 * n];
        unsafe {
            fused_incomplete_ntt_mult_neon(&mut neon_result, op1, op2, &shift_f64, n, modulus);
        }

        assert_eq!(scalar_result, neon_result, "NEON diverged from scalar ({ctx})");
    }

    /// Randomized scalar-vs-NEON bit-exactness across both moduli, all bench
    /// sizes, and many seeds — differential testing over a wide input sample
    /// rather than a single seed per size. Also guards against a dispatcher-only
    /// test hiding a NEON-kernel regression behind a dispatcher misroute.
    #[test]
    fn test_fused_neon_matches_scalar() {
        let sizes = [8usize, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
        for &modulus in MODULI.iter() {
            for &n in sizes.iter() {
                for seed in 0..32u64 {
                    let op1 = deterministic_vec(2 * n, modulus, 0x1000 + seed);
                    let op2 = deterministic_vec(2 * n, modulus, 0x2000 + seed);
                    let shift = deterministic_vec(n, modulus, 0x3000 + seed);
                    assert_neon_matches_scalar(
                        &op1,
                        &op2,
                        &shift,
                        n,
                        modulus,
                        &format!("random m={modulus} n={n} seed={seed}"),
                    );
                }
            }
        }
    }

    /// Exhaustive boundary sweep. Float-Barrett off-by-ones live where the
    /// quotient `floor(h * mu)` lands next to an integer — i.e. at extreme
    /// operands such as `(p-1) * (p-1)`. Each lane independently draws every
    /// input from the edge set, so one call covers all `6^5` (a,b,c,d,s)
    /// combinations.
    #[test]
    fn test_fused_neon_boundary_values() {
        for &modulus in MODULI.iter() {
            let edges = [0u64, 1, 2, modulus / 2, modulus - 2, modulus - 1];
            let k = edges.len();
            let n = k.pow(5); // 6^5 = 7776, even

            let mut op1 = vec![0u64; 2 * n]; // [a.. | b..]
            let mut op2 = vec![0u64; 2 * n]; // [c.. | d..]
            let mut shift = vec![0u64; n];

            for i in 0..n {
                let mut idx = i;
                let a = edges[idx % k];
                idx /= k;
                let b = edges[idx % k];
                idx /= k;
                let c = edges[idx % k];
                idx /= k;
                let d = edges[idx % k];
                idx /= k;
                let s = edges[idx % k];
                op1[i] = a;
                op1[n + i] = b;
                op2[i] = c;
                op2[n + i] = d;
                shift[i] = s;
            }

            assert_neon_matches_scalar(
                &op1,
                &op2,
                &shift,
                n,
                modulus,
                &format!("boundary m={modulus}"),
            );
        }
    }
}

#[cfg(test)]
mod slot_tests {
    use super::*;
    use crate::number_theory::{multiply_mod, sub_uint_mod};

    const SLOT_CASES: [(usize, u64); 4] = [
        (128, 1125899906842177),
        (256, 1125899906839937),
        (128, 1125899906839937),
        (256, 1125899906840833),
    ];

    fn xorshift_vec(len: usize, modulus: u64, seed: u64) -> Vec<u64> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state % modulus
            })
            .collect()
    }

    fn edge_vec(len: usize, modulus: u64, seed: u64) -> Vec<u64> {
        let edges = [0, 1, 2, modulus / 2, modulus - 2, modulus - 1];
        xorshift_vec(len, 6, seed)
            .iter()
            .map(|&k| edges[k as usize])
            .collect()
    }

    fn schoolbook(a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
        let n = a.len();
        let mut c = vec![0u64; n];
        for i in 0..n {
            for j in 0..n {
                let p = multiply_mod(a[i], b[j], modulus);
                let k = (i + j) % n;
                c[k] = if i + j < n {
                    add_uint_mod(c[k], p, modulus)
                } else {
                    sub_uint_mod(c[k], p, modulus)
                };
            }
        }
        c
    }

    fn native_slot_product(a: &[u64], b: &[u64], ring_degree: usize, modulus: u64) -> Vec<u64> {
        let d = crate::slot_degree(ring_degree, modulus);
        let n = ring_degree / d;
        let forward = |x: &[u64]| {
            let mut s = vec![0u64; ring_degree];
            crate::coefficients_to_strided(&mut s, x, ring_degree, modulus);
            crate::strided_ntt_forward_in_place(&mut s, ring_degree, modulus);
            s
        };
        let (sa, sb) = (forward(a), forward(b));
        let zetas = crate::get_ntt(n, modulus).shift_factors().to_vec();
        let mut sc = vec![0u64; ring_degree];
        match d {
            2 => fused_incomplete_ntt_mult_native(&mut sc, &sa, &sb, &zetas, n, modulus),
            _ => fused_slot4_mult_native(&mut sc, &sa, &sb, &zetas, n, modulus),
        }
        crate::strided_ntt_inverse_in_place(&mut sc, ring_degree, modulus);
        let mut c = vec![0u64; ring_degree];
        crate::strided_to_coefficients(&mut c, &sc, ring_degree, modulus);
        c
    }

    #[test]
    fn test_native_slot_product_matches_schoolbook() {
        for (ring_degree, modulus) in SLOT_CASES {
            for seed in 0..4u64 {
                let inputs = [
                    (
                        xorshift_vec(ring_degree, modulus, 0x100 + seed),
                        xorshift_vec(ring_degree, modulus, 0x200 + seed),
                    ),
                    (
                        edge_vec(ring_degree, modulus, 0x300 + seed),
                        edge_vec(ring_degree, modulus, 0x400 + seed),
                    ),
                    (
                        vec![modulus - 1; ring_degree],
                        vec![modulus - 1; ring_degree],
                    ),
                ];
                for (a, b) in &inputs {
                    assert_eq!(
                        native_slot_product(a, b, ring_degree, modulus),
                        schoolbook(a, b, modulus),
                        "N={ring_degree} q={modulus} seed={seed}"
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_slot4_avx512_matches_native() {
        if !*HAS_AVX512DQ {
            return;
        }
        let n = 1 << 14;
        for modulus in [1125899906842177u64, 1125899906839937, 1125899906826241] {
            for seed in 0..8u64 {
                let (op1, op2, zetas) = if seed % 2 == 0 {
                    (
                        xorshift_vec(4 * n, modulus, 0x1000 + seed),
                        xorshift_vec(4 * n, modulus, 0x2000 + seed),
                        xorshift_vec(n, modulus, 0x3000 + seed),
                    )
                } else {
                    (
                        edge_vec(4 * n, modulus, 0x1000 + seed),
                        edge_vec(4 * n, modulus, 0x2000 + seed),
                        edge_vec(n, modulus, 0x3000 + seed),
                    )
                };
                let zetas_f64: Vec<f64> = zetas.iter().map(|&z| z as f64).collect();
                let mut native = vec![0u64; 4 * n];
                fused_slot4_mult_native(&mut native, &op1, &op2, &zetas, n, modulus);
                let mut avx = vec![0u64; 4 * n];
                unsafe {
                    fused_slot4_mult_avx512_float(&mut avx, &op1, &op2, &zetas_f64, n, modulus);
                }
                assert_eq!(native, avx, "m={modulus} seed={seed}");
            }
        }
    }

    fn slot4_schoolbook(a: &[u64], b: &[u64], zetas: &[u64], n: usize, modulus: u64) -> Vec<u64> {
        let q = modulus as u128;
        let mut c = vec![0u64; 4 * n];
        for i in 0..n {
            let mut acc = [0u128; 7];
            for j in 0..4 {
                for l in 0..4 {
                    acc[j + l] += (a[j * n + i] as u128 * b[l * n + i] as u128) % q;
                }
            }
            for k in 0..4 {
                let high = if k < 3 { acc[k + 4] % q } else { 0 };
                c[k * n + i] = ((acc[k] + zetas[i] as u128 * high) % q) as u64;
            }
        }
        c
    }

    #[test]
    fn test_slot4_karatsuba_matches_schoolbook() {
        let n = 1 << 13;
        let moduli = [
            1125899906842177u64,
            1125899906839937,
            (1 << 50) - 1,
            (1 << 50) - 3,
            (1 << 49) + 1,
            1 << 40 | 1,
            (1 << 20) + 7,
        ];
        for modulus in moduli {
            for seed in 0..6u64 {
                let (op1, op2, zetas) = match seed % 3 {
                    0 => (
                        xorshift_vec(4 * n, modulus, 0x5000 + seed),
                        xorshift_vec(4 * n, modulus, 0x6000 + seed),
                        xorshift_vec(n, modulus, 0x7000 + seed),
                    ),
                    1 => (
                        edge_vec(4 * n, modulus, 0x5000 + seed),
                        edge_vec(4 * n, modulus, 0x6000 + seed),
                        edge_vec(n, modulus, 0x7000 + seed),
                    ),
                    _ => (
                        vec![modulus - 1; 4 * n],
                        edge_vec(4 * n, modulus, 0x6000 + seed),
                        vec![modulus - 1; n],
                    ),
                };
                let expected = slot4_schoolbook(&op1, &op2, &zetas, n, modulus);
                let mut native = vec![0u64; 4 * n];
                fused_slot4_mult_native(&mut native, &op1, &op2, &zetas, n, modulus);
                assert_eq!(native, expected, "native m={modulus} seed={seed}");
                #[cfg(target_arch = "x86_64")]
                if *HAS_AVX512DQ {
                    let zetas_f64: Vec<f64> = zetas.iter().map(|&z| z as f64).collect();
                    let mut avx = vec![0u64; 4 * n];
                    unsafe {
                        fused_slot4_mult_avx512_float(&mut avx, &op1, &op2, &zetas_f64, n, modulus);
                    }
                    assert_eq!(avx, expected, "avx m={modulus} seed={seed}");
                }
            }
        }
    }
}
