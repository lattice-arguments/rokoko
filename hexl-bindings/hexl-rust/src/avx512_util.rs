#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use crate::cpu_features::HAS_AVX512VBMI2;
use crate::util::CmpInt;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
pub unsafe fn extract_values(x: __m512i) -> [u64; 8] {
    let mut values = [0u64; 8];
    _mm512_storeu_si512(values.as_mut_ptr() as *mut __m512i, x);
    values
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
pub unsafe fn extract_int_values(x: __m512i) -> [i64; 8] {
    let mut values = [0i64; 8];
    _mm512_storeu_si512(values.as_mut_ptr() as *mut __m512i, x);
    values
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
pub unsafe fn extract_double_values(x: __m512d) -> [f64; 8] {
    let mut values = [0f64; 8];
    _mm512_storeu_pd(values.as_mut_ptr(), x);
    values
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
pub unsafe fn clear_top_bits64<const NUM_BITS: i32>(x: __m512i) -> __m512i {
    let mask = if NUM_BITS == 64 {
        u64::MAX
    } else {
        (1u64 << NUM_BITS) - 1
    };
    let low_mask = _mm512_set1_epi64(mask as i64);
    _mm512_and_epi64(x, low_mask)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
unsafe fn mm512_hexl_mulhi_epi_64(x: __m512i, y: __m512i) -> __m512i {
    let lo_mask = _mm512_set1_epi64(0x00000000ffffffffu64 as i64);
    let x_hi = _mm512_shuffle_epi32(x, 0xB1);
    let y_hi = _mm512_shuffle_epi32(y, 0xB1);
    let z_lo_lo = _mm512_mul_epu32(x, y);
    let z_lo_hi = _mm512_mul_epu32(x, y_hi);
    let z_hi_lo = _mm512_mul_epu32(x_hi, y);
    let z_hi_hi = _mm512_mul_epu32(x_hi, y_hi);

    let z_lo_lo_shift = _mm512_srli_epi64(z_lo_lo, 32);
    let sum_tmp = _mm512_add_epi64(z_lo_hi, z_lo_lo_shift);
    let sum_lo = _mm512_and_si512(sum_tmp, lo_mask);
    let sum_mid = _mm512_srli_epi64(sum_tmp, 32);
    let sum_mid2 = _mm512_add_epi64(z_hi_lo, sum_lo);
    let sum_mid2_hi = _mm512_srli_epi64(sum_mid2, 32);
    let sum_hi = _mm512_add_epi64(z_hi_hi, sum_mid);
    _mm512_add_epi64(sum_hi, sum_mid2_hi)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
#[inline(always)]
unsafe fn mm512_hexl_mulhi_epi_52(x: __m512i, y: __m512i) -> __m512i {
    let zero = _mm512_set1_epi64(0);
    _mm512_madd52hi_epu64(zero, x, y)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
pub unsafe fn mm512_hexl_mulhi_epi<const BITSHIFT: i32>(x: __m512i, y: __m512i) -> __m512i {
    if BITSHIFT == 64 {
        mm512_hexl_mulhi_epi_64(x, y)
    } else if BITSHIFT == 52 {
        mm512_hexl_mulhi_epi_52(x, y)
    } else {
        core::hint::unreachable_unchecked()
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
unsafe fn mm512_hexl_mulhi_approx_epi_64(x: __m512i, y: __m512i) -> __m512i {
    let lo_mask = _mm512_set1_epi64(0x00000000ffffffffu64 as i64);
    let x_hi = _mm512_shuffle_epi32(x, 0xB1);
    let y_hi = _mm512_shuffle_epi32(y, 0xB1);
    let z_lo_hi = _mm512_mul_epu32(x, y_hi);
    let z_hi_lo = _mm512_mul_epu32(x_hi, y);
    let z_hi_hi = _mm512_mul_epu32(x_hi, y_hi);

    let sum_lo = _mm512_and_si512(z_lo_hi, lo_mask);
    let sum_mid = _mm512_srli_epi64(z_lo_hi, 32);
    let sum_mid2 = _mm512_add_epi64(z_hi_lo, sum_lo);
    let sum_mid2_hi = _mm512_srli_epi64(sum_mid2, 32);
    let sum_hi = _mm512_add_epi64(z_hi_hi, sum_mid);
    _mm512_add_epi64(sum_hi, sum_mid2_hi)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
#[inline(always)]
unsafe fn mm512_hexl_mulhi_approx_epi_52(x: __m512i, y: __m512i) -> __m512i {
    let zero = _mm512_set1_epi64(0);
    _mm512_madd52hi_epu64(zero, x, y)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
pub unsafe fn mm512_hexl_mulhi_approx_epi<const BITSHIFT: i32>(x: __m512i, y: __m512i) -> __m512i {
    if BITSHIFT == 64 {
        mm512_hexl_mulhi_approx_epi_64(x, y)
    } else if BITSHIFT == 52 {
        mm512_hexl_mulhi_approx_epi_52(x, y)
    } else {
        core::hint::unreachable_unchecked()
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
unsafe fn mm512_hexl_mullo_epi_64(x: __m512i, y: __m512i) -> __m512i {
    _mm512_mullo_epi64(x, y)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
#[inline(always)]
unsafe fn mm512_hexl_mullo_epi_52(x: __m512i, y: __m512i) -> __m512i {
    let zero = _mm512_set1_epi64(0);
    _mm512_madd52lo_epu64(zero, x, y)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
pub unsafe fn mm512_hexl_mullo_epi<const BITSHIFT: i32>(x: __m512i, y: __m512i) -> __m512i {
    if BITSHIFT == 64 {
        mm512_hexl_mullo_epi_64(x, y)
    } else if BITSHIFT == 52 {
        mm512_hexl_mullo_epi_52(x, y)
    } else {
        core::hint::unreachable_unchecked()
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
unsafe fn mm512_hexl_mullo_add_lo_epi_64(x: __m512i, y: __m512i, z: __m512i) -> __m512i {
    let prod = _mm512_mullo_epi64(y, z);
    _mm512_add_epi64(x, prod)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
#[inline(always)]
unsafe fn mm512_hexl_mullo_add_lo_epi_52(x: __m512i, y: __m512i, z: __m512i) -> __m512i {
    let mut result = _mm512_madd52lo_epu64(x, y, z);
    result = clear_top_bits64::<52>(result);
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
pub unsafe fn mm512_hexl_mullo_add_lo_epi<const BITSHIFT: i32>(
    x: __m512i,
    y: __m512i,
    z: __m512i,
) -> __m512i {
    if BITSHIFT == 64 {
        mm512_hexl_mullo_add_lo_epi_64(x, y, z)
    } else if BITSHIFT == 52 {
        mm512_hexl_mullo_add_lo_epi_52(x, y, z)
    } else {
        core::hint::unreachable_unchecked()
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
pub unsafe fn mm512_hexl_small_mod_epu64<const INPUT_MOD_FACTOR: i32>(
    mut x: __m512i,
    q: __m512i,
    q_times_2: Option<&__m512i>,
    q_times_4: Option<&__m512i>,
) -> __m512i {
    debug_assert!(
        INPUT_MOD_FACTOR == 1
            || INPUT_MOD_FACTOR == 2
            || INPUT_MOD_FACTOR == 4
            || INPUT_MOD_FACTOR == 8
    );
    if INPUT_MOD_FACTOR == 1 {
        return x;
    }
    if INPUT_MOD_FACTOR == 2 {
        return _mm512_min_epu64(x, _mm512_sub_epi64(x, q));
    }
    if INPUT_MOD_FACTOR == 4 {
        let q_times_2 = q_times_2.expect("q_times_2 must not be None");
        x = _mm512_min_epu64(x, _mm512_sub_epi64(x, *q_times_2));
        return _mm512_min_epu64(x, _mm512_sub_epi64(x, q));
    }
    if INPUT_MOD_FACTOR == 8 {
        let q_times_2 = q_times_2.expect("q_times_2 must not be None");
        let q_times_4 = q_times_4.expect("q_times_4 must not be None");
        x = _mm512_min_epu64(x, _mm512_sub_epi64(x, *q_times_4));
        x = _mm512_min_epu64(x, _mm512_sub_epi64(x, *q_times_2));
        return _mm512_min_epu64(x, _mm512_sub_epi64(x, q));
    }
    x
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
pub unsafe fn mm512_hexl_small_add_mod_epi64(x: __m512i, y: __m512i, q: __m512i) -> __m512i {
    let sum = _mm512_add_epi64(x, y);
    mm512_hexl_small_mod_epu64::<2>(sum, q, None, None)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
pub unsafe fn mm512_hexl_small_sub_mod_epi64(x: __m512i, y: __m512i, q: __m512i) -> __m512i {
    let diff = _mm512_sub_epi64(x, y);
    let sign_bits = _mm512_movepi64_mask(diff);
    _mm512_mask_add_epi64(diff, sign_bits, diff, q)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
pub unsafe fn mm512_hexl_cmp_epu64_mask(a: __m512i, b: __m512i, cmp: CmpInt) -> __mmask8 {
    match cmp {
        CmpInt::Eq => _mm512_cmp_epu64_mask(a, b, 0),
        CmpInt::Lt => _mm512_cmp_epu64_mask(a, b, 1),
        CmpInt::Le => _mm512_cmp_epu64_mask(a, b, 2),
        CmpInt::False => _mm512_cmp_epu64_mask(a, b, 3),
        CmpInt::Ne => _mm512_cmp_epu64_mask(a, b, 4),
        CmpInt::Nlt => _mm512_cmp_epu64_mask(a, b, 5),
        CmpInt::Nle => _mm512_cmp_epu64_mask(a, b, 6),
        CmpInt::True => _mm512_cmp_epu64_mask(a, b, 7),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
pub unsafe fn mm512_hexl_cmp_epi64(
    a: __m512i,
    b: __m512i,
    cmp: CmpInt,
    match_value: u64,
) -> __m512i {
    let mask = mm512_hexl_cmp_epu64_mask(a, b, cmp);
    _mm512_maskz_broadcastq_epi64(mask, _mm_set1_epi64x(match_value as i64))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
pub unsafe fn mm512_hexl_cmpge_epu64(a: __m512i, b: __m512i, match_value: u64) -> __m512i {
    mm512_hexl_cmp_epi64(a, b, CmpInt::Nlt, match_value)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
pub unsafe fn mm512_hexl_cmplt_epu64(a: __m512i, b: __m512i, match_value: u64) -> __m512i {
    mm512_hexl_cmp_epi64(a, b, CmpInt::Lt, match_value)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
pub unsafe fn mm512_hexl_cmple_epu64(a: __m512i, b: __m512i, match_value: u64) -> __m512i {
    mm512_hexl_cmp_epi64(a, b, CmpInt::Le, match_value)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
pub unsafe fn mm512_hexl_barrett_reduce64<const BITSHIFT: i32, const OUTPUT_MOD_FACTOR: i32>(
    mut x: __m512i,
    q: __m512i,
    q_barr_64: __m512i,
    q_barr_52: __m512i,
    prod_right_shift: u64,
    v_neg_mod: __m512i,
) -> __m512i {
    if BITSHIFT == 52 {
        let two_pow_fiftytwo = _mm512_set1_epi64(2251799813685248i64);
        let mask = mm512_hexl_cmp_epu64_mask(x, two_pow_fiftytwo, CmpInt::Nlt);
        if mask != 0 {
            let x_hi = _mm512_srli_epi64(x, 52);
            let x_lo = clear_top_bits64::<52>(x);
            let v_shift = _mm512_set1_epi64(prod_right_shift as i64);
            let v_shift_hi = _mm512_set1_epi64((52u64 - prod_right_shift) as i64);
            let c1_lo = _mm512_srlv_epi64(x_lo, v_shift);
            let c1_hi = _mm512_sllv_epi64(x_hi, v_shift_hi);
            let c1 = _mm512_or_epi64(c1_lo, c1_hi);
            let q_hat = mm512_hexl_mulhi_epi::<52>(c1, q_barr_64);
            x = mm512_hexl_mullo_add_lo_epi::<52>(x_lo, q_hat, v_neg_mod);
        } else {
            let rnd1_hi = mm512_hexl_mulhi_epi::<52>(x, q_barr_52);
            let tmp1_times_mod = mm512_hexl_mullo_epi::<52>(rnd1_hi, q);
            x = _mm512_sub_epi64(x, tmp1_times_mod);
        }
    }

    if BITSHIFT == 64 {
        let rnd1_hi = mm512_hexl_mulhi_epi::<64>(x, q_barr_64);
        let tmp1_times_mod = mm512_hexl_mullo_epi::<64>(rnd1_hi, q);
        x = _mm512_sub_epi64(x, tmp1_times_mod);
    }

    if OUTPUT_MOD_FACTOR == 1 {
        x = mm512_hexl_small_mod_epu64::<2>(x, q, None, None);
    }

    x
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
pub unsafe fn mm512_hexl_shrdi_epi64_runtime(x: __m512i, y: __m512i, bit_shift: u32) -> __m512i {
    let v_shift = _mm512_set1_epi64(bit_shift as i64);
    let v_shift_hi = _mm512_set1_epi64((64 - bit_shift) as i64);
    let c_lo = _mm512_srlv_epi64(x, v_shift);
    let c_hi = _mm512_sllv_epi64(y, v_shift_hi);
    _mm512_add_epi64(c_lo, c_hi)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vbmi2")]
#[inline(always)]
unsafe fn mm512_hexl_shrdi_epi64_vbmi2<const BITSHIFT: i32>(x: __m512i, y: __m512i) -> __m512i {
    _mm512_shrdi_epi64::<BITSHIFT>(x, y)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
pub unsafe fn mm512_hexl_shrdi_epi64<const BITSHIFT: i32>(x: __m512i, y: __m512i) -> __m512i {
    if *HAS_AVX512VBMI2 {
        mm512_hexl_shrdi_epi64_vbmi2::<BITSHIFT>(x, y)
    } else {
        mm512_hexl_shrdi_epi64_runtime(x, y, BITSHIFT as u32)
    }
}
