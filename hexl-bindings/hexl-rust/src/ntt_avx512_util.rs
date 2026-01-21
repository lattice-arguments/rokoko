#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx2")]
#[inline(always)]
pub unsafe fn load_fwd_interleaved_t1(arg: *const u64, out1: &mut __m512i, out2: &mut __m512i) {
    let arg_512 = arg as *const __m512i;
    let v1 = _mm512_loadu_si512(arg_512);
    let v2 = _mm512_loadu_si512(arg_512.add(1));

    let perm_idx = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
    let v1_perm = _mm512_permutexvar_epi64(perm_idx, v1);
    let v2_perm = _mm512_permutexvar_epi64(perm_idx, v2);

    *out1 = _mm512_mask_blend_epi64(0xaa_u8, v1, v2_perm);
    *out2 = _mm512_mask_blend_epi64(0xaa_u8, v1_perm, v2);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx2")]
#[inline(always)]
pub unsafe fn load_inv_interleaved_t1(arg: *const u64, out1: &mut __m512i, out2: &mut __m512i) {
    let vperm_hi_idx = _mm512_set_epi64(6, 4, 2, 0, 7, 5, 3, 1);
    let vperm_lo_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
    let vperm2_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);

    let arg_512 = arg as *const __m512i;
    let v_7to0 = _mm512_loadu_si512(arg_512);
    let v_15to8 = _mm512_loadu_si512(arg_512.add(1));

    let perm_lo = _mm512_permutexvar_epi64(vperm_lo_idx, v_7to0);
    let perm_hi = _mm512_permutexvar_epi64(vperm_hi_idx, v_15to8);

    *out1 = _mm512_mask_blend_epi64(0x0f_u8, perm_hi, perm_lo);
    *out2 = _mm512_mask_blend_epi64(0xf0_u8, perm_hi, perm_lo);
    *out2 = _mm512_permutexvar_epi64(vperm2_idx, *out2);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx2")]
#[inline(always)]
pub unsafe fn load_fwd_interleaved_t2(arg: *const u64, out1: &mut __m512i, out2: &mut __m512i) {
    let arg_512 = arg as *const __m512i;
    let v1 = _mm512_loadu_si512(arg_512);
    let v2 = _mm512_loadu_si512(arg_512.add(1));

    let v1_perm_idx = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
    let v1_perm = _mm512_permutexvar_epi64(v1_perm_idx, v1);
    let v2_perm = _mm512_permutexvar_epi64(v1_perm_idx, v2);

    *out1 = _mm512_mask_blend_epi64(0xcc_u8, v1, v2_perm);
    *out2 = _mm512_mask_blend_epi64(0xcc_u8, v1_perm, v2);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx2")]
#[inline(always)]
pub unsafe fn load_inv_interleaved_t2(arg: *const u64, out1: &mut __m512i, out2: &mut __m512i) {
    let arg_512 = arg as *const __m512i;
    let v1 = _mm512_loadu_si512(arg_512);
    let v2 = _mm512_loadu_si512(arg_512.add(1));

    let v1_perm_idx = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
    let v1_perm = _mm512_permutexvar_epi64(v1_perm_idx, v1);
    let v2_perm = _mm512_permutexvar_epi64(v1_perm_idx, v2);

    *out1 = _mm512_mask_blend_epi64(0xaa_u8, v1, v2_perm);
    *out2 = _mm512_mask_blend_epi64(0xaa_u8, v1_perm, v2);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx2")]
#[inline(always)]
pub unsafe fn load_fwd_interleaved_t4(arg: *const u64, out1: &mut __m512i, out2: &mut __m512i) {
    let arg_512 = arg as *const __m512i;
    let vperm2_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
    let v_7to0 = _mm512_loadu_si512(arg_512);
    let v_15to8 = _mm512_loadu_si512(arg_512.add(1));
    let perm_hi = _mm512_permutexvar_epi64(vperm2_idx, v_15to8);
    *out1 = _mm512_mask_blend_epi64(0x0f_u8, perm_hi, v_7to0);
    *out2 = _mm512_mask_blend_epi64(0xf0_u8, perm_hi, v_7to0);
    *out2 = _mm512_permutexvar_epi64(vperm2_idx, *out2);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx2")]
#[inline(always)]
pub unsafe fn load_inv_interleaved_t4(arg: *const u64, out1: &mut __m512i, out2: &mut __m512i) {
    let arg_512 = arg as *const __m512i;
    let v1 = _mm512_loadu_si512(arg_512);
    let v2 = _mm512_loadu_si512(arg_512.add(1));
    let perm_idx = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);

    let v1_perm = _mm512_permutexvar_epi64(perm_idx, v1);
    let v2_perm = _mm512_permutexvar_epi64(perm_idx, v2);

    *out1 = _mm512_mask_blend_epi64(0xcc_u8, v1, v2_perm);
    *out2 = _mm512_mask_blend_epi64(0xcc_u8, v1_perm, v2);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx2")]
#[inline(always)]
pub unsafe fn write_fwd_interleaved_t1(arg1: __m512i, arg2: __m512i, out: *mut __m512i) {
    let vperm2_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
    let v_x_out_idx = _mm512_set_epi64(7, 3, 6, 2, 5, 1, 4, 0);
    let v_y_out_idx = _mm512_set_epi64(3, 7, 2, 6, 1, 5, 0, 4);

    let arg2 = _mm512_permutexvar_epi64(vperm2_idx, arg2);
    let perm_lo = _mm512_mask_blend_epi64(0x0f_u8, arg1, arg2);
    let perm_hi = _mm512_mask_blend_epi64(0xf0_u8, arg1, arg2);

    let arg1 = _mm512_permutexvar_epi64(v_x_out_idx, perm_hi);
    let arg2 = _mm512_permutexvar_epi64(v_y_out_idx, perm_lo);

    _mm512_storeu_si512(out, arg1);
    _mm512_storeu_si512(out.add(1), arg2);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx2")]
#[inline(always)]
pub unsafe fn write_inv_interleaved_t4(arg1: __m512i, arg2: __m512i, out: *mut __m512i) {
    let x0 = _mm512_extracti64x4_epi64(arg1, 0);
    let x1 = _mm512_extracti64x4_epi64(arg1, 1);
    let y0 = _mm512_extracti64x4_epi64(arg2, 0);
    let y1 = _mm512_extracti64x4_epi64(arg2, 1);
    let out_256 = out as *mut __m256i;
    _mm256_storeu_si256(out_256, x0);
    _mm256_storeu_si256(out_256.add(1), y0);
    _mm256_storeu_si256(out_256.add(2), x1);
    _mm256_storeu_si256(out_256.add(3), y1);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx2")]
#[inline(always)]
pub unsafe fn load_w_op_t2(arg: *const u64) -> __m512i {
    let vperm_w_idx = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);
    let v_w_256 = _mm256_loadu_si256(arg as *const __m256i);
    let mut v_w = _mm512_broadcast_i64x4(v_w_256);
    v_w = _mm512_permutexvar_epi64(vperm_w_idx, v_w);
    v_w
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx2")]
#[inline(always)]
pub unsafe fn load_w_op_t4(arg: *const u64) -> __m512i {
    let vperm_w_idx = _mm512_set_epi64(1, 1, 1, 1, 0, 0, 0, 0);
    let v_w_128 = _mm_loadu_si128(arg as *const __m128i);
    let mut v_w = _mm512_broadcast_i64x2(v_w_128);
    v_w = _mm512_permutexvar_epi64(vperm_w_idx, v_w);
    v_w
}
