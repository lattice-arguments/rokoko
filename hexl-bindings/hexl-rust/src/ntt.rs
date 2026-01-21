use crate::cpu_features::{HAS_AVX512DQ, HAS_AVX512IFMA};
use crate::number_theory::{
    add_uint_mod, inverse_mod, is_power_of_two, is_prime, log2, multiply_mod,
    multiply_mod_lazy, reduce_mod, sub_uint_mod,
    MultiplyFactor,
};
use crate::aligned_vec::AlignedVecU64;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use crate::avx512_util::{
    mm512_hexl_mulhi_approx_epi, mm512_hexl_mulhi_epi, mm512_hexl_mullo_add_lo_epi,
    mm512_hexl_mullo_epi, mm512_hexl_small_add_mod_epi64, mm512_hexl_small_mod_epu64,
};
#[cfg(target_arch = "x86_64")]
use crate::ntt_avx512_util::{
    load_fwd_interleaved_t1, load_fwd_interleaved_t2, load_fwd_interleaved_t4,
    load_inv_interleaved_t1, load_inv_interleaved_t2, load_inv_interleaved_t4, load_w_op_t2,
    load_w_op_t4, write_fwd_interleaved_t1, write_inv_interleaved_t4,
};

#[derive(Clone)]
pub struct Ntt {
    degree: u64,
    modulus: u64,
    root_of_unity: u64,
    degree_bits: u64,
    root_of_unity_powers: Vec<u64>,
    precon32_root_of_unity_powers: Vec<u64>,
    precon64_root_of_unity_powers: Vec<u64>,
    avx512_root_of_unity_powers: AlignedVecU64,
    avx512_precon32_root_of_unity_powers: AlignedVecU64,
    avx512_precon52_root_of_unity_powers: AlignedVecU64,
    avx512_precon64_root_of_unity_powers: AlignedVecU64,
    precon32_inv_root_of_unity_powers: Vec<u64>,
    precon52_inv_root_of_unity_powers: Vec<u64>,
    precon64_inv_root_of_unity_powers: Vec<u64>,
    inv_root_of_unity_powers: Vec<u64>,
}

impl Ntt {
    pub const DEFAULT_SHIFT_BITS: usize = 64;
    pub const IFMA_SHIFT_BITS: usize = 52;
    pub const MAX_FWD_32_MODULUS: u64 = 1u64 << (32 - 2);
    pub const MAX_INV_32_MODULUS: u64 = 1u64 << (32 - 2);
    pub const MAX_FWD_IFMA_MODULUS: u64 = 1u64 << (Self::IFMA_SHIFT_BITS - 2);
    pub const MAX_INV_IFMA_MODULUS: u64 = 1u64 << (Self::IFMA_SHIFT_BITS - 2);

    pub fn new(degree: u64, modulus: u64) -> Self {
        let root_of_unity = crate::number_theory::minimal_primitive_root(2 * degree, modulus);
        Self::with_root(degree, modulus, root_of_unity)
    }

    pub fn with_root(degree: u64, modulus: u64, root_of_unity: u64) -> Self {
        debug_assert!(Self::check_arguments(degree, modulus));
        debug_assert!(crate::number_theory::is_primitive_root(
            root_of_unity,
            2 * degree,
            modulus
        ));

        let degree_bits = log2(degree);
        let mut ntt = Self {
            degree,
            modulus,
            root_of_unity,
            degree_bits,
            root_of_unity_powers: Vec::new(),
            precon32_root_of_unity_powers: Vec::new(),
            precon64_root_of_unity_powers: Vec::new(),
            avx512_root_of_unity_powers: AlignedVecU64::default(),
            avx512_precon32_root_of_unity_powers: AlignedVecU64::default(),
            avx512_precon52_root_of_unity_powers: AlignedVecU64::default(),
            avx512_precon64_root_of_unity_powers: AlignedVecU64::default(),
            precon32_inv_root_of_unity_powers: Vec::new(),
            precon52_inv_root_of_unity_powers: Vec::new(),
            precon64_inv_root_of_unity_powers: Vec::new(),
            inv_root_of_unity_powers: Vec::new(),
        };
        ntt.compute_root_of_unity_powers();
        ntt
    }

    pub fn root_of_unity_powers(&self) -> &Vec<u64> {
        &self.root_of_unity_powers
    }

    pub fn inv_root_of_unity_powers(&self) -> &Vec<u64> {
        &self.inv_root_of_unity_powers
    }

    pub fn precon32_root_of_unity_powers(&self) -> &Vec<u64> {
        &self.precon32_root_of_unity_powers
    }

    pub fn precon64_root_of_unity_powers(&self) -> &Vec<u64> {
        &self.precon64_root_of_unity_powers
    }

    pub fn avx512_root_of_unity_powers(&self) -> &AlignedVecU64 {
        &self.avx512_root_of_unity_powers
    }

    pub fn avx512_precon32_root_of_unity_powers(&self) -> &AlignedVecU64 {
        &self.avx512_precon32_root_of_unity_powers
    }

    pub fn avx512_precon52_root_of_unity_powers(&self) -> &AlignedVecU64 {
        &self.avx512_precon52_root_of_unity_powers
    }

    pub fn avx512_precon64_root_of_unity_powers(&self) -> &AlignedVecU64 {
        &self.avx512_precon64_root_of_unity_powers
    }

    pub fn precon32_inv_root_of_unity_powers(&self) -> &Vec<u64> {
        &self.precon32_inv_root_of_unity_powers
    }

    pub fn precon52_inv_root_of_unity_powers(&self) -> &Vec<u64> {
        &self.precon52_inv_root_of_unity_powers
    }

    pub fn precon64_inv_root_of_unity_powers(&self) -> &Vec<u64> {
        &self.precon64_inv_root_of_unity_powers
    }

    pub fn compute_forward(
        &self,
        result: &mut [u64],
        operand: &[u64],
        input_mod_factor: u64,
        output_mod_factor: u64,
    ) {
        debug_assert!(result.len() >= self.degree as usize);
        debug_assert!(operand.len() >= self.degree as usize);
        debug_assert!(input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4);
        debug_assert!(output_mod_factor == 1 || output_mod_factor == 4);

        #[cfg(target_arch = "x86_64")]
        unsafe {
            if *HAS_AVX512IFMA
                && self.modulus < Self::MAX_FWD_IFMA_MODULUS
                && self.degree >= 16
            {
                forward_transform_to_bit_reverse_avx512::<{ Self::IFMA_SHIFT_BITS as i32 }>(
                    result.as_mut_ptr(),
                    operand.as_ptr(),
                    self.degree,
                    self.modulus,
                    self.avx512_root_of_unity_powers.as_ptr(),
                    self.avx512_precon52_root_of_unity_powers.as_ptr(),
                    input_mod_factor,
                    output_mod_factor,
                    0,
                    0,
                );
                return;
            }

            if *HAS_AVX512DQ && self.degree >= 16 {
                if self.modulus < Self::MAX_FWD_32_MODULUS {
                    forward_transform_to_bit_reverse_avx512::<32>(
                        result.as_mut_ptr(),
                        operand.as_ptr(),
                        self.degree,
                        self.modulus,
                        self.avx512_root_of_unity_powers.as_ptr(),
                        self.avx512_precon32_root_of_unity_powers.as_ptr(),
                        input_mod_factor,
                        output_mod_factor,
                        0,
                        0,
                    );
                } else {
                    forward_transform_to_bit_reverse_avx512::<{ Self::DEFAULT_SHIFT_BITS as i32 }>(
                        result.as_mut_ptr(),
                        operand.as_ptr(),
                        self.degree,
                        self.modulus,
                        self.avx512_root_of_unity_powers.as_ptr(),
                        self.avx512_precon64_root_of_unity_powers.as_ptr(),
                        input_mod_factor,
                        output_mod_factor,
                        0,
                        0,
                    );
                }
                return;
            }
        }

        forward_transform_to_bit_reverse_radix2(
            result,
            operand,
            self.degree,
            self.modulus,
            &self.root_of_unity_powers,
            &self.precon64_root_of_unity_powers,
            input_mod_factor,
            output_mod_factor,
        );
    }

    pub fn compute_inverse(
        &self,
        result: &mut [u64],
        operand: &[u64],
        input_mod_factor: u64,
        output_mod_factor: u64,
    ) {
        debug_assert!(result.len() >= self.degree as usize);
        debug_assert!(operand.len() >= self.degree as usize);
        debug_assert!(input_mod_factor == 1 || input_mod_factor == 2);
        debug_assert!(output_mod_factor == 1 || output_mod_factor == 2);

        #[cfg(target_arch = "x86_64")]
        unsafe {
            if *HAS_AVX512IFMA
                && self.modulus < Self::MAX_INV_IFMA_MODULUS
                && self.degree >= 16
            {
                inverse_transform_from_bit_reverse_avx512::<{ Self::IFMA_SHIFT_BITS as i32 }>(
                    result.as_mut_ptr(),
                    operand.as_ptr(),
                    self.degree,
                    self.modulus,
                    self.inv_root_of_unity_powers.as_ptr(),
                    self.precon52_inv_root_of_unity_powers.as_ptr(),
                    input_mod_factor,
                    output_mod_factor,
                    0,
                    0,
                );
                return;
            }

            if *HAS_AVX512DQ && self.degree >= 16 {
                if self.modulus < Self::MAX_INV_32_MODULUS {
                    inverse_transform_from_bit_reverse_avx512::<32>(
                        result.as_mut_ptr(),
                        operand.as_ptr(),
                        self.degree,
                        self.modulus,
                        self.inv_root_of_unity_powers.as_ptr(),
                        self.precon32_inv_root_of_unity_powers.as_ptr(),
                        input_mod_factor,
                        output_mod_factor,
                        0,
                        0,
                    );
                } else {
                    inverse_transform_from_bit_reverse_avx512::<{ Self::DEFAULT_SHIFT_BITS as i32 }>(
                        result.as_mut_ptr(),
                        operand.as_ptr(),
                        self.degree,
                        self.modulus,
                        self.inv_root_of_unity_powers.as_ptr(),
                        self.precon64_inv_root_of_unity_powers.as_ptr(),
                        input_mod_factor,
                        output_mod_factor,
                        0,
                        0,
                    );
                }
                return;
            }
        }

        inverse_transform_from_bit_reverse_radix2(
            result,
            operand,
            self.degree,
            self.modulus,
            &self.inv_root_of_unity_powers,
            &self.precon64_inv_root_of_unity_powers,
            input_mod_factor,
            output_mod_factor,
        );
    }

    pub fn check_arguments(degree: u64, modulus: u64) -> bool {
        debug_assert!(is_power_of_two(degree), "degree is not power of two");
        debug_assert!(
            degree <= (1u64 << Self::max_degree_bits()),
            "degree too large"
        );
        debug_assert!(
            modulus <= (1u64 << Self::max_modulus_bits()),
            "modulus too large"
        );
        debug_assert!(modulus % (2 * degree) == 1, "modulus mod 2n != 1");
        debug_assert!(is_prime(modulus), "modulus is not prime");
        true
    }

    pub fn max_degree_bits() -> usize {
        20
    }

    pub fn max_modulus_bits() -> usize {
        62
    }

    pub fn max_fwd_modulus(bit_shift: i32) -> u64 {
        if bit_shift == 32 {
            Self::MAX_FWD_32_MODULUS
        } else if bit_shift == 52 {
            Self::MAX_FWD_IFMA_MODULUS
        } else if bit_shift == 64 {
            1u64 << Self::max_modulus_bits()
        } else {
            0
        }
    }

    pub fn max_inv_modulus(bit_shift: i32) -> u64 {
        if bit_shift == 32 {
            Self::MAX_INV_32_MODULUS
        } else if bit_shift == 52 {
            Self::MAX_INV_IFMA_MODULUS
        } else if bit_shift == 64 {
            1u64 << Self::max_modulus_bits()
        } else {
            0
        }
    }

    fn compute_root_of_unity_powers(&mut self) {
        let n = self.degree as usize;
        let mut root_of_unity_powers = vec![0u64; n];
        let mut inv_root_of_unity_powers = vec![0u64; n];

        root_of_unity_powers[0] = 1;
        inv_root_of_unity_powers[0] = inverse_mod(1, self.modulus);
        let mut prev_idx = 0u64;

        for i in 1..self.degree {
            let idx = crate::number_theory::reverse_bits(i, self.degree_bits);
            root_of_unity_powers[idx as usize] =
                multiply_mod(root_of_unity_powers[prev_idx as usize], self.root_of_unity, self.modulus);
            inv_root_of_unity_powers[idx as usize] =
                inverse_mod(root_of_unity_powers[idx as usize], self.modulus);
            prev_idx = idx;
        }

        self.root_of_unity_powers = root_of_unity_powers.clone();
        let mut avx512_root_of_unity_powers = root_of_unity_powers.clone();

        let mut w2_roots = Vec::with_capacity((self.degree / 2) as usize);
        for i in (self.degree / 4)..(self.degree / 2) {
            let val = self.root_of_unity_powers[i as usize];
            w2_roots.push(val);
            w2_roots.push(val);
        }
        avx512_root_of_unity_powers.splice(
            (self.degree / 4) as usize..(self.degree / 2) as usize,
            w2_roots,
        );

        let mut w4_roots = Vec::with_capacity((self.degree / 2) as usize);
        for i in (self.degree / 8)..(self.degree / 4) {
            let val = self.root_of_unity_powers[i as usize];
            w4_roots.push(val);
            w4_roots.push(val);
            w4_roots.push(val);
            w4_roots.push(val);
        }
        avx512_root_of_unity_powers.splice(
            (self.degree / 8) as usize..(self.degree / 4) as usize,
            w4_roots,
        );
        self.avx512_root_of_unity_powers =
            AlignedVecU64::from_vec(avx512_root_of_unity_powers);

        let compute_barrett_vector = |values: &[u64], bit_shift: u64| -> Vec<u64> {
            let mut barrett_vector = Vec::with_capacity(values.len());
            for &value in values {
                let mf = MultiplyFactor::new(value, bit_shift, self.modulus);
                barrett_vector.push(mf.barrett_factor());
            }
            barrett_vector
        };

        self.precon32_root_of_unity_powers = compute_barrett_vector(&root_of_unity_powers, 32);
        self.precon64_root_of_unity_powers = compute_barrett_vector(&root_of_unity_powers, 64);

        if *HAS_AVX512IFMA {
            let precon = compute_barrett_vector(self.avx512_root_of_unity_powers.as_slice(), 52);
            self.avx512_precon52_root_of_unity_powers = AlignedVecU64::from_vec(precon);
        }

        if *HAS_AVX512DQ {
            let precon32 = compute_barrett_vector(self.avx512_root_of_unity_powers.as_slice(), 32);
            let precon64 = compute_barrett_vector(self.avx512_root_of_unity_powers.as_slice(), 64);
            self.avx512_precon32_root_of_unity_powers = AlignedVecU64::from_vec(precon32);
            self.avx512_precon64_root_of_unity_powers = AlignedVecU64::from_vec(precon64);
        }

        let mut temp = vec![0u64; n];
        temp[0] = inv_root_of_unity_powers[0];
        let mut idx = 1usize;
        let mut m = self.degree >> 1;
        while m > 0 {
            for i in 0..m {
                temp[idx] = inv_root_of_unity_powers[(m + i) as usize];
                idx += 1;
            }
            m >>= 1;
        }
        self.inv_root_of_unity_powers = temp;

        self.precon32_inv_root_of_unity_powers =
            compute_barrett_vector(&self.inv_root_of_unity_powers, 32);
        if *HAS_AVX512IFMA {
            self.precon52_inv_root_of_unity_powers =
                compute_barrett_vector(&self.inv_root_of_unity_powers, 52);
        }
        self.precon64_inv_root_of_unity_powers =
            compute_barrett_vector(&self.inv_root_of_unity_powers, 64);
    }
}

fn fwd_butterfly_radix2(
    x_op: u64,
    y_op: u64,
    w: u64,
    w_precon: u64,
    modulus: u64,
    twice_modulus: u64,
) -> (u64, u64) {
    let tx = reduce_mod::<2>(x_op, twice_modulus, None, None);
    let t = multiply_mod_lazy::<64>(y_op, w, w_precon, modulus);
    let x_r = tx + t;
    let y_r = tx + twice_modulus - t;
    (x_r, y_r)
}

fn inv_butterfly_radix2(
    x_op: u64,
    y_op: u64,
    w: u64,
    w_precon: u64,
    modulus: u64,
    twice_modulus: u64,
) -> (u64, u64) {
    let tx = x_op + y_op;
    let mut y_r = x_op + twice_modulus - y_op;
    let x_r = reduce_mod::<2>(tx, twice_modulus, None, None);
    y_r = multiply_mod_lazy::<64>(y_r, w, w_precon, modulus);
    (x_r, y_r)
}

fn forward_transform_to_bit_reverse_radix2(
    result: &mut [u64],
    operand: &[u64],
    n: u64,
    modulus: u64,
    root_of_unity_powers: &[u64],
    precon_root_of_unity_powers: &[u64],
    input_mod_factor: u64,
    output_mod_factor: u64,
) {
    debug_assert!(Ntt::check_arguments(n, modulus));
    debug_assert!(input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4);
    debug_assert!(output_mod_factor == 1 || output_mod_factor == 4);

    let twice_modulus = modulus << 1;
    let mut t = n >> 1;

    {
        let w = root_of_unity_powers[1];
        let w_precon = precon_root_of_unity_powers[1];
        let mut x_r_idx = 0usize;
        let mut y_r_idx = t as usize;
        let mut x_op_idx = 0usize;
        let mut y_op_idx = t as usize;

        match t {
            8 => {
                for _ in 0..8 {
                    let (x_new, y_new) = fwd_butterfly_radix2(
                        operand[x_op_idx],
                        operand[y_op_idx],
                        w,
                        w_precon,
                        modulus,
                        twice_modulus,
                    );
                    result[x_r_idx] = x_new;
                    result[y_r_idx] = y_new;
                    x_r_idx += 1;
                    y_r_idx += 1;
                    x_op_idx += 1;
                    y_op_idx += 1;
                }
            }
            4 => {
                for _ in 0..4 {
                    let (x_new, y_new) = fwd_butterfly_radix2(
                        operand[x_op_idx],
                        operand[y_op_idx],
                        w,
                        w_precon,
                        modulus,
                        twice_modulus,
                    );
                    result[x_r_idx] = x_new;
                    result[y_r_idx] = y_new;
                    x_r_idx += 1;
                    y_r_idx += 1;
                    x_op_idx += 1;
                    y_op_idx += 1;
                }
            }
            2 => {
                for _ in 0..2 {
                    let (x_new, y_new) = fwd_butterfly_radix2(
                        operand[x_op_idx],
                        operand[y_op_idx],
                        w,
                        w_precon,
                        modulus,
                        twice_modulus,
                    );
                    result[x_r_idx] = x_new;
                    result[y_r_idx] = y_new;
                    x_r_idx += 1;
                    y_r_idx += 1;
                    x_op_idx += 1;
                    y_op_idx += 1;
                }
            }
            1 => {
                let (x_new, y_new) = fwd_butterfly_radix2(
                    operand[x_op_idx],
                    operand[y_op_idx],
                    w,
                    w_precon,
                    modulus,
                    twice_modulus,
                );
                result[x_r_idx] = x_new;
                result[y_r_idx] = y_new;
            }
            _ => {
                let mut j = 0usize;
                while j < t as usize {
                    for _ in 0..8 {
                        let (x_new, y_new) = fwd_butterfly_radix2(
                            operand[x_op_idx],
                            operand[y_op_idx],
                            w,
                            w_precon,
                            modulus,
                            twice_modulus,
                        );
                        result[x_r_idx] = x_new;
                        result[y_r_idx] = y_new;
                        x_r_idx += 1;
                        y_r_idx += 1;
                        x_op_idx += 1;
                        y_op_idx += 1;
                    }
                    j += 8;
                }
            }
        }
        t >>= 1;
    }

    let mut m = 2u64;
    while m < n {
        let mut offset = 0usize;
        match t {
            8 => {
                for i in 0..m {
                    if i != 0 {
                        offset += (t << 1) as usize;
                    }
                    let w = root_of_unity_powers[(m + i) as usize];
                    let w_precon = precon_root_of_unity_powers[(m + i) as usize];

                    let mut x_r_idx = offset;
                    let mut y_r_idx = offset + t as usize;
                    for _ in 0..8 {
                        let x_op = result[x_r_idx];
                        let y_op = result[y_r_idx];
                        let (x_new, y_new) = fwd_butterfly_radix2(
                            x_op,
                            y_op,
                            w,
                            w_precon,
                            modulus,
                            twice_modulus,
                        );
                        result[x_r_idx] = x_new;
                        result[y_r_idx] = y_new;
                        x_r_idx += 1;
                        y_r_idx += 1;
                    }
                }
            }
            4 => {
                for i in 0..m {
                    if i != 0 {
                        offset += (t << 1) as usize;
                    }
                    let w = root_of_unity_powers[(m + i) as usize];
                    let w_precon = precon_root_of_unity_powers[(m + i) as usize];

                    let mut x_r_idx = offset;
                    let mut y_r_idx = offset + t as usize;
                    for _ in 0..4 {
                        let x_op = result[x_r_idx];
                        let y_op = result[y_r_idx];
                        let (x_new, y_new) = fwd_butterfly_radix2(
                            x_op,
                            y_op,
                            w,
                            w_precon,
                            modulus,
                            twice_modulus,
                        );
                        result[x_r_idx] = x_new;
                        result[y_r_idx] = y_new;
                        x_r_idx += 1;
                        y_r_idx += 1;
                    }
                }
            }
            2 => {
                for i in 0..m {
                    if i != 0 {
                        offset += (t << 1) as usize;
                    }
                    let w = root_of_unity_powers[(m + i) as usize];
                    let w_precon = precon_root_of_unity_powers[(m + i) as usize];

                    let mut x_r_idx = offset;
                    let mut y_r_idx = offset + t as usize;
                    for _ in 0..2 {
                        let x_op = result[x_r_idx];
                        let y_op = result[y_r_idx];
                        let (x_new, y_new) = fwd_butterfly_radix2(
                            x_op,
                            y_op,
                            w,
                            w_precon,
                            modulus,
                            twice_modulus,
                        );
                        result[x_r_idx] = x_new;
                        result[y_r_idx] = y_new;
                        x_r_idx += 1;
                        y_r_idx += 1;
                    }
                }
            }
            1 => {
                for i in 0..m {
                    if i != 0 {
                        offset += (t << 1) as usize;
                    }
                    let w = root_of_unity_powers[(m + i) as usize];
                    let w_precon = precon_root_of_unity_powers[(m + i) as usize];

                    let x_r_idx = offset;
                    let y_r_idx = offset + t as usize;
                    let x_op = result[x_r_idx];
                    let y_op = result[y_r_idx];
                    let (x_new, y_new) =
                        fwd_butterfly_radix2(x_op, y_op, w, w_precon, modulus, twice_modulus);
                    result[x_r_idx] = x_new;
                    result[y_r_idx] = y_new;
                }
            }
            _ => {
                for i in 0..m {
                    if i != 0 {
                        offset += (t << 1) as usize;
                    }
                    let w = root_of_unity_powers[(m + i) as usize];
                    let w_precon = precon_root_of_unity_powers[(m + i) as usize];

                    let mut x_r_idx = offset;
                    let mut y_r_idx = offset + t as usize;
                    let mut j = 0usize;
                    while j < t as usize {
                        for _ in 0..8 {
                            let x_op = result[x_r_idx];
                            let y_op = result[y_r_idx];
                            let (x_new, y_new) = fwd_butterfly_radix2(
                                x_op,
                                y_op,
                                w,
                                w_precon,
                                modulus,
                                twice_modulus,
                            );
                            result[x_r_idx] = x_new;
                            result[y_r_idx] = y_new;
                            x_r_idx += 1;
                            y_r_idx += 1;
                        }
                        j += 8;
                    }
                }
            }
        }
        t >>= 1;
        m <<= 1;
    }

    if output_mod_factor == 1 {
        for value in result.iter_mut().take(n as usize) {
            *value = reduce_mod::<4>(*value, modulus, Some(&twice_modulus), None);
        }
    }
}

fn reference_forward_transform_to_bit_reverse(
    operand: &mut [u64],
    n: u64,
    modulus: u64,
    root_of_unity_powers: &[u64],
) {
    debug_assert!(Ntt::check_arguments(n, modulus));

    let mut t = n >> 1;
    let mut m = 1u64;
    while m < n {
        let mut offset = 0usize;
        for i in 0..m {
            let offset2 = offset + t as usize;
            let w = root_of_unity_powers[(m + i) as usize];
            let mut x_idx = offset;
            let mut y_idx = offset + t as usize;
            while x_idx < offset2 {
                let tx = operand[x_idx];
                let w_x_y = multiply_mod(operand[y_idx], w, modulus);
                operand[x_idx] = add_uint_mod(tx, w_x_y, modulus);
                operand[y_idx] = sub_uint_mod(tx, w_x_y, modulus);
                x_idx += 1;
                y_idx += 1;
            }
            offset += (t << 1) as usize;
        }
        t >>= 1;
        m <<= 1;
    }
}

fn reference_inverse_transform_from_bit_reverse(
    operand: &mut [u64],
    n: u64,
    modulus: u64,
    inv_root_of_unity_powers: &[u64],
) {
    debug_assert!(Ntt::check_arguments(n, modulus));

    let mut t = 1u64;
    let mut root_index = 1usize;
    let mut m = n >> 1;
    while m >= 1 {
        let mut offset = 0usize;
        for _ in 0..m {
            let w = inv_root_of_unity_powers[root_index];
            root_index += 1;
            let mut x_idx = offset;
            let mut y_idx = offset + t as usize;
            for _ in 0..t {
                let x_op = operand[x_idx];
                let y_op = operand[y_idx];
                operand[x_idx] = add_uint_mod(x_op, y_op, modulus);
                operand[y_idx] = multiply_mod(w, sub_uint_mod(x_op, y_op, modulus), modulus);
                x_idx += 1;
                y_idx += 1;
            }
            offset += (t << 1) as usize;
        }
        t <<= 1;
        if m == 1 {
            break;
        }
        m >>= 1;
    }

    let inv_n = inverse_mod(n, modulus);
    for value in operand.iter_mut().take(n as usize) {
        *value = multiply_mod(*value, inv_n, modulus);
    }
}

fn inverse_transform_from_bit_reverse_radix2(
    result: &mut [u64],
    operand: &[u64],
    n: u64,
    modulus: u64,
    inv_root_of_unity_powers: &[u64],
    precon_inv_root_of_unity_powers: &[u64],
    input_mod_factor: u64,
    output_mod_factor: u64,
) {
    debug_assert!(Ntt::check_arguments(n, modulus));
    debug_assert!(input_mod_factor == 1 || input_mod_factor == 2);
    debug_assert!(output_mod_factor == 1 || output_mod_factor == 2);

    let twice_modulus = modulus << 1;
    let n_div_2 = n >> 1;
    let mut t = 1u64;
    let mut root_index = 1usize;
    let mut m = n_div_2;

    while m > 1 {
        let mut offset = 0usize;
        match t {
            1 => {
                for i in 0..m {
                    if i != 0 {
                        offset += (t << 1) as usize;
                    }
                    let w = inv_root_of_unity_powers[root_index];
                    let w_precon = precon_inv_root_of_unity_powers[root_index];
                    root_index += 1;

                    let x_r_idx = offset;
                    let y_r_idx = offset + t as usize;
                    let x_op_idx = offset;
                    let y_op_idx = offset + t as usize;
                    let (x_new, y_new) = inv_butterfly_radix2(
                        operand[x_op_idx],
                        operand[y_op_idx],
                        w,
                        w_precon,
                        modulus,
                        twice_modulus,
                    );
                    result[x_r_idx] = x_new;
                    result[y_r_idx] = y_new;
                }
            }
            2 => {
                for i in 0..m {
                    if i != 0 {
                        offset += (t << 1) as usize;
                    }
                    let w = inv_root_of_unity_powers[root_index];
                    let w_precon = precon_inv_root_of_unity_powers[root_index];
                    root_index += 1;

                    let mut x_r_idx = offset;
                    let mut y_r_idx = offset + t as usize;
                    for _ in 0..2 {
                        let x_op = result[x_r_idx];
                        let y_op = result[y_r_idx];
                        let (x_new, y_new) =
                            inv_butterfly_radix2(x_op, y_op, w, w_precon, modulus, twice_modulus);
                        result[x_r_idx] = x_new;
                        result[y_r_idx] = y_new;
                        x_r_idx += 1;
                        y_r_idx += 1;
                    }
                }
            }
            4 => {
                for i in 0..m {
                    if i != 0 {
                        offset += (t << 1) as usize;
                    }
                    let w = inv_root_of_unity_powers[root_index];
                    let w_precon = precon_inv_root_of_unity_powers[root_index];
                    root_index += 1;

                    let mut x_r_idx = offset;
                    let mut y_r_idx = offset + t as usize;
                    for _ in 0..4 {
                        let x_op = result[x_r_idx];
                        let y_op = result[y_r_idx];
                        let (x_new, y_new) =
                            inv_butterfly_radix2(x_op, y_op, w, w_precon, modulus, twice_modulus);
                        result[x_r_idx] = x_new;
                        result[y_r_idx] = y_new;
                        x_r_idx += 1;
                        y_r_idx += 1;
                    }
                }
            }
            8 => {
                for i in 0..m {
                    if i != 0 {
                        offset += (t << 1) as usize;
                    }
                    let w = inv_root_of_unity_powers[root_index];
                    let w_precon = precon_inv_root_of_unity_powers[root_index];
                    root_index += 1;

                    let mut x_r_idx = offset;
                    let mut y_r_idx = offset + t as usize;
                    for _ in 0..8 {
                        let x_op = result[x_r_idx];
                        let y_op = result[y_r_idx];
                        let (x_new, y_new) =
                            inv_butterfly_radix2(x_op, y_op, w, w_precon, modulus, twice_modulus);
                        result[x_r_idx] = x_new;
                        result[y_r_idx] = y_new;
                        x_r_idx += 1;
                        y_r_idx += 1;
                    }
                }
            }
            _ => {
                for i in 0..m {
                    if i != 0 {
                        offset += (t << 1) as usize;
                    }
                    let w = inv_root_of_unity_powers[root_index];
                    let w_precon = precon_inv_root_of_unity_powers[root_index];
                    root_index += 1;

                    let mut x_r_idx = offset;
                    let mut y_r_idx = offset + t as usize;
                    let mut j = 0usize;
                    while j < t as usize {
                        for _ in 0..8 {
                            let x_op = result[x_r_idx];
                            let y_op = result[y_r_idx];
                            let (x_new, y_new) = inv_butterfly_radix2(
                                x_op,
                                y_op,
                                w,
                                w_precon,
                                modulus,
                                twice_modulus,
                            );
                            result[x_r_idx] = x_new;
                            result[y_r_idx] = y_new;
                            x_r_idx += 1;
                            y_r_idx += 1;
                        }
                        j += 8;
                    }
                }
            }
        }
        t <<= 1;
        m >>= 1;
    }

    if !std::ptr::eq(result.as_ptr(), operand.as_ptr()) && n == 2 {
        result[..n as usize].copy_from_slice(&operand[..n as usize]);
    }

    let w = inv_root_of_unity_powers[(n - 1) as usize];
    let inv_n = inverse_mod(n, modulus);
    let inv_n_precon = MultiplyFactor::new(inv_n, 64, modulus).barrett_factor();
    let inv_n_w = multiply_mod(inv_n, w, modulus);
    let inv_n_w_precon = MultiplyFactor::new(inv_n_w, 64, modulus).barrett_factor();

    let x_base = 0usize;
    let y_base = n_div_2 as usize;
    for j in 0..n_div_2 as usize {
        let tx = add_uint_mod(result[x_base + j], result[y_base + j], twice_modulus);
        let ty = result[x_base + j] + twice_modulus - result[y_base + j];
        result[x_base + j] = multiply_mod_lazy::<64>(tx, inv_n, inv_n_precon, modulus);
        result[y_base + j] = multiply_mod_lazy::<64>(ty, inv_n_w, inv_n_w_precon, modulus);
    }

    if output_mod_factor == 1 {
        for value in result.iter_mut().take(n as usize) {
            *value = reduce_mod::<2>(*value, modulus, None, None);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
unsafe fn fwd_butterfly_avx512<const BITSHIFT: i32, const INPUT_LESS_THAN_MOD: bool>(
    x: &mut __m512i,
    y: &mut __m512i,
    w: __m512i,
    w_precon: __m512i,
    neg_modulus: __m512i,
    twice_modulus: __m512i,
) {
    if !INPUT_LESS_THAN_MOD {
        *x = mm512_hexl_small_mod_epu64::<2>(*x, twice_modulus, None, None);
    }

    let t = if BITSHIFT == 32 {
        let mut q = mm512_hexl_mullo_epi::<64>(w_precon, *y);
        q = _mm512_srli_epi64(q, 32);
        let w_y = mm512_hexl_mullo_epi::<64>(w, *y);
        mm512_hexl_mullo_add_lo_epi::<64>(w_y, q, neg_modulus)
    } else if BITSHIFT == 52 {
        let q = mm512_hexl_mulhi_epi::<BITSHIFT>(w_precon, *y);
        let w_y = mm512_hexl_mullo_epi::<BITSHIFT>(w, *y);
        mm512_hexl_mullo_add_lo_epi::<BITSHIFT>(w_y, q, neg_modulus)
    } else if BITSHIFT == 64 {
        let q = mm512_hexl_mulhi_approx_epi::<BITSHIFT>(w_precon, *y);
        let w_y = mm512_hexl_mullo_epi::<BITSHIFT>(w, *y);
        let mut t = mm512_hexl_mullo_add_lo_epi::<BITSHIFT>(w_y, q, neg_modulus);
        t = mm512_hexl_small_mod_epu64::<2>(t, twice_modulus, None, None);
        t
    } else {
        core::hint::unreachable_unchecked()
    };

    let twice_mod_minus_t = _mm512_sub_epi64(twice_modulus, t);
    *y = _mm512_add_epi64(*x, twice_mod_minus_t);
    *x = _mm512_add_epi64(*x, t);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
unsafe fn fwd_t1<const BITSHIFT: i32>(
    operand: *mut u64,
    v_neg_modulus: __m512i,
    v_twice_mod: __m512i,
    m: u64,
    w: *const u64,
    w_precon: *const u64,
) {
    let mut w_ptr = w as *const __m512i;
    let mut w_precon_ptr = w_precon as *const __m512i;
    let mut j1 = 0usize;

    macro_rules! fwd_t1_step {
        () => {{
            let x = operand.add(j1);
            let v_x_pt = x as *mut __m512i;
            let mut v_x = _mm512_setzero_si512();
            let mut v_y = _mm512_setzero_si512();
            load_fwd_interleaved_t1(x as *const u64, &mut v_x, &mut v_y);
            let v_w = _mm512_loadu_si512(w_ptr);
            let v_w_precon = _mm512_loadu_si512(w_precon_ptr);
            w_ptr = w_ptr.add(1);
            w_precon_ptr = w_precon_ptr.add(1);

            fwd_butterfly_avx512::<BITSHIFT, false>(
                &mut v_x,
                &mut v_y,
                v_w,
                v_w_precon,
                v_neg_modulus,
                v_twice_mod,
            );
            write_fwd_interleaved_t1(v_x, v_y, v_x_pt);

            j1 += 16;
        }};
    }

    let mut i = m / 8;
    while i >= 8 {
        fwd_t1_step!();
        fwd_t1_step!();
        fwd_t1_step!();
        fwd_t1_step!();
        fwd_t1_step!();
        fwd_t1_step!();
        fwd_t1_step!();
        fwd_t1_step!();
        i -= 8;
    }
    while i > 0 {
        fwd_t1_step!();
        i -= 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
unsafe fn fwd_t2<const BITSHIFT: i32>(
    operand: *mut u64,
    v_neg_modulus: __m512i,
    v_twice_mod: __m512i,
    m: u64,
    w: *const u64,
    w_precon: *const u64,
) {
    let mut w_ptr = w as *const __m512i;
    let mut w_precon_ptr = w_precon as *const __m512i;
    let mut j1 = 0usize;

    macro_rules! fwd_t2_step {
        () => {{
            let x = operand.add(j1);
            let v_x_pt = x as *mut __m512i;
            let mut v_x = _mm512_setzero_si512();
            let mut v_y = _mm512_setzero_si512();
            load_fwd_interleaved_t2(x as *const u64, &mut v_x, &mut v_y);
            let v_w = _mm512_loadu_si512(w_ptr);
            let v_w_precon = _mm512_loadu_si512(w_precon_ptr);
            w_ptr = w_ptr.add(1);
            w_precon_ptr = w_precon_ptr.add(1);

            fwd_butterfly_avx512::<BITSHIFT, false>(
                &mut v_x,
                &mut v_y,
                v_w,
                v_w_precon,
                v_neg_modulus,
                v_twice_mod,
            );

            _mm512_storeu_si512(v_x_pt, v_x);
            _mm512_storeu_si512(v_x_pt.add(1), v_y);

            j1 += 16;
        }};
    }

    let mut i = m / 4;
    while i >= 4 {
        fwd_t2_step!();
        fwd_t2_step!();
        fwd_t2_step!();
        fwd_t2_step!();
        i -= 4;
    }
    while i > 0 {
        fwd_t2_step!();
        i -= 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
unsafe fn fwd_t4<const BITSHIFT: i32>(
    operand: *mut u64,
    v_neg_modulus: __m512i,
    v_twice_mod: __m512i,
    m: u64,
    w: *const u64,
    w_precon: *const u64,
) {
    let mut j1 = 0usize;
    let mut w_ptr = w as *const __m512i;
    let mut w_precon_ptr = w_precon as *const __m512i;

    macro_rules! fwd_t4_step {
        () => {{
            let x = operand.add(j1);
            let v_x_pt = x as *mut __m512i;
            let mut v_x = _mm512_setzero_si512();
            let mut v_y = _mm512_setzero_si512();
            load_fwd_interleaved_t4(x as *const u64, &mut v_x, &mut v_y);

            let v_w = _mm512_loadu_si512(w_ptr);
            let v_w_precon = _mm512_loadu_si512(w_precon_ptr);
            w_ptr = w_ptr.add(1);
            w_precon_ptr = w_precon_ptr.add(1);

            fwd_butterfly_avx512::<BITSHIFT, false>(
                &mut v_x,
                &mut v_y,
                v_w,
                v_w_precon,
                v_neg_modulus,
                v_twice_mod,
            );

            _mm512_storeu_si512(v_x_pt, v_x);
            _mm512_storeu_si512(v_x_pt.add(1), v_y);

            j1 += 16;
        }};
    }

    let mut i = m / 2;
    while i >= 4 {
        fwd_t4_step!();
        fwd_t4_step!();
        fwd_t4_step!();
        fwd_t4_step!();
        i -= 4;
    }
    while i > 0 {
        fwd_t4_step!();
        i -= 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
unsafe fn fwd_t8<const BITSHIFT: i32, const INPUT_LESS_THAN_MOD: bool>(
    result: *mut u64,
    operand: *const u64,
    v_neg_modulus: __m512i,
    v_twice_mod: __m512i,
    t: u64,
    m: u64,
    w: *const u64,
    w_precon: *const u64,
) {
    let mut j1 = 0usize;
    let mut w_ptr = w;
    let mut w_precon_ptr = w_precon;

    macro_rules! fwd_t8_step {
        () => {{
            let x_op = operand.add(j1);
            let y_op = x_op.add(t as usize);
            let x_r = result.add(j1);
            let y_r = x_r.add(t as usize);

            let v_w = _mm512_set1_epi64(*w_ptr as i64);
            let v_w_precon = _mm512_set1_epi64(*w_precon_ptr as i64);
            w_ptr = w_ptr.add(1);
            w_precon_ptr = w_precon_ptr.add(1);

            let mut x_op_pt = x_op as *const __m512i;
            let mut y_op_pt = y_op as *const __m512i;
            let mut x_r_pt = x_r as *mut __m512i;
            let mut y_r_pt = y_r as *mut __m512i;

            let mut j = t / 8;
            while j > 0 {
                let mut v_x = _mm512_loadu_si512(x_op_pt);
                let mut v_y = _mm512_loadu_si512(y_op_pt);

                fwd_butterfly_avx512::<BITSHIFT, INPUT_LESS_THAN_MOD>(
                    &mut v_x,
                    &mut v_y,
                    v_w,
                    v_w_precon,
                    v_neg_modulus,
                    v_twice_mod,
                );

                _mm512_storeu_si512(x_r_pt, v_x);
                _mm512_storeu_si512(y_r_pt, v_y);

                x_op_pt = x_op_pt.add(1);
                y_op_pt = y_op_pt.add(1);
                x_r_pt = x_r_pt.add(1);
                y_r_pt = y_r_pt.add(1);
                j -= 1;
            }
            j1 += (t << 1) as usize;
        }};
    }

    let mut i = 0u64;
    while i + 4 <= m {
        fwd_t8_step!();
        fwd_t8_step!();
        fwd_t8_step!();
        fwd_t8_step!();
        i += 4;
    }
    while i < m {
        fwd_t8_step!();
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
unsafe fn forward_transform_to_bit_reverse_avx512<const BITSHIFT: i32>(
    result: *mut u64,
    operand: *const u64,
    n: u64,
    modulus: u64,
    root_of_unity_powers: *const u64,
    precon_root_of_unity_powers: *const u64,
    input_mod_factor: u64,
    output_mod_factor: u64,
    recursion_depth: u64,
    recursion_half: u64,
) {
    debug_assert!(Ntt::check_arguments(n, modulus));
    debug_assert!(modulus < Ntt::max_fwd_modulus(BITSHIFT));
    debug_assert!(n >= 16);
    debug_assert!(input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4);
    debug_assert!(output_mod_factor == 1 || output_mod_factor == 4);

    let twice_mod = modulus << 1;
    let v_modulus = _mm512_set1_epi64(modulus as i64);
    let v_neg_modulus = _mm512_set1_epi64(-(modulus as i64));
    let v_twice_mod = _mm512_set1_epi64(twice_mod as i64);

    const BASE_NTT_SIZE: u64 = 1024;

    if n <= BASE_NTT_SIZE {
        let mut t = n >> 1;
        let mut m = 1u64;
        let mut w_idx = (m << recursion_depth) + (recursion_half * m);

        if result as *const u64 != operand {
            std::ptr::copy_nonoverlapping(operand, result, n as usize);
        }

        if m < (n >> 3) {
            let w = root_of_unity_powers.add(w_idx as usize);
            let w_precon = precon_root_of_unity_powers.add(w_idx as usize);

            if input_mod_factor <= 2 && recursion_depth == 0 {
                fwd_t8::<BITSHIFT, true>(
                    result,
                    result,
                    v_neg_modulus,
                    v_twice_mod,
                    t,
                    m,
                    w,
                    w_precon,
                );
            } else {
                fwd_t8::<BITSHIFT, false>(
                    result,
                    result,
                    v_neg_modulus,
                    v_twice_mod,
                    t,
                    m,
                    w,
                    w_precon,
                );
            }

            t >>= 1;
            m <<= 1;
            w_idx <<= 1;
        }

        while m < (n >> 3) {
            let w = root_of_unity_powers.add(w_idx as usize);
            let w_precon = precon_root_of_unity_powers.add(w_idx as usize);
            fwd_t8::<BITSHIFT, false>(
                result,
                result,
                v_neg_modulus,
                v_twice_mod,
                t,
                m,
                w,
                w_precon,
            );
            t >>= 1;
            w_idx <<= 1;
            m <<= 1;
        }

        let compute_new_w_idx = |idx: u64| -> u64 {
            let n_total = n << recursion_depth;
            if idx <= n_total / 8 {
                idx
            } else if idx <= n_total / 4 {
                (idx - n_total / 8) * 4 + (n_total / 8)
            } else if idx <= n_total / 2 {
                (idx - n_total / 4) * 2 + (5 * n_total / 8)
            } else {
                idx + (5 * n_total / 8)
            }
        };

        let mut new_w_idx = compute_new_w_idx(w_idx);
        let mut w = root_of_unity_powers.add(new_w_idx as usize);
        let mut w_precon = precon_root_of_unity_powers.add(new_w_idx as usize);
        fwd_t4::<BITSHIFT>(result, v_neg_modulus, v_twice_mod, m, w, w_precon);

        m <<= 1;
        w_idx <<= 1;
        new_w_idx = compute_new_w_idx(w_idx);
        w = root_of_unity_powers.add(new_w_idx as usize);
        w_precon = precon_root_of_unity_powers.add(new_w_idx as usize);
        fwd_t2::<BITSHIFT>(result, v_neg_modulus, v_twice_mod, m, w, w_precon);

        m <<= 1;
        w_idx <<= 1;
        new_w_idx = compute_new_w_idx(w_idx);
        w = root_of_unity_powers.add(new_w_idx as usize);
        w_precon = precon_root_of_unity_powers.add(new_w_idx as usize);
        fwd_t1::<BITSHIFT>(result, v_neg_modulus, v_twice_mod, m, w, w_precon);

        if output_mod_factor == 1 {
            let mut v_x_pt = result as *mut __m512i;
            for _ in 0..(n / 8) {
                let mut v_x = _mm512_loadu_si512(v_x_pt);
                v_x = mm512_hexl_small_mod_epu64::<2>(v_x, v_twice_mod, None, None);
                v_x = mm512_hexl_small_mod_epu64::<2>(v_x, v_modulus, None, None);
                _mm512_storeu_si512(v_x_pt, v_x);
                v_x_pt = v_x_pt.add(1);
            }
        }
    } else {
        let t = n >> 1;
        let w_idx = (1u64 << recursion_depth) + recursion_half;
        let w = root_of_unity_powers.add(w_idx as usize);
        let w_precon = precon_root_of_unity_powers.add(w_idx as usize);

        fwd_t8::<BITSHIFT, false>(
            result,
            operand,
            v_neg_modulus,
            v_twice_mod,
            t,
            1,
            w,
            w_precon,
        );

        forward_transform_to_bit_reverse_avx512::<BITSHIFT>(
            result,
            result,
            n / 2,
            modulus,
            root_of_unity_powers,
            precon_root_of_unity_powers,
            input_mod_factor,
            output_mod_factor,
            recursion_depth + 1,
            recursion_half * 2,
        );

        forward_transform_to_bit_reverse_avx512::<BITSHIFT>(
            result.add((n / 2) as usize),
            result.add((n / 2) as usize),
            n / 2,
            modulus,
            root_of_unity_powers,
            precon_root_of_unity_powers,
            input_mod_factor,
            output_mod_factor,
            recursion_depth + 1,
            recursion_half * 2 + 1,
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
unsafe fn inv_butterfly_avx512<const BITSHIFT: i32, const INPUT_LESS_THAN_MOD: bool>(
    x: &mut __m512i,
    y: &mut __m512i,
    w: __m512i,
    w_precon: __m512i,
    neg_modulus: __m512i,
    twice_modulus: __m512i,
) {
    let y_minus_2q = _mm512_sub_epi64(*y, twice_modulus);
    let t = _mm512_sub_epi64(*x, y_minus_2q);

    if INPUT_LESS_THAN_MOD {
        *x = _mm512_add_epi64(*x, *y);
    } else {
        *x = _mm512_add_epi64(*x, y_minus_2q);
        let sign_bits = _mm512_movepi64_mask(*x);
        *x = _mm512_mask_add_epi64(*x, sign_bits, *x, twice_modulus);
    }

    if BITSHIFT == 32 {
        let mut q = mm512_hexl_mullo_epi::<64>(w_precon, t);
        q = _mm512_srli_epi64(q, 32);
        let q_p = mm512_hexl_mullo_epi::<64>(q, neg_modulus);
        *y = mm512_hexl_mullo_add_lo_epi::<64>(q_p, w, t);
    } else if BITSHIFT == 52 {
        let q = mm512_hexl_mulhi_epi::<BITSHIFT>(w_precon, t);
        let q_p = mm512_hexl_mullo_epi::<BITSHIFT>(q, neg_modulus);
        *y = mm512_hexl_mullo_add_lo_epi::<BITSHIFT>(q_p, w, t);
    } else if BITSHIFT == 64 {
        let q = mm512_hexl_mulhi_approx_epi::<BITSHIFT>(w_precon, t);
        let q_p = mm512_hexl_mullo_epi::<BITSHIFT>(q, neg_modulus);
        *y = mm512_hexl_mullo_add_lo_epi::<BITSHIFT>(q_p, w, t);
        *y = mm512_hexl_small_mod_epu64::<2>(*y, twice_modulus, None, None);
    } else {
        core::hint::unreachable_unchecked()
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
unsafe fn inv_t1<const BITSHIFT: i32, const INPUT_LESS_THAN_MOD: bool>(
    operand: *mut u64,
    v_neg_modulus: __m512i,
    v_twice_mod: __m512i,
    m: u64,
    w: *const u64,
    w_precon: *const u64,
) {
    let mut w_ptr = w as *const __m512i;
    let mut w_precon_ptr = w_precon as *const __m512i;
    let mut j1 = 0usize;

    macro_rules! inv_t1_step {
        () => {{
            let x = operand.add(j1);
            let v_x_pt = x as *mut __m512i;
            let mut v_x = _mm512_setzero_si512();
            let mut v_y = _mm512_setzero_si512();
            load_inv_interleaved_t1(x as *const u64, &mut v_x, &mut v_y);
            let v_w = _mm512_loadu_si512(w_ptr);
            let v_w_precon = _mm512_loadu_si512(w_precon_ptr);
            w_ptr = w_ptr.add(1);
            w_precon_ptr = w_precon_ptr.add(1);

            inv_butterfly_avx512::<BITSHIFT, INPUT_LESS_THAN_MOD>(
                &mut v_x,
                &mut v_y,
                v_w,
                v_w_precon,
                v_neg_modulus,
                v_twice_mod,
            );

            _mm512_storeu_si512(v_x_pt, v_x);
            _mm512_storeu_si512(v_x_pt.add(1), v_y);

            j1 += 16;
        }};
    }

    let mut i = m / 8;
    while i >= 8 {
        inv_t1_step!();
        inv_t1_step!();
        inv_t1_step!();
        inv_t1_step!();
        inv_t1_step!();
        inv_t1_step!();
        inv_t1_step!();
        inv_t1_step!();
        i -= 8;
    }
    while i > 0 {
        inv_t1_step!();
        i -= 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
unsafe fn inv_t2<const BITSHIFT: i32>(
    mut x: *mut u64,
    v_neg_modulus: __m512i,
    v_twice_mod: __m512i,
    m: u64,
    mut w: *const u64,
    mut w_precon: *const u64,
) {
    macro_rules! inv_t2_step {
        () => {{
            let v_x_pt = x as *mut __m512i;
            let mut v_x = _mm512_setzero_si512();
            let mut v_y = _mm512_setzero_si512();
            load_inv_interleaved_t2(x as *const u64, &mut v_x, &mut v_y);

            let v_w = load_w_op_t2(w);
            let v_w_precon = load_w_op_t2(w_precon);

            inv_butterfly_avx512::<BITSHIFT, false>(
                &mut v_x,
                &mut v_y,
                v_w,
                v_w_precon,
                v_neg_modulus,
                v_twice_mod,
            );

            _mm512_storeu_si512(v_x_pt, v_x);
            _mm512_storeu_si512(v_x_pt.add(1), v_y);
            x = x.add(16);
            w = w.add(4);
            w_precon = w_precon.add(4);
        }};
    }

    let mut i = m / 4;
    while i >= 4 {
        inv_t2_step!();
        inv_t2_step!();
        inv_t2_step!();
        inv_t2_step!();
        i -= 4;
    }
    while i > 0 {
        inv_t2_step!();
        i -= 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
unsafe fn inv_t4<const BITSHIFT: i32>(
    operand: *mut u64,
    v_neg_modulus: __m512i,
    v_twice_mod: __m512i,
    m: u64,
    mut w: *const u64,
    mut w_precon: *const u64,
) {
    let mut x = operand;
    macro_rules! inv_t4_step {
        () => {{
            let v_x_pt = x as *mut __m512i;
            let mut v_x = _mm512_setzero_si512();
            let mut v_y = _mm512_setzero_si512();
            load_inv_interleaved_t4(x as *const u64, &mut v_x, &mut v_y);

            let v_w = load_w_op_t4(w);
            let v_w_precon = load_w_op_t4(w_precon);

            inv_butterfly_avx512::<BITSHIFT, false>(
                &mut v_x,
                &mut v_y,
                v_w,
                v_w_precon,
                v_neg_modulus,
                v_twice_mod,
            );

            write_inv_interleaved_t4(v_x, v_y, v_x_pt);
            x = x.add(16);
            w = w.add(2);
            w_precon = w_precon.add(2);
        }};
    }

    let mut i = m / 2;
    while i >= 4 {
        inv_t4_step!();
        inv_t4_step!();
        inv_t4_step!();
        inv_t4_step!();
        i -= 4;
    }
    while i > 0 {
        inv_t4_step!();
        i -= 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
unsafe fn inv_t8<const BITSHIFT: i32>(
    operand: *mut u64,
    v_neg_modulus: __m512i,
    v_twice_mod: __m512i,
    t: u64,
    m: u64,
    w: *const u64,
    w_precon: *const u64,
) {
    let mut j1 = 0usize;
    let mut w_ptr = w;
    let mut w_precon_ptr = w_precon;

    macro_rules! inv_t8_step {
        () => {{
            let x = operand.add(j1);
            let y = x.add(t as usize);

            let v_w = _mm512_set1_epi64(*w_ptr as i64);
            let v_w_precon = _mm512_set1_epi64(*w_precon_ptr as i64);
            w_ptr = w_ptr.add(1);
            w_precon_ptr = w_precon_ptr.add(1);

            let mut v_x_pt = x as *mut __m512i;
            let mut v_y_pt = y as *mut __m512i;
            let mut j = t / 8;
            while j > 0 {
                let mut v_x = _mm512_loadu_si512(v_x_pt);
                let mut v_y = _mm512_loadu_si512(v_y_pt);

                inv_butterfly_avx512::<BITSHIFT, false>(
                    &mut v_x,
                    &mut v_y,
                    v_w,
                    v_w_precon,
                    v_neg_modulus,
                    v_twice_mod,
                );

                _mm512_storeu_si512(v_x_pt, v_x);
                _mm512_storeu_si512(v_y_pt, v_y);
                v_x_pt = v_x_pt.add(1);
                v_y_pt = v_y_pt.add(1);
                j -= 1;
            }
            j1 += (t << 1) as usize;
        }};
    }

    let mut i = 0u64;
    while i + 4 <= m {
        inv_t8_step!();
        inv_t8_step!();
        inv_t8_step!();
        inv_t8_step!();
        i += 4;
    }
    while i < m {
        inv_t8_step!();
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline(always)]
unsafe fn inverse_transform_from_bit_reverse_avx512<const BITSHIFT: i32>(
    result: *mut u64,
    operand: *const u64,
    n: u64,
    modulus: u64,
    inv_root_of_unity_powers: *const u64,
    precon_inv_root_of_unity_powers: *const u64,
    input_mod_factor: u64,
    output_mod_factor: u64,
    recursion_depth: u64,
    recursion_half: u64,
) {
    debug_assert!(Ntt::check_arguments(n, modulus));
    debug_assert!(n >= 16);
    debug_assert!(modulus < Ntt::max_inv_modulus(BITSHIFT));
    debug_assert!(input_mod_factor == 1 || input_mod_factor == 2);
    debug_assert!(output_mod_factor == 1 || output_mod_factor == 2);

    let twice_mod = modulus << 1;
    let v_modulus = _mm512_set1_epi64(modulus as i64);
    let v_neg_modulus = _mm512_set1_epi64(-(modulus as i64));
    let v_twice_mod = _mm512_set1_epi64(twice_mod as i64);

    let mut t = 1u64;
    let mut m = n >> 1;
    let mut w_idx = 1 + m * recursion_half;

    const BASE_NTT_SIZE: u64 = 1024;

    if n <= BASE_NTT_SIZE {
        if result as *const u64 != operand {
            std::ptr::copy_nonoverlapping(operand, result, n as usize);
        }

        {
            let w = inv_root_of_unity_powers.add(w_idx as usize);
            let w_precon = precon_inv_root_of_unity_powers.add(w_idx as usize);
            if input_mod_factor == 1 && recursion_depth == 0 {
                inv_t1::<BITSHIFT, true>(result, v_neg_modulus, v_twice_mod, m, w, w_precon);
            } else {
                inv_t1::<BITSHIFT, false>(result, v_neg_modulus, v_twice_mod, m, w, w_precon);
            }

            t <<= 1;
            m >>= 1;
            let mut w_idx_delta = m * ((1u64 << (recursion_depth + 1)) - recursion_half);
            w_idx += w_idx_delta;

            let w = inv_root_of_unity_powers.add(w_idx as usize);
            let w_precon = precon_inv_root_of_unity_powers.add(w_idx as usize);
            inv_t2::<BITSHIFT>(result, v_neg_modulus, v_twice_mod, m, w, w_precon);

            t <<= 1;
            m >>= 1;
            w_idx_delta >>= 1;
            w_idx += w_idx_delta;

            let w = inv_root_of_unity_powers.add(w_idx as usize);
            let w_precon = precon_inv_root_of_unity_powers.add(w_idx as usize);
            inv_t4::<BITSHIFT>(result, v_neg_modulus, v_twice_mod, m, w, w_precon);

            t <<= 1;
            m >>= 1;
            w_idx_delta >>= 1;
            w_idx += w_idx_delta;

            while m > 1 {
                let w = inv_root_of_unity_powers.add(w_idx as usize);
                let w_precon = precon_inv_root_of_unity_powers.add(w_idx as usize);
                inv_t8::<BITSHIFT>(result, v_neg_modulus, v_twice_mod, t, m, w, w_precon);
                t <<= 1;
                m >>= 1;
                w_idx_delta >>= 1;
                w_idx += w_idx_delta;
            }
        }
    } else {
        inverse_transform_from_bit_reverse_avx512::<BITSHIFT>(
            result,
            operand,
            n / 2,
            modulus,
            inv_root_of_unity_powers,
            precon_inv_root_of_unity_powers,
            input_mod_factor,
            output_mod_factor,
            recursion_depth + 1,
            2 * recursion_half,
        );
        inverse_transform_from_bit_reverse_avx512::<BITSHIFT>(
            result.add((n / 2) as usize),
            operand.add((n / 2) as usize),
            n / 2,
            modulus,
            inv_root_of_unity_powers,
            precon_inv_root_of_unity_powers,
            input_mod_factor,
            output_mod_factor,
            recursion_depth + 1,
            2 * recursion_half + 1,
        );

        let mut w_idx_delta = m * ((1u64 << (recursion_depth + 1)) - recursion_half);
        while m > 2 {
            t <<= 1;
            m >>= 1;
            w_idx_delta >>= 1;
            w_idx += w_idx_delta;
        }
        if m == 2 {
            let w = inv_root_of_unity_powers.add(w_idx as usize);
            let w_precon = precon_inv_root_of_unity_powers.add(w_idx as usize);
            inv_t8::<BITSHIFT>(result, v_neg_modulus, v_twice_mod, t, m, w, w_precon);
            t <<= 1;
            m >>= 1;
            w_idx_delta >>= 1;
            w_idx += w_idx_delta;
        }
    }

    if recursion_depth == 0 {
        let w = *inv_root_of_unity_powers.add(w_idx as usize);
        let mf_inv_n = MultiplyFactor::new(inverse_mod(n, modulus), BITSHIFT as u64, modulus);
        let inv_n = mf_inv_n.operand();
        let inv_n_prime = mf_inv_n.barrett_factor();

        let mf_inv_n_w = MultiplyFactor::new(multiply_mod(inv_n, w, modulus), BITSHIFT as u64, modulus);
        let inv_n_w = mf_inv_n_w.operand();
        let inv_n_w_prime = mf_inv_n_w.barrett_factor();

        let x = result;
        let y = result.add((n >> 1) as usize);

        let v_inv_n = _mm512_set1_epi64(inv_n as i64);
        let v_inv_n_prime = _mm512_set1_epi64(inv_n_prime as i64);
        let v_inv_n_w = _mm512_set1_epi64(inv_n_w as i64);
        let v_inv_n_w_prime = _mm512_set1_epi64(inv_n_w_prime as i64);

        let mut v_x_pt = x as *mut __m512i;
        let mut v_y_pt = y as *mut __m512i;

        let mut j = n / 16;
        while j > 0 {
            let v_x = _mm512_loadu_si512(v_x_pt);
            let v_y = _mm512_loadu_si512(v_y_pt);

            let y_minus_2q = _mm512_sub_epi64(v_y, v_twice_mod);
            let x_plus_y_mod2q = mm512_hexl_small_add_mod_epi64(v_x, v_y, v_twice_mod);
            let t_val = _mm512_sub_epi64(v_x, y_minus_2q);

            let mut v_x_out;
            let mut v_y_out;

            if BITSHIFT == 32 {
                let mut q1 = mm512_hexl_mullo_epi::<64>(v_inv_n_prime, x_plus_y_mod2q);
                q1 = _mm512_srli_epi64(q1, 32);
                let inv_n_tx = mm512_hexl_mullo_epi::<64>(v_inv_n, x_plus_y_mod2q);
                v_x_out = mm512_hexl_mullo_add_lo_epi::<64>(inv_n_tx, q1, v_neg_modulus);

                let mut q2 = mm512_hexl_mullo_epi::<64>(v_inv_n_w_prime, t_val);
                q2 = _mm512_srli_epi64(q2, 32);
                let inv_n_w_t = mm512_hexl_mullo_epi::<64>(v_inv_n_w, t_val);
                v_y_out = mm512_hexl_mullo_add_lo_epi::<64>(inv_n_w_t, q2, v_neg_modulus);
            } else {
                let q1 = mm512_hexl_mulhi_epi::<BITSHIFT>(v_inv_n_prime, x_plus_y_mod2q);
                let inv_n_tx = mm512_hexl_mullo_epi::<BITSHIFT>(v_inv_n, x_plus_y_mod2q);
                v_x_out = mm512_hexl_mullo_add_lo_epi::<BITSHIFT>(inv_n_tx, q1, v_neg_modulus);

                let q2 = mm512_hexl_mulhi_epi::<BITSHIFT>(v_inv_n_w_prime, t_val);
                let inv_n_w_t = mm512_hexl_mullo_epi::<BITSHIFT>(v_inv_n_w, t_val);
                v_y_out = mm512_hexl_mullo_add_lo_epi::<BITSHIFT>(inv_n_w_t, q2, v_neg_modulus);
            }

            if output_mod_factor == 1 {
                v_x_out = mm512_hexl_small_mod_epu64::<2>(v_x_out, v_modulus, None, None);
                v_y_out = mm512_hexl_small_mod_epu64::<2>(v_y_out, v_modulus, None, None);
            }

            _mm512_storeu_si512(v_x_pt, v_x_out);
            _mm512_storeu_si512(v_y_pt, v_y_out);

            v_x_pt = v_x_pt.add(1);
            v_y_pt = v_y_pt.add(1);
            j -= 1;
        }
    }
}
