use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::{LazyLock, Mutex};

use crate::util::{
    barrett_reduce_128, divide_u128_u64_lo, log2_u64, msb, multiply_u64_full, multiply_u64_hi,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct MultiplyFactor {
    operand: u64,
    barrett_factor: u64,
}

impl MultiplyFactor {
    pub fn new(operand: u64, bit_shift: u64, modulus: u64) -> Self {
        debug_assert!(operand <= modulus, "operand must be <= modulus");
        debug_assert!(
            bit_shift == 32 || bit_shift == 52 || bit_shift == 64,
            "Unsupported bit shift"
        );
        let op_hi = operand >> (64 - bit_shift);
        let op_lo = if bit_shift == 64 {
            0
        } else {
            operand << bit_shift
        };
        let barrett_factor = divide_u128_u64_lo(op_hi, op_lo, modulus);
        Self {
            operand,
            barrett_factor,
        }
    }

    pub fn barrett_factor(&self) -> u64 {
        self.barrett_factor
    }

    pub fn operand(&self) -> u64 {
        self.operand
    }
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

#[inline(always)]
pub fn log2(x: u64) -> u64 {
    msb(x)
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

pub fn inverse_mod(input: u64, modulus: u64) -> u64 {
    let mut a = input % modulus;
    if a == 0 {
        panic!("input does not have inverse");
    }

    if modulus == 1 {
        return 0;
    }

    let m0 = modulus as i64;
    let mut modulus = modulus;
    let mut y: i64 = 0;
    let mut x: i64 = 1;

    while a > 1 {
        let q = (a / modulus) as i64;
        let t = modulus as i64;
        modulus = a % modulus;
        a = t as u64;

        let t = y;
        y = x - q * y;
        x = t;
    }

    if x < 0 {
        x += m0;
    }

    x as u64
}

pub fn multiply_mod(x: u64, y: u64, modulus: u64) -> u64 {
    debug_assert!(modulus != 0, "modulus == 0");
    debug_assert!(x < modulus, "x must be < modulus");
    debug_assert!(y < modulus, "y must be < modulus");
    let (prod_hi, prod_lo) = multiply_u64_full(x, y);
    barrett_reduce_128(prod_hi, prod_lo, modulus)
}

pub fn multiply_mod_precon(x: u64, y: u64, y_precon: u64, modulus: u64) -> u64 {
    let mut q = multiply_u64_hi::<64>(x, y_precon);
    q = x.wrapping_mul(y).wrapping_sub(q.wrapping_mul(modulus));
    if q >= modulus {
        q - modulus
    } else {
        q
    }
}

pub fn add_uint_mod(x: u64, y: u64, modulus: u64) -> u64 {
    debug_assert!(x < modulus, "x must be < modulus");
    debug_assert!(y < modulus, "y must be < modulus");
    let sum = x + y;
    if sum >= modulus {
        sum - modulus
    } else {
        sum
    }
}

pub fn sub_uint_mod(x: u64, y: u64, modulus: u64) -> u64 {
    debug_assert!(x < modulus, "x must be < modulus");
    debug_assert!(y < modulus, "y must be < modulus");
    let diff = x + modulus - y;
    if diff >= modulus {
        diff - modulus
    } else {
        diff
    }
}

pub fn pow_mod(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    base %= modulus;
    let mut result = 1u64;
    while exp > 0 {
        if exp & 1 == 1 {
            result = multiply_mod(result, base, modulus);
        }
        base = multiply_mod(base, base, modulus);
        exp >>= 1;
    }
    result
}

pub fn is_primitive_root(root: u64, degree: u64, modulus: u64) -> bool {
    if root == 0 {
        return false;
    }
    if !is_power_of_two(degree) {
        return false;
    }
    pow_mod(root, degree / 2, modulus) == modulus - 1
}

pub fn multiply_mod_lazy<const BITSHIFT: usize>(
    x: u64,
    y_operand: u64,
    y_barrett_factor: u64,
    modulus: u64,
) -> u64 {
    debug_assert!(y_operand < modulus, "y_operand must be < modulus");
    debug_assert!(
        modulus <= maximum_value(BITSHIFT as u64),
        "modulus exceeds bound"
    );
    debug_assert!(
        x <= maximum_value(BITSHIFT as u64),
        "operand exceeds bound"
    );

    let q = multiply_u64_hi::<BITSHIFT>(x, y_barrett_factor);
    y_operand.wrapping_mul(x).wrapping_sub(q.wrapping_mul(modulus))
}

pub fn multiply_mod_lazy_simple<const BITSHIFT: usize>(x: u64, y: u64, modulus: u64) -> u64 {
    debug_assert!(BITSHIFT == 64 || BITSHIFT == 52, "Unsupported BitShift");
    debug_assert!(
        x <= maximum_value(BITSHIFT as u64),
        "operand exceeds bound"
    );
    debug_assert!(y < modulus, "y must be < modulus");
    debug_assert!(
        modulus <= maximum_value(BITSHIFT as u64),
        "modulus exceeds bound"
    );

    let y_barrett = MultiplyFactor::new(y, BITSHIFT as u64, modulus).barrett_factor();
    multiply_mod_lazy::<BITSHIFT>(x, y, y_barrett, modulus)
}

pub fn barrett_reduce64<const OUTPUT_MOD_FACTOR: usize>(
    input: u64,
    modulus: u64,
    q_barr: u64,
) -> u64 {
    debug_assert!(modulus != 0, "modulus == 0");
    let q = multiply_u64_hi::<64>(input, q_barr);
    let q_times_input = input.wrapping_sub(q.wrapping_mul(modulus));
    if OUTPUT_MOD_FACTOR == 2 {
        q_times_input
    } else if q_times_input >= modulus {
        q_times_input - modulus
    } else {
        q_times_input
    }
}

static RNG: LazyLock<Mutex<StdRng>> = LazyLock::new(|| {
    let mut seed_rng = rand::rng();
    let rng = StdRng::from_rng(&mut seed_rng);
    Mutex::new(rng)
});

pub fn generate_primitive_root(degree: u64, modulus: u64) -> u64 {
    let size_entire_group = modulus - 1;
    let size_quotient_group = size_entire_group / degree;

    for _ in 0..200 {
        let mut rng = RNG.lock().expect("rng lock");
        let mut root = rng.random_range(0..modulus);
        root = pow_mod(root, size_quotient_group, modulus);
        if is_primitive_root(root, degree, modulus) {
            return root;
        }
    }
    panic!("no primitive root found for degree {} modulus {}", degree, modulus);
}

pub fn minimal_primitive_root(degree: u64, modulus: u64) -> u64 {
    if !is_power_of_two(degree) {
        panic!("degree not power of 2");
    }

    let root = generate_primitive_root(degree, modulus);
    let generator_sq = multiply_mod(root, root, modulus);
    let mut current_generator = root;
    let mut min_root = root;

    for _ in 0..degree {
        if current_generator < min_root {
            min_root = current_generator;
        }
        current_generator = multiply_mod(current_generator, generator_sq, modulus);
    }

    min_root
}

pub fn is_prime(n: u64) -> bool {
    let bases: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    for &a in &bases {
        if n == a {
            return true;
        }
        if n % a == 0 {
            return false;
        }
    }

    let mut r = 63u64;
    while r > 0 {
        let two_pow_r = 1u64 << r;
        if (n - 1) % two_pow_r == 0 {
            break;
        }
        r -= 1;
    }
    if r == 0 {
        return false;
    }
    let d = (n - 1) / (1u64 << r);

    for &a in &bases {
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        let mut prime = false;
        for _ in 1..r {
            x = pow_mod(x, 2, n);
            if x == n - 1 {
                prime = true;
                break;
            }
        }
        if !prime {
            return false;
        }
    }
    true
}

pub fn generate_primes(
    num_primes: usize,
    bit_size: usize,
    prefer_small_primes: bool,
    ntt_size: usize,
) -> Vec<u64> {
    if num_primes == 0 {
        panic!("num_primes == 0");
    }
    if !is_power_of_two(ntt_size as u64) {
        panic!("ntt_size is not power of two");
    }
    if log2_u64(ntt_size as u64) >= bit_size as u64 {
        panic!("log2(ntt_size) should be less than bit_size");
    }

    let prime_lower_bound = (1i64 << bit_size) + 1;
    let prime_upper_bound = (1i64 << (bit_size + 1)) - 1;

    let mut prime_candidate = if prefer_small_primes {
        prime_lower_bound
    } else {
        prime_upper_bound - (prime_upper_bound % (2 * ntt_size as i64)) + 1
    };

    let prime_candidate_step =
        (if prefer_small_primes { 1 } else { -1 }) * 2 * ntt_size as i64;

    let continue_condition = |candidate: i64| {
        if prefer_small_primes {
            candidate < prime_upper_bound
        } else {
            candidate > prime_lower_bound
        }
    };

    let mut primes = Vec::new();
    while continue_condition(prime_candidate) && primes.len() < num_primes {
        if is_prime(prime_candidate as u64) {
            primes.push(prime_candidate as u64);
        }
        prime_candidate += prime_candidate_step;
    }

    if primes.len() != num_primes {
        panic!("failed to generate enough primes");
    }
    primes
}

pub fn reduce_mod<const INPUT_MOD_FACTOR: u64>(
    mut x: u64,
    modulus: u64,
    twice_modulus: Option<&u64>,
    four_times_modulus: Option<&u64>,
) -> u64 {
    debug_assert!(
        INPUT_MOD_FACTOR == 1
            || INPUT_MOD_FACTOR == 2
            || INPUT_MOD_FACTOR == 4
            || INPUT_MOD_FACTOR == 8,
        "InputModFactor should be 1, 2, 4, or 8"
    );
    if INPUT_MOD_FACTOR == 1 {
        return x;
    }
    if INPUT_MOD_FACTOR == 2 {
        if x >= modulus {
            x -= modulus;
        }
        return x;
    }
    if INPUT_MOD_FACTOR == 4 {
        let twice = *twice_modulus.expect("twice_modulus should not be None");
        if x >= twice {
            x -= twice;
        }
        if x >= modulus {
            x -= modulus;
        }
        return x;
    }
    if INPUT_MOD_FACTOR == 8 {
        let twice = *twice_modulus.expect("twice_modulus should not be None");
        let four = *four_times_modulus.expect("four_times_modulus should not be None");
        if x >= four {
            x -= four;
        }
        if x >= twice {
            x -= twice;
        }
        if x >= modulus {
            x -= modulus;
        }
        return x;
    }
    panic!("Invalid InputModFactor");
}

pub fn add_uint64(operand1: u64, operand2: u64) -> (u64, u8) {
    let sum = operand1.wrapping_add(operand2);
    let carry = if sum < operand1 { 1 } else { 0 };
    (sum, carry)
}

pub fn hensel_lemma_2adic_root(r: u32, q: u64) -> u64 {
    let mut a_prev = 1u64;
    let mut c = 2u64;
    let mut mod_mask = 3u64;

    for _k in 2..=r {
        let mut f = 0u64;
        let mut t = 0u64;
        let mut a = 0u64;
        loop {
            a = a_prev + c * t;
            t += 1;
            f = q * a + 1u64;
            if (f & mod_mask) == 0 {
                break;
            }
        }
        mod_mask = mod_mask * 2 + 1u64;
        c *= 2;
        a_prev = a;
    }

    a_prev
}
