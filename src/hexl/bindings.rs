#[link(name = "hexl_wrapper")]
extern "C" {
    pub fn multiply_mod(a: u64, b: u64, modulus: u64) -> u64;

    pub fn power_mod(a: u64, b: u64, modulus: u64) -> u64;

    pub fn add_mod(a: u64, b: u64, modulus: u64) -> u64;

    pub fn sub_mod(a: u64, b: u64, modulus: u64) -> u64;

    pub fn eltwise_mult_mod(
        result: *mut u64,
        operand1: *const u64,
        operand2: *const u64,
        n: u64,
        modulus: u64,
    );

    pub fn get_roots(n: u64, modulus: u64) -> *const u64;

    pub fn inv_mod(a: u64, modulus: u64) -> u64;

    pub fn get_inv_roots(n: u64, modulus: u64) -> *const u64;

    pub fn eltwise_add_mod(
        result: *mut u64,
        operand1: *const u64,
        operand2: *const u64,
        n: u64,
        modulus: u64,
    );

    pub fn eltwise_sub_mod(
        result: *mut u64,
        operand1: *const u64,
        operand2: *const u64,
        n: u64,
        modulus: u64,
    );

    pub fn eltwise_mod_inverse(result: *mut u64, operand1: *const u64, n: u64, modulus: u64);

    pub fn multiply_poly(
        result: *mut u64,
        operand1: *const u64,
        operand2: *const u64,
        n: u64,
        modulus: u64,
    );

    pub fn eltwise_reduce_mod(result: *mut u64, operand: *const u64, n: u64, modulus: u64);

    pub fn polynomial_multiply_cyclotomic_mod(
        result: *mut u64,
        operand1: *const u64,
        operand2: *const u64,
        phi: u64,
        mod_q: u64,
    );
    pub fn ntt_forward_in_place(operand: *mut u64, n: usize, modulus: u64);

    pub fn ntt_inverse_in_place(operand: *mut u64, n: usize, modulus: u64);

    pub fn eltwise_fma_mod(
        result: *mut u64,
        operand1: *const u64,
        operand2: u64,
        operand3: *const u64,
        n: u64,
        modulus: u64,
    );
}
