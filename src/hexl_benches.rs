use std::time::Instant;

pub fn run_hexl_benches() {
    use cowboys_and_aliens::common::config::{DEGREE, HALF_DEGREE, MOD_Q};
    use cowboys_and_aliens::hexl::bindings::{
        eltwise_add_mod, eltwise_fma_mod, eltwise_mult_mod, eltwise_reduce_mod,
        ntt_forward_in_place, ntt_inverse_in_place,
    };

    let wrapper = unsafe {
        HexlWrapper::load().unwrap_or_else(|err| panic!("Failed to load HEXL wrapper: {err}"))
    };
    verify_hexl_against_cpp(&wrapper);

    let n = HALF_DEGREE;
    let mut a = vec![0u64; n];
    let mut b = vec![0u64; n];
    let mut c = vec![0u64; n];
    let mut out = vec![0u64; n];
    let mut reduce_input = vec![0u64; n];
    for i in 0..n {
        a[i] = (i as u64 * 3) % MOD_Q;
        b[i] = (i as u64 * 5 + 7) % MOD_Q;
        c[i] = (i as u64 * 11 + 13) % MOD_Q;
        reduce_input[i] = MOD_Q + (i as u64 * 7);
    }

    let iters = 10_000_000usize;

    let rust_time = bench(iters * 10, || unsafe {
        eltwise_mult_mod(out.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n as u64, MOD_Q);
    });
    let cpp_time = bench(iters * 10, || unsafe {
        (wrapper.eltwise_mult_mod)(out.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n as u64, MOD_Q);
    });
    print_bench("eltwise_mult_mod", rust_time, cpp_time, iters * 10);

    let rust_time = bench(iters, || unsafe {
        eltwise_add_mod(out.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n as u64, MOD_Q);
    });
    let cpp_time = bench(iters, || unsafe {
        (wrapper.eltwise_add_mod)(out.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n, MOD_Q);
    });
    print_bench("eltwise_add_mod", rust_time, cpp_time, iters);

    let rust_time = bench(iters, || unsafe {
        eltwise_fma_mod(out.as_mut_ptr(), a.as_ptr(), 7, c.as_ptr(), n as u64, MOD_Q);
    });
    let cpp_time = bench(iters, || unsafe {
        (wrapper.eltwise_fma_mod)(out.as_mut_ptr(), a.as_ptr(), 7, c.as_ptr(), n as u64, MOD_Q);
    });
    print_bench("eltwise_fma_mod", rust_time, cpp_time, iters);

    let rust_time = bench(iters, || unsafe {
        eltwise_reduce_mod(out.as_mut_ptr(), reduce_input.as_ptr(), n as u64, MOD_Q);
    });
    let cpp_time = bench(iters, || unsafe {
        (wrapper.eltwise_reduce_mod)(out.as_mut_ptr(), reduce_input.as_ptr(), n, MOD_Q);
    });
    print_bench("eltwise_reduce_mod", rust_time, cpp_time, iters);

    let mut ntt_buf = vec![0u64; n];
    ntt_buf.copy_from_slice(&a);
    let rust_time = bench(iters, || unsafe {
        ntt_forward_in_place(ntt_buf.as_mut_ptr(), n, MOD_Q);
        ntt_inverse_in_place(ntt_buf.as_mut_ptr(), n, MOD_Q);
    });
    let cpp_time = bench(iters, || unsafe {
        (wrapper.ntt_forward_in_place)(ntt_buf.as_mut_ptr(), n, MOD_Q);
        (wrapper.ntt_inverse_in_place)(ntt_buf.as_mut_ptr(), n, MOD_Q);
    });
    print_bench("ntt fwd+inv", rust_time, cpp_time, iters);

    let mut full = vec![0u64; DEGREE];
    full.copy_from_slice(&[a.clone(), b.clone()].concat());
    println!("bench data prepared (degree {})", DEGREE);
}

fn verify_hexl_against_cpp(wrapper: &HexlWrapper) {
    use cowboys_and_aliens::common::config::{HALF_DEGREE, MOD_Q};
    use cowboys_and_aliens::hexl::bindings as rust_hexl;

    let n = HALF_DEGREE;
    let modulus = MOD_Q;

    let mut a = vec![0u64; n];
    let mut b = vec![0u64; n];
    let mut c = vec![0u64; n];
    let mut reduce_input = vec![0u64; n];
    for i in 0..n {
        a[i] = (i as u64 * 3) % modulus;
        b[i] = (i as u64 * 5 + 7) % modulus;
        c[i] = (i as u64 * 11 + 13) % modulus;
        reduce_input[i] = modulus + (i as u64 * 7);
    }

    unsafe {
        let a0 = modulus - 1;
        let b0 = modulus - 2;
        assert_eq!(
            rust_hexl::add_mod(a0, b0, modulus),
            (wrapper.add_mod)(a0, b0, modulus),
            "add_mod mismatch"
        );
        assert_eq!(
            rust_hexl::sub_mod(5, 7, modulus),
            (wrapper.sub_mod)(5, 7, modulus),
            "sub_mod mismatch"
        );
        assert_eq!(
            rust_hexl::multiply_mod(12345, 67890, modulus),
            (wrapper.multiply_mod)(12345, 67890, modulus),
            "multiply_mod mismatch"
        );
        assert_eq!(
            rust_hexl::power_mod(5, 123, modulus),
            (wrapper.power_mod)(5, 123, modulus),
            "power_mod mismatch"
        );
        assert_eq!(
            rust_hexl::inv_mod(7, modulus),
            (wrapper.inv_mod)(7, modulus),
            "inv_mod mismatch"
        );

        let rust_roots = std::slice::from_raw_parts(rust_hexl::get_roots(n as u64, modulus), n);
        let cpp_roots = std::slice::from_raw_parts((wrapper.get_roots)(n, modulus), n);
        assert_eq!(rust_roots, cpp_roots, "get_roots mismatch");

        let rust_inv_roots =
            std::slice::from_raw_parts(rust_hexl::get_inv_roots(n as u64, modulus), n);
        let cpp_inv_roots = std::slice::from_raw_parts((wrapper.get_inv_roots)(n, modulus), n);
        assert_eq!(rust_inv_roots, cpp_inv_roots, "get_inv_roots mismatch");

        let mut rust_out = vec![0u64; n];
        let mut cpp_out = vec![0u64; n];
        rust_hexl::eltwise_mult_mod(
            rust_out.as_mut_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            n as u64,
            modulus,
        );
        (wrapper.eltwise_mult_mod)(
            cpp_out.as_mut_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            n as u64,
            modulus,
        );
        assert_eq!(rust_out, cpp_out, "eltwise_mult_mod mismatch");

        rust_hexl::eltwise_add_mod(
            rust_out.as_mut_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            n as u64,
            modulus,
        );
        (wrapper.eltwise_add_mod)(cpp_out.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n, modulus);
        assert_eq!(rust_out, cpp_out, "eltwise_add_mod mismatch");

        rust_hexl::eltwise_sub_mod(
            rust_out.as_mut_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            n as u64,
            modulus,
        );
        (wrapper.eltwise_sub_mod)(cpp_out.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n, modulus);
        assert_eq!(rust_out, cpp_out, "eltwise_sub_mod mismatch");

        rust_hexl::eltwise_fma_mod(
            rust_out.as_mut_ptr(),
            a.as_ptr(),
            7,
            c.as_ptr(),
            n as u64,
            modulus,
        );
        (wrapper.eltwise_fma_mod)(
            cpp_out.as_mut_ptr(),
            a.as_ptr(),
            7,
            c.as_ptr(),
            n as u64,
            modulus,
        );
        assert_eq!(rust_out, cpp_out, "eltwise_fma_mod mismatch");

        rust_hexl::eltwise_reduce_mod(
            rust_out.as_mut_ptr(),
            reduce_input.as_ptr(),
            n as u64,
            modulus,
        );
        (wrapper.eltwise_reduce_mod)(cpp_out.as_mut_ptr(), reduce_input.as_ptr(), n, modulus);
        assert_eq!(rust_out, cpp_out, "eltwise_reduce_mod mismatch");

        let mut rust_ntt = a.clone();
        let mut cpp_ntt = a.clone();
        rust_hexl::ntt_forward_in_place(rust_ntt.as_mut_ptr(), n, modulus);
        (wrapper.ntt_forward_in_place)(cpp_ntt.as_mut_ptr(), n, modulus);
        assert_eq!(rust_ntt, cpp_ntt, "ntt_forward_in_place mismatch");

        let mut rust_inv = rust_ntt.clone();
        let mut cpp_inv = cpp_ntt.clone();
        rust_hexl::ntt_inverse_in_place(rust_inv.as_mut_ptr(), n, modulus);
        (wrapper.ntt_inverse_in_place)(cpp_inv.as_mut_ptr(), n, modulus);
        assert_eq!(rust_inv, cpp_inv, "ntt_inverse_in_place mismatch");
        assert_eq!(rust_inv, a, "ntt roundtrip mismatch");
    }
    println!("cpp comparison ok");
}

fn bench<F: FnMut()>(iters: usize, mut f: F) -> std::time::Duration {
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed()
}

fn print_bench(label: &str, rust: std::time::Duration, cpp: std::time::Duration, iters: usize) {
    let rust_ns = rust.as_nanos();
    let cpp_ns = cpp.as_nanos();
    let ratio = if cpp_ns == 0 {
        f64::INFINITY
    } else {
        rust_ns as f64 / cpp_ns as f64
    };
    println!(
        "{}: rust={:?}, cpp={:?}, rust/cpp={:.2} ({} iters)",
        label, rust, cpp, ratio, iters
    );
}

struct HexlWrapper {
    _lib: libloading::Library,
    multiply_mod: unsafe extern "C" fn(u64, u64, u64) -> u64,
    add_mod: unsafe extern "C" fn(u64, u64, u64) -> u64,
    sub_mod: unsafe extern "C" fn(u64, u64, u64) -> u64,
    get_roots: unsafe extern "C" fn(usize, u64) -> *const u64,
    inv_mod: unsafe extern "C" fn(u64, u64) -> u64,
    get_inv_roots: unsafe extern "C" fn(usize, u64) -> *const u64,
    power_mod: unsafe extern "C" fn(u64, u64, u64) -> u64,
    eltwise_mult_mod: unsafe extern "C" fn(*mut u64, *const u64, *const u64, u64, u64),
    eltwise_fma_mod: unsafe extern "C" fn(*mut u64, *const u64, u64, *const u64, u64, u64),
    eltwise_reduce_mod: unsafe extern "C" fn(*mut u64, *const u64, usize, u64),
    eltwise_add_mod: unsafe extern "C" fn(*mut u64, *const u64, *const u64, usize, u64),
    eltwise_sub_mod: unsafe extern "C" fn(*mut u64, *const u64, *const u64, usize, u64),
    ntt_forward_in_place: unsafe extern "C" fn(*mut u64, usize, u64),
    ntt_inverse_in_place: unsafe extern "C" fn(*mut u64, usize, u64),
}

impl HexlWrapper {
    unsafe fn load() -> Result<Self, String> {
        let candidates = [
            "./libhexl_wrapper.so",
            "hexl-bindings/libhexl_wrapper.so",
            "libhexl_wrapper.so",
        ];
        let mut errors = Vec::new();
        for candidate in candidates {
            match libloading::Library::new(candidate) {
                Ok(lib) => return Self::from_lib(lib),
                Err(err) => errors.push(format!("{candidate}: {err}")),
            }
        }
        Err(format!(
            "libhexl_wrapper.so not found. Tried:\n{}",
            errors.join("\n")
        ))
    }

    unsafe fn from_lib(lib: libloading::Library) -> Result<Self, String> {
        let multiply_mod = *lib
            .get::<unsafe extern "C" fn(u64, u64, u64) -> u64>(b"multiply_mod\0")
            .map_err(|err| err.to_string())?;
        let add_mod = *lib
            .get::<unsafe extern "C" fn(u64, u64, u64) -> u64>(b"add_mod\0")
            .map_err(|err| err.to_string())?;
        let sub_mod = *lib
            .get::<unsafe extern "C" fn(u64, u64, u64) -> u64>(b"sub_mod\0")
            .map_err(|err| err.to_string())?;
        let get_roots = *lib
            .get::<unsafe extern "C" fn(usize, u64) -> *const u64>(b"get_roots\0")
            .map_err(|err| err.to_string())?;
        let inv_mod = *lib
            .get::<unsafe extern "C" fn(u64, u64) -> u64>(b"inv_mod\0")
            .map_err(|err| err.to_string())?;
        let get_inv_roots = *lib
            .get::<unsafe extern "C" fn(usize, u64) -> *const u64>(b"get_inv_roots\0")
            .map_err(|err| err.to_string())?;
        let power_mod = *lib
            .get::<unsafe extern "C" fn(u64, u64, u64) -> u64>(b"power_mod\0")
            .map_err(|err| err.to_string())?;
        let eltwise_mult_mod = *lib
            .get::<unsafe extern "C" fn(*mut u64, *const u64, *const u64, u64, u64)>(
                b"eltwise_mult_mod\0",
            )
            .map_err(|err| err.to_string())?;
        let eltwise_fma_mod = *lib
            .get::<unsafe extern "C" fn(*mut u64, *const u64, u64, *const u64, u64, u64)>(
                b"eltwise_fma_mod\0",
            )
            .map_err(|err| err.to_string())?;
        let eltwise_reduce_mod = *lib
            .get::<unsafe extern "C" fn(*mut u64, *const u64, usize, u64)>(b"eltwise_reduce_mod\0")
            .map_err(|err| err.to_string())?;
        let eltwise_add_mod = *lib
            .get::<unsafe extern "C" fn(*mut u64, *const u64, *const u64, usize, u64)>(
                b"eltwise_add_mod\0",
            )
            .map_err(|err| err.to_string())?;
        let eltwise_sub_mod = *lib
            .get::<unsafe extern "C" fn(*mut u64, *const u64, *const u64, usize, u64)>(
                b"eltwise_sub_mod\0",
            )
            .map_err(|err| err.to_string())?;
        let ntt_forward_in_place = *lib
            .get::<unsafe extern "C" fn(*mut u64, usize, u64)>(b"ntt_forward_in_place\0")
            .map_err(|err| err.to_string())?;
        let ntt_inverse_in_place = *lib
            .get::<unsafe extern "C" fn(*mut u64, usize, u64)>(b"ntt_inverse_in_place\0")
            .map_err(|err| err.to_string())?;

        Ok(Self {
            _lib: lib,
            multiply_mod,
            add_mod,
            sub_mod,
            get_roots,
            inv_mod,
            get_inv_roots,
            power_mod,
            eltwise_mult_mod,
            eltwise_fma_mod,
            eltwise_reduce_mod,
            eltwise_add_mod,
            eltwise_sub_mod,
            ntt_forward_in_place,
            ntt_inverse_in_place,
        })
    }
}
