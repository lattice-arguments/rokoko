use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::hint::black_box;

const MODULUS: u64 = 1125899906826241;

// ─── 64-byte aligned buffers for AVX-512 ────────────────────────────────────

struct AlignedBuf<T: Copy> {
    ptr: *mut T,
    len: usize,
}

impl<T: Copy> AlignedBuf<T> {
    fn new(len: usize) -> Self {
        let layout =
            Layout::from_size_align(len * std::mem::size_of::<T>(), 64).expect("bad layout");
        let ptr = unsafe { alloc_zeroed(layout) as *mut T };
        assert!(!ptr.is_null());
        Self { ptr, len }
    }

    fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T: Copy> Drop for AlignedBuf<T> {
    fn drop(&mut self) {
        let layout =
            Layout::from_size_align(self.len * std::mem::size_of::<T>(), 64).expect("bad layout");
        unsafe {
            dealloc(self.ptr as *mut u8, layout);
        }
    }
}

fn fill_random(buf: &mut AlignedBuf<u64>) {
    use rand::Rng;
    let mut rng = rand::rng();
    for v in buf.as_mut_slice().iter_mut() {
        *v = rng.random::<u64>() % MODULUS;
    }
}

/// Compute shift factors for half-degree `n` by NTT-forwarding [0, 1, 0, …].
fn compute_shift_factors(n: usize) -> (AlignedBuf<u64>, AlignedBuf<f64>) {
    let mut factors = AlignedBuf::<u64>::new(n);
    if n > 1 {
        factors.as_mut_slice()[1] = 1;
    }
    hexl_rust::ntt_forward_in_place(factors.as_mut_slice(), n, MODULUS);

    let mut factors_f64 = AlignedBuf::<f64>::new(n);
    for (dst, &src) in factors_f64.as_mut_slice().iter_mut().zip(factors.as_slice()) {
        *dst = src as f64;
    }
    (factors, factors_f64)
}

fn bench_eltwise_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("eltwise_mult_comparison");

    for log_n in 6..=13 {
        let n = 1usize << log_n;
        let label = format!("2^{log_n}");

        // ── Data for eltwise_mult_mod (n elements) ──────────────────
        let mut operand1 = AlignedBuf::<u64>::new(n);
        let mut operand2 = AlignedBuf::<u64>::new(n);
        let mut result = AlignedBuf::<u64>::new(n);
        fill_random(&mut operand1);
        fill_random(&mut operand2);

        // hexl-rust eltwise_mult_mod (slice API)
        group.bench_with_input(
            BenchmarkId::new("hexl_rust/eltwise_mult_mod", &label),
            &n,
            |bencher, _| {
                bencher.iter(|| {
                    hexl_rust::eltwise_mult_mod(
                        black_box(result.as_mut_slice()),
                        black_box(operand1.as_slice()),
                        black_box(operand2.as_slice()),
                        black_box(MODULUS),
                    );
                    black_box(result.as_slice());
                });
            },
        );

        // rokoko bindings eltwise_mult_mod (unsafe raw-pointer API)
        group.bench_with_input(
            BenchmarkId::new("bindings/eltwise_mult_mod", &label),
            &n,
            |bencher, _| {
                bencher.iter(|| unsafe {
                    rokoko::hexl::bindings::eltwise_mult_mod(
                        black_box(result.as_mut_ptr()),
                        black_box(operand1.as_ptr()),
                        black_box(operand2.as_ptr()),
                        black_box(n as u64),
                        black_box(MODULUS),
                    );
                    black_box(result.as_slice());
                });
            },
        );

        // ── Data for fused_incomplete_ntt_mult (2·n elements) ───────
        let mut fused_op1 = AlignedBuf::<u64>::new(2 * n);
        let mut fused_op2 = AlignedBuf::<u64>::new(2 * n);
        let mut fused_result = AlignedBuf::<u64>::new(2 * n);
        fill_random(&mut fused_op1);
        fill_random(&mut fused_op2);
        let (shift_factors, shift_factors_f64) = compute_shift_factors(n);

        // hexl-rust fused_incomplete_ntt_mult
        group.bench_with_input(
            BenchmarkId::new("hexl_rust/fused_incomplete_ntt_mult", &label),
            &n,
            |bencher, _| {
                bencher.iter(|| {
                    hexl_rust::fused_incomplete_ntt_mult(
                        black_box(fused_result.as_mut_slice()),
                        black_box(fused_op1.as_slice()),
                        black_box(fused_op2.as_slice()),
                        black_box(shift_factors.as_slice()),
                        black_box(shift_factors_f64.as_slice()),
                        black_box(n),
                        black_box(MODULUS),
                    );
                    black_box(fused_result.as_slice());
                });
            },
        );
    }

    group.finish();
}

criterion_group! {
    name = eltwise_benches;
    config = Criterion::default();
    targets = bench_eltwise_ops
}
criterion_main!(eltwise_benches);
