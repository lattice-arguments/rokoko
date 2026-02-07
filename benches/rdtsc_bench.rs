//! Standalone rdtsc-based microbenchmark for fused_incomplete_ntt_mult.
//! Build & run: cargo run --release --features rust-hexl --example rdtsc_bench

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::hint::black_box;

const MODULUS: u64 = 1125899906826241;
const WARMUP_ITERS: u64 = 50_000;
const BENCH_ITERS: u64 = 200_000;

struct AlignedBuf<T: Copy> {
    ptr: *mut T,
    len: usize,
}

impl<T: Copy> AlignedBuf<T> {
    fn new(len: usize) -> Self {
        let layout = Layout::from_size_align(len * std::mem::size_of::<T>(), 64).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) as *mut T };
        assert!(!ptr.is_null());
        Self { ptr, len }
    }
    fn as_slice(&self) -> &[T] { unsafe { std::slice::from_raw_parts(self.ptr, self.len) } }
    fn as_mut_slice(&mut self) -> &mut [T] { unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) } }
}

impl<T: Copy> Drop for AlignedBuf<T> {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.len * std::mem::size_of::<T>(), 64).unwrap();
        unsafe { dealloc(self.ptr as *mut u8, layout); }
    }
}

fn fill_random(buf: &mut AlignedBuf<u64>) {
    use rand::Rng;
    let mut rng = rand::rng();
    for v in buf.as_mut_slice().iter_mut() {
        *v = rng.random::<u64>() % MODULUS;
    }
}

fn compute_shift_factors(n: usize) -> (AlignedBuf<u64>, AlignedBuf<f64>) {
    let mut factors = AlignedBuf::<u64>::new(n);
    if n > 1 { factors.as_mut_slice()[1] = 1; }
    hexl_rust::ntt_forward_in_place(factors.as_mut_slice(), n, MODULUS);
    let mut factors_f64 = AlignedBuf::<f64>::new(n);
    for (dst, &src) in factors_f64.as_mut_slice().iter_mut().zip(factors.as_slice()) {
        *dst = src as f64;
    }
    (factors, factors_f64)
}

#[inline(always)]
fn rdtsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

fn bench_fused(log_n: u32) {
    let n = 1usize << log_n;

    let mut op1 = AlignedBuf::<u64>::new(2 * n);
    let mut op2 = AlignedBuf::<u64>::new(2 * n);
    let mut result = AlignedBuf::<u64>::new(2 * n);
    fill_random(&mut op1);
    fill_random(&mut op2);
    let (shift_u64, shift_f64) = compute_shift_factors(n);

    // Warm up
    for _ in 0..WARMUP_ITERS {
        hexl_rust::fused_incomplete_ntt_mult(
            black_box(result.as_mut_slice()),
            black_box(op1.as_slice()),
            black_box(op2.as_slice()),
            black_box(shift_u64.as_slice()),
            black_box(shift_f64.as_slice()),
            black_box(n),
            black_box(MODULUS),
        );
    }

    // Benchmark with rdtsc
    let mut best = u64::MAX;
    let mut total = 0u64;
    for _ in 0..BENCH_ITERS {
        let start = rdtsc();
        hexl_rust::fused_incomplete_ntt_mult(
            black_box(result.as_mut_slice()),
            black_box(op1.as_slice()),
            black_box(op2.as_slice()),
            black_box(shift_u64.as_slice()),
            black_box(shift_f64.as_slice()),
            black_box(n),
            black_box(MODULUS),
        );
        let elapsed = rdtsc() - start;
        if elapsed < best { best = elapsed; }
        total += elapsed;
    }
    let avg = total / BENCH_ITERS;

    // Also time-based measurement
    let start_time = std::time::Instant::now();
    let time_iters = BENCH_ITERS;
    for _ in 0..time_iters {
        hexl_rust::fused_incomplete_ntt_mult(
            black_box(result.as_mut_slice()),
            black_box(op1.as_slice()),
            black_box(op2.as_slice()),
            black_box(shift_u64.as_slice()),
            black_box(shift_f64.as_slice()),
            black_box(n),
            black_box(MODULUS),
        );
    }
    let elapsed_ns = start_time.elapsed().as_nanos() as u64;
    let ns_per_iter = elapsed_ns / time_iters;

    println!(
        "N={:>5} (n={:>5}): rdtsc best={:>5} cy, avg={:>5} cy | time: {:>5} ns/iter",
        2 * n, n, best, avg, ns_per_iter
    );
}

fn bench_eltwise(log_n: u32) {
    let n = 1usize << log_n;

    let mut op1 = AlignedBuf::<u64>::new(n);
    let mut op2 = AlignedBuf::<u64>::new(n);
    let mut result = AlignedBuf::<u64>::new(n);
    fill_random(&mut op1);
    fill_random(&mut op2);

    // Warm up
    for _ in 0..WARMUP_ITERS {
        hexl_rust::eltwise_mult_mod(
            black_box(result.as_mut_slice()),
            black_box(op1.as_slice()),
            black_box(op2.as_slice()),
            black_box(MODULUS),
        );
    }

    let mut best = u64::MAX;
    let mut total = 0u64;
    for _ in 0..BENCH_ITERS {
        let start = rdtsc();
        hexl_rust::eltwise_mult_mod(
            black_box(result.as_mut_slice()),
            black_box(op1.as_slice()),
            black_box(op2.as_slice()),
            black_box(MODULUS),
        );
        let elapsed = rdtsc() - start;
        if elapsed < best { best = elapsed; }
        total += elapsed;
    }
    let avg = total / BENCH_ITERS;

    let start_time = std::time::Instant::now();
    for _ in 0..BENCH_ITERS {
        hexl_rust::eltwise_mult_mod(
            black_box(result.as_mut_slice()),
            black_box(op1.as_slice()),
            black_box(op2.as_slice()),
            black_box(MODULUS),
        );
    }
    let elapsed_ns = start_time.elapsed().as_nanos() as u64;
    let ns_per_iter = elapsed_ns / BENCH_ITERS;

    println!(
        "eltwise N={:>5}: rdtsc best={:>5} cy, avg={:>5} cy | time: {:>5} ns/iter",
        n, best, avg, ns_per_iter
    );
}

fn main() {
    println!("=== Standalone rdtsc benchmark ===");
    println!("Warmup: {} iters, Bench: {} iters", WARMUP_ITERS, BENCH_ITERS);
    println!();

    // Eltwise mult mod reference
    for log_n in [6, 7, 8, 9, 10, 11, 12, 13] {
        bench_eltwise(log_n);
    }
    println!();

    // Fused kernel
    for log_n in [6, 7, 8, 9, 10, 11, 12, 13] {
        bench_fused(log_n);
    }
}
