use criterion::{criterion_group, criterion_main, Criterion};
use rokoko::common::hash::HashWrapper;
use rokoko::common::ring_arithmetic::{Representation, RingElement};
use rokoko::common::short_challenge::{sample_attempt, sample_short_challenge};
use std::hint::black_box;

fn bench_ring_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_multiplication");

    // a *= (b, c)  — out-of-place: a = b * c
    group.bench_function("mul_assign_tuple", |bencher| {
        let b = RingElement::random(Representation::IncompleteNTT);
        let c = RingElement::random(Representation::IncompleteNTT);
        let mut a = RingElement::new(Representation::IncompleteNTT);

        bencher.iter(|| {
            a *= (black_box(&b), black_box(&c));
            black_box(&a);
        });
    });

    // a *= &b  — in-place multiplication
    group.bench_function("mul_assign_in_place", |bencher| {
        let b = RingElement::random(Representation::IncompleteNTT);
        let mut a = RingElement::random(Representation::IncompleteNTT);

        bencher.iter(|| {
            a *= black_box(&b);
            black_box(&a);
        });
    });

    group.finish();
}

fn bench_short_challenge(c: &mut Criterion) {
    let mut group = c.benchmark_group("short_challenge");
    group.bench_function("sample_accepted", |bencher| {
        let mut hasher = HashWrapper::new();
        bencher.iter(|| {
            let (challenge, _attempts) = sample_short_challenge(black_box(&mut hasher));
            black_box(challenge);
        });
    });
    group.bench_function("sample_attempt", |bencher| {
        let mut hasher = HashWrapper::new();
        bencher.iter(|| {
            let res = sample_attempt(black_box(&mut hasher));
            black_box(res);
        });
    });
    group.bench_function("op_norm_sq_sparse", |bencher| {
        let positions: [u8; 22] = [
            0, 3, 7, 11, 17, 23, 29, 31, 37, 41, 47, 53, 59, 61, 67, 73, 79, 83, 97, 103, 109, 127,
        ];
        let signs: [i8; 22] = [
            1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1,
        ];
        bencher.iter(|| {
            let v = rokoko::common::short_challenge::op_norm_sq_sparse(
                black_box(&positions),
                black_box(&signs),
            );
            black_box(v);
        });
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_ring_multiplication, bench_short_challenge
}
criterion_main!(benches);
