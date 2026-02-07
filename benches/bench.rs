use criterion::{criterion_group, criterion_main, Criterion};
use rokoko::common::ring_arithmetic::{Representation, RingElement};
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

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_ring_multiplication
}
criterion_main!(benches);
