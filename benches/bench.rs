use criterion::{criterion_group, criterion_main, Criterion};
use rokoko::common::hash::HashWrapper;
use rokoko::common::matrix::VerticallyAlignedMatrix;
use rokoko::common::projection_matrix::ProjectionMatrix;
use rokoko::common::ring_arithmetic::{Representation, RingElement};
use rokoko::common::short_challenge::sample_short_challenge;
use rokoko::protocol::project_coarse::{prepare_i16_witness, project};
use std::hint::black_box;

fn bench_project_coarse(c: &mut Criterion) {
    rokoko::common::init_common();
    let mut group = c.benchmark_group("project_coarse");
    group.sample_size(10);

    let height = 2usize.pow(13);
    let width = 2usize.pow(3);
    let mut projection_matrix = ProjectionMatrix::new(2usize.pow(5), 2usize.pow(8));
    projection_matrix.sample(&mut HashWrapper::new());

    let witness = VerticallyAlignedMatrix {
        data: (0..height * width)
            .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 1 << 12))
            .collect(),
        width,
        height,
        used_cols: width,
    };
    let witness_i16 = prepare_i16_witness(&witness);

    group.bench_function("prepare_i16_witness", |bencher| {
        bencher.iter(|| {
            black_box(prepare_i16_witness(black_box(&witness)));
        });
    });

    group.bench_function("project", |bencher| {
        bencher.iter(|| {
            black_box(project(
                black_box(&witness_i16),
                black_box(&projection_matrix),
            ));
        });
    });

    group.finish();
}

fn bench_decompose(c: &mut Criterion) {
    use rokoko::common::decomposition::decompose;

    rokoko::common::init_common();
    let mut group = c.benchmark_group("decompose");
    group.sample_size(10);

    let input: Vec<RingElement> = (0..1usize << 16)
        .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 1 << 30))
        .collect();

    group.bench_function("base 2^16, radix 2", |bencher| {
        bencher.iter(|| black_box(decompose(black_box(&input), 16, 2)));
    });

    group.finish();
}

fn bench_commitment(c: &mut Criterion) {
    use rokoko::protocol::commitment::commit_basic;
    use rokoko::protocol::commitment_crt::{commit_basic_crt, digits_l2, CrtKey, Plan};
    use rokoko::protocol::crs::CRS;

    rokoko::common::init_common();
    let height: usize = std::env::var("ROKOKO_BENCH_HEIGHT")
        .map(|v| v.parse().unwrap())
        .unwrap_or(2usize.pow(11));
    let width: usize = std::env::var("ROKOKO_BENCH_WIDTH")
        .map(|v| v.parse().unwrap())
        .unwrap_or(16);
    let rank = 10;
    let mut group = c.benchmark_group("commitment");
    group.sample_size(10);

    let crs = CRS::gen_crs(height, rank + 2);
    let witness = VerticallyAlignedMatrix {
        data: (0..height * width)
            .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 1 << 15))
            .collect(),
        width,
        height,
        used_cols: width,
    };
    let start = std::time::Instant::now();
    let digits = prepare_i16_witness(&witness);
    let narrowing = start.elapsed();
    let plan = Plan::new(digits_l2(&digits), rank);
    let start = std::time::Instant::now();
    let key = CrtKey::preprocess(crs.ck_for_wit_dim(height), rank, &plan);
    println!(
        "key {:.0} MB, digits {:.0} MB, ring witness {:.0} MB",
        key.bytes() as f64 / 1e6,
        (digits.data.len() * 256) as f64 / 1e6,
        (witness.data.len() * 1088) as f64 / 1e6,
    );
    println!(
        "plan: {:?}; digits {:.0} ms, key {:.0} ms",
        plan.primes,
        narrowing.as_secs_f64() * 1e3,
        start.elapsed().as_secs_f64() * 1e3
    );

    group.bench_function("ring", |bencher| {
        bencher.iter(|| black_box(commit_basic(black_box(&crs), black_box(&witness), rank)));
    });

    if std::env::var("ROKOKO_BENCH_LIMBS").is_ok() {
        for set in [
            vec![7681, 7937, 9473, 10753, 11777, 12289],
            vec![9473, 10753, 11777, 12289, 13313, 7681],
            vec![3329, 7681, 7937, 9473, 10753, 11777, 12289],
            vec![3329, 7681, 7937, 9473, 10753, 11777],
        ] {
            let bits: f64 = set.iter().map(|p| (*p as f64).log2()).sum();
            if bits < (2.0 * plan.bound).log2() {
                continue;
            }
            let forced = Plan {
                primes: set.clone(),
                bound: plan.bound,
            };
            let forced_key = CrtKey::preprocess(crs.ck_for_wit_dim(height), rank, &forced);
            group.bench_function(format!("crt {set:?}"), |bencher| {
                bencher.iter(|| {
                    black_box(commit_basic_crt(
                        black_box(&forced_key),
                        black_box(&digits),
                        &forced,
                        rank,
                    ))
                });
            });
        }
    }

    group.bench_function("crt", |bencher| {
        bencher.iter(|| {
            black_box(commit_basic_crt(
                black_box(&key),
                black_box(&digits),
                &plan,
                rank,
            ))
        });
    });

    group.bench_function("crt streaming", |bencher| {
        bencher.iter(|| {
            black_box(
                rokoko::protocol::commitment_crt::commit_basic_crt_streaming(
                    black_box(&key),
                    black_box(&witness),
                    &plan,
                    rank,
                ),
            )
        });
    });

    group.bench_function("crt end to end", |bencher| {
        bencher.iter(|| {
            black_box(rokoko::protocol::commitment_crt::commit_basic(
                black_box(&crs),
                black_box(&witness),
                rank,
            ))
        });
    });

    group.finish();
}

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
    targets = bench_decompose, bench_commitment, bench_ring_multiplication, bench_short_challenge, bench_project_coarse
}
criterion_main!(benches);
