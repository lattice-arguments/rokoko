use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rokoko::common::{
    ring_arithmetic::{Representation, RingElement},
    structured_row::StructuredRow,
};
use rokoko::protocol::sumcheck_utils::{
    common::EvaluationSumcheckData,
    diff::DiffSumcheckEvaluation,
    elephant_cell::ElephantCell,
    linear::{BasicEvaluationLinearSumcheck, StructuredRowEvaluationLinearSumcheck},
    product::ProductSumcheckEvaluation,
    selector_eq::SelectorEqEvaluation,
};
use std::hint::black_box;

struct MsBasicEvaluationLinearSumcheck {
    data: Vec<RingElement>,
    variable_count: usize,
    suffix: usize,
    evaluated: bool,
}

impl MsBasicEvaluationLinearSumcheck {
    fn new_with_prefixed_sufixed_data(count: usize, prefix_size: usize, suffix_size: usize) -> Self {
        MsBasicEvaluationLinearSumcheck {
            data: vec![RingElement::zero(Representation::IncompleteNTT); count],
            variable_count: count.ilog2() as usize + prefix_size + suffix_size,
            suffix: suffix_size,
            evaluated: false,
        }
    }

    fn new(count: usize) -> Self {
        Self::new_with_prefixed_sufixed_data(count, 0, 0)
    }

    fn load_from(&mut self, src: &[RingElement]) {
        self.data.clone_from_slice(src);
        self.evaluated = false;
    }
}

impl EvaluationSumcheckData for MsBasicEvaluationLinearSumcheck {
    type Element = RingElement;

    fn evaluate(&mut self, point: &Vec<Self::Element>) -> &Self::Element {
        if self.evaluated {
            return &self.data[0];
        }

        if point.len() != self.variable_count {
            panic!("Point has incorrect number of variables");
        }

        let data_variable_count = self.data.len().ilog2() as usize;
        let prefix_size = self.variable_count - data_variable_count - self.suffix;
        let mut current_len = self.data.len();
        let mut current_variable = 0;

        for r in point.iter() {
            if current_variable < prefix_size {
                current_variable += 1;
                continue;
            }
            if current_variable >= prefix_size + data_variable_count {
                current_variable += 1;
                continue;
            }

            let half = current_len / 2;
            for i in 0..half {
                let mut delta = self.data[i + half].clone();
                delta -= &self.data[i];
                delta *= r;
                self.data[i] += &delta;
            }
            current_len = half;
            current_variable += 1;
        }

        self.evaluated = true;
        &self.data[0]
    }
}

struct MsSelectorEqEvaluation {
    selector: usize,
    selector_variable_count: usize,
    total_variable_count: usize,
    result: RingElement,
    scratch: RingElement,
    evaluated: bool,
}

impl MsSelectorEqEvaluation {
    fn new(selector: usize, selector_variable_count: usize, total_variable_count: usize) -> Self {
        MsSelectorEqEvaluation {
            selector,
            selector_variable_count,
            total_variable_count,
            result: RingElement::constant(1, Representation::IncompleteNTT),
            scratch: RingElement::zero(Representation::IncompleteNTT),
            evaluated: false,
        }
    }
}

impl EvaluationSumcheckData for MsSelectorEqEvaluation {
    type Element = RingElement;

    fn evaluate(&mut self, point: &Vec<Self::Element>) -> &Self::Element {
        if self.evaluated {
            return &self.result;
        }
        if point.len() != self.total_variable_count {
            panic!("Point has incorrect number of variables");
        }

        self.result = RingElement::constant(1, Representation::IncompleteNTT);
        for i in 0..self.selector_variable_count {
            let selector_bit = (self.selector >> (self.selector_variable_count - 1 - i)) & 1;
            let r = &point[i];
            if selector_bit == 1 {
                self.result *= r;
            } else {
                self.scratch.set_from(&self.result);
                self.scratch *= r;
                self.result -= &self.scratch;
            }
        }
        self.evaluated = true;
        &self.result
    }
}

fn make_data(count: usize) -> Vec<RingElement> {
    (0..count)
        .map(|i| RingElement::constant((i as u64).wrapping_mul(17).wrapping_add(3), Representation::IncompleteNTT))
        .collect()
}

fn make_point(vars: usize) -> Vec<RingElement> {
    (0..vars)
        .map(|i| RingElement::constant((i as u64).wrapping_mul(97).wrapping_add(11), Representation::IncompleteNTT))
        .collect()
}

fn bench_linear_diff_prod_selector(c: &mut Criterion) {
    let mut group = c.benchmark_group("sumcheck_eval_order");
    let vars = 15usize;
    let count = 1usize << vars;
    let data_l = make_data(count);
    let data_r = make_data(count).into_iter().rev().collect::<Vec<_>>();

    let point_ls = make_point(vars);
    let point_ms = point_ls.iter().rev().cloned().collect::<Vec<_>>();

    group.bench_function("linear_ls", |b| {
        b.iter_batched(
            || {
                let mut ev = BasicEvaluationLinearSumcheck::<RingElement>::new(count);
                ev.load_from(&data_l);
                ev
            },
            |mut ev| {
                let out = ev.evaluate(black_box(&point_ls));
                black_box(out);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("linear_ms", |b| {
        b.iter_batched(
            || {
                let mut ev = MsBasicEvaluationLinearSumcheck::new(count);
                ev.load_from(&data_l);
                ev
            },
            |mut ev| {
                let out = ev.evaluate(black_box(&point_ms));
                black_box(out);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("diff_ls", |b| {
        b.iter_batched(
            || {
                let mut lhs = BasicEvaluationLinearSumcheck::<RingElement>::new(count);
                lhs.load_from(&data_l);
                let mut rhs = BasicEvaluationLinearSumcheck::<RingElement>::new(count);
                rhs.load_from(&data_r);
                DiffSumcheckEvaluation::new(ElephantCell::new(lhs), ElephantCell::new(rhs))
            },
            |mut ev| {
                let out = ev.evaluate(black_box(&point_ls));
                black_box(out);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("diff_ms", |b| {
        b.iter_batched(
            || {
                let mut lhs = MsBasicEvaluationLinearSumcheck::new(count);
                lhs.load_from(&data_l);
                let mut rhs = MsBasicEvaluationLinearSumcheck::new(count);
                rhs.load_from(&data_r);
                DiffSumcheckEvaluation::new(ElephantCell::new(lhs), ElephantCell::new(rhs))
            },
            |mut ev| {
                let out = ev.evaluate(black_box(&point_ms));
                black_box(out);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("product_ls", |b| {
        b.iter_batched(
            || {
                let mut lhs = BasicEvaluationLinearSumcheck::<RingElement>::new(count);
                lhs.load_from(&data_l);
                let mut rhs = BasicEvaluationLinearSumcheck::<RingElement>::new(count);
                rhs.load_from(&data_r);
                ProductSumcheckEvaluation::new(ElephantCell::new(lhs), ElephantCell::new(rhs))
            },
            |mut ev| {
                let out = ev.evaluate(black_box(&point_ls));
                black_box(out);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("product_ms", |b| {
        b.iter_batched(
            || {
                let mut lhs = MsBasicEvaluationLinearSumcheck::new(count);
                lhs.load_from(&data_l);
                let mut rhs = MsBasicEvaluationLinearSumcheck::new(count);
                rhs.load_from(&data_r);
                ProductSumcheckEvaluation::new(ElephantCell::new(lhs), ElephantCell::new(rhs))
            },
            |mut ev| {
                let out = ev.evaluate(black_box(&point_ms));
                black_box(out);
            },
            BatchSize::SmallInput,
        );
    });

    // Structured-row linear evaluation benchmark.
    let row = StructuredRow {
        tensor_layers: make_point(vars),
    };
    group.bench_function("structured_linear_ls", |b| {
        b.iter_batched(
            || {
                let mut ev = StructuredRowEvaluationLinearSumcheck::<RingElement>::new(count);
                ev.load_from(row.clone());
                ev
            },
            |mut ev| {
                let out = ev.evaluate(black_box(&point_ls));
                black_box(out);
            },
            BatchSize::SmallInput,
        );
    });

    // Selector-eq evaluation benchmark.
    let selector_vars = 6usize;
    let total_vars = 16usize;
    let selector = 0b101101usize;
    let selector_point_ls = make_point(total_vars);
    let mut selector_point_ms = selector_point_ls[total_vars - selector_vars..]
        .iter()
        .rev()
        .cloned()
        .collect::<Vec<_>>();
    selector_point_ms.extend_from_slice(&selector_point_ls[..total_vars - selector_vars]);

    group.bench_function("selector_ls", |b| {
        b.iter_batched(
            || SelectorEqEvaluation::new(selector, selector_vars, total_vars),
            |mut ev| {
                let out = ev.evaluate(black_box(&selector_point_ls));
                black_box(out);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("selector_ms", |b| {
        b.iter_batched(
            || MsSelectorEqEvaluation::new(selector, selector_vars, total_vars),
            |mut ev| {
                let out = ev.evaluate(black_box(&selector_point_ms));
                black_box(out);
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_linear_diff_prod_selector);
criterion_main!(benches);
