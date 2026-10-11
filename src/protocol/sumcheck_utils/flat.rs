use crate::{
    common::{
        config::MOD_Q,
        ring_arithmetic::{incomplete_ntt_dot_into, Representation, RingElement},
    },
    protocol::sumcheck_utils::{
        common::{FlatTerm, HighOrderSumcheckData},
        elephant_cell::ElephantCell,
        hypercube_point::HypercubePoint,
        linear::LinearSumcheck,
        polynomial::{add_poly_in_place, Polynomial},
    },
};

type Cell = ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>;
type Term<'a> = FlatTerm<'a, RingElement>;

const CHUNK: usize = 32;

pub struct FlatRounds {
    rounds: usize,
    previous: std::cell::RefCell<Vec<Polynomial<RingElement>>>,
}

struct Plan<'a> {
    output: usize,
    negative: bool,
    start: usize,
    end: usize,
    slots: Vec<usize>,
    inactive: Vec<&'a LinearSumcheck<RingElement>>,
    claims: Vec<&'a RingElement>,
    rounds: Vec<(bool, &'a RingElement)>,
    shift: Option<u32>,
    karatsuba_pair: bool,
}

impl Plan<'_> {
    fn degree(&self) -> usize {
        self.slots.len() + self.rounds.len()
    }

    fn weighted(&self) -> bool {
        !self.inactive.is_empty() || !self.claims.is_empty() || !self.rounds.is_empty()
    }

    fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

struct Slot<'a> {
    table: &'a LinearSumcheck<RingElement>,
    pairs: usize,
    mapped: Vec<RingElement>,
    slopes: Vec<RingElement>,
    twos: Vec<RingElement>,
    base: usize,
    needs_two: bool,
}

struct Accumulator {
    sums: Vec<RingElement>,
    run: Vec<RingElement>,
    key: Option<usize>,
    anchor: usize,
}

fn zero() -> RingElement {
    RingElement::zero(Representation::IncompleteNTT)
}

fn points(degree: usize) -> usize {
    if degree == 1 {
        2
    } else {
        degree + 1
    }
}

#[cfg(feature = "unsafe-sumcheck")]
fn flatten(cell: &Cell) -> Option<Vec<Term<'_>>> {
    let mut terms = vec![];
    cell.get_ref().flat_terms(&mut terms).map(|_| terms)
}

#[cfg(not(feature = "unsafe-sumcheck"))]
fn flatten(_: &Cell) -> Option<Vec<Term<'_>>> {
    None
}

impl FlatRounds {
    pub fn new(outputs: &[Cell], total_vars: usize) -> FlatRounds {
        let mut any = false;
        let mut full = true;
        for terms in outputs.iter().filter_map(flatten) {
            for term in &terms {
                any = true;
                full &= term
                    .tables
                    .iter()
                    .any(|table| table.len() == 1 << total_vars && table.suffix() == 0);
            }
        }
        FlatRounds {
            rounds: if any && full {
                total_vars.saturating_sub(CHUNK.ilog2() as usize)
            } else {
                0
            },
            previous: std::cell::RefCell::new(vec![]),
        }
    }

    pub fn rounds(&self) -> usize {
        self.rounds
    }

    pub fn round_polynomial(
        &self,
        outputs: &[Cell],
        gammas: &[RingElement],
        challenge: Option<&RingElement>,
    ) -> Polynomial<RingElement> {
        let flattened: Vec<Option<Vec<Term>>> = outputs.iter().map(flatten).collect();
        let mut slots: Vec<Slot> = vec![];
        let mut plans: Vec<Plan> = vec![];
        for (output, terms) in flattened.iter().enumerate() {
            for term in terms.iter().flatten() {
                plans.push(plan(output, term, &mut slots));
            }
        }
        let previous = self.previous.take();
        let skip = challenge.is_some() && previous.len() == plans.len();
        let mut accumulators: Vec<Accumulator> = plans
            .iter()
            .map(|plan| Accumulator {
                sums: vec![zero(); points(plan.degree())],
                run: vec![zero(); if plan.weighted() { points(plan.degree()) } else { 0 }],
                key: None,
                anchor: 0,
            })
            .collect();
        for plan in &plans {
            if points(plan.degree()) == 4 {
                for &slot in &plan.slots {
                    slots[slot].needs_two = true;
                }
            }
        }

        let mut bounds: Vec<usize> = plans
            .iter()
            .filter(|plan| !plan.is_empty())
            .flat_map(|plan| [plan.start, plan.end])
            .collect();
        bounds.sort_unstable();
        bounds.dedup();

        let mut product = zero();
        let mut scratch = if plans.iter().any(|plan| plan.slots.len() == 3) {
            vec![zero(); CHUNK]
        } else {
            vec![]
        };
        let mut used = vec![false; slots.len()];
        let mut members: Vec<usize> = vec![];
        for window in bounds.windows(2) {
            let (from, to) = (window[0], window[1]);
            members.clear();
            members.extend((0..plans.len()).filter(|&m| plans[m].start <= from && to <= plans[m].end));
            if members.is_empty() {
                continue;
            }
            used.iter_mut().for_each(|u| *u = false);
            for &m in &members {
                for &slot in &plans[m].slots {
                    used[slot] = true;
                }
            }
            let width = slots
                .iter()
                .zip(&used)
                .filter(|(_, u)| **u)
                .map(|(slot, _)| slot.pairs)
                .fold(CHUNK, usize::min);
            let mut x = from;
            while x < to {
                let y = to.min((x / width + 1) * width);
                for (slot, _) in slots.iter_mut().zip(&used).filter(|(_, u)| **u) {
                    prepare(slot, x, y);
                }
                for &m in &members {
                    let plan = &plans[m];
                    let accumulator = &mut accumulators[m];
                    let mut u = x;
                    while u < y {
                        let v = match plan.shift {
                            Some(shift) => y.min(((u >> shift) + 1) << shift),
                            None => y,
                        };
                        let key = plan.shift.map_or(0, |shift| u >> shift);
                        if accumulator.key != Some(key) {
                            flush(plan, accumulator, skip, &mut product);
                            accumulator.key = Some(key);
                            accumulator.anchor = u;
                        }
                        let target = if plan.weighted() {
                            &mut accumulator.run
                        } else {
                            &mut accumulator.sums
                        };
                        accumulate(plan, &slots, u - x, v - u, skip, target, &mut scratch, &mut product);
                        u = v;
                    }
                }
                x = y;
            }
        }
        for (plan, accumulator) in plans.iter().zip(accumulators.iter_mut()) {
            flush(plan, accumulator, skip, &mut product);
        }
        if let Some(r) = challenge.filter(|_| skip) {
            for (accumulator, previous) in accumulators.iter_mut().zip(previous.iter()) {
                let mut at = previous.coefficients[previous.num_coefficients - 1].clone();
                for i in (0..previous.num_coefficients - 1).rev() {
                    at *= r;
                    at += &previous.coefficients[i];
                }
                let (zero_point, rest) = accumulator.sums.split_at_mut(1);
                rest[0] -= (&at, &zero_point[0]);
            }
        }
        let mut current = Vec::with_capacity(plans.len());

        let mut total = Polynomial::<RingElement>::new(0);
        total.num_coefficients = 0;
        let mut coefficients = Polynomial::<RingElement>::new(0);
        for (output, terms) in flattened.iter().enumerate() {
            let mut poly = Polynomial::<RingElement>::new(0);
            poly.num_coefficients = 0;
            match terms {
                None => outputs[output].get_ref().univariate_polynomial_into(&mut poly),
                Some(_) => {
                    poly.num_coefficients = 1;
                    for (plan, accumulator) in plans.iter().zip(&accumulators) {
                        if plan.output != output {
                            continue;
                        }
                        let degree = plan.degree();
                        if plan.karatsuba_pair {
                            poly.num_coefficients = poly.num_coefficients.max(3);
                        }
                        if plan.is_empty() {
                            current.push(Polynomial::<RingElement>::new(0));
                            current.last_mut().unwrap().num_coefficients = 1;
                            continue;
                        }
                        poly.num_coefficients = poly.num_coefficients.max(degree + 1);
                        interpolate(degree, &accumulator.sums, &mut coefficients);
                        let mut kept = Polynomial::<RingElement>::new(0);
                        for i in 0..=degree {
                            kept.coefficients[i].set_from(&coefficients.coefficients[i]);
                        }
                        kept.num_coefficients = degree + 1;
                        current.push(kept);
                        for i in 0..=degree {
                            if plan.negative {
                                poly.coefficients[i] -= &coefficients.coefficients[i];
                            } else {
                                poly.coefficients[i] += &coefficients.coefficients[i];
                            }
                        }
                    }
                }
            }
            for i in 0..poly.num_coefficients {
                poly.coefficients[i] *= &gammas[output];
            }
            add_poly_in_place(&mut total, &poly);
        }
        if current.len() == plans.len() {
            *self.previous.borrow_mut() = current;
        }
        total
    }
}

fn plan<'a>(output: usize, term: &Term<'a>, slots: &mut Vec<Slot<'a>>) -> Plan<'a> {
    let mut start = 0;
    let mut end = usize::MAX;
    let mut half = 0;
    let mut active = vec![];
    let mut inactive = vec![];
    let mut shift: Option<u32> = None;
    let mut interleaved = 0;
    let mut claims = vec![];
    let mut rounds = vec![];
    for &selector in &term.selectors {
        if let Some((from, to)) = selector.non_zero_range() {
            start = start.max(from);
            end = end.min(to);
        }
        match selector.round_bit() {
            Some(bit) => rounds.push((bit, selector.current_claim())),
            None if !selector.is_unit_constant() => claims.push(selector.current_claim()),
            None => {}
        }
    }
    for &table in &term.tables {
        interleaved += table.as_interleaved_data().is_some() as usize;
        if let Some((from, to)) = table.non_zero_range() {
            start = start.max(from);
            end = end.min(to);
        }
        if table.is_folding() {
            half = half.max(table.len() / 2);
            let slot = match slots.iter().position(|s| std::ptr::eq(s.table, table)) {
                Some(slot) => slot,
                None => {
                    let width = CHUNK.min(table.len() / 2);
                    slots.push(Slot {
                        table,
                        pairs: table.len() / 2,
                        mapped: if table.values().is_none() {
                            vec![zero(); 2 * width]
                        } else {
                            vec![]
                        },
                        slopes: vec![zero(); width],
                        twos: vec![],
                        base: 0,
                        needs_two: false,
                    });
                    slots.len() - 1
                }
            };
            active.push(slot);
        } else {
            if table.suffix() > 0 {
                let s = table.suffix() as u32 - 1;
                shift = Some(shift.map_or(s, |t| t.min(s)));
            }
            inactive.push(table);
        }
    }
    end = end.min(half);
    assert!(!active.is_empty() && active.len() + rounds.len() <= 3);
    Plan {
        output,
        negative: term.negative,
        start,
        end,
        slots: active,
        inactive,
        claims,
        rounds,
        shift,
        karatsuba_pair: term.pair && term.tables.len() == 2 && interleaved == 2,
    }
}

fn prepare(slot: &mut Slot, from: usize, to: usize) {
    let base = from & (slot.pairs - 1);
    slot.base = base;
    let count = to - from;
    if !slot.mapped.is_empty() {
        for k in 0..count {
            let (even, odd) = slot.mapped[2 * k..2 * k + 2].split_at_mut(1);
            slot.table.pair_into(base + k, &mut even[0], &mut odd[0]);
        }
    }
    if slot.needs_two && slot.twos.is_empty() {
        slot.twos = vec![zero(); slot.slopes.len()];
    }
    let Slot {
        table,
        mapped,
        slopes,
        twos,
        needs_two,
        ..
    } = slot;
    let values: &[RingElement] = if mapped.is_empty() {
        &table.values().unwrap()[2 * base..]
    } else {
        mapped
    };
    for k in 0..count {
        slopes[k] -= (&values[2 * k + 1], &values[2 * k]);
        if *needs_two {
            twos[k] += (&values[2 * k + 1], &slopes[k]);
        }
    }
}

fn values<'b>(slot: &'b Slot) -> &'b [RingElement] {
    if slot.mapped.is_empty() {
        let data = slot.table.values().unwrap();
        &data[2 * slot.base..]
    } else {
        &slot.mapped
    }
}

fn operand<'b>(slot: &'b Slot, point: usize, offset: usize) -> (&'b [RingElement], usize) {
    match point {
        0 => (&values(slot)[2 * offset..], 2),
        1 => (&values(slot)[2 * offset + 1..], 2),
        2 => (&slot.slopes[offset..], 1),
        _ => (&slot.twos[offset..], 1),
    }
}

fn accumulate(
    plan: &Plan,
    slots: &[Slot],
    offset: usize,
    count: usize,
    skip: bool,
    target: &mut [RingElement],
    scratch: &mut [RingElement],
    product: &mut RingElement,
) {
    match plan.slots[..] {
        [a] => {
            for (k, sum) in target.iter_mut().enumerate().filter(|(k, _)| !skip || *k != 1) {
                let (values, step) = operand(&slots[a], k, offset);
                for i in 0..count {
                    *sum += &values[i * step];
                }
            }
        }
        [a, b] => {
            for (k, sum) in target.iter_mut().enumerate().filter(|(k, _)| !skip || *k != 1) {
                let (left, left_step) = operand(&slots[a], k, offset);
                let (right, right_step) = operand(&slots[b], k, offset);
                if count == 1 {
                    *product *= (&left[0], &right[0]);
                    *sum += &*product;
                } else {
                    incomplete_ntt_dot_into(sum, left, left_step, right, right_step, count);
                }
            }
        }
        [a, b, c] => {
            for (k, sum) in target.iter_mut().enumerate().filter(|(k, _)| !skip || *k != 1) {
                let (left, left_step) = operand(&slots[a], k, offset);
                let (middle, middle_step) = operand(&slots[b], k, offset);
                let (right, right_step) = operand(&slots[c], k, offset);
                for i in 0..count {
                    scratch[i] *= (&left[i * left_step], &middle[i * middle_step]);
                }
                incomplete_ntt_dot_into(sum, scratch, 1, right, right_step, count);
            }
        }
        _ => unreachable!(),
    }
}

fn flush(plan: &Plan, accumulator: &mut Accumulator, skip: bool, product: &mut RingElement) {
    if !plan.weighted() || accumulator.key.is_none() {
        return;
    }
    let point = HypercubePoint::new(accumulator.anchor);
    let mut weight: Option<RingElement> = None;
    for value in plan
        .inactive
        .iter()
        .map(|table| &table[point])
        .chain(plan.claims.iter().copied())
    {
        match &mut weight {
            Some(weight) => *weight *= value,
            None => weight = Some(value.clone()),
        }
    }
    for (k, (sum, run)) in accumulator
        .sums
        .iter_mut()
        .zip(accumulator.run.iter_mut())
        .enumerate()
        .filter(|(k, _)| !skip || *k != 1)
    {
        let mut negative = false;
        let mut doubled = 0;
        let mut vanishes = false;
        match &weight {
            Some(weight) => *product *= (&*run, weight),
            None => product.set_from(run),
        }
        for &(bit, claim) in &plan.rounds {
            match (bit, k) {
                (false, 1) | (true, 0) => vanishes = true,
                (false, 2) | (false, 3) => negative = !negative,
                (true, 3) => doubled += 1,
                _ => {}
            }
            *product *= claim;
        }
        for _ in 0..doubled {
            let copy = product.clone();
            *product += &copy;
        }
        if !vanishes {
            if negative {
                *sum -= &*product;
            } else {
                *sum += &*product;
            }
        }
        run.set_zero();
    }
    accumulator.key = None;
}

fn interpolate(degree: usize, sums: &[RingElement], out: &mut Polynomial<RingElement>) {
    let c = &mut out.coefficients;
    match degree {
        1 => {
            c[0].set_from(&sums[0]);
            c[1] -= (&sums[1], &sums[0]);
        }
        2 => {
            c[0].set_from(&sums[0]);
            c[2].set_from(&sums[2]);
            c[1] -= (&sums[1], &sums[0]);
            c[1] -= &sums[2];
        }
        _ => {
            c[0].set_from(&sums[0]);
            c[3].set_from(&sums[2]);
            let mut u = zero();
            u -= (&sums[1], &sums[0]);
            u -= &sums[2];
            let mut eight = zero();
            eight += (&sums[2], &sums[2]);
            let four = eight.clone();
            eight += &four;
            let four = eight.clone();
            eight += &four;
            let mut v = zero();
            v -= (&sums[3], &sums[0]);
            v -= &eight;
            v -= &u;
            v -= &u;
            v *= &RingElement::constant((MOD_Q + 1) / 2, Representation::IncompleteNTT);
            c[2].set_from(&v);
            c[1] -= (&u, &v);
        }
    }
}
