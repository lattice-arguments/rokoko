use crate::{
    common::{
        config::MOD_Q,
        ring_arithmetic::{incomplete_ntt_dot_into, Representation, RingElement},
    },
    protocol::sumcheck_utils::{
        common::{flat_cell, FlatTerm, HighOrderSumcheckData},
        elephant_cell::{CellRef, ElephantCell},
        hypercube_point::HypercubePoint,
        linear::LinearSumcheck,
        polynomial::{add_poly_in_place, Polynomial},
        selector_eq::SelectorEq,
    },
};

type Cell = ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>;
type Node = dyn HighOrderSumcheckData<Element = RingElement>;

struct Term<'c> {
    negative: bool,
    pair: bool,
    tables: Vec<CellRef<'c, Node>>,
    selectors: Vec<CellRef<'c, Node>>,
}

fn hold(term: &FlatTerm<RingElement>) -> Term<'_> {
    Term {
        negative: term.negative,
        pair: term.pair,
        tables: term.tables.iter().map(|cell| cell.get_ref()).collect(),
        selectors: term.selectors.iter().map(|cell| cell.get_ref()).collect(),
    }
}

fn hold_all(terms: &[FlatTerm<RingElement>]) -> Vec<Term<'_>> {
    terms.iter().map(hold).collect()
}

fn table<'b>(node: &'b CellRef<'_, Node>) -> &'b LinearSumcheck<RingElement> {
    node.as_linear().unwrap()
}

fn selector<'b>(node: &'b CellRef<'_, Node>) -> &'b SelectorEq<RingElement> {
    node.as_selector().unwrap()
}

const CHUNK: usize = 32;

pub struct FlatRounds {
    rounds: usize,
    previous: std::cell::RefCell<Vec<Polynomial<RingElement>>>,
    pending: std::cell::RefCell<Vec<(usize, usize, Vec<RingElement>)>>,
}

struct PairPlan<'a> {
    output: usize,
    negative: bool,
    start: usize,
    end: usize,
    slots: Vec<usize>,
    small: Vec<Small<'a>>,
    inactive: Vec<&'a LinearSumcheck<RingElement>>,
    claims: Vec<&'a RingElement>,
    shift: Option<u32>,
    degrees: (usize, usize),
}

impl PairPlan<'_> {
    fn sides(&self) -> (usize, usize) {
        (points(self.degrees.0), points(self.degrees.1))
    }

    fn weighted(&self) -> bool {
        !self.inactive.is_empty() || !self.claims.is_empty()
    }
}

enum Small<'a> {
    Table(&'a LinearSumcheck<RingElement>, (bool, bool)),
    Selector(&'a SelectorEq<RingElement>, (bool, bool)),
}

struct QuadSlot<'a> {
    table: &'a LinearSumcheck<RingElement>,
    quads: usize,
    width: usize,
    sides: (usize, usize),
    corners: Vec<RingElement>,
    grid: [Vec<RingElement>; 16],
    base: usize,
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

fn flatten(cell: &Cell) -> Option<Vec<FlatTerm<RingElement>>> {
    let mut terms = vec![];
    flat_cell(cell, &mut terms).map(|_| terms)
}

impl FlatRounds {
    pub fn new(outputs: &[Cell], total_vars: usize) -> FlatRounds {
        let mut any = false;
        let mut full = true;
        for terms in outputs.iter().filter_map(flatten) {
            for term in hold_all(&terms) {
                any = true;
                full &= term
                    .tables
                    .iter()
                    .map(table)
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
            pending: std::cell::RefCell::new(vec![]),
        }
    }

    pub fn pair_polynomial(
        &self,
        outputs: &[Cell],
        gammas: &[RingElement],
        degrees: (usize, usize),
    ) -> Option<Vec<Polynomial<RingElement>>> {
        let flattened: Vec<Vec<FlatTerm<RingElement>>> = outputs.iter().map(flatten).collect::<Option<_>>()?;
        let flattened: Vec<Vec<Term>> = flattened.iter().map(|terms| hold_all(terms)).collect();
        let mut slots: Vec<QuadSlot> = vec![];
        let mut plans: Vec<PairPlan> = vec![];
        for (output, terms) in flattened.iter().enumerate() {
            for term in terms {
                let plan = pair_plan(output, term, &mut slots)?;
                if plan.degrees.0 > degrees.0 || plan.degrees.1 > degrees.1 {
                    return None;
                }
                plans.push(plan);
            }
        }
        let mut accumulators: Vec<Accumulator> = plans
            .iter()
            .map(|plan| {
                let (s1, s2) = plan.sides();
                Accumulator {
                    sums: vec![zero(); s1 * s2],
                    run: vec![zero(); if plan.weighted() { s1 * s2 } else { 0 }],
                    key: None,
                    anchor: 0,
                }
            })
            .collect();

        let mut bounds: Vec<usize> = plans
            .iter()
            .filter(|plan| plan.start < plan.end)
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
        let mut members: Vec<usize> = vec![];
        for window in bounds.windows(2) {
            let (from, to) = (window[0], window[1]);
            members.clear();
            members.extend((0..plans.len()).filter(|&m| plans[m].start <= from && to <= plans[m].end));
            if members.is_empty() {
                continue;
            }
            for slot in slots.iter_mut() {
                slot.sides = (0, 0);
            }
            for &m in &members {
                let (s1, s2) = plans[m].sides();
                for &slot in &plans[m].slots {
                    let sides = &mut slots[slot].sides;
                    *sides = (sides.0.max(s1), sides.1.max(s2));
                }
            }
            let width = slots
                .iter()
                .filter(|slot| slot.sides.0 > 0)
                .map(|slot| slot.width)
                .fold(CHUNK, usize::min);
            let mut x = from;
            while x < to {
                let y = to.min((x / width + 1) * width);
                for slot in slots.iter_mut().filter(|slot| slot.sides.0 > 0) {
                    prepare_quads(slot, x, y);
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
                            flush_pair(plan, accumulator, &mut product);
                            accumulator.key = Some(key);
                            accumulator.anchor = u;
                        }
                        let target = if plan.weighted() {
                            &mut accumulator.run
                        } else {
                            &mut accumulator.sums
                        };
                        accumulate_pair(plan, &slots, u, u - x, v - u, target, &mut scratch, &mut product);
                        u = v;
                    }
                }
                x = y;
            }
        }
        for (plan, accumulator) in plans.iter().zip(accumulators.iter_mut()) {
            flush_pair(plan, accumulator, &mut product);
        }

        let (rows, columns) = (degrees.0 + 1, degrees.1 + 1);
        let mut totals = vec![zero(); outputs.len() * rows * columns];
        let mut pending = Vec::with_capacity(plans.len());
        for (plan, accumulator) in plans.iter().zip(&accumulators) {
            let (d1, d2) = plan.degrees;
            let mut coefficients = vec![zero(); (d1 + 1) * (d2 + 1)];
            if plan.start < plan.end {
                interpolate_grid(plan.degrees, &accumulator.sums, &mut coefficients);
                let total = &mut totals[plan.output * rows * columns..];
                for i in 0..=d1 {
                    for j in 0..=d2 {
                        let c = &coefficients[i * (d2 + 1) + j];
                        if plan.negative {
                            total[i * columns + j] -= c;
                        } else {
                            total[i * columns + j] += c;
                        }
                    }
                }
            }
            pending.push((d1, d2, coefficients));
        }
        *self.pending.borrow_mut() = pending;
        let mut result: Vec<Polynomial<RingElement>> = (0..rows)
            .map(|_| {
                let mut row = Polynomial::<RingElement>::new(0);
                row.num_coefficients = columns;
                row
            })
            .collect();
        for (output, gamma) in gammas.iter().enumerate().take(outputs.len()) {
            for i in 0..rows {
                for j in 0..columns {
                    product *= (&totals[(output * rows + i) * columns + j], gamma);
                    result[i].coefficients[j] += &product;
                }
            }
        }
        Some(result)
    }

    pub fn fix_first(&self, challenge: &RingElement) {
        let pending = self.pending.take();
        let mut previous = Vec::with_capacity(pending.len());
        for (d1, d2, coefficients) in pending {
            let mut poly = Polynomial::<RingElement>::new(0);
            poly.num_coefficients = d2 + 1;
            for j in 0..=d2 {
                let c = &mut poly.coefficients[j];
                c.set_from(&coefficients[d1 * (d2 + 1) + j]);
                for i in (0..d1).rev() {
                    *c *= challenge;
                    *c += &coefficients[i * (d2 + 1) + j];
                }
            }
            previous.push(poly);
        }
        *self.previous.borrow_mut() = previous;
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
        let flattened: Vec<Option<Vec<FlatTerm<RingElement>>>> = outputs.iter().map(flatten).collect();
        let flattened: Vec<Option<Vec<Term>>> = flattened
            .iter()
            .map(|terms| terms.as_deref().map(hold_all))
            .collect();
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

fn plan<'a>(output: usize, term: &'a Term<'_>, slots: &mut Vec<Slot<'a>>) -> Plan<'a> {
    let mut start = 0;
    let mut end = usize::MAX;
    let mut half = 0;
    let mut active = vec![];
    let mut inactive = vec![];
    let mut shift: Option<u32> = None;
    let mut interleaved = 0;
    let mut claims = vec![];
    let mut rounds = vec![];
    for selector in term.selectors.iter().map(selector) {
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
    for table in term.tables.iter().map(table) {
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

fn pair_plan<'a>(output: usize, term: &'a Term<'_>, slots: &mut Vec<QuadSlot<'a>>) -> Option<PairPlan<'a>> {
    let mut start = 0;
    let mut end = usize::MAX;
    let mut quarter = 0;
    let mut active = vec![];
    let mut small = vec![];
    let mut inactive = vec![];
    let mut claims = vec![];
    let mut shift: Option<u32> = None;
    let (mut d1, mut d2) = (0, 0);
    for selector in term.selectors.iter().map(selector) {
        let (bits, count, total) = selector.window();
        if count == 0 || total - count >= 2 {
            if count > 0 {
                let free = total - count - 2;
                start = start.max(bits << free);
                end = end.min((bits + 1) << free);
            }
            if !selector.is_unit_constant() {
                claims.push(selector.current_claim());
            }
        } else {
            let free = total - count;
            let quad = bits >> (2 - free);
            start = start.max(quad);
            end = end.min(quad + 1);
            let dependence = (free == 0, true);
            d1 += dependence.0 as usize;
            d2 += dependence.1 as usize;
            small.push(Small::Selector(selector, dependence));
        }
    }
    for table in term.tables.iter().map(table) {
        let (len, suffix) = (table.len(), table.suffix());
        if len == 1 || suffix >= 2 {
            if len > 1 {
                let s = suffix as u32 - 2;
                shift = Some(shift.map_or(s, |t| t.min(s)));
            }
            inactive.push(table);
        } else if suffix == 1 || len == 2 {
            let dependence = (suffix == 0, suffix == 1);
            d1 += dependence.0 as usize;
            d2 += dependence.1 as usize;
            small.push(Small::Table(table, dependence));
        } else {
            if let Some((_, to)) = table.non_zero_range() {
                end = end.min(to.div_ceil(2));
            }
            quarter = quarter.max(len / 4);
            let slot = match slots.iter().position(|s| std::ptr::eq(s.table, table)) {
                Some(slot) => slot,
                None => {
                    let width = CHUNK.min(len / 4);
                    slots.push(QuadSlot {
                        table,
                        quads: len / 4,
                        width,
                        sides: (0, 0),
                        corners: if table.values().is_none() {
                            vec![zero(); 4 * width]
                        } else {
                            vec![]
                        },
                        grid: std::array::from_fn(|_| vec![]),
                        base: 0,
                    });
                    slots.len() - 1
                }
            };
            d1 += 1;
            d2 += 1;
            active.push(slot);
        }
    }
    if active.is_empty() || active.len() > 3 || d1 > 3 || d2 > 3 {
        return None;
    }
    Some(PairPlan {
        output,
        negative: term.negative,
        start,
        end: end.min(quarter),
        slots: active,
        small,
        inactive,
        claims,
        shift,
        degrees: (d1, d2),
    })
}

fn prepare_quads(slot: &mut QuadSlot, from: usize, to: usize) {
    let base = from & (slot.quads - 1);
    slot.base = base;
    let count = to - from;
    let QuadSlot {
        table,
        corners,
        grid,
        sides,
        width,
        ..
    } = slot;
    if !corners.is_empty() {
        for k in 0..count {
            let quad = &mut corners[4 * k..4 * k + 4];
            let (low, high) = quad.split_at_mut(2);
            let (c0, c1) = low.split_at_mut(1);
            table.pair_into(2 * (base + k), &mut c0[0], &mut c1[0]);
            let (c2, c3) = high.split_at_mut(1);
            table.pair_into(2 * (base + k) + 1, &mut c2[0], &mut c3[0]);
        }
    }
    let values: &[RingElement] = if corners.is_empty() {
        &table.values().unwrap()[4 * base..]
    } else {
        corners
    };
    let (s1, s2) = *sides;
    let needed = |p1: usize, p2: usize| p1 < s1 && p2 < s2;
    let [_, _, g2, g3, _, _, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15] = grid;
    for (g, used) in [
        (&mut *g2, needed(0, 2) || needed(0, 3)),
        (&mut *g3, needed(0, 3)),
        (&mut *g6, needed(1, 2) || needed(1, 3) || needed(3, 2)),
        (&mut *g7, needed(1, 3)),
        (&mut *g8, needed(2, 0) || needed(3, 0)),
        (&mut *g9, needed(2, 1) || needed(3, 1) || needed(2, 3)),
        (&mut *g10, needed(2, 2)),
        (&mut *g11, needed(2, 3)),
        (&mut *g12, needed(3, 0)),
        (&mut *g13, needed(3, 1)),
        (&mut *g14, needed(3, 2)),
        (&mut *g15, needed(3, 3)),
    ] {
        if used && g.is_empty() {
            *g = vec![zero(); *width];
        }
    }
    for k in 0..count {
        let c = &values[4 * k..4 * k + 4];
        if s1 > 2 {
            g8[k] -= (&c[1], &c[0]);
            g9[k] -= (&c[3], &c[2]);
        }
        if s2 > 2 {
            g2[k] -= (&c[2], &c[0]);
            g6[k] -= (&c[3], &c[1]);
        }
        if s1 > 2 && s2 > 2 {
            g10[k] -= (&g9[k], &g8[k]);
        }
        if s1 > 3 {
            g12[k] += (&c[1], &g8[k]);
            g13[k] += (&c[3], &g9[k]);
        }
        if s2 > 3 {
            g3[k] += (&c[2], &g2[k]);
            g7[k] += (&c[3], &g6[k]);
        }
        if s1 > 3 && s2 > 2 {
            g14[k] += (&g6[k], &g10[k]);
        }
        if s1 > 2 && s2 > 3 {
            g11[k] += (&g9[k], &g10[k]);
        }
        if s1 > 3 && s2 > 3 {
            g15[k] += (&g13[k], &g14[k]);
        }
    }
}

fn quad_operand<'b>(slot: &'b QuadSlot, p1: usize, p2: usize, offset: usize) -> (&'b [RingElement], usize) {
    if p1 < 2 && p2 < 2 {
        let values: &[RingElement] = if slot.corners.is_empty() {
            &slot.table.values().unwrap()[4 * slot.base..]
        } else {
            &slot.corners
        };
        (&values[4 * offset + 2 * p2 + p1..], 4)
    } else {
        (&slot.grid[4 * p1 + p2][offset..], 1)
    }
}

fn along(low: &RingElement, high: &RingElement, point: usize, dependent: bool, out: &mut RingElement) {
    match point {
        0 => out.set_from(low),
        1 => out.set_from(high),
        2 if dependent => *out -= (high, low),
        2 => out.set_from(low),
        _ => {
            *out -= (high, low);
            *out += high;
        }
    }
}

fn small_corners(small: &Small, quad: usize, corners: &mut [RingElement; 4]) -> (bool, bool) {
    match small {
        Small::Table(table, dependence) => {
            let [c0, c1, c2, c3] = corners;
            if dependence.0 {
                table.pair_into(0, c0, c1);
                c2.set_from(c0);
                c3.set_from(c1);
            } else {
                table.pair_into(quad, c0, c2);
                c1.set_from(c0);
                c3.set_from(c2);
            }
            *dependence
        }
        Small::Selector(selector, dependence) => {
            let (bits, count, total) = selector.window();
            let free = total - count;
            let low = bits & ((1 << (2 - free)) - 1);
            for (j, corner) in corners.iter_mut().enumerate() {
                if j >> free == low {
                    corner.set_from(selector.current_claim());
                } else {
                    corner.set_zero();
                }
            }
            *dependence
        }
    }
}

fn accumulate_pair(
    plan: &PairPlan,
    slots: &[QuadSlot],
    first: usize,
    offset: usize,
    count: usize,
    target: &mut [RingElement],
    scratch: &mut [RingElement],
    product: &mut RingElement,
) {
    let (s1, s2) = plan.sides();
    if plan.small.is_empty() {
        for p1 in 0..s1 {
            for p2 in 0..s2 {
                let sum = &mut target[p1 * s2 + p2];
                match plan.slots[..] {
                    [a] => {
                        let (values, step) = quad_operand(&slots[a], p1, p2, offset);
                        for i in 0..count {
                            *sum += &values[i * step];
                        }
                    }
                    [a, b] => {
                        let (left, left_step) = quad_operand(&slots[a], p1, p2, offset);
                        let (right, right_step) = quad_operand(&slots[b], p1, p2, offset);
                        if count == 1 {
                            *product *= (&left[0], &right[0]);
                            *sum += &*product;
                        } else {
                            incomplete_ntt_dot_into(sum, left, left_step, right, right_step, count);
                        }
                    }
                    [a, b, c] => {
                        let (left, left_step) = quad_operand(&slots[a], p1, p2, offset);
                        let (middle, middle_step) = quad_operand(&slots[b], p1, p2, offset);
                        let (right, right_step) = quad_operand(&slots[c], p1, p2, offset);
                        for i in 0..count {
                            scratch[i] *= (&left[i * left_step], &middle[i * middle_step]);
                        }
                        incomplete_ntt_dot_into(sum, scratch, 1, right, right_step, count);
                    }
                    _ => unreachable!(),
                }
            }
        }
        return;
    }
    let mut corners: [RingElement; 4] = std::array::from_fn(|_| zero());
    let (mut low, mut high) = (zero(), zero());
    let mut grids: Vec<Vec<RingElement>> = vec![vec![zero(); s1 * s2]; plan.small.len()];
    for i in 0..count {
        for (small, grid) in plan.small.iter().zip(grids.iter_mut()) {
            let dependence = small_corners(small, first + i, &mut corners);
            for p1 in 0..s1 {
                along(&corners[0], &corners[1], p1, dependence.0, &mut low);
                along(&corners[2], &corners[3], p1, dependence.0, &mut high);
                for p2 in 0..s2 {
                    along(&low, &high, p2, dependence.1, &mut grid[p1 * s2 + p2]);
                }
            }
        }
        for p1 in 0..s1 {
            for p2 in 0..s2 {
                let mut started = false;
                for &slot in &plan.slots {
                    let (values, step) = quad_operand(&slots[slot], p1, p2, offset);
                    if started {
                        *product *= &values[i * step];
                    } else {
                        product.set_from(&values[i * step]);
                        started = true;
                    }
                }
                for grid in &grids {
                    *product *= &grid[p1 * s2 + p2];
                }
                target[p1 * s2 + p2] += &*product;
            }
        }
    }
}

fn flush_pair(plan: &PairPlan, accumulator: &mut Accumulator, product: &mut RingElement) {
    if !plan.weighted() || accumulator.key.is_none() {
        return;
    }
    let point = HypercubePoint::new(2 * accumulator.anchor);
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
    let weight = weight.unwrap();
    for (sum, run) in accumulator.sums.iter_mut().zip(accumulator.run.iter_mut()) {
        *product *= (&*run, &weight);
        *sum += &*product;
        run.set_zero();
    }
    accumulator.key = None;
}

fn interpolate_grid(degrees: (usize, usize), values: &[RingElement], out: &mut [RingElement]) {
    let (d1, d2) = degrees;
    let (s1, s2) = (points(d1), points(d2));
    let mut rows = vec![zero(); s1 * (d2 + 1)];
    let mut line = Polynomial::<RingElement>::new(0);
    for p1 in 0..s1 {
        interpolate(d2, &values[p1 * s2..p1 * s2 + s2], &mut line);
        for j in 0..=d2 {
            rows[p1 * (d2 + 1) + j].set_from(&line.coefficients[j]);
        }
    }
    let mut column = vec![zero(); s1];
    for j in 0..=d2 {
        for p1 in 0..s1 {
            column[p1].set_from(&rows[p1 * (d2 + 1) + j]);
        }
        interpolate(d1, &column, &mut line);
        for i in 0..=d1 {
            out[i * (d2 + 1) + j].set_from(&line.coefficients[i]);
        }
    }
}
