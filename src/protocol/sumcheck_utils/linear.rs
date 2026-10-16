use std::{
    cell::RefCell,
    ops::{Deref, Index},
    sync::Arc,
};

use crate::{
    common::{
        arithmetic::field_to_ring_element_into,
        ring_arithmetic::{FieldExtension, Representation, RingElement},
        structured_row::StructuredRow,
        sumcheck_element::SumcheckElement,
    },
    protocol::sumcheck_utils::{
        common::{EvaluationSumcheckData, FlatTerm, HighOrderSumcheckData, SumcheckBaseData},
        elephant_cell::ElephantCell,
        hypercube_point::HypercubePoint,
        polynomial::Polynomial,
    },
};

#[cfg(test)]
use crate::common::{config::MOD_Q, structured_row::PreprocessedRow};

enum Source<E: 'static> {
    Caller(&'static [E]),
    Shared(Arc<Vec<E>>),
}

impl<E> Deref for Source<E> {
    type Target = [E];

    fn deref(&self) -> &[E] {
        match self {
            Source::Caller(data) => data,
            Source::Shared(data) => data,
        }
    }
}

struct Borrowed<E: 'static> {
    data: Source<E>,
    map: Option<fn(&E, &mut E)>,
    conjugate: bool,
}

const BORROW_MIN_LEN: usize = 1024;

#[cfg(any(test, feature = "unsafe-sumcheck"))]
fn ring_conjugate(x: &RingElement, out: &mut RingElement) {
    x.conjugate_into(out)
}

#[cfg(any(test, feature = "unsafe-sumcheck"))]
impl LinearSumcheck<RingElement> {
    pub(crate) unsafe fn conjugate_of(data: &[RingElement], non_zero_end: usize) -> Self {
        let mut table = unsafe { Self::from_borrowed(data, non_zero_end, Some(ring_conjugate)) };
        if let Some(borrowed) = &mut table.borrowed {
            borrowed.conjugate = true;
        }
        table
    }
}

/// Standard linear sumcheck over a vector that represents a multilinear extension.
pub struct LinearSumcheck<E: SumcheckElement = RingElement> {
    pub data: Vec<E>,
    borrowed: Option<Borrowed<E>>,
    variable_count: usize,
    index_mask: usize,
    suffix: usize,
    poly_scratch: RefCell<Polynomial<E>>,
    // Index past the last non-zero entry in `data`. Everything in
    // data[non_zero_end..] is guaranteed zero. Used to skip work in
    // partial_evaluate, non_zero_range, and batched Karatsuba.
    non_zero_end: usize,
    viewed: bool,
}

impl<E: SumcheckElement> LinearSumcheck<E> {
    pub fn new(count: usize) -> Self {
        Self::new_with_prefixed_sufixed_data(count, 0, 0)
    }

    // here we want to handle the case when
    // we have f(x_0, x_1, ..., x_{n-1}, y_0, y_1, ..., y_{m-1}) = MLE[data](y_0, y_1, ..., y_{m-1})
    // so we need to prefix the variable count with n extra variables
    // This will be used if we have a large data, but our claim is only on a subset of the variables
    // for example, if we have a data of size 2^{10} but our claim is only on 2^{6} points,
    // Then, imagine we have a claim <s, subdata> = v, where subdata is of size 2^{6}
    // Since we want to run larger sumchecks on the full data, we need to embed subdata into data
    // so we have \sum_{z \in HC^10} MLE[data](z) * eq(x_0, ... x_3, SEL) * MLE[s](x_4 ... x_9) = v
    // where SEL is the selection vector that selects the subdata from data.
    // to conveniently operate we will assume that MLE[s](x_4 ... x_9) is prefixed with 4 dummy variables x_0 ... x_3
    pub fn new_with_prefixed_sufixed_data(
        count: usize,
        prefix_size: usize, // number of extra most significant dummy variables
        suffix_size: usize, // number of extra least significant dummy variables
    ) -> Self {
        LinearSumcheck {
            data: E::allocate_zero_vec(count),
            borrowed: None,
            variable_count: count.ilog2() as usize + prefix_size + suffix_size,
            index_mask: count - 1, // this mask will be used to ignore prefixed variables
            poly_scratch: RefCell::new(Polynomial::new(2)),
            suffix: suffix_size,
            non_zero_end: count,
            viewed: false,
        }
    }
    pub fn from_data(data: Vec<E>) -> Self {
        Self::from_data_with_prefixed_sufixed_data(data, 0, 0)
    }

    pub fn from_data_with_prefixed_sufixed_data(
        data: Vec<E>,
        prefix_size: usize,
        suffix_size: usize,
    ) -> Self {
        let count = data.len();
        LinearSumcheck {
            data,
            borrowed: None,
            variable_count: count.ilog2() as usize + prefix_size + suffix_size,
            index_mask: count - 1,
            poly_scratch: RefCell::new(Polynomial::new(2)),
            suffix: suffix_size,
            non_zero_end: count,
            viewed: false,
        }
    }

    pub fn from_data_with_non_zero_end(data: Vec<E>, non_zero_end: usize) -> Self {
        let mut table = Self::from_data(data);
        table.non_zero_end = non_zero_end.min(table.data.len());
        table
    }

    pub unsafe fn from_borrowed(
        data: &[E],
        non_zero_end: usize,
        map: Option<fn(&E, &mut E)>,
    ) -> Self {
        let count = data.len();
        if count < BORROW_MIN_LEN {
            let owned = match map {
                None => data.to_vec(),
                Some(map) => data
                    .iter()
                    .map(|x| {
                        let mut out = E::zero();
                        map(x, &mut out);
                        out
                    })
                    .collect(),
            };
            return Self::from_data_with_non_zero_end(owned, non_zero_end);
        }
        LinearSumcheck {
            data: Vec::new(),
            borrowed: Some(Borrowed {
                data: Source::Caller(unsafe { std::mem::transmute::<&[E], &'static [E]>(data) }),
                map,
                conjugate: false,
            }),
            variable_count: count.ilog2() as usize,
            index_mask: count - 1,
            poly_scratch: RefCell::new(Polynomial::new(2)),
            suffix: 0,
            non_zero_end: non_zero_end.min(count),
            viewed: false,
        }
    }

    pub(crate) fn len(&self) -> usize {
        match &self.borrowed {
            Some(borrowed) => borrowed.data.len(),
            None => self.data.len(),
        }
    }

    pub(crate) fn values(&self) -> Option<&[E]> {
        match &self.borrowed {
            None => Some(self.data.as_slice()),
            Some(Borrowed {
                data, map: None, ..
            }) => Some(data),
            Some(_) => None,
        }
    }

    fn own_shared(&mut self) {
        match self.borrowed.take() {
            Some(Borrowed {
                data: Source::Shared(data),
                ..
            }) => self.data = Arc::unwrap_or_clone(data),
            borrowed => self.borrowed = borrowed,
        }
    }

    pub fn partial_evaluate_with_mapped(&mut self, mapped: &mut Self, value: &E) -> bool {
        let (other, map, conjugate) = match &mapped.borrowed {
            Some(Borrowed {
                data,
                map: Some(map),
                conjugate,
            }) => (&**data as *const [E], *map, *conjugate),
            _ => return false,
        };
        let Some(values) = self.values() else {
            return false;
        };
        let shared = std::ptr::eq(values, other)
            && self.non_zero_end == mapped.non_zero_end
            && self.suffix == 0
            && mapped.suffix == 0;
        if !shared || values.len() < 2 {
            return false;
        }
        let fixed = conjugate && {
            let mut image = E::zero();
            map(value, &mut image);
            image == *value
        };
        if fixed && values.len() / 2 >= BORROW_MIN_LEN {
            mapped.borrowed = None;
            self.own_shared();
            self.viewed = false;
            self.partial_evaluate(value);
            let folded = Arc::new(std::mem::take(&mut self.data));
            self.borrowed = Some(Borrowed {
                data: Source::Shared(folded.clone()),
                map: None,
                conjugate: false,
            });
            mapped.borrowed = Some(Borrowed {
                data: Source::Shared(folded),
                map: Some(map),
                conjugate,
            });
            mapped.non_zero_end = self.non_zero_end;
            mapped.variable_count -= 1;
            self.viewed = true;
            return true;
        }
        let Some(own) = &self.borrowed else {
            mapped.partial_evaluate(value);
            self.viewed = false;
            self.partial_evaluate(value);
            return true;
        };
        let half = own.data.len() / 2;
        let fold_end = ((self.non_zero_end + 1) / 2).min(half);
        let (mut first, mut second) = (Vec::with_capacity(half), Vec::with_capacity(half));
        let (mut even, mut odd) = (E::zero(), E::zero());
        for pair in own.data.chunks_exact(2).take(fold_end) {
            first.push(E::zero());
            let out = first.last_mut().unwrap();
            *out -= (&pair[1], &pair[0]);
            *out *= value;
            *out += &pair[0];
            map(&pair[0], &mut even);
            map(&pair[1], &mut odd);
            second.push(E::zero());
            let out = second.last_mut().unwrap();
            *out -= (&odd, &even);
            *out *= value;
            *out += &even;
        }
        self.viewed = false;
        self.install_fold(first, half, fold_end);
        mapped.install_fold(second, half, fold_end);
        true
    }

    fn install_fold(&mut self, mut folded: Vec<E>, half: usize, fold_end: usize) {
        folded.resize(half, E::zero());
        self.borrowed = None;
        self.data = folded;
        self.non_zero_end = fold_end;
        self.variable_count -= 1;
    }

    pub(crate) fn is_folding(&self) -> bool {
        self.suffix == 0 && self.len() > 1
    }

    pub(crate) fn suffix(&self) -> usize {
        self.suffix
    }

    pub(crate) fn pair_into(&self, point: usize, even: &mut E, odd: &mut E) {
        let p = point & (self.len() / 2 - 1);
        self.load(2 * p, even);
        self.load(2 * p + 1, odd);
    }

    fn load(&self, index: usize, out: &mut E) {
        match &self.borrowed {
            Some(Borrowed {
                data,
                map: Some(map),
                ..
            }) => map(&data[index], out),
            _ => out.set_from(&self.values().unwrap()[index]),
        }
    }

    /// Populate the internal buffer with the provided values.
    /// Marks the whole buffer as potentially non-zero.
    pub fn load_from(&mut self, src: &[E]) {
        assert_eq!(
            src.len(),
            self.data.len(),
            "load_from length mismatch: buffer {} (vars {}, suffix {}), source {}",
            self.data.len(),
            self.variable_count,
            self.suffix,
            src.len()
        );
        self.data.clone_from_slice(src);
        self.non_zero_end = self.data.len();
    }

    /// Load data and explicitly set the non-zero boundary. The caller
    /// guarantees that src[non_zero_end..] is all zero. Avoids the
    /// backward scan when the boundary is already known (e.g. from
    /// the config's usage count).
    pub fn load_from_with_non_zero_end(&mut self, src: &[E], non_zero_end: usize) {
        assert_eq!(
            src.len(),
            self.data.len(),
            "Source data length must match the sumcheck data length, expected {}, got {}",
            self.data.len(),
            src.len()
        );
        assert!(
            non_zero_end <= src.len(),
            "non_zero_end ({}) must not exceed source length ({})",
            non_zero_end,
            src.len()
        );
        self.data.clone_from_slice(src);
        self.non_zero_end = non_zero_end;
    }

    pub fn load_from_at_index(&mut self, src: &[E], index: usize) {
        let len = src.len();
        self.data[index..index + len].clone_from_slice(src);
    }
}

impl<E: SumcheckElement> Index<HypercubePoint> for LinearSumcheck<E> {
    type Output = E;

    fn index(&self, index: HypercubePoint) -> &Self::Output {
        // LS-first indexing: the current variable being folded is the LS bit.
        // Even indices = value at 0, odd indices = value at 1.
        // A HypercubePoint p identifies one pair within the data array.
        // When suffix > 0, we're still in dummy rounds (suffix vars are LS,
        // folded first, so they're consumed before data).  During suffix
        // rounds the data doesn't split, so we just mask the prefix away.
        //
        // During data rounds the point selects which even/odd pair we look at.
        // self[p] returns data[2*p_masked] (the val-at-0 side of the pair).
        let data = self
            .values()
            .expect("a mapped table has no stored entries to index");
        if data.len() == 1 {
            return &data[0];
        }
        if self.suffix > 0
            || self.variable_count > data.len().trailing_zeros() as usize + self.suffix
        {
            // Prefix/suffix round — data isn't being split.
            // For suffix rounds under LS-first, the current round consumes
            // one suffix bit now, so the half-hypercube point still contains
            // the remaining (suffix - 1) suffix bits.
            let suffix_shift = if self.suffix > 0 { self.suffix - 1 } else { 0 };
            let index_shifted = index.shifted(suffix_shift);
            let index_masked = index_shifted.masked(self.index_mask);
            return &data[index_masked.coordinates];
        }
        // Data round, LS-first: point p → even slot 2p.
        let p = index.masked(data.len() / 2 - 1).coordinates;
        &data[2 * p]
    }
}

impl<E: SumcheckElement> HighOrderSumcheckData for LinearSumcheck<E> {
    type Element = E;

    fn gadget_span(&self) -> tracing::Span {
        tracing::trace_span!("sumcheck::gadget::linear")
    }

    fn get_scratch_poly(&self) -> &RefCell<Polynomial<E>> {
        &self.poly_scratch
    }
    fn max_num_polynomial_coefficients(&self) -> usize {
        2
    }

    #[inline]
    fn constant_univariate_polynomial_at_point_available_by_ref(
        &self,
        point: HypercubePoint,
    ) -> Option<&Self::Element> {
        // LS-first: suffix variables are folded first. While suffix > 0,
        // we're in a dummy LS round — the polynomial is constant (just
        // the data value at this point).
        if self.suffix > 0 {
            return Some(&self[point]);
        }

        if self.len() == 1 {
            // All data variables consumed; remaining are prefix dummies.
            return Some(&self[point]);
        }

        None
    }

    #[inline]
    fn univariate_polynomial_at_point_into(
        &self,
        point: HypercubePoint,
        polynomial: &mut Polynomial<E>,
    ) {
        // LS-first: current variable is the least-significant (bit 0 of index).
        // For data round: data[2p] = value@0, data[2p+1] = value@1.

        // Suffix round (LS dummies, folded first under LS-first): constant.
        if self.suffix > 0 {
            polynomial.coefficients[0].set_from(&self[point]);
            polynomial.num_coefficients = 1;
            return;
        }

        // Data is fully folded but prefix variables remain: constant.
        if self.len() == 1 {
            self.load(0, &mut polynomial.coefficients[0]);
            polynomial.num_coefficients = 1;
            return;
        }

        // Data round: pair at (2p, 2p+1).
        let p = point.masked(self.len() / 2 - 1).coordinates;
        let idx0 = 2 * p;
        let idx1 = 2 * p + 1;
        let (low, high) = polynomial.coefficients.split_at_mut(1);
        match self.values() {
            Some(data) => {
                low[0].set_from(&data[idx0]);
                high[0] -= (&data[idx1], &data[idx0]);
            }
            None => {
                self.load(idx0, &mut low[0]);
                self.load(idx1, &mut high[0]);
                high[0] -= &low[0];
            }
        }
        polynomial.num_coefficients = 2;
    }

    fn flat_terms<'a>(&'a self, out: &mut Vec<FlatTerm<'a, E>>) -> Option<usize> {
        out.push(FlatTerm {
            negative: false,
            leaf: true,
            pair: false,
            tables: vec![self],
            selectors: vec![],
        });
        Some(1)
    }

    fn linear_pair_at_point(&self, point: HypercubePoint) -> Option<(&E, &E)> {
        if self.suffix > 0 || self.len() <= 1 {
            return None;
        }
        let data = self.values()?;
        let p = point.masked(data.len() / 2 - 1).coordinates;
        Some((&data[2 * p], &data[2 * p + 1]))
    }

    fn is_univariate_polynomial_zero_at_point(&self, point: HypercubePoint) -> bool {
        // For suffix/prefix rounds we intentionally expose no sparse range.
        if self.suffix > 0 || self.len() <= 1 {
            return false;
        }
        // Sparse tail tracking is only valid when data fills the full hypercube.
        if self.len() != (1usize << self.variable_count) {
            return false;
        }
        let half = self.len() / 2;
        let pair_nz = (self.non_zero_end + 1) / 2;
        if pair_nz >= half {
            return false;
        }
        let p = point.masked(half - 1).coordinates;
        p >= pair_nz
    }

    fn variable_count(&self) -> usize {
        self.variable_count
    }

    fn dependence_window_vars(&self) -> usize {
        if self.suffix > 0 {
            return self.suffix - 1 + self.len().ilog2() as usize;
        }
        if self.len() <= 1 {
            return 0;
        }
        self.len().ilog2() as usize - 1
    }

    fn as_data_slices(&self) -> Option<(&[Self::Element], &[Self::Element])> {
        // Under LS-first, data is interleaved: even indices = val@0, odd = val@1.
        // We can't provide two contiguous halves. Return None so callers
        // (ProductSumcheck Karatsuba) fall back to the per-point path.
        None
    }

    fn as_interleaved_data(&self) -> Option<&[Self::Element]> {
        // Only expose interleaved data when the data fills the full hypercube
        // for the current variable count (no prefix/suffix remaining).
        // For sumchecks with prefix variables, data.len() < 2^variable_count
        // during early rounds, so this correctly returns None.
        if self.suffix > 0 {
            return None;
        }
        if self.len() <= 1 {
            return None;
        }
        // data.len() must equal 2^variable_count (no pending prefix variables).
        if self.len() != (1usize << self.variable_count) {
            return None;
        }
        let nz_end = self.non_zero_end.min(self.len());
        self.values().map(|data| &data[..nz_end])
    }

    fn non_zero_range(&self) -> Option<(usize, usize)> {
        // LS-first: during suffix rounds or prefix rounds, no meaningful range.
        if self.suffix > 0 {
            return None;
        }
        if self.len() <= 1 {
            return None;
        }
        // Only valid when data fills the full hypercube (no prefix active).
        if self.len() != (1usize << self.variable_count) {
            return None;
        }
        let half = self.len() / 2;
        // non_zero_end tracks pairs: pair i covers data[2i] and data[2i+1].
        // Points run over [0, half), and pair i is non-zero if 2i < non_zero_end,
        // i.e. i < ceil(non_zero_end / 2).
        let pair_nz = (self.non_zero_end + 1) / 2;
        if pair_nz < half {
            Some((0, pair_nz))
        } else {
            None
        }
    }

    fn final_evaluations_test_only(&self) -> Self::Element {
        if self.len() != 1 {
            panic!("Sumcheck is not fully evaluated yet");
        }
        let mut value = E::zero();
        self.load(0, &mut value);
        value
    }
}

impl<E: SumcheckElement> SumcheckBaseData for LinearSumcheck<E> {
    fn partial_evaluate(&mut self, value: &E) {
        // LS-first folding: suffix (LS dummies) are consumed first,
        // then actual data variables, then prefix (MS dummies) last.

        // Suffix round: LS dummy variable, data is constant across it.
        if self.suffix > 0 {
            self.variable_count -= 1;
            self.suffix -= 1;
            return;
        }

        // Prefix round: MS dummy variable, data is constant across it.
        if self.len() == 1 {
            self.variable_count -= 1;
            return;
        }

        // Data round: fold even/odd pairs.
        // data[2i] = value at current_var=0, data[2i+1] = value at current_var=1.
        // folded[i] = data[2i] + (data[2i+1] - data[2i]) * r
        let n = self.len();
        if n % 2 != 0 {
            panic!("Sumcheck data length must be a power of 2");
        }
        let half = n / 2;

        // even_nz: number of folded output slots that might be non-zero.
        // Pair i has even=data[2i], odd=data[2i+1]. If 2i >= non_zero_end
        // then both are zero → folded[i] is zero.
        let pair_nz = (self.non_zero_end + 1) / 2; // ceil(non_zero_end / 2)
        let fold_end = pair_nz.min(half);

        assert!(
            !self.viewed,
            "a table viewed through its conjugate folds only with the view"
        );
        if let Some(borrowed) = self.borrowed.take() {
            let mut folded = Vec::with_capacity(half);
            match borrowed.map {
                None => {
                    for pair in borrowed.data.chunks_exact(2).take(fold_end) {
                        folded.push(E::zero());
                        let out = folded.last_mut().unwrap();
                        *out -= (&pair[1], &pair[0]);
                        *out *= value;
                        *out += &pair[0];
                    }
                }
                Some(map) => {
                    let mut even = E::zero();
                    for pair in borrowed.data.chunks_exact(2).take(fold_end) {
                        folded.push(E::zero());
                        let out = folded.last_mut().unwrap();
                        map(&pair[0], &mut even);
                        map(&pair[1], out);
                        *out -= &even;
                        *out *= value;
                        *out += &even;
                    }
                }
            }
            self.install_fold(folded, half, fold_end);
        } else {
            if fold_end > 0 {
                let (left, right) = self.data.split_at_mut(1);
                right[0] -= &left[0];
                right[0] *= value;
                left[0] += &right[0];
            }
            for i in 1..fold_end {
                let (head, tail) = self.data.split_at_mut(2 * i);
                let out = &mut head[i];
                *out -= (&tail[1], &tail[0]);
                *out *= value;
                *out += &tail[0];
            }
            for i in fold_end..half {
                self.data[i].set_zero();
            }

            self.data.truncate(half);
            self.non_zero_end = fold_end;
            self.variable_count -= 1;
        }
    }

    fn final_evaluations(&self) -> &E {
        if self.data.len() != 1 {
            panic!("Sumcheck is not fully evaluated yet");
        }
        &self.data[0]
    }
}

pub struct BasicEvaluationLinearSumcheck<E: SumcheckElement = RingElement> {
    pub data: Vec<E>,
    variable_count: usize,
    #[allow(dead_code)]
    index_mask: usize,
    suffix: usize,
    evaluated: bool,
}

impl<E: SumcheckElement> BasicEvaluationLinearSumcheck<E> {
    pub fn new(count: usize) -> Self {
        Self::new_with_prefixed_sufixed_data(count, 0, 0)
    }

    pub fn new_with_prefixed_sufixed_data(
        count: usize,
        prefix_size: usize,
        suffix_size: usize,
    ) -> Self {
        BasicEvaluationLinearSumcheck {
            data: E::allocate_zero_vec(count),
            variable_count: count.ilog2() as usize + prefix_size + suffix_size,
            index_mask: count - 1,
            suffix: suffix_size,
            evaluated: false,
        }
    }

    pub fn load_from(&mut self, src: &[E]) {
        self.data.clone_from_slice(src);
    }
}

impl<E: SumcheckElement> EvaluationSumcheckData for BasicEvaluationLinearSumcheck<E> {
    type Element = E;

    fn evaluate(&mut self, point: &Vec<Self::Element>) -> &Self::Element {
        // If already evaluated, return cached result
        if self.evaluated {
            return &self.data[0];
        }

        // LS-first evaluation: suffix vars first, then data vars, then prefix vars.
        if point.len() != self.variable_count {
            panic!("Point has incorrect number of variables");
        }

        let data_variable_count = self.data.len().ilog2() as usize;
        let mut current_len = self.data.len();
        let data_point = &point[self.suffix..self.suffix + data_variable_count];

        // Keep MS-style in-memory folding, but consume LS-ordered challenges by reversing.
        for r in data_point.iter().rev() {
            let half = current_len / 2;
            if half > 0 {
                let (left, right) = self.data[..current_len].split_at_mut(half);
                for i in 0..half {
                    right[i] -= &left[i];
                    right[i] *= r;
                    left[i] += &right[i];
                }
            }
            current_len = half;
        }

        // After all folds, data[0] contains the evaluation
        self.evaluated = true;
        &self.data[0]
    }
}

pub struct RingToFieldWrapperEvaluation {
    field_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = FieldExtension>>,
    result: RingElement,
    evaluated: bool,
}

impl RingToFieldWrapperEvaluation {
    pub fn new(
        field_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = FieldExtension>>,
    ) -> Self {
        RingToFieldWrapperEvaluation {
            field_evaluation,
            result: RingElement::zero(Representation::IncompleteNTT),
            evaluated: false,
        }
    }
}

impl EvaluationSumcheckData for RingToFieldWrapperEvaluation {
    type Element = RingElement;

    fn evaluate(&mut self, point: &Vec<Self::Element>) -> &Self::Element {
        if self.evaluated {
            return &self.result;
        }
        let point_field: Vec<FieldExtension> = point.iter().map(|r| r.slot_zero()).collect();

        // Evaluate the field evaluation at the converted point
        field_to_ring_element_into(
            &mut self.result,
            self.field_evaluation.borrow_mut().evaluate(&point_field),
        );
        self.result
            .from_homogenized_field_extensions_to_incomplete_ntt();
        self.evaluated = true;
        &self.result
    }
}

pub struct StructuredRowEvaluationLinearSumcheck<E: SumcheckElement = RingElement> {
    pub data: Option<StructuredRow<E>>,
    variable_count: usize,
    suffix: usize,
    prefix: usize,
    /// Constant the tensor product is multiplied by. Restricting a tensor row to a dyadic slice
    /// fixes its leading layers, and what they contribute is exactly this scalar.
    scale: Option<E>,
    result: E,
    scratch: E,
    slope: E,
    evaluated: bool,
}

impl<E: SumcheckElement> StructuredRowEvaluationLinearSumcheck<E> {
    pub fn new(count: usize) -> Self {
        Self::new_with_prefixed_sufixed_data(count, 0, 0)
    }

    pub fn new_with_prefixed_sufixed_data(
        count: usize,
        prefix_size: usize,
        suffix_size: usize,
    ) -> Self {
        StructuredRowEvaluationLinearSumcheck {
            data: None,
            variable_count: count.ilog2() as usize + prefix_size + suffix_size,
            suffix: suffix_size,
            prefix: prefix_size,
            scale: None,
            result: E::one(),
            scratch: E::zero(),
            slope: E::zero(),
            evaluated: false,
        }
    }

    pub fn load_from(&mut self, src: StructuredRow<E>) {
        self.load_normalized(src, None);
    }

    pub fn load_scaled_from(&mut self, src: StructuredRow<E>, scale: E) {
        self.load_normalized(src, Some(scale));
    }

    fn load_normalized(&mut self, src: StructuredRow<E>, scale: Option<E>) {
        debug_assert!(src.tensor_layers.len() == self.variable_count - self.suffix - self.prefix);
        // Normalize once to LS-first so evaluate() can zip directly with LS-first challenges.
        let mut normalized = src;
        normalized.tensor_layers.reverse();
        self.data = Some(normalized);
        self.scale = scale;
    }
}

impl<E: SumcheckElement + 'static> EvaluationSumcheckData
    for StructuredRowEvaluationLinearSumcheck<E>
{
    type Element = E;

    fn evaluate(&mut self, point: &Vec<Self::Element>) -> &Self::Element {
        if self.evaluated {
            return &self.result;
        }
        if point.len() != self.variable_count {
            panic!(
                "Point has incorrect number of variables, expected {}, got {}",
                self.variable_count,
                point.len()
            );
        }

        let data = self.data.as_ref().expect("Data not loaded");
        let data_variable_count = data.tensor_layers.len();

        // LS-first: point layout is [suffix..., data..., prefix...]
        // Data challenges are at indices [self.suffix .. self.suffix + data_variable_count).
        // Both tensor_layers and data_point are in the same LS-first order,
        // so layer[k] pairs with data_point[k] directly.
        let data_point = &point[self.suffix..self.suffix + data_variable_count];

        // tensor_layers were normalized to LS-first in load_from(),
        // so layer[k] pairs directly with data_point[k].
        let mut first = true;
        for (layer, r) in data.tensor_layers.iter().zip(data_point.iter()) {
            // (1 - layer)(1 - r) + layer r = (1 - r) + layer (2r - 1)
            self.slope += (r, r);
            self.slope -= E::one_ref();
            self.scratch *= (layer, &self.slope);
            self.scratch += E::one_ref();
            self.scratch -= r;

            match (first, &self.scale) {
                (true, Some(scale)) => self.result *= (scale, &self.scratch),
                (true, None) => self.result.set_from(&self.scratch),
                (false, _) => self.result *= &self.scratch,
            }
            first = false;
        }
        if first {
            self.result
                .set_from(self.scale.as_ref().unwrap_or(E::one_ref()));
        }

        self.evaluated = true;
        &self.result
    }
}

pub struct FakeEvaluationLinearSumcheck<E: SumcheckElement = RingElement> {
    result: E,
}

impl<E: SumcheckElement> FakeEvaluationLinearSumcheck<E> {
    pub fn new() -> Self {
        FakeEvaluationLinearSumcheck { result: E::zero() }
    }

    pub fn set_result(&mut self, result: E) {
        self.result = result;
    }
}

impl<E: SumcheckElement> EvaluationSumcheckData for FakeEvaluationLinearSumcheck<E> {
    type Element = E;

    fn evaluate(&mut self, _point: &Vec<Self::Element>) -> &Self::Element {
        &self.result
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_are_send() {
        fn send<T: Send>() {}
        send::<LinearSumcheck<RingElement>>();
        send::<LinearSumcheck<FieldExtension>>();
    }

    fn conjugate(x: &RingElement, out: &mut RingElement) {
        x.conjugate_into(out)
    }

    fn snapshot(table: &LinearSumcheck<RingElement>) -> (usize, usize, usize, Vec<RingElement>) {
        let entries = (0..table.len())
            .map(|i| {
                let mut x = RingElement::zero(Representation::IncompleteNTT);
                table.load(i, &mut x);
                x
            })
            .collect();
        (table.variable_count, table.suffix, table.non_zero_end, entries)
    }

    #[test]
    fn test_fused_fold_matches_two_folds() {
        crate::common::init_common();
        let n = 2048;
        let data = crate::common::sampling::sample_random_short_vector(n, 60, Representation::IncompleteNTT);
        let challenges =
            crate::common::sampling::sample_random_short_vector(16, 1 << 20, Representation::IncompleteNTT);
        let map: fn(&RingElement, &mut RingElement) = conjugate;
        let table = |map: Option<fn(&RingElement, &mut RingElement)>, end: usize, prefix: usize, suffix: usize| {
            let mut table = unsafe { LinearSumcheck::from_borrowed(&data, end, map) };
            table.variable_count += prefix + suffix;
            table.suffix = suffix;
            table
        };
        for (end, prefix, suffix) in [(n, 0, 0), (n / 2 + 3, 0, 0), (777, 2, 0), (n, 0, 1), (n / 3, 1, 2)] {
            let (mut a, mut b) = (table(None, end, prefix, suffix), table(Some(map), end, prefix, suffix));
            let (mut c, mut d) = (table(None, end, prefix, suffix), table(Some(map), end, prefix, suffix));
            let mut fused_rounds = 0;
            for r in &challenges[..a.variable_count] {
                let eligible = a.borrowed.is_some() && b.borrowed.is_some() && a.suffix == 0;
                let fused = a.partial_evaluate_with_mapped(&mut b, r);
                assert_eq!(fused, eligible);
                fused_rounds += fused as usize;
                if !fused {
                    a.partial_evaluate(r);
                    b.partial_evaluate(r);
                }
                c.partial_evaluate(r);
                d.partial_evaluate(r);
                assert_eq!(snapshot(&a), snapshot(&c));
                assert_eq!(snapshot(&b), snapshot(&d));
            }
            assert_eq!(fused_rounds, 1);
            assert_eq!(a.final_evaluations(), c.final_evaluations());
            assert_eq!(b.final_evaluations(), d.final_evaluations());
        }
        let (mut a, mut b) = (table(None, n, 0, 0), table(Some(map), n - 2, 0, 0));
        assert!(!a.partial_evaluate_with_mapped(&mut b, &challenges[0]));
        let (mut a, mut b) = (table(None, n, 0, 0), table(None, n, 0, 0));
        assert!(!a.partial_evaluate_with_mapped(&mut b, &challenges[0]));
        assert!(a.borrowed.is_some() && b.borrowed.is_some());
    }

    #[test]
    fn test_a_conjugate_view_folds_with_its_witness_under_fixed_challenges() {
        crate::common::init_common();
        let n = 8 * BORROW_MIN_LEN;
        let data = crate::common::sampling::sample_random_short_vector(n, 60, Representation::IncompleteNTT);
        let fixed = |x: &RingElement| x + &x.conjugate();
        let challenges: Vec<RingElement> =
            crate::common::sampling::sample_random_short_vector(14, 1 << 20, Representation::IncompleteNTT)
                .iter()
                .enumerate()
                .map(|(k, x)| if k == 3 { x.clone() } else { fixed(x) })
                .collect();
        assert!(challenges[3].conjugate() != challenges[3]);
        let map: fn(&RingElement, &mut RingElement) = conjugate;
        for (end, rounds) in [(n, [true, true, true, false, false]), (n - 5, [true, true, false, false, false])] {
            let mut a = unsafe { LinearSumcheck::from_borrowed(&data, end, None) };
            let mut b = unsafe { LinearSumcheck::conjugate_of(&data, end) };
            let mut c = unsafe { LinearSumcheck::from_borrowed(&data, end, None) };
            let mut d = unsafe { LinearSumcheck::from_borrowed(&data, end, Some(map)) };
            let rs = if end == n {
                challenges[..13].to_vec()
            } else {
                [&challenges[..2], &challenges[3..]].concat()
            };
            for (k, r) in rs.iter().enumerate() {
                let fused = a.partial_evaluate_with_mapped(&mut b, r);
                if !fused {
                    a.partial_evaluate(r);
                    b.partial_evaluate(r);
                }
                c.partial_evaluate(r);
                d.partial_evaluate(r);
                assert_eq!(b.borrowed.is_some(), rounds.get(k).copied().unwrap_or(false), "round {k}");
                assert_eq!(snapshot(&a), snapshot(&c));
                assert_eq!(snapshot(&b), snapshot(&d));
            }
            assert_eq!(a.final_evaluations(), c.final_evaluations());
            assert_eq!(b.final_evaluations(), d.final_evaluations());
        }
    }

    #[test]
    #[should_panic(expected = "a table viewed through its conjugate folds only with the view")]
    fn test_a_viewed_table_refuses_a_fold_without_its_view() {
        crate::common::init_common();
        let data = crate::common::sampling::sample_random_short_vector(
            4 * BORROW_MIN_LEN,
            60,
            Representation::IncompleteNTT,
        );
        let x = crate::common::sampling::sample_random_short_vector(
            1,
            1 << 20,
            Representation::IncompleteNTT,
        )
        .remove(0);
        let r = &x + &x.conjugate();
        let mut a = unsafe { LinearSumcheck::from_borrowed(&data, data.len(), None) };
        let mut b = unsafe { LinearSumcheck::conjugate_of(&data, data.len()) };
        assert!(a.partial_evaluate_with_mapped(&mut b, &r));
        assert!(b.borrowed.is_some());
        a.partial_evaluate(&r);
    }

    #[test]
    fn test_an_arbitrary_map_never_becomes_a_view() {
        crate::common::init_common();
        let data = crate::common::sampling::sample_random_short_vector(
            4 * BORROW_MIN_LEN,
            60,
            Representation::IncompleteNTT,
        );
        let r = RingElement::constant(3, Representation::IncompleteNTT);
        let map: fn(&RingElement, &mut RingElement) = conjugate;
        let mut a = unsafe { LinearSumcheck::from_borrowed(&data, data.len(), None) };
        let mut b = unsafe { LinearSumcheck::from_borrowed(&data, data.len(), Some(map)) };
        assert!(a.partial_evaluate_with_mapped(&mut b, &r));
        assert!(a.borrowed.is_none() && b.borrowed.is_none() && !a.viewed);
    }

    #[test]
    fn test_an_arbitrary_map_never_sees_the_challenge() {
        crate::common::init_common();
        let data: Vec<RingElement> = (0..4 * BORROW_MIN_LEN as u64)
            .map(|i| RingElement::constant(i & 1, Representation::IncompleteNTT))
            .collect();
        let negate: fn(&RingElement, &mut RingElement) = |x, out| {
            let one = RingElement::constant(1, Representation::IncompleteNTT);
            let zero = RingElement::zero(Representation::IncompleteNTT);
            assert!(*x == zero || *x == one, "the map is defined on bits only");
            out.set_from(&(&one - x));
        };
        let r = RingElement::constant(3, Representation::IncompleteNTT);
        let mut a = unsafe { LinearSumcheck::from_borrowed(&data, data.len(), None) };
        let mut b = unsafe { LinearSumcheck::from_borrowed(&data, data.len(), Some(negate)) };
        let mut c = unsafe { LinearSumcheck::from_borrowed(&data, data.len(), None) };
        let mut d = unsafe { LinearSumcheck::from_borrowed(&data, data.len(), Some(negate)) };
        assert!(a.partial_evaluate_with_mapped(&mut b, &r));
        c.partial_evaluate(&r);
        d.partial_evaluate(&r);
        assert_eq!(snapshot(&a), snapshot(&c));
        assert_eq!(snapshot(&b), snapshot(&d));
    }

    #[test]
    fn a_conjugate_view_outlives_the_table_it_was_folded_from() {
        crate::common::init_common();
        let data = crate::common::sampling::sample_random_short_vector(
            4 * BORROW_MIN_LEN,
            60,
            Representation::IncompleteNTT,
        );
        let x = crate::common::sampling::sample_random_short_vector(
            1,
            1 << 20,
            Representation::IncompleteNTT,
        )
        .remove(0);
        let r = &x + &x.conjugate();
        let s = RingElement::constant(5, Representation::IncompleteNTT);
        let map: fn(&RingElement, &mut RingElement) = conjugate;
        let mut want = unsafe { LinearSumcheck::from_borrowed(&data, data.len(), Some(map)) };
        want.partial_evaluate(&r);
        for replace in [false, true] {
            let mut a = unsafe { LinearSumcheck::from_borrowed(&data, data.len(), None) };
            let mut b = unsafe { LinearSumcheck::conjugate_of(&data, data.len()) };
            assert!(a.partial_evaluate_with_mapped(&mut b, &r));
            assert!(b.borrowed.is_some() && a.data.is_empty());
            if replace {
                a.data = data.clone();
            }
            drop(a);
            let reused = vec![RingElement::constant(1, Representation::IncompleteNTT); data.len()];
            assert_eq!(snapshot(&b), snapshot(&want));
            drop(reused);
            let mut again = unsafe { LinearSumcheck::from_borrowed(&data, data.len(), Some(map)) };
            again.partial_evaluate(&r);
            again.partial_evaluate(&s);
            b.partial_evaluate(&s);
            assert_eq!(snapshot(&b), snapshot(&again));
        }
    }

    #[test]
    fn test_linear_sumcheck() {
        use crate::common::ring_arithmetic::RingElement;

        let data = vec![
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(2, Representation::IncompleteNTT),
            RingElement::constant(3, Representation::IncompleteNTT),
            RingElement::constant(4, Representation::IncompleteNTT),
            RingElement::constant(5, Representation::IncompleteNTT),
            RingElement::constant(6, Representation::IncompleteNTT),
            RingElement::constant(7, Representation::IncompleteNTT),
            RingElement::constant(8, Representation::IncompleteNTT),
        ];

        let mut sumcheck = LinearSumcheck::new(data.len());
        sumcheck.load_from(&data);

        // Simulate three verifier challenges and ensure the running claim matches
        // the multilinear extension evaluated at the folded point.

        let r0 = RingElement::constant(524, Representation::IncompleteNTT);

        sumcheck.partial_evaluate(&r0);

        let r1 = RingElement::constant(1337, Representation::IncompleteNTT);

        sumcheck.partial_evaluate(&r1);

        let r2 = RingElement::constant(42, Representation::IncompleteNTT);

        sumcheck.partial_evaluate(&r2);

        debug_assert!(sumcheck.data.len() == 1);

        // LS-first: r0 folds x0 (bit 0), r1 folds x1 (bit 1), r2 folds x2 (bit 2).
        // data[b0 + 2*b1 + 4*b2] evaluated at x0=r0, x1=r1, x2=r2.
        debug_assert_eq!(
            sumcheck.data[0],
            RingElement::constant(
                (MOD_Q as i64
                    + 1 * (1 - 524) * (1 - 1337) * (1 - 42)
                    + 2 * 524 * (1 - 1337) * (1 - 42)
                    + 3 * (1 - 524) * 1337 * (1 - 42)
                    + 4 * 524 * 1337 * (1 - 42)
                    + 5 * (1 - 524) * (1 - 1337) * 42
                    + 6 * 524 * (1 - 1337) * 42
                    + 7 * (1 - 524) * 1337 * 42
                    + 8 * 524 * 1337 * 42) as u64,
                Representation::IncompleteNTT
            )
        )
    }

    #[test]
    fn test_linear_sumcheck_univariate_polynomial() {
        use crate::common::ring_arithmetic::RingElement;

        let data = vec![
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(2, Representation::IncompleteNTT),
            RingElement::constant(3, Representation::IncompleteNTT),
            RingElement::constant(4, Representation::IncompleteNTT),
            RingElement::constant(5, Representation::IncompleteNTT),
            RingElement::constant(6, Representation::IncompleteNTT),
            RingElement::constant(7, Representation::IncompleteNTT),
            RingElement::constant(8, Representation::IncompleteNTT),
        ];

        let mut sumcheck = LinearSumcheck::new(data.len());
        sumcheck.load_from(&data);

        let mut poly = Polynomial::new(2);

        // First round polynomial encodes x0 (the LSB under LS-first).
        // data[2i]=val@x0=0, data[2i+1]=val@x0=1.
        // constant = sum of data[0]+data[2]+data[4]+data[6] = 1+3+5+7
        // linear   = sum of (data[1]-data[0])+(data[3]-data[2])+(data[5]-data[4])+(data[7]-data[6])

        sumcheck.univariate_polynomial_into(&mut poly);

        debug_assert_eq!(
            poly.coefficients[0],
            RingElement::constant(1 + 3 + 5 + 7, Representation::IncompleteNTT)
        );

        debug_assert_eq!(
            poly.coefficients[1],
            RingElement::constant(
                (2 - 1) + (4 - 3) + (6 - 5) + (8 - 7),
                Representation::IncompleteNTT
            )
        );
    }

    #[test]
    fn test_masked_sumcheck_indexing() {
        use crate::common::ring_arithmetic::RingElement;

        let data = vec![
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(2, Representation::IncompleteNTT),
            RingElement::constant(3, Representation::IncompleteNTT),
            RingElement::constant(4, Representation::IncompleteNTT),
            RingElement::constant(5, Representation::IncompleteNTT),
            RingElement::constant(6, Representation::IncompleteNTT),
            RingElement::constant(7, Representation::IncompleteNTT),
            RingElement::constant(8, Representation::IncompleteNTT),
        ];

        let mut sumcheck = LinearSumcheck::new_with_prefixed_sufixed_data(data.len(), 2, 0);
        sumcheck.load_from(&data);

        // prefix=2, suffix=0, data_size=8 → variable_count=5.
        // LS-first order: rounds 0,1,2 fold data vars (x0 LSB → x2 MSB),
        //   rounds 3,4 are prefix dummy rounds (data.len()==1).

        let mut poly = Polynomial::new(0);

        sumcheck.univariate_polynomial_into(&mut poly);

        // Round 0: data round, folding x0. Hypercube has 2^4=16 points.
        // Each pair (data[2i], data[2i+1]) appears 4 times (prefix bits give 2^2=4 aliases).
        // constant = 4*(1+3+5+7) = 64, linear = 4*(1+1+1+1) = 16
        debug_assert_eq!(
            poly.coefficients[0],
            RingElement::constant((1 + 3 + 5 + 7) * 4, Representation::IncompleteNTT)
        );

        debug_assert_eq!(
            poly.coefficients[1],
            RingElement::constant(
                ((2 - 1) + (4 - 3) + (6 - 5) + (8 - 7)) * 4,
                Representation::IncompleteNTT
            )
        );

        debug_assert_eq!(poly.num_coefficients, 2);

        let mut claim = poly.at_zero();
        claim += &poly.at_one();

        // claim should equal sum over full hypercube = sum(data) * 2^(prefix+suffix) = 36 * 4 = 144
        debug_assert_eq!(
            claim,
            RingElement::constant(
                (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) * 4,
                Representation::IncompleteNTT,
            )
        );

        let r0 = RingElement::constant(524, Representation::IncompleteNTT);

        let new_claim = poly.at(&r0);

        sumcheck.partial_evaluate(&r0);

        sumcheck.univariate_polynomial_into(&mut poly);

        // Round 1: data round, folding x1. data_size=4. variable_count=4.
        // Hypercube has 2^3=8 points. Each pair appears 4 times (prefix).
        debug_assert_eq!(poly.num_coefficients, 2);

        debug_assert_eq!(&poly.at_zero() + &poly.at_one(), new_claim);

        let r1 = RingElement::constant(1337, Representation::IncompleteNTT);

        let new_claim = poly.at(&r1);

        sumcheck.partial_evaluate(&r1);

        sumcheck.univariate_polynomial_into(&mut poly);

        // Round 2: data round, folding x2 (MSB of data). data_size=2. variable_count=3.
        // Hypercube has 2^2=4 points. Each pair appears 4 times (prefix).
        debug_assert_eq!(poly.num_coefficients, 2);

        debug_assert_eq!(&poly.at_zero() + &poly.at_one(), new_claim);

        let r2 = RingElement::constant(42, Representation::IncompleteNTT);

        let new_claim = poly.at(&r2);

        sumcheck.partial_evaluate(&r2);

        sumcheck.univariate_polynomial_into(&mut poly);

        // Round 3: prefix dummy round. data.len()==1. Constant polynomial.
        debug_assert_eq!(poly.num_coefficients, 1);

        debug_assert_eq!(&poly.at_zero() + &poly.at_one(), new_claim);

        let r3 = RingElement::constant(7, Representation::IncompleteNTT);

        let new_claim = poly.at(&r3);

        sumcheck.partial_evaluate(&r3);

        sumcheck.univariate_polynomial_into(&mut poly);

        // Round 4: prefix dummy round. Constant polynomial.
        debug_assert_eq!(poly.num_coefficients, 1);

        debug_assert_eq!(&poly.at_zero() + &poly.at_one(), new_claim);

        let r4 = RingElement::constant(19, Representation::IncompleteNTT);

        let new_claim = poly.at(&r4);

        sumcheck.partial_evaluate(&r4);

        debug_assert!(sumcheck.data.len() == 1);

        // LS-first: r0=x0, r1=x1, r2=x2. Final evaluation = MLE(x0=r0, x1=r1, x2=r2).
        debug_assert_eq!(
            sumcheck.final_evaluations(),
            &RingElement::constant(
                (MOD_Q as i64
                    + 1 * (1 - 524) * (1 - 1337) * (1 - 42)
                    + 2 * 524 * (1 - 1337) * (1 - 42)
                    + 3 * (1 - 524) * 1337 * (1 - 42)
                    + 4 * 524 * 1337 * (1 - 42)
                    + 5 * (1 - 524) * (1 - 1337) * 42
                    + 6 * 524 * (1 - 1337) * 42
                    + 7 * (1 - 524) * 1337 * 42
                    + 8 * 524 * 1337 * 42) as u64,
                Representation::IncompleteNTT
            )
        );

        debug_assert_eq!(&new_claim, sumcheck.final_evaluations());
    }

    #[test]
    fn test_linear_sumcheck_with_suffixed_data() {
        use crate::common::ring_arithmetic::RingElement;

        let data = vec![
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(2, Representation::IncompleteNTT),
            RingElement::constant(3, Representation::IncompleteNTT),
            RingElement::constant(4, Representation::IncompleteNTT),
            RingElement::constant(5, Representation::IncompleteNTT),
            RingElement::constant(6, Representation::IncompleteNTT),
            RingElement::constant(7, Representation::IncompleteNTT),
            RingElement::constant(8, Representation::IncompleteNTT),
        ];

        // we head with a vector that has 2 suffixed variables, i.e. (1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7, 8,8,8,8)

        let mut sumcheck = LinearSumcheck::new_with_prefixed_sufixed_data(data.len(), 0, 2);
        sumcheck.load_from(&data);

        // Now, the sumcheck has 2 suffixed variables, so when we index with HypercubePoint.

        let mut poly = Polynomial::new(0);

        sumcheck.univariate_polynomial_into(&mut poly);

        let claim = RingElement::constant(
            (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) * 4,
            Representation::IncompleteNTT,
        );

        debug_assert_eq!(&poly.at_zero() + &poly.at_one(), claim);
    }

    #[test]
    fn test_evaluation_sumcheck() {
        let data = vec![
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(2, Representation::IncompleteNTT),
            RingElement::constant(3, Representation::IncompleteNTT),
            RingElement::constant(4, Representation::IncompleteNTT),
            RingElement::constant(5, Representation::IncompleteNTT),
            RingElement::constant(6, Representation::IncompleteNTT),
            RingElement::constant(7, Representation::IncompleteNTT),
            RingElement::constant(8, Representation::IncompleteNTT),
        ];

        let mut evaluation_sumcheck =
            BasicEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(8, 3, 3);

        evaluation_sumcheck.load_from(&data);

        let point = vec![
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(2, Representation::IncompleteNTT),
            RingElement::constant(5, Representation::IncompleteNTT),
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(6, Representation::IncompleteNTT),
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(3, Representation::IncompleteNTT),
            RingElement::constant(4, Representation::IncompleteNTT),
        ];

        let mut ref_sumcheck = LinearSumcheck::new_with_prefixed_sufixed_data(8, 3, 3);
        ref_sumcheck.load_from(&data);

        for r in point.iter() {
            ref_sumcheck.partial_evaluate(r);
        }
        let expected_evaluation = ref_sumcheck.final_evaluations();

        debug_assert_eq!(evaluation_sumcheck.evaluate(&point), expected_evaluation);
    }

    #[test]
    fn test_structured_row_evaluation_sumcheck() {
        // Create a structured row with 3 tensor layers
        // This represents 2^3 = 8 data points
        let tensor_layers = vec![
            RingElement::random(Representation::IncompleteNTT),
            RingElement::random(Representation::IncompleteNTT),
            RingElement::random(Representation::IncompleteNTT),
        ];

        let structured_row = StructuredRow { tensor_layers };

        let mut evaluation_sumcheck =
            StructuredRowEvaluationLinearSumcheck::new_with_prefixed_sufixed_data(8, 2, 3);

        evaluation_sumcheck.load_from(structured_row.clone());

        let point = vec![
            RingElement::random(Representation::IncompleteNTT), // prefix 0
            RingElement::random(Representation::IncompleteNTT), // prefix 1
            RingElement::random(Representation::IncompleteNTT), // data 0
            RingElement::random(Representation::IncompleteNTT), // data 1
            RingElement::random(Representation::IncompleteNTT), // data 2
            RingElement::random(Representation::IncompleteNTT), // suffix 0
            RingElement::random(Representation::IncompleteNTT), // suffix 1
            RingElement::random(Representation::IncompleteNTT), // suffix 2
        ];

        let mut ref_sumcheck = LinearSumcheck::new_with_prefixed_sufixed_data(8, 2, 3);

        let prepared_data = PreprocessedRow::from_structured_row(&structured_row);

        ref_sumcheck.load_from(&prepared_data.preprocessed_row);

        for r in point.iter() {
            ref_sumcheck.partial_evaluate(r);
        }
        let expected_evaluation = ref_sumcheck.final_evaluations();
        debug_assert_eq!(evaluation_sumcheck.evaluate(&point), expected_evaluation);
    }
}
