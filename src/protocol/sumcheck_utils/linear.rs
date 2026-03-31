use std::{cell::RefCell, ops::Index};

use crate::{
    common::{
        arithmetic::field_to_ring_element_into,
        config::HALF_DEGREE,
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        structured_row::StructuredRow,
        sumcheck_element::SumcheckElement,
    },
    protocol::sumcheck_utils::{
        common::{EvaluationSumcheckData, HighOrderSumcheckData, SumcheckBaseData},
        elephant_cell::ElephantCell,
        hypercube_point::HypercubePoint,
        polynomial::Polynomial,
    },
};

#[cfg(test)]
use crate::common::{config::MOD_Q, structured_row::PreprocessedRow};

/// Standard linear sumcheck over a vector that represents a multilinear extension.
pub struct LinearSumcheck<E: SumcheckElement = RingElement> {
    pub data: Vec<E>,
    variable_count: usize,
    index_mask: usize,
    suffix: usize,
    poly_scratch: RefCell<Polynomial<E>>,
    // Index past the last non-zero entry in `data`. Everything in
    // data[non_zero_end..] is guaranteed zero. Used to skip work in
    // partial_evaluate, non_zero_range, and batched Karatsuba.
    non_zero_end: usize,
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
            variable_count: count.ilog2() as usize + prefix_size + suffix_size,
            index_mask: count - 1, // this mask will be used to ignore prefixed variables
            poly_scratch: RefCell::new(Polynomial::new(2)),
            suffix: suffix_size,
            non_zero_end: count,
        }
    }
    /// Populate the internal buffer with the provided values.
    /// Marks the whole buffer as potentially non-zero.
    pub fn load_from(&mut self, src: &[E]) {
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
        if self.data.len() == 1 {
            return &self.data[0];
        }
        if self.suffix > 0
            || self.variable_count > self.data.len().trailing_zeros() as usize + self.suffix
        {
            // Prefix/suffix round — data isn't being split.
            // For suffix rounds under LS-first, the current round consumes
            // one suffix bit now, so the half-hypercube point still contains
            // the remaining (suffix - 1) suffix bits.
            let suffix_shift = if self.suffix > 0 { self.suffix - 1 } else { 0 };
            let index_shifted = index.shifted(suffix_shift);
            let index_masked = index_shifted.masked(self.index_mask);
            return &self.data[index_masked.coordinates];
        }
        // Data round, LS-first: point p → even slot 2p.
        let p = index.masked(self.data.len() / 2 - 1).coordinates;
        &self.data[2 * p]
    }
}

impl<E: SumcheckElement> HighOrderSumcheckData for LinearSumcheck<E> {
    type Element = E;

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

        if self.data.len() == 1 {
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
        if self.data.len() == 1 {
            polynomial.coefficients[0].set_from(&self.data[0]);
            polynomial.num_coefficients = 1;
            return;
        }

        // Data round: pair at (2p, 2p+1).
        let p = point.masked(self.data.len() / 2 - 1).coordinates;
        let idx0 = 2 * p;
        let idx1 = 2 * p + 1;
        polynomial.coefficients[0].set_from(&self.data[idx0]);
        polynomial.coefficients[1].set_from(&self.data[idx1]);
        polynomial.coefficients[1] -= &self.data[idx0];
        polynomial.num_coefficients = 2;
    }

    fn is_univariate_polynomial_zero_at_point(&self, point: HypercubePoint) -> bool {
        // For suffix/prefix rounds we intentionally expose no sparse range.
        if self.suffix > 0 || self.data.len() <= 1 {
            return false;
        }
        // Sparse tail tracking is only valid when data fills the full hypercube.
        if self.data.len() != (1usize << self.variable_count) {
            return false;
        }
        let half = self.data.len() / 2;
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
        if self.data.len() <= 1 {
            return None;
        }
        // data.len() must equal 2^variable_count (no pending prefix variables).
        if self.data.len() != (1usize << self.variable_count) {
            return None;
        }
        let nz_end = self.non_zero_end.min(self.data.len());
        Some(&self.data[..nz_end])
    }

    fn non_zero_range(&self) -> Option<(usize, usize)> {
        // LS-first: during suffix rounds or prefix rounds, no meaningful range.
        if self.suffix > 0 {
            return None;
        }
        if self.data.len() <= 1 {
            return None;
        }
        // Only valid when data fills the full hypercube (no prefix active).
        if self.data.len() != (1usize << self.variable_count) {
            return None;
        }
        let half = self.data.len() / 2;
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
        if self.data.len() != 1 {
            panic!("Sumcheck is not fully evaluated yet");
        }
        self.data[0].clone()
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
        if self.data.len() == 1 {
            self.variable_count -= 1;
            return;
        }

        // Data round: fold even/odd pairs.
        // data[2i] = value at current_var=0, data[2i+1] = value at current_var=1.
        // folded[i] = data[2i] + (data[2i+1] - data[2i]) * r
        let n = self.data.len();
        if n % 2 != 0 {
            panic!("Sumcheck data length must be a power of 2");
        }
        let half = n / 2;

        // even_nz: number of folded output slots that might be non-zero.
        // Pair i has even=data[2i], odd=data[2i+1]. If 2i >= non_zero_end
        // then both are zero → folded[i] is zero.
        let pair_nz = (self.non_zero_end + 1) / 2; // ceil(non_zero_end / 2)
        let fold_end = pair_nz.min(half);

        for i in 0..fold_end {
            let idx0 = 2 * i;
            let idx1 = 2 * i + 1;
            // delta = data[2i+1] - data[2i]
            // Use split_at_mut to get non-overlapping references.
            let (left, right) = self.data.split_at_mut(idx1);
            // left[idx0] = data[2i], right[0] = data[2i+1]
            right[0] -= &left[idx0];
            right[0] *= value;
            left[idx0] += &right[0];
            // Move folded result from idx0 → slot i.
            if i != idx0 {
                // idx0 = 2*i, which equals i only when i=0.
                // For i>=1, copy to the compacted position without allocation.
                // Split left at idx0 so source and destination are disjoint.
                let (prefix, from_idx0) = left.split_at_mut(idx0);
                prefix[i].set_from(&from_idx0[0]);
            }
        }
        // Clear the compacted tail: indices [fold_end, half) are outside the
        // guaranteed non-zero region after folding and may still contain stale
        // values from the previous layout.
        for i in fold_end..half {
            self.data[i].set_zero();
        }

        self.data.truncate(half);
        self.non_zero_end = fold_end;
        self.variable_count -= 1;
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
    field_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = QuadraticExtension>>,
    result: RingElement,
    evaluated: bool,
}

impl RingToFieldWrapperEvaluation {
    pub fn new(
        field_evaluation: ElephantCell<dyn EvaluationSumcheckData<Element = QuadraticExtension>>,
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
        let point_field: Vec<QuadraticExtension> = point
            .iter()
            .map(|r| QuadraticExtension {
                coeffs: [r.v[0], r.v[HALF_DEGREE]],
            })
            .collect();

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
    result: E,
    scratch: E,
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
            result: E::one(),
            scratch: E::zero(),
        }
    }

    pub fn load_from(&mut self, src: StructuredRow<E>) {
        debug_assert!(src.tensor_layers.len() == self.variable_count - self.suffix - self.prefix);
        // Normalize once to LS-first so evaluate() can zip directly with LS-first challenges.
        let mut normalized = src;
        normalized.tensor_layers.reverse();
        self.data = Some(normalized);
    }
}

impl<E: SumcheckElement + 'static> EvaluationSumcheckData
    for StructuredRowEvaluationLinearSumcheck<E>
{
    type Element = E;

    fn evaluate(&mut self, point: &Vec<Self::Element>) -> &Self::Element {
        self.result.set_from(&*E::one_ref());
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
        for (layer, r) in data.tensor_layers.iter().zip(data_point.iter()) {
            // Compute: (1-layer)*(1-r) + layer*r = 1 - layer - r + 2*layer*r
            self.scratch.set_from(layer);
            self.scratch *= r; // layer*r
            self.scratch *= &*E::two_ref(); // 2*layer*r
            self.scratch -= layer; // 2*layer*r - layer
            self.scratch -= r; // 2*layer*r - layer - r
            self.scratch += &*E::one_ref(); // 1 - layer - r + 2*layer*r

            self.result *= &self.scratch;
        }

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
