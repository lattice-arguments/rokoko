//! Proof serialisation. A coefficient is split into its octave `b = bitlen(|c|)`, coded with an
//! interleaved rANS against a histogram measured on the proof being sent — one per field role,
//! quantised and written in the header, so nothing is read from the norm schedule — and a raw
//! `b`-bit field holding the mantissa and the sign. Uniform residues land on the octave law
//! `Pr[b = k] = 2^(k - bitlen(q))`, of entropy 2, and so pay `ceil(log2 q)` exactly; the folded witness
//! and the norm claims pay their own scale, and zero carries no field at all. Ring elements are
//! coded in the representation they are held in unless the coefficient one saves more than a bit
//! per coefficient, and shapes travel as varints rather than being re-derived from `Config`. At
//! p-26 this is 112.1 KB against the 100.4 KB `size_in_bits` reports, and about 10 kB of that
//! gap is unreachable: that accounting charges `log2|c|`, the ideal code length for a scale-free
//! source, whereas the entropy floor of the coefficients actually sent is 110.7 KB.

use crate::common::config::{DEGREE, MOD_Q, NOF_BATCHES, SLOT_DEGREE};
use crate::common::matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix};
use crate::common::ring_arithmetic::{FieldExtension, Representation, RingElement};
use crate::protocol::config::{
    config_base_from_config, Config, IntermediateConfig, IntermediateRoundProof,
    NextRoundCommitment, Projection, RoundProof, SimpleConfig, SimpleRoundProof, SumcheckConfig,
    SumcheckRoundProof,
};
use crate::protocol::snark::{ClaimsProof, WitnessShape};
use crate::protocol::sumcheck_utils::polynomial::Polynomial;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WireError {
    Truncated,
    Malformed,
}

const SCALE_BITS: u32 = 12;
const SCALE: u32 = 1 << SCALE_BITS;
const RANS_LOW: u32 = 1 << 16;
const LANES: usize = 8;
const SYMBOLS: usize = 56;
const ROLES: usize = 16;
const MAX_ELEMENTS: usize = 1 << 20;
const VERSION: u8 = 1;
/// The rows of an opening below the root: the sumcheck point and its conjugate.
const OPENED_POINTS: usize = 2;

/// What a decoder may allocate, in coefficients, and how many rounds it may recurse into. A zero
/// in a role where zeros dominate codes into next to nothing, so the length of the input does not
/// bound what a proof declares; the config does. Declared lengths are charged against the
/// allowance before anything is allocated.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Allowance {
    pub(crate) cells: usize,
    pub(crate) rounds: usize,
}

impl Allowance {
    /// Exactly what an honest proof of `config` declares. The verifier's shape checks then hold
    /// each field to its own size.
    pub(crate) fn of(config: &SumcheckConfig) -> Allowance {
        let mut allowance = Allowance {
            cells: 0,
            rounds: 0,
        };
        allowance.sumcheck(config);
        allowance
    }

    /// The claims proof over a witness of `shape`: a round polynomial per variable and the two
    /// openings.
    fn claims(shape: WitnessShape) -> Allowance {
        let mut allowance = Allowance {
            cells: 0,
            rounds: 1,
        };
        allowance.polys((shape.height * shape.width).ilog2() as usize);
        allowance.rings(2);
        allowance
    }

    fn rings(&mut self, count: usize) {
        self.cells += count * DEGREE;
    }

    fn polys(&mut self, count: usize) {
        self.cells += count * 4 * SLOT_DEGREE;
    }

    fn images(&mut self, image_height: usize, width: usize) {
        self.rings((image_height + NOF_BATCHES) * width);
    }

    fn sumcheck(&mut self, config: &SumcheckConfig) {
        self.rounds += 1;
        self.polys(config.composed_witness_length.ilog2() as usize);
        self.rings(4 + config.projection_norm_scope().is_some() as usize);
        self.rings(config.opening_recursion.most_inner_config().rank);
        match &config.projection_recursion {
            Projection::Coarse(recursion) => self.rings(recursion.most_inner_config().rank),
            Projection::Fine(recursion) => self.rings(
                recursion.recursion_constant_term.most_inner_config().rank
                    + recursion
                        .recursion_batched_projection
                        .most_inner_config()
                        .rank
                    + NOF_BATCHES,
            ),
            Projection::Skip => {}
        }
        self.next(config.next.as_deref());
    }

    /// The next round and the commitment the round before sends for it.
    fn next(&mut self, next: Option<&Config>) {
        let Some(next) = next else {
            return;
        };
        if let Config::Sumcheck(next) = next {
            self.rings(next.commitment_recursion.most_inner_config().rank);
            return self.sumcheck(next);
        }
        let base = config_base_from_config(next);
        self.rings(base.basic_commitment_rank().next_power_of_two() * base.witness_width());
        match next {
            Config::Intermediate(next) => self.intermediate(next),
            Config::Simple(next) => self.simple(next),
            Config::Sumcheck(_) => unreachable!(),
        }
    }

    fn intermediate(&mut self, config: &IntermediateConfig) {
        self.rounds += 1;
        self.rings(OPENED_POINTS * config.witness_width);
        self.polys((config.witness_height * config.witness_decomposition_chunks).ilog2() as usize);
        self.rings(3);
        self.images(
            config.witness_height / config.projection_ratio,
            config.witness_width,
        );
        self.next(config.next.as_deref());
    }

    fn simple(&mut self, config: &SimpleConfig) {
        self.rounds += 1;
        self.rings(config.witness_height + OPENED_POINTS * config.witness_width);
        self.images(
            config.witness_height / config.projection_ratio,
            config.witness_width,
        );
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
enum Role {
    Poly = 0,
    Claim,
    Conjugate,
    Norm,
    InnerNorm,
    ProjectionNorm,
    Opening,
    CoarseProjection,
    FineProjection,
    ConstantTerm,
    Commitment,
    Folded,
    ProjectionCt,
    BatchedProjection,
    OpeningRhs,
    WitnessEval,
}

impl Role {
    /// The representation the protocol holds the role's ring elements in. The label the format
    /// carries is not read by the transcript or the arithmetic, so it is accepted only when it is
    /// this one.
    fn representation(self) -> Representation {
        match self {
            Role::ProjectionCt => Representation::Coefficients,
            _ => Representation::IncompleteNTT,
        }
    }
}

fn residue_bits() -> u32 {
    u64::BITS - (MOD_Q - 1).leading_zeros()
}

fn centred(v: u64) -> i64 {
    if v > MOD_Q / 2 {
        -((MOD_Q - v) as i64)
    } else {
        v as i64
    }
}

fn octave(c: i64) -> usize {
    if c == 0 {
        0
    } else {
        64 - c.unsigned_abs().leading_zeros() as usize
    }
}

fn cost(values: &[u64]) -> usize {
    values.iter().map(|v| octave(centred(*v))).sum()
}

fn flips(
    held: Representation,
    len: usize,
    resident: usize,
    coefficient_cost: impl FnOnce() -> usize,
) -> bool {
    let coefficients = len * DEGREE;
    held != Representation::Coefficients
        && resident > coefficients * (residue_bits() as usize - 4)
        && coefficient_cost() + coefficients < resident
}

// =================================================================================================
// bit streams
// =================================================================================================

#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    acc: u64,
    n: u32,
}

impl BitWriter {
    fn with_capacity(bytes: usize) -> BitWriter {
        BitWriter {
            bytes: Vec::with_capacity(bytes),
            acc: 0,
            n: 0,
        }
    }

    #[inline]
    fn put(&mut self, value: u64, bits: u32) {
        debug_assert!(bits <= 32);
        self.acc |= (value & ((1u64 << bits) - 1)) << self.n;
        self.n += bits;
        while self.n >= 8 {
            self.bytes.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }

    #[inline]
    fn put_wide(&mut self, value: u64, bits: u32) {
        if bits <= 32 {
            self.put(value, bits);
        } else {
            self.put(value & 0xffff_ffff, 32);
            self.put(value >> 32, bits - 32);
        }
    }

    fn put_gamma(&mut self, value: u64) {
        debug_assert!(value >= 1);
        let k = 63 - value.leading_zeros();
        self.put(0, k);
        self.put(1, 1);
        self.put_wide(value & ((1u64 << k) - 1), k);
    }

    fn put_varint(&mut self, mut value: u64) {
        loop {
            let byte = (value & 0x7f) as u64;
            value >>= 7;
            self.put(byte | ((value != 0) as u64) << 7, 8);
            if value == 0 {
                return;
            }
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.bytes.push(self.acc as u8);
        }
        self.bytes
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
    acc: u64,
    n: u32,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> BitReader<'a> {
        BitReader {
            bytes,
            pos: 0,
            acc: 0,
            n: 0,
        }
    }

    #[inline]
    fn refill(&mut self) {
        let rest = &self.bytes[self.pos..];
        let word = if rest.len() >= 8 {
            u64::from_le_bytes(rest[..8].try_into().unwrap())
        } else {
            let mut b = [0u8; 8];
            b[..rest.len()].copy_from_slice(rest);
            u64::from_le_bytes(b)
        };
        self.acc |= word << self.n;
        let take = (((63 - self.n) >> 3) as usize).min(rest.len());
        self.pos += take;
        self.n += 8 * take as u32;
    }

    #[inline]
    fn get(&mut self, bits: u32) -> Result<u64, WireError> {
        debug_assert!(bits <= 32);
        if self.n < bits {
            self.refill();
            if self.n < bits {
                return Err(WireError::Truncated);
            }
        }
        let value = self.acc & ((1u64 << bits) - 1);
        self.acc >>= bits;
        self.n -= bits;
        Ok(value)
    }

    #[inline]
    fn get_wide(&mut self, bits: u32) -> Result<u64, WireError> {
        if bits <= 32 {
            self.get(bits)
        } else {
            let low = self.get(32)?;
            Ok(low | (self.get(bits - 32)? << 32))
        }
    }

    fn get_gamma(&mut self) -> Result<u64, WireError> {
        let mut k = 0;
        while self.get(1)? == 0 {
            k += 1;
            if k > 40 {
                return Err(WireError::Malformed);
            }
        }
        Ok((1u64 << k) | self.get_wide(k)?)
    }

    fn get_len(&mut self) -> Result<usize, WireError> {
        let mut len = 0;
        for shift in (0..usize::BITS - MAX_ELEMENTS.leading_zeros()).step_by(7) {
            let byte = self.get(8)? as usize;
            len |= (byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                if (byte == 0 && shift > 0) || len > MAX_ELEMENTS {
                    return Err(WireError::Malformed);
                }
                return Ok(len);
            }
        }
        Err(WireError::Malformed)
    }

    fn end(mut self) -> Result<(), WireError> {
        self.refill();
        if self.pos < self.bytes.len() || self.n >= 8 || self.acc != 0 {
            return Err(WireError::Malformed);
        }
        Ok(())
    }
}

// =================================================================================================
// the octave model
// =================================================================================================

#[derive(Clone)]
struct Model {
    coded: bool,
    freq: [u16; SYMBOLS],
    cum: [u16; SYMBOLS],
    slot: Vec<u8>,
}

impl Model {
    fn raw() -> Model {
        Model {
            coded: false,
            freq: [0; SYMBOLS],
            cum: [0; SYMBOLS],
            slot: Vec::new(),
        }
    }

    fn from_counts(counts: &[u64; SYMBOLS]) -> Model {
        let total: u64 = counts.iter().sum();
        let mut freq = [0u16; SYMBOLS];
        let mut sum = 0u32;
        for s in 0..SYMBOLS {
            if counts[s] == 0 {
                continue;
            }
            freq[s] = ((counts[s] * SCALE as u64 / total) as u32).max(1) as u16;
            sum += freq[s] as u32;
        }
        let mut largest = 0;
        for s in 0..SYMBOLS {
            if freq[s] > freq[largest] {
                largest = s;
            }
        }
        freq[largest] = (freq[largest] as i64 + SCALE as i64 - sum as i64) as u16;

        let mut cum = [0u16; SYMBOLS];
        let mut running = 0u32;
        for s in 0..SYMBOLS {
            cum[s] = running as u16;
            running += freq[s] as u32;
        }
        Model {
            coded: true,
            freq,
            cum,
            slot: Vec::new(),
        }
    }

    fn chosen(counts: &[u64; SYMBOLS]) -> Model {
        let total: u64 = counts.iter().sum();
        if total == 0 {
            return Model::raw();
        }
        let candidate = Model::from_counts(counts);
        let coded = candidate.bits(counts) + (SYMBOLS * 16) as f64;
        if coded < 0.98 * (total * residue_bits() as u64) as f64 {
            candidate
        } else {
            Model::raw()
        }
    }

    fn build_slots(&mut self) {
        self.slot = vec![0u8; SCALE as usize];
        for s in 0..SYMBOLS {
            let start = self.cum[s] as usize;
            self.slot[start..start + self.freq[s] as usize].fill(s as u8);
        }
    }

    fn bits(&self, counts: &[u64; SYMBOLS]) -> f64 {
        let mut bits = 0.0;
        for s in 0..SYMBOLS {
            if counts[s] == 0 {
                continue;
            }
            bits += counts[s] as f64
                * ((SCALE as f64 / self.freq[s] as f64).log2()
                    + if s == 0 { 0.0 } else { s as f64 });
        }
        bits
    }

    fn write(&self, out: &mut BitWriter) {
        if !self.coded {
            out.put(0, 1);
            return;
        }
        out.put(1, 1);
        let low = self.freq.iter().position(|f| *f > 0).unwrap();
        let high = self.freq.iter().rposition(|f| *f > 0).unwrap();
        out.put(low as u64, 6);
        out.put(high as u64, 6);
        for s in low..=high {
            out.put_gamma(self.freq[s] as u64 + 1);
        }
    }

    fn read(input: &mut BitReader) -> Result<Model, WireError> {
        if input.get(1)? == 0 {
            return Ok(Model::raw());
        }
        let low = input.get(6)? as usize;
        let high = input.get(6)? as usize;
        if high >= SYMBOLS || low > high {
            return Err(WireError::Malformed);
        }
        let mut freq = [0u16; SYMBOLS];
        let mut cum = [0u16; SYMBOLS];
        let mut running = 0u32;
        for s in low..=high {
            let f = input.get_gamma()? - 1;
            if f > SCALE as u64 {
                return Err(WireError::Malformed);
            }
            freq[s] = f as u16;
        }
        if freq[low] == 0 || freq[high] == 0 {
            return Err(WireError::Malformed);
        }
        for s in 0..SYMBOLS {
            cum[s] = running as u16;
            running += freq[s] as u32;
        }
        if running != SCALE {
            return Err(WireError::Malformed);
        }
        let mut model = Model {
            coded: true,
            freq,
            cum,
            slot: Vec::new(),
        };
        model.build_slots();
        Ok(model)
    }
}

// =================================================================================================
// rANS
// =================================================================================================

struct RansEncoder {
    state: [u32; LANES],
    words: Vec<u16>,
}

impl RansEncoder {
    fn new(symbols: usize) -> RansEncoder {
        RansEncoder {
            state: [RANS_LOW; LANES],
            words: Vec::with_capacity(symbols / 2 + LANES),
        }
    }

    #[inline]
    fn put(&mut self, lane: usize, model: &Model, symbol: usize) {
        let freq = model.freq[symbol] as u32;
        debug_assert!(freq > 0);
        let mut x = self.state[lane];
        // 2^32 for a one-symbol model, whose symbol costs nothing and never renormalises.
        let ceiling = (((RANS_LOW >> SCALE_BITS) << 16) as u64) * freq as u64;
        if x as u64 >= ceiling {
            self.words.push(x as u16);
            x >>= 16;
        }
        self.state[lane] = ((x / freq) << SCALE_BITS) + (x % freq) + model.cum[symbol] as u32;
    }

    fn finish(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(4 * LANES + 2 * self.words.len());
        for x in self.state {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        for word in self.words.iter().rev() {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

struct RansDecoder<'a> {
    state: [u32; LANES],
    words: &'a [u8],
    pos: usize,
}

impl<'a> RansDecoder<'a> {
    fn new(bytes: &'a [u8]) -> Result<RansDecoder<'a>, WireError> {
        if bytes.len() < 4 * LANES {
            return Err(WireError::Truncated);
        }
        let mut state = [0u32; LANES];
        for (lane, slot) in state.iter_mut().enumerate() {
            *slot = u32::from_le_bytes(bytes[4 * lane..4 * lane + 4].try_into().unwrap());
        }
        if state.iter().any(|&x| x < RANS_LOW) {
            return Err(WireError::Malformed);
        }
        Ok(RansDecoder {
            state,
            words: &bytes[4 * LANES..],
            pos: 0,
        })
    }

    #[inline]
    fn get(&mut self, lane: usize, model: &Model) -> Result<usize, WireError> {
        let x = self.state[lane];
        let slot = (x & (SCALE - 1)) as usize;
        let symbol = model.slot[slot] as usize;
        let mut next =
            model.freq[symbol] as u32 * (x >> SCALE_BITS) + slot as u32 - model.cum[symbol] as u32;
        if next < RANS_LOW {
            if self.pos + 2 > self.words.len() {
                return Err(WireError::Truncated);
            }
            let word = u16::from_le_bytes(self.words[self.pos..self.pos + 2].try_into().unwrap());
            self.pos += 2;
            next = (next << 16) | word as u32;
        }
        self.state[lane] = next;
        Ok(symbol)
    }

    fn end(&self) -> Result<(), WireError> {
        if self.pos != self.words.len() || self.state.iter().any(|&x| x != RANS_LOW) {
            return Err(WireError::Malformed);
        }
        Ok(())
    }
}

// =================================================================================================
// encoding
// =================================================================================================

struct Encoder {
    shape: BitWriter,
    cells: Vec<(i64, u8)>,
    counts: [[u64; SYMBOLS]; ROLES],
    scratch: Vec<RingElement>,
}

impl Encoder {
    fn new() -> Encoder {
        Encoder {
            shape: BitWriter::with_capacity(4096),
            cells: Vec::with_capacity(1 << 16),
            counts: [[0; SYMBOLS]; ROLES],
            scratch: Vec::new(),
        }
    }

    #[inline]
    fn push(&mut self, role: Role, v: u64) {
        let c = centred(v);
        self.counts[role as usize][octave(c)] += 1;
        self.cells.push((c, role as u8));
    }

    fn ring_cost(els: &[RingElement]) -> usize {
        els.iter().map(|e| cost(&e.v)).sum()
    }

    fn ring_slice(&mut self, role: Role, els: &[RingElement]) {
        self.shape.put_varint(els.len() as u64);
        if els.is_empty() {
            return;
        }
        let uniform = els
            .iter()
            .all(|e| e.representation == els[0].representation);
        self.shape.put(uniform as u64, 1);
        if uniform {
            self.shape.put(els[0].representation as u64, 2);
        } else {
            for e in els {
                self.shape.put(e.representation as u64, 2);
            }
        }

        let scratch = &mut self.scratch;
        let flip = flips(
            els[0].representation,
            els.len(),
            Self::ring_cost(els),
            || {
                scratch.clear();
                scratch.extend(els.iter().map(|e| {
                    let mut e = e.clone();
                    e.to_representation(Representation::Coefficients);
                    e
                }));
                Self::ring_cost(scratch)
            },
        );
        self.shape.put(flip as u64, 1);

        let coded = if flip {
            std::mem::take(&mut self.scratch)
        } else {
            Vec::new()
        };
        for e in if flip { &coded[..] } else { els } {
            for v in &e.v {
                self.push(role, *v);
            }
        }
        self.scratch = coded;
    }

    fn ring(&mut self, role: Role, el: &RingElement) {
        self.ring_slice(role, std::slice::from_ref(el));
    }

    fn option_ring(&mut self, role: Role, el: &Option<RingElement>) {
        self.shape.put(el.is_some() as u64, 1);
        if let Some(el) = el {
            self.ring(role, el);
        }
    }

    fn option_slice(&mut self, role: Role, els: &Option<Vec<RingElement>>) {
        self.shape.put(els.is_some() as u64, 1);
        if let Some(els) = els {
            self.ring_slice(role, els);
        }
    }

    fn matrix_v(&mut self, role: Role, m: &VerticallyAlignedMatrix<RingElement>) {
        self.shape.put_varint(m.width as u64);
        self.shape.put_varint(m.height as u64);
        self.shape.put_varint(m.used_cols as u64);
        self.ring_slice(role, &m.data);
    }

    fn matrix_h(&mut self, role: Role, m: &HorizontallyAlignedMatrix<RingElement>) {
        self.shape.put_varint(m.width as u64);
        self.shape.put_varint(m.height as u64);
        self.ring_slice(role, &m.data);
    }

    /// A round polynomial is written without its leading zero coefficients, which neither its
    /// evaluations nor the transcript see.
    fn polys(&mut self, polys: &[Polynomial<FieldExtension>]) {
        self.shape.put_varint(polys.len() as u64);
        for poly in polys {
            let len = poly.coefficients[..poly.num_coefficients]
                .iter()
                .rposition(|coeff| coeff.coeffs.iter().any(|&v| v != 0))
                .map_or(0, |top| top + 1);
            self.shape.put(len as u64, 3);
            for coeff in &poly.coefficients[..len] {
                for v in &coeff.coeffs {
                    self.push(Role::Poly, *v);
                }
            }
        }
    }

    fn next_commitment(&mut self, commitment: &Option<NextRoundCommitment>) {
        self.shape.put(commitment.is_some() as u64, 1);
        match commitment {
            None => {}
            Some(NextRoundCommitment::Recursive(rc)) => {
                self.shape.put(0, 1);
                self.ring_slice(Role::Commitment, rc);
            }
            Some(NextRoundCommitment::Simple(m)) => {
                self.shape.put(1, 1);
                self.matrix_h(Role::Commitment, m);
            }
        }
    }

    fn round(&mut self, proof: &RoundProof) {
        match proof {
            RoundProof::Sumcheck(p) => {
                self.shape.put(0, 2);
                self.sumcheck(p);
            }
            RoundProof::Simple(p) => {
                self.shape.put(1, 2);
                self.simple(p);
            }
            RoundProof::Intermediate(p) => {
                self.shape.put(2, 2);
                self.intermediate(p);
            }
        }
    }

    fn sumcheck(&mut self, p: &SumcheckRoundProof) {
        self.polys(&p.polys);
        self.ring(Role::Claim, &p.claim_over_witness);
        self.ring(Role::Conjugate, &p.claim_over_witness_conjugate);
        self.ring(Role::Norm, &p.norm_claim);
        self.ring(Role::InnerNorm, &p.most_inner_norm_claim);
        self.option_ring(Role::ProjectionNorm, &p.projection_norm_claim);
        self.ring_slice(Role::Opening, &p.rc_opening_inner);
        self.option_slice(Role::CoarseProjection, &p.rc_coarse_projection_inner);
        self.shape
            .put(p.rc_fine_projection_inner.is_some() as u64, 1);
        if let Some((left, right)) = &p.rc_fine_projection_inner {
            self.ring_slice(Role::FineProjection, left);
            self.ring_slice(Role::FineProjection, right);
        }
        self.option_slice(Role::ConstantTerm, &p.constant_term_claims);
        self.next_commitment(&p.next_round_commitment);
        self.next(&p.next);
    }

    fn simple(&mut self, p: &SimpleRoundProof) {
        self.matrix_v(Role::Folded, &p.folded_witness);
        self.matrix_v(Role::ProjectionCt, &p.projection_image_ct);
        self.matrix_h(Role::BatchedProjection, &p.batched_projection_image);
        self.matrix_h(Role::OpeningRhs, &p.opening_rhs);
    }

    fn intermediate(&mut self, p: &IntermediateRoundProof) {
        self.matrix_h(Role::OpeningRhs, &p.opening_rhs);
        self.polys(&p.polys);
        self.ring(Role::Claim, &p.claim_over_witness);
        self.ring(Role::Conjugate, &p.claim_over_witness_conjugate);
        self.ring(Role::Norm, &p.norm_claim);
        self.next_commitment(&p.next_round_commitment);
        self.matrix_v(Role::ProjectionCt, &p.projection_image_ct);
        self.matrix_h(Role::BatchedProjection, &p.batched_projection_image);
        self.next(&p.next);
    }

    fn next(&mut self, next: &Option<Box<RoundProof>>) {
        self.shape.put(next.is_some() as u64, 1);
        if let Some(next) = next {
            self.round(next);
        }
    }

    fn finish(self) -> Vec<u8> {
        let models = self.counts.iter().map(Model::chosen).collect();
        self.finish_with(models)
    }

    fn finish_with(self, models: Vec<Model>) -> Vec<u8> {
        let bits = residue_bits();
        let mut head = BitWriter::with_capacity(512);
        for model in &models {
            model.write(&mut head);
        }

        let mut raw = BitWriter::with_capacity(self.cells.len() * 8);
        let mut symbols: Vec<(usize, u8)> = Vec::with_capacity(self.cells.len());
        for (c, role) in &self.cells {
            let model = &models[*role as usize];
            if !model.coded {
                raw.put_wide(
                    if *c < 0 {
                        (MOD_Q as i64 + *c) as u64
                    } else {
                        *c as u64
                    },
                    bits,
                );
                continue;
            }
            let b = octave(*c);
            symbols.push((b, *role));
            if b > 0 {
                let magnitude = c.unsigned_abs();
                let field = (magnitude - (1 << (b - 1))) | ((*c < 0) as u64) << (b - 1);
                raw.put_wide(field, b as u32);
            }
        }

        let mut rans = RansEncoder::new(symbols.len());
        for i in (0..symbols.len()).rev() {
            let (symbol, role) = symbols[i];
            rans.put(i % LANES, &models[role as usize], symbol);
        }

        let head = head.finish();
        let shape = self.shape.finish();
        let rans = rans.finish();
        let raw = raw.finish();

        let mut out = Vec::with_capacity(13 + head.len() + shape.len() + rans.len() + raw.len());
        out.push(VERSION);
        for len in [head.len(), shape.len(), rans.len()] {
            out.extend_from_slice(&(len as u32).to_le_bytes());
        }
        out.extend_from_slice(&head);
        out.extend_from_slice(&shape);
        out.extend_from_slice(&rans);
        out.extend_from_slice(&raw);
        out
    }
}

// =================================================================================================
// decoding
// =================================================================================================

struct Body<'a> {
    raw: BitReader<'a>,
    rans: RansDecoder<'a>,
    models: Vec<Model>,
    counts: [[u64; SYMBOLS]; ROLES],
    bits: u32,
    index: usize,
}

impl<'a> Body<'a> {
    fn fill(&mut self, role: Role, out: &mut [u64]) -> Result<usize, WireError> {
        let model = &self.models[role as usize];
        let counts = &mut self.counts[role as usize];
        let mut sent = 0;
        if !model.coded {
            for slot in out.iter_mut() {
                let v = self.raw.get_wide(self.bits)?;
                if v >= MOD_Q {
                    return Err(WireError::Malformed);
                }
                let b = octave(centred(v));
                counts[b] += 1;
                sent += b;
                *slot = v;
            }
            return Ok(sent);
        }
        let mut index = self.index;
        for slot in out.iter_mut() {
            let b = self.rans.get(index % LANES, model)?;
            index += 1;
            counts[b] += 1;
            sent += b;
            if b == 0 {
                *slot = 0;
                continue;
            }
            let field = self.raw.get_wide(b as u32)?;
            let magnitude = (1u64 << (b - 1)) | (field & ((1u64 << (b - 1)) - 1));
            if magnitude > MOD_Q / 2 {
                return Err(WireError::Malformed);
            }
            *slot = if field >> (b - 1) != 0 {
                MOD_Q - magnitude
            } else {
                magnitude
            };
        }
        self.index = index;
        Ok(sent)
    }
}

struct Decoder<'a> {
    shape: BitReader<'a>,
    body: Body<'a>,
    rounds: usize,
    cells: usize,
}

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8], allowance: Allowance) -> Result<Decoder<'a>, WireError> {
        if bytes.len() < 13 || bytes[0] != VERSION {
            return Err(if bytes.len() < 13 {
                WireError::Truncated
            } else {
                WireError::Malformed
            });
        }
        let mut cut = [0usize; 3];
        for (i, len) in cut.iter_mut().enumerate() {
            *len = u32::from_le_bytes(bytes[1 + 4 * i..5 + 4 * i].try_into().unwrap()) as usize;
        }
        let mut at = 13;
        let section = |len: usize, at: &mut usize| -> Result<&'a [u8], WireError> {
            if *at + len > bytes.len() {
                return Err(WireError::Truncated);
            }
            let slice = &bytes[*at..*at + len];
            *at += len;
            Ok(slice)
        };
        let head = section(cut[0], &mut at)?;
        let shape = section(cut[1], &mut at)?;
        let rans = section(cut[2], &mut at)?;
        let raw = section(bytes.len() - at, &mut at)?;

        let mut head = BitReader::new(head);
        let mut models = Vec::with_capacity(ROLES);
        for _ in 0..ROLES {
            models.push(Model::read(&mut head)?);
        }
        head.end()?;
        Ok(Decoder {
            shape: BitReader::new(shape),
            body: Body {
                raw: BitReader::new(raw),
                rans: RansDecoder::new(rans)?,
                models,
                counts: [[0; SYMBOLS]; ROLES],
                bits: residue_bits(),
                index: 0,
            },
            rounds: allowance
                .rounds
                .checked_sub(1)
                .ok_or(WireError::Malformed)?,
            cells: allowance.cells,
        })
    }

    /// Spends `cells` of the allowance.
    fn charge(&mut self, cells: usize) -> Result<(), WireError> {
        self.cells = self.cells.checked_sub(cells).ok_or(WireError::Malformed)?;
        Ok(())
    }

    fn ring_slice(&mut self, role: Role) -> Result<Vec<RingElement>, WireError> {
        let len = self.shape.get_len()?;
        if len == 0 {
            return Ok(Vec::new());
        }
        self.charge(len * DEGREE)?;
        let held = role.representation();
        if self.shape.get(1)? == 0 || representation(self.shape.get(2)?)? != held {
            return Err(WireError::Malformed);
        }
        let flip = self.shape.get(1)? != 0;

        let mut els = Vec::with_capacity(len.min(4096));
        let mut sent = 0;
        for _ in 0..len {
            let mut el = RingElement::new(if flip {
                Representation::Coefficients
            } else {
                held
            });
            sent += self.body.fill(role, &mut el.v)?;
            if flip {
                el.to_representation(held);
            }
            els.push(el);
        }
        let chosen = if flip {
            flips(held, len, Encoder::ring_cost(&els), || sent)
        } else {
            flips(held, len, sent, || {
                let mut strided = [0; DEGREE];
                els.iter()
                    .map(|e| {
                        e.strided_coefficients_into(&mut strided);
                        cost(&strided)
                    })
                    .sum()
            })
        };
        if chosen != flip {
            return Err(WireError::Malformed);
        }
        Ok(els)
    }

    fn ring(&mut self, role: Role) -> Result<RingElement, WireError> {
        let mut els = self.ring_slice(role)?;
        if els.len() != 1 {
            return Err(WireError::Malformed);
        }
        Ok(els.pop().unwrap())
    }

    fn option_ring(&mut self, role: Role) -> Result<Option<RingElement>, WireError> {
        if self.shape.get(1)? == 0 {
            return Ok(None);
        }
        Ok(Some(self.ring(role)?))
    }

    fn option_slice(&mut self, role: Role) -> Result<Option<Vec<RingElement>>, WireError> {
        if self.shape.get(1)? == 0 {
            return Ok(None);
        }
        Ok(Some(self.ring_slice(role)?))
    }

    fn matrix_v(&mut self, role: Role) -> Result<VerticallyAlignedMatrix<RingElement>, WireError> {
        let width = self.shape.get_len()?;
        let height = self.shape.get_len()?;
        let used_cols = self.shape.get_len()?;
        let data = self.ring_slice(role)?;
        if width.checked_mul(height) != Some(data.len()) || used_cols > width {
            return Err(WireError::Malformed);
        }
        Ok(VerticallyAlignedMatrix {
            data,
            width,
            height,
            used_cols,
        })
    }

    fn matrix_h(
        &mut self,
        role: Role,
    ) -> Result<HorizontallyAlignedMatrix<RingElement>, WireError> {
        let width = self.shape.get_len()?;
        let height = self.shape.get_len()?;
        let data = self.ring_slice(role)?;
        if width.checked_mul(height) != Some(data.len()) {
            return Err(WireError::Malformed);
        }
        Ok(HorizontallyAlignedMatrix {
            data,
            width,
            height,
        })
    }

    fn polys(&mut self) -> Result<Vec<Polynomial<FieldExtension>>, WireError> {
        let len = self.shape.get_len()?;
        self.charge(len * 4 * SLOT_DEGREE)?;
        let mut polys = Vec::with_capacity(len);
        for _ in 0..len {
            let num_coefficients = self.shape.get(3)? as usize;
            if num_coefficients > 4 {
                return Err(WireError::Malformed);
            }
            let mut poly = Polynomial::<FieldExtension>::new(0);
            poly.num_coefficients = num_coefficients;
            for i in 0..num_coefficients {
                let mut coeffs = [0u64; SLOT_DEGREE];
                self.body.fill(Role::Poly, &mut coeffs)?;
                poly.coefficients[i].coeffs = coeffs;
            }
            if poly.coefficients[..num_coefficients]
                .last()
                .is_some_and(|top| top.coeffs.iter().all(|&v| v == 0))
            {
                return Err(WireError::Malformed);
            }
            polys.push(poly);
        }
        Ok(polys)
    }

    fn next_commitment(&mut self) -> Result<Option<NextRoundCommitment>, WireError> {
        if self.shape.get(1)? == 0 {
            return Ok(None);
        }
        Ok(Some(if self.shape.get(1)? == 0 {
            NextRoundCommitment::Recursive(self.ring_slice(Role::Commitment)?)
        } else {
            NextRoundCommitment::Simple(self.matrix_h(Role::Commitment)?)
        }))
    }

    fn round(&mut self) -> Result<RoundProof, WireError> {
        Ok(match self.shape.get(2)? {
            0 => RoundProof::Sumcheck(self.sumcheck()?),
            1 => RoundProof::Simple(self.simple()?),
            2 => RoundProof::Intermediate(self.intermediate()?),
            _ => return Err(WireError::Malformed),
        })
    }

    fn sumcheck(&mut self) -> Result<SumcheckRoundProof, WireError> {
        let polys = self.polys()?;
        let claim_over_witness = self.ring(Role::Claim)?;
        let claim_over_witness_conjugate = self.ring(Role::Conjugate)?;
        let norm_claim = self.ring(Role::Norm)?;
        let most_inner_norm_claim = self.ring(Role::InnerNorm)?;
        let projection_norm_claim = self.option_ring(Role::ProjectionNorm)?;
        let rc_opening_inner = self.ring_slice(Role::Opening)?;
        let rc_coarse_projection_inner = self.option_slice(Role::CoarseProjection)?;
        let rc_fine_projection_inner = if self.shape.get(1)? != 0 {
            Some((
                self.ring_slice(Role::FineProjection)?,
                self.ring_slice(Role::FineProjection)?,
            ))
        } else {
            None
        };
        let constant_term_claims = self.option_slice(Role::ConstantTerm)?;
        let next_round_commitment = self.next_commitment()?;
        Ok(SumcheckRoundProof {
            polys,
            claim_over_witness,
            claim_over_witness_conjugate,
            norm_claim,
            most_inner_norm_claim,
            projection_norm_claim,
            rc_opening_inner,
            rc_coarse_projection_inner,
            rc_fine_projection_inner,
            constant_term_claims,
            next_round_commitment,
            next: self.next()?,
        })
    }

    fn simple(&mut self) -> Result<SimpleRoundProof, WireError> {
        Ok(SimpleRoundProof {
            folded_witness: self.matrix_v(Role::Folded)?,
            projection_image_ct: self.matrix_v(Role::ProjectionCt)?,
            batched_projection_image: self.matrix_h(Role::BatchedProjection)?,
            opening_rhs: self.matrix_h(Role::OpeningRhs)?,
        })
    }

    fn intermediate(&mut self) -> Result<IntermediateRoundProof, WireError> {
        let opening_rhs = self.matrix_h(Role::OpeningRhs)?;
        let polys = self.polys()?;
        let claim_over_witness = self.ring(Role::Claim)?;
        let claim_over_witness_conjugate = self.ring(Role::Conjugate)?;
        let norm_claim = self.ring(Role::Norm)?;
        let next_round_commitment = self.next_commitment()?;
        let projection_image_ct = self.matrix_v(Role::ProjectionCt)?;
        let batched_projection_image = self.matrix_h(Role::BatchedProjection)?;
        Ok(IntermediateRoundProof {
            opening_rhs,
            polys,
            claim_over_witness,
            claim_over_witness_conjugate,
            norm_claim,
            next_round_commitment,
            projection_image_ct,
            batched_projection_image,
            next: self.next()?,
        })
    }

    fn next(&mut self) -> Result<Option<Box<RoundProof>>, WireError> {
        if self.shape.get(1)? == 0 {
            return Ok(None);
        }
        self.rounds = self.rounds.checked_sub(1).ok_or(WireError::Malformed)?;
        Ok(Some(Box::new(self.round()?)))
    }

    fn finish(self) -> Result<(), WireError> {
        let Body {
            raw,
            rans,
            models,
            counts,
            ..
        } = self.body;
        for (model, counts) in models.iter().zip(&counts) {
            let chosen = Model::chosen(counts);
            if (model.coded, model.freq) != (chosen.coded, chosen.freq) {
                return Err(WireError::Malformed);
            }
        }
        self.shape.end()?;
        raw.end()?;
        rans.end()
    }
}

fn representation(tag: u64) -> Result<Representation, WireError> {
    Ok(match tag {
        0 => Representation::Coefficients,
        1 => Representation::StridedCoefficients,
        2 => Representation::IncompleteNTT,
        3 => Representation::HomogenizedFieldExtensions,
        _ => return Err(WireError::Malformed),
    })
}

pub fn initial_to_bytes(proof: &ClaimsProof) -> Vec<u8> {
    let mut encoder = Encoder::new();
    encoder.polys(&proof.polys);
    encoder.ring(Role::WitnessEval, &proof.witness_eval);
    encoder.option_ring(Role::WitnessEval, &proof.conj_witness_eval);
    encoder.finish()
}

/// Like `from_bytes_for`, for the claims proof over a witness of `shape`.
pub fn initial_from_bytes_for(
    shape: impl Into<WitnessShape>,
    bytes: &[u8],
) -> Result<ClaimsProof, WireError> {
    let mut decoder = Decoder::new(bytes, Allowance::claims(shape.into()))?;
    let proof = ClaimsProof {
        polys: decoder.polys()?,
        witness_eval: decoder.ring(Role::WitnessEval)?,
        conj_witness_eval: decoder.option_ring(Role::WitnessEval)?,
    };
    decoder.finish()?;
    Ok(proof)
}

pub fn to_bytes(proof: &SumcheckRoundProof) -> Vec<u8> {
    let mut encoder = Encoder::new();
    encoder.sumcheck(proof);
    encoder.finish()
}

/// Decodes a proof of `config`'s chain, declaring no more than an honest one does. It accepts
/// exactly the bytes `to_bytes` writes for the proof they decode to, so an accepted proof has one
/// encoding: the coding choices the format leaves open (the histograms, which elements are coded
/// as coefficients, the padding of each section) are the encoder's, never the sender's.
pub fn from_bytes_for(
    config: &SumcheckConfig,
    bytes: &[u8],
) -> Result<SumcheckRoundProof, WireError> {
    from_bytes_within(bytes, Allowance::of(config))
}

/// `from_bytes_for` within an explicit allowance.
pub(crate) fn from_bytes_within(
    bytes: &[u8],
    allowance: Allowance,
) -> Result<SumcheckRoundProof, WireError> {
    let mut decoder = Decoder::new(bytes, allowance)?;
    let proof = decoder.sumcheck()?;
    decoder.finish()?;
    Ok(proof)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::common::init_common;
    use crate::common::matrix::ZeroNew;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// The fixtures follow no config; they decode within an allowance well above theirs.
    const FIXTURE: Allowance = Allowance {
        cells: 1 << 20,
        rounds: 8,
    };

    fn from_bytes(bytes: &[u8]) -> Result<SumcheckRoundProof, WireError> {
        from_bytes_within(bytes, FIXTURE)
    }

    /// A claims fixture with `polys` round polynomials.
    fn initial_from_bytes(polys: usize, bytes: &[u8]) -> Result<ClaimsProof, WireError> {
        initial_from_bytes_for((1 << polys, 1), bytes)
    }

    fn short(bound: u64) -> RingElement {
        let mut el = RingElement::random_bounded(Representation::Coefficients, bound);
        el.to_representation(Representation::IncompleteNTT);
        el
    }

    fn uniform() -> RingElement {
        RingElement::random(Representation::IncompleteNTT)
    }

    fn vertical(height: usize, width: usize, bound: u64) -> VerticallyAlignedMatrix<RingElement> {
        let mut m = VerticallyAlignedMatrix::new_zero(height, width, &uniform());
        for slot in m.data.iter_mut() {
            *slot = short(bound);
        }
        m.used_cols = width - 1;
        m
    }

    /// A projection image, which the protocol holds as coefficients.
    fn image(height: usize, width: usize, bound: u64) -> VerticallyAlignedMatrix<RingElement> {
        let mut m = vertical(height, width, bound);
        for slot in m.data.iter_mut() {
            slot.to_representation(Representation::Coefficients);
        }
        m
    }

    fn horizontal(height: usize, width: usize) -> HorizontallyAlignedMatrix<RingElement> {
        let mut m = HorizontallyAlignedMatrix::new_zero(height, width, &uniform());
        for slot in m.data.iter_mut() {
            *slot = uniform();
        }
        m.data[0] = RingElement::zero(Representation::IncompleteNTT);
        m
    }

    fn poly(num_coefficients: usize) -> Polynomial<FieldExtension> {
        let mut poly = Polynomial::<FieldExtension>::new(0);
        poly.num_coefficients = num_coefficients;
        for i in 0..num_coefficients {
            poly.coefficients[i] = FieldExtension {
                coeffs: std::array::from_fn(|k| uniform().v[k]),
            };
        }
        poly
    }

    fn simple() -> RoundProof {
        RoundProof::Simple(SimpleRoundProof {
            folded_witness: vertical(4, 3, 1 << 11),
            projection_image_ct: image(2, 2, 1 << 5),
            batched_projection_image: horizontal(2, 2),
            opening_rhs: horizontal(2, 2),
        })
    }

    fn intermediate() -> RoundProof {
        RoundProof::Intermediate(IntermediateRoundProof {
            opening_rhs: horizontal(1, 2),
            polys: (1..=4).map(poly).collect(),
            claim_over_witness: uniform(),
            claim_over_witness_conjugate: uniform(),
            norm_claim: short(1 << 28),
            next_round_commitment: Some(NextRoundCommitment::Simple(horizontal(2, 1))),
            projection_image_ct: image(2, 1, 1 << 5),
            batched_projection_image: horizontal(1, 2),
            next: Some(Box::new(simple())),
        })
    }

    fn sumcheck(next: Option<Box<RoundProof>>, full: bool) -> SumcheckRoundProof {
        SumcheckRoundProof {
            polys: (2..=3).map(poly).collect(),
            claim_over_witness: uniform(),
            claim_over_witness_conjugate: uniform(),
            norm_claim: short(1 << 28),
            most_inner_norm_claim: short(1 << 19),
            projection_norm_claim: full.then(|| short(1 << 24)),
            rc_opening_inner: vec![uniform(), uniform()],
            rc_coarse_projection_inner: full.then(|| vec![uniform()]),
            rc_fine_projection_inner: full.then(|| (vec![uniform()], vec![uniform(), uniform()])),
            constant_term_claims: full.then(|| vec![uniform()]),
            next_round_commitment: Some(NextRoundCommitment::Recursive(vec![uniform(); 3])),
            next,
        }
    }

    fn same_ring(left: &[RingElement], right: &[RingElement]) {
        assert_eq!(left.len(), right.len());
        for (l, r) in left.iter().zip(right) {
            assert_eq!(l.representation, r.representation);
            assert_eq!(l.v, r.v);
        }
    }

    #[test]
    fn wire_roundtrip_is_exact() {
        init_common();
        let proof = sumcheck(
            Some(Box::new(RoundProof::Sumcheck(sumcheck(
                Some(Box::new(intermediate())),
                false,
            )))),
            true,
        );

        let bytes = to_bytes(&proof);
        let back = from_bytes(&bytes).unwrap();
        assert_eq!(bytes, to_bytes(&back));

        same_ring(&proof.rc_opening_inner, &back.rc_opening_inner);
        same_ring(
            std::slice::from_ref(&proof.norm_claim),
            std::slice::from_ref(&back.norm_claim),
        );
        for (l, r) in proof.polys.iter().zip(&back.polys) {
            assert_eq!(l.num_coefficients, r.num_coefficients);
            assert_eq!(
                l.coefficients[..l.num_coefficients],
                r.coefficients[..r.num_coefficients]
            );
        }
        let (RoundProof::Sumcheck(inner), RoundProof::Sumcheck(inner_back)) = (
            &**proof.next.as_ref().unwrap(),
            &**back.next.as_ref().unwrap(),
        ) else {
            panic!("the chain must survive the roundtrip")
        };
        let (RoundProof::Intermediate(mid), RoundProof::Intermediate(mid_back)) = (
            &**inner.next.as_ref().unwrap(),
            &**inner_back.next.as_ref().unwrap(),
        ) else {
            panic!("the chain must survive the roundtrip")
        };
        let (RoundProof::Simple(last), RoundProof::Simple(last_back)) = (
            &**mid.next.as_ref().unwrap(),
            &**mid_back.next.as_ref().unwrap(),
        ) else {
            panic!("the chain must survive the roundtrip")
        };
        same_ring(&last.folded_witness.data, &last_back.folded_witness.data);
        assert_eq!(
            last.folded_witness.used_cols,
            last_back.folded_witness.used_cols
        );
        same_ring(&mid.opening_rhs.data, &mid_back.opening_rhs.data);
    }

    #[test]
    fn wire_rejects_a_truncated_proof() {
        init_common();
        let bytes = to_bytes(&sumcheck(None, false));
        assert!(from_bytes(&bytes[..bytes.len() - 8]).is_err());
        assert!(from_bytes(&bytes[..12]).is_err());
    }

    /// Decoding recurses once per round, so a chain longer than its allowance is refused at the
    /// first round past it rather than overflowing the stack.
    #[test]
    fn wire_refuses_a_chain_past_its_rounds() {
        init_common();
        let chain = |rounds: usize| {
            let mut proof = sumcheck(None, false);
            for _ in 1..rounds {
                proof = sumcheck(Some(Box::new(RoundProof::Sumcheck(proof))), false);
            }
            to_bytes(&proof)
        };
        let allowance = |rounds| Allowance { rounds, ..FIXTURE };
        assert!(from_bytes_within(&chain(4), allowance(4)).is_ok());
        assert_eq!(
            from_bytes_within(&chain(4), allowance(3)).err(),
            Some(WireError::Malformed)
        );
        let deep = std::thread::Builder::new()
            .stack_size(1 << 28)
            .spawn(move || chain(1000))
            .unwrap()
            .join()
            .unwrap();
        assert_eq!(
            from_bytes_within(&deep, allowance(4)).err(),
            Some(WireError::Malformed)
        );
    }

    /// A zero in a role where zeros dominate codes into next to nothing, so a proof can declare
    /// far more ring elements than it carries; the decoder refuses what its allowance does not
    /// cover before allocating it.
    #[test]
    fn wire_refuses_more_coefficients_than_its_allowance() {
        init_common();
        let mut proof = sumcheck(None, false);
        proof.rc_opening_inner.extend(
            std::iter::repeat_with(|| RingElement::zero(Representation::IncompleteNTT))
                .take(1 << 16),
        );
        assert_eq!(
            from_bytes(&to_bytes(&proof)).err(),
            Some(WireError::Malformed)
        );

        let mut claims = ClaimsProof {
            polys: (2..=3).map(poly).collect(),
            witness_eval: uniform(),
            conj_witness_eval: Some(uniform()),
        };
        let bytes = initial_to_bytes(&claims);
        assert!(initial_from_bytes(2, &bytes).is_ok());
        assert_eq!(
            initial_from_bytes(1, &bytes).err(),
            Some(WireError::Malformed)
        );
        claims.polys = vec![Polynomial::new(0); 1 << 16];
        assert_eq!(
            initial_from_bytes(20, &initial_to_bytes(&claims)).err(),
            Some(WireError::Malformed)
        );
    }

    /// The decoder accepts only the bytes the encoder writes: a change to a coding choice (a
    /// histogram, a section's padding, which elements travel as coefficients, a rANS state) or
    /// to the length is refused unless it changes the proof, and what decodes encodes back to
    /// the bytes it came from.
    #[test]
    fn wire_accepts_only_the_bytes_it_writes() {
        init_common();
        let proof = sumcheck(
            Some(Box::new(RoundProof::Sumcheck(sumcheck(
                Some(Box::new(intermediate())),
                false,
            )))),
            true,
        );
        let bytes = to_bytes(&proof);
        let section =
            |i: usize| u32::from_le_bytes(bytes[1 + 4 * i..5 + 4 * i].try_into().unwrap()) as usize;
        let choices = 13 + section(0) + section(1) + 4 * LANES;
        for bit in 0..8 * choices {
            let mut flipped = bytes.clone();
            flipped[bit / 8] ^= 1 << (bit % 8);
            if let Ok(decoded) = from_bytes(&flipped) {
                assert_eq!(to_bytes(&decoded), flipped, "bit {bit}");
            }
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(from_bytes(&longer).is_err());
    }

    /// The transcript and the arithmetic read a ring element's coefficients and not its label, so
    /// a relabelled element would be a second encoding of the same proof; the decoder accepts
    /// only the representation the protocol holds each role in.
    #[test]
    fn wire_refuses_a_relabelled_ring_element() {
        init_common();
        let relabelled: [fn(&mut SumcheckRoundProof); 3] = [
            |p| p.claim_over_witness.representation = Representation::Coefficients,
            |p| p.norm_claim.representation = Representation::Coefficients,
            |p| {
                let Some(RoundProof::Simple(last)) = p.next.as_deref_mut() else {
                    unreachable!()
                };
                last.projection_image_ct.data[0].representation = Representation::IncompleteNTT;
            },
        ];
        for relabel in relabelled {
            let mut proof = sumcheck(Some(Box::new(simple())), false);
            assert!(from_bytes(&to_bytes(&proof)).is_ok());
            relabel(&mut proof);
            assert_eq!(
                from_bytes(&to_bytes(&proof)).err(),
                Some(WireError::Malformed)
            );
        }

        let mut claims = ClaimsProof {
            polys: (2..=3).map(poly).collect(),
            witness_eval: uniform(),
            conj_witness_eval: Some(uniform()),
        };
        assert!(initial_from_bytes(2, &initial_to_bytes(&claims)).is_ok());
        claims.witness_eval.representation = Representation::Coefficients;
        assert_eq!(
            initial_from_bytes(2, &initial_to_bytes(&claims)).err(),
            Some(WireError::Malformed)
        );
    }

    /// A round polynomial is written at its degree, so one held with a zero leading coefficient
    /// has the same single encoding.
    #[test]
    fn wire_writes_a_round_polynomial_at_its_degree() {
        init_common();
        let proof = sumcheck(None, false);
        let mut padded = from_bytes(&to_bytes(&proof)).unwrap();
        padded.polys[0].num_coefficients += 1;
        assert_eq!(to_bytes(&padded), to_bytes(&proof));
        assert_eq!(
            from_bytes(&to_bytes(&padded)).unwrap().polys[0].num_coefficients,
            proof.polys[0].num_coefficients
        );
    }

    /// A role whose coefficients all fall in one octave, as an all-zero one does, has a model of
    /// one symbol, which costs no bits. Its renormalisation bound overflowed and left the lanes
    /// it shares with the other roles out of step, so such a proof did not decode.
    #[test]
    fn wire_roundtrips_a_role_of_one_octave() {
        init_common();
        let zeroed: [fn(&mut SumcheckRoundProof); 2] = [
            |p| {
                for e in p.rc_opening_inner.iter_mut() {
                    e.set_zero();
                }
            },
            |p| {
                let Some(RoundProof::Simple(last)) = p.next.as_deref_mut() else {
                    unreachable!()
                };
                for e in last.folded_witness.data.iter_mut() {
                    e.set_zero();
                }
            },
        ];
        for zero in zeroed {
            let mut proof = sumcheck(Some(Box::new(simple())), true);
            zero(&mut proof);
            let bytes = to_bytes(&proof);
            assert_eq!(to_bytes(&from_bytes(&bytes).unwrap()), bytes);
        }
    }

    /// A matrix travels with its shape, which must describe its entries.
    #[test]
    fn wire_refuses_a_matrix_shape_that_disagrees_with_its_entries() {
        init_common();
        let tampered: [fn(&mut SimpleRoundProof); 3] = [
            |p| p.batched_projection_image.height = 0,
            |p| p.folded_witness.width += 1,
            |p| p.projection_image_ct.used_cols = p.projection_image_ct.width + 1,
        ];
        for tamper in tampered {
            let mut proof = sumcheck(Some(Box::new(simple())), false);
            let Some(RoundProof::Simple(last)) = proof.next.as_deref_mut() else {
                unreachable!()
            };
            tamper(last);
            assert_eq!(
                from_bytes(&to_bytes(&proof)).err(),
                Some(WireError::Malformed)
            );
        }
    }

    pub(crate) fn mutations(bytes: &[u8], count: usize, seed: u64) -> Vec<Vec<u8>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let choices =
            (13 + spliced_lengths(bytes)[..2].iter().sum::<usize>() + 4 * LANES).min(bytes.len());
        (0..count)
            .map(|i| {
                let mut mutated = bytes.to_vec();
                let at = match i & 3 {
                    0 | 1 => rng.random_range(0..bytes.len()),
                    2 => rng.random_range(0..choices),
                    _ => rng.random_range(bytes.len().saturating_sub(16)..bytes.len()),
                };
                match rng.random_range(0..5) {
                    0 => mutated[at] ^= 1 << rng.random_range(0..8),
                    1 => mutated[at] = rng.random(),
                    2 => mutated.insert(at, rng.random()),
                    3 => drop(mutated.remove(at)),
                    _ => mutated.truncate(at),
                }
                mutated
            })
            .collect()
    }

    fn spliced_lengths(bytes: &[u8]) -> [usize; 3] {
        std::array::from_fn(|i| {
            u32::from_le_bytes(bytes[1 + 4 * i..5 + 4 * i].try_into().unwrap()) as usize
        })
    }

    fn spliced(bytes: &[u8], section: usize, edit: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
        let mut parts = Vec::with_capacity(4);
        let mut at = 13;
        for len in spliced_lengths(bytes) {
            parts.push(bytes[at..at + len].to_vec());
            at += len;
        }
        parts.push(bytes[at..].to_vec());
        edit(&mut parts[section]);
        let mut out = vec![bytes[0]];
        for part in &parts[..3] {
            out.extend_from_slice(&(part.len() as u32).to_le_bytes());
        }
        for part in &parts {
            out.extend_from_slice(part);
        }
        out
    }

    fn bits(writer: &BitWriter) -> usize {
        8 * writer.bytes.len() + writer.n as usize
    }

    fn written(proof: &SumcheckRoundProof) -> [usize; 3] {
        let mut encoder = Encoder::new();
        encoder.sumcheck(proof);
        let models: Vec<Model> = encoder.counts.iter().map(Model::chosen).collect();
        let mut head = BitWriter::default();
        for model in &models {
            model.write(&mut head);
        }
        let raw = encoder
            .cells
            .iter()
            .map(|(c, role)| {
                if models[*role as usize].coded {
                    octave(*c)
                } else {
                    residue_bits() as usize
                }
            })
            .sum();
        [bits(&head), bits(&encoder.shape), raw]
    }

    fn single_octaves() -> SumcheckRoundProof {
        let mut proof = sumcheck(None, false);
        proof.norm_claim.set_zero();
        proof.most_inner_norm_claim.set_zero();
        proof
    }

    fn fixtures() -> Vec<SumcheckRoundProof> {
        let mut fixtures = vec![
            sumcheck(None, false),
            sumcheck(None, true),
            sumcheck(Some(Box::new(simple())), false),
            sumcheck(Some(Box::new(simple())), true),
            sumcheck(Some(Box::new(intermediate())), false),
            sumcheck(
                Some(Box::new(RoundProof::Sumcheck(sumcheck(
                    Some(Box::new(intermediate())),
                    false,
                )))),
                true,
            ),
            single_octaves(),
        ];
        let mut zeroed = sumcheck(Some(Box::new(simple())), true);
        for e in zeroed.rc_opening_inner.iter_mut() {
            e.set_zero();
        }
        let Some(RoundProof::Simple(last)) = zeroed.next.as_deref_mut() else {
            unreachable!()
        };
        for e in last.folded_witness.data.iter_mut() {
            e.set_zero();
        }
        fixtures.push(zeroed);
        fixtures
    }

    fn claims_fixtures() -> Vec<ClaimsProof> {
        vec![
            ClaimsProof {
                polys: (2..=3).map(poly).collect(),
                witness_eval: uniform(),
                conj_witness_eval: Some(uniform()),
            },
            ClaimsProof {
                polys: (1..=3).map(poly).collect(),
                witness_eval: short(1 << 5),
                conj_witness_eval: None,
            },
            ClaimsProof {
                polys: vec![Polynomial::new(0); 2],
                witness_eval: RingElement::zero(Representation::IncompleteNTT),
                conj_witness_eval: Some(short(1 << 20)),
            },
        ]
    }

    #[test]
    fn wire_roundtrips_every_fixture() {
        init_common();
        for proof in fixtures() {
            let bytes = to_bytes(&proof);
            assert_eq!(to_bytes(&from_bytes(&bytes).unwrap()), bytes);
        }
        for claims in claims_fixtures() {
            let bytes = initial_to_bytes(&claims);
            assert_eq!(
                initial_to_bytes(&initial_from_bytes(3, &bytes).unwrap()),
                bytes
            );
        }
    }

    #[test]
    fn wire_accepts_a_mutation_only_as_its_own_encoding() {
        init_common();
        let mut accepted = 0;
        for (seed, proof) in fixtures().iter().enumerate() {
            for mutated in mutations(&to_bytes(proof), 2000, seed as u64) {
                if let Ok(decoded) = from_bytes(&mutated) {
                    assert_eq!(to_bytes(&decoded), mutated);
                    accepted += 1;
                }
            }
        }
        for (seed, claims) in claims_fixtures().iter().enumerate() {
            for mutated in mutations(&initial_to_bytes(claims), 2000, 100 + seed as u64) {
                if let Ok(decoded) = initial_from_bytes(3, &mutated) {
                    assert_eq!(initial_to_bytes(&decoded), mutated);
                    accepted += 1;
                }
            }
        }
        assert!(accepted > 0);
    }

    #[test]
    fn wire_refuses_a_section_padded_otherwise() {
        init_common();
        let mut padded = [0; 3];
        for proof in fixtures() {
            let bytes = to_bytes(&proof);
            assert_eq!(spliced(&bytes, 2, |_| {}), bytes);
            for (i, (section, bits)) in [0, 1, 3].into_iter().zip(written(&proof)).enumerate() {
                assert_eq!(
                    from_bytes(&spliced(&bytes, section, |part| part.push(0))).err(),
                    Some(WireError::Malformed)
                );
                if bits & 7 != 0 {
                    let marked = spliced(&bytes, section, |part| *part.last_mut().unwrap() |= 0x80);
                    assert_eq!(from_bytes(&marked).err(), Some(WireError::Malformed));
                    padded[i] += 1;
                }
            }
            for extra in [&[0u8][..], &[0, 0]] {
                let longer = spliced(&bytes, 2, |rans| rans.extend_from_slice(extra));
                assert_eq!(from_bytes(&longer).err(), Some(WireError::Malformed));
            }
        }
        assert!(padded.iter().all(|&tested| tested > 0), "{padded:?}");
    }

    #[test]
    fn wire_refuses_a_lane_outside_the_encoders_states() {
        init_common();
        let bytes = to_bytes(&single_octaves());
        assert_eq!(spliced_lengths(&bytes)[2], 4 * LANES);
        assert!(from_bytes(&bytes).is_ok());
        let low = spliced(&bytes, 2, |rans| {
            rans[..4].copy_from_slice(&1u32.to_le_bytes());
            rans.extend_from_slice(&[0, 0]);
        });
        let high = spliced(&bytes, 2, |rans| {
            rans[12..16].copy_from_slice(&(RANS_LOW + 1).to_le_bytes())
        });
        for bytes in [low, high] {
            assert_eq!(from_bytes(&bytes).err(), Some(WireError::Malformed));
        }
    }

    #[test]
    fn wire_refuses_a_histogram_other_than_the_encoders() {
        init_common();
        let proof = sumcheck(Some(Box::new(simple())), true);
        let encoder = || {
            let mut encoder = Encoder::new();
            encoder.sumcheck(&proof);
            encoder
        };
        let counts = encoder().counts;
        let chosen: Vec<Model> = counts.iter().map(Model::chosen).collect();
        assert_eq!(encoder().finish_with(chosen.clone()), to_bytes(&proof));
        let coded = (0..ROLES)
            .find(|&r| chosen[r].freq.iter().filter(|&&f| f > 0).count() > 1)
            .unwrap();
        let raw = (0..ROLES)
            .find(|&r| !chosen[r].coded && counts[r].iter().sum::<u64>() > 0)
            .unwrap();

        let mut freq = chosen[coded].freq;
        let top = (0..SYMBOLS).max_by_key(|&s| freq[s]).unwrap();
        let other = (0..SYMBOLS).find(|&s| s != top && freq[s] > 0).unwrap();
        freq[top] -= 1;
        freq[other] += 1;
        let mut requantised = chosen[coded].clone();
        requantised.freq = freq;
        for s in 1..SYMBOLS {
            requantised.cum[s] = requantised.cum[s - 1] + freq[s - 1];
        }

        let mut variants = vec![chosen.clone(); 3];
        variants[0][coded] = Model::raw();
        variants[1][raw] = Model::from_counts(&counts[raw]);
        variants[2][coded] = requantised;
        for models in variants {
            assert_eq!(
                from_bytes(&encoder().finish_with(models)).err(),
                Some(WireError::Malformed)
            );
        }

        let model = &chosen[coded];
        let low = model.freq.iter().position(|&f| f > 0).unwrap();
        let high = model.freq.iter().rposition(|&f| f > 0).unwrap() + 1;
        let mut head = BitWriter::default();
        for (role, model) in chosen.iter().enumerate() {
            if role != coded {
                model.write(&mut head);
                continue;
            }
            head.put(1, 1);
            head.put(low as u64, 6);
            head.put(high as u64, 6);
            for s in low..=high {
                head.put_gamma(model.freq[s] as u64 + 1);
            }
        }
        let wide = spliced(&to_bytes(&proof), 0, |part| *part = head.finish());
        assert_eq!(from_bytes(&wide).err(), Some(WireError::Malformed));
    }

    fn claims_written(
        claims: &ClaimsProof,
        padded: bool,
        len: &[u8],
        uniform: u64,
        flip: bool,
    ) -> Vec<u8> {
        let mut encoder = Encoder::new();
        encoder.shape.put_varint(claims.polys.len() as u64);
        for poly in &claims.polys {
            let written = poly.num_coefficients + padded as usize;
            encoder.shape.put(written as u64, 3);
            for coeff in &poly.coefficients[..written] {
                for v in &coeff.coeffs {
                    encoder.push(Role::Poly, *v);
                }
            }
        }
        for byte in len {
            encoder.shape.put(*byte as u64, 8);
        }
        encoder.shape.put(uniform, 1);
        encoder.shape.put(Representation::IncompleteNTT as u64, 2);
        encoder.shape.put(flip as u64, 1);
        let mut el = claims.witness_eval.clone();
        if flip {
            el.to_representation(Representation::Coefficients);
        }
        for v in &el.v {
            encoder.push(Role::WitnessEval, *v);
        }
        encoder.option_ring(Role::WitnessEval, &claims.conj_witness_eval);
        encoder.finish()
    }

    #[test]
    fn wire_refuses_a_shape_written_otherwise() {
        init_common();
        let claims = |witness_eval| ClaimsProof {
            polys: (2..=3).map(poly).collect(),
            witness_eval,
            conj_witness_eval: Some(uniform()),
        };
        for (claims, flip) in [(claims(uniform()), false), (claims(short(1 << 5)), true)] {
            let canonical = claims_written(&claims, false, &[1], 1, flip);
            assert_eq!(canonical, initial_to_bytes(&claims));
            assert!(initial_from_bytes(2, &canonical).is_ok());
            for written in [
                claims_written(&claims, true, &[1], 1, flip),
                claims_written(&claims, false, &[0x81, 0], 1, flip),
                claims_written(&claims, false, &[1], 0, flip),
                claims_written(&claims, false, &[1], 1, !flip),
            ] {
                assert_eq!(
                    initial_from_bytes(2, &written).err(),
                    Some(WireError::Malformed)
                );
            }
        }

        let proof = sumcheck(Some(Box::new(simple())), false);
        let bytes = to_bytes(&proof);
        let overlong = spliced(&bytes, 1, |shape| {
            shape[0] |= 0x80;
            shape.insert(1, 0);
        });
        assert_eq!(from_bytes(&overlong).err(), Some(WireError::Malformed));

        let Some(RoundProof::Simple(last)) = proof.next.as_deref() else {
            unreachable!()
        };
        let mut tail = Encoder::new();
        tail.matrix_h(Role::BatchedProjection, &last.batched_projection_image);
        tail.matrix_h(Role::OpeningRhs, &last.opening_rhs);
        let at = written(&proof)[1] - bits(&tail.shape) - 1;
        let shape = &bytes[13 + spliced_lengths(&bytes)[0]..];
        assert_eq!(shape[at >> 3] >> (at & 7) & 1, 0);
        let flipped = spliced(&bytes, 1, |shape| shape[at >> 3] ^= 1 << (at & 7));
        assert_eq!(from_bytes(&flipped).err(), Some(WireError::Malformed));
    }

    fn with_cell(proof: &SumcheckRoundProof, coded: bool, c: i64) -> Vec<u8> {
        let mut encoder = Encoder::new();
        encoder.sumcheck(proof);
        let models: Vec<bool> = encoder
            .counts
            .iter()
            .map(|counts| Model::chosen(counts).coded)
            .collect();
        let (old, role) = encoder
            .cells
            .iter_mut()
            .find(|(_, role)| models[*role as usize] == coded)
            .unwrap();
        encoder.counts[*role as usize][octave(*old)] -= 1;
        encoder.counts[*role as usize][octave(c)] += 1;
        *old = c;
        encoder.finish()
    }

    #[test]
    fn wire_refuses_a_coefficient_outside_its_residues() {
        init_common();
        let proof = sumcheck(None, false);
        for coded in [false, true] {
            let bytes = with_cell(&proof, coded, 5);
            assert_eq!(to_bytes(&from_bytes(&bytes).unwrap()), bytes);
        }
        for (coded, c) in [(false, MOD_Q as i64), (true, (MOD_Q / 2 + 1) as i64)] {
            assert_eq!(
                from_bytes(&with_cell(&proof, coded, c)).err(),
                Some(WireError::Malformed)
            );
        }
    }
}
