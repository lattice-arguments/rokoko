//! Proof serialisation. A coefficient is split into its octave `b = bitlen(|c|)`, coded with an
//! interleaved rANS against a histogram measured on the proof being sent — one per field role,
//! quantised and written in the header, so nothing is read from the norm schedule — and a raw
//! `b`-bit field holding the mantissa and the sign. Uniform residues land on the octave law
//! `Pr[b = k] = 2^(k-50)`, of entropy 2, and so pay `ceil(log2 q)` exactly; the folded witness
//! and the norm claims pay their own scale, and zero carries no field at all. Ring elements are
//! coded in the representation they are held in unless the coefficient one saves more than a bit
//! per coefficient, and shapes travel as varints rather than being re-derived from `Config`. At
//! p-26 this is 112.1 KB against the 100.4 KB `size_in_bits` reports, and about 10 kB of that
//! gap is unreachable: that accounting charges `log2|c|`, the ideal code length for a scale-free
//! source, whereas the entropy floor of the coefficients actually sent is 110.7 KB.

use crate::common::config::{DEGREE, MOD_Q};
use crate::common::matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix};
use crate::common::ring_arithmetic::{QuadraticExtension, Representation, RingElement};
use crate::protocol::config::{
    IntermediateRoundProof, NextRoundCommitment, RoundProof, SimpleRoundProof, SumcheckRoundProof,
};
use crate::protocol::snark::ClaimsProof;
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

    fn get_varint(&mut self) -> Result<u64, WireError> {
        let mut value = 0u64;
        let mut shift = 0;
        loop {
            let byte = self.get(8)?;
            value |= (byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
            if shift > 63 {
                return Err(WireError::Malformed);
            }
        }
    }

    fn get_len(&mut self) -> Result<usize, WireError> {
        let len = self.get_varint()? as usize;
        if len > MAX_ELEMENTS {
            return Err(WireError::Malformed);
        }
        Ok(len)
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
        let ceiling = ((RANS_LOW >> SCALE_BITS) << 16) * freq;
        if x >= ceiling {
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
        els.iter()
            .flat_map(|e| e.v.iter())
            .map(|v| octave(centred(*v)))
            .sum()
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

        let resident = Self::ring_cost(els);
        let coefficients = els.len() * DEGREE;
        let mut flip = false;
        if els[0].representation != Representation::Coefficients
            && resident > coefficients * (residue_bits() as usize - 4)
        {
            self.scratch.clear();
            self.scratch.extend(els.iter().map(|e| {
                let mut e = e.clone();
                e.to_representation(Representation::Coefficients);
                e
            }));
            flip = Self::ring_cost(&self.scratch) + coefficients < resident;
        }
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

    fn polys(&mut self, polys: &[Polynomial<QuadraticExtension>]) {
        self.shape.put_varint(polys.len() as u64);
        for poly in polys {
            self.shape.put(poly.num_coefficients as u64, 3);
            for coeff in &poly.coefficients[..poly.num_coefficients] {
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
        let bits = residue_bits();
        let mut head = BitWriter::with_capacity(512);
        let mut models: Vec<Model> = Vec::with_capacity(ROLES);
        for role in 0..ROLES {
            let counts = &self.counts[role];
            let total: u64 = counts.iter().sum();
            if total == 0 {
                models.push(Model::raw());
                Model::raw().write(&mut head);
                continue;
            }
            let candidate = Model::from_counts(counts);
            let coded = candidate.bits(counts) + (SYMBOLS * 16) as f64;
            let model = if coded < 0.98 * (total * bits as u64) as f64 {
                candidate
            } else {
                Model::raw()
            };
            model.write(&mut head);
            models.push(model);
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
    bits: u32,
    index: usize,
}

impl<'a> Body<'a> {
    fn fill(&mut self, model: &Model, out: &mut [u64]) -> Result<(), WireError> {
        if !model.coded {
            for slot in out.iter_mut() {
                let v = self.raw.get_wide(self.bits)?;
                if v >= MOD_Q {
                    return Err(WireError::Malformed);
                }
                *slot = v;
            }
            return Ok(());
        }
        let mut index = self.index;
        for slot in out.iter_mut() {
            let b = self.rans.get(index % LANES, model)?;
            index += 1;
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
        Ok(())
    }
}

struct Decoder<'a> {
    shape: BitReader<'a>,
    body: Body<'a>,
    models: Vec<Model>,
}

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8]) -> Result<Decoder<'a>, WireError> {
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
        Ok(Decoder {
            shape: BitReader::new(shape),
            body: Body {
                raw: BitReader::new(raw),
                rans: RansDecoder::new(rans)?,
                bits: residue_bits(),
                index: 0,
            },
            models,
        })
    }

    fn ring_slice(&mut self, role: Role) -> Result<Vec<RingElement>, WireError> {
        let len = self.shape.get_len()?;
        if len == 0 {
            return Ok(Vec::new());
        }
        let uniform = self.shape.get(1)? != 0;
        let mut representations = Vec::with_capacity(len);
        if uniform {
            let r = representation(self.shape.get(2)?)?;
            representations.resize(len, r);
        } else {
            for _ in 0..len {
                representations.push(representation(self.shape.get(2)?)?);
            }
        }
        let flip = self.shape.get(1)? != 0;

        let mut els = Vec::with_capacity(len.min(4096));
        for representation in representations {
            let mut el = RingElement::new(if flip {
                Representation::Coefficients
            } else {
                representation
            });
            self.body.fill(&self.models[role as usize], &mut el.v)?;
            if flip {
                el.to_representation(representation);
            }
            els.push(el);
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
        Ok(HorizontallyAlignedMatrix {
            data,
            width,
            height,
        })
    }

    fn polys(&mut self) -> Result<Vec<Polynomial<QuadraticExtension>>, WireError> {
        let len = self.shape.get_len()?;
        let mut polys = Vec::with_capacity(len.min(4096));
        for _ in 0..len {
            let num_coefficients = self.shape.get(3)? as usize;
            if num_coefficients > 4 {
                return Err(WireError::Malformed);
            }
            let mut poly = Polynomial::<QuadraticExtension>::new(0);
            poly.num_coefficients = num_coefficients;
            for i in 0..num_coefficients {
                let mut coeffs = [0u64; 2];
                self.body
                    .fill(&self.models[Role::Poly as usize], &mut coeffs)?;
                poly.coefficients[i].coeffs = coeffs;
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
        Ok(Some(Box::new(self.round()?)))
    }
}

fn representation(tag: u64) -> Result<Representation, WireError> {
    Ok(match tag {
        0 => Representation::Coefficients,
        1 => Representation::EvenOddCoefficients,
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

pub fn initial_from_bytes(bytes: &[u8]) -> Result<ClaimsProof, WireError> {
    let mut decoder = Decoder::new(bytes)?;
    Ok(ClaimsProof {
        polys: decoder.polys()?,
        witness_eval: decoder.ring(Role::WitnessEval)?,
        conj_witness_eval: decoder.option_ring(Role::WitnessEval)?,
    })
}

pub fn to_bytes(proof: &SumcheckRoundProof) -> Vec<u8> {
    let mut encoder = Encoder::new();
    encoder.sumcheck(proof);
    encoder.finish()
}

pub fn from_bytes(bytes: &[u8]) -> Result<SumcheckRoundProof, WireError> {
    Decoder::new(bytes)?.sumcheck()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::init_common;
    use crate::common::matrix::ZeroNew;

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

    fn horizontal(height: usize, width: usize) -> HorizontallyAlignedMatrix<RingElement> {
        let mut m = HorizontallyAlignedMatrix::new_zero(height, width, &uniform());
        for slot in m.data.iter_mut() {
            *slot = uniform();
        }
        m.data[0] = RingElement::zero(Representation::IncompleteNTT);
        m
    }

    fn poly(num_coefficients: usize) -> Polynomial<QuadraticExtension> {
        let mut poly = Polynomial::<QuadraticExtension>::new(0);
        poly.num_coefficients = num_coefficients;
        for i in 0..num_coefficients {
            poly.coefficients[i] = QuadraticExtension {
                coeffs: [uniform().v[0], uniform().v[1]],
            };
        }
        poly
    }

    fn simple() -> RoundProof {
        RoundProof::Simple(SimpleRoundProof {
            folded_witness: vertical(4, 3, 1 << 11),
            projection_image_ct: vertical(2, 2, 1 << 5),
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
            projection_image_ct: vertical(2, 1, 1 << 5),
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
}
