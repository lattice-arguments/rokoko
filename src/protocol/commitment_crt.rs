//! The basic commitment over a CRT basis of 16-bit NTT primes instead of `q`. `A w` is computed
//! exactly over the integers modulo `Q = prod p_i`, one 128-point negacyclic NTT per prime with
//! the slot products accumulated by VNNI, and reduced to `q` once at the end. The limb count
//! comes from a statistical bound on `|A w|` and the digits' measured norm, not from the
//! worst-case one, and every reconstructed coefficient is checked against that bound.

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use core::arch::x86_64::*;

use crate::common::structured_row::PreprocessedRow;
use crate::common::{
    arithmetic::centered_i16_from_u64_mod_q,
    config::{DEGREE, HALF_DEGREE, MOD_Q},
    matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
    ring_arithmetic::{Representation, RingElement},
};
use crate::hexl::bindings::ntt_inverse;
use crate::protocol::{
    commitment::BasicCommitment,
    crs::{CK, CRS},
    project_coarse::{prepare_i16_witness, Signed16RingElement},
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub const PRIMES: [i32; 8] = [3329, 7681, 7937, 9473, 10753, 11777, 12289, 13313];

const SLOTS: usize = HALF_DEGREE;
const STAGES: u32 = 6;

fn power(base: i64, exponent: u64, modulus: i64) -> i64 {
    let mut result = 1i64;
    let mut base = base % modulus;
    let mut exponent = exponent;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = result * base % modulus;
        }
        base = base * base % modulus;
        exponent >>= 1;
    }
    result
}

fn generator(p: i64) -> i64 {
    let factors: Vec<i64> = {
        let mut n = p - 1;
        let mut factors = vec![];
        let mut d = 2;
        while d * d <= n {
            if n % d == 0 {
                factors.push(d);
                while n % d == 0 {
                    n /= d;
                }
            }
            d += 1;
        }
        if n > 1 {
            factors.push(n);
        }
        factors
    };
    (2..p)
        .find(|g| {
            factors
                .iter()
                .all(|f| power(*g, ((p - 1) / f) as u64, p) != 1)
        })
        .expect("a prime field has a generator")
}

#[derive(Clone)]
pub struct Limb {
    pub p: i32,
    pinv: i16,
    barrett: i32,
    /// `floor(2^62 / p)` and `floor(2^42 / p)`: Barrett reciprocals for the two widths that need
    /// reducing, a key coefficient centred mod `q` and a VNNI accumulator lane.
    reciprocal_i64: i64,
    shift: i16,
    half: i16,
    inverse_zetas: [i16; SLOTS],
    stage_zetas: [[i16; DEGREE]; STAGES as usize],
    stage_shoup: [[i16; DEGREE]; STAGES as usize],
    slot_pairs: [i16; DEGREE],
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    reciprocal_i32: i32,
    scale: i16,
    pub depth: usize,
    /// Butterfly stages the `i16` lanes survive between reductions: `|r|` grows by `p` a stage.
    pub stride: u32,
}

fn montgomery_one(modulus: i64) -> i16 {
    let lifted = (1i64 << 16) % modulus;
    (if lifted > modulus / 2 {
        lifted - modulus
    } else {
        lifted
    }) as i16
}

fn stride(p: i32) -> u32 {
    ((i16::MAX as f64 / p as f64 - 0.5) as u32).clamp(1, STAGES)
}

fn stage_offset(stage: u32) -> usize {
    (1usize << stage) - 1
}

impl Limb {
    pub fn new(p: i32) -> Limb {
        let modulus = p as i64;
        let root = power(generator(modulus), ((modulus - 1) / 256) as u64, modulus);
        let montgomery = |x: i64| -> i16 {
            let lifted = (x.rem_euclid(modulus) << 16) % modulus;
            let centred = if lifted > modulus / 2 {
                lifted - modulus
            } else {
                lifted
            };
            centred as i16
        };

        let mut pinv = 1i64;
        for _ in 0..4 {
            pinv = pinv * (2 - modulus * pinv) % 65536;
        }

        let mut inverse_zetas = [0i16; SLOTS];
        let mut block_exponents: Vec<Vec<u64>> = Vec::new();
        let mut exponents = vec![128u64];
        for stage in 0..STAGES {
            block_exponents.push(exponents.clone());
            let mut next = Vec::with_capacity(exponents.len() * 2);
            for (block, exponent) in exponents.iter().enumerate() {
                let half = exponent / 2;
                inverse_zetas[stage_offset(stage) + block] =
                    montgomery(power(root, 256 - half, modulus));
                next.push(half);
                next.push(half + 128);
            }
            exponents = next;
        }

        let mut slot_pairs = [0i16; DEGREE];
        for (slot, exponent) in exponents.iter().enumerate() {
            slot_pairs[2 * slot] = montgomery_one(modulus);
            slot_pairs[2 * slot + 1] = montgomery(power(root, *exponent, modulus));
        }

        let mut stage_zetas = [[0i16; DEGREE]; STAGES as usize];
        let mut stage_shoup = [[0i16; DEGREE]; STAGES as usize];
        for stage in 0..STAGES as usize {
            let len = DEGREE >> (stage + 1);
            for i in 0..DEGREE {
                let block = i / (2 * len);
                let high = i % (2 * len) >= len;
                let exponent = block_exponents[stage][block] / 2;
                let plain = {
                    let value = power(root, exponent, modulus);
                    let centred = if value > modulus / 2 {
                        value - modulus
                    } else {
                        value
                    };
                    (if high { -centred } else { centred }) as i16
                };
                stage_zetas[stage][i] = plain;
                stage_shoup[stage][i] =
                    ((plain as i64 * 32768 + modulus / 2 * plain.signum() as i64) / modulus) as i16;
            }
        }

        let centre = |x: i64| -> i16 {
            let reduced = x.rem_euclid(modulus);
            (if reduced > modulus / 2 {
                reduced - modulus
            } else {
                reduced
            }) as i16
        };

        Limb {
            p,
            pinv: (pinv as u64 as u16) as i16,
            barrett: (((1i64 << 26) + modulus / 2) / modulus) as i32,
            reciprocal_i64: ((1i128 << 62) / modulus as i128) as i64,
            shift: centre(1i64 << 32),
            half: centre(32768),
            inverse_zetas,
            stage_zetas,
            stage_shoup,
            slot_pairs,
            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
            reciprocal_i32: ((1i64 << 42) / modulus) as i32,
            scale: montgomery(power(SLOTS as i64, modulus as u64 - 2, modulus)),
            depth: ((1u64 << 31) / (p as u64 * p as u64 / 2)) as usize,
            stride: stride(p),
        }
    }

    #[inline]
    fn montgomery(&self, a: i32) -> i16 {
        let t = (a as i16).wrapping_mul(self.pinv);
        ((a - t as i32 * self.p) >> 16) as i16
    }

    #[inline]
    fn mul(&self, a: i16, b: i16) -> i16 {
        self.montgomery(a as i32 * b as i32)
    }

    #[inline]
    fn shoup(&self, x: i16, zeta: i16, quotient: i16) -> i16 {
        let q = ((x as i32 * quotient as i32 + (1 << 14)) >> 15) as i16;
        x.wrapping_mul(zeta)
            .wrapping_sub(q.wrapping_mul(self.p as i16))
    }

    #[inline]
    fn centre(&self, a: i16) -> i16 {
        let t = ((a as i32 * self.barrett) >> 16) as i16;
        let t = (t + 512) >> 10;
        a.wrapping_sub(t.wrapping_mul(self.p as i16))
    }

    #[inline]
    fn reduce_wide(&self, value: i64) -> i16 {
        let quotient = ((value as i128 * self.reciprocal_i64 as i128) >> 62) as i64;
        self.centre((value - quotient * self.p as i64) as i16)
    }

    #[inline]
    fn centre32(&self, a: i32) -> i16 {
        let high = (a >> 16) as i16;
        let low = (a as u16 as i32 - 32768) as i16;
        self.centre(self.mul(high, self.shift) + self.centre(low) + self.half)
    }

    pub fn ntt(&self, r: &mut [i16; DEGREE]) {
        let mut len = DEGREE / 2;
        for stage in 0..STAGES {
            for start in (0..DEGREE).step_by(2 * len) {
                let zeta = self.stage_zetas[stage as usize][start];
                let quotient = self.stage_shoup[stage as usize][start];
                for j in start..start + len {
                    let t = self.shoup(r[j + len], zeta, quotient);
                    r[j + len] = r[j] - t;
                    r[j] = r[j] + t;
                }
            }
            len /= 2;
            if (stage + 1) % self.stride == 0 {
                for slot in r.iter_mut() {
                    *slot = self.centre(*slot);
                }
            }
        }
        for slot in r.iter_mut() {
            *slot = self.centre(*slot);
        }
    }

    pub fn inverse_ntt(&self, r: &mut [i16; DEGREE]) {
        let mut len = 2;
        for stage in (0..STAGES).rev() {
            for (block, start) in (0..DEGREE).step_by(2 * len).enumerate() {
                let zeta = self.inverse_zetas[stage_offset(stage) + block];
                for j in start..start + len {
                    let t = r[j];
                    r[j] = self.centre(t + r[j + len]);
                    r[j + len] = self.mul(zeta, t - r[j + len]);
                }
            }
            len *= 2;
        }
        for slot in r.iter_mut() {
            *slot = self.mul(*slot, self.scale);
        }
    }

    #[cfg(any(test, not(all(target_arch = "x86_64", target_feature = "avx512f"))))]
    fn arrange(&self, a: &[i16; DEGREE]) -> [i16; DEGREE] {
        std::array::from_fn(|i| self.centre(self.mul(a[i], self.slot_pairs[i])))
    }
}

pub struct Plan {
    pub primes: Vec<i32>,
    pub bound: f64,
}

fn limb_cost(p: i32, rank: usize) -> f64 {
    let depth = ((1u64 << 31) / (p as u64 * p as u64 / 2)) as f64;
    const TRANSFORM: f64 = 100.0;
    const PASS: f64 = 16.0;
    const PRODUCTS: f64 = 16.0;
    const REDUCTION: f64 = 80.0;
    TRANSFORM + PASS * (STAGES / stride(p)) as f64 + rank as f64 * (PRODUCTS + REDUCTION / depth)
}

impl Plan {
    pub fn new(digits_l2: f64, rank: usize) -> Plan {
        const MARGIN: f64 = 32.0;
        let bound = MARGIN * MOD_Q as f64 * digits_l2 / 12f64.sqrt();
        let wanted = (2.0 * bound).log2();

        let mut best: Option<(f64, Vec<i32>)> = None;
        for mask in 1u32..1 << PRIMES.len() {
            let primes: Vec<i32> = (0..PRIMES.len())
                .filter(|i| mask >> i & 1 == 1)
                .map(|i| PRIMES[i])
                .collect();
            let bits: f64 = primes.iter().map(|p| (*p as f64).log2()).sum();
            if bits < wanted {
                continue;
            }
            let cost: f64 = primes.iter().map(|p| limb_cost(*p, rank)).sum();
            if best.as_ref().is_none_or(|(low, _)| cost < *low) {
                best = Some((cost, primes));
            }
        }
        let (_, primes) = best.expect("the prime set must cover the bound");
        Plan { primes, bound }
    }

    /// The same bound from the schedule rather than from the digits: a balanced base-`2 bound`
    /// decomposition leaves digits uniform over `[-bound, bound)`, so `||w||_2` is
    /// `sqrt(rows * DEGREE * bound^2 / 3)`. Lets the key be preprocessed before a witness exists;
    /// the per-coefficient check still catches a proof whose digits come out wider.
    pub fn for_shape(rows: usize, bound: u64, rank: usize) -> Plan {
        let square = (rows * DEGREE) as f64 * (bound * bound) as f64 / 3.0;
        Plan::new(square.sqrt(), rank)
    }

    pub fn limbs(&self) -> Vec<Limb> {
        self.primes.iter().map(|p| Limb::new(*p)).collect()
    }
}

pub fn digits_l2(digits: &VerticallyAlignedMatrix<Signed16RingElement>) -> f64 {
    let mut widest = 0u128;
    for col in 0..digits.used_cols {
        let mut square = 0u128;
        for element in &digits.data[col * digits.height..][..digits.height] {
            square += element
                .0
                .iter()
                .map(|c| (*c as i64 * *c as i64) as u128)
                .sum::<u128>();
        }
        widest = widest.max(square);
    }
    (widest as f64).sqrt()
}

pub struct CrtKey {
    pub limbs: Vec<Limb>,
    pub rows: usize,
    pub n: usize,
    data: Vec<i16>,
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn transform_rows(limb: &Limb, source: &[i16], out: &mut [i16]) {
    let mut k = 0;
    let rows = out.len() / (2 * DEGREE);
    while k + 4 <= rows {
        unsafe {
            transform::<4, 2>(
                limb,
                source.as_ptr().add(k * DEGREE),
                out.as_mut_ptr().add(2 * k * DEGREE),
            )
        };
        k += 4;
    }
    while k < rows {
        unsafe {
            transform::<1, 2>(
                limb,
                source.as_ptr().add(k * DEGREE),
                out.as_mut_ptr().add(2 * k * DEGREE),
            )
        };
        k += 1;
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
fn transform_rows(limb: &Limb, source: &[i16], out: &mut [i16]) {
    for k in 0..out.len() / (2 * DEGREE) {
        let mut z = natural(&Signed16RingElement(
            source[k * DEGREE..(k + 1) * DEGREE].try_into().unwrap(),
        ));
        for slot in z.iter_mut() {
            *slot = limb.centre(*slot);
        }
        limb.ntt(&mut z);
        let scaled = limb.arrange(&z);
        out[2 * k * DEGREE..(2 * k + 1) * DEGREE].copy_from_slice(&scaled);
        out[(2 * k + 1) * DEGREE..(2 * k + 2) * DEGREE].copy_from_slice(&z);
    }
}

fn centre_row(key: &PreprocessedRow, out: &mut [i64]) {
    for (k, element) in key.preprocessed_row.iter().enumerate() {
        let mut even_odd = element.clone();
        even_odd.to_representation(Representation::EvenOddCoefficients);
        for (slot, value) in even_odd.v.iter().enumerate() {
            out[k * DEGREE + slot] = if *value > MOD_Q / 2 {
                *value as i64 - MOD_Q as i64
            } else {
                *value as i64
            };
        }
    }
}

fn narrow_row(limb: &Limb, centred: &[i64], out: &mut [i16]) {
    for (slot, value) in centred.iter().enumerate() {
        out[slot] = limb.reduce_wide(*value);
    }
}

impl CrtKey {
    pub fn preprocess(ck: &CK, rank: usize, plan: &Plan) -> CrtKey {
        CrtKey::preprocess_with(ck, rank, plan, cfg!(feature = "parallel"))
    }

    /// Each (limb, row) block is an independent transform of the same centred key row, so the
    /// rows are centred once and the blocks then filled in any order.
    fn preprocess_with(ck: &CK, rank: usize, plan: &Plan, parallel: bool) -> CrtKey {
        let limbs = plan.limbs();
        let n = ck[0].preprocessed_row.len();
        let mut data = vec![0i16; limbs.len() * rank * n * 2 * DEGREE];

        #[cfg(feature = "parallel")]
        if parallel {
            let mut centred = vec![0i64; rank * n * DEGREE];
            centred
                .par_chunks_mut(n * DEGREE)
                .enumerate()
                .for_each(|(row, out)| centre_row(&ck[row], out));
            data.par_chunks_mut(2 * n * DEGREE)
                .enumerate()
                .for_each_init(
                    || vec![0i16; n * DEGREE],
                    |narrowed, (block, out)| {
                        let limb = &limbs[block / rank];
                        narrow_row(
                            limb,
                            &centred[(block % rank) * n * DEGREE..][..n * DEGREE],
                            narrowed,
                        );
                        transform_rows(limb, narrowed, out);
                    },
                );
            return CrtKey {
                limbs,
                rows: rank,
                n,
                data,
            };
        }
        let _ = parallel;

        let mut centred = vec![0i64; n * DEGREE];
        let mut narrowed = vec![0i16; n * DEGREE];
        for row in 0..rank {
            centre_row(&ck[row], &mut centred);
            for (index, limb) in limbs.iter().enumerate() {
                narrow_row(limb, &centred, &mut narrowed);
                let at = ((index * rank + row) * n) * 2 * DEGREE;
                transform_rows(limb, &narrowed, &mut data[at..at + 2 * n * DEGREE]);
            }
        }

        CrtKey {
            limbs,
            rows: rank,
            n,
            data,
        }
    }

    pub fn bytes(&self) -> usize {
        self.data.len() * 2
    }

    fn row(&self, limb: usize, row: usize, k: usize) -> &[i16] {
        let at = ((limb * self.rows + row) * self.n + k) * 2 * DEGREE;
        &self.data[at..at + 2 * DEGREE]
    }
}

fn natural(element: &Signed16RingElement) -> [i16; DEGREE] {
    let mut coefficients = [0i16; DEGREE];
    for i in 0..HALF_DEGREE {
        coefficients[2 * i] = element.0[i];
        coefficients[2 * i + 1] = element.0[HALF_DEGREE + i];
    }
    coefficients
}

/// The same call as `commitment::commit_basic`, planning the limbs and preprocessing the key on
/// the spot. Repeated commitments against one key should keep [`Plan`] and [`CrtKey`] instead.
pub fn commit_basic(
    crs: &CRS,
    witness: &VerticallyAlignedMatrix<RingElement>,
    rank: usize,
) -> BasicCommitment {
    let digits = prepare_i16_witness(witness);
    let plan = Plan::new(digits_l2(&digits), rank);
    let key = CrtKey::preprocess(crs.ck_for_wit_dim(witness.height), rank, &plan);
    commit_basic_crt(&key, &digits, &plan, rank, 1)
}

/// A block-diagonal commitment reads the same buffer as `blocks` times as many columns, each
/// `blocks` times shorter: a block of a column is a contiguous run. The result is that reshaped
/// commitment, `rank` rows by `width * blocks` columns.
fn blocked(height: usize, width: usize, used_cols: usize, blocks: usize) -> (usize, usize, usize) {
    debug_assert_eq!(height % blocks, 0);
    (height / blocks, width * blocks, used_cols * blocks)
}

pub fn commit_basic_crt(
    key: &CrtKey,
    digits: &VerticallyAlignedMatrix<Signed16RingElement>,
    plan: &Plan,
    rank: usize,
    blocks: usize,
) -> BasicCommitment {
    let residues = residues(key, digits, rank, blocks);
    reconstruct(key, plan, rank, digits.width * blocks, &residues)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn residues(
    key: &CrtKey,
    digits: &VerticallyAlignedMatrix<Signed16RingElement>,
    rank: usize,
    blocks: usize,
) -> Vec<i16> {
    let (n, width, used_cols) = blocked(digits.height, digits.width, digits.used_cols, blocks);
    residues_by_tile(
        key,
        rank,
        width,
        n,
        used_cols,
        Source::Narrowed(digits),
        cfg!(feature = "parallel"),
    )
}

/// Without AVX-512 the reference stands in: correct, and slower by the factor the kernels win.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
fn residues(
    key: &CrtKey,
    digits: &VerticallyAlignedMatrix<Signed16RingElement>,
    rank: usize,
    blocks: usize,
) -> Vec<i16> {
    reference_residues(key, digits, rank, blocks)
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub fn commit_basic_crt_streaming(
    key: &CrtKey,
    witness: &VerticallyAlignedMatrix<RingElement>,
    plan: &Plan,
    rank: usize,
    blocks: usize,
) -> BasicCommitment {
    let mut digits = vec![Signed16RingElement([0i16; DEGREE]); witness.data.len()];
    narrow(&witness.data, &mut digits);
    let digits = VerticallyAlignedMatrix {
        data: digits,
        width: witness.width,
        height: witness.height,
        used_cols: witness.used_cols,
    };
    commit_basic_crt(key, &digits, plan, rank, blocks)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
enum Source<'a> {
    Narrowed(&'a VerticallyAlignedMatrix<Signed16RingElement>),
    Ring(&'a VerticallyAlignedMatrix<RingElement>),
}

/// The same, narrowing one column tile of the ring witness at a time so that the whole i16 copy
/// never exists: the peak is `COLUMNS * n` elements instead of `width * n`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub fn commit_basic_crt_streaming(
    key: &CrtKey,
    witness: &VerticallyAlignedMatrix<RingElement>,
    plan: &Plan,
    rank: usize,
    blocks: usize,
) -> BasicCommitment {
    let (n, width, used_cols) = blocked(witness.height, witness.width, witness.used_cols, blocks);
    let residues = residues_by_tile(
        key,
        rank,
        width,
        n,
        used_cols,
        Source::Ring(witness),
        cfg!(feature = "parallel"),
    );
    reconstruct(key, plan, rank, width, &residues)
}

fn narrow(source: &[RingElement], out: &mut [Signed16RingElement]) {
    #[repr(align(64))]
    struct Buffer([u64; DEGREE]);
    let mut coefficients = Buffer([0u64; DEGREE]);
    for (element, slot) in source.iter().zip(out.iter_mut()) {
        debug_assert_eq!(element.representation, Representation::IncompleteNTT);
        unsafe {
            ntt_inverse(
                coefficients.0.as_mut_ptr(),
                element.v.as_ptr(),
                HALF_DEGREE,
                MOD_Q,
            );
            ntt_inverse(
                coefficients.0.as_mut_ptr().add(HALF_DEGREE),
                element.v.as_ptr().add(HALF_DEGREE),
                HALF_DEGREE,
                MOD_Q,
            );
        }
        centered_i16_from_u64_mod_q(&mut slot.0, &coefficients.0);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
struct GroupScratch {
    buffer: Vec<i16>,
    accumulators: Vec<i32>,
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl GroupScratch {
    fn new(rank: usize, columns_per_tile: usize) -> GroupScratch {
        GroupScratch {
            buffer: vec![0i16; CHUNK * columns_per_tile * DEGREE],
            accumulators: vec![0i32; GROUP * rank * columns_per_tile * DEGREE],
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn tile_digits<'a>(
    source: &'a Source,
    scratch: &'a mut [Signed16RingElement],
    n: usize,
    at: usize,
    columns: usize,
) -> &'a [Signed16RingElement] {
    match source {
        Source::Narrowed(digits) => &digits.data[at * n..(at + columns) * n],
        Source::Ring(witness) => {
            narrow(
                &witness.data[at * n..(at + columns) * n],
                &mut scratch[..columns * n],
            );
            &scratch[..columns * n]
        }
    }
}

/// A tile's `(block, column)` residues into the flat `(block, column)` array of full width.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn scatter(
    tile: &[i16],
    residues: &mut [i16],
    blocks: usize,
    width: usize,
    columns_per_tile: usize,
    at: usize,
    columns: usize,
) {
    for block in 0..blocks {
        let from = block * columns_per_tile * DEGREE;
        let to = (block * width + at) * DEGREE;
        residues[to..to + columns * DEGREE].copy_from_slice(&tile[from..from + columns * DEGREE]);
    }
}

/// One limb group of one column tile. Every (row, column) accumulates over `k` in ascending
/// order regardless of who computes it, so which thread takes which group does not affect the
/// result.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn accumulate_group(
    key: &CrtKey,
    first: usize,
    rank: usize,
    n: usize,
    columns: usize,
    columns_per_tile: usize,
    tile_digits: &[Signed16RingElement],
    scratch: &mut GroupScratch,
    out: &mut [i16],
) {
    let group = &key.limbs[first..(first + GROUP).min(key.limbs.len())];
    let period = group
        .iter()
        .map(|limb| limb.depth / CHUNK * CHUNK)
        .min()
        .unwrap_or(CHUNK);
    let stride = rank * columns_per_tile * DEGREE;
    let buffer = &mut scratch.buffer;
    let accumulators = &mut scratch.accumulators;
    accumulators.fill(0);

    for base in (0..n).step_by(CHUNK) {
        let taken = (n - base).min(CHUNK);
        for (offset, limb) in group.iter().enumerate() {
            for c in 0..columns {
                let source = tile_digits[c * n + base].0.as_ptr();
                let out = unsafe { buffer.as_mut_ptr().add(c * CHUNK * DEGREE) };
                let mut k = 0;
                while k + 4 <= taken {
                    unsafe { transform::<4, 0>(limb, source.add(k * DEGREE), out.add(k * DEGREE)) };
                    k += 4;
                }
                while k < taken {
                    unsafe { transform::<1, 0>(limb, source.add(k * DEGREE), out.add(k * DEGREE)) };
                    k += 1;
                }
            }

            let mut row = 0;
            while row < rank {
                let paired = row + 1 < rank;
                for c in 0..columns {
                    let at: [*mut i32; 2] = std::array::from_fn(|r| unsafe {
                        accumulators.as_mut_ptr().add(
                            offset * stride
                                + ((row + r * paired as usize) * columns_per_tile + c) * DEGREE,
                        )
                    });
                    let mut acc = unsafe { load(at.map(|p| p as *const i32)) };
                    let bases: [*const i16; 2] = std::array::from_fn(|r| {
                        key.row(first + offset, row + r * paired as usize, base)
                            .as_ptr()
                    });
                    let operands = unsafe { buffer.as_ptr().add(c * CHUNK * DEGREE) };
                    for k in 0..taken {
                        let keys: [*const i16; 2] =
                            std::array::from_fn(|r| unsafe { bases[r].add(2 * k * DEGREE) });
                        let operand = unsafe { operands.add(k * DEGREE) };
                        unsafe {
                            if paired {
                                accumulate::<2>(&mut acc, operand, keys)
                            } else {
                                accumulate::<1>(&mut acc, operand, [keys[0]])
                            }
                        };
                    }
                    unsafe {
                        if paired {
                            store::<2>(&acc, at)
                        } else {
                            store::<1>(&acc, [at[0]])
                        }
                    };
                }
                row += 1 + paired as usize;
            }
        }

        if (base + CHUNK) % period == 0 {
            for (offset, limb) in group.iter().enumerate() {
                unsafe {
                    reduce(
                        limb,
                        &mut accumulators[offset * stride..(offset + 1) * stride],
                    )
                };
            }
        }
    }

    for (offset, limb) in group.iter().enumerate() {
        unsafe {
            reduce(
                limb,
                &mut accumulators[offset * stride..(offset + 1) * stride],
            )
        };
        for row in 0..rank {
            for c in 0..columns {
                let mut image = [0i16; DEGREE];
                gather(
                    limb,
                    &accumulators[offset * stride + (row * columns_per_tile + c) * DEGREE..],
                    &mut image,
                );
                limb.inverse_ntt(&mut image);
                let at = ((offset * rank + row) * columns_per_tile + c) * DEGREE;
                out[at..at + DEGREE].copy_from_slice(&image);
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn residues_by_tile(
    key: &CrtKey,
    rank: usize,
    width: usize,
    n: usize,
    used_cols: usize,
    source: Source,
    parallel: bool,
) -> Vec<i16> {
    use {COLUMNS, GROUP};

    let limbs = key.limbs.len();
    let columns_per_tile = COLUMNS.min(width.next_power_of_two());
    let stride = rank * columns_per_tile * DEGREE;
    let mut residues = vec![0i16; limbs * rank * width * DEGREE];
    let held = match source {
        Source::Ring(_) => columns_per_tile * n,
        Source::Narrowed(_) => 0,
    };
    let tiles: Vec<(usize, usize)> = (0..used_cols)
        .step_by(columns_per_tile)
        .map(|at| (at, (used_cols - at).min(columns_per_tile)))
        .collect();

    // Groups fan out under the tiles rather than beside them: the narrowed witness a tile's
    // groups share is then held once per tile in flight instead of once per task.
    #[cfg(feature = "parallel")]
    if parallel {
        let computed: Vec<Vec<i16>> = tiles
            .par_iter()
            .map_init(
                || vec![Signed16RingElement([0i16; DEGREE]); held],
                |scratch, &(at, columns)| {
                    let digits = tile_digits(&source, scratch, n, at, columns);
                    let mut tile = vec![0i16; limbs * stride];
                    tile.par_chunks_mut(GROUP * stride)
                        .enumerate()
                        .for_each_init(
                            || GroupScratch::new(rank, columns_per_tile),
                            |scratch, (group, out)| {
                                accumulate_group(
                                    key,
                                    group * GROUP,
                                    rank,
                                    n,
                                    columns,
                                    columns_per_tile,
                                    digits,
                                    scratch,
                                    out,
                                )
                            },
                        );
                    tile
                },
            )
            .collect();
        for (&(at, columns), tile) in tiles.iter().zip(&computed) {
            scatter(
                tile,
                &mut residues,
                limbs * rank,
                width,
                columns_per_tile,
                at,
                columns,
            );
        }
        return residues;
    }
    let _ = parallel;

    let mut scratch = vec![Signed16RingElement([0i16; DEGREE]); held];
    let mut group_scratch = GroupScratch::new(rank, columns_per_tile);
    let mut tile = vec![0i16; limbs * stride];
    for &(at, columns) in &tiles {
        let digits = tile_digits(&source, &mut scratch, n, at, columns);
        for (first, out) in (0..limbs)
            .step_by(GROUP)
            .zip(tile.chunks_mut(GROUP * stride))
        {
            accumulate_group(
                key,
                first,
                rank,
                n,
                columns,
                columns_per_tile,
                digits,
                &mut group_scratch,
                out,
            );
        }
        scatter(
            &tile,
            &mut residues,
            limbs * rank,
            width,
            columns_per_tile,
            at,
            columns,
        );
    }
    residues
}

pub fn reference_residues(
    key: &CrtKey,
    digits: &VerticallyAlignedMatrix<Signed16RingElement>,
    rank: usize,
    blocks: usize,
) -> Vec<i16> {
    let (n, width, used_cols) = blocked(digits.height, digits.width, digits.used_cols, blocks);
    let mut residues = vec![0i16; key.limbs.len() * rank * width * DEGREE];

    for (index, limb) in key.limbs.iter().enumerate() {
        let mut accumulators = vec![0i32; rank * width * DEGREE];
        for col in 0..used_cols {
            for k in 0..n {
                let mut z = natural(&digits.data[col * n + k]);
                for slot in z.iter_mut() {
                    *slot = limb.centre(*slot);
                }
                limb.ntt(&mut z);
                for row in 0..rank {
                    let key_row = key.row(index, row, k);
                    let at = (row * width + col) * DEGREE;
                    let accumulator = &mut accumulators[at..at + DEGREE];
                    for slot in 0..SLOTS {
                        accumulator[2 * slot] = limb.centre32(
                            accumulator[2 * slot]
                                + z[2 * slot] as i32 * key_row[2 * slot] as i32
                                + z[2 * slot + 1] as i32 * key_row[2 * slot + 1] as i32,
                        ) as i32;
                        accumulator[2 * slot + 1] = limb.centre32(
                            accumulator[2 * slot + 1]
                                + z[2 * slot + 1] as i32 * key_row[DEGREE + 2 * slot] as i32
                                + z[2 * slot] as i32 * key_row[DEGREE + 2 * slot + 1] as i32,
                        ) as i32;
                    }
                }
            }
        }

        for row in 0..rank {
            for col in 0..width {
                let at = (row * width + col) * DEGREE;
                let mut image = [0i16; DEGREE];
                for slot in 0..DEGREE {
                    image[slot] = limb.centre32(accumulators[at + slot]);
                }
                limb.inverse_ntt(&mut image);
                let out = ((index * rank + row) * width + col) * DEGREE;
                residues[out..out + DEGREE].copy_from_slice(&image);
            }
        }
    }
    residues
}

fn reconstruct(
    key: &CrtKey,
    plan: &Plan,
    rank: usize,
    width: usize,
    residues: &[i16],
) -> BasicCommitment {
    let limbs: Vec<i64> = key.limbs.iter().map(|limb| limb.p as i64).collect();
    let mut inverses = vec![vec![0i64; limbs.len()]; limbs.len()];
    for i in 0..limbs.len() {
        for j in 0..i {
            inverses[i][j] = power(limbs[j] % limbs[i], (limbs[i] - 2) as u64, limbs[i]);
        }
    }
    let modulus: i128 = limbs.iter().map(|p| *p as i128).product();
    let bound = plan.bound;

    let mut commitment = HorizontallyAlignedMatrix {
        data: vec![
            RingElement::zero(Representation::IncompleteNTT);
            rank.next_power_of_two() * width
        ],
        width,
        height: rank.next_power_of_two(),
    };

    for row in 0..rank {
        for col in 0..width {
            let mut element = RingElement::new(Representation::Coefficients);
            for slot in 0..DEGREE {
                let mut digits = [0i64; PRIMES.len()];
                for i in 0..limbs.len() {
                    let at = ((i * rank + row) * width + col) * DEGREE + slot;
                    let mut value = residues[at] as i64 % limbs[i];
                    for j in 0..i {
                        value = (value - digits[j]) * inverses[i][j] % limbs[i];
                    }
                    digits[i] = value.rem_euclid(limbs[i]);
                }
                let mut lifted = 0i128;
                for i in (0..limbs.len()).rev() {
                    lifted = lifted * limbs[i] as i128 + digits[i] as i128;
                }
                if lifted > modulus / 2 {
                    lifted -= modulus;
                }
                assert!(
                    (lifted.unsigned_abs() as f64) <= bound,
                    "the commitment wrapped the CRT basis: |{}| exceeds {:.3e}",
                    lifted,
                    bound
                );
                element.v[slot] = lifted.rem_euclid(MOD_Q as i128) as u64;
            }
            element.to_representation(Representation::IncompleteNTT);
            commitment.data[row * width + col] = element;
        }
    }

    commitment
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const CHUNK: usize = 12;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const GROUP: usize = 2;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const COLUMNS: usize = 32;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn montgomery(a: __m512i, b: __m512i, p: __m512i, pinv: __m512i) -> __m512i {
    let low = _mm512_mullo_epi16(a, b);
    let m = _mm512_mullo_epi16(low, pinv);
    _mm512_sub_epi16(_mm512_mulhi_epi16(a, b), _mm512_mulhi_epi16(m, p))
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn shoup(x: __m512i, zeta: __m512i, quotient: __m512i, p: __m512i) -> __m512i {
    let q = _mm512_mulhrs_epi16(x, quotient);
    _mm512_sub_epi16(_mm512_mullo_epi16(x, zeta), _mm512_mullo_epi16(q, p))
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn centre(a: __m512i, p: __m512i, barrett: __m512i) -> __m512i {
    let t = _mm512_mulhi_epi16(a, barrett);
    let t = _mm512_srai_epi16(_mm512_add_epi16(t, _mm512_set1_epi16(512)), 10);
    _mm512_sub_epi16(a, _mm512_mullo_epi16(t, p))
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const LOW_LANES: [u32; 6] = [0, 0, 0x0000_ffff, 0x00ff_00ff, 0x0f0f_0f0f, 0x3333_3333];

/// Even-odd storage into natural coefficient order: output lane `2i` takes the even half,
/// lane `2i + 1` the odd one.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const INTERLEAVE: [[i16; 32]; 2] = {
    let mut index = [[0i16; 32]; 2];
    let mut parity = 0;
    while parity < 2 {
        let mut i = 0;
        while i < 16 {
            index[parity][2 * i] = (16 * parity + i) as i16;
            index[parity][2 * i + 1] = (32 + 16 * parity + i) as i16;
            i += 1;
        }
        parity += 1;
    }
    index
};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn interleave(source: *const i16, register: usize) -> __m512i {
    let half = 32 * (register / 2);
    _mm512_permutex2var_epi16(
        _mm512_loadu_si512(source.add(half) as *const _),
        _mm512_loadu_si512(INTERLEAVE[register % 2].as_ptr() as *const _),
        _mm512_loadu_si512(source.add(64 + half) as *const _),
    )
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn partner_of<const STAGE: usize>(x: __m512i) -> __m512i {
    match STAGE {
        2 => _mm512_shuffle_i64x2(x, x, 0x4e),
        3 => _mm512_shuffle_i64x2(x, x, 0xb1),
        4 => _mm512_shuffle_epi32(x, _MM_PERM_BADC),
        _ => _mm512_shuffle_epi32(x, _MM_PERM_CDAB),
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn in_register<const STAGE: usize, const BATCH: usize>(
    limb: &Limb,
    z: &mut [__m512i],
    p: __m512i,
) {
    for r in 0..4 {
        let zeta = _mm512_loadu_si512(limb.stage_zetas[STAGE].as_ptr().add(32 * r) as *const _);
        let quotient = _mm512_loadu_si512(limb.stage_shoup[STAGE].as_ptr().add(32 * r) as *const _);
        for e in 0..BATCH {
            let x = z[4 * e + r];
            let other = partner_of::<STAGE>(x);
            let upper = _mm512_mask_blend_epi16(LOW_LANES[STAGE], x, other);
            let lower = _mm512_mask_blend_epi16(LOW_LANES[STAGE], other, x);
            z[4 * e + r] = _mm512_add_epi16(lower, shoup(upper, zeta, quotient, p));
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn crossing<const STAGE: usize, const BATCH: usize>(
    limb: &Limb,
    z: &mut [__m512i],
    p: __m512i,
) {
    let step = 2 >> STAGE;
    for low in 0..4 {
        if low & step != 0 {
            continue;
        }
        let zeta = _mm512_set1_epi16(limb.stage_zetas[STAGE][32 * low]);
        let quotient = _mm512_set1_epi16(limb.stage_shoup[STAGE][32 * low]);
        for e in 0..BATCH {
            let t = shoup(z[4 * e + low + step], zeta, quotient, p);
            z[4 * e + low + step] = _mm512_sub_epi16(z[4 * e + low], t);
            z[4 * e + low] = _mm512_add_epi16(z[4 * e + low], t);
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn transform<const BATCH: usize, const MODE: usize>(
    limb: &Limb,
    source: *const i16,
    out: *mut i16,
) {
    let p = _mm512_set1_epi16(limb.p as i16);
    let pinv = _mm512_set1_epi16(limb.pinv);
    let barrett = _mm512_set1_epi16(limb.barrett as i16);

    let mut z = [_mm512_setzero_si512(); 16];
    for e in 0..BATCH {
        for r in 0..4 {
            z[4 * e + r] = centre(interleave(source.add(DEGREE * e), r), p, barrett);
        }
    }

    let reduce = |z: &mut [__m512i], stage: u32| {
        if (stage + 1) % limb.stride == 0 {
            for slot in z.iter_mut().take(4 * BATCH) {
                *slot = centre(*slot, p, barrett);
            }
        }
    };

    crossing::<0, BATCH>(limb, &mut z, p);
    reduce(&mut z, 0);
    crossing::<1, BATCH>(limb, &mut z, p);
    reduce(&mut z, 1);
    in_register::<2, BATCH>(limb, &mut z, p);
    reduce(&mut z, 2);
    in_register::<3, BATCH>(limb, &mut z, p);
    reduce(&mut z, 3);
    in_register::<4, BATCH>(limb, &mut z, p);
    reduce(&mut z, 4);
    in_register::<5, BATCH>(limb, &mut z, p);
    for slot in z.iter_mut().take(4 * BATCH) {
        *slot = centre(*slot, p, barrett);
    }

    for r in 0..4 {
        let zeta = _mm512_loadu_si512(limb.slot_pairs.as_ptr().add(32 * r) as *const _);
        for e in 0..BATCH {
            let x = z[4 * e + r];
            if MODE == 0 {
                _mm512_storeu_si512(out.add(DEGREE * e + 32 * r) as *mut _, x);
                continue;
            }
            let scaled = centre(montgomery(x, zeta, p, pinv), p, barrett);
            _mm512_storeu_si512(out.add(2 * DEGREE * e + 32 * r) as *mut _, scaled);
            _mm512_storeu_si512(out.add(2 * DEGREE * e + DEGREE + 32 * r) as *mut _, x);
        }
    }
}

/// One transformed witness element against `ROWS` key rows, so that its eight loads are
/// shared instead of repeated per row.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn accumulate<const ROWS: usize>(
    acc: &mut [__m512i],
    operand: *const i16,
    key: [*const i16; ROWS],
) {
    for r in 0..4 {
        let plain = _mm512_loadu_si512(operand.add(32 * r) as *const _);
        let swapped = _mm512_rol_epi32(plain, 16);
        for row in 0..ROWS {
            let scaled = _mm512_loadu_si512(key[row].add(32 * r) as *const _);
            let straight = _mm512_loadu_si512(key[row].add(DEGREE + 32 * r) as *const _);
            acc[8 * row + r] = _mm512_dpwssd_epi32(acc[8 * row + r], plain, scaled);
            acc[8 * row + 4 + r] = _mm512_dpwssd_epi32(acc[8 * row + 4 + r], swapped, straight);
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn reduce(limb: &Limb, accumulators: &mut [i32]) {
    let p = _mm512_set1_epi32(limb.p);
    let m = _mm512_set1_epi32(limb.reciprocal_i32);
    for lane in accumulators.chunks_exact_mut(16) {
        let a = _mm512_loadu_si512(lane.as_ptr() as *const _);
        let even = _mm512_and_si512(
            _mm512_srai_epi64(_mm512_mul_epi32(a, m), 42),
            _mm512_set1_epi64(0xffff_ffff),
        );
        let odd = _mm512_slli_epi64(
            _mm512_srai_epi64(_mm512_mul_epi32(_mm512_srli_epi64(a, 32), m), 42),
            32,
        );
        let quotient = _mm512_or_si512(even, odd);
        _mm512_storeu_si512(
            lane.as_mut_ptr() as *mut _,
            _mm512_sub_epi32(a, _mm512_mullo_epi32(quotient, p)),
        );
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn load(at: [*const i32; 2]) -> [__m512i; 16] {
    std::array::from_fn(|slot| _mm512_loadu_si512(at[slot / 8].add(16 * (slot % 8)) as *const _))
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn store<const ROWS: usize>(acc: &[__m512i], at: [*mut i32; ROWS]) {
    for row in 0..ROWS {
        for r in 0..8 {
            _mm512_storeu_si512(at[row].add(16 * r) as *mut _, acc[8 * row + r]);
        }
    }
}

/// The accumulated slots back into coefficient order: constant terms at even indices.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn gather(limb: &Limb, long: &[i32], out: &mut [i16; DEGREE]) {
    for slot in 0..SLOTS {
        out[2 * slot] = limb.centre(limb.centre32(long[slot]));
        out[2 * slot + 1] = limb.centre(limb.centre32(long[SLOTS + slot]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::init_common;
    use crate::protocol::commitment::commit_basic;
    use crate::protocol::crs::CRS;
    use crate::protocol::project_coarse::prepare_i16_witness;

    fn schoolbook(a: &[i16; DEGREE], b: &[i16; DEGREE], p: i32) -> [i16; DEGREE] {
        let mut out = [0i64; DEGREE];
        for i in 0..DEGREE {
            for j in 0..DEGREE {
                let term = a[i] as i64 * b[j] as i64;
                if i + j < DEGREE {
                    out[i + j] += term;
                } else {
                    out[i + j - DEGREE] -= term;
                }
            }
        }
        let mut reduced = [0i16; DEGREE];
        for i in 0..DEGREE {
            let value = out[i].rem_euclid(p as i64);
            reduced[i] = if value > p as i64 / 2 {
                (value - p as i64) as i16
            } else {
                value as i16
            };
        }
        reduced
    }

    fn sample(limb: &Limb, seed: u64) -> [i16; DEGREE] {
        let mut state = seed | 1;
        let mut out = [0i16; DEGREE];
        for slot in out.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *slot = limb.centre(((state >> 33) as i32 % limb.p) as i16);
        }
        out
    }

    #[test]
    fn crt_plan_is_reported() {
        for (n, rank) in [(8192usize, 10usize), (16384, 10), (64, 2)] {
            let l2 = ((n * DEGREE) as f64 * (1u64 << 30) as f64 / 3.0).sqrt();
            let plan = Plan::new(l2, rank);
            let bits: f64 = plan.primes.iter().map(|p| (*p as f64).log2()).sum();
            println!(
                "n = {n:5} rank = {rank}: primes {:?} ({} limbs, {bits:.1} bits, bound 2^{:.1})",
                plan.primes,
                plan.primes.len(),
                plan.bound.log2()
            );
        }
    }

    #[test]
    fn crt_simd_transform_matches_the_reference() {
        for p in PRIMES {
            let limb = Limb::new(p);
            let stored = Signed16RingElement(sample(&limb, p as u64 + 3));

            let mut reference = natural(&stored);
            for slot in reference.iter_mut() {
                *slot = limb.centre(*slot);
            }
            limb.ntt(&mut reference);
            let scaled = limb.arrange(&reference);

            let mut plain = [0i16; DEGREE];
            unsafe { super::transform::<1, 0>(&limb, stored.0.as_ptr(), plain.as_mut_ptr()) };
            let mut paired = [0i16; 2 * DEGREE];
            unsafe { super::transform::<1, 2>(&limb, stored.0.as_ptr(), paired.as_mut_ptr()) };
            for i in 0..DEGREE {
                assert_eq!(plain[i], reference[i], "p = {p}, plain {i}");
                assert_eq!(paired[i], scaled[i], "p = {p}, scaled {i}");
                assert_eq!(paired[DEGREE + i], reference[i], "p = {p}, straight {i}");
            }
        }
    }

    #[test]
    fn crt_simd_residues_match_the_reference() {
        init_common();
        let height = 64;
        let width = 5;
        let rank = 3;
        let crs = CRS::gen_crs(height, rank + 1);
        let witness = VerticallyAlignedMatrix {
            data: (0..height * width)
                .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 1 << 15))
                .collect(),
            width,
            height,
            used_cols: width,
        };
        let digits = prepare_i16_witness(&witness);
        let plan = Plan::new(digits_l2(&digits), rank);
        let key = CrtKey::preprocess(crs.ck_for_wit_dim(height), rank, &plan);
        assert_eq!(
            residues(&key, &digits, rank, 1),
            reference_residues(&key, &digits, rank, 1)
        );
    }

    #[test]
    fn crt_transform_inverts() {
        for p in PRIMES {
            let limb = Limb::new(p);
            let original = sample(&limb, p as u64);
            let mut transformed = original;
            limb.ntt(&mut transformed);
            limb.inverse_ntt(&mut transformed);
            for slot in 0..DEGREE {
                assert_eq!(
                    limb.centre(transformed[slot]),
                    limb.centre(original[slot]),
                    "p = {p}, slot = {slot}"
                );
            }
        }
    }

    #[test]
    fn crt_slot_products_are_the_negacyclic_ones() {
        for p in PRIMES {
            let limb = Limb::new(p);
            let a = sample(&limb, p as u64);
            let b = sample(&limb, p as u64 + 7);

            let mut a_ntt = a;
            let mut b_ntt = b;
            limb.ntt(&mut a_ntt);
            limb.ntt(&mut b_ntt);
            let scaled = limb.arrange(&b_ntt);

            let mut product = [0i16; DEGREE];
            for slot in 0..SLOTS {
                product[2 * slot] = limb.centre32(
                    a_ntt[2 * slot] as i32 * scaled[2 * slot] as i32
                        + a_ntt[2 * slot + 1] as i32 * scaled[2 * slot + 1] as i32,
                );
                product[2 * slot + 1] = limb.centre32(
                    a_ntt[2 * slot + 1] as i32 * b_ntt[2 * slot] as i32
                        + a_ntt[2 * slot] as i32 * b_ntt[2 * slot + 1] as i32,
                );
            }
            limb.inverse_ntt(&mut product);

            let expected = schoolbook(&a, &b, p);
            for slot in 0..DEGREE {
                assert_eq!(
                    limb.centre(product[slot]),
                    expected[slot],
                    "p = {p}, slot = {slot}"
                );
            }
        }
    }

    #[test]
    fn crt_bisect() {
        for (height, width, rank) in [(8, 1, 1), (16, 1, 1), (4, 3, 1), (4, 1, 2), (8, 3, 2)] {
            crt_bisect_at(height, width, rank);
        }
    }

    fn crt_bisect_at(height: usize, width: usize, rank: usize) {
        init_common();
        let crs = CRS::gen_crs(height, rank + 1);
        let witness = VerticallyAlignedMatrix {
            data: (0..height * width)
                .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 1 << 12))
                .collect(),
            width,
            height,
            used_cols: width,
        };
        let expected = commit_basic(&crs, &witness, rank, 1);

        let ck = crs.ck_for_wit_dim(height);
        let centred = |e: &RingElement| -> Vec<i128> {
            let mut c = e.clone();
            c.to_representation(Representation::Coefficients);
            c.v.iter()
                .map(|v| {
                    if *v > MOD_Q / 2 {
                        *v as i128 - MOD_Q as i128
                    } else {
                        *v as i128
                    }
                })
                .collect()
        };
        let mut reference = vec![0i128; DEGREE];
        for k in 0..height.min(if width == 1 && rank == 1 { height } else { 0 }) {
            let a = centred(&ck[0].preprocessed_row[k]);
            let w = centred(&witness.data[k]);
            for i in 0..DEGREE {
                for j in 0..DEGREE {
                    let term = a[i] * w[j];
                    if i + j < DEGREE {
                        reference[i + j] += term;
                    } else {
                        reference[i + j - DEGREE] -= term;
                    }
                }
            }
        }
        let mut lifted = RingElement::new(Representation::Coefficients);
        for i in 0..DEGREE {
            lifted.v[i] = reference[i].rem_euclid(MOD_Q as i128) as u64;
        }
        lifted.to_representation(Representation::IncompleteNTT);
        if width == 1 && rank == 1 {
            assert_eq!(
                expected.data[0].v, lifted.v,
                "integer model of the commitment"
            );
        }

        let digits = prepare_i16_witness(&witness);
        for k in 0..height {
            let w = centred(&witness.data[k]);
            let got = natural(&digits.data[k]);
            for i in 0..DEGREE {
                assert_eq!(w[i], got[i] as i128, "digit {i} of element {k}");
            }
        }

        let plan = Plan {
            primes: PRIMES.to_vec(),
            bound: f64::INFINITY,
        };
        let key = CrtKey::preprocess(ck, rank, &plan);
        let got = commit_basic_crt(&key, &digits, &plan, rank, 1);
        for row in 0..rank {
            for col in 0..width {
                assert_eq!(
                    expected.data[row * width + col].v,
                    got.data[row * width + col].v,
                    "crt commitment at {height}x{width} rank {rank}, row {row} col {col}"
                );
            }
        }
    }

    #[test]
    fn crt_streaming_matches_the_ring_one() {
        init_common();
        let height = 64;
        let width = 5;
        let rank = 3;
        let crs = CRS::gen_crs(height, rank + 1);
        let witness = VerticallyAlignedMatrix {
            data: (0..height * width)
                .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 1 << 15))
                .collect(),
            width,
            height,
            used_cols: width,
        };
        let expected = crate::protocol::commitment::commit_basic(&crs, &witness, rank, 1);

        let digits = prepare_i16_witness(&witness);
        let plan = Plan::new(digits_l2(&digits), rank);
        let key = CrtKey::preprocess(crs.ck_for_wit_dim(height), rank, &plan);
        let got = commit_basic_crt_streaming(&key, &witness, &plan, rank, 1);
        for element in 0..rank * width {
            assert_eq!(
                expected.data[element].v, got.data[element].v,
                "element {element}"
            );
        }
    }

    #[cfg(all(
        feature = "parallel",
        target_arch = "x86_64",
        target_feature = "avx512f"
    ))]
    #[test]
    fn crt_parallel_matches_the_serial_path() {
        init_common();
        let height = 64;
        let width = 70;
        let rank = 3;
        let crs = CRS::gen_crs(height, rank + 1);
        let witness = VerticallyAlignedMatrix {
            data: (0..height * width)
                .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 1 << 15))
                .collect(),
            width,
            height,
            used_cols: width,
        };
        let digits = prepare_i16_witness(&witness);
        let plan = Plan::new(digits_l2(&digits), rank);
        let ck = crs.ck_for_wit_dim(height);

        let key = CrtKey::preprocess_with(ck, rank, &plan, true);
        assert_eq!(
            key.data,
            CrtKey::preprocess_with(ck, rank, &plan, false).data,
            "preprocessed key"
        );

        let from_digits = |parallel| {
            residues_by_tile(
                &key,
                rank,
                width,
                height,
                width,
                Source::Narrowed(&digits),
                parallel,
            )
        };
        let streamed = |parallel| {
            residues_by_tile(
                &key,
                rank,
                width,
                height,
                width,
                Source::Ring(&witness),
                parallel,
            )
        };
        assert_eq!(from_digits(true), from_digits(false), "residues");
        assert_eq!(streamed(true), streamed(false), "streamed residues");
        assert_eq!(from_digits(true), streamed(true), "the two sources");

        let expected = crate::protocol::commitment::commit_basic(&crs, &witness, rank, 1);
        let got = commit_basic_crt(&key, &digits, &plan, rank, 1);
        for element in 0..rank * width {
            assert_eq!(
                expected.data[element].v, got.data[element].v,
                "element {element}"
            );
        }
    }

    #[test]
    fn crt_commitment_matches_the_ring_one() {
        init_common();
        let height = 64;
        let width = 3;
        let rank = 2;
        let crs = CRS::gen_crs(height, rank + 1);

        let witness = VerticallyAlignedMatrix {
            data: (0..height * width)
                .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 1 << 15))
                .collect(),
            width,
            height,
            used_cols: width,
        };

        let expected = crate::protocol::commitment::commit_basic(&crs, &witness, rank, 1);

        let digits = prepare_i16_witness(&witness);
        let plan = Plan::new(digits_l2(&digits), rank);
        let key = CrtKey::preprocess(crs.ck_for_wit_dim(height), rank, &plan);
        let got = commit_basic_crt(&key, &digits, &plan, rank, 1);

        for row in 0..rank {
            for col in 0..width {
                assert_eq!(
                    expected.data[row * width + col].v,
                    got.data[row * width + col].v,
                    "row {row}, col {col}"
                );
            }
        }
    }
}
