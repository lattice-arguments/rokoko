use std::io;
use std::process::Command;

use crate::common::config::{DEGREE, MOD_Q};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Norm {
    L2,
    Infinity,
}

impl Norm {
    pub fn as_str(&self) -> &str {
        match self {
            Norm::L2 => "2",
            Norm::Infinity => "oo",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SISParameters {
    pub n: u64,
    pub m: u64,
    pub q: u64,
    pub length_bound: u64,
    pub norm: Norm,
}

pub struct RSISParameters {
    pub n: u64,
    pub m: u64,
    pub length_bound: u64,
}

#[derive(Debug)]
pub struct EstimatorResult {
    pub secpar: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedCostModel {
    Matzov,
    Adps16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedShapeModel {
    Gsa,
    Lgsa,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Zeta {
    Fixed(u64),
    Optimised,
}

#[derive(Debug, Clone, Copy)]
pub struct SISLatticeOptions {
    pub red_cost_model: RedCostModel,
    pub red_shape_model: RedShapeModel,
    pub zeta: Zeta,
}

impl Default for SISLatticeOptions {
    /// `SIS.lattice(params, zeta=0, red_cost_model=RC.ADPS16, red_shape_model="lgsa")`
    fn default() -> Self {
        SISLatticeOptions {
            red_cost_model: RedCostModel::Adps16,
            red_shape_model: RedShapeModel::Lgsa,
            zeta: Zeta::Fixed(0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SISLatticeEstimate {
    pub log2_rop: f64,
    pub beta: u64,
    pub d: u64,
    pub zeta: u64,
    pub log2_repetitions: f64,
}

pub fn estimate_rsis_security(params: &RSISParameters) -> Result<EstimatorResult, io::Error> {
    let m_sis = params.m * DEGREE as u64;
    let n_sis = params.n * DEGREE as u64;
    let sis_params = SISParameters {
        n: n_sis,
        m: m_sis,
        q: MOD_Q,
        length_bound: params.length_bound,
        norm: Norm::L2,
    };
    estimate_sis_security(&sis_params)
}

/// Bit security of SIS against lattice reduction, rounded up, following `SIS.lattice` of the
/// lattice estimator. The Euclidean norm is costed with MATZOV. The infinity norm uses the
/// conservative convention of [`SISLatticeOptions::default`]: ADPS16, the LGSA shape and no
/// ignored coordinates; [`estimate_sis_lattice_infinity`] takes other options.
pub fn estimate_sis_security(params: &SISParameters) -> Result<EstimatorResult, io::Error> {
    let log2_rop = match params.norm {
        Norm::L2 => {
            if params.length_bound >= (params.q - 1) / 2 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "SIS trivially easy. Please set norm bound < (q-1)/2.",
                ));
            }
            sis_lattice::cost_euclidean_log2(params)
        }
        Norm::Infinity => {
            estimate_sis_lattice_infinity(params, &SISLatticeOptions::default())?.log2_rop
        }
    };

    let rop = log2_rop.ceil();

    if rop < 128.0 {
        println!(
            "Warning: Estimated security parameter is low: {} bits. Consider increasing parameters.",
            rop
        );
    }
    Ok(EstimatorResult { secpar: rop })
}

/// `SIS.lattice` for the infinity norm under `options`, unrounded. `Err` where the lattice
/// estimator raises: a trivial norm bound, or a Euclidean baseline within BKZ-40.
pub fn estimate_sis_lattice_infinity(
    params: &SISParameters,
    options: &SISLatticeOptions,
) -> Result<SISLatticeEstimate, io::Error> {
    if params.norm != Norm::Infinity {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Only the infinity norm is supported.",
        ));
    }
    if params.n == 0 || params.length_bound == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "The rank and the norm bound must be positive.",
        ));
    }
    if params.length_bound >= params.q / 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "SIS trivially easy. Please set norm bound < (q-1)/2.",
        ));
    }
    sis_lattice::cost_infinity_lattice(params, options).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Block size search is empty: the Euclidean baseline is within BKZ-40.",
        )
    })
}

/// Legacy implementation that shells out to Sage/Python lattice-estimator.
pub fn estimate_sis_security_lattice_estimator(
    params: &SISParameters,
) -> Result<EstimatorResult, io::Error> {
    let script_path = std::env::current_dir()?
        .join("scripts")
        .join("run_sage_estimator.sh");

    let output = Command::new("bash")
        .arg(script_path)
        .arg(params.n.to_string())
        .arg(params.m.to_string())
        .arg(params.q.to_string())
        .arg(params.length_bound.to_string())
        .arg(params.norm.as_str())
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("Estimator script failed: {}", stderr),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let secpar: f64 = stdout.trim().parse().map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Failed to parse output: {}", e),
        )
    })?;

    Ok(EstimatorResult {
        secpar: secpar.ceil(),
    })
}

mod sis_lattice {
    //! Pure-Rust port of `estimator.sis_lattice.SISLattice` from the
    //! `lattice-estimator` project:
    //! - the Euclidean cost with `red_cost_model = RC.MATZOV` (list_decoding-classical)
    //! - the infinity-norm cost with `RC.MATZOV` or `RC.ADPS16` and the `"gsa"` or `"lgsa"` shape

    use super::{
        Norm, RedCostModel, RedShapeModel, SISLatticeEstimate, SISLatticeOptions, SISParameters,
        Zeta,
    };
    use std::collections::HashSet;

    // ---- MATZOV nearest-neighbor constants (list_decoding-classical) ----
    const MATZOV_A: f64 = 0.29613500308205365;
    const MATZOV_B: f64 = 20.387885985467914;

    // ---------- Reduction cost: δ(β) and its inverse ----------

    /// `_delta(β)` — Python version takes integer β in production but accepts
    /// float β through `_beta_find_root`. We mirror that here.
    fn delta_f(beta: f64) -> f64 {
        if beta <= 2.0 {
            return 1.0219;
        }
        if beta < 5.0 {
            return 1.02190;
        }
        if beta < 10.0 {
            return 1.01862;
        }
        if beta < 15.0 {
            return 1.01616;
        }
        if beta < 20.0 {
            return 1.01485;
        }
        if beta < 25.0 {
            return 1.01420;
        }
        if beta < 28.0 {
            return 1.01342;
        }
        if beta < 40.0 {
            return 1.01331;
        }
        if beta == 40.0 {
            return 1.01295;
        }
        let pi = std::f64::consts::PI;
        let e = std::f64::consts::E;
        let base = beta / (2.0 * pi * e) * (pi * beta).powf(1.0 / beta);
        base.powf(1.0 / (2.0 * (beta - 1.0)))
    }

    /// Public delta with integer β (Python: `delta(β) = _delta(ZZ(round(β)))`).
    pub(super) fn delta(beta: u64) -> f64 {
        delta_f(beta as f64)
    }

    /// Inverse of δ: given target δ return β ∈ ℤ with `_delta(β) ≤ target`.
    /// Mirrors `_beta_find_root`: bisection in `[40, 2^16]` on
    /// `_delta(β) - target`, then `ceil(β - 1e-8)`.
    fn beta_from_delta(target: f64) -> u64 {
        // Corresponds to the explicit early-exit in `_beta_find_root`.
        if delta_f(40.0) < target {
            return 40;
        }
        let lo = 40.0_f64;
        let hi = 65536.0_f64;
        let f_lo = delta_f(lo) - target;
        let f_hi = delta_f(hi) - target;
        if f_lo * f_hi > 0.0 {
            // No sign change — fall back to the "simple" search as the Python
            // code does on failure.
            return beta_simple(target);
        }
        let mut a = lo;
        let mut b = hi;
        // Plain bisection to high precision — matches the behavior of
        // `sage.find_root` (brentq) on a monotone function for our purposes.
        for _ in 0..200 {
            let m = 0.5 * (a + b);
            if (b - a).abs() < 1e-12 {
                break;
            }
            let fm = delta_f(m) - target;
            let fa = delta_f(a) - target;
            if fa * fm <= 0.0 {
                b = m;
            } else {
                a = m;
            }
        }
        let root = 0.5 * (a + b);
        (root - 1e-8).ceil() as u64
    }

    /// Fallback: `_beta_simple`, doubling/stepping search from β=40.
    fn beta_simple(target: f64) -> u64 {
        let mut beta = 40u64;
        while delta_f((2 * beta) as f64) > target {
            beta *= 2;
        }
        while delta_f((beta + 10) as f64) > target {
            beta += 10;
        }
        while delta_f(beta as f64) >= target {
            beta += 1;
        }
        beta
    }

    // ---------- MATZOV/GJ21 cost and short_vectors ----------

    /// `Kyber.d4f(β)` — dimensions "for free".
    fn d4f(beta: f64) -> f64 {
        let pi = std::f64::consts::PI;
        let e = std::f64::consts::E;
        let num = beta * (4.0f64 / 3.0).ln();
        let den = (beta / (2.0 * pi * e)).ln();
        let v = num / den;
        if v.is_nan() || v < 0.0 {
            0.0
        } else {
            v
        }
    }

    fn log2_add(lhs: f64, rhs: f64) -> f64 {
        if lhs.is_infinite() && lhs.is_sign_negative() {
            return rhs;
        }
        if rhs.is_infinite() && rhs.is_sign_negative() {
            return lhs;
        }
        let max = lhs.max(rhs);
        let min = lhs.min(rhs);
        max + (1.0 + 2f64.powf(min - max)).log2()
    }

    /// `CheNgu12.__call__(β, d)` — fallback for β < 20 inside `Kyber.__call__`.
    fn chengu12_call_log2(beta: f64, d: f64) -> f64 {
        let repeat_log2 = svp_repeat(beta, d).log2();
        let svp_cost_log2 = 0.270188776350190 * beta * beta.ln() - 1.0192050451318417 * beta
            + 16.10253135200765
            + (100.0_f64).log2();
        log2_add(lll(d).log2(), repeat_log2 + svp_cost_log2)
    }

    fn svp_repeat(beta: f64, d: f64) -> f64 {
        if beta < d {
            8.0 * d
        } else {
            1.0
        }
    }

    fn lll(d: f64) -> f64 {
        d.powi(3)
    }

    /// `MATZOV.__call__(β, d)` — Kyber cost with list_decoding-classical.
    fn matzov_call_log2(beta: f64, d: f64) -> f64 {
        if beta < 20.0 {
            return chengu12_call_log2(beta, d);
        }
        let c_prog_log2 = (1.0 / (1.0 - 2f64.powf(-MATZOV_A))).log2();
        let svp_calls_log2 = c_prog_log2 + (d - beta).max(1.0).log2();
        let beta_ = beta - d4f(beta);
        let gate_count_log2 = c_prog_log2 + MATZOV_A * beta_ + MATZOV_B;
        log2_add(lll(d).log2(), svp_calls_log2 + gate_count_log2)
    }

    fn log2_floor_pow2(exponent: f64) -> f64 {
        if exponent < 63.0 {
            2f64.powf(exponent).floor().log2()
        } else {
            exponent
        }
    }

    /// `MATZOV.short_vectors(β, d)` with `N=None`, no sieve_dim hint.
    /// Returns `(ρ, log2(cost_red), log2(N), sieve_dim)`.
    fn matzov_short_vectors(beta: u64, d: u64) -> (f64, f64, f64, u64) {
        let beta_f = beta as f64;
        let d_f = d as f64;
        let c_prog = 1.0 / (1.0 - 2f64.powf(-MATZOV_A));
        let c_prog_log2 = c_prog.log2();
        let beta_minus = beta_f - d4f(beta_f).floor();
        let sieve_dim = if beta < d {
            let cand = beta_minus + (((d_f - beta_f) * c_prog).log2() / MATZOV_A);
            d_f.min(cand.floor()) as u64
        } else {
            beta_minus as u64
        };
        let sd_f = sieve_dim as f64;
        // ρ = sqrt(4/3) * δ(sieve_dim)^(sieve_dim-1) * δ(β)^(1-sieve_dim)
        let rho = (4.0f64 / 3.0).sqrt()
            * delta(sieve_dim).powf(sd_f - 1.0)
            * delta(beta).powf(1.0 - sd_f);
        // N defaulted → N = floor(2^(0.2075 * sieve_dim)), so c = N / floor(c1) = 1.
        let log2_n = log2_floor_pow2(0.2075 * sd_f);
        let sieve_cost_log2 = c_prog_log2 + MATZOV_A * sd_f + MATZOV_B;
        let cost_red_log2 = log2_add(matzov_call_log2(beta_f, d_f), sieve_cost_log2);
        (rho, cost_red_log2, log2_n, sieve_dim)
    }

    /// `costf(MATZOV, β, d)["rop"]` — just `MATZOV(β, d)`.
    fn bkz_rop_log2(beta: u64, d: u64) -> f64 {
        matzov_call_log2(beta as f64, d as f64)
    }

    // ---------- Probability helpers ----------

    fn gaussian_cdf(mu: f64, sigma: f64, t: f64) -> f64 {
        0.5 * (1.0 + libm::erf((t - mu) / (std::f64::consts::SQRT_2 * sigma)))
    }

    fn coordinate_success_log2(length_bound: f64, sigma: f64) -> f64 {
        if sigma <= 0.0 {
            return if length_bound > 0.0 {
                0.0
            } else {
                f64::NEG_INFINITY
            };
        }
        let phi = gaussian_cdf(0.0, sigma, -length_bound);
        let success = 1.0 - 2.0 * phi;
        if success <= 0.0 {
            f64::NEG_INFINITY
        } else {
            success.log2()
        }
    }

    // ---------- Euclidean (L2) cost ----------

    fn opt_sis_d(params: &SISParameters) -> f64 {
        let lb_log2 = (params.length_bound as f64).log2();
        let q_log2 = (params.q as f64).log2();
        let n_f = params.n as f64;
        let log_delta = lb_log2 * lb_log2 / (4.0 * n_f * q_log2);
        (n_f * q_log2 / log_delta).sqrt()
    }

    fn solve_for_delta_euclidean(params: &SISParameters, d: u64) -> f64 {
        let d_f = d as f64;
        let n_f = params.n as f64;
        let q_log2 = (params.q as f64).log2();
        let root_volume = (n_f / d_f) * q_log2;
        let log_delta = ((params.length_bound as f64).log2() - root_volume) / (d_f - 1.0);
        2f64.powf(log_delta)
    }

    pub(super) fn cost_euclidean_log2(params: &SISParameters) -> f64 {
        // d = min(floor(opt), m)
        let d_opt = opt_sis_d(params).floor() as u64;
        let d = d_opt.min(params.m);

        let delta_val = solve_for_delta_euclidean(params, d);
        let (beta, reduction_possible) = if delta_val >= 1.0 {
            let b = beta_from_delta(delta_val);
            if b <= d {
                (b, true)
            } else {
                (d, false)
            }
        } else {
            (d, false)
        };

        // lb = min(sqrt(n * ln(q)), sqrt(d) * q^(n/d))
        let n_f = params.n as f64;
        let d_f = d as f64;
        let q_f = params.q as f64;
        let lb1 = (n_f * q_f.ln()).sqrt();
        let lb2 = d_f.sqrt() * q_f.powf(n_f / d_f);
        let lb = lb1.min(lb2);

        let predicate = (params.length_bound as f64) > lb && reduction_possible;
        if !predicate {
            return f64::INFINITY;
        }
        bkz_rop_log2(beta, d)
    }

    // ---------- Infinity norm cost ----------

    fn adps16_short_vectors(beta: u64) -> (f64, f64, f64, u64) {
        let beta_f = beta as f64;
        let delta_val = delta(beta);
        let rho =
            (4.0f64 / 3.0).sqrt() * (delta_val.powf(beta_f - 1.0) * delta_val.powf(1.0 - beta_f));
        (rho, 0.292 * beta_f, log2_floor_pow2(0.2075 * beta_f), beta)
    }

    // prob.amplify rounds 1 - p to max(53, 2 * ceil(-log2 p)) bits, capped at 2048
    pub(super) fn amplify_trials_log2(target: f64, log2_success: f64) -> f64 {
        let success = log2_success.exp2();
        if target < success {
            return 0.0;
        }
        let numerator = (1.0 - target).ln();
        let precision = (2.0 * (-log2_success).ceil()).clamp(53.0, 2048.0);
        if precision == 53.0 {
            return (numerator / (1.0 - success).ln()).ceil().log2();
        }
        let scaled = log2_success + precision;
        if scaled >= 53.0 {
            if log2_success < -1000.0 {
                return (-numerator).log2() - log2_success;
            }
            return (numerator / (-success).ln_1p()).ceil().log2();
        }
        let units = scaled.exp2().round_ties_even();
        if units == 0.0 {
            return f64::INFINITY;
        }
        let log2_rounded = units.log2() - precision;
        if log2_rounded < -1000.0 {
            return (-numerator).log2() - log2_rounded;
        }
        let rounded = libm::scalbn(units, -(precision as i32));
        (numerator / (-rounded).ln_1p()).ceil().log2()
    }

    enum Shape<'a> {
        Gsa { d: u64, log_delta: f64, level: f64 },
        Lgsa { d: u64, ramp: &'a [f64], shift: f64 },
    }

    impl Shape<'_> {
        fn lgsa(d: u64, log_vol: f64, log_delta: f64, ramp: &mut Vec<f64>) -> Shape<'_> {
            let slope = -2.0 * log_delta;
            let mut log_vec_len = 0.0;
            let mut profile_log_vol = 0.0;
            ramp.clear();
            for _ in 0..d {
                log_vec_len -= slope;
                profile_log_vol += log_vec_len;
                ramp.push(log_vec_len);
                if profile_log_vol > log_vol {
                    break;
                }
            }
            let profile_log_vol = ramp.iter().rev().fold(0.0, |sum, &x| sum + x);
            let shift = (profile_log_vol - log_vol) / ramp.len() as f64;
            Shape::Lgsa { d, ramp, shift }
        }

        fn d(&self) -> u64 {
            match *self {
                Shape::Gsa { d, .. } | Shape::Lgsa { d, .. } => d,
            }
        }

        fn sloped(&self) -> u64 {
            match *self {
                Shape::Gsa { d, .. } => d,
                Shape::Lgsa { ramp, .. } => ramp.len() as u64,
            }
        }

        fn r(&self, i: u64) -> f64 {
            let r_log = match *self {
                Shape::Gsa {
                    d,
                    log_delta,
                    level,
                } => (d as f64 - 1.0 - 2.0 * i as f64) * log_delta + level,
                Shape::Lgsa { ramp, shift, .. } => match ramp.len().checked_sub(i as usize + 1) {
                    Some(j) => ramp[j] - shift,
                    None => 0.0,
                },
            };
            (2.0 * r_log).exp2()
        }

        fn first_below(&self, bound: f64) -> u64 {
            (1..=self.sloped().min(self.d() - 1))
                .find(|&i| self.r(i) < bound)
                .unwrap_or(0)
        }

        fn first_unit(&self) -> u64 {
            let mut i = self.sloped();
            while i > 0 && self.r(i - 1).sqrt() <= 1.0 + 1e-8 {
                i -= 1;
            }
            i
        }
    }

    struct InfinitySearch<'a> {
        params: &'a SISParameters,
        options: &'a SISLatticeOptions,
        log_vol: f64,
        dilithium: bool,
        baseline_beta: u64,
        ramp: Vec<f64>,
    }

    fn keep_smaller(best: &mut Option<SISLatticeEstimate>, cost: SISLatticeEstimate) {
        if best.is_none_or(|b| cost.log2_rop <= b.log2_rop) {
            *best = Some(cost);
        }
    }

    impl InfinitySearch<'_> {
        fn cost(&mut self, beta: u64, zeta: u64) -> SISLatticeEstimate {
            let d = self.params.m.saturating_sub(zeta);
            let mut cost = SISLatticeEstimate {
                log2_rop: f64::INFINITY,
                beta,
                d,
                zeta,
                log2_repetitions: f64::INFINITY,
            };
            if d < beta {
                return cost;
            }

            let (rho, cost_red_log2, log2_big_n, sieve_dim) = match self.options.red_cost_model {
                RedCostModel::Matzov => matzov_short_vectors(beta, d),
                RedCostModel::Adps16 => adps16_short_vectors(beta),
            };
            let log_delta = delta(beta).log2();
            let shape = match self.options.red_shape_model {
                RedShapeModel::Gsa => Shape::Gsa {
                    d,
                    log_delta,
                    level: self.log_vol / d as f64,
                },
                RedShapeModel::Lgsa => Shape::lgsa(d, self.log_vol, log_delta, &mut self.ramp),
            };

            let lb = self.params.length_bound as f64;
            let log_trial_prob = if self.dilithium {
                let r0 = shape.r(0);
                let idx_start = if (r0.sqrt() - self.params.q as f64).abs() < 1e-8 {
                    shape.first_below(r0)
                } else {
                    0
                };
                let idx_end = if (shape.r(d - 1) - 1.0).abs() < 1e-8 {
                    shape.first_unit() as i64 - 1
                } else {
                    d as i64 - 1
                };
                let gaussian_coords = (idx_end - idx_start as i64 + 1).max(sieve_dim as i64) as f64;
                let sigma = shape.r(idx_start).sqrt() / gaussian_coords.sqrt();
                coordinate_success_log2(lb, sigma) * gaussian_coords
                    + ((2.0 * lb + 1.0) / self.params.q as f64).log2() * idx_start as f64
            } else {
                let sigma = rho * shape.r(0).sqrt() / (d as f64).sqrt();
                d as f64 * coordinate_success_log2(lb, sigma)
            };

            let log2_solutions = log_trial_prob + log2_big_n;
            let log2_probability = if log2_solutions < 0.0 {
                log2_solutions
            } else {
                0.0
            };
            if log2_probability == f64::NEG_INFINITY {
                return cost;
            }
            cost.log2_repetitions = amplify_trials_log2(0.99, log2_probability);
            cost.log2_rop = cost_red_log2 + cost.log2_repetitions;
            cost
        }

        fn cost_zeta(&mut self, zeta: u64) -> Option<SISLatticeEstimate> {
            let mut best = None;
            let mut lm = LocalMinimum::new(40, self.baseline_beta + 1, 2);
            while let Some(beta) = lm.next_x() {
                let cost = self.cost(beta, zeta);
                lm.update(cost.log2_rop);
                keep_smaller(&mut best, cost);
            }
            for beta in lm.neighborhood() {
                keep_smaller(&mut best, self.cost(beta, zeta));
            }
            best
        }
    }

    fn euclidean_beta(params: &SISParameters) -> u64 {
        let d = (opt_sis_d(params).floor() as u64).min(params.m);
        let delta_val = solve_for_delta_euclidean(params, d);
        let reachable = delta_f(65536.0).max(delta_f(d as f64 + 1.0)).max(1.0);
        if delta_val >= reachable {
            let beta = beta_from_delta(delta_val);
            if beta <= d {
                return beta;
            }
        }
        d
    }

    pub(super) fn cost_infinity_lattice(
        params: &SISParameters,
        options: &SISLatticeOptions,
    ) -> Option<SISLatticeEstimate> {
        let baseline = SISParameters {
            norm: Norm::L2,
            length_bound: if params.length_bound == 1 {
                2
            } else {
                params.length_bound
            },
            ..params.clone()
        };
        let mut search = InfinitySearch {
            params,
            options,
            log_vol: params.n as f64 * (params.q as f64).log2(),
            dilithium: (params.m as f64).sqrt() * params.length_bound as f64 > params.q as f64,
            baseline_beta: euclidean_beta(&baseline),
            ramp: Vec::new(),
        };

        let mut cost = match options.zeta {
            Zeta::Fixed(zeta) => search.cost_zeta(zeta)?,
            Zeta::Optimised => {
                let zero = search.cost_zeta(0)?;
                let mut best = None;
                let mut lm = LocalMinimum::new(0, params.m, 1);
                while let Some(zeta) = lm.next_x() {
                    let cost = search.cost_zeta(zeta)?;
                    lm.update(cost.log2_rop);
                    keep_smaller(&mut best, cost);
                }
                best.filter(|b| zero.log2_rop >= b.log2_rop).unwrap_or(zero)
            }
        };
        if cost.log2_rop > 10000.0 {
            cost.log2_rop = f64::INFINITY;
        }
        Some(cost)
    }

    // ---------- Binary search (local_minimum) ----------

    /// Port of `estimator.util.local_minimum` with `precision` support and the
    /// `smallerf = x <= best` comparator.
    struct LocalMinimum {
        precision: i64,
        orig_start: i64,
        orig_stop: i64,
        start: i64,
        stop: i64,
        initial_bounds: (i64, i64),
        direction: i32,
        last_x: Option<i64>,
        next_x: Option<i64>,
        best: Option<(i64, f64)>,
        all_x: HashSet<i64>,
    }

    impl LocalMinimum {
        fn new(start: u64, stop: u64, precision: u64) -> Self {
            let orig_start = start as i64;
            let orig_stop = stop as i64;
            let prec = precision as i64;
            let s = (orig_start as f64 / prec as f64).ceil() as i64;
            let e = (orig_stop as f64 / prec as f64).floor() as i64;
            let internal_stop = e - 1;
            Self {
                precision: prec,
                orig_start,
                orig_stop,
                start: s,
                stop: internal_stop,
                initial_bounds: (s, internal_stop),
                direction: -1,
                last_x: None,
                next_x: Some(internal_stop),
                best: None,
                all_x: HashSet::new(),
            }
        }

        /// Advance iterator; returns next β (or ζ) value scaled by precision.
        fn next_x(&mut self) -> Option<u64> {
            if let Some(nx) = self.next_x {
                if !self.all_x.contains(&nx)
                    && nx >= self.initial_bounds.0
                    && nx <= self.initial_bounds.1
                {
                    self.last_x = Some(nx);
                    self.next_x = None;
                    return Some((nx * self.precision) as u64);
                }
            }
            None
        }

        /// Apply the gradient/bisection transitions from Python's `update()`.
        fn update(&mut self, res: f64) {
            let last = self.last_x.expect("update called before next_x");
            self.all_x.insert(last);

            if self.best.is_none() {
                self.best = Some((last, res));
            }

            // res "is not False" and smallerf(res, best.high)  <=> res <= best.high and finite-ish
            let is_better = self.best.as_ref().map_or(false, |(_, bh)| res <= *bh);
            let is_valid = !res.is_nan();

            if is_valid && is_better {
                self.best = Some((last, res));
                if self.direction.abs() != 1 {
                    self.direction = -1;
                    self.next_x = Some(last - 1);
                } else if self.direction == -1 {
                    self.direction = -2;
                    self.stop = last;
                    self.next_x = Some(((self.start + self.stop) as f64 / 2.0).ceil() as i64);
                } else {
                    self.direction = 2;
                    self.start = last;
                    self.next_x = Some(((self.start + self.stop) as f64 / 2.0).floor() as i64);
                }
            } else {
                if self.direction == -1 {
                    self.direction = 1;
                    self.next_x = Some(last + 2);
                } else if self.direction == 1 {
                    self.next_x = None;
                } else if self.direction == -2 {
                    self.start = last;
                    self.next_x = Some(((self.start + self.stop) as f64 / 2.0).ceil() as i64);
                } else {
                    self.stop = last;
                    self.next_x = Some(((self.start + self.stop) as f64 / 2.0).floor() as i64);
                }
            }

            if self.next_x == self.last_x {
                self.next_x = None;
            }
        }

        fn best_x(&self) -> Option<u64> {
            self.best.as_ref().map(|(x, _)| (x * self.precision) as u64)
        }

        fn neighborhood(&self) -> Vec<u64> {
            let Some(x) = self.best_x() else {
                return Vec::new();
            };
            let prec = self.precision as u64;
            let start = (self.orig_start as u64).max(x.saturating_sub(prec));
            let stop = (self.orig_stop as u64).min(x + prec);
            (start..stop).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;
    const REFERENCE_TOL: f64 = 1e-9;
    const LATTICE_ESTIMATOR_DEFAULT: SISLatticeOptions = SISLatticeOptions {
        red_cost_model: RedCostModel::Matzov,
        red_shape_model: RedShapeModel::Gsa,
        zeta: Zeta::Optimised,
    };

    fn raw_log2(params: &SISParameters, options: &SISLatticeOptions) -> f64 {
        match params.norm {
            Norm::L2 => super::sis_lattice::cost_euclidean_log2(params),
            Norm::Infinity => {
                estimate_sis_lattice_infinity(params, options)
                    .unwrap()
                    .log2_rop
            }
        }
    }

    fn legacy_raw_log2(params: &SISParameters) -> Option<f64> {
        let script_path = std::env::current_dir().ok()?
            .join("scripts")
            .join("run_sage_estimator.sh");
        let output = Command::new("bash")
            .arg(script_path)
            .arg(params.n.to_string())
            .arg(params.m.to_string())
            .arg(params.q.to_string())
            .arg(params.length_bound.to_string())
            .arg(params.norm.as_str())
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let stdout = String::from_utf8(output.stdout).ok()?;
        stdout.trim().parse().ok()
    }

    fn check_matches_legacy(params: SISParameters) {
        let Some(expected) = legacy_raw_log2(&params) else {
            return;
        };
        let actual = raw_log2(&params, &SISLatticeOptions::default());
        let diff = (actual - expected).abs();
        assert!(
            diff < TOL || (actual.is_infinite() && expected.is_infinite()),
            "legacy mismatch for {:?}: got {}, want {}, diff {}",
            params,
            actual,
            expected,
            diff,
        );
    }

    fn check(params: SISParameters, expected: f64) {
        // Call the internal function before .ceil() so we can compare precisely.
        let actual = raw_log2(&params, &LATTICE_ESTIMATOR_DEFAULT);
        let diff = (actual - expected).abs();
        assert!(
            diff < TOL || (actual.is_infinite() && expected.is_infinite()),
            "params {:?}: got {}, want {}, diff {}",
            params,
            actual,
            expected,
            diff,
        );
    }

    // ---- Euclidean norm ----

    #[test]
    fn l2_113_1000_2048_512() {
        check(
            SISParameters {
                n: 113,
                m: 1000,
                q: 2048,
                length_bound: 512,
                norm: Norm::L2,
            },
            46.977962983288236,
        );
    }

    #[test]
    fn l2_113_1000_2048_64() {
        check(
            SISParameters {
                n: 113,
                m: 1000,
                q: 2048,
                length_bound: 64,
                norm: Norm::L2,
            },
            107.50012989420334,
        );
    }

    #[test]
    fn l2_256_2048_8380417_1024() {
        check(
            SISParameters {
                n: 256,
                m: 2048,
                q: 8380417,
                length_bound: 1024,
                norm: Norm::L2,
            },
            200.7798045191604,
        );
    }

    #[test]
    fn l2_100_500_1024_50() {
        check(
            SISParameters {
                n: 100,
                m: 500,
                q: 1024,
                length_bound: 50,
                norm: Norm::L2,
            },
            96.46938197277763,
        );
    }

    #[test]
    fn l2_infeasible() {
        // Length bound too small → predicate False → rop = +∞.
        check(
            SISParameters {
                n: 50,
                m: 300,
                q: 512,
                length_bound: 10,
                norm: Norm::L2,
            },
            f64::INFINITY,
        );
    }

    // ---- Infinity norm ----

    #[test]
    fn linf_113_1000_2048_16() {
        check(
            SISParameters {
                n: 113,
                m: 1000,
                q: 2048,
                length_bound: 16,
                norm: Norm::Infinity,
            },
            73.66827522815265,
        );
    }

    #[test]
    fn linf_113_1000_2048_1() {
        check(
            SISParameters {
                n: 113,
                m: 1000,
                q: 2048,
                length_bound: 1,
                norm: Norm::Infinity,
            },
            376.42572646103207,
        );
    }

    #[test]
    fn linf_256_2048_8380417_32() {
        check(
            SISParameters {
                n: 256,
                m: 2048,
                q: 8380417,
                length_bound: 32,
                norm: Norm::Infinity,
            },
            250.4850444022964,
        );
    }

    #[test]
    fn linf_60_500_1024_32() {
        check(
            SISParameters {
                n: 60,
                m: 500,
                q: 1024,
                length_bound: 32,
                norm: Norm::Infinity,
            },
            40.33188031426958,
        );
    }

    #[test]
    fn linf_200_1500_2048_80_dilithium_branch() {
        // sqrt(m)*length_bound > q triggers the Dilithium-style analysis.
        check(
            SISParameters {
                n: 200,
                m: 1500,
                q: 2048,
                length_bound: 80,
                norm: Norm::Infinity,
            },
            72.13516402262441,
        );
    }

    #[test]
    fn linf_50_300_512_10() {
        check(
            SISParameters {
                n: 50,
                m: 300,
                q: 512,
                length_bound: 10,
                norm: Norm::Infinity,
            },
            40.663222438430374,
        );
    }

    #[test]
    fn linf_150_1500_8192_100() {
        check(
            SISParameters {
                n: 150,
                m: 1500,
                q: 8192,
                length_bound: 100,
                norm: Norm::Infinity,
            },
            62.91059889544071,
        );
    }

    #[test]
    fn linf_512_4096_big_q_10() {
        check(
            SISParameters {
                n: 512,
                m: 4096,
                q: 1073741824,
                length_bound: 10,
                norm: Norm::Infinity,
            },
            1000.7594302165222,
        );
    }

    fn reference_mismatches(csv: &str) -> (usize, f64, Vec<String>) {
        let mut rows = 0;
        let mut worst = 0.0f64;
        let mut mismatches = Vec::new();
        for line in csv.lines().skip(1) {
            let f: Vec<&str> = line.split(',').collect();
            let params = SISParameters {
                n: f[3].parse().unwrap(),
                m: f[4].parse().unwrap(),
                q: f[5].parse().unwrap(),
                length_bound: f[6].parse().unwrap(),
                norm: Norm::Infinity,
            };
            let options = SISLatticeOptions {
                red_cost_model: match f[0] {
                    "adps16" => RedCostModel::Adps16,
                    "matzov" => RedCostModel::Matzov,
                    other => panic!("cost model {other}"),
                },
                red_shape_model: match f[1] {
                    "lgsa" => RedShapeModel::Lgsa,
                    "gsa" => RedShapeModel::Gsa,
                    other => panic!("shape model {other}"),
                },
                zeta: match f[2] {
                    "opt" => Zeta::Optimised,
                    fixed => Zeta::Fixed(fixed.parse().unwrap()),
                },
            };
            let actual = estimate_sis_lattice_infinity(&params, &options);
            let matches = match (f[7], &actual) {
                ("ok" | "overflow", Ok(actual)) => {
                    let rop: f64 = f[8].parse().unwrap();
                    let integers = f[9] == "-"
                        || [actual.beta, actual.d, actual.zeta]
                            .iter()
                            .zip(&f[9..12])
                            .all(|(x, y)| *x == y.parse::<u64>().unwrap());
                    let deviation = if rop.is_infinite() && actual.log2_rop.is_infinite() {
                        0.0
                    } else {
                        let repetitions: f64 = f[12].parse().unwrap();
                        ((actual.log2_rop - rop) / rop)
                            .abs()
                            .max(((actual.log2_repetitions - repetitions) / rop).abs())
                    };
                    worst = worst.max(deviation);
                    integers && deviation < REFERENCE_TOL
                }
                ("OverflowError", Ok(_)) => true,
                ("ok" | "overflow" | "OverflowError", Err(_)) => false,
                (_, actual) => actual.is_err(),
            };
            if !matches {
                mismatches.push(format!("{line} -> {actual:?}"));
            }
            rows += 1;
        }
        (rows, worst, mismatches)
    }

    #[test]
    fn linf_lattice_matches_pinned_reference() {
        let (rows, _, mismatches) =
            reference_mismatches(include_str!("../../scripts/sis_lattice_pinned.csv"));
        assert!(rows > 100);
        assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
    }

    #[test]
    #[ignore]
    fn linf_lattice_matches_reference_file() {
        let path = std::env::var("SIS_LATTICE_REFERENCE").unwrap();
        let start = std::time::Instant::now();
        let (rows, worst, mismatches) =
            reference_mismatches(&std::fs::read_to_string(path).unwrap());
        println!(
            "{rows} rows, {} mismatches, max relative deviation {worst:e}, {:?} per row",
            mismatches.len(),
            start.elapsed() / rows as u32
        );
        assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
    }

    #[test]
    fn amplify_matches_prob_amplify() {
        for (log2_success, expected) in [
            (-0.02, 1.0),
            (-0.5, 2.0),
            (-1.0, 2.807354922057604),
            (-3.3, 5.459431618637297),
            (-10.0, 12.202736043250168),
            (-26.4, 28.60325446438008),
            (-27.0, 29.2032544693931),
            (-52.0, 54.20325447269972),
            (-53.5, 55.70325447269972),
            (-100.0, 102.20325447269973),
            (-1000.0, 1002.2032544726998),
            (-1024.5, 1026.7032544726997),
            (-2040.3, 2042.5028147545586),
            (-2044.7, 2046.8813263778125),
            (-2046.5, 2048.6182919719786),
            (-2047.5, 2050.2032544727),
            (-2048.99, 2050.2032544727),
            (-2049.0, f64::INFINITY),
            (-2049.01, f64::INFINITY),
        ] {
            let actual = super::sis_lattice::amplify_trials_log2(0.99, log2_success);
            assert!(
                actual == expected || (actual - expected).abs() < 1e-12 * expected,
                "log2 p = {log2_success}: got {actual}, want {expected}",
            );
        }
    }

    #[test]
    fn linf_lattice_rejects_unsupported_instances() {
        let options = SISLatticeOptions {
            red_cost_model: RedCostModel::Adps16,
            red_shape_model: RedShapeModel::Lgsa,
            zeta: Zeta::Fixed(0),
        };
        for (n, length_bound, norm) in [
            (64, 1, Norm::L2),
            (0, 1, Norm::Infinity),
            (64, 0, Norm::Infinity),
            (64, 128, Norm::Infinity),
            (2, 64, Norm::Infinity),
        ] {
            let params = SISParameters {
                n,
                m: 256,
                q: 257,
                length_bound,
                norm,
            };
            let err = estimate_sis_lattice_infinity(&params, &options).unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        }
    }

    // Public entry point sanity check.
    #[test]
    fn entry_point_ceils() {
        let p = SISParameters {
            n: 113,
            m: 1000,
            q: 2048,
            length_bound: 512,
            norm: Norm::L2,
        };
        let r = estimate_sis_security(&p).unwrap();
        assert_eq!(r.secpar, 47.0);
    }

    #[test]
    fn entry_point_ceils_linf() {
        for (n, m, q, secpar) in [
            (113, 1000, 2048, 242.0),
            (16, 2048, 2147483489, 69.0),
            (48, 4096, 2147483489, 222.0),
        ] {
            let p = SISParameters {
                n,
                m,
                q,
                length_bound: 1,
                norm: Norm::Infinity,
            };
            assert_eq!(estimate_sis_security(&p).unwrap().secpar, secpar);
        }
    }

    #[test]
    fn entry_point_rejects_easy_linf_instance() {
        let p = SISParameters {
            n: 2,
            m: 512,
            q: 1048573,
            length_bound: 1000,
            norm: Norm::Infinity,
        };
        assert!(estimate_sis_security(&p).is_err());
    }

    #[test]
    fn rejects_trivial_l2_instance() {
        let p = SISParameters {
            n: 64,
            m: 256,
            q: 257,
            length_bound: 128,
            norm: Norm::L2,
        };
        let err = estimate_sis_security(&p).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn rejects_trivial_linf_instance() {
        let p = SISParameters {
            n: 64,
            m: 256,
            q: 257,
            length_bound: 128,
            norm: Norm::Infinity,
        };
        let err = estimate_sis_security(&p).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn legacy_estimator_entry_point_is_preserved() {
        let _legacy: fn(&SISParameters) -> Result<EstimatorResult, io::Error> =
            estimate_sis_security_lattice_estimator;
    }

    #[test]
    fn matches_legacy_estimator_when_available() {
        for params in [
            SISParameters {
                n: 113,
                m: 1000,
                q: 2048,
                length_bound: 512,
                norm: Norm::L2,
            },
            SISParameters {
                n: 113,
                m: 1000,
                q: 2048,
                length_bound: 16,
                norm: Norm::Infinity,
            },
            SISParameters {
                n: 113,
                m: 1000,
                q: 2048,
                length_bound: 1,
                norm: Norm::Infinity,
            },
            SISParameters {
                n: 256,
                m: 2048,
                q: 8380417,
                length_bound: 32,
                norm: Norm::Infinity,
            },
            SISParameters {
                n: 356,
                m: 5048,
                q: 8382343240417,
                length_bound: 323423,
                norm: Norm::L2,
            },
            SISParameters {
                n: 512,
                m: 4096,
                q: 1073741824,
                length_bound: 10,
                norm: Norm::Infinity,
            },
        ] {
            check_matches_legacy(params);
        }
    }
}
