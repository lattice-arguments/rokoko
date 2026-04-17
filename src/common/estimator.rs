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

pub fn estimate_sis_security(params: &SISParameters) -> Result<EstimatorResult, io::Error> {
    if params.length_bound >= (params.q - 1) / 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "SIS trivially easy. Please set norm bound < (q-1)/2.",
        ));
    }

    let log2_rop = match params.norm {
        Norm::L2 => sis_lattice::cost_euclidean_log2(params),
        Norm::Infinity => sis_lattice::cost_infinity_top_log2(params),
    };

    Ok(EstimatorResult {
        secpar: log2_rop.ceil(),
    })
}

/// Legacy implementation that shells out to Sage/Python lattice-estimator.
pub fn estimate_sis_security_lattice_estimator(
    params: &SISParameters,
) -> Result<EstimatorResult, io::Error> {
    let script_path = std::env::current_dir()?.join("run_sage_estimator.sh");

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
    //! `lattice-estimator` project, specialized for the default cost models:
    //! - `red_cost_model = RC.MATZOV` (list_decoding-classical)
    //! - `red_shape_model = "gsa"` (Geometric Series Assumption)

    use super::{Norm, SISParameters};
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

    // ---------- GSA simulator ----------

    /// `simulator.GSA(d, n, q, β, xi=1, tau=False)` with `tau=False`.
    /// Returns squared Gram-Schmidt norms `r_i`, i = 0..d-1.
    fn gsa_tau_false(d: u64, n: u64, q: f64, beta: u64) -> Vec<f64> {
        assert!(beta >= 2 && beta <= d);
        let d_f = d as f64;
        // log_vol in base 2, xi=1 so log2(xi)=0
        let log_vol = q.log2() * (d_f - n as f64);
        let delta_val = delta(beta);
        let log_delta = delta_val.log2();
        let mut r = Vec::with_capacity(d as usize);
        for i in 0..d {
            let r_log = (d_f - 1.0 - 2.0 * (i as f64)) * log_delta + log_vol / d_f;
            r.push(2f64.powf(2.0 * r_log));
        }
        r
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

    /// `prob.amplify(target, success, majority=False)` in log-domain.
    fn amplify_log2(target: f64, log2_success: f64) -> f64 {
        if !log2_success.is_finite() {
            return f64::INFINITY;
        }

        let log2_target = target.log2();
        if log2_success >= log2_target {
            return 0.0;
        }

        // `prob.amplify` caps the working precision at 2048 bits. Once `p`
        // falls below that resolution, `1 - p` rounds to 1 and the estimator
        // returns infinity.
        if log2_success < -2048.0 {
            return f64::INFINITY;
        }

        let success = 2f64.powf(log2_success);
        if success == 0.0 {
            return (-(1.0 - target).ln()).log2() - log2_success;
        }

        let denom = (-success).ln_1p();
        if !denom.is_finite() || denom == 0.0 {
            return f64::INFINITY;
        }

        let trials = ((1.0 - target).ln() / denom).ceil();
        if trials <= 1.0 {
            0.0
        } else {
            trials.log2()
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

    /// `cost_infinity(β, params, zeta, d=m)` from Python. Returns log2(rop)
    /// (possibly `+∞`).
    fn cost_infinity_log2(beta: u64, params: &SISParameters, zeta: u64) -> f64 {
        let d = params.m;
        if zeta >= d {
            return f64::INFINITY;
        }
        let d_ = d - zeta;
        if d_ < beta {
            return f64::INFINITY;
        }

        // r = GSA(d_, d_ - n, q, β, xi=1, tau=False)
        // Safety: d_ - n must be sensible; Python GSA asserts n ≥ 0 implicitly.
        if d_ < params.n {
            return f64::INFINITY;
        }
        let r = gsa_tau_false(d_, d_ - params.n, params.q as f64, beta);
        let (rho, cost_red_log2, log2_big_n, sieve_dim) = matzov_short_vectors(beta, d_);
        let bkz = bkz_rop_log2(beta, d_);

        let d_f = d as f64;
        let d_under_f = d_ as f64;
        let lb = params.length_bound as f64;
        let q_f = params.q as f64;

        let log_trial_prob = if d_f.sqrt() * lb <= q_f {
            // Non-Dilithium style
            let vector_length = rho * r[0].sqrt();
            let sigma = vector_length / d_under_f.sqrt();
            d_under_f * coordinate_success_log2(lb, sigma)
        } else {
            // Dilithium style
            let r0 = r[0];
            let idx_start = if (r0.sqrt() - q_f).abs() < 1e-8 {
                // first index where r[i] < r[0]
                r.iter().position(|&x| x < r0).unwrap_or(0) as u64
            } else {
                0
            };
            let idx_end = if (r[(d_ - 1) as usize] - 1.0).abs() < 1e-8 {
                // first i where sqrt(r[i]) <= 1+1e-8, then i-1
                match r.iter().position(|&x| x.sqrt() <= 1.0 + 1e-8) {
                    Some(i) => (i as i64 - 1) as u64,
                    None => d_ - 1,
                }
            } else {
                d_ - 1
            };
            let vector_length = r[idx_start as usize].sqrt();
            let gaussian_coords =
                ((idx_end as i64) - (idx_start as i64) + 1).max(sieve_dim as i64) as f64;
            let sigma = vector_length / gaussian_coords.sqrt();
            let coords_log2 = coordinate_success_log2(lb, sigma);
            if !coords_log2.is_finite() {
                return f64::INFINITY;
            }
            let mut ltp = coords_log2 * gaussian_coords;
            ltp += ((2.0 * lb + 1.0) / q_f).log2() * (idx_start as f64);
            ltp
        };

        let log2_probability = (log_trial_prob + log2_big_n).min(0.0);
        if log2_probability.is_nan() {
            return f64::INFINITY;
        }
        let log2_amp = amplify_log2(0.99, log2_probability);
        if !log2_amp.is_finite() {
            return f64::INFINITY;
        }
        // rop = cost_red * amp; stay in the log-domain to avoid underflow or
        // overflow in the repeat count.
        let _ = bkz; // red entry not used outside reporting
        cost_red_log2 + log2_amp
    }

    /// `cost_zeta(ζ, params)` — optimizes β using `local_minimum(40, baseline_beta+1, precision=2)`.
    fn cost_zeta_log2(zeta: u64, params: &SISParameters, baseline_beta: u64) -> f64 {
        let f = |beta: u64| -> f64 { cost_infinity_log2(beta, params, zeta) };

        let mut lm = LocalMinimum::new(40, baseline_beta + 1, 2);
        while let Some(beta) = lm.next_x() {
            let v = f(beta);
            lm.update(v);
        }
        for beta in lm.neighborhood() {
            let v = f(beta);
            lm.update_point(beta, v);
        }
        lm.best_y().unwrap_or(f64::INFINITY)
    }

    pub(super) fn cost_infinity_top_log2(params: &SISParameters) -> f64 {
        // 1. Baseline: params with norm=2, length_bound adjusted.
        let baseline_lb = if params.length_bound == 1 {
            2
        } else {
            params.length_bound
        };
        let baseline = SISParameters {
            norm: Norm::L2,
            length_bound: baseline_lb,
            ..params.clone()
        };
        // We need baseline_beta: recompute as in cost_euclidean but keep β.
        let baseline_beta = baseline_cost_beta(&baseline);

        let f_zeta = |zeta: u64| -> f64 { cost_zeta_log2(zeta, params, baseline_beta) };

        // Outer binary search over zeta in [0, m).
        let mut lm: LocalMinimum = LocalMinimum::new(0, params.m, 1);
        while let Some(zeta) = lm.next_x() {
            let v = f_zeta(zeta);
            lm.update(v);
        }
        let binary_best = lm.best_y().unwrap_or(f64::INFINITY);
        // Explicitly include ζ = 0 (Python: `cost = min(it.y, f(0))`).
        let zero = f_zeta(0);
        binary_best.min(zero)
    }

    /// Compute the β used by `cost_euclidean` without the predicate short-circuit
    /// (we only need it for `cost_zeta`'s baseline).
    fn baseline_cost_beta(params: &SISParameters) -> u64 {
        let d_opt = opt_sis_d(params).floor() as u64;
        let d = d_opt.min(params.m);
        let delta_val = solve_for_delta_euclidean(params, d);
        if delta_val >= 1.0 {
            let b = beta_from_delta(delta_val);
            if b <= d {
                return b;
            }
        }
        d
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

        /// Evaluate a specific point from the neighborhood scan — bypasses the
        /// iterator state machine but still updates `best`.
        fn update_point(&mut self, x_scaled: u64, res: f64) {
            let x = (x_scaled as i64) / self.precision;
            self.last_x = Some(x);
            if self.best.is_none() {
                self.best = Some((x, res));
            } else if !res.is_nan() && res <= self.best.as_ref().unwrap().1 {
                self.best = Some((x, res));
            }
            self.all_x.insert(x);
        }

        fn best_x(&self) -> Option<u64> {
            self.best.as_ref().map(|(x, _)| (x * self.precision) as u64)
        }

        fn best_y(&self) -> Option<f64> {
            self.best.as_ref().map(|(_, y)| *y)
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

    fn legacy_raw_log2(params: &SISParameters) -> Option<f64> {
        let script_path = std::env::current_dir().ok()?.join("run_sage_estimator.sh");
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
        let actual = match params.norm {
            Norm::L2 => super::sis_lattice::cost_euclidean_log2(&params),
            Norm::Infinity => super::sis_lattice::cost_infinity_top_log2(&params),
        };
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
        let actual = match params.norm {
            Norm::L2 => super::sis_lattice::cost_euclidean_log2(&params),
            Norm::Infinity => super::sis_lattice::cost_infinity_top_log2(&params),
        };
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
        let p = SISParameters {
            n: 113,
            m: 1000,
            q: 2048,
            length_bound: 1,
            norm: Norm::Infinity,
        };
        let r = estimate_sis_security(&p).unwrap();
        assert_eq!(r.secpar, 377.0);
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
