use crate::common::{config::MOD_Q, ring_arithmetic::RingElement};

const HARD_ASSERTION: bool = false;

macro_rules! maybe_assert {
    ($cond:expr $(,)?) => {
        if HARD_ASSERTION {
            assert!($cond);
        } else if !($cond) {
            eprintln!("Assertion failed: {}", stringify!($cond));
        }
    };
    ($cond:expr, $($arg:tt)+) => {
        if HARD_ASSERTION {
            assert!($cond, $($arg)+);
        } else if !($cond) {
            eprintln!(
                "Assertion failed: {} | {}",
                stringify!($cond),
                format_args!($($arg)+)
            );
        }
    };
}


pub fn inf_norm(vec: &[RingElement]) -> u64 {
    vec.iter()
        .map(|el| {
            let mut el_cloned = el.clone();
            el_cloned.from_incomplete_ntt_to_even_odd_coefficients();
            el_cloned
                .v
                .map(|x| x)
                .iter()
                .map(|&x| {
                    if x > MOD_Q / 2 {
                        MOD_Q - x as u64
                    } else {
                        x as u64
                    }
                })
                .max()
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0)
}

pub fn l2_norm(vec: &[RingElement]) -> f64 {
    let mut sum = 0u64;
    for el in vec {
        let mut el_cloned = el.clone();
        el_cloned.from_incomplete_ntt_to_even_odd_coefficients();
        for &x in el_cloned.v.map(|x| x).iter() {
            let centered = if x < MOD_Q / 2 { x } else { MOD_Q - x };
            sum += centered * centered;
        }
    }
    (sum as f64).sqrt() as f64
}

pub fn l2_norm_coeffs(vec: &[RingElement]) -> f64 {
    let mut sum = 0u64;
    for el in vec {
        for &x in el.v.map(|x| x).iter() {
            let centered = if x < MOD_Q / 2 { x } else { MOD_Q - x };
            sum += centered * centered;
        }
    }
    (sum as f64).sqrt() as f64
}

pub fn assert_norm_bounded(label: &str, value: f64, bound: f64) {
    tracing::debug!("L2 norm of {label}: {value} (bound {bound})");
    #[cfg(feature = "calibration")]
    calibration::record(label, value, bound);
    maybe_assert!(
        value <= bound,
        "L2 norm of {label} = {value} exceeds the registered bound {bound}"
    );
}

/// Measures the norm schedule a run actually produces, in the shape `assign_norm_bounds`
/// consumes: one row per round of the chain, `[first column, second column]`.
#[cfg(feature = "calibration")]
pub mod calibration {
    use std::sync::Mutex;

    static MEASURED: Mutex<Vec<(String, f64, f64)>> = Mutex::new(Vec::new());

    /// Where a measurement belongs in a round's row. `Column` matches the field order
    /// `assign_norm_bounds` assigns: `[norm_bound, most_inner_norm_bound]` for a sumcheck round,
    /// `[norm_bound, projection_norm_bound]` for an intermediate one, and
    /// `[witness_norm_bound, projection_norm_bound]` for a simple one. `Projection` is the exact
    /// projection-image claim a sumcheck round makes, and fills the third column.
    enum Slot {
        Column(usize),
        Projection,
    }

    fn slot(label: &str) -> Slot {
        match label {
            "norm claim via inner-product"
            | "norm claim via inner-product (intermediate)"
            | "folded witness in simple verifier" => Slot::Column(0),
            "most inner norm claim via inner-product"
            | "projection image in intermediate verifier"
            | "projection image in simple verifier" => Slot::Column(1),
            "projection image norm claim via inner-product" => Slot::Projection,
            other => panic!("calibration: unknown norm label {other:?}; add it to `slot`"),
        }
    }

    /// One round's measurements: the two calibrated columns plus the optional projection claim.
    struct Row {
        columns: [(f64, f64); 2],
        projection: Option<(f64, f64)>,
    }

    pub(super) fn record(label: &str, value: f64, bound: f64) {
        MEASURED
            .lock()
            .unwrap()
            .push((label.to_string(), value, bound));
    }

    /// Print what this run measured, as a table of headroom against the registered bounds and as
    /// an array literal to paste over the chain's `NB_*` constant. The literal holds the raw
    /// measurements; `assign_norm_bounds` applies `NORM_MARGIN` on top.
    ///
    /// A round contributes a variable number of measurements, so rows are cut at each column-0
    /// label rather than by counting.
    pub fn print_table() {
        let measured = MEASURED.lock().unwrap();
        if measured.is_empty() {
            println!("\ncalibration: no norms were measured");
            return;
        }

        let mut rows: Vec<Row> = Vec::new();
        for (label, value, bound) in measured.iter() {
            match slot(label) {
                Slot::Column(0) => rows.push(Row {
                    columns: [(*value, *bound), (f64::NAN, f64::NAN)],
                    projection: None,
                }),
                Slot::Column(col) => {
                    let row = rows.last_mut().unwrap_or_else(|| {
                        panic!("calibration: {label:?} arrived before its round's first column")
                    });
                    assert!(
                        row.columns[col].0.is_nan(),
                        "calibration: two measurements for column {col} in one round ({label})"
                    );
                    row.columns[col] = (*value, *bound);
                }
                Slot::Projection => {
                    let row = rows.last_mut().unwrap_or_else(|| {
                        panic!("calibration: {label:?} arrived before its round's first column")
                    });
                    assert!(
                        row.projection.is_none(),
                        "calibration: two projection measurements in one round"
                    );
                    row.projection = Some((*value, *bound));
                }
            }
        }

        println!("\n=== calibration: measured norms vs registered bounds ===");
        for (i, row) in rows.iter().enumerate() {
            for (col, (value, bound)) in row.columns.iter().enumerate() {
                if value.is_nan() {
                    println!("round {i} col {col}: not measured");
                    continue;
                }
                let flag = if value > bound { "  OVER" } else { "" };
                println!(
                    "round {i} col {col}: measured {value:.6}  bound {bound:.6}  ratio {:.4}{flag}",
                    value / bound
                );
            }
            if let Some((value, bound)) = row.projection {
                let flag = if value > bound { "  OVER" } else { "" };
                println!(
                    "round {i} projection: measured {value:.6}  bound {bound:.6}  ratio {:.4}{flag}",
                    value / bound
                );
            }
        }

        println!("\n=== calibration: paste over this chain's NB_* constant ===");
        println!("[[f64; 3]; {}] = [", rows.len());
        for row in rows.iter() {
            // A round that makes no projection claim leaves its upper bound unset.
            let projection = match row.projection {
                Some((value, _)) => value.to_string(),
                None => "f64::INFINITY".to_string(),
            };
            println!(
                "    [{}, {}, {}],",
                row.columns[0].0, row.columns[1].0, projection
            );
        }
        println!("];");
    }
}
