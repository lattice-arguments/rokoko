// Minimal UTC formatting. Replaces `chrono`, which pulls a timezone database and
// a strftime engine to produce the two fixed-shape timestamps profiling needs.

use std::time::{SystemTime, UNIX_EPOCH};

/// Days since 1970-01-01 -> (year, month, day). Hinnant's `civil_from_days`.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as i64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

pub struct Utc {
    y: i64,
    mo: u32,
    d: u32,
    h: u32,
    mi: u32,
    s: u32,
}

pub fn now_utc() -> Utc {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs()) as i64;
    let (y, mo, d) = civil_from_days(secs.div_euclid(86_400));
    let rem = secs.rem_euclid(86_400);
    Utc {
        y,
        mo,
        d,
        h: (rem / 3600) as u32,
        mi: (rem % 3600 / 60) as u32,
        s: (rem % 60) as u32,
    }
}

impl Utc {
    /// `YYYY-MM-DDTHH:MM:SSZ`
    pub fn iso8601(&self) -> String {
        let Utc { y, mo, d, h, mi, s } = *self;
        format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
    }

    /// `YYYYMMDD-HHMMSS`
    pub fn filename(&self) -> String {
        let Utc { y, mo, d, h, mi, s } = *self;
        format!("{y:04}{mo:02}{d:02}-{h:02}{mi:02}{s:02}")
    }
}
