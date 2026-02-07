//! Memory pool for preallocated vectors (RingElement and QuadraticExtension)
//!
//! # Usage
//!
//! ## Step 1: Run your code with access tracking
//! Your code will automatically track all pool accesses via the abstraction.
//!
//! ## Step 2: Save access statistics
//! At the end of your run, call:
//! ```rust,ignore
//! use rokoko::common::pool::save_access_stats;
//! save_access_stats("pool_stats.txt").expect("Failed to save stats");
//! ```
//!
//! ## Step 3: Preallocate pools for subsequent runs
//! At the start of your next run, call:
//! ```rust,ignore
//! use rokoko::common::pool::load_and_preallocate;
//! load_and_preallocate("pool_stats.txt").expect("Failed to load stats");
//! ```
//!
//! This will preallocate all the vectors based on the access patterns from the previous run,
//! eliminating allocation overhead and warnings during execution.
//!
//! Note: The pool tracks vectors by their length, not by matrix dimensions,
//! since matrices are just flattened vectors.

use std::{
    collections::HashMap,
    fs,
    io::{self, Write},
    path::Path,
    sync::{LazyLock, Mutex},
};

use crate::common::{
    ring_arithmetic::{QuadraticExtension, Representation, RingElement},
    sumcheck_element::SumcheckElement,
};

static ZERO_REP_INCOMPLETE_NTT: LazyLock<RingElement> =
    LazyLock::new(|| RingElement::zero(Representation::IncompleteNTT));

/// Pool for preallocated RingElement vectors
static PREALLOCATED_RING: LazyLock<Mutex<HashMap<usize, Vec<Vec<RingElement>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Pool for preallocated QuadraticExtension vectors
static PREALLOCATED_QUAD: LazyLock<Mutex<HashMap<usize, Vec<Vec<QuadraticExtension>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Tracks accesses to the pool (sizes requested)
static ACCESS_TRACKER: LazyLock<Mutex<HashMap<usize, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Tracks accesses to the quadratic extension pool
static ACCESS_TRACKER_QUAD: LazyLock<Mutex<HashMap<usize, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Get a preallocated vector from the RingElement pool
#[inline]
pub fn get_preallocated_ring_element_vec(len: usize) -> Vec<RingElement> {
    // Track this access
    {
        let mut tracker = ACCESS_TRACKER.lock().expect("tracker poisoned");
        *tracker.entry(len).or_insert(0) += 1;
    }

    let mut pool = PREALLOCATED_RING.lock().expect("pool poisoned");
    pool.get_mut(&len).and_then(|v| v.pop()).unwrap_or_else(|| {
        println!(
            "Preallocated RingElement pool miss for size {}, allocating new",
            len
        );
        vec![ZERO_REP_INCOMPLETE_NTT.clone(); len]
    })
}

/// Get a preallocated vector from the QuadraticExtension pool
#[inline]
pub fn get_preallocated_quad_vec(len: usize) -> Vec<QuadraticExtension> {
    // Track this access
    {
        let mut tracker = ACCESS_TRACKER_QUAD.lock().expect("tracker poisoned");
        *tracker.entry(len).or_insert(0) += 1;
    }

    let mut pool = PREALLOCATED_QUAD.lock().expect("pool poisoned");
    pool.get_mut(&len).and_then(|v| v.pop()).unwrap_or_else(|| {
        println!(
            "Preallocated QuadraticExtension pool miss for size {}, allocating new",
            len
        );
        vec![QuadraticExtension::zero(); len]
    })
}

/// Preallocate RingElement vectors in the pool
pub fn preallocate_ring_element_vecs(len: usize, count: usize) {
    let mut pool = PREALLOCATED_RING.lock().expect("pool poisoned");
    let entry = pool.entry(len).or_insert_with(Vec::new);
    entry.reserve(count);
    for _ in 0..count {
        entry.push(vec![ZERO_REP_INCOMPLETE_NTT.clone(); len]);
    }
}

/// Preallocate QuadraticExtension vectors in the pool
pub fn preallocate_quad_vecs(len: usize, count: usize) {
    let mut pool = PREALLOCATED_QUAD.lock().expect("pool poisoned");
    let entry = pool.entry(len).or_insert_with(Vec::new);
    entry.reserve(count);
    for _ in 0..count {
        entry.push(vec![QuadraticExtension::zero(); len]);
    }
}

/// Save access statistics to a file
pub fn save_access_stats<P: AsRef<Path>>(path: P) -> io::Result<()> {
    let tracker = ACCESS_TRACKER.lock().expect("tracker poisoned");
    let tracker_quad = ACCESS_TRACKER_QUAD.lock().expect("tracker poisoned");

    let mut file = fs::File::create(path)?;

    writeln!(file, "# RingElement pool accesses")?;
    let mut ring_accesses: Vec<_> = tracker.iter().collect();
    ring_accesses.sort_by_key(|(k, _)| *k);
    for (len, count) in ring_accesses {
        writeln!(file, "ring {} {}", len, count)?;
    }

    writeln!(file, "\n# QuadraticExtension pool accesses")?;
    let mut quad_accesses: Vec<_> = tracker_quad.iter().collect();
    quad_accesses.sort_by_key(|(k, _)| *k);
    for (len, count) in quad_accesses {
        writeln!(file, "quad {} {}", len, count)?;
    }

    Ok(())
}

/// Load access statistics from a file and preallocate accordingly.
/// If the file doesn't exist, initializes an empty pool (no error).
pub fn load_and_preallocate<P: AsRef<Path>>(path: P) -> io::Result<()> {
    let contents = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            println!("Pool stats file not found, starting with empty pool");
            return Ok(());
        }
        Err(e) => return Err(e),
    };

    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 3 {
            continue;
        }

        let pool_type = parts[0];
        let len: usize = parts[1].parse().ok().unwrap_or(0);
        let count: usize = parts[2].parse().ok().unwrap_or(0);

        if len == 0 || count == 0 {
            continue;
        }

        match pool_type {
            "ring" => {
                preallocate_ring_element_vecs(len, count);
            }
            "quad" => {
                preallocate_quad_vecs(len, count);
            }
            _ => {}
        }
    }

    Ok(())
}
