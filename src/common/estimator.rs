use std::io;
use std::process::Command;

use crate::common::config::{DEGREE, MOD_Q};

#[derive(Debug, Clone, Copy)]
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

    Ok(EstimatorResult { secpar: secpar.ceil() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimator() {
        let params = SISParameters {
            n: 113,
            m: 1000,
            q: 2048,
            length_bound: 512,
            norm: Norm::L2,
        };

        let result = estimate_sis_security(&params).unwrap();
        println!("Log2(ROP): {}", result.secpar);
        debug_assert!(result.secpar > 0.0);
    }
}
