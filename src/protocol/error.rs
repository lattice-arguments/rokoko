//! Errors returned when checking a proof.
//!
//! Configuration, CRS construction and statement builders retain their own invariants.
//! A failed verification may consume the transcript and sumcheck context; retry with fresh ones.
use std::{error::Error, fmt};

pub type VerificationResult<T> = Result<T, VerificationError>;

/// A rejected proof or an invalid use of the verifier.
#[derive(Debug, Clone, PartialEq)]
pub enum VerificationError {
    /// A proof or statement has the wrong dimensions, optional fields or polynomial encoding.
    InvalidShape(String),
    /// The proof, configuration, commitment or context disagree on the next round.
    InvalidRound(String),
    /// An algebraic consistency check failed.
    CheckFailed(String),
    /// A sumcheck polynomial does not sum to the current claim (zero-based local round).
    SumcheckRound { round: usize },
    /// The final sumcheck evaluation disagrees with the claimed opening.
    SumcheckEvaluation,
    /// A measured norm exceeds the configured bound.
    NormExceeded {
        label: String,
        value: f64,
        bound: f64,
    },
    /// Sumcheck contexts carry mutable state and can only serve one verification attempt.
    ContextReused,
}

impl fmt::Display for VerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape(message)
            | Self::InvalidRound(message)
            | Self::CheckFailed(message) => f.write_str(message),
            Self::SumcheckRound { round } => {
                write!(f, "round claim mismatch in sumcheck round {round}")
            }
            Self::SumcheckEvaluation => f.write_str("sumcheck final evaluation mismatch"),
            Self::NormExceeded {
                label,
                value,
                bound,
            } => write!(
                f,
                "L2 norm of {label} = {value} exceeds the registered bound {bound}"
            ),
            Self::ContextReused => f.write_str(
                "a sumcheck context serves one proof; build a fresh one with init_verifier",
            ),
        }
    }
}

impl Error for VerificationError {}
