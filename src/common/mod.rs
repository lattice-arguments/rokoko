//! Operations on ring-element vectors and the Fiat-Shamir transcript. Start with `hash`: its
//! `HashWrapper` transcript is the source of every verifier challenge, which `short_challenge`
//! turns into short ring challenges. `decomposition` splits ring vectors into short digits,
//! `norms` measures them, `matrix` holds the column- and row-major containers, `structured_row`
//! the tensor-structured rows, and `sampling` draws randomness outside the transcript (the
//! CRS, test witnesses). `init` holds `init_common`, re-exported here, which seeds the
//! thread-local RNG and builds the ring's lazy tables.

pub mod decomposition;
pub mod hash;
mod init;
pub mod matrix;
pub mod norms;
pub mod sampling;
pub mod short_challenge;
pub mod structured_row;

pub use init::init_common;
