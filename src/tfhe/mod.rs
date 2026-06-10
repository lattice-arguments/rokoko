//! WIP (PR3): CGGI/TFHE bootstrapping implemented natively over the proof
//! system's modulus q = MOD_Q (no auxiliary 2^64 torus). Layers done: poly,
//! glwe (untested beyond poly). Still to write: lwe.rs (LWE + keyswitch),
//! bootstrap.rs (mod switch, redundant LUT, blind rotation, sample extract),
//! params (toy N=256/n=8 and scaled-Zama N=2048/n=918/k=1, B=2^25 l=2,
//! TUniform(31)/TUniform(3)), correctness tests.

pub mod glwe;
pub mod poly;
