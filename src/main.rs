#[cfg_attr(rokoko_ring = "n128_d4", allow(dead_code))]
mod instantiation;
#[cfg(rokoko_ring = "n128_d4")]
mod instantiation_n128_d4;

/// The parameter sets of the ring spec the crate is built with.
#[cfg(not(rokoko_ring = "n128_d4"))]
use instantiation as ring_instantiation;
#[cfg(rokoko_ring = "n128_d4")]
use instantiation_n128_d4 as ring_instantiation;

use instantiation::ParamSet;
use rokoko::common::init_common;
use rokoko::common::short_challenge::repetition_rate;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
#[cfg(not(feature = "snark"))]
use rokoko::protocol::parties::executor::execute;

fn main() {
    let set = match std::env::args().nth(1) {
        Some(arg) => arg.parse::<ParamSet>().unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2);
        }),
        None => ParamSet::P28,
    };
    let (inst, chain) = if cfg!(feature = "snark") {
        (ring_instantiation::snark_instantiation(set), "exact-norm")
    } else {
        (ring_instantiation::instantiation(set), "plain")
    };
    let inst = inst.unwrap_or_else(|| {
        eprintln!("{} has no {chain} chain", set.name());
        std::process::exit(2);
    });
    println!("Using {}...", set.name().replace('-', ""));

    #[cfg(feature = "unsafe-sumcheck")]
    {
        println!("Sumcheck unsafe...");
    }

    // Check AVX-512F support
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            println!("✓ AVX-512F is enabled in runtime detection and available on this CPU");
            #[cfg(all(target_feature = "avx512f"))]
            {
                println!("✓✓ AVX-512F is enabled at compile time");
            }
            #[cfg(not(target_feature = "avx512f"))]
            {
                println!("✗ AVX-512F is NOT enabled at compile time");
            }
        } else {
            println!("✗ AVX-512F is NOT available on this CPU");
        }

        if is_x86_feature_detected!("avx512dq") {
            println!("✓ AVX-512DQ is enabled in runtime detection and available on this CPU");
            #[cfg(all(target_feature = "avx512dq"))]
            {
                println!("✓✓ AVX-512DQ is enabled at compile time");
            }
            #[cfg(not(target_feature = "avx512dq"))]
            {
                println!("✗ AVX-512DQ is NOT enabled at compile time");
            }
        } else {
            println!("✗ AVX-512DQ is NOT available on this CPU");
        }
        if is_x86_feature_detected!("avx512vbmi2") {
            println!("✓ AVX-512VBMI2 is enabled in runtime detection and available on this CPU");
            #[cfg(all(target_feature = "avx512vbmi2"))]
            {
                println!("✓✓ AVX-512VBMI2 is enabled at compile time");
            }
            #[cfg(not(target_feature = "avx512vbmi2"))]
            {
                println!("✗ AVX-512VBMI2 is NOT enabled at compile time");
            }
        } else {
            println!("✗ AVX-512VBMI2 is NOT available on this CPU");
        }
    }

    // Trigger CPU feature detection and print the detected features
    incomplete_rexl::cpu_features::print_features();

    #[cfg(feature = "crt-commitment")]
    {
        println!("Using CRT commitment...");
    }

    #[cfg(not(feature = "crt-commitment"))]
    {
        println!(
            "Using {}-bits commitment...",
            rokoko::common::config::MOD_Q.ilog2() + 1
        );
    }

    #[cfg(feature = "parallel")]
    {
        println!("Using parallel commitment...");
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("✗ AVX-512 is only available on x86_64 architecture");
    }
    let challenge_set_repetition_rate = repetition_rate();
    println!(
        "Expected repetition rate for challenge set: {:.2}",
        challenge_set_repetition_rate
    );

    let tracing_guards = rokoko::tracing::setup(set.name());

    init_common();
    #[cfg(feature = "snark")]
    {
        println!("Running executor in SNARK mode...");
        rokoko::protocol::parties::executor::execute_snark(&inst);
    }
    #[cfg(not(feature = "snark"))]
    {
        println!("Running executor...");
        execute(&inst);
    }
    #[cfg(feature = "calibration")]
    rokoko::common::norms::calibration::print_table();

    drop(tracing_guards);
    #[cfg(feature = "profile")]
    rokoko::tracing::print_artifact_paths(rokoko::tracing::run_dir());
}

#[cfg(test)]
mod tests {
    use super::ring_instantiation::instantiation;
    use super::ParamSet;
    use rokoko::common::init_common;
    use rokoko::protocol::parties::executor::{execute, execute_to_boundary};
    use std::num::NonZeroUsize;

    #[test]
    fn initial_witness_fills_the_root() {
        for inst in ParamSet::ALL.into_iter().filter_map(instantiation) {
            let root = inst.root();
            assert_eq!(
                inst.witness.height * inst.witness.decomposition_chunks,
                root.witness_height
            );
            assert_eq!(inst.witness.width, root.witness_width);
        }
    }

    /// The boundary tests stop a few rounds in, so they never reach the last sumcheck round,
    /// whose recursions are single levels and whose level 0 is therefore itself a leaf. Only a
    /// whole-chain run covers it.
    #[test]
    fn full_chain_verifies() {
        init_common();
        execute(&instantiation(ParamSet::P28).unwrap());
    }

    #[test]
    fn round_boundary_extraction() {
        init_common();
        let inst = instantiation(ParamSet::P28).unwrap();
        let mut run = execute_to_boundary(&inst, NonZeroUsize::new(3).unwrap());

        assert_eq!(run.prover.witness.height, 256);
        assert_eq!(run.prover.witness.width, 32);
        assert_eq!(run.verifier.commitment_root.len(), 1);
        assert_eq!(run.prover.claims.len(), 2);
        assert_eq!(run.verifier.claims.len(), 2);
        assert_eq!(run.prover.evaluation_points, run.verifier.evaluation_points);

        let mut prover_bytes = [0u8; 16];
        let mut verifier_bytes = [0u8; 16];
        run.prover
            .transcript
            .fill_from_xof(b"round-boundary-test", &mut prover_bytes);
        run.verifier
            .transcript
            .fill_from_xof(b"round-boundary-test", &mut verifier_bytes);
        assert_eq!(prover_bytes, verifier_bytes);

        assert_eq!(run.crs.cks.len(), run.verifier_crs.structured_cks.len());
        let first_row = &run.verifier_crs.structured_cks[0][0];
        assert_eq!(first_row.tensor_layers.len(), 1);

        let run4 = execute_to_boundary(&inst, NonZeroUsize::new(4).unwrap());
        assert_eq!(run4.prover.witness.height, 512);
        assert_eq!(run4.prover.witness.width, 8);
        assert_eq!(
            run4.prover.evaluation_points,
            run4.verifier.evaluation_points
        );
    }
}
