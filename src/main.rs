use rokoko::common::init_common;
use rokoko::common::short_challenge::repetition_rate;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
#[cfg(not(feature = "snark"))]
use rokoko::protocol::parties::executor::execute;

fn main() {
    #[cfg(feature = "p-26")]
    {
        println!("Using p26...");
    }
    #[cfg(feature = "p-30")]
    {
        println!("Using p30...");
    }
    #[cfg(not(any(feature = "p-26", feature = "p-30")))]
    {
        println!("Using p28...");
    }

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

    #[cfg(feature = "incomplete-rexl")]
    {
        // Trigger CPU feature detection and print features if enabled
        incomplete_rexl::cpu_features::print_features();
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

    #[cfg(any(feature = "events", feature = "profile"))]
    let trace_base = format!(
        "{}_{}",
        trace_name(),
        rokoko_profiling::timestamp_for_filename()
    );

    #[cfg(any(feature = "events", feature = "profile"))]
    let _tracing_guards = rokoko_profiling::setup(
        cfg!(feature = "events"),
        cfg!(feature = "profile"),
        cfg!(feature = "debug"),
        &trace_base,
        &active_features(),
    );

    init_common();
    #[cfg(feature = "snark")]
    {
        println!("Running executor in SNARK mode...");
        rokoko::protocol::parties::executor::execute_snark();
    }
    #[cfg(not(feature = "snark"))]
    {
        println!("Running executor...");
        execute();
    }

    // Drop guards so the chrome trace flushes before we print its path.
    #[cfg(any(feature = "events", feature = "profile"))]
    drop(_tracing_guards);
    #[cfg(feature = "profile")]
    rokoko_profiling::print_artifact_paths(&trace_base);
}

#[cfg(any(feature = "events", feature = "profile"))]
fn trace_name() -> &'static str {
    match (
        cfg!(feature = "p-26"),
        cfg!(feature = "p-28"),
        cfg!(feature = "p-30"),
    ) {
        (true, _, _) => "p26",
        (_, true, _) => "p28",
        (_, _, true) => "p30",
        _ => panic!("--features events|profile requires one of p-26, p-28, p-30"),
    }
}

#[cfg(any(feature = "events", feature = "profile"))]
fn active_features() -> String {
    [
        cfg!(feature = "p-26").then_some("p-26"),
        cfg!(feature = "p-28").then_some("p-28"),
        cfg!(feature = "p-30").then_some("p-30"),
        cfg!(feature = "incomplete-rexl").then_some("incomplete-rexl"),
        cfg!(feature = "unsafe-sumcheck").then_some("unsafe-sumcheck"),
        cfg!(feature = "debug-hardness").then_some("debug-hardness"),
        cfg!(feature = "debug-decomp").then_some("debug-decomp"),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join(",")
}
