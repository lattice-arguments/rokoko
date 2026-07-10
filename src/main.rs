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

    let _tracing_guards = rokoko::tracing::setup();

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

    drop(_tracing_guards);
    #[cfg(feature = "profile")]
    {
        let trace_base = format!(
            "{}_{}",
            rokoko::tracing::trace_name(),
            rokoko::tracing::timestamp_for_filename()
        );
        rokoko::tracing::print_artifact_paths(&trace_base);
    }
}
