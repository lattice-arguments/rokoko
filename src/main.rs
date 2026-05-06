use rokoko::common::init_common;
use rokoko::common::pool::{load_and_preallocate, save_access_stats};
use rokoko::protocol::parties::executor::execute;
// use rokoko::protocol::execution::execute;

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

    #[cfg(feature = "profile")]
    let _tracing_guards = rokoko_profiling::setup_tracing(
        &[
            rokoko_profiling::TracingFormat::Default,
            rokoko_profiling::TracingFormat::Chrome,
            rokoko_profiling::TracingFormat::Snapshot,
        ],
        trace_name(),
        &active_features(),
    );

    load_and_preallocate("pool_stats.txt").expect("Failed to load stats");
    init_common();
    println!("Running executor...");
    execute();
    save_access_stats("pool_stats.txt").expect("Failed to save stats");
}

#[cfg(feature = "profile")]
fn trace_name() -> &'static str {
    if cfg!(feature = "p-26") {
        "p26"
    } else if cfg!(feature = "p-30") {
        "p30"
    } else {
        "p28"
    }
}

#[cfg(feature = "profile")]
fn active_features() -> String {
    let mut features = Vec::new();
    if cfg!(feature = "p-26") {
        features.push("p-26");
    }
    if cfg!(feature = "p-28") {
        features.push("p-28");
    }
    if cfg!(feature = "p-30") {
        features.push("p-30");
    }
    if cfg!(feature = "incomplete-rexl") {
        features.push("incomplete-rexl");
    }
    if cfg!(feature = "unsafe-sumcheck") {
        features.push("unsafe-sumcheck");
    }
    if cfg!(feature = "debug-hardness") {
        features.push("debug-hardness");
    }
    if cfg!(feature = "debug-decomp") {
        features.push("debug-decomp");
    }
    features.join(",")
}
