use cowboys_and_aliens::common::init_common;
use cowboys_and_aliens::common::pool::{load_and_preallocate, save_access_stats};
use cowboys_and_aliens::protocol::parties::executor::execute;

mod hexl_benches;

fn main() {
    // hexl_benches::run_hexl_benches();
    // return;
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
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("✗ AVX-512F is only available on x86_64 architecture");
    }

    load_and_preallocate("pool_stats.txt").expect("Failed to load stats");
    init_common();
    println!("Running executor...");
    execute();
    save_access_stats("pool_stats.txt").expect("Failed to save stats");
}
