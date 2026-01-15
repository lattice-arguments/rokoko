use cowboys_and_aliens::common::config::HALF_DEGREE;
use cowboys_and_aliens::common::init_common;
use cowboys_and_aliens::common::pool::{load_and_preallocate, save_access_stats};
use cowboys_and_aliens::common::ring_arithmetic::*;
use cowboys_and_aliens::protocol::execution;
use cowboys_and_aliens::protocol::execution::execute;

fn main() {
    #[cfg(feature = "unsafe-sumcheck")]
    {
        println!("Sumcheck unsafe...");
    }
    
    // Check AVX-512F support
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            println!("✓ AVX-512F is enabled and available");
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
    execute();
    save_access_stats("pool_stats.txt").expect("Failed to save stats");
}
