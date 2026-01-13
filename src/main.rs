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
    load_and_preallocate("pool_stats.txt").expect("Failed to load stats");
    init_common();
    execute();
    save_access_stats("pool_stats.txt").expect("Failed to save stats");
}
