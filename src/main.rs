use cowboys_and_aliens::common::config::HALF_DEGREE;
use cowboys_and_aliens::common::init_common;
use cowboys_and_aliens::common::ring_arithmetic::*;
use cowboys_and_aliens::protocol::execution;
use cowboys_and_aliens::protocol::execution::execute;
use std::sync::LazyLock;

fn main() {
    init_common();
    execute();
}
