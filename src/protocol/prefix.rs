use crate::protocol::commitment::RecursionConfig;
use crate::protocol::config::{Config, Projection};

pub fn is_prefix(a_prefix: usize, a_len: usize, b_prefix: usize, b_len: usize) -> bool {
    if a_len == 0 {
        return true;
    }
    if a_len > b_len {
        return false;
    }
    let shift = b_len - a_len;
    let mask = (1usize << a_len) - 1;
    (a_prefix & mask) == ((b_prefix >> shift) & mask)
}

#[inline]
fn push_recursion_prefixes(recursion: &RecursionConfig, prefixes: &mut Vec<(usize, usize)>) {
    let mut current = recursion;
    loop {
        prefixes.push((current.prefix.prefix, current.prefix.length));
        match &current.next {
            Some(next) => current = next,
            None => break,
        }
    }
}

pub fn check_prefixing_correctness(config: &Config) {
    let mut prefixes = Vec::with_capacity(16);
    let mut current_config = config;

    loop {
        prefixes.push((
            current_config.folded_witness_prefix.prefix,
            current_config.folded_witness_prefix.length,
        ));
        push_recursion_prefixes(&current_config.commitment_recursion, &mut prefixes);
        push_recursion_prefixes(&current_config.opening_recursion, &mut prefixes);
        match &current_config.projection_recursion {
            Projection::Type0(proj_config) => {
                push_recursion_prefixes(&proj_config, &mut prefixes);
            }
            Projection::Type1(proj_config) => {
                push_recursion_prefixes(&proj_config.recursion_constant_term, &mut prefixes);
                push_recursion_prefixes(&proj_config.recursion_batched_projection, &mut prefixes);
            }
        }

        match &current_config.next {
            Some(next) => current_config = next,
            None => break,
        }
    }

    for i in 0..prefixes.len() {
        for j in i + 1..prefixes.len() {
            let (a_prefix, a_len) = prefixes[i];
            let (b_prefix, b_len) = prefixes[j];

            if is_prefix(a_prefix, a_len, b_prefix, b_len)
                || is_prefix(b_prefix, b_len, a_prefix, a_len)
            {
                panic!(
                    "Config prefixes conflict: ({}, {}) vs ({}, {})",
                    a_prefix, a_len, b_prefix, b_len
                );
            }
        }
    }

    // TODO: print some stuff about e.g. how many slots are free (i.e. uncovered by any prefix)

    if config.next.is_some() {
        check_prefixing_correctness(config.next.as_ref().unwrap());
    }
}

#[cfg(test)]
mod tests {
    use super::is_prefix;

    #[test]
    fn detects_prefix_relationships() {
        assert!(is_prefix(0b110, 3, 0b1100, 4));
        assert!(is_prefix(0, 0, 0b111, 3));
        assert!(is_prefix(0b1, 1, 0b1, 1));
        assert!(is_prefix(0b101, 3, 0b101, 3));
        assert!(!is_prefix(0b111, 3, 0b110, 3));
        assert!(!is_prefix(0b1101, 4, 0b110, 3));
        assert!(!is_prefix(0b10, 2, 0b1100, 4));
    }
}
