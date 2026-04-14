use crate::{
    common::config::*,
    protocol::{
        commitment::Prefix,
        config::{RoundConfig, RoundConfigCommon},
    },
};

/// Recursively builds the round config chain.
/// - First round: uses NUM_COLUMNS_INITIAL columns, projection_ratio=2, VDF+exact_binariness enabled.
/// - Subsequent rounds: 8 columns, projection_ratio=8, L2 enabled.
/// - Recursion stops when the *next* round's single_col_height would be < PROJECTION_HEIGHT * projection_ratio
///   (i.e., the next round couldn't support projection).
pub fn build_round_config(extended_witness_length: usize, is_first_round: bool) -> RoundConfig {
    let main_witness_columns = if is_first_round {
        NUM_COLUMNS_INITIAL
    } else {
        8
    };

    // only for structured case
    let projection_ratio = if is_first_round { 2 } else { 8 };

    let single_col_height = extended_witness_length / 2 / main_witness_columns;
    // After fold+split+decompose, next round's column height = single_col_height / 2
    let next_single_col_height = single_col_height / 2;
    let next_main_witness_columns = 8usize;
    let next_projection_ratio = 8usize;
    let can_recurse = next_single_col_height >= PROJECTION_HEIGHT * next_projection_ratio;
    println!("Building round config: extended_witness_length={}, single_col_height={}, next_single_col_height={}, can_recurse={}", extended_witness_length, single_col_height, next_single_col_height, can_recurse);

    let inner_evaluation_claims = if is_first_round { 0 } else { 2 };

    let common = RoundConfigCommon {
        extended_witness_length,
        exact_binariness: is_first_round,
        l2: !is_first_round,
        vdf: is_first_round,
        inner_evaluation_claims,
        main_witness_columns,
        main_witness_prefix: Prefix {
            prefix: 0b0,
            length: 1,
        },
    };

    if can_recurse {
        let next_extended_witness_length = next_single_col_height * next_main_witness_columns * 2;
        RoundConfig::Intermediate {
            common,
            decomposition_base_log: 8,
            projection_ratio,
            projection_prefix: Prefix {
                prefix: main_witness_columns,
                length: main_witness_columns.ilog2() as usize + 1,
            },
            next: Box::new(build_round_config(next_extended_witness_length, false)),
        }
    } else {
        // Transition to unstructured rounds (no projection).
        // The first unstructured round has 8 input columns (from
        // the Intermediate decomposition). With prefix=0, extended_witness_length
        // does not include the factor-of-2 doubling.
        let unstructured_cols = 8usize; // first unstructured inherits 8 cols from Intermediate output
        let unstructured_extended_witness_length = next_single_col_height * unstructured_cols;
        let unstructured_single_col_height = next_single_col_height;
        let next_unstructured_height = unstructured_single_col_height / 2;
        let next_unstructured_cols = 4usize;
        let next_unstructured_wl = next_unstructured_height * next_unstructured_cols;

        let unstructured_common = RoundConfigCommon {
            extended_witness_length: unstructured_extended_witness_length,
            exact_binariness: false,
            l2: true,
            vdf: false,
            inner_evaluation_claims: 2,
            main_witness_columns: unstructured_cols,
            main_witness_prefix: Prefix {
                prefix: 0,
                length: 0,
            },
        };

        println!(
            "Building unstructured round config: extended_witness_length={}, single_col_height={}, next_height={}",
            unstructured_extended_witness_length, unstructured_single_col_height, next_unstructured_height
        );

        let next_unstructured_config = if next_unstructured_height >= PROJECTION_HEIGHT {
            build_unstructured_round_config(next_unstructured_wl)
        } else {
            RoundConfig::Last {
                common: RoundConfigCommon {
                    extended_witness_length: next_unstructured_wl,
                    exact_binariness: false,
                    l2: true,
                    vdf: false,
                    inner_evaluation_claims: 2,
                    main_witness_columns: next_unstructured_cols,
                    main_witness_prefix: Prefix {
                        prefix: 0,
                        length: 0,
                    },
                },
                projection_ratio: std::cmp::min(
                    DEGREE * next_unstructured_height / PROJECTION_HEIGHT,
                    MAX_UNSTRUCT_PROJ_RATIO,
                ),
            }
        };

        let next_config = RoundConfig::IntermediateUnstructured {
            common: unstructured_common,
            decomposition_base_log: 8,
            projection_ratio: std::cmp::min(
                DEGREE * unstructured_single_col_height / PROJECTION_HEIGHT,
                MAX_UNSTRUCT_PROJ_RATIO,
            ),
            next: Box::new(next_unstructured_config),
        };

        RoundConfig::Intermediate {
            common,
            decomposition_base_log: 8,
            projection_ratio,
            projection_prefix: Prefix {
                prefix: main_witness_columns,
                length: main_witness_columns.ilog2() as usize + 1,
            },
            next: Box::new(next_config),
        }
    }
}

/// Builds unstructured round configs (4 columns, prefix 0, unstructured projection).
/// Continues until single_col_height / 2 < PROJECTION_HEIGHT, then produces Last.
pub fn build_unstructured_round_config(extended_witness_length: usize) -> RoundConfig {
    let main_witness_columns = 4usize;
    let single_col_height = extended_witness_length / main_witness_columns;
    let next_single_col_height = single_col_height / 2;
    let next_cols = 4usize;
    let next_wl = next_single_col_height * next_cols;

    println!(
        "Building unstructured round config: extended_witness_length={}, single_col_height={}, next_height={}",
        extended_witness_length, single_col_height, next_single_col_height
    );

    let common = RoundConfigCommon {
        extended_witness_length,
        exact_binariness: false,
        l2: true,
        vdf: false,
        inner_evaluation_claims: 2,
        main_witness_columns,
        main_witness_prefix: Prefix {
            prefix: 0,
            length: 0,
        },
    };

    let next_config = if next_single_col_height >= LAST_ROUND_THRESHOLD {
        build_unstructured_round_config(next_wl)
    } else {
        RoundConfig::Last {
            common: RoundConfigCommon {
                extended_witness_length: next_wl,
                exact_binariness: false,
                l2: true,
                vdf: false,
                inner_evaluation_claims: 2,
                main_witness_columns: next_cols,
                main_witness_prefix: Prefix {
                    prefix: 0,
                    length: 0,
                },
            },
            projection_ratio: std::cmp::min(
                DEGREE * next_single_col_height / PROJECTION_HEIGHT,
                MAX_UNSTRUCT_PROJ_RATIO,
            ),
        }
    };

    RoundConfig::IntermediateUnstructured {
        common,
        decomposition_base_log: 8,
        projection_ratio: std::cmp::min(
            DEGREE * single_col_height / PROJECTION_HEIGHT,
            MAX_UNSTRUCT_PROJ_RATIO,
        ), // for now, we assume that each column is projected to PROJECTION_HEIGHT Zq elements.
        next: Box::new(next_config),
    }
}
