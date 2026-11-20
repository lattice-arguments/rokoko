//! The CRS a round chain needs: `CRS::gen_prover_crs` expands the key rows the chain reads,
//! and `CRS::gen_verifier_crs` keeps them structured except for the simple round's key.

use crate::{
    commitment::{
        crs::{gen_structured_cks, VerifierCRS, CRS},
        RecursionConfig,
    },
    common::structured_row::PreprocessedRow,
    rok::config::{Config, Projection, SimpleConfig, SumcheckConfig},
};

fn simple_round(config: &SumcheckConfig) -> Option<&SimpleConfig> {
    let mut next = config.next.as_deref();
    while let Some(round) = next {
        next = match round {
            Config::Sumcheck(c) => c.next.as_deref(),
            Config::Intermediate(c) => c.next.as_deref(),
            Config::Simple(c) => return Some(c),
        };
    }
    None
}

fn key_rows(rows: &mut [usize], wit_dim: usize, rank: usize) {
    let dim = &mut rows[wit_dim.ilog2() as usize - 1];
    *dim = (*dim).max(rank);
}

fn rows_of_tree(rows: &mut [usize], tree: &RecursionConfig) {
    let mut level = Some(tree);
    while let Some(t) = level {
        key_rows(rows, t.committed_len(), t.rank);
        level = t.next.as_deref();
    }
}

fn rows_of_round(rows: &mut [usize], config: &SumcheckConfig) {
    key_rows(rows, config.witness_height, config.basic_commitment_rank);
    rows_of_tree(rows, &config.commitment_recursion);
    rows_of_tree(rows, &config.opening_recursion);
    match &config.projection_recursion {
        Projection::Coarse(tree) => rows_of_tree(rows, tree),
        Projection::Fine(fine) => {
            rows_of_tree(rows, &fine.recursion_constant_term);
            rows_of_tree(rows, &fine.recursion_batched_projection);
        }
        Projection::Skip => {}
    }
    let mut next = config.next.as_deref();
    while let Some(round) = next {
        next = match round {
            Config::Sumcheck(c) => return rows_of_round(rows, c),
            Config::Intermediate(c) => {
                key_rows(rows, c.witness_height, c.basic_commitment_rank);
                c.next.as_deref()
            }
            Config::Simple(c) => {
                key_rows(rows, c.witness_height, c.basic_commitment_rank);
                None
            }
        };
    }
}

fn key_rows_of(config: &SumcheckConfig) -> Vec<usize> {
    let mut rows = vec![0; config.composed_witness_length.ilog2() as usize];
    rows_of_round(&mut rows, config);
    rows
}

impl CRS {
    pub fn gen_prover_crs(config: &SumcheckConfig) -> CRS {
        let rows = key_rows_of(config);
        let structured_cks = gen_structured_cks(
            config.composed_witness_length,
            rows.iter().copied().max().expect("a chain reads at least one key row"),
        );
        let cks = structured_cks
            .iter()
            .zip(&rows)
            .map(|(sck, &n)| {
                sck[..n]
                    .iter()
                    .map(PreprocessedRow::from_structured_row)
                    .collect()
            })
            .collect();
        CRS {
            cks,
            structured_cks,
        }
    }

    pub fn gen_verifier_crs(config: &SumcheckConfig) -> VerifierCRS {
        let mut crs = VerifierCRS {
            structured_cks: gen_structured_cks(
                config.composed_witness_length,
                key_rows_of(config).iter().copied().max().expect("a chain reads at least one key row"),
            ),
            simple_ck: Vec::new(),
        };
        if let Some(simple) = simple_round(config) {
            crs.simple_ck = crs
                .structured_ck_for_wit_dim(simple.witness_height)
                .iter()
                .take(simple.basic_commitment_rank)
                .map(PreprocessedRow::from_structured_row)
                .collect();
        }
        crs
    }
}
