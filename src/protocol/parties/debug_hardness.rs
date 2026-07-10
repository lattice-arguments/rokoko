//! Per-round norm tracking and RSIS hardness estimation (`debug-hardness`).
//! The extracted witness norm is the worse of the rewinding bound and the JL
//! projection bound, as in the paper's extraction analysis.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{
    common::{
        config::MOD_Q,
        estimator::{estimate_rsis_security, RSISParameters},
        norms,
        ring_arithmetic::RingElement,
        short_challenge::T_OP_NORM_BOUND,
    },
    protocol::{
        commitment::{RecursionConfig, RecursiveCommitmentWithAux},
        config::{IntermediateConfig, Projection, SimpleConfig, SumcheckConfig},
    },
};

/// Paper: alpha_rp = sqrt(30), the lower JL bound (Lemma "JL", kappa = 2^-128).
const JL_ALPHA_RP: f64 = 5.477225575051661;

/// Rewinding slack: factor 4 for the difference quotient in extraction,
/// factor 2 for ISIS-to-SIS.
const EXTRACTION_SLACK: f64 = 8.0;

static ROUND_ID: AtomicUsize = AtomicUsize::new(0);
static DEBUG_HARDNESS_FROM_ROUND: usize = 0;

fn check_recursive_commitment(
    rc: &RecursiveCommitmentWithAux,
    config: &RecursionConfig,
    name: &str,
    extracted_norm: f64,
    extracted_norm_most_inner: f64,
    depth: usize,
) {
    let ell_inf_norm = norms::inf_norm(&rc.committed_data);
    let ell_2_norm = norms::l2_norm(&rc.committed_data);

    let current_extracted_norm = match config.next {
        Some(_) => extracted_norm,
        None => extracted_norm_most_inner,
    };

    let hardness = estimate_rsis_security(&RSISParameters {
        m: rc.committed_data.len() as u64,
        n: config.rank as u64,
        length_bound: current_extracted_norm.ceil() as u64,
    });
    let indent = "  ".repeat(depth);
    println!(
        "{}Recursive Commitment '{}' norms: L_2 = {}, bit_len = {}, MOD_Q = {} => estimated security for extraction: {:?}",
        indent,
        name,
        ell_2_norm,
        ell_inf_norm.ilog2(),
        MOD_Q,
        hardness,
    );

    if let (Some(next_rc), Some(next_config)) = (&rc.next, &config.next) {
        check_recursive_commitment(
            next_rc,
            next_config,
            name,
            extracted_norm,
            extracted_norm_most_inner,
            depth + 1,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn check_sumcheck_round(
    config: &SumcheckConfig,
    next_round_data: &[RingElement],
    rc_commitment: &RecursiveCommitmentWithAux,
    rc_opening: &RecursiveCommitmentWithAux,
    rc_coarse_projection: Option<&RecursiveCommitmentWithAux>,
    rc_fine_projection: Option<(&RecursiveCommitmentWithAux, &RecursiveCommitmentWithAux)>,
    next_level_width: usize,
) {
    if ROUND_ID.fetch_add(1, Ordering::Relaxed) < DEBUG_HARDNESS_FROM_ROUND {
        return;
    }

    println!("=== Debug Hardness Check ===");

    let recommited_ell_inf_norm = norms::inf_norm(next_round_data);
    let recommited_ell_2_norm = norms::l2_norm(next_round_data);

    let most_inner_commitment_data_ell_2 = {
        let commitment_data = &rc_commitment
            .most_inner_commitment_with_aux()
            .committed_data;
        let norm_commitment_data_ell_2_sq = norms::l2_norm(commitment_data).powf(2.0) as u64;

        let opening_data = &rc_opening.most_inner_commitment_with_aux().committed_data;
        let norm_opening_data_ell_2_sq = norms::l2_norm(opening_data).powf(2.0) as u64;

        let norm_projection_data_ell_2_sq = match (rc_coarse_projection, rc_fine_projection) {
            (Some(rc_proj), _) => {
                let proj_data = &rc_proj.most_inner_commitment_with_aux().committed_data;
                norms::l2_norm(proj_data).powf(2.0) as u64
            }
            (_, Some((rc_ct, rc_batched))) => {
                let proj_ct_data = &rc_ct.most_inner_commitment_with_aux().committed_data;
                let proj_batched_data = &rc_batched.most_inner_commitment_with_aux().committed_data;
                norms::l2_norm(proj_ct_data).powf(2.0) as u64
                    + norms::l2_norm(proj_batched_data).powf(2.0) as u64
            }
            _ => 0,
        };
        ((norm_commitment_data_ell_2_sq + norm_opening_data_ell_2_sq + norm_projection_data_ell_2_sq)
            as f64)
            .sqrt()
    };
    println!(
        "Most inner commitment data L_2 norm: {}",
        most_inner_commitment_data_ell_2
    );

    // the packed vector minus the most-inner commitment data: decomposed folded witness etc.
    let recommited_ell_2_norm_rest =
        (recommited_ell_2_norm.powf(2.0) - most_inner_commitment_data_ell_2.powf(2.0)).sqrt();

    check_recursive_commitment(
        rc_commitment,
        &config.commitment_recursion,
        "Commitment",
        recommited_ell_2_norm_rest,
        most_inner_commitment_data_ell_2,
        0,
    );

    check_recursive_commitment(
        rc_opening,
        &config.opening_recursion,
        "Opening",
        recommited_ell_2_norm_rest,
        most_inner_commitment_data_ell_2,
        0,
    );

    if let (Some(rc_projection), Projection::Coarse(projection_config)) =
        (rc_coarse_projection, &config.projection_recursion)
    {
        check_recursive_commitment(
            rc_projection,
            projection_config,
            "Projection Image",
            recommited_ell_2_norm_rest,
            most_inner_commitment_data_ell_2,
            0,
        );
    }

    if let (Some((rc_ct, rc_batched)), Projection::Fine(projection_config)) =
        (rc_fine_projection, &config.projection_recursion)
    {
        check_recursive_commitment(
            rc_ct,
            &projection_config.recursion_constant_term,
            "Fine Projection Constant Term",
            recommited_ell_2_norm_rest,
            most_inner_commitment_data_ell_2,
            0,
        );
        check_recursive_commitment(
            rc_batched,
            &projection_config.recursion_batched_projection,
            "Fine Projection Batched",
            recommited_ell_2_norm_rest,
            most_inner_commitment_data_ell_2,
            0,
        );
    }
    println!(
        "Next round data norms: L_inf = {}, bit_len = {}, L_2 = {}, MOD_Q = {}",
        recommited_ell_inf_norm,
        recommited_ell_inf_norm.ilog2(),
        recommited_ell_2_norm,
        MOD_Q
    );

    let recomposed_witness_bound = recommited_ell_2_norm_rest
        * (config
            .witness_decomposition_base_log
            .pow((config.witness_decomposition_chunks - 1) as u32)) as f64;

    let extracted_witness_bound = recomposed_witness_bound * T_OP_NORM_BOUND * EXTRACTION_SLACK;

    let recomposed_projection_bound = match &config.projection_recursion {
        Projection::Coarse(proj_config) => {
            // full norm, not the rest: projection may live in the most inner commitment
            recommited_ell_2_norm
                * (proj_config
                    .decomposition_base_log
                    .pow((proj_config.decomposition_chunks - 1) as u32)) as f64
        }
        Projection::Fine(proj_config) => {
            recommited_ell_2_norm
                * (proj_config
                    .recursion_constant_term
                    .decomposition_base_log
                    .pow((proj_config.recursion_constant_term.decomposition_chunks - 1) as u32))
                    as f64
        }
        Projection::Skip => 0.0, // not used
    };

    let argued_witness_bound = recomposed_projection_bound / JL_ALPHA_RP;

    let worse_bound = if extracted_witness_bound > argued_witness_bound {
        println!(
            "Using extracted witness bound {} for security estimation.",
            extracted_witness_bound
        );
        extracted_witness_bound
    } else {
        println!(
            "Using projection-argued witness bound {} for security estimation.",
            argued_witness_bound
        );
        argued_witness_bound
    };

    match &config.projection_recursion {
        Projection::Skip => {
            // no projection: inner-product norm extraction is not available anyway
        }
        _ => {
            assert!(
                next_level_width as f64 * argued_witness_bound * argued_witness_bound
                    < (MOD_Q as f64 / 2f64),
                "Witness bound too large for inner-product norm extraction!"
            );
        }
    }

    let basic_commitment_security = estimate_rsis_security(&RSISParameters {
        m: config.witness_height as u64,
        n: config.basic_commitment_rank as u64,
        length_bound: worse_bound.ceil() as u64,
    });
    println!(
        "Basic commitment estimated security for extraction: {:?} with rank {}",
        basic_commitment_security, config.basic_commitment_rank
    );
}

pub fn check_intermediate_round(
    config: &IntermediateConfig,
    next_round_witness_data: &[RingElement],
    folded_witness_data: &[RingElement],
    projection_image_ct_data: &[RingElement],
) {
    println!("=== Debug Hardness Check for Intermediate Round ===");

    let recommited_ell_2_norm = norms::l2_norm(next_round_witness_data);
    let recommited_ell_inf_norm = norms::inf_norm(next_round_witness_data);
    println!(
        "Next round witness norms: L_2 = {}, L_inf = {}, bit_len = {}, MOD_Q = {}",
        recommited_ell_2_norm,
        recommited_ell_inf_norm,
        recommited_ell_inf_norm.ilog2(),
        MOD_Q
    );

    let folded_witness_ell_2_norm = norms::l2_norm(folded_witness_data);
    let folded_witness_inf_norm = norms::inf_norm(folded_witness_data);
    println!(
        "Folded witness norms: L_2 = {}, L_inf = {}, bit_len = {}, MOD_Q = {}",
        folded_witness_ell_2_norm,
        folded_witness_inf_norm,
        folded_witness_inf_norm.ilog2(),
        MOD_Q
    );

    let recomposed_witness_bound = recommited_ell_2_norm
        * (config
            .witness_decomposition_base_log
            .pow((config.witness_decomposition_chunks - 1) as u32)) as f64;

    println!("Folded witness norm: {}", recomposed_witness_bound);

    let projection_l2_norm = norms::l2_norm_coeffs(projection_image_ct_data);

    let extracted_witness_bound = recomposed_witness_bound * T_OP_NORM_BOUND * EXTRACTION_SLACK;

    let argued_witness_bound = projection_l2_norm / JL_ALPHA_RP;

    assert!(
        argued_witness_bound * argued_witness_bound < (MOD_Q as f64 / 2f64),
        "Projection-argued witness bound too large for inner-product norm extraction!"
    );

    let worse_bound = if extracted_witness_bound > argued_witness_bound {
        println!(
            "Using extracted witness bound {} for security estimation.",
            extracted_witness_bound
        );
        extracted_witness_bound
    } else {
        println!(
            "Using projection-argued witness bound {} for security estimation.",
            argued_witness_bound
        );
        argued_witness_bound
    };

    let basic_commitment_security = estimate_rsis_security(&RSISParameters {
        m: config.witness_height as u64,
        n: config.basic_commitment_rank as u64,
        length_bound: worse_bound.ceil() as u64,
    });
    println!(
        "Basic commitment estimated security for extraction: {:?} with rank {}",
        basic_commitment_security, config.basic_commitment_rank
    );
}

pub fn check_simple_round(
    config: &SimpleConfig,
    folded_witness_data: &[RingElement],
    projection_image_ct_data: &[RingElement],
) {
    println!("=== Debug Hardness Check for Simple Round ===");

    let folded_witness_l2_norm = norms::l2_norm(folded_witness_data);
    println!("Folded witness norm: {}", folded_witness_l2_norm);

    let projection_l2_norm = norms::l2_norm_coeffs(projection_image_ct_data);

    let extracted_witness_bound = folded_witness_l2_norm * T_OP_NORM_BOUND * EXTRACTION_SLACK;

    let argued_witness_bound = projection_l2_norm / JL_ALPHA_RP;
    let worse_bound = if extracted_witness_bound > argued_witness_bound {
        println!(
            "Using extracted witness bound {} for security estimation.",
            extracted_witness_bound
        );
        extracted_witness_bound
    } else {
        println!(
            "Using projection-argued witness bound {} for security estimation.",
            argued_witness_bound
        );
        argued_witness_bound
    };

    let basic_commitment_security = estimate_rsis_security(&RSISParameters {
        m: config.witness_height as u64,
        n: config.basic_commitment_rank as u64,
        length_bound: worse_bound.ceil() as u64,
    });
    println!(
        "Basic commitment estimated security for extraction: {:?} with rank {}",
        basic_commitment_security, config.basic_commitment_rank
    );
}
