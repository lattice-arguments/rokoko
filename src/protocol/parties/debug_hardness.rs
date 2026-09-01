//! Per-round norm tracking and RSIS hardness estimation (`debug-hardness`).
//! The extracted witness norm is the worse of the rewinding bound and the JL
//! projection bound, as in the paper's extraction analysis.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{
    common::{
        config::MOD_Q,
        estimator::{estimate_rsis_security, RSISParameters},
        norms,
        ring_arithmetic::{Representation, RingElement},
        short_challenge::T_OP_NORM_BOUND,
    },
    protocol::{
        commitment::{RecursionConfig, RecursiveCommitmentWithAux},
        config::{IntermediateConfig, Projection, SimpleConfig, SumcheckConfig},
        params::NORM_MARGIN,
        sumchecks::helpers::plane_weight,
    },
};

/// Paper: alpha_rp = sqrt(30), the lower JL bound (Lemma "JL", kappa = 2^-128).
const JL_ALPHA_RP: f64 = 5.477225575051661;

/// Rewinding slack: factor 4 for the difference quotient in extraction,
/// factor 2 for ISIS-to-SIS.
const EXTRACTION_SLACK: f64 = 8.0;

/// The L_2 norm of the level's undecomposed input, recomposed out of its digit planes. This is
/// what the recomposition factor bounds, so printing it shows how much of that factor is slack.
fn recomposed_input_l2(rc: &RecursiveCommitmentWithAux, config: &RecursionConfig) -> f64 {
    let row_len = rc.committed_data.len() / config.padded_chunks();
    let mut recomposed = vec![RingElement::zero(Representation::IncompleteNTT); row_len];
    let mut term = RingElement::zero(Representation::IncompleteNTT);
    for plane in 0..config.decomposition_chunks {
        let weight = plane_weight(config.decomposition_base_log, plane);
        for element in 0..row_len {
            term.set_from(&rc.committed_data[plane * row_len + element]);
            term *= &weight;
            recomposed[element] += &term;
        }
    }

    let mut sum = 0f64;
    for element in recomposed.iter_mut() {
        element.from_incomplete_ntt_to_even_odd_coefficients();
        for &x in element.v.iter() {
            let centered = if x < MOD_Q / 2 { x } else { MOD_Q - x } as f64;
            sum += centered * centered;
        }
    }
    sum.sqrt()
}

/// What the projection norm claim measures: sqrt(SUM_j 2^{2 . base_log . j} . ||d_j||^2) over
/// the level's own digit planes, taken from the composed witness the round commits to.
fn weighted_plane_norm(composed_witness: &[RingElement], config: &RecursionConfig) -> f64 {
    let total_vars = composed_witness.len().ilog2() as usize;
    let placement = config.placement();
    let mut sum = 0f64;
    for plane in 0..config.decomposition_chunks {
        let prefix = placement.slice(plane, config.decomposition_chunks);
        let length = 1 << (total_vars - prefix.length);
        let start = prefix.prefix << (total_vars - prefix.length);
        let weight = 2f64.powi((config.decomposition_base_log * plane) as i32);
        let plane_norm = norms::l2_norm(&composed_witness[start..start + length]);
        sum += weight * weight * plane_norm * plane_norm;
    }
    sum.sqrt()
}

fn recomposition_factor(base_log: usize, chunks: usize) -> f64 {
    (0..chunks as i32)
        .map(|i| 2f64.powi(2 * i * base_log as i32))
        .sum::<f64>()
        .sqrt()
}

/// The length bound an estimate is asked for, as the estimator takes it. `as u64` turns a NaN
/// into zero, which the estimator reads as a bound nothing can meet and answers with security the
/// data does not have, so every bound crosses into it through here.
fn length_bound(norm: f64) -> u64 {
    assert!(
        norm.is_finite() && norm >= 0.0,
        "{norm} is not a length bound the estimator can be given"
    );
    norm.ceil() as u64
}

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
        length_bound: length_bound(current_extracted_norm),
    });
    let indent = "  ".repeat(depth);
    println!(
        "{}Recursive Commitment '{}' norms: L_2 = {}, bit_len = {}, MOD_Q = {} => estimated security for extraction: {:?} with rank {}",
        indent,
        name,
        ell_2_norm,
        ell_inf_norm.ilog2(),
        MOD_Q,
        hardness,
        config.rank,
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
    _next_level_width: usize, // this seems not needed, but we keep it for now to avoid changing the function signature
) {
    // we run in on reference run used as a avg case and we put some slack 
    if ROUND_ID.fetch_add(1, Ordering::Relaxed) < DEBUG_HARDNESS_FROM_ROUND {
        return;
    }

    println!("=== Debug Hardness Check ===");

    let recommited_ell_inf_norm = norms::inf_norm(next_round_data);
    let recommited_ell_2_norm = norms::l2_norm(next_round_data);

    // The squares are accumulated in f64: a squared L_2 norm of this data does not fit a u64.
    let most_inner_commitment_data_ell_2 = {
        let squared = |data: &[RingElement]| norms::l2_norm(data).powi(2);

        let commitment_data = &rc_commitment
            .most_inner_commitment_with_aux()
            .committed_data;
        let norm_commitment_data_ell_2_sq = squared(commitment_data);

        let opening_data = &rc_opening.most_inner_commitment_with_aux().committed_data;
        let norm_opening_data_ell_2_sq = squared(opening_data);

        let norm_projection_data_ell_2_sq = match (rc_coarse_projection, rc_fine_projection) {
            (Some(rc_proj), _) => squared(&rc_proj.most_inner_commitment_with_aux().committed_data),
            (_, Some((rc_ct, rc_batched))) => {
                squared(&rc_ct.most_inner_commitment_with_aux().committed_data)
                    + squared(&rc_batched.most_inner_commitment_with_aux().committed_data)
            }
            _ => 0.0,
        };
        (norm_commitment_data_ell_2_sq + norm_opening_data_ell_2_sq + norm_projection_data_ell_2_sq)
            .sqrt()
    };
    println!(
        "Most inner commitment data L_2 norm: {}",
        most_inner_commitment_data_ell_2
    );

    // the packed vector minus the most-inner commitment data: decomposed folded witness etc.
    // A non-finite difference would reach the estimator as `length_bound: 0` -- the `as u64` cast
    // of a NaN -- and be reported as security this data does not have, so it fails loudly here
    // instead; float slack alone is allowed to take the difference below zero.
    let rest_squared = recommited_ell_2_norm.powi(2) - most_inner_commitment_data_ell_2.powi(2);
    assert!(
        rest_squared.is_finite(),
        "norm accounting is not finite: packed vector {recommited_ell_2_norm}, \
         most inner commitment data {most_inner_commitment_data_ell_2}"
    );
    let recommited_ell_2_norm_rest = rest_squared.max(0.0).sqrt();

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
        * recomposition_factor(
            config.witness_decomposition_base_log,
            config.witness_decomposition_chunks,
        );


    let extracted_witness_bound = recomposed_witness_bound * T_OP_NORM_BOUND * EXTRACTION_SLACK;

    // Without the round's own projection claim the image is only covered by a containing
    // aggregate: the decomposed projection image is the outermost block of its own recursion and
    // so sits in the rest, unless a single-level recursion puts it among the most inner
    // commitment data.
    let projection_block_norm = |proj_config: &RecursionConfig| {
        if proj_config.next.is_some() {
            recommited_ell_2_norm_rest
        } else {
            most_inner_commitment_data_ell_2
        }
    };

    if let Some(proj_config) = config.projection_norm_scope() {
        let recomposed = match (rc_coarse_projection, rc_fine_projection) {
            (Some(rc_proj), _) => recomposed_input_l2(rc_proj, proj_config),
            (_, Some((rc_ct, _))) => recomposed_input_l2(rc_ct, proj_config),
            _ => 0.0,
        };
        let claim = weighted_plane_norm(next_round_data, proj_config);
        let factor = recomposition_factor(
            proj_config.decomposition_base_log,
            proj_config.decomposition_chunks,
        );
        println!(
            "Projection image L_2 {recomposed}: claimed bound {}, aggregate bound {}",
            claim * (proj_config.decomposition_chunks as f64).sqrt(),
            recommited_ell_2_norm_rest * factor
        );
    }

    // The claim carries SUM_j 2^{2 . base_log . j} . ||d_j||^2; Cauchy-Schwarz over the
    // `chunks` planes turns it into a bound on the recomposed image. Lacking that per-plane
    // split, the aggregate path pays the full recomposition factor.
    let recomposed_projection_bound = match config.projection_norm_scope() {
        Some(proj_config) => {
            weighted_plane_norm(next_round_data, proj_config)
                * (proj_config.decomposition_chunks as f64).sqrt()
        }
        None => match &config.projection_recursion {
            Projection::Coarse(proj_config) => {
                projection_block_norm(proj_config)
                    * recomposition_factor(
                        proj_config.decomposition_base_log,
                        proj_config.decomposition_chunks,
                    )
            }
            Projection::Fine(proj_config) => {
                let constant_term = &proj_config.recursion_constant_term;
                projection_block_norm(constant_term)
                    * recomposition_factor(
                        constant_term.decomposition_base_log,
                        constant_term.decomposition_chunks,
                    )
            }
            Projection::Skip => 0.0, // not used
        },
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

    // we have a joint boudn on random projection from the next round extraction (i.e. not per column) 
    // that is extracted_witness_bound which argues about the joint norm of the witness and one column must be less that that.

    match &config.projection_recursion {
        Projection::Skip => {
            // no projection: inner-product norm extraction is not available anyway
        }
        _ => {
            let bound = argued_witness_bound * argued_witness_bound * NORM_MARGIN * NORM_MARGIN;
            assert!(
                  bound < (MOD_Q as f64 / 2f64),
                "Witness bound too large for inner-product norm extraction! {} * {}^2 * {} = {} >= {} / 2 = {}, ratio: {}",
                argued_witness_bound,
                argued_witness_bound,
                NORM_MARGIN,
                bound,
                MOD_Q,
                MOD_Q as f64 / 2f64,
                bound / (MOD_Q as f64 / 2f64)
            );
        }
    }

    let basic_commitment_security = estimate_rsis_security(&RSISParameters {
        m: config.witness_height as u64,
        n: config.basic_commitment_rank as u64,
        length_bound: length_bound(worse_bound),
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
    println!("Debug hardness check for intermediate round is broken, but we don't use it! Fix it if you need it.");
    std::process::exit(1);
//     println!("=== Debug Hardness Check for Intermediate Round ===");

//     let recommited_ell_2_norm = norms::l2_norm(next_round_witness_data);
//     let recommited_ell_inf_norm = norms::inf_norm(next_round_witness_data);
//     println!(
//         "Next round witness norms: L_2 = {}, L_inf = {}, bit_len = {}, MOD_Q = {}",
//         recommited_ell_2_norm,
//         recommited_ell_inf_norm,
//         recommited_ell_inf_norm.ilog2(),
//         MOD_Q
//     );

//     let folded_witness_ell_2_norm = norms::l2_norm(folded_witness_data);
//     let folded_witness_inf_norm = norms::inf_norm(folded_witness_data);
//     println!(
//         "Folded witness norms: L_2 = {}, L_inf = {}, bit_len = {}, MOD_Q = {}",
//         folded_witness_ell_2_norm,
//         folded_witness_inf_norm,
//         folded_witness_inf_norm.ilog2(),
//         MOD_Q
//     );

//     let recomposed_witness_bound = recommited_ell_2_norm
//         * (config
//             .witness_decomposition_base_log
//             .pow((config.witness_decomposition_chunks - 1) as u32)) as f64;

//     println!("Folded witness norm: {}", recomposed_witness_bound);

//     let projection_l2_norm = norms::l2_norm_coeffs(projection_image_ct_data);

//     let extracted_witness_bound = recomposed_witness_bound * T_OP_NORM_BOUND * EXTRACTION_SLACK;

//     let argued_witness_bound = projection_l2_norm / JL_ALPHA_RP;

//     assert!(
//         argued_witness_bound * argued_witness_bound < (MOD_Q as f64 / 2f64),
//         "Projection-argued witness bound too large for inner-product norm extraction!"
//     );

//     let worse_bound = if extracted_witness_bound > argued_witness_bound {
//         println!(
//             "Using extracted witness bound {} for security estimation.",
//             extracted_witness_bound
//         );
//         extracted_witness_bound
//     } else {
//         println!(
//             "Using projection-argued witness bound {} for security estimation.",
//             argued_witness_bound
//         );
//         argued_witness_bound
//     };

//     let basic_commitment_security = estimate_rsis_security(&RSISParameters {
//         m: config.witness_height as u64,
//         n: config.basic_commitment_rank as u64,
//         length_bound: worse_bound.ceil() as u64,
//     });
//     println!(
//         "Basic commitment estimated security for extraction: {:?} with rank {}",
//         basic_commitment_security, config.basic_commitment_rank
//     );
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
        length_bound: length_bound(worse_bound),
    });
    println!(
        "Basic commitment estimated security for extraction: {:?} with rank {}",
        basic_commitment_security, config.basic_commitment_rank
    );
}
