//! Ajtai-hash Merkle tree over a committed trace, proven with one sumcheck
//! claim per layer.
//!
//! The data is a vector of `d` ring elements, represented in the witness only
//! by its `4d` balanced base-2^13 digits. The hash is `H(x) = A x` for a
//! public random `A` in `R_q^{2x32}`: it takes 32 short elements to 2 full
//! ones, whose `2 x 4 = 8` digits join three sibling outputs to form the next
//! layer's 32-element preimage, an arity-4 tree ending in a public root.
//! Every layer is one zero-valued claim batching all its node equations
//! `A x_m = sum_j 2^{13j} y_{m,j}` under an eq-tensor over the node index:
//! the input side pairs the layer segment with a public whose closed form is
//! `eq(alpha, m) * K_i` for the 32 fixed ring elements `K_i = sum_r tau_r
//! A[r][i]`, and the recomposition side is a single field tensor over the
//! next layer's segment, sharing the node tensor's layers. Only the root
//! claim carries a value, the public root. `log_4` many claims, one chain
//! opening per layer.
//!
//! # Soundness, roughly
//!
//! The argument certifies one aggregate l2 bound on the whole committed
//! trace, carrying the chain's extraction slack. Binding therefore reduces
//! to module-SIS for `A` at twice that aggregate bound: a forged tree
//! yields two distinct short preimages of one node image, each a priori
//! spending the full trace budget. At the p-28 shape the honest trace has
//! l2 near 2^26, and the lattice estimator puts SIS at rank 2,
//! `q = 2^50 - 2687`, `beta = 2^27` near 25 bits (classical Core-SVP): the
//! 32 -> 2 hash exercises the claim machinery and is not a binding tree at
//! these parameters. Rank 5 at the same norm estimates near 131 bits;
//! certifying per-node rather than aggregate norms (`beta ~ 2^19`) would
//! put rank 2 near 86 bits. Both are parameter changes orthogonal to the
//! claim structure shown here.

use std::sync::Arc;

use rokoko::common::arithmetic::pow_mod;
use rokoko::common::config::{HALF_DEGREE, MOD_Q as Q};
use rokoko::common::decomposition::decompose;
use rokoko::common::hash::HashWrapper;
use rokoko::common::matrix::VerticallyAlignedMatrix;
use rokoko::common::ring_arithmetic::{QuadraticExtension, Representation, RingElement};
use rokoko::common::sumcheck_element::SumcheckElement;
use rokoko::protocol::commitment::Prefix;
use rokoko::protocol::config::Config;
use rokoko::protocol::crs::CRS;
use rokoko::protocol::parties::{commiter::commit, prover::prover_round, verifier::verifier_round};
use rokoko::protocol::snark::{
    embed_qe, eq_layers_qe, inv_pow2_q, prove_initial_claims, qe_mul, qe_one_minus,
    sample_qe_layers, tensor_at, verify_initial_claims, weighted_layer, ClaimFactor, ClaimTerm,
    LazyPublicEval, PublicFactor, SnarkClaim,
};
use rokoko::protocol::sumcheck::init_sumcheck;
use rokoko::protocol::sumchecks::builder_verifier::init_verifier;

const IN: usize = 32;
const OUT: usize = 2;

const DIGITS: usize = 4;
const BASE_LOG: u64 = 13;
const ARITY: usize = IN / (OUT * DIGITS);

fn zero() -> RingElement {
    RingElement::zero(Representation::IncompleteNTT)
}

fn eq_bits(zs_msb: &[QuadraticExtension], value: usize) -> QuadraticExtension {
    tensor_at(zs_msb, value)
}

/// The public hash matrix, derived from a domain-separated transcript.
fn public_matrix() -> Vec<Vec<RingElement>> {
    let mut hw = HashWrapper::new();
    hw.update_with_u64(0x414a544149);
    (0..OUT)
        .map(|_| {
            (0..IN)
                .map(|_| {
                    let mut x = zero();
                    hw.sample_ring_element_into(&mut x);
                    x
                })
                .collect()
        })
        .collect()
}

fn hash_node(a: &[Vec<RingElement>], x: &[RingElement]) -> [RingElement; OUT] {
    let mut out = [zero(), zero()];
    let mut t = zero();
    for r in 0..OUT {
        for i in 0..IN {
            t *= (&a[r][i], &x[i]);
            out[r] += &t;
        }
    }
    out
}

/// Layer inputs bottom-up: `layers[l][m * 32 + i]` is preimage element `i` of
/// node `m`; `layers[0]` holds the data digits (8 data elements per node).
struct Tree {
    layers: Vec<Vec<RingElement>>,
    root: [RingElement; OUT],
}

fn build_tree(a: &[Vec<RingElement>], data: &[RingElement]) -> Tree {
    let leaf_nodes = data.len() * DIGITS / IN;
    assert!(leaf_nodes.is_power_of_two());
    assert_eq!(
        leaf_nodes.trailing_zeros() as usize % ARITY.trailing_zeros() as usize,
        0,
        "node count must be a power of the arity"
    );
    let mut layers = vec![decompose(data, BASE_LOG, DIGITS)];
    loop {
        let inputs = layers.last().unwrap();
        let m_nodes = inputs.len() / IN;
        let mut outputs = Vec::with_capacity(m_nodes * OUT);
        for m in 0..m_nodes {
            outputs.extend(hash_node(a, &inputs[m * IN..(m + 1) * IN]));
        }
        if m_nodes == 1 {
            return Tree {
                layers,
                root: [outputs[0].clone(), outputs[1].clone()],
            };
        }
        let digits = decompose(&outputs, BASE_LOG, DIGITS);
        // output (m, r, j) lands at preimage slot (s, r, j) of node m / arity
        let mut next = vec![zero(); digits.len()];
        for m in 0..m_nodes {
            let (mn, s) = (m / ARITY, m % ARITY);
            for r in 0..OUT {
                for j in 0..DIGITS {
                    next[(mn * IN) + (s * OUT * DIGITS) + (r * DIGITS) + j] =
                        digits[(m * OUT + r) * DIGITS + j].clone();
                }
            }
        }
        layers.push(next);
    }
}

struct MerkleWitness {
    matrix: VerticallyAlignedMatrix<RingElement>,
    segments: Vec<Prefix>,
}

fn place_witness(tree: &Tree, height: usize, width: usize) -> MerkleWitness {
    let n = height * width;
    let total_vars = n.ilog2() as usize;
    let mut data = vec![zero(); n];
    let mut cursor = 0usize;
    let mut segments = vec![];
    for layer in &tree.layers {
        let size = layer.len();
        assert!(size.is_power_of_two());
        let start = cursor.next_multiple_of(size);
        assert!(start + size <= n, "witness overflow");
        data[start..start + size].clone_from_slice(layer);
        segments.push(Prefix {
            prefix: start / size,
            length: total_vars - size.ilog2() as usize,
        });
        cursor = start + size;
    }
    MerkleWitness {
        matrix: VerticallyAlignedMatrix {
            height,
            width,
            used_cols: width,
            data,
        },
        segments,
    }
}

/// One claim per layer. The transcript must already hold the witness
/// commitment; the verifier rebuilds the same claims with `materialize`
/// off, so the lazy publics carry no data and evaluate in closed form.
fn build_claims(
    a: &[Vec<RingElement>],
    tree_dims: &[usize],
    root: &[RingElement; OUT],
    witness: &MerkleWitness,
    hw: &mut HashWrapper,
    materialize: bool,
) -> Vec<SnarkClaim> {
    let depth = tree_dims.len();
    let mut claims = vec![];
    for l in 0..depth {
        let m_nodes = tree_dims[l];
        let mb = m_nodes.ilog2() as usize;
        let alpha = sample_qe_layers(hw, mb);
        let rho = sample_qe_layers(hw, 1);
        let tau = [qe_one_minus(&rho[0]), rho[0].clone()];
        let k: Vec<RingElement> = (0..IN)
            .map(|i| {
                let mut s = zero();
                let mut t = zero();
                for r in 0..OUT {
                    t *= (&embed_qe(&tau[r]), &a[r][i]);
                    s += &t;
                }
                s
            })
            .collect();

        let pref = witness.segments[l].length;
        let data = materialize.then(|| {
            let mut w = vec![zero(); m_nodes * IN];
            for m in 0..m_nodes {
                let em = embed_qe(&tensor_at(&alpha, m));
                for i in 0..IN {
                    let mut t = em.clone();
                    t *= &k[i];
                    w[m * IN + i] = t;
                }
            }
            Arc::new(w)
        });
        let (alpha_cl, k_cl) = (alpha.clone(), k.clone());
        let eval: LazyPublicEval = Arc::new(move |_ring, qe| {
            let zs: Vec<QuadraticExtension> = qe.iter().rev().cloned().collect();
            let (z_m, z_i) = zs.split_at(zs.len() - 5);
            let em = embed_qe(&eq_layers_qe(&alpha_cl, z_m));
            let mut s = zero();
            let mut t = zero();
            for (i, ki) in k_cl.iter().enumerate() {
                t *= (&embed_qe(&eq_bits(z_i, i)), ki);
                s += &t;
            }
            s *= &em;
            s
        });
        let mut terms = vec![ClaimTerm::scaled(
            RingElement::constant(inv_pow2_q(pref), Representation::IncompleteNTT),
            vec![
                ClaimFactor::Public(PublicFactor::LazyPrefixed {
                    prefix_len: pref,
                    suffix_len: 0,
                    data,
                    eval,
                }),
                ClaimFactor::WitnessSegment(witness.segments[l].clone()),
            ],
        )];

        let mut value = zero();
        if l + 1 < depth {
            // recomposition over the next layer: position (m', s, r, j) with
            // (m', s) = m carries weight eq(alpha, m) tau_r base^j
            let jb = DIGITS.ilog2() as usize;
            let mut j_layers = vec![];
            let mut j_scale: u128 = 1;
            for t in (0..jb).rev() {
                let (aj, sj) = weighted_layer(pow_mod(2, (1 << t) as u64 * BASE_LOG));
                j_layers.push(aj);
                j_scale = j_scale * sj as u128 % Q as u128;
            }
            let layers: Vec<QuadraticExtension> =
                [alpha.clone(), rho.clone(), j_layers].concat();
            let pref_next = witness.segments[l + 1].length;
            let mut coeff = RingElement::constant(
                j_scale as u64,
                Representation::IncompleteNTT,
            );
            coeff *= &RingElement::constant(inv_pow2_q(pref_next), Representation::IncompleteNTT);
            coeff *= &RingElement::constant(Q - 1, Representation::IncompleteNTT);
            terms.push(ClaimTerm::scaled(
                coeff,
                vec![
                    ClaimFactor::Public(PublicFactor::FieldTensor {
                        prefix_len: pref_next,
                        suffix_len: 0,
                        layers: Arc::new(layers),
                    }),
                    ClaimFactor::WitnessSegment(witness.segments[l + 1].clone()),
                ],
            ));
        } else {
            let mut t = zero();
            for r in 0..OUT {
                t *= (&embed_qe(&tau[r]), &root[r]);
                value += &t;
            }
        }
        claims.push(SnarkClaim {
            terms,
            value,
            ct_zero_from_proof: false,
        });
    }
    claims
}

fn main() {
    rokoko::common::init_common();
    let config = match &*params::P_MERKLE {
        Config::Sumcheck(config) => config,
        _ => panic!("Expected sumcheck config at the top level."),
    };
    let n = config.witness_height * config.witness_width;
    let d = n / (2 * DIGITS);
    println!(
        "Ajtai-Merkle: d = 2^{} data elements ({} MB), {} digits of 2^{}",
        d.ilog2(),
        d * 2 * HALF_DEGREE * 50 / 8 / 1_000_000,
        DIGITS,
        BASE_LOG,
    );

    let a = public_matrix();
    let data: Vec<RingElement> = (0..d)
        .map(|_| RingElement::random(Representation::IncompleteNTT))
        .collect();

    let t = std::time::Instant::now();
    let tree = build_tree(&a, &data);
    let tree_dims: Vec<usize> = tree.layers.iter().map(|l| l.len() / IN).collect();
    println!(
        "Tree built: {} ms ({} layers, root public)",
        t.elapsed().as_millis(),
        tree.layers.len()
    );
    let witness = place_witness(&tree, config.witness_height, config.witness_width);

    println!("Generating CRS...");
    let crs = CRS::gen_crs(
        config.composed_witness_length,
        config.basic_commitment_rank + 2,
    );
    let mut sumcheck_context = init_sumcheck(&crs, &config);
    let mut sumcheck_context_verifier = init_verifier(&crs, &config);

    let t = std::time::Instant::now();
    let (commitment_with_aux, rc_commitment) = commit(&crs, &config, &witness.matrix);
    println!("Committed: {} ms", t.elapsed().as_millis());

    let t = std::time::Instant::now();
    let mut hw = HashWrapper::new();
    hw.update_with_ring_element_slice(
        &commitment_with_aux
            .rc_commitment_with_aux
            .most_inner_commitment(),
    );
    let claims = build_claims(&a, &tree_dims, &tree.root, &witness, &mut hw, true);
    let (proof, chain_inputs) = prove_initial_claims(&witness.matrix, &claims, &mut hw);
    println!(
        "Entry sumcheck: {} ms ({} claims, {} openings)",
        t.elapsed().as_millis(),
        claims.len(),
        chain_inputs.claims.len()
    );
    let (chain_proof, _) = prover_round(
        &crs,
        &config,
        &commitment_with_aux,
        &witness.matrix,
        &chain_inputs.evaluation_points_inner,
        &chain_inputs.evaluation_points_outer,
        &mut sumcheck_context,
        false,
        Some(hw),
    );
    println!("Prover total: {} ms", t.elapsed().as_millis());

    let t = std::time::Instant::now();
    let mut hw_v = HashWrapper::new();
    hw_v.update_with_ring_element_slice(&rc_commitment);
    let claims_v = build_claims(&a, &tree_dims, &tree.root, &witness, &mut hw_v, false);
    let inputs_v = verify_initial_claims(
        config.witness_height,
        config.witness_width,
        &claims_v,
        &proof,
        &mut hw_v,
    );
    verifier_round(
        &crs,
        &config,
        &rc_commitment,
        &chain_proof,
        &inputs_v.evaluation_points_inner,
        &inputs_v.evaluation_points_outer,
        &inputs_v.claims,
        &mut sumcheck_context_verifier,
        Some(hw_v),
    );
    println!("Verifier: {} ms", t.elapsed().as_millis());
}

mod params {
    use rokoko::protocol::config::Config;
    use rokoko::protocol::params::p_root_aux;
    use std::sync::LazyLock;

    /// The parameter set as compiled, sixteen chain openings (one per tree
    /// layer plus the two standard witness evaluations), no other change.
    pub static P_MERKLE: LazyLock<Config> =
        LazyLock::new(|| p_root_aux(16).generate_config());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_instance() -> (Vec<Vec<RingElement>>, Tree, Vec<usize>, MerkleWitness) {
        rokoko::common::init_common();
        let a = public_matrix();
        let data: Vec<RingElement> = (0..512)
            .map(|_| RingElement::random(Representation::IncompleteNTT))
            .collect();
        let tree = build_tree(&a, &data);
        let dims: Vec<usize> = tree.layers.iter().map(|l| l.len() / IN).collect();
        let witness = place_witness(&tree, 256, 64);
        (a, tree, dims, witness)
    }

    #[test]
    fn test_entry_roundtrip() {
        let (a, tree, dims, witness) = tiny_instance();
        let mut hw = HashWrapper::new();
        let claims = build_claims(&a, &dims, &tree.root, &witness, &mut hw, true);
        let (proof, _) = prove_initial_claims(&witness.matrix, &claims, &mut hw);
        let mut hw_v = HashWrapper::new();
        let claims_v = build_claims(&a, &dims, &tree.root, &witness, &mut hw_v, false);
        verify_initial_claims(256, 64, &claims_v, &proof, &mut hw_v);
    }

    #[test]
    #[should_panic]
    fn test_tampered_leaf_rejected() {
        let (a, tree, dims, mut witness) = tiny_instance();
        let total_vars = (witness.matrix.height * witness.matrix.width).ilog2() as usize;
        let start = witness.segments[0].prefix << (total_vars - witness.segments[0].length);
        witness.matrix.data[start] += &RingElement::constant(1, Representation::IncompleteNTT);
        let mut hw = HashWrapper::new();
        let claims = build_claims(&a, &dims, &tree.root, &witness, &mut hw, true);
        let (proof, _) = prove_initial_claims(&witness.matrix, &claims, &mut hw);
        let mut hw_v = HashWrapper::new();
        let claims_v = build_claims(&a, &dims, &tree.root, &witness, &mut hw_v, false);
        verify_initial_claims(256, 64, &claims_v, &proof, &mut hw_v);
    }

    #[test]
    #[should_panic]
    fn test_wrong_root_rejected() {
        let (a, tree, dims, witness) = tiny_instance();
        let mut root = tree.root.clone();
        root[0] += &RingElement::constant(1, Representation::IncompleteNTT);
        let mut hw = HashWrapper::new();
        let claims = build_claims(&a, &dims, &root, &witness, &mut hw, true);
        let (proof, _) = prove_initial_claims(&witness.matrix, &claims, &mut hw);
        let mut hw_v = HashWrapper::new();
        let claims_v = build_claims(&a, &dims, &root, &witness, &mut hw_v, false);
        verify_initial_claims(256, 64, &claims_v, &proof, &mut hw_v);
    }
}
