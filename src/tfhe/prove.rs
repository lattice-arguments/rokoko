use super::bootstrap::{BlindRotationTrace, BootstrapKey};
use super::embed::pack;
use super::glwe::GlweCiphertext;
use super::poly::{Poly, Q};
use crate::common::config::HALF_DEGREE;
use crate::common::decomposition::decompose;
use crate::common::hash::HashWrapper;
use crate::common::matrix::VerticallyAlignedMatrix;
use crate::common::ring_arithmetic::{Representation, RingElement};
use crate::protocol::commitment::Prefix;
use crate::protocol::snark::{ClaimFactor, ClaimTerm, PublicFactor, SnarkClaim};

#[derive(Clone)]
pub struct Segment {
    pub prefix: Prefix,
    pub len: usize,
}

pub struct CggiWitness {
    pub matrix: VerticallyAlignedMatrix<RingElement>,
    pub seg_acc: Segment,
    pub seg_dig: Segment,
    pub seg_bsk: Segment,
    pub plane_acc: usize,
    pub plane_dig: usize,
    pub plane_bsk: usize,
    pub seg_lop: Vec<Segment>,
    pub seg_rop: Vec<Segment>,
    pub chunk_base_log: u32,
    pub chunks: usize,
    pub l_coords: usize,
    pub k1: usize,
    pub levels: usize,
    pub gadget_base_log: u32,
    pub n_acc_elems: usize,
    pub n_dig_elems: usize,
    pub n_bsk_elems: usize,
    pub n_pairs: usize,
    pub acc0_packed: Vec<Vec<RingElement>>,
    pub acc_final_packed: Vec<Vec<RingElement>>,
}

fn zero() -> RingElement {
    RingElement::zero(Representation::IncompleteNTT)
}

fn constant(v: u64) -> RingElement {
    RingElement::constant(v, Representation::IncompleteNTT)
}

fn x_element() -> RingElement {
    let mut x = RingElement::zero(Representation::EvenOddCoefficients);
    x.v[HALF_DEGREE] = 1;
    x.from_even_odd_coefficients_to_incomplete_ntt_representation();
    x
}

fn pow_q(base: u64, exp: u64) -> u64 {
    let mut acc = 1u128;
    let mut b = base as u128;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            acc = acc * b % Q as u128;
        }
        b = b * b % Q as u128;
        e >>= 1;
    }
    acc as u64
}

fn inv_pow2_q(bits: usize) -> u64 {
    pow_q(pow_q(2, bits as u64), Q - 2)
}

fn glwe_polys(ct: &GlweCiphertext) -> Vec<&Poly> {
    ct.mask.iter().chain(std::iter::once(&ct.body)).collect()
}

fn chunk_planes(elems: &[RingElement], base_log: u32, chunks: usize) -> Vec<Vec<RingElement>> {
    let interleaved = decompose(elems, base_log as u64, chunks);
    (0..chunks)
        .map(|c| {
            (0..elems.len())
                .map(|i| interleaved[i * chunks + c].clone())
                .collect()
        })
        .collect()
}

struct PairIndex {
    lop_src: Vec<usize>,
    rop_src: Vec<usize>,
    out_v: Vec<usize>,
    out_t: Vec<usize>,
    twisted: Vec<bool>,
    eq_of_pair: Vec<usize>,
}

fn pair_index(
    traces: &[&BlindRotationTrace],
    k1: usize,
    levels: usize,
    l: usize,
) -> (PairIndex, std::collections::HashMap<(usize, usize, usize, usize), usize>, usize) {
    let mut bsk_pos = std::collections::HashMap::new();
    let mut bsk_cursor = 0usize;
    let mut used = std::collections::BTreeSet::new();
    for trace in traces {
        for step in &trace.steps {
            used.insert(step.index);
        }
    }
    for gi in used {
        for u in 0..k1 {
            for j in 0..levels {
                for v in 0..k1 {
                    bsk_pos.insert((gi, u, j, v), bsk_cursor);
                    bsk_cursor += l;
                }
            }
        }
    }

    let mut idx = PairIndex {
        lop_src: vec![],
        rop_src: vec![],
        out_v: vec![],
        out_t: vec![],
        twisted: vec![],
        eq_of_pair: vec![],
    };
    let mut dig_base = 0usize;
    let mut eq_base = 0usize;
    for trace in traces {
        for step in &trace.steps {
            for u in 0..k1 {
                for j in 0..levels {
                    for v in 0..k1 {
                        let d_idx = dig_base + (u * levels + j) * l;
                        let g_idx = bsk_pos[&(step.index, u, j, v)];
                        for t1 in 0..l {
                            for t2 in 0..l {
                                idx.lop_src.push(d_idx + t1);
                                idx.rop_src.push(g_idx + t2);
                                idx.out_v.push(v);
                                let (t, tw) = if t1 + t2 < l {
                                    (t1 + t2, false)
                                } else {
                                    (t1 + t2 - l, true)
                                };
                                idx.out_t.push(t);
                                idx.twisted.push(tw);
                                idx.eq_of_pair.push(eq_base + v * l + t);
                            }
                        }
                    }
                }
            }
            dig_base += k1 * levels * l;
            eq_base += k1 * l;
        }
    }
    (idx, bsk_pos, bsk_cursor)
}

impl CggiWitness {
    pub fn build(
        bsk: &BootstrapKey,
        traces: &[&BlindRotationTrace],
        chunk_base_log: u32,
        chunks: usize,
        height: usize,
        width: usize,
    ) -> CggiWitness {
        let k1 = bsk.ggsws[0].rows.len();
        let levels = bsk.ggsws[0].levels;
        let l = pack(&traces[0].acc_states[0].body).len();

        let mut acc_vals: Vec<RingElement> = vec![];
        for trace in traces {
            for state in &trace.acc_states {
                for poly in glwe_polys(state) {
                    acc_vals.extend(pack(poly));
                }
            }
        }
        let mut dig_vals: Vec<RingElement> = vec![];
        for trace in traces {
            for step in &trace.steps {
                for poly_digits in &step.digits {
                    for d in poly_digits {
                        dig_vals.extend(pack(d));
                    }
                }
            }
        }

        let (pairs, bsk_pos, bsk_len) = pair_index(traces, k1, levels, l);
        let mut bsk_vals = vec![zero(); bsk_len];
        for ((gi, u, j, v), pos) in &bsk_pos {
            let packed = pack(glwe_polys(&bsk.ggsws[*gi].rows[*u][*j])[*v]);
            bsk_vals[*pos..pos + l].clone_from_slice(&packed);
        }

        let lop_vals: Vec<RingElement> = pairs.lop_src.iter().map(|&i| dig_vals[i].clone()).collect();
        let rop_vals: Vec<RingElement> = pairs.rop_src.iter().map(|&i| bsk_vals[i].clone()).collect();

        let n = height * width;
        let total_vars = n.ilog2() as usize;
        let mut data = vec![zero(); n];
        let mut cursor = 0usize;
        fn place(
            data: &mut [RingElement],
            cursor: &mut usize,
            total_vars: usize,
            plane: &[RingElement],
            size: usize,
        ) -> Segment {
            let start = (*cursor + size - 1) / size * size;
            assert!(start + size <= data.len(), "witness overflow");
            data[start..start + plane.len()].clone_from_slice(plane);
            *cursor = start + size;
            Segment {
                prefix: Prefix {
                    prefix: start / size,
                    length: total_vars - size.ilog2() as usize,
                },
                len: size,
            }
        }
        fn alloc_stacked(
            data: &mut [RingElement],
            cursor: &mut usize,
            total_vars: usize,
            vals: &[RingElement],
            chunk_base_log: u32,
            chunks: usize,
        ) -> (Segment, usize) {
            let planes = chunk_planes(vals, chunk_base_log, chunks);
            let plane_pad = vals.len().next_power_of_two().max(1);
            let mut stacked = vec![zero(); plane_pad * chunks.next_power_of_two()];
            for (c, plane) in planes.iter().enumerate() {
                stacked[c * plane_pad..c * plane_pad + plane.len()].clone_from_slice(plane);
            }
            let len = stacked.len();
            let seg = place(data, cursor, total_vars, &stacked, len);
            (seg, plane_pad)
        }
        fn alloc_planes(
            data: &mut [RingElement],
            cursor: &mut usize,
            total_vars: usize,
            vals: &[RingElement],
            chunk_base_log: u32,
            chunks: usize,
        ) -> Vec<Segment> {
            chunk_planes(vals, chunk_base_log, chunks)
                .into_iter()
                .map(|plane| {
                    let size = plane.len().next_power_of_two().max(1);
                    place(data, cursor, total_vars, &plane, size)
                })
                .collect()
        }

        let (seg_acc, plane_acc) =
            alloc_stacked(&mut data, &mut cursor, total_vars, &acc_vals, chunk_base_log, chunks);
        let (seg_dig, plane_dig) =
            alloc_stacked(&mut data, &mut cursor, total_vars, &dig_vals, chunk_base_log, chunks);
        let (seg_bsk, plane_bsk) =
            alloc_stacked(&mut data, &mut cursor, total_vars, &bsk_vals, chunk_base_log, chunks);
        let seg_lop =
            alloc_planes(&mut data, &mut cursor, total_vars, &lop_vals, chunk_base_log, chunks);
        let seg_rop =
            alloc_planes(&mut data, &mut cursor, total_vars, &rop_vals, chunk_base_log, chunks);

        CggiWitness {
            matrix: VerticallyAlignedMatrix {
                height,
                width,
                used_cols: width,
                data,
            },
            seg_acc,
            seg_dig,
            seg_bsk,
            plane_acc,
            plane_dig,
            plane_bsk,
            seg_lop,
            seg_rop,
            chunk_base_log,
            chunks,
            l_coords: l,
            k1,
            levels,
            gadget_base_log: bsk.ggsws[0].base_log,
            n_acc_elems: acc_vals.len(),
            n_dig_elems: dig_vals.len(),
            n_bsk_elems: bsk_vals.len(),
            n_pairs: pairs.lop_src.len(),
            acc0_packed: traces
                .iter()
                .map(|t| {
                    glwe_polys(&t.acc_states[0])
                        .into_iter()
                        .flat_map(|p| pack(p))
                        .collect()
                })
                .collect(),
            acc_final_packed: traces
                .iter()
                .map(|t| {
                    glwe_polys(t.acc_states.last().unwrap())
                        .into_iter()
                        .flat_map(|p| pack(p))
                        .collect()
                })
                .collect(),
        }
    }
}

fn seg_factor(seg: &Segment) -> ClaimFactor {
    ClaimFactor::WitnessSegment(seg.prefix.clone())
}

fn pub_factor(seg: &Segment, mut values: Vec<RingElement>) -> ClaimFactor {
    values.resize(seg.len, zero());
    ClaimFactor::Public(PublicFactor::DensePrefixed(seg.prefix.length, values))
}

fn normalized(seg: &Segment, extra: u64) -> RingElement {
    let mut c = constant(inv_pow2_q(seg.prefix.length));
    c *= &constant(extra);
    c
}

fn sample_rho(hash_wrapper: &mut HashWrapper, len: usize) -> Vec<RingElement> {
    let mut rho = vec![zero(); len];
    hash_wrapper.sample_ring_element_vec_into(&mut rho);
    rho
}

fn neg(e: &RingElement) -> RingElement {
    let mut m = e.clone();
    m *= &constant(Q - 1);
    m
}

fn copy_claim(
    dst_segs: &[Segment],
    src_seg: &Segment,
    src_plane_pad: usize,
    src_of: &[usize],
    chunks: usize,
    hash_wrapper: &mut HashWrapper,
) -> Vec<SnarkClaim> {
    (0..chunks)
        .map(|c| {
            let rho = sample_rho(hash_wrapper, src_of.len());
            let mut scatter = vec![zero(); src_seg.len];
            for (o, &src) in src_of.iter().enumerate() {
                scatter[c * src_plane_pad + src] += &rho[o];
            }
            SnarkClaim {
                terms: vec![
                    ClaimTerm::scaled(
                        normalized(&dst_segs[c], 1),
                        vec![pub_factor(&dst_segs[c], rho), seg_factor(&dst_segs[c])],
                    ),
                    ClaimTerm::scaled(
                        normalized(src_seg, Q - 1),
                        vec![pub_factor(src_seg, scatter), seg_factor(src_seg)],
                    ),
                ],
                value: zero(),
            }
        })
        .collect()
}

fn conv_coeff(shift_packed: &[RingElement], x: &RingElement, t: usize, t1: usize, l: usize) -> RingElement {
    if t >= t1 {
        shift_packed[t - t1].clone()
    } else {
        let mut m = shift_packed[t + l - t1].clone();
        m *= x;
        m
    }
}

pub fn build_claims(
    witness: &CggiWitness,
    traces: &[&BlindRotationTrace],
    hash_wrapper: &mut HashWrapper,
) -> Vec<SnarkClaim> {
    let k1 = witness.k1;
    let levels = witness.levels;
    let l = witness.l_coords;
    let chunks = witness.chunks;
    let delta = witness.chunk_base_log as u64;
    let x = x_element();

    let (pairs, _, _) = pair_index(traces, k1, levels, l);

    let mut claims = vec![];

    claims.extend(copy_claim(
        &witness.seg_lop,
        &witness.seg_dig,
        witness.plane_dig,
        &pairs.lop_src,
        chunks,
        hash_wrapper,
    ));
    claims.extend(copy_claim(
        &witness.seg_rop,
        &witness.seg_bsk,
        witness.plane_bsk,
        &pairs.rop_src,
        chunks,
        hash_wrapper,
    ));

    let n_lin_eqs: usize = traces
        .iter()
        .map(|t| t.steps.len() * k1 * l + 2 * k1 * l)
        .sum();
    let rho = sample_rho(hash_wrapper, n_lin_eqs);
    let mut pub_acc = vec![zero(); witness.n_acc_elems];
    let mut pub_dig = vec![zero(); witness.n_dig_elems];
    let mut value = zero();
    let mut tmp = zero();

    let mut eq = 0usize;
    let mut acc_base = 0usize;
    let mut dig_base = 0usize;
    for (b, trace) in traces.iter().enumerate() {
        for (s, step) in trace.steps.iter().enumerate() {
            let n_prime = trace.acc_states[0].body.n();
            let shift_packed = {
                let mut mono = Poly::zero(n_prime);
                mono.coeffs[step.a_switched % n_prime] =
                    if step.a_switched < n_prime { 1 } else { Q - 1 };
                pack(&mono)
            };
            for u in 0..k1 {
                for t in 0..l {
                    let r = &rho[eq];
                    eq += 1;
                    for j in 0..levels {
                        let mut w = r.clone();
                        w *= &constant(pow_q(2, witness.gadget_base_log as u64 * j as u64));
                        pub_dig[dig_base + (u * levels + j) * l + t] += &w;
                    }
                    let acc_idx = acc_base + s * k1 * l + u * l;
                    for t1 in 0..l {
                        let mut coeff = conv_coeff(&shift_packed, &x, t, t1, l);
                        coeff *= r;
                        pub_acc[acc_idx + t1] += &neg(&coeff);
                    }
                    pub_acc[acc_idx + t] += r;
                }
            }
            dig_base += k1 * levels * l;
        }

        let first = acc_base;
        let last = acc_base + (trace.acc_states.len() - 1) * k1 * l;
        for (offset, public) in [
            (first, &witness.acc0_packed[b]),
            (last, &witness.acc_final_packed[b]),
        ] {
            for i in 0..k1 * l {
                let r = &rho[eq];
                eq += 1;
                pub_acc[offset + i] += r;
                tmp *= (r, &public[i]);
                value += &tmp;
            }
        }
        acc_base += trace.acc_states.len() * k1 * l;
    }
    assert_eq!(eq, n_lin_eqs);

    claims.push(SnarkClaim {
        terms: vec![
            ClaimTerm::scaled(
                normalized(&witness.seg_dig, 1),
                vec![
                    pub_factor(
                        &witness.seg_dig,
                        stacked_public(&pub_dig, witness.plane_dig, witness.seg_dig.len, chunks, delta),
                    ),
                    seg_factor(&witness.seg_dig),
                ],
            ),
            ClaimTerm::scaled(
                normalized(&witness.seg_acc, 1),
                vec![
                    pub_factor(
                        &witness.seg_acc,
                        stacked_public(&pub_acc, witness.plane_acc, witness.seg_acc.len, chunks, delta),
                    ),
                    seg_factor(&witness.seg_acc),
                ],
            ),
        ],
        value,
    });

    let n_prod_eqs: usize = traces.iter().map(|t| t.steps.len() * k1 * l).sum();
    let rho2 = sample_rho(hash_wrapper, n_prod_eqs);
    let mut pub2_acc = vec![zero(); witness.n_acc_elems];
    let mut weight = vec![zero(); witness.n_pairs];

    let mut eq_base = 0usize;
    let mut acc_base = 0usize;
    for trace in traces {
        for (s, _) in trace.steps.iter().enumerate() {
            for v in 0..k1 {
                for t in 0..l {
                    let r = &rho2[eq_base + v * l + t];
                    pub2_acc[acc_base + (s + 1) * k1 * l + v * l + t] += r;
                    pub2_acc[acc_base + s * k1 * l + v * l + t] += &neg(r);
                }
            }
            eq_base += k1 * l;
        }
        acc_base += trace.acc_states.len() * k1 * l;
    }
    for o in 0..witness.n_pairs {
        let mut w = rho2[pairs.eq_of_pair[o]].clone();
        if pairs.twisted[o] {
            w *= &x;
        }
        weight[o] = neg(&w);
    }

    let mut prod_terms = vec![ClaimTerm::scaled(
        normalized(&witness.seg_acc, 1),
        vec![
            pub_factor(
                &witness.seg_acc,
                stacked_public(&pub2_acc, witness.plane_acc, witness.seg_acc.len, chunks, delta),
            ),
            seg_factor(&witness.seg_acc),
        ],
    )];
    for c in 0..chunks {
        for c2 in 0..chunks {
            let scale2 = pow_q(2, delta * (c as u64 + c2 as u64));
            prod_terms.push(ClaimTerm::scaled(
                normalized(&witness.seg_lop[c], scale2),
                vec![
                    pub_factor(&witness.seg_lop[c], weight.clone()),
                    seg_factor(&witness.seg_lop[c]),
                    seg_factor(&witness.seg_rop[c2]),
                ],
            ));
        }
    }
    claims.push(SnarkClaim {
        terms: prod_terms,
        value: zero(),
    });

    claims
}

fn stacked_public(
    public: &[RingElement],
    plane_pad: usize,
    seg_len: usize,
    chunks: usize,
    delta: u64,
) -> Vec<RingElement> {
    let mut out = vec![zero(); seg_len];
    for c in 0..chunks {
        let scale = constant(pow_q(2, delta * c as u64));
        for (e, p) in public.iter().enumerate() {
            let mut v = p.clone();
            v *= &scale;
            out[c * plane_pad + e] = v;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::snark::{prove_initial_claims, verify_initial_claims};
    use rand::SeedableRng;

    fn toy_instance() -> (CggiWitness, super::super::bootstrap::BlindRotationTrace) {
        crate::common::init_common();
        let params = &super::super::TOY;
        let mut rng = rand::rngs::StdRng::seed_from_u64(21);
        let keys = super::super::keygen(params, &mut rng);
        let ct = super::super::encrypt(params, &keys, 5, &mut rng);
        let lut = super::super::bootstrap::generate_lut(
            params.polynomial_size,
            (params.message_modulus * params.carry_modulus) as usize,
            params.delta(),
            |m| m,
        );
        let (_, trace) = super::super::bootstrap::blind_rotate_traced(
            &keys.bsk,
            &ct,
            &lut,
            params.glwe_dimension,
        );
        let witness = CggiWitness::build(&keys.bsk, &[&trace], 13, 4, 2048, 4);
        (witness, trace)
    }

    #[test]
    fn test_cggi_claims_roundtrip() {
        let (witness, trace) = toy_instance();

        let mut hw_p = HashWrapper::new();
        let claims_p = build_claims(&witness, &[&trace], &mut hw_p);
        let (proof, chain_p) = prove_initial_claims(&witness.matrix, &claims_p, &mut hw_p);

        let mut hw_v = HashWrapper::new();
        let claims_v = build_claims(&witness, &[&trace], &mut hw_v);
        let chain_v = verify_initial_claims(
            witness.matrix.height,
            witness.matrix.width,
            &claims_v,
            &proof,
            &mut hw_v,
        );
        assert_eq!(chain_p.claims, chain_v.claims);

        for j in 0..chain_p.claims.len() {
            let direct = crate::protocol::open::claim(
                &witness.matrix,
                &chain_p.evaluation_points_inner[j],
                &chain_p.evaluation_points_outer[j],
            );
            assert_eq!(direct, chain_p.claims[j], "opening {}", j);
        }
    }

    #[test]
    fn test_cggi_four_bootstraps_shared_bsk() {
        crate::common::init_common();
        let params = &super::super::TOY;
        let mut rng = rand::rngs::StdRng::seed_from_u64(33);
        let keys = super::super::keygen(params, &mut rng);
        let lut = super::super::bootstrap::generate_lut(
            params.polynomial_size,
            (params.message_modulus * params.carry_modulus) as usize,
            params.delta(),
            |m| (3 * m + 1) % 16,
        );
        let traces: Vec<_> = (0..4)
            .map(|m| {
                let ct = super::super::encrypt(params, &keys, m as u64, &mut rng);
                super::super::bootstrap::blind_rotate_traced(
                    &keys.bsk,
                    &ct,
                    &lut,
                    params.glwe_dimension,
                )
                .1
            })
            .collect();
        let trace_refs: Vec<&_> = traces.iter().collect();
        let witness = CggiWitness::build(&keys.bsk, &trace_refs, 13, 4, 4096, 8);

        let mut hw_p = HashWrapper::new();
        let claims_p = build_claims(&witness, &trace_refs, &mut hw_p);
        let (proof, chain_p) = prove_initial_claims(&witness.matrix, &claims_p, &mut hw_p);

        let mut hw_v = HashWrapper::new();
        let claims_v = build_claims(&witness, &trace_refs, &mut hw_v);
        let chain_v = verify_initial_claims(
            witness.matrix.height,
            witness.matrix.width,
            &claims_v,
            &proof,
            &mut hw_v,
        );
        assert_eq!(chain_p.claims, chain_v.claims);
    }

    #[test]
    #[should_panic(expected = "round claim mismatch")]
    fn test_cggi_wrong_output_rejected() {
        let (mut witness, trace) = toy_instance();

        let mut hw_p = HashWrapper::new();
        let claims_p = build_claims(&witness, &[&trace], &mut hw_p);
        let (proof, _) = prove_initial_claims(&witness.matrix, &claims_p, &mut hw_p);

        witness.acc_final_packed[0][0] += &constant(1);
        let mut hw_v = HashWrapper::new();
        let claims_v = build_claims(&witness, &[&trace], &mut hw_v);
        verify_initial_claims(
            witness.matrix.height,
            witness.matrix.width,
            &claims_v,
            &proof,
            &mut hw_v,
        );
    }
}
