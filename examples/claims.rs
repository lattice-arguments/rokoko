use rokoko::common::decomposition::decompose;
use rokoko::common::init_common;
use rokoko::common::ring_arithmetic::{Representation, RingElement};
use rokoko::common::sampling::sample_random_short_vector;
use rokoko::protocol::snark::{
    challenge_point, eq, eq_weighted_sum, powers, prove_claims, table, verify_claims, witness_in,
    Claim, Region, Transcript, WitnessBuilder, WitnessShape,
};

const BASE: u64 = 1 << 8;
const DIGITS_PER_TOTAL: usize = 8;

fn main() {
    init_common();

    let balances = sample_random_short_vector(512, 1 << 7, Representation::IncompleteNTT);
    let audit_totals: Vec<RingElement> = (0..32)
        .map(|_| RingElement::random(Representation::IncompleteNTT))
        .collect();
    let audit_digits = decompose(&audit_totals, 8, DIGITS_PER_TOTAL);

    let mut layout = WitnessBuilder::new(256, 8);
    let balances_at = layout.push(&balances);
    let mirror_at = layout.push(&balances);
    let digits_at = layout.push(&audit_digits);
    let everything = Region::whole(2048);
    let witness = layout.finish();

    let prices: Vec<u64> = (1..=512).collect();
    let revenue = (table(prices.clone()).on(balances_at) * witness_in(balances_at)).sum(&witness);
    let energy = (witness_in(everything) * witness_in(everything).conjugate()).sum(&witness);

    let build_claims = |transcript: &mut Transcript| -> Vec<Claim> {
        let spot_check = challenge_point(transcript, balances_at.vars().len());
        let (total_index, digit_index) = digits_at.vars().split_at(5);
        let audit_point = challenge_point(transcript, total_index.len());

        vec![
            Claim::sums_to(
                table(prices.clone()).on(balances_at) * witness_in(balances_at),
                revenue.clone(),
            ),
            Claim::sums_to_zero(
                eq(&spot_check).on(balances_at) * (witness_in(balances_at) - witness_in(mirror_at)),
            ),
            Claim::sums_to(
                eq(&audit_point).on(total_index)
                    * powers(BASE, digit_index.len()).on(digit_index)
                    * witness_in(digits_at),
                eq_weighted_sum(&audit_point, &audit_totals),
            ),
            Claim::sums_to(
                witness_in(everything) * witness_in(everything).conjugate(),
                energy.clone(),
            ),
        ]
    };

    let mut prover_transcript = Transcript::new();
    let prover_claims = build_claims(&mut prover_transcript);
    let (proof, _openings) = prove_claims(&witness, &prover_claims, &mut prover_transcript);
    println!("prover: four claims batched into one sumcheck");

    let mut verifier_transcript = Transcript::new();
    let verifier_claims = build_claims(&mut verifier_transcript);
    verify_claims(
        WitnessShape::new(256, 8),
        &verifier_claims,
        &proof,
        &mut verifier_transcript,
    );
    println!("verifier: accepted the weighted revenue, the mirrored balances,");
    println!("          the digit recomposition of every audit total,");

    let mut norm = energy;
    norm.to_representation(Representation::Coefficients);
    println!(
        "          and the witness energy (squared l2 norm = {})",
        norm.v[0]
    );
}
