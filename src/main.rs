use cowboys_and_aliens::common::config::HALF_DEGREE;
use cowboys_and_aliens::common::init_common;
use cowboys_and_aliens::common::ring_arithmetic::*;
use std::sync::LazyLock;

fn main() {
    init_common();
    let mut a = RingElement::random(Representation::Coefficients);
    let mut b = RingElement::random(Representation::Coefficients);
    let mut c = RingElement::new(Representation::Coefficients);

    naive_polynomial_multiplication(&mut c, &a, &b);

    a.from_coefficients_to_even_odd_coefficients();
    b.from_coefficients_to_even_odd_coefficients();

    a.from_even_odd_coefficients_to_incomplete_ntt_representation();
    b.from_even_odd_coefficients_to_incomplete_ntt_representation();

    let mut d = RingElement::new(Representation::IncompleteNTT);

    let start_time = std::time::Instant::now();
    incomplete_ntt_multiplication(&mut d, &a, &b);
    let duration = start_time.elapsed();
    println!("NTT multiplication took: {:?}", duration);

    d.from_incomplete_ntt_to_even_odd_coefficients();
    d.from_even_odd_coefficients_to_coefficients();

    assert_eq!(c.v, d.v);
    println!("Result matches naive multiplication!");

    let mut b_c = b.clone();
    b_c.from_incomplete_ntt_to_homogenized_field_extensions();
    b_c.from_homogenized_field_extensions_to_incomplete_ntt();
    assert_eq!(b.v, b_c.v);
    println!("Homogenized field extension conversion works!");

    b_c.from_incomplete_ntt_to_homogenized_field_extensions();
    let ext_b: [QuadraticExtension; HALF_DEGREE] = b_c.split_into_quadratic_extensions();
    let mut b_reconstructed = RingElement::new(Representation::HomogenizedFieldExtensions);
    b_reconstructed.combine_from_quadratic_extensions(&ext_b);
    assert_eq!(b_c.v, b_reconstructed.v);
    println!("Splitting and combining extensions works!");

    let mut a_c = a.clone();
    a_c.from_incomplete_ntt_to_homogenized_field_extensions();
    let ext_a: [QuadraticExtension; HALF_DEGREE] = a_c.split_into_quadratic_extensions();

    let quadratic_fields_hadamard: [QuadraticExtension; HALF_DEGREE] = ext_a
        .iter()
        .zip(ext_b.iter())
        .map(|(x, y)| (*x * *y))
        .collect::<Vec<QuadraticExtension>>()
        .try_into()
        .unwrap();

    let mut c_c = RingElement::new(Representation::HomogenizedFieldExtensions);
    c_c.combine_from_quadratic_extensions(&quadratic_fields_hadamard);
    c_c.from_homogenized_field_extensions_to_incomplete_ntt();
    c_c.from_incomplete_ntt_to_even_odd_coefficients();
    c_c.from_even_odd_coefficients_to_coefficients();
    assert_eq!(c.v, c_c.v);
    println!("Hadamard multiplication in quadratic extensions matches naive multiplication!");

    let mut e = RingElement::new(Representation::HomogenizedFieldExtensions);

    a.from_incomplete_ntt_to_homogenized_field_extensions();
    b.from_incomplete_ntt_to_homogenized_field_extensions();

    incomplete_ntt_multiplication_homogenized(&mut e, &a, &b);
    e.from_homogenized_field_extensions_to_incomplete_ntt();
    e.from_incomplete_ntt_to_even_odd_coefficients();
    e.from_even_odd_coefficients_to_coefficients();

    assert_eq!(c.v, e.v);
    println!("Homogenized multiplication matches naive multiplication!");
}
