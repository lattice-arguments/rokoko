use crate::common::{
    matrix::Matrix,
    ring_arithmetic::{
        addition, addition_in_place, incomplete_ntt_multiplication, Representation, RingElement,
    },
};

#[derive(Debug, Clone)]
pub struct PowerSeries {
    pub expanded_layers: Vec<RingElement>,
    pub tensors: Matrix<RingElement>,
}

pub fn dot_series_matrix(
    power_series: &[PowerSeries],
    matrix: &Matrix<RingElement>,
) -> Matrix<RingElement> {
    assert!(matrix.width % 2 == 0);

    let extracted_rows: Vec<&[RingElement]> = power_series
        .iter()
        .map(|series| &series.expanded_layers.as_slice()[0..matrix.width])
        .collect();

    let h = matrix.height;
    //let nrows = extracted_rows.len();

    let mut result: Matrix<RingElement> = Matrix::new(h, extracted_rows.len());

    for (r, row) in extracted_rows.iter().enumerate() {
        for c in 0..h {
            let mut acc = RingElement::zero(Representation::IncompleteNTT);
            let mut temp = RingElement::zero(Representation::IncompleteNTT);

            for i in 0..matrix.width {
                incomplete_ntt_multiplication(&mut temp, &row[i], &matrix[(c, i)]);
                addition_in_place(&mut acc, &temp);
            }
            result[(c, r)] = acc;
        }
    }

    result
}

#[cfg(test)]
mod test {
    use std::{result, time::Instant};

    use crate::common::config::MOD_Q;
    use crate::common::power_series::PowerSeries;
    use crate::common::ring_arithmetic::Representation;
    use crate::{
        common::{
            config::DEGREE, sampling::sample_random_short_mat, sampling::sample_random_vector,
        },
        subroutines::crs::CRS,
    };

    use super::*;
    use salsaa::cyclotomic_ring::*;

    fn salsaa_random_power_series(size: usize) -> salsaa::arithmetic::PowerSeries<MOD_Q, DEGREE> {
        let row = salsaa::arithmetic::sample_random_vector(size);

        let mut ps = salsaa::arithmetic::PowerSeries {
            expanded_layers: vec![],
            tensors: vec![],
        };
        let mut current_dim = size;
        while current_dim % 2 == 0 {
            ps.expanded_layers.push(row[0..current_dim].to_vec());
            current_dim /= 2;
            ps.tensors
                .push(vec![CyclotomicRing::one(), row[current_dim - 1]]);
        }
        ps.expanded_layers.push(row[0..current_dim].to_vec());
        ps
    }

    pub fn matrix_from_nested_vec(
        v: Vec<Vec<CyclotomicRing<MOD_Q, DEGREE>>>,
    ) -> Matrix<RingElement> {
        let height = v.len();
        let width = v[0].len();

        for row in &v {
            assert_eq!(row.len(), width, "rows must have same number of columns");
        }

        let mut data = Vec::with_capacity(width * height);
        for row in v {
            for elem in row {
                let mut ring_el = RingElement::new(Representation::IncompleteNTT);
                ring_el.v = elem.data;
                data.push(ring_el);
            }
        }

        Matrix {
            data,
            width,
            height,
        }
    }

    #[test]
    fn compare_dot_series_many() {
        let wit_len: usize = 1 << 18;
        let n_ps: usize = 10;

        let mut old_random_mat = salsaa::arithmetic::sample_random_short_mat(1, wit_len, 2);

        for row in old_random_mat.iter_mut() {
            for elem in row.iter_mut() {
                elem.to_incomplete_ntt_representation();
            }
        }

        let new_random_mat = matrix_from_nested_vec(old_random_mat.clone());

        let mut old_series_vec = Vec::new();
        let mut new_series_vec = Vec::new();

        for _ in 0..n_ps {
            let mut old_random_ps = salsaa_random_power_series(wit_len);

            for t in old_random_ps.tensors.iter_mut() {
                for e in t.iter_mut() {
                    e.to_incomplete_ntt_representation();
                }
            }

            for layer in old_random_ps.expanded_layers.iter_mut() {
                for e in layer.iter_mut() {
                    e.to_incomplete_ntt_representation();
                }
            }

            old_series_vec.push(old_random_ps.clone());

            let expanded_layers = matrix_from_nested_vec(vec![old_random_ps
                .expanded_layers
                .first()
                .unwrap()
                .to_vec()]);

            let random_ps_new = PowerSeries {
                expanded_layers: expanded_layers.data,
                tensors: matrix_from_nested_vec(old_random_ps.tensors.clone()),
            };

            new_series_vec.push(random_ps_new);
        }
        let time = Instant::now();
        let new_result = dot_series_matrix(&new_series_vec, &new_random_mat);
        println!("elapsed for new dot series matrix {:?}", time.elapsed());
        let time = Instant::now();
        let result =
            salsaa::arithmetic::parallel_dot_series_matrix(&old_series_vec, &old_random_mat);
        println!("elapsed for salsaa dot series matrix {:?}", time.elapsed());

        assert_eq!(matrix_from_nested_vec(result), new_result,);
    }
}
