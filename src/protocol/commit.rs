mod test {
    use crate::{
        common::{power_series::dot_series_matrix, sampling::sample_random_short_mat},
        subroutines::crs::CRS,
    };

    #[test]
    pub fn commit_verify() {
        let wit_len = 1 << 16;
        let module_size = 4;
        let crs = CRS::gen_crs(wit_len, module_size);

        let reps = 1;
        let witness = sample_random_short_mat(reps, wit_len, 2);
        let commitment = dot_series_matrix(&crs.ck, &witness);

        let statement = crs.ck;
        assert_eq!(dot_series_matrix(&statement, &witness), commitment);
    }
}
