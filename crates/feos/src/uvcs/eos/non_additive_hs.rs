use super::hard_sphere::zeta;
use feos_core::StateHD;
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use std::f64::consts::PI;

/// Helmholtz energy density of the non-additive hard-sphere correction (as in SAFT-VRQ Mie).
///
/// First-order correction of the additive BMCSL mixture with diameters d_ii for unlike
/// diameters d_ij that differ from (d_ii + d_jj) / 2:
/// a / V = -2π Σ_ij ρ_i ρ_j d_add,ij² g_hs,ij (d_add,ij - d_ij).
pub fn non_additive_hard_sphere_helmholtz_energy_density<D: DualNum<f64> + Copy>(
    state: &StateHD<D>,
    d_ij: &DMatrix<D>,
) -> D {
    let rho = &state.partial_density;
    let n = rho.len();
    let d = DVector::from_fn(n, |i, _| d_ij[(i, i)]);
    let zeta = zeta(rho, &d);
    let frac_1mz3 = -(zeta[3] - 1.0).recip();

    let mut a = D::zero();
    for i in 0..n {
        for j in 0..n {
            let d_add = (d[i] + d[j]) * 0.5;
            // BMCSL contact value of the additive mixture
            let mu = d[i] * d[j] / (d[i] + d[j]);
            let g_hs = frac_1mz3
                + mu * zeta[2] * frac_1mz3.powi(2) * 3.0
                + (mu * zeta[2]).powi(2) * frac_1mz3.powi(3) * 2.0;
            a -= rho[i] * rho[j] * d_add.powi(2) * g_hs * (d_add - d_ij[(i, j)]);
        }
    }
    a * (2.0 * PI)
}

#[cfg(test)]
mod test {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_non_additive_hs_sign() {
        let state = StateHD::new(1.0, 100.0, &dvector![0.4, 0.6]);
        let a = |d12: f64| {
            let d_ij = dmatrix![2.6, d12; d12, 3.0];
            non_additive_hard_sphere_helmholtz_energy_density(&state, &d_ij)
        };
        // vanishes for additive diameters, positive if the unlike pair is larger
        assert!(a(2.8).abs() < 1e-15);
        assert!(a(2.85) > 0.0);
        assert!(a(2.75) < 0.0);
        // linear in the non-additivity d_ij - d_add
        assert!((a(2.85) + a(2.75)).abs() < 1e-14);
    }
}
