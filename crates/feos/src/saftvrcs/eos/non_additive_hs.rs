use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use std::f64::consts::{FRAC_PI_6, PI};

/// Helmholtz energy density of the non-additive hard-sphere correction (as in SAFT-VRQ Mie).
///
/// First-order correction of the additive BMCSL mixture with diameters d_ii for unlike
/// diameters d_ij that differ from (d_ii + d_jj) / 2:
/// a / V = -2π Σ_ij ρ_s,i ρ_s,j d_add,ij² g_hs,ij (d_add,ij - d_ij).
pub fn helmholtz_energy_density_non_additive_hs<D: DualNum<f64> + Copy>(
    m: &DVector<f64>,
    partial_density: &DVector<D>,
    d_ij: &DMatrix<D>,
) -> D {
    let n = m.len();
    let d = DVector::from_fn(n, |i, _| d_ij[(i, i)]);
    let rho_s = DVector::from_fn(n, |i, _| partial_density[i] * m[i]);

    let mut zeta = [D::zero(); 4];
    for i in 0..n {
        for (k, z) in zeta.iter_mut().enumerate() {
            *z += rho_s[i] * d[i].powi(k as i32) * FRAC_PI_6;
        }
    }
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
            a -= rho_s[i] * rho_s[j] * d_add.powi(2) * g_hs * (d_add - d_ij[(i, j)]);
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
        let m = dvector![1.0, 1.5];
        let rho = dvector![0.004, 0.006];
        let a = |d12: f64| {
            let d_ij = dmatrix![2.6, d12; d12, 3.0];
            helmholtz_energy_density_non_additive_hs(&m, &rho, &d_ij)
        };
        // vanishes for additive diameters, positive if the unlike pair is larger
        assert!(a(2.8).abs() < 1e-15);
        assert!(a(2.85) > 0.0);
        assert!(a(2.75) < 0.0);
        // linear in the non-additivity d_ij - d_add
        assert!((a(2.85) + a(2.75)).abs() < 1e-14);
    }
}
