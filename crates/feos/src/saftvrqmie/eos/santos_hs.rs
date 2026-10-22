use super::TemperatureDependentProperties;
use crate::saftvrqmie::parameters::SaftVRQMiePars;
use feos_core::StateHD;
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use std::f64::consts::FRAC_PI_6;
use std::fmt;

/// Non-additive hard-sphere mixture of Santos et al. (J. Chem. Phys. 123, 234512, 2005)
/// as used for SAFT-VRQ Mie by Aasen et al. (J. Chem. Phys. 152, 074507, 2020), Eqs. (14)-(24).
///
/// For chain molecules, the model is applied to the mixture of segments.
pub struct SantosHardSphere;

impl SantosHardSphere {
    pub fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        &self,
        parameters: &SaftVRQMiePars,
        state: &StateHD<D>,
        properties: &TemperatureDependentProperties<D>,
    ) -> D {
        let p = parameters;
        let n = p.m.len();
        let d = &properties.hs_diameter_ij;

        // segment density and segment fractions
        let m_mix = DVector::from_fn(n, |i, _| state.molefracs[i] * p.m[i]).sum();
        let rho_s = state.partial_density.sum() * m_mix;
        let x_s = DVector::from_fn(n, |i, _| state.molefracs[i] * p.m[i] / m_mix);

        // Eq. (14) and (15)
        let d_na3 = DVector::from_fn(n, |i, _| x_s[i] * d[(i, i)].powi(3)).sum();
        let eta = rho_s * d_na3 * FRAC_PI_6;

        // Eqs. (17) - (23)
        let (b2, b3) = virial_coefficients(&x_s, d);
        let a1 = (d_na3 * b2 * 10.0 - b3 * 4.0) / (d_na3.powi(2) * 6.0);
        let a2 = (b3 - d_na3 * b2) / (d_na3.powi(2) * 6.0);

        // Eq. (24) with the Carnahan-Starling equation of state
        let a_cs = (eta * 4.0 - eta.powi(2) * 3.0) / (-eta + 1.0).powi(2);
        rho_s * (-(-eta).ln_1p() * a1 + a_cs * a2)
    }
}

impl fmt::Display for SantosHardSphere {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hard Sphere (Santos)")
    }
}

/// Second and third virial coefficients of the non-additive hard-sphere mixture, Eqs. (17) - (21).
fn virial_coefficients<D: DualNum<f64> + Copy>(x: &DVector<D>, d: &DMatrix<D>) -> (D, D) {
    let n = x.len();
    let delta = |k: usize, i: usize, j: usize| {
        let delta = d[(i, k)] + d[(j, k)] - d[(i, j)];
        if delta.re() > 0.0 { delta } else { D::zero() }
    };
    let c = |k: usize, i: usize, j: usize| {
        let delta_kij = delta(k, i, j);
        delta_kij.powi(3) + delta_kij.powi(2) / d[(i, j)] * delta(i, j, k) * delta(j, i, k) * 1.5
    };

    let mut b2 = D::zero();
    let mut b3 = D::zero();
    for i in 0..n {
        for j in 0..n {
            b2 += x[i] * x[j] * d[(i, j)].powi(3) * 4.0;
            for k in 0..n {
                let b_ijk = (c(k, i, j) * d[(i, j)].powi(3)
                    + c(j, i, k) * d[(i, k)].powi(3)
                    + c(i, j, k) * d[(j, k)].powi(3))
                    * (4.0 / 3.0);
                b3 += x[i] * x[j] * x[k] * b_ijk;
            }
        }
    }
    (b2, b3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::saftvrqmie::eos::hard_sphere::HardSphere;
    use crate::saftvrqmie::parameters::utils::h2_ne_fh;
    use crate::saftvrqmie::parameters::utils::hydrogen_fh;
    use approx::assert_relative_eq;
    use nalgebra::dvector;

    #[test]
    fn test_santos_hs_helmholtz_energy_pure() {
        // reduces to Carnahan-Starling (= BMCSL) for pure components
        let parameters = hydrogen_fh("1");
        let na = 6.02214076e23;
        let t = 26.7060;
        let v = 1.0e26;
        let n = na * 1.1;
        let s = StateHD::new(t, v / n, &dvector![1.0]);
        let properties = TemperatureDependentProperties::new(&parameters, s.temperature);
        let a_santos = SantosHardSphere.helmholtz_energy_density(&parameters, &s, &properties);
        let a_bmcsl = HardSphere.helmholtz_energy_density(&parameters, &s, &properties);
        assert_relative_eq!(a_santos, a_bmcsl, max_relative = 1e-12);
        assert_relative_eq!(a_santos * v / na, 0.54586730268029837, epsilon = 5e-7);
    }

    #[test]
    fn test_santos_hs_helmholtz_energy_mix() {
        let parameters = h2_ne_fh("1");
        let na = 6.02214076e23;
        let t = 30.0;
        let v = 1.0e26;
        let n = dvector![na * 1.1, na * 1.0];
        let s = StateHD::new(t, v / n.sum(), &(&n / n.sum()));
        let properties = TemperatureDependentProperties::new(&parameters, s.temperature);
        let a_rust = SantosHardSphere.helmholtz_energy_density(&parameters, &s, &properties) * v;
        dbg!(a_rust / na);
        assert_relative_eq!(a_rust / na, 1.8249307925054206, epsilon = 5e-7);
    }
}
