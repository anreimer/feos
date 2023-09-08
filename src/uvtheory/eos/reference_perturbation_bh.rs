use super::hard_sphere_bh::{
    diameter_bh, packing_fraction, packing_fraction_a, packing_fraction_b,
};
use crate::uvtheory::parameters::*;
use feos_core::{HelmholtzEnergyDual, StateHD};
use num_dual::DualNum;
use std::fmt;
use std::{f64::consts::PI, sync::Arc};

#[derive(Clone)]
pub struct ReferencePerturbationBH {
    pub parameters: Arc<UVParameters>,
}

impl fmt::Display for ReferencePerturbationBH {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Reference Perturbation BH")
    }
}

impl<D: DualNum<f64> + Copy> HelmholtzEnergyDual<D> for ReferencePerturbationBH {
    /// Helmholtz energy for perturbation reference (Mayer-f), eq. 29
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        let p = &self.parameters;
        let n = p.sigma.len();
        let x = &state.molefracs;
        let d = diameter_bh(p, state.temperature);
        let eta = packing_fraction(&p.m, &state.partial_density, &d);
        let eta_a = packing_fraction_a(p, &d, eta);
        let eta_b = packing_fraction_b(p, &d, eta);
        let mut a = D::zero();
        for i in 0..n {
            for j in 0..n {
                let d_ij = (d[i] + d[j]) * 0.5;
                let i0_ij: D = (((-eta_a[[i, j]] * 0.5 + 1.0) / (-eta_a[[i, j]] + 1.0).powi(3))
                    - ((-eta_b[[i, j]] * 0.5 + 1.0) / (-eta_b[[i, j]] + 1.0).powi(3)))
                    * (-d_ij.powi(3) / p.sigma_ij[[i, j]].powi(3) + 1.0)
                    / (2.0 - 2.0 / (p.m[i] + p.m[j]))
                    / 3.0;

                a += x[i] * x[j] * p.m[i] * p.m[j] * p.sigma_ij[[i, j]].powi(3) * i0_ij;
            }
        }

        -a * state.moles.sum().powi(2) * 2.0 / state.volume * PI
    }
}

// fn diameter_bh_chain<D: DualNum<f64>>(parameters: &UVParameters, temperature: D) -> Array1<D> {
//     let p = &parameters;
//     let reduced_temperature = p.epsilon_k.mapv(|eps_k_i| temperature / eps_k_i);
//     let b =[0.852987920795915,-0.128229846701676,
//                         0.833664689185409,0.0240477795238045,
//                         0.0177618321999164,0.127015906854396,
//                         -0.528941139160234,-0.147289922797747];

//     let fac1 = (&p.m - 1.0) / &p.m;
//     let fac2 = fac1 * (&p.m-2.0)/&p.m;
//     let a_1 = b[0] + b[1] * fac1 + b[2] * fac2;
//     let a_2 = b[3] + b[4] * fac1 + b[5] * fac2;
//     let fac3 = ((1.0 / 24.0)*(1.0 + b[6] * fac1 + b[7] * fac2));

//     (1.0 / (1.0 + a_1 * t_st + a_2 * t_st**2 ))**fac3
// }

#[cfg(test)]
mod test {
    use super::*;
    use crate::uvtheory::parameters::utils::test_parameters;
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn test_delta_a0_bh() {
        let moles = arr1(&[2.0]);

        // m = 12.0, t = 4.0, rho = 1.0
        let reduced_temperature = 4.0;
        let reduced_density = 1.0;
        let reduced_volume = moles[0] / reduced_density;

        let p = test_parameters(1.0, 24.0, 6.0, 1.0, 1.0);
        let pt = ReferencePerturbationBH {
            parameters: Arc::new(p),
        };
        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());
        let a = pt.helmholtz_energy(&state) / moles[0];
        assert_relative_eq!(a, -0.0611105573289734, epsilon = 1e-10);
    }
}
