use super::hard_sphere::{
    dimensionless_diameter_q_wca, packing_fraction, packing_fraction_a, packing_fraction_b,
};
use crate::uvcs::corresponding_states::CorrespondingParameters;
use feos_core::StateHD;
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use std::f64::consts::PI;

/// Helmholtz energy density of the soft-repulsive perturbation Delta a0 (Mayer-f),
/// Eq. (27) of the uv-mixtures write-up (Eq. 29 of the uv-theory paper), times rho:
/// -2/3 pi sum_ij rho_i rho_j [g(eta_A,ij) (r_m,ij^3 - q_ij^3) - g(eta_B,ij) (r_m,ij^3 - d_ij^3)]
///
/// `d_ij` are the (additive or non-additive) WCA hard-sphere diameters.
pub fn reference_perturbation_helmholtz_energy_density<D: DualNum<f64> + Copy>(
    parameters: &CorrespondingParameters<D>,
    state: &StateHD<D>,
    d_ij: &DMatrix<D>,
) -> D {
    let p = &parameters;
    let n = p.sigma.len();
    // rho_i rho_j instead of x_i x_j rho^2: identical for normalized molefracs, and exact
    // also when single partial densities are perturbed (sum x != 1 in the dual part)
    let rho = &state.partial_density;
    let d =DVector::from_fn(n, |i, _| d_ij[(i, i)]);
    let eta = packing_fraction(&state.partial_density, &d);
    let eta_a = packing_fraction_a(p, eta, d_ij);
    let eta_b = packing_fraction_b(p, eta, d_ij);
    let mut a = D::zero();

    for i in 0..n {
        for j in 0..n {
            let rs_ij = (p.rep_ij[(i, j)] / p.att_ij[(i, j)])
                .powd((p.rep_ij[(i, j)] - p.att_ij[(i, j)]).inv());
            let t_ij = state.temperature / p.eps_k_ij[(i, j)];
            let rep_ij = p.rep_ij[(i, j)];
            let att_ij = p.att_ij[(i, j)];
            let q_ij = dimensionless_diameter_q_wca(t_ij, rep_ij, att_ij) * p.sigma_ij[(i, j)];

            a += rho[i]
                * rho[j]
                *((-eta_a[(i, j)] * 0.5 + 1.0) / (-eta_a[(i, j)] + 1.0).powi(3)
                    * (-q_ij.powi(3) + (rs_ij * p.sigma_ij[(i, j)]).powi(3))
                    - ((-eta_b[(i, j)] * 0.5 + 1.0) / (-eta_b[(i, j)] + 1.0).powi(3))
                        * (-d_ij[(i, j)].powi(3) + (rs_ij * p.sigma_ij[(i, j)]).powi(3)))
        }
    }

    -a * 2.0 / 3.0 * PI
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::uvcs::eos::hard_sphere::diameter_wca_ij;
    use crate::uvcs::{
        UVCSPars,
        parameters::utils::{test_parameters, test_parameters_mixture},
    };
    use approx::assert_relative_eq;
    use nalgebra::dvector;

    #[test]
    fn test_delta_a0_wca_pure() {
        let reduced_temperature = 4.0;
        let reduced_density = 1.0;

        let _p = test_parameters(24.0, 6.0, 1.0, 1.0);
        let p = UVCSPars::new(&_p);
        let state = StateHD::new(reduced_temperature, 1.0 / reduced_density, &dvector![1.0]);
        let cp = CorrespondingParameters::new(&p, state.temperature);
        let d_ij = diameter_wca_ij(&cp, state.temperature, false);
        let a = reference_perturbation_helmholtz_energy_density(&cp, &state, &d_ij) / reduced_density;
        assert_relative_eq!(a, 0.258690311450425, epsilon = 1e-10);
    }
    #[test]
    fn test_delta_a0_wca_mixture() {
        let molefracs = dvector![0.40000000000000002, 0.59999999999999998];
        let reduced_temperature = 1.0;
        let reduced_density = 0.90000000000000002;

        let _p = test_parameters_mixture(
            dvector![12.0, 12.0],
            dvector![6.0, 6.0],
            dvector![1.0, 1.0],
            dvector![1.0, 0.5],
        );
        let p = UVCSPars::new(&_p);

        let state = StateHD::new(reduced_temperature, 1.0 / reduced_density, &molefracs);
        let cp = CorrespondingParameters::new(&p, state.temperature);
        let d_ij = diameter_wca_ij(&cp, state.temperature, false);
        let a = reference_perturbation_helmholtz_energy_density(&cp, &state, &d_ij) / reduced_density;

        assert_relative_eq!(a, 0.308268896386771, epsilon = 1e-6);
    }

    #[test]
    fn test_delta_b20_mixture() {
        // low-density limit (write-up Eq. 33): Delta a0 / rho -> 2 pi / 3 sum_ij x_i x_j (q_ij^3 - d_ij^3)
        // with pair quantities, also for unequal exponents and non-additive d_ij
        let _p = test_parameters_mixture(
            dvector![12.0, 20.0],
            dvector![6.0, 6.0],
            dvector![1.0, 1.3],
            dvector![1.0, 2.0],
        );
        let p = UVCSPars::new(&_p);
        let x = dvector![0.3, 0.7];
        let rho = 1e-9;
        let temperature = 1.5;
        let state = StateHD::new(temperature, 1.0 / rho, &x);
        let cp = CorrespondingParameters::new(&p, temperature);
        for non_additive in [false, true] {
            let d_ij = diameter_wca_ij(&cp, temperature, non_additive);
            let b20 = reference_perturbation_helmholtz_energy_density(&cp, &state, &d_ij) / rho.powi(2);
            let mut b20_exact = 0.0;
            for i in 0..2 {
                for j in 0..2 {
                    let q = dimensionless_diameter_q_wca(
                        temperature / cp.eps_k_ij[(i, j)],
                        cp.rep_ij[(i, j)],
                        cp.att_ij[(i, j)],
                    ) * cp.sigma_ij[(i, j)];
                    b20_exact += x[i] * x[j] * (q.powi(3) - d_ij[(i, j)].powi(3));
                }
            }
            b20_exact *= 2.0 / 3.0 * PI;
            assert_relative_eq!(b20, b20_exact, max_relative = 1e-7);
        }
    }
}
