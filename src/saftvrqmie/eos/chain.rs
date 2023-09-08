use crate::saftvrqmie::eos::dispersion::sutherland;
use crate::saftvrqmie::parameters::SaftVRQMieParameters;
use feos_core::{HelmholtzEnergyDual, StateHD};
use ndarray::{Array1, Array2};
use num_dual::DualNum;
use std::f64::consts::{FRAC_PI_6, PI};
use std::fmt;
use std::sync::Arc;
pub struct ChainContribution {
    pub parameters: Arc<SaftVRQMieParameters>,
}

impl<D: DualNum<f64> + Copy> HelmholtzEnergyDual<D> for ChainContribution {
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        let p = &self.parameters;
        let rho = &state.partial_density.sum();
        let mut achain = D::zero();
        let nc = p.m.len();
        let eta_x = onefluid_packingfraction(p, state);

        for i in 0..nc {
            let beta_ii = state.temperature.recip() * p.epsilon_k[i];

            let x0_ii = p
                .hs_diameter_ij(i, i, state.temperature, D::one() * p.sigma[i]) // NOT implemented for Quantum stuff
                .recip()
                * p.sigma[i];

            let ln_gii = log_gmie(
                beta_ii,
                x0_ii,
                eta_x,
                state.partial_density.sum(),
                state.temperature,
                p.m[i],
                p.lr[i],
                p.la[i],
                p.sigma[i],
                p.epsilon_k[i],
            );

            achain += state.molefracs[i] * (p.m[i] - 1.0) * ln_gii;
        }
        -state.moles.sum() * achain
    }
}
impl fmt::Display for ChainContribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ChainContribution")
    }
}

fn log_gmie<D: DualNum<f64> + Copy>(
    beta: D,
    x0: D,
    eta: D,
    density: D,
    temperature: D,
    m: f64,
    lr: f64,
    la: f64,
    sigma: f64,
    eps_k: f64,
) -> D {
    let gdhs = g_hard_sphere(beta, x0, eta);
    //dbg!(gdhs.re());
    //let density = state.partial_density.sum();
    let g_1 = g1(m, lr, la, x0, eta, sigma, density);
    //dbg!(g_1.re());
    let g_2 = g2(m, lr, la, x0, eta, sigma, eps_k, density, temperature);
    // dbg!(g_2.re());
    gdhs.ln() + ((beta * g_1 / gdhs) + (beta.powi(2) * g_2 / gdhs))
}
fn g_hard_sphere<D: DualNum<f64> + Copy>(beta: D, x0: D, eta: D) -> D {
    // Hard sphere RDF
    let k0 = -(-eta + 1.0).ln()
        + (eta * 42.0 - eta.powi(2) * 39.0 + eta.powi(3) * 9.0 - eta.powi(4) * 2.0)
            / ((-eta + 1.0).powi(3) * 6.0);
    let k1 = (eta.powi(4) + eta.powi(2) * 6.0 - eta * 12.0) / ((-eta + 1.0).powi(3) * 2.0);
    let k2 = (-eta.powi(2) * 3.0) / ((-eta + 1.0).powi(2) * 8.0);
    let k3 = (-eta.powi(4) + eta.powi(2) * 3.0 + eta * 3.0) / ((-eta + 1.0).powi(3) * 6.0);
    let g_dhs = (k0 + k1 * x0 + k2 * x0.powi(2) + k3 * x0.powi(3)).exp();
    //

    g_dhs
}
fn B<D: DualNum<f64> + Copy>(eta: D, x0: D, exponent: f64) -> D {
    // Dimensionless (/eps)

    let I_lambda = -((x0).powf(3.0 - exponent) - 1.0) / (exponent - 3.0);
    let J_lambda = -(x0.powf(4.0 - exponent) * (exponent - 3.0)
        - x0.powf(3.0 - exponent) * (exponent - 4.0)
        - 1.0)
        / (exponent - 3.0)
        / (exponent - 4.0);

    eta * 12.0
        * ((-eta / 2.0 + 1.0) / (-eta + 1.0).powi(3) * I_lambda
            - eta * 9.0 * (eta + 1.0) / ((-eta + 1.0).powi(3) * 2.0) * J_lambda)
}

fn dB_deta<D: DualNum<f64> + Copy>(eta: D, x0: D, exponent: f64) -> D {
    let I_lambda = -((x0).powf(3.0 - exponent) - 1.0) / (exponent - 3.0);
    let J_lambda = -(x0.powf(4.0 - exponent) * (exponent - 3.0)
        - x0.powf(3.0 - exponent) * (exponent - 4.0)
        - 1.0)
        / (exponent - 3.0)
        / (exponent - 4.0);

    -(D::one()
        * 6.0
        * ((J_lambda * 36.0 + I_lambda) * eta.powi(2) + eta * (J_lambda * 18.0 - I_lambda * 2.0)
            - I_lambda * 2.0))
        / ((eta - 1.0).powi(4))
}

fn g1<D: DualNum<f64> + Copy>(
    m: f64,
    lr: f64,
    la: f64,
    x0: D,
    eta: D,
    sigma: f64,
    density: D,
) -> D {
    let potential_prefactor_C = lr / (lr - la) * (lr / la).powf(la / (lr - la));

    //First order contribution
    let B_att = B(eta, x0, la);
    let B_rep = B(eta, x0, lr);
    let a1s_att = a1s(la, eta);
    let a1s_rep = a1s(lr, eta);
    let derive_a1_rhos = da1_drhos(lr, la, x0.recip() * sigma, x0, eta);
    let prefactor = x0.powi(3) / 2.0 / PI / sigma.powi(3);

    prefactor
        * (derive_a1_rhos * 3.0
            - x0.powf(la) * potential_prefactor_C * la * (a1s_att + B_att) / (density * m)
            + x0.powf(lr) * potential_prefactor_C * lr * (a1s_rep + B_rep) / (density * m))
}

fn a1s<D: DualNum<f64> + Copy>(exponent: f64, eta: D) -> D {
    let c1 = 0.81096 + 1.7888 / exponent - 37.578 / exponent.powi(2) + 92.284 / exponent.powi(3);
    let c2 = 1.0205 - 19.341 / exponent + 151.26 / exponent.powi(2) - 463.50 / exponent.powi(3);
    let c3 = -1.9057 + 22.845 / exponent - 228.14 / exponent.powi(2) + 973.92 / exponent.powi(3);
    let c4 = 1.0885 + -6.1962 / exponent + 106.98 / exponent.powi(2) + -677.64 / exponent.powi(3);
    let eta_eff = eta * c1 + eta.powi(2) * c2 + eta.powi(3) * c3 + eta.powi(4) * c4;
    -eta * 12.0 * (1.0 / (exponent - 3.0)) * (-eta_eff / 2.0 + 1.0) / (-eta_eff + 1.0).powi(3)
}

fn da1s_deta<D: DualNum<f64> + Copy>(exponent: f64, eta: D) -> D {
    let c1 = 0.81096 + 1.7888 / exponent - 37.578 / exponent.powi(2) + 92.284 / exponent.powi(3);
    let c2 = 1.0205 - 19.341 / exponent + 151.26 / exponent.powi(2) - 463.50 / exponent.powi(3);
    let c3 = -1.9057 + 22.845 / exponent - 228.14 / exponent.powi(2) + 973.92 / exponent.powi(3);
    let c4 = 1.0885 + -6.1962 / exponent + 106.98 / exponent.powi(2) + -677.64 / exponent.powi(3);

    (D::one()
        * 6.0
        * (D::one() * 7.0 * c4.powf(2.0) * eta.powf(8.0)
            + D::one() * 12.0 * c3 * c4 * eta.powf(7.0)
            + D::one() * (10.0 * c2 * c4 + 5.0 * c3.powf(2.0)) * eta.powf(6.0)
            + D::one() * (8.0 * c1 * c4 + 8.0 * c2 * c3) * eta.powf(5.0)
            + D::one() * (-17.0 * c4 + 6.0 * c1 * c3 + 3.0 * c2.powf(2.0)) * eta.powf(4.0)
            + D::one() * (4.0 * c1 * c2 - 12.0 * c3) * eta.powf(3.0)
            + D::one() * (c1.powf(2.0) - 7.0 * c2) * eta.powf(2.0)
            - eta * 2.0 * c1
            - 2.0))
        / ((eta.powf(4.0) * c4 + eta.powf(3.0) * c3 + eta.powf(2.0) * c2 + eta * c1 - 1.0)
            .powf(4.0)
            * (exponent - 3.0))
}
fn da1_drhos<D: DualNum<f64> + Copy>(lr: f64, la: f64, d: D, x0: D, eta: D) -> D {
    let potential_prefactor_C = lr / (lr - la) * (lr / la).powf(la / (lr - la));

    let dB_deta_lambda_a = dB_deta(eta, x0, la);
    let dB_deta_lambda_r = dB_deta(eta, x0, lr);

    let da1s_deta_lambda_a = da1s_deta(la, eta);
    let da1s_deta_lambda_r = da1s_deta(lr, eta);

    let da1_deta = (x0.powf(la) * (da1s_deta_lambda_a + dB_deta_lambda_a)
        - x0.powf(lr) * (da1s_deta_lambda_r + dB_deta_lambda_r))
        * potential_prefactor_C;
    d.powi(3) * da1_deta * FRAC_PI_6 // d with dimension!!!!
}

fn da2_drhos<D: DualNum<f64> + Copy>(lr: f64, la: f64, d: D, x0: D, eta: D, eps_k: f64) -> D {
    let potential_prefactor_C = lr / (lr - la) * (lr / la).powf(la / (lr - la));
    let Khs = (-eta + 1.0).powi(4)
        / (eta * 4.0 + eta.powi(2) * 4.0 - eta.powi(3) * 4.0 + eta.powi(4) + 1.0);
    let B_2att = B(eta, x0, 2.0 * la);
    let B_2rep = B(eta, x0, 2.0 * lr);

    let derive_Khs =
        -(D::one() * 4.0 * (eta - 1.0).powf(3.0) * (eta.powf(2.0) - D::one() * 5.0 * eta - 2.0))
            / (eta.powf(4.0) - D::one() * 4.0 * eta.powf(3.0)
                + D::one() * 4.0 * eta.powf(2.0)
                + D::one() * 4.0 * eta
                + 1.0)
                .powf(2.0);
    let a1s_2att = a1s(2.0 * la, eta);
    let a1s_2rep = a1s(2.0 * lr, eta);
    let B_mix = B(eta, x0, lr + la);
    let a1s_mix = a1s(lr + la, eta);

    let in_bracket = x0.powf(2.0 * la) * (a1s_2att + B_2att)
        - x0.powf(la + lr) * 2.0 * (a1s_mix + B_mix)
        + x0.powf(2.0 * lr) * (a1s_2rep + B_2rep);

    let dB_deta_lambda_2a = dB_deta(eta, x0, 2.0 * la);
    let dB_deta_lambda_2r = dB_deta(eta, x0, 2.0 * lr);
    let da1s_deta_lambda_2a = da1s_deta(2.0 * la, eta);
    let da1s_deta_lambda_2r = da1s_deta(2.0 * lr, eta);
    let dB_deta_lambda_ar = dB_deta(eta, x0, lr + la);
    let da1s_deta_lambda_ar = da1s_deta(la + lr, eta);

    let derive_in_bracket = x0.powf(2.0 * la) * (da1s_deta_lambda_2a + dB_deta_lambda_2a)
        - x0.powf(la + lr) * 2.0 * (da1s_deta_lambda_ar + dB_deta_lambda_ar)
        + x0.powf(2.0 * lr) * (da1s_deta_lambda_2r + dB_deta_lambda_2r);

    let da2_deta =
        (Khs * derive_in_bracket + derive_Khs * in_bracket) / 2.0 * potential_prefactor_C.powi(2);

    d.powi(3) * da2_deta * FRAC_PI_6 // d with dimension!!!!
}
// m, lr, la, x0, eta, eps_k, sigma, density
fn g2<D: DualNum<f64> + Copy>(
    m: f64,
    lr: f64,
    la: f64,
    x0: D,
    eta: D,
    sigma: f64,
    eps_k: f64,
    state_density: D,
    state_temperature: D,
) -> D {
    let potential_prefactor_C = lr / (lr - la) * (lr / la).powf(la / (lr - la));
    let potential_prefactor_C2 = potential_prefactor_C.powi(2);
    let B_2att = B(eta, x0, 2.0 * la);
    let B_2rep = B(eta, x0, 2.0 * lr);
    let a1s_2att = a1s(2.0 * la, eta);
    let a1s_2rep = a1s(2.0 * lr, eta);
    let B_mix = B(eta, x0, lr + la);
    let a1s_mix = a1s(lr + la, eta);
    let Khs = (-eta + 1.0).powi(4)
        / (eta * 4.0 + eta.powi(2) * 4.0 - eta.powi(3) * 4.0 + eta.powi(4) + 1.0);
    let prefactor = x0.powi(3) / 2.0 / PI / sigma.powi(3);

    let da2_drho = da2_drhos(lr, la, x0.recip() * sigma, x0, eta, eps_k);
    let g2_mca = prefactor
        * (da2_drho * 3.0
            - Khs * potential_prefactor_C2 * lr * x0.powf(2.0 * lr) * (a1s_2rep + B_2rep)
                / (state_density * m)
            + Khs * potential_prefactor_C2 * (la + lr) * x0.powf(lr + la) * (a1s_mix + B_mix)
                / (state_density * m)
            - Khs * potential_prefactor_C2 * la * x0.powf(2.0 * la) * (a1s_2att + B_2att)
                / (state_density * m));

    let alpha = potential_prefactor_C * (1.0 / (la - 3.0) - 1.0 / (lr - 3.0));
    let theta = (state_temperature.recip() * eps_k).exp() - 1.0;

    let gamma_c = D::one()
        * (-(10.0 * (0.57 - alpha)).tanh() + 1.0)
        * eta
        * x0.powi(3)
        * theta
        * (-D::one() * 6.7 * eta * x0.powi(3) + D::one() * -8.0 * eta.powi(2) * x0.powi(6)).exp()
        * 10.0;

    g2_mca * (gamma_c + 1.0)
}

fn onefluid_packingfraction<D: DualNum<f64> + Copy>(
    p: &SaftVRQMieParameters,
    state: &StateHD<D>,
) -> D {
    let mut zeta_x = D::zero();
    let nc = p.m.len();
    let mut sum_mx = D::zero();
    for i in 0..nc {
        sum_mx += state.molefracs[i] * p.m[i];
    }
    let rho_s = state.partial_density.sum() * sum_mx;

    for i in 0..nc {
        let xsi = state.molefracs[i] * p.m[i] / sum_mx;
        for j in 0..nc {
            let xsj = state.molefracs[j] * p.m[j] / sum_mx;
            zeta_x += xsi
                * xsj
                * p.hs_diameter_ij(i, j, state.temperature, D::one() * p.sigma_ij[[i, j]])
                    .powi(3)
        }
    }
    zeta_x * FRAC_PI_6 * rho_s
}
#[allow(clippy::excessive_precision)]
#[cfg(test)]
mod test {
    use super::*;
    use crate::saftvrqmie::parameters::*;
    use approx::assert_relative_eq;
    use ndarray::arr1;
    #[test]
    fn test_chain_contribution() {
        let m = 2.0;
        let lr = 16.0;
        let la = 7.0;
        let sigma = 4.0;
        let eps_k = 300.0;
        let fh = 0;
        let mw = 1.0;

        let p = SaftVRQMieParameters::new_simple(m, lr, la, sigma, eps_k, mw, fh);
        let moles = arr1(&[1.0]);
        let reduced_temperature = 1.0;
        let reduced_density = 0.1;
        let reduced_volume = (moles[0]) / reduced_density;
        let pt = ChainContribution {
            parameters: Arc::new(p),
        };
        let state = StateHD::new(
            reduced_temperature * eps_k,
            reduced_volume * sigma.powi(3),
            moles.clone(),
        );
        let a = pt.helmholtz_energy(&state) / (moles[0]);
        dbg!(a.re());
        assert_relative_eq!(a, -0.0438790034673494, epsilon = 1e-10);
    }
}
