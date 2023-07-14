//use super::attractive_perturbation_wca::one_fluid_properties;
use super::hard_sphere_bh::diameter_bh;
use crate::uvtheory::parameters::*;
use feos_core::{HelmholtzEnergyDual, StateHD};
use ndarray::{Array1, Array2};
use num_dual::DualNum;
use std::{
    f64::consts::{FRAC_PI_3, PI},
    fmt,
    sync::Arc,
};

const C_BH: [[f64; 4]; 2] = [
    [
        0.168966996450507,
        -0.991545819144238,
        0.743142180601202,
        -4.32349593441145,
    ],
    [
        -0.532628162859638,
        2.66039013993583,
        -1.95070279905704,
        -0.000137219512394905,
    ],
];

/// Constants for BH u-fraction.
const CU_BH: [[f64; 2]; 4] = [
    [0.72188, 0.0],
    [-0.0059822, 2.4676],
    [2.2919, 14.9735],
    [5.1647, 2.4017],
];

/// Constants for BH effective inverse reduced temperature.
const C2: [[f64; 2]; 3] = [
    [1.50542979585173e-03, 3.90426109607451e-02],
    [3.23388827421376e-04, 1.29508541592689e-02],
    [5.25749466058948e-05, 5.26748277148572e-04],
];

const C_BH_CHAIN: [f64; 20] = [
    0.03062297,
    0.02559041,
    0.14768986,
    -0.11195757,
    0.09753004,
    0.01922224,
    0.0336288,
    -0.33977454,
    -0.04639408,
    0.05173679,
    -0.02174187,
    0.3025229,
    -0.06062949,
    -2.38534573,
    0.04236449,
    0.44611938,
    0.60159647,
    -0.38411535,
    -0.44319917,
    0.40444317,
];

const C_BH_CHAIN_INTRA: [f64; 4] = [-0.07897173, -0.32513486, 0.19299206, -4.11256466];

const NU: f64 = 0.25;

#[derive(Clone)]
pub struct AttractivePerturbationBH {
    pub parameters: Arc<UVParameters>,
}

impl fmt::Display for AttractivePerturbationBH {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Attractive Perturbation BH")
    }
}

#[derive(Debug)]
struct OneFluidProperties<D> {
    m: D,
    rep: D,
    att: D,
    sigma: D,
    epsilon_k: D,
    /// sigma cubed using quadratic mixture rule
    sigma3_quadratic: D,
    reduced_diameter: D,
    reduced_segment_density: D,
    reduced_temperature: D,
}

impl<D: DualNum<f64> + Copy> OneFluidProperties<D> {
    fn new(
        parameters: &UVParameters,
        x: &Array1<D>,
        partial_density: &Array1<D>,
        temperature: D,
    ) -> Self {
        // non-reduced temperature dependent diameter
        let d = diameter_bh(parameters, temperature);

        let mut epsilon_k = D::zero();
        let mut sigma3_quadratic = D::zero();
        let mut rep = D::zero();
        let mut att = D::zero();
        let mut d_x_3 = D::zero();
        let mut m = D::zero();
        let mut mbar_quadratic = D::zero();
        for i in 0..parameters.ncomponents {
            let xi = x[i];
            let mi = parameters.m[i];
            m += x[i] * mi;
            d_x_3 += x[i] * mi * d[i].powi(3);
            for j in 0..parameters.ncomponents {
                mbar_quadratic += xi * x[j] * mi * parameters.m[j];
                let _y = xi * x[j] * mi * parameters.m[j] * parameters.sigma_ij[[i, j]].powi(3);
                sigma3_quadratic += _y;
                epsilon_k += _y * parameters.eps_k_ij[[i, j]];

                rep += xi * x[j] * parameters.rep_ij[[i, j]];
                att += xi * x[j] * parameters.att_ij[[i, j]];
            }
        }
        sigma3_quadratic /= mbar_quadratic;
        let sigma =
            ((x * &parameters.m * &parameters.sigma.mapv(|v| v.powi(3))).sum() / m).powf(1.0 / 3.0);
        let reduced_diameter = (d_x_3 / m).powf(1.0 / 3.0) / sigma;
        let reduced_segment_density = partial_density.sum() * sigma.powi(3) * m;
        epsilon_k /= sigma3_quadratic * mbar_quadratic;
        let reduced_temperature = temperature / epsilon_k;
        Self {
            m,
            rep,
            att,
            sigma,
            epsilon_k,
            sigma3_quadratic,
            reduced_diameter,
            reduced_segment_density,
            reduced_temperature,
        }
    }
}

impl<D: DualNum<f64> + Copy> HelmholtzEnergyDual<D> for AttractivePerturbationBH {
    /// Helmholtz energy for attractive perturbation, eq. 52
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        let p = &self.parameters;
        let x = &state.molefracs;
        let t = state.temperature;
        let density = state.partial_density.sum();
        let n = x.len();

        // vdw effective one fluid properties
        let one_fluid = OneFluidProperties::new(p, x, &state.partial_density, t);

        // Not intended to work for mixtures
        let m2 = (one_fluid.m - 2.0) / one_fluid.m;
        let m2_squared = m2 * m2;
        let a_intra = m2_squared * C_BH_CHAIN_INTRA[0];
        let b_intra = m2_squared * C_BH_CHAIN_INTRA[1];
        let c_intra = m2_squared * C_BH_CHAIN_INTRA[2];

        let mut delta_a1u_intra = D::zero();
        let mut delta_b12u_intra = D::zero();
        let cmie = mie_prefactor(one_fluid.rep, one_fluid.att);
        if p.m[0] > 2.0 {
            delta_a1u_intra = one_fluid.reduced_segment_density
                * cmie
                * (a_intra
                    + b_intra * one_fluid.reduced_segment_density
                    + c_intra * one_fluid.reduced_segment_density.powf(2.0));
            let fac = one_fluid.m.recip() * C_BH_CHAIN_INTRA[3] + 1.0;
            delta_a1u_intra *= one_fluid.m.powd(fac) / one_fluid.reduced_temperature;
            delta_b12u_intra = cmie * a_intra * one_fluid.m.powd(fac)
                / one_fluid.reduced_temperature
                * (x * &p.m * &p.sigma.mapv(|s| s.powi(3))).sum();
        }

        let mut delta_a1u = D::zero();
        let mut delta = D::zero();
        let phi_x = u_fraction_bh_chain(
            one_fluid.m,
            one_fluid.reduced_segment_density,
            one_fluid.reduced_temperature.recip(),
        );
        let d = diameter_bh(p, state.temperature);
        for i in 0..n {
            let xi = x[i];
            let mi = p.m[i];
            for j in 0..n {
                let alpha_ij = mean_field_constant_f64(p.rep_ij[[i, j]], p.att_ij[[i, j]], 1.0);
                let (i_bh_ij, b21_inter_ij) = correlation_integral_and_b21_bh(
                    one_fluid.reduced_segment_density,
                    one_fluid.reduced_temperature,
                    alpha_ij,
                    one_fluid.m,
                    p.rep_ij[[i, j]],
                    p.att_ij[[i, j]],
                    one_fluid.sigma3_quadratic,
                    one_fluid.reduced_diameter,
                );
                delta_a1u += xi
                    * x[j]
                    * mi
                    * p.m[j]
                    * p.sigma_ij[[i, j]].powi(3)
                    * p.eps_k_ij[[i, j]]
                    * i_bh_ij;
                delta += xi
                    * x[j]
                    * (delta_b2_lj_chain(
                        state.temperature / p.eps_k_ij[[i, j]],
                        0.5 * (mi + p.m[j]),
                        p.sigma_ij[[i, j]],
                        (d[i] + d[j]) / (p.sigma[i] + p.sigma[j]),
                    ) - (b21_inter_ij + delta_b12u_intra));
            }
        }
        delta_a1u = density / state.temperature * delta_a1u * 2.0 * PI + delta_a1u_intra;
        state.moles.sum() * (delta_a1u + (-phi_x + 1.0) * delta * density)
    }
}

fn delta_b12u<D: DualNum<f64>>(t_x: D, mean_field_constant_x: D, weighted_sigma3_ij: D) -> D {
    -mean_field_constant_x / t_x * 2.0 * PI * weighted_sigma3_ij
}

fn delta_b2_lj_chain<D: DualNum<f64> + Copy>(
    reduced_temperature: D,
    m: f64,
    sigma: f64,
    d: D,
) -> D {
    let m1 = (m - 1.0) / m;
    let m12 = (m - 2.0) / m * m1;
    let m123 = (m - 3.0) / m * m12 / m;
    let c_mie = mie_prefactor(12.0, 6.0);
    let mean_field_constant = mean_field_constant(12.0, 6.0, 1.0) / c_mie;
    let fac = d - 1.0;
    let m_nu = m.powf(-NU);
    let t_recip = reduced_temperature.recip();

    let c1 = [
        4.18938869e-02,
        1.27313440e-02,
        1.37047712e-01,
        -1.02530116e-01,
        8.49115847e-01,
        -6.70977982e-01,
        -6.71875321e-01,
        6.13530457e-01,
    ];

    let a1 = -mean_field_constant;
    let a2 = fac * (m_nu * c1[5] + c1[4]) + c1[0] + m_nu * c1[1];
    let a3 = fac * (m_nu * c1[7] + c1[6]) + c1[2] + m_nu * c1[3];

    let prefac = t_recip * 2.0;
    let b_21 = prefac * c_mie * (a2 * m1 + a3 * m12 + a1) * m_nu;

    let c2 = [
        0.10161078,
        -0.37280908,
        0.54190839,
        0.21956392,
        -0.11806782,
        5.19555905,
        -6.48646822,
        -4.72599161,
        3.63486848,
    ];

    let a1 = c2[0];
    let a2 = fac * (m_nu * c2[6] + c2[5]) + c2[1] + m_nu * c2[2];
    let a3 = fac * (m_nu * c2[8] + c2[7]) + c2[3] + m_nu * c2[4];

    let prefac = -t_recip.powi(2);
    let b_22 = prefac * c_mie * (a2 * m1 + a3 * m12 + a1);

    let c3 = [
        -0.07388219,
        3.41316599,
        -4.37999075,
        -3.49348043,
        2.84275469,
        -21.46347169,
        25.96844421,
        27.40251901,
        -25.51614374,
    ];

    let a1 = c3[0];
    let a2 = fac * (m_nu * c3[6] + c3[5]) + c3[1] + m_nu * c3[2];
    let a3 = fac * (m_nu * c3[8] + c3[7]) + c3[3] + m_nu * c3[4];

    let prefac = t_recip.powi(3) * (1.0 / 3.0);
    let b_23 = prefac * c_mie * (a2 * m1 + a3 * m12 + a1);

    let phi = (t_recip * 0.0208820673).tanh().powf(1.51646922);

    let par = [
        286.831547,
        -449.394468,
        237.934383,
        0.0,
        0.555105093,
        3.32488055,
        3.08699368,
        0.0,
        311.346254,
        -495.057496,
        185.343661,
        29.9125809,
        -0.314196403,
        -5.07772512,
        17.5503540,
        -40.2150869,
    ];

    let a0 = m1 * par[1] + m12 * par[2] + m123 * par[3] + par[0];
    let a1 = m1 * par[5] + m12 * par[6] + m123 * par[7] + par[4];
    let a2 = m1 * par[9] + m12 * par[10] + m123 * par[11] + par[8];
    let a3 = m1 * par[13] + m12 * par[14] + m123 * par[15] + par[12];

    let psi = -((t_recip * a1).exp() - 1.0) * a0 - ((t_recip * 2.0 * a3).exp() - 1.0) * a2;

    (b_21 + b_22 + b_23 + phi * psi / 6.0) * PI * m.powi(2) * sigma.powi(3)
}

fn residual_virial_coefficient<D: DualNum<f64> + Copy>(p: &UVParameters, x: &Array1<D>, t: D) -> D {
    let mut delta_b2bar = D::zero();
    for i in 0..p.ncomponents {
        let xi = x[i];
        for j in 0..p.ncomponents {
            delta_b2bar += xi
                * x[j]
                * p.sigma_ij[[i, j]].powi(3)
                * delta_b2(t / p.eps_k_ij[[i, j]], p.rep_ij[[i, j]], p.att_ij[[i, j]]);
        }
    }
    delta_b2bar
}

fn correlation_integral_and_b21_bh<D: DualNum<f64> + Copy>(
    rho_x: D,
    t_x: D,
    mean_field_constant: f64,
    mbar: D,
    rep: f64,
    att: f64,
    sigma3_x_quadratic: D,
    d_x: D,
) -> (D, D) {
    let [b_monomer, c_monomer, d_monomer] = coefficients_bh(rep, att, d_x);
    let m1 = (mbar - 1.0) / mbar;
    let m2 = (mbar - 2.0) / mbar;
    let m2_squared = m2 * m2;
    let m12 = m1 * m2;

    let fac = d_x - 1.0;
    let a2 = mbar.powf(-NU) * C_BH_CHAIN[1]
        + C_BH_CHAIN[0]
        + (mbar.powf(-NU) * C_BH_CHAIN[17] + C_BH_CHAIN[16]) * fac;
    let a3 = mbar.powf(-NU) * C_BH_CHAIN[3]
        + C_BH_CHAIN[2]
        + (mbar.powf(-NU) * C_BH_CHAIN[19] + C_BH_CHAIN[18]) * fac;
    let a = ((m1 * a2 + m12 * a3) - mie_prefactor(rep, att).recip() * mean_field_constant)
        * mbar.powf(-NU);

    let b2 = fac * C_BH_CHAIN[5] + C_BH_CHAIN[4];
    let b3 = fac * C_BH_CHAIN[7] + C_BH_CHAIN[6];
    let b = b_monomer - m1 * b2 - m12 * b3;

    let c2 = fac * C_BH_CHAIN[9] + C_BH_CHAIN[8];
    let c3 = fac * C_BH_CHAIN[11] + C_BH_CHAIN[10];
    let c = c_monomer - m1 * c2 - m12 * c3;

    let d2 = fac * C_BH_CHAIN[13] + C_BH_CHAIN[12];
    let d3 = fac * C_BH_CHAIN[15] + C_BH_CHAIN[14];
    let d = d_monomer - m1 * d2 - m12 * d3;
    let i_inter = (a + (b * rho_x + c * rho_x.powf(2.0)) / (d * rho_x + 1.0).powf(2.0))
        * mie_prefactor(rep, att);
    let b21_inter =
        a * mie_prefactor(rep, att) / t_x * mbar.powi(2) * sigma3_x_quadratic * 2.0 * PI;
    (i_inter, b21_inter)
}

/// U-fraction according to Barker-Henderson division.
/// Eq. 15
// fn u_fraction_bh<D: DualNum<f64> + Copy>(m: f64, reduced_density: D, beta: D) -> D {
//     let mut c = [0.0; 4];
//     for i in 0..4 {
//         c[i] = CU_BH[i][1] + CU_BH[i][0] / rep;
//     }
//     let a = 1.2187;
//     let b = 4.2773;
//     todo!();
//     // (activation(c[1], beta) * (-c[0] + 1.0) + c[0])
//     //     * (reduced_density.powf(a) * c[2] + reduced_density.powf(b) * c[3]).tanh()
// }

fn u_fraction_bh_chain<D: DualNum<f64> + Copy>(m: D, reduced_density: D, reduced_beta: D) -> D {
    let m1 = (m - 1.0) / m;
    let m12 = m1 * (m - 2.0) / m;
    // old
    // let c1 = m1 * 0.21474970219148440 + m12 * 8.4559397335985956e-002 + 0.74157730394994559;
    // let c2 = m1 * 0.71684577296450291 + m12 * 2.0107795080641869e-002 + 0.14102431817992339;
    // let c3 = m1 * 0.21231582074821093 + m12 * 0.49161177623333113 + 2.3966433172179635;
    // let c4 = -m1 * 2.2021047225140755 + m12 * 0.57208989230294349 + 4.698403261064168;

    // new (11.07.2023)
    let c1 = m1 * 0.21474970219148440 + m12 * 8.5778843117388259e-002 + 0.74157730394994559;
    let c2 = m1 * 0.71684577296450291 + m12 * 2.1873075377140071e-002 + 0.14102431817992339;
    let c3 = m1 * 0.21231582074821093 + m12 * 0.54998664928214569 + 2.3966433172179635;
    let c4 = -m1 * 2.2021047225140755 + m12 * 0.41251049251352323 + 4.698403261064168;
    let activation = reduced_beta * c2.sqrt() / (reduced_beta.powi(2) * c2 + 1.0).sqrt();
    (activation * (-c1 + 1.0) + c1) * (reduced_density * c3 + reduced_density.powf(3.0) * c4).tanh()
}

/// Activation function used for u-fraction according to Barker-Henderson division.
/// Eq. 16
fn activation<D: DualNum<f64> + Copy>(c: D, one_fluid_beta: D) -> D {
    one_fluid_beta * c.sqrt() / (one_fluid_beta.powi(2) * c + 1.0).sqrt()
}

fn one_fluid_properties<D: DualNum<f64> + Copy>(
    p: &UVParameters,
    x: &Array1<D>,
    t: D,
) -> (D, D, D, D, D, D, D, D) {
    let d = diameter_bh(p, t);
    // &p.sigma;
    let mut epsilon_k = D::zero();
    let mut sigma3_x_quadratic = D::zero();
    let mut rep = D::zero();
    let mut att = D::zero();
    let mut d_x_3 = D::zero();
    let mut mbar = D::zero();
    let mut mbar_quadratic = D::zero();
    for i in 0..p.ncomponents {
        let xi = x[i];
        let mi = p.m[i];
        mbar += x[i] * mi;
        d_x_3 += x[i] * mi * d[i].powi(3);
        for j in 0..p.ncomponents {
            mbar_quadratic += xi * x[j] * mi * p.m[j];
            let _y = xi * x[j] * mi * p.m[j] * p.sigma_ij[[i, j]].powi(3);
            sigma3_x_quadratic += _y;
            epsilon_k += _y * p.eps_k_ij[[i, j]];

            rep += xi * x[j] * p.rep_ij[[i, j]];
            att += xi * x[j] * p.att_ij[[i, j]];
        }
    }
    sigma3_x_quadratic /= mbar_quadratic;
    let sigma_x = ((x * &p.m * &p.sigma.mapv(|v| v.powi(3))).sum() / mbar).powf(1.0 / 3.0);
    let dx = (d_x_3 / mbar).powf(1.0 / 3.0) / sigma_x;

    (
        mbar,
        mbar_quadratic,
        rep,
        att,
        sigma_x,
        sigma3_x_quadratic,
        epsilon_k / sigma3_x_quadratic / mbar_quadratic,
        dx,
    )
}

fn coefficients_bh<D: DualNum<f64> + Copy>(rep: f64, att: f64, d: D) -> [D; 3] {
    let c11 = d.powf(-rep + 6.0) * ((D::one() * 2.0f64).powf(-rep + 3.0) - d.powf(rep - 3.0))
        / (-rep + 3.0)
        + (-d.powi(3) * 8.0 + 1.0) / 24.0;
    let c12 = (d.powf(-rep + 6.0) * ((D::one() * 2.0f64).powf(-rep + 4.0) - d.powf(rep - 4.0))
        / (-rep + 4.0)
        + (-d.powi(2) * 4.0 + 1.0) / 8.0)
        * -0.75;
    let c13 = (((d * 2.0).powf(-rep + 6.0) - 1.0) / (-rep + 6.0)
        - (d * 2.0).ln() * d.powf(-att + 6.0))
        / 16.0;
    let rep_inv = rep.recip();
    let c1 = (c11 + c12 + c13) * FRAC_PI_3 * 4.0;
    let c2 = -(-d + 1.0) * (rep_inv * C_BH[0][3] + C_BH[0][2]) + rep_inv * C_BH[0][1] + C_BH[0][0];
    let c3 = -(-d + 1.0) * (rep_inv * C_BH[1][3] + C_BH[1][2]) + rep_inv * C_BH[1][1] + C_BH[1][0];
    [c1, c2, c3]
}

fn delta_b2<D: DualNum<f64> + Copy>(reduced_temperature: D, rep: f64, att: f64) -> D {
    let rc = 5.0;
    let alpha = mean_field_constant(rep, att, rc);
    let yeff = y_eff(reduced_temperature, rep, att);
    -(yeff * (rc.powi(3) - 1.0) / 3.0 + reduced_temperature.recip() * alpha) * 2.0 * PI
}

fn y_eff<D: DualNum<f64> + Copy>(reduced_temperature: D, rep: f64, att: f64) -> D {
    // optimize: move this part to parameter initialization
    let rc = 5.0;
    let rs = 1.0;
    let c0 = 1.0
        - 3.0 * (mean_field_constant(rep, att, rs) - mean_field_constant(rep, att, rc))
            / (rc.powi(3) - rs.powi(3));
    let c1 = C2[0][0] + C2[0][1] / rep;
    let c2 = C2[1][0] + C2[1][1] / rep;
    let c3 = C2[2][0] + C2[2][1] / rep;

    let beta = reduced_temperature.recip();
    let beta_eff = beta * (-(beta * (beta * c2 + beta.powi(3) * c3 + c1) + 1.0).recip() * c0 + 1.0);
    beta_eff.exp() - 1.0
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::uvtheory::parameters::utils::test_parameters;
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn test_delta_b2_lj_chain() {
        let moles = arr1(&[2.0]);

        let reduced_temperature = 2.0;
        let reduced_density = 0.6;
        let reduced_volume = moles[0] / reduced_density;

        let p = test_parameters(2.0, 12.0, 6.0, 1.0, 1.0);
        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());
        let d = diameter_bh(&p, state.temperature);
        dbg!(&d);
        let delta_b2 =
            delta_b2_lj_chain(reduced_temperature, p.m[0], p.sigma[0], d[0] / p.sigma[0]);
        dbg!(&delta_b2);
        assert_eq!(1, 2);
    }

    #[test]
    fn test_helmholtz_energy_perturbation() {
        let moles = arr1(&[2.0]);
        let reduced_temperature = 2.0;
        let reduced_density = 0.04;
        let reduced_volume = moles[0] / reduced_density;

        let p = test_parameters(8.0, 12.0, 6.0, 1.0, 1.0);
        let pt = AttractivePerturbationBH {
            parameters: Arc::new(p.clone()),
        };
        let state = StateHD::new(
            reduced_temperature * p.epsilon_k[0],
            reduced_volume * p.sigma[0].powi(3),
            moles.clone(),
        );
        let a = pt.helmholtz_energy(&state) / moles.sum();
        dbg!(a);
        assert_eq!(1, 2)
    }
    //     #[test]
    //     fn test_attractive_perturbation() {
    //         // m = 12, t = 4.0, rho = 1.0
    //         let moles = arr1(&[2.0]);
    //         let reduced_temperature = 4.0;
    //         let reduced_density = 1.0;
    //         let reduced_volume = moles[0] / reduced_density;

    //         let p = methane_parameters(24.0, 6.0);
    //         let pt = AttractivePerturbationBH {
    //             parameters: Arc::new(p.clone()),
    //         };
    //         let state = StateHD::new(
    //             reduced_temperature * p.epsilon_k[0],
    //             reduced_volume * p.sigma[0].powi(3),
    //             moles.clone(),
    //         );
    //         let x = &state.molefracs;

    //         let (rep_x, att_x, sigma_x, weighted_sigma3_ij, epsilon_k_x, d_x) =
    //             one_fluid_properties(&p, &state.molefracs, state.temperature);
    //         let t_x = state.temperature / epsilon_k_x;
    //         let rho_x = state.partial_density.sum() * sigma_x.powi(3);

    //         let mean_field_constant_x = mean_field_constant(rep_x, att_x, 1.0);

    //         let i_bh = correlation_integral_bh(rho_x, mean_field_constant_x, rep_x, att_x, d_x);
    //         let delta_a1u = state.partial_density.sum() / t_x * i_bh * 2.0 * PI * weighted_sigma3_ij;
    //         dbg!(delta_a1u);
    //         //assert!(delta_a1u.re() == -1.1470186919354);
    //         assert_relative_eq!(delta_a1u.re(), -1.1470186919354, epsilon = 1e-12);

    //         let u_fraction_bh = u_fraction_bh(
    //             rep_x,
    //             state.partial_density.sum() * (x * &p.sigma.mapv(|s| s.powi(3))).sum(),
    //             t_x.recip(),
    //         );
    //         dbg!(u_fraction_bh);
    //         //assert!(u_fraction_bh.re() == 0.743451055308332);
    //         assert_relative_eq!(u_fraction_bh.re(), 0.743451055308332, epsilon = 1e-5);

    //         let b21u = delta_b12u(t_x, mean_field_constant_x, weighted_sigma3_ij);
    //         dbg!(b21u);
    //         assert!(b21u.re() / p.sigma[0].powi(3) == -0.949898568221715);

    //         let b2bar = residual_virial_coefficient(&p, x, state.temperature);
    //         dbg!(b2bar);
    //         assert_relative_eq!(
    //             b2bar.re() / p.sigma[0].powi(3),
    //             -1.00533412744652,
    //             epsilon = 1e-12
    //         );
    //         //assert!(b2bar.re() ==-1.00533412744652);

    //         //let a_test = state.moles.sum()
    //         //  * (delta_a1u + (-u_fraction_bh + 1.0) * (b2bar - b21u) * state.partial_density.sum());
    //         let a = pt.helmholtz_energy(&state) / moles[0];
    //         dbg!(a.re());
    //         //assert!(-1.16124062615291 == a.re())
    //         assert_relative_eq!(-1.16124062615291, a.re(), epsilon = 1e-5);
    //     }
}
