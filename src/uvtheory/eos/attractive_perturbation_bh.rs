//use super::attractive_perturbation_wca::one_fluid_properties;
use super::hard_sphere_bh::diameter_bh;
use crate::uvtheory::eos::CombinationRule;
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
    pub combination_rule: CombinationRule,
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
    /// Helmholtz energy for attractive perturbation
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        let p = &self.parameters;
        let x = &state.molefracs;
        let t = state.temperature;
        let density = state.partial_density.sum();
        let n = x.len();

        // vdw effective one fluid properties
        let one_fluid = OneFluidProperties::new(p, x, &state.partial_density, t);

        let mut delta_a1u = D::zero();
        let mut virial = D::zero();

        let d = diameter_bh(p, state.temperature);

        for i in 0..n {
            let xi = x[i];
            let mi = p.m[i];

            let ufraction_i = u_fraction_bh_chain(
                D::one() * mi,
                one_fluid.reduced_segment_density,
                //density * mi * p.sigma[i].powi(3),
                t.recip() * p.epsilon_k[i],
            );

            for j in 0..n {
                let alpha_ij = mean_field_constant_f64(p.rep_ij[[i, j]], p.att_ij[[i, j]], 1.0);

                let ufraction_j = u_fraction_bh_chain(
                    D::one() * p.m[j],
                    one_fluid.reduced_segment_density,
                    //density * p.m[j] * p.sigma[j].powi(3),
                    t.recip() * p.epsilon_k[j],
                );

                let m_ij: D = one_fluid.m;
                //  let m_ij = D::one() * 0.5 * (mi + p.m[j]);

                let psi_ij = match self.combination_rule {
                    CombinationRule::ArithmeticPhi => {
                        let ufraction_ij = (ufraction_i + ufraction_j) * 0.5;
                        -ufraction_ij + 1.0
                    }
                    CombinationRule::GeometricPhi => {
                        let ufraction_ij = (ufraction_i * ufraction_j).sqrt();
                        -ufraction_ij + 1.0
                    }
                    CombinationRule::GeometricPsi => {
                        ((-ufraction_i + 1.0) * (-ufraction_j + 1.0)).sqrt()
                    }
                    // A. one-fluid like u-fraction
                    CombinationRule::OneFluidPsi => {
                        let ufraction_x = u_fraction_bh_chain(
                            one_fluid.m,
                            one_fluid.reduced_segment_density,
                            one_fluid.reduced_temperature.recip(),
                        );
                        -ufraction_x + 1.0
                    }
                };

                // Intramolecular part
                if i == j {
                    if mi > 2.0 {
                        let (i_intra_ij, i_intra_ii_ldl) =
                            intramolecular_correlation_integral_and_ldl(
                                one_fluid.reduced_segment_density,
                                m_ij,
                                p.rep_ij[[i, j]],
                                p.att_ij[[i, j]],
                            );

                        delta_a1u += xi
                            * mi
                            * mi
                            * p.sigma_ij[[i, i]].powi(3)
                            * p.eps_k_ij[[i, i]]
                            * i_intra_ij
                            * 0.5
                            / PI;

                        let b21u_intra = (t.recip() * p.eps_k_ij[[i, i]])
                            * mi
                            * mi
                            * p.sigma_ij[[i, i]].powi(3)
                            * i_intra_ii_ldl;

                        match self.combination_rule {
                            CombinationRule::OneFluidPsi => {
                                let psi_x_intra = -u_fraction_bh_chain(
                                    D::one() * mi,
                                    one_fluid.reduced_segment_density,
                                    one_fluid.reduced_temperature.recip(),
                                ) + 1.0;
                                virial -= xi * psi_x_intra * b21u_intra
                            } // One fluid combining rule
                            _ => virial -= xi * (-ufraction_i + 1.0) * b21u_intra, //all other cases
                        };

                        //   dbg!(&phi_u);
                        //   dbg!(&i_intra_ij_ldl);
                        //   dbg!(&b2u1_intra);
                    }
                }

                // Intermolecular part
                let (i_inter_ij, i_inter_ij_ldl) = intermolecular_correlation_integral_and_ldl(
                    one_fluid.reduced_segment_density,
                    alpha_ij,
                    m_ij,
                    p.rep_ij[[i, j]],
                    p.att_ij[[i, j]],
                    one_fluid.reduced_diameter,
                );

                delta_a1u += xi
                    * x[j]
                    * mi
                    * p.m[j]
                    * p.sigma_ij[[i, j]].powi(3)
                    * p.eps_k_ij[[i, j]]
                    * i_inter_ij;

                let delta_b2 = delta_b2_lj_chain(
                    state.temperature / p.eps_k_ij[[i, j]],
                    0.5 * (mi + p.m[j]),
                    p.sigma_ij[[i, j]],
                    // (0.5 * (p.sigma_ij[[i, i]].powi(3) + p.sigma_ij[[j, j]].powi(3))).powf(1.0/3.0),
                    (d[i] + d[j]) / (p.sigma[i] + p.sigma[j]),
                );

                let b21u_inter = (t.recip() * p.eps_k_ij[[i, j]])
                    * mi
                    * p.m[j]
                    * p.sigma_ij[[i, j]].powi(3)
                    * i_inter_ij_ldl
                    * 2.0
                    * PI;

                virial += xi * x[j] * psi_ij * (delta_b2 - b21u_inter);

                //  dbg!(&i_inter_ij_ldl);
                //  dbg!(&virial);
                //  dbg!(&delta_b2);
                //  dbg!(&b2u1_inter);
            }
        }
        delta_a1u = density / state.temperature * delta_a1u * 2.0 * PI;

        //   dbg!(&delta_a1u);
        //   dbg!(&virial);
        //   assert!(1 == 2);

        state.moles.sum() * (delta_a1u + virial * density)
    }
}

// fn delta_b12u<D: DualNum<f64>>(t_x: D, mean_field_constant_x: D, weighted_sigma3_ij: D) -> D {
//     -mean_field_constant_x / t_x * 2.0 * PI * weighted_sigma3_ij
// }

fn delta_b2_lj_chain<D: DualNum<f64> + Copy>(
    reduced_temperature: D,
    m: f64,
    sigma: f64,
    d_reduced: D,
) -> D {
    let m1 = (m - 1.0) / m;
    let m12 = (m - 2.0) / m * m1;
    let m123 = (m - 3.0) / m * m12 / m;
    let c_mie = mie_prefactor(12.0, 6.0);
    let mean_field_constant = mean_field_constant(12.0, 6.0, 1.0) / c_mie;
    let fac = d_reduced - 1.0;
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

    // first fit (PPEPPD)
    // let par = [
    //     2.08820673e-02,
    //     1.51646922e+00,
    //     2.86831547e+02,
    //     5.55105093e-01,
    //     3.11346254e+02,
    //     -3.14196403e-01,
    //     -4.49394468e+02,
    //     3.32488055e+00,
    //     -4.95057496e+02,
    //     -5.07772512e+00,
    //     2.37934383e+02,
    //     3.08699368e+00,
    //     1.85343661e+02,
    //     1.75503540e+01,
    //     2.99125809e+01,
    //     -4.02150869e+01,
    // ];

    // new fit with zeno data up to m=50
    // let par = [
    //     2.90693336e-03,
    //     1.17478491e+00,
    //     1.66258506e+02,
    //     1.15603682e+00,
    //     4.15414295e+02,
    //     -3.56334057e-01,
    //     -2.15779185e+02,
    //     3.82507533e+00,
    //     -6.44597816e+02,
    //     -5.34793865e+00,
    //     1.52425427e+02,
    //     2.02375273e+00,
    //     2.32097107e+02,
    //     1.79423250e+01,
    //     -7.86470657e+00,
    //     -3.84378799e+01,
    // ];
    // new fit with zeno data up to m=100
    let par = [
        6.20033073e-03,
        1.31861108e+00,
        1.66887145e+02,
        1.05954078e+00,
        4.08454752e+02,
        -3.09041211e-01,
        -2.09556043e+02,
        3.56377030e+00,
        -6.46323571e+02,
        -5.37654949e+00,
        1.48106611e+02,
        2.33257323e+00,
        2.38659097e+02,
        1.91108418e+01,
        5.83123929e+01,
        -5.42997820e+01,
    ];

    let phi = (t_recip * par[0]).tanh().powf(par[1]);

    let a0 = m1 * par[6] + m12 * par[10] + par[2];
    let a1 = m1 * par[7] + m12 * par[11] + par[3];
    let a2 = m1 * par[8] + m12 * par[12] + m123 * par[14] + par[4];
    let a3 = m1 * par[9] + m12 * par[13] + m123 * par[15] + par[5];

    let psi = -((t_recip * a1).exp() - 1.0) * a0 - ((t_recip * 2.0 * a3).exp() - 1.0) * a2;

    (b_21 + b_22 + b_23 + phi * psi / 6.0) * PI * m.powi(2) * sigma.powi(3)
}

// fn residual_virial_coefficient<D: DualNum<f64> + Copy>(p: &UVParameters, x: &Array1<D>, t: D) -> D {
//     let mut delta_b2bar = D::zero();
//     for i in 0..p.ncomponents {
//         let xi = x[i];
//         for j in 0..p.ncomponents {
//             delta_b2bar += xi
//                 * x[j]
//                 * p.sigma_ij[[i, j]].powi(3)
//                 * delta_b2(t / p.eps_k_ij[[i, j]], p.rep_ij[[i, j]], p.att_ij[[i, j]]);
//         }
//     }
//     delta_b2bar
// }

fn intermolecular_correlation_integral_and_ldl<D: DualNum<f64> + Copy>(
    rho_reduced: D,
    mean_field_constant: f64,
    m: D,
    rep: f64,
    att: f64,
    d_reduced: D,
) -> (D, D) {
    let [b_monomer, c_monomer, d_monomer] = coefficients_bh(rep, att, d_reduced);
    let m1 = (m - 1.0) / m;
    let m2 = (m - 2.0) / m;
    let m12 = m1 * m2;

    let fac = d_reduced - 1.0;
    let a2 = m.powf(-NU) * C_BH_CHAIN[1]
        + C_BH_CHAIN[0]
        + (m.powf(-NU) * C_BH_CHAIN[17] + C_BH_CHAIN[16]) * fac;
    let a3 = m.powf(-NU) * C_BH_CHAIN[3]
        + C_BH_CHAIN[2]
        + (m.powf(-NU) * C_BH_CHAIN[19] + C_BH_CHAIN[18]) * fac;
    let a = ((m1 * a2 + m12 * a3) - mie_prefactor(rep, att).recip() * mean_field_constant)
        * m.powf(-NU);

    let b2 = fac * C_BH_CHAIN[5] + C_BH_CHAIN[4];
    let b3 = fac * C_BH_CHAIN[7] + C_BH_CHAIN[6];
    let b = b_monomer - m1 * b2 - m12 * b3;

    let c2 = fac * C_BH_CHAIN[9] + C_BH_CHAIN[8];
    let c3 = fac * C_BH_CHAIN[11] + C_BH_CHAIN[10];
    let c = c_monomer - m1 * c2 - m12 * c3;

    let d2 = fac * C_BH_CHAIN[13] + C_BH_CHAIN[12];
    let d3 = fac * C_BH_CHAIN[15] + C_BH_CHAIN[14];
    let d = d_monomer - m1 * d2 - m12 * d3;

    let i_inter = (a
        + (b * rho_reduced + c * rho_reduced.powf(2.0)) / (d * rho_reduced + 1.0).powf(2.0))
        * mie_prefactor(rep, att);

    let i_inter_ldl = a * mie_prefactor(rep, att);

    (i_inter, i_inter_ldl)
}

fn intramolecular_correlation_integral_and_ldl<D: DualNum<f64> + Copy>(
    rho_reduced: D,
    m: D,
    rep: f64,
    att: f64,
) -> (D, D) {
    let cmie = mie_prefactor(rep, att);
    let m2 = (m - 2.0) / m;

    let m2_squared = m2 * m2;
    let a_intra = m2_squared * C_BH_CHAIN_INTRA[0];
    let b_intra = m2_squared * C_BH_CHAIN_INTRA[1];
    let c_intra = m2_squared * C_BH_CHAIN_INTRA[2];

    let fac = m.powd(m.recip() * C_BH_CHAIN_INTRA[3]);

    let i_intra = fac * cmie * (a_intra + b_intra * rho_reduced + c_intra * rho_reduced.powi(2));
    let i_intra_ldl = fac * cmie * a_intra;

    (i_intra, i_intra_ldl)
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
// fn activation<D: DualNum<f64> + Copy>(c: D, one_fluid_beta: D) -> D {
//     one_fluid_beta * c.sqrt() / (one_fluid_beta.powi(2) * c + 1.0).sqrt()
// }

// fn one_fluid_properties<D: DualNum<f64> + Copy>(
//     p: &UVParameters,
//     x: &Array1<D>,
//     t: D,
// ) -> (D, D, D, D, D, D, D, D) {
//     let d = diameter_bh(p, t);
//     // &p.sigma;
//     let mut epsilon_k = D::zero();
//     let mut sigma3_x_quadratic = D::zero();
//     let mut rep = D::zero();
//     let mut att = D::zero();
//     let mut d_x_3 = D::zero();
//     let mut mbar = D::zero();
//     let mut mbar_quadratic = D::zero();
//     for i in 0..p.ncomponents {
//         let xi = x[i];
//         let mi = p.m[i];
//         mbar += x[i] * mi;
//         d_x_3 += x[i] * mi * d[i].powi(3);
//         for j in 0..p.ncomponents {
//             mbar_quadratic += xi * x[j] * mi * p.m[j];
//             let _y = xi * x[j] * mi * p.m[j] * p.sigma_ij[[i, j]].powi(3);
//             sigma3_x_quadratic += _y;
//             epsilon_k += _y * p.eps_k_ij[[i, j]];

//             rep += xi * x[j] * p.rep_ij[[i, j]];
//             att += xi * x[j] * p.att_ij[[i, j]];
//         }
//     }
//     sigma3_x_quadratic /= mbar_quadratic;
//     let sigma_x = ((x * &p.m * &p.sigma.mapv(|v| v.powi(3))).sum() / mbar).powf(1.0 / 3.0);
//     let dx = (d_x_3 / mbar).powf(1.0 / 3.0) / sigma_x;

//     (
//         mbar,
//         mbar_quadratic,
//         rep,
//         att,
//         sigma_x,
//         sigma3_x_quadratic,
//         epsilon_k / sigma3_x_quadratic / mbar_quadratic,
//         dx,
//     )
// }

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

// fn delta_b2<D: DualNum<f64> + Copy>(reduced_temperature: D, rep: f64, att: f64) -> D {
//     let rc = 5.0;
//     let alpha = mean_field_constant(rep, att, rc);
//     let yeff = y_eff(reduced_temperature, rep, att);
//     -(yeff * (rc.powi(3) - 1.0) / 3.0 + reduced_temperature.recip() * alpha) * 2.0 * PI
// }

// fn y_eff<D: DualNum<f64> + Copy>(reduced_temperature: D, rep: f64, att: f64) -> D {
//     // optimize: move this part to parameter initialization
//     let rc = 5.0;
//     let rs = 1.0;
//     let c0 = 1.0
//         - 3.0 * (mean_field_constant(rep, att, rs) - mean_field_constant(rep, att, rc))
//             / (rc.powi(3) - rs.powi(3));
//     let c1 = C2[0][0] + C2[0][1] / rep;
//     let c2 = C2[1][0] + C2[1][1] / rep;
//     let c3 = C2[2][0] + C2[2][1] / rep;

//     let beta = reduced_temperature.recip();
//     let beta_eff = beta * (-(beta * (beta * c2 + beta.powi(3) * c3 + c1) + 1.0).recip() * c0 + 1.0);
//     beta_eff.exp() - 1.0
// }

#[cfg(test)]
mod test {
    use super::*;
    use crate::uvtheory::eos::CombinationRule;
    use crate::uvtheory::parameters::utils::test_parameters;
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn test_delta_b2_lj_chain() {
        let moles = arr1(&[2.0]);

        let reduced_temperature = 2.0;
        let reduced_density = 0.6;
        let reduced_volume = moles[0] / reduced_density;

        let p = test_parameters(2.0, 12.0, 6.0, 1.5, 1.5);
        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());
        let d = diameter_bh(&p, state.temperature);
        dbg!(&d);
        let delta_b2 =
            delta_b2_lj_chain(reduced_temperature, p.m[0], p.sigma[0], d[0] / p.sigma[0]);
        dbg!(&delta_b2);
        dbg!(&state.temperature);
        dbg!(&reduced_temperature);
        dbg!(&p.epsilon_k[0]);
        assert_eq!(delta_b2, -36.83548303945616);
    }

    #[test]
    fn test_helmholtz_energy_perturbation() {
        let moles = arr1(&[2.0]);
        let reduced_temperature = 2.0;
        let reduced_density = 0.04;
        let reduced_volume = moles[0] / reduced_density;

        let p = test_parameters(1.0, 12.0, 6.0, 1.0, 1.0);
        let pt = AttractivePerturbationBH {
            parameters: Arc::new(p.clone()),
            combination_rule: CombinationRule::OneFluidPsi,
        };
        let state = StateHD::new(
            reduced_temperature * p.epsilon_k[0],
            reduced_volume * p.sigma[0].powi(3),
            moles.clone(),
        );
        let a = pt.helmholtz_energy(&state) / moles.sum();
        dbg!(a);
        assert_eq!(a, -0.12670343489444227)
    }
}
