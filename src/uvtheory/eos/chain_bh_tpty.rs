use super::hard_sphere_bh::{diameter_bh, packing_fraction};
use crate::uvtheory::parameters::*;
use feos_core::{HelmholtzEnergyDual, StateHD};
use ndarray::Array1;
use num_dual::DualNum;
use std::fmt;
use std::{f64::consts::FRAC_PI_6, f64::consts::PI, sync::Arc};

const BH_DIAMETER: [f64; 8] = [
    0.852987920795915,
    -0.128229846701676,
    0.833664689185409,
    0.0240477795238045,
    0.0177618321999164,
    0.127015906854396,
    -0.528941139160234,
    -0.147289922797747,
];

#[derive(Clone)]
pub struct ChainBH {
    pub parameters: Arc<UVParameters>,
}

impl fmt::Display for ChainBH {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Reference Perturbation")
    }
}

impl<D: DualNum<f64> + Copy> HelmholtzEnergyDual<D> for ChainBH {
    /// Helmholtz energy for perturbation reference (Mayer-f), eq. 29
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        let p = &self.parameters;
        let n = p.sigma.len();
        let x = &state.molefracs;
        let m = &p.m;
        let d = diameter_bh(p, state.temperature);
        let eta = packing_fraction(m, &state.partial_density, &d);
        let zms = (-eta + 1.0).recip();
        let zms2 = zms * zms;
        let z2t = (x * m * d.mapv(|di| di.powi(2))).sum() * FRAC_PI_6;
        let z2 = state.partial_density.sum() * z2t;
        let mut a = D::zero();
        for i in 0..n {
            a += -x[i]
                * (m[i] - 1.0)
                * (g_mspt(
                    state.partial_density.sum(),
                    x,
                    m,
                    &d,
                    i,
                    d[i].recip() * p.sigma[i],
                ))
                .ln();

            let para = [0.35749, -1.2591, 0.84841, -1.3420, 3.2487, -2.5243];

            let fac3 = (m[i] - 2.0) / m[i];

            let a3 = para[0] + para[1] * fac3 + para[2] * fac3.powi(2);
            let b3 = para[3] + para[4] * fac3 + para[5] * fac3.powi(2);
            let ynn = zms + d[i] * a3 * z2 * zms2 + (d[i] * z2).powi(2) * b3 * zms.sqrt();

            if p.m[i] > 2.0 {
                a += x[i] * (m[i] - 2.0) * fac3 * ynn.ln();
                //    B20 = B20 - x(i) (mseg(i)-2.0)fac3 *z3t
            }
        }

        state.moles.sum() * a
    }
}

/// Radial distribution function
///  The cavity-correlation function of two (heteronuclear) HS cavities
///  of index i and j, at reduced distance xx, in a hard-sphere fluid
///  mixture, according to the Modified-Scaled-Particle-Theory (MSPT)
///  of Boublik (1986).
fn g_mspt<D: DualNum<f64> + Copy>(
    density: D,
    x: &Array1<D>,
    m: &Array1<f64>,
    d: &Array1<D>,
    i: usize,
    xx: D,
) -> D {
    let mbar = (x * m).sum();
    let rho_s = density * mbar;
    let x_s = (x * m).mapv(|e| e / mbar);

    let nu = rho_s * FRAC_PI_6 * (&x_s * d.mapv(|di| di.powi(3))).sum();
    let s = rho_s * PI * (&x_s * d.mapv(|di| di.powi(2))).sum();
    let q = rho_s * (&x_s * d.mapv(|di| di.powi(2))).sum() / 4.0;
    let r = rho_s * (&x_s * d).sum() * 0.5;

    let v_i = d[i].powi(3) * FRAC_PI_6;
    let s_i = d[i].powi(2) * PI;
    let r_i = d[i] * 0.5;

    let muhs_i = -(-nu + 1.0).ln()
        + (r_i * s + s_i * r + v_i * rho_s) / (-nu + 1.0)
        + (r_i.powi(2) * s.powi(2) + s_i * q * s * 2.0 + v_i * r * s * 6.0)
            / ((-nu + 1.0).powi(2) * 6.0)
        + (v_i * q * s.powi(2) * (-nu / 3.0 + 2.0)) / ((-nu + 1.0).powi(3) * 9.0);

    // !-------------------
    // ! chemical potential of Hard-Dimer,
    // ! formed from HS i and j at bond-lengt xx=2.0*l/(di+dj),
    // ! infinitely diluted in HS mixture
    // !-------------------
    let v_ij =
        (-xx.powi(3) / 8.0 * 2.0.powi(3) + 2.0 + xx * 0.75 * 4.0) * d[i].powi(3) * (PI / 12.0);
    let s_ij = (xx + 1.0) * d[i].powi(2) * PI;
    let r_ij = (xx / 2.0 + 1.0) * d[i] * 0.5;

    dbg!(v_ij);
    dbg!(s_ij);
    dbg!(r_ij);

    let muhd_ij = -(-nu + 1.0).ln()
        + (r_ij * s + s_ij * r + v_ij * rho_s) / (-nu + 1.0)
        + (r_ij.powi(2) * s.powi(2) + s_ij * q * s * 2.0 + v_ij * r * s * 6.0)
            / ((-nu + 1.0).powi(2) * 6.0)
        + (v_ij * q * s.powi(2) * (-nu / 3.0 + 2.0)) / ((-nu + 1.0).powi(3) * 9.0);

    // !-------------------
    // ! cavity function of HS i and HS j at distance L,
    // ! in HS mixture
    // !-------------------
    (muhs_i * 2.0 - muhd_ij).exp()
}

fn diameter_bh_chain_ij<D: DualNum<f64> + Copy>(
    m_i: f64,
    m_j: f64,
    epsilon_k_ij: f64,
    temperature: D,
) -> D {
    let reduced_temperature = temperature / epsilon_k_ij;

    let m_ij = 0.5 * (m_i + m_j);
    let fac1 = (m_ij - 1.0) / m_ij;
    let fac2 = fac1 * (m_ij - 2.0) / m_ij;
    let a_1 = fac1 * BH_DIAMETER[1] + fac2 * BH_DIAMETER[2] + BH_DIAMETER[0];
    let a_2 = fac1 * BH_DIAMETER[4] + fac2 * BH_DIAMETER[5] + BH_DIAMETER[3];
    let fac3 = (1.0 / 24.0) * (1.0 + BH_DIAMETER[6] * fac1 + BH_DIAMETER[7] * fac2);

    (reduced_temperature * a_1 + reduced_temperature.powi(2) * a_2 + 1.0)
        .recip()
        .powf(fac3)
}

/// Total second virial coefficient of a mixture of Barker Henderson LJ chains.
///
/// # Note
///
/// Not tested for mixtures.
pub fn b20_lj_chain<D: DualNum<f64> + Copy>(
    parameters: &UVParameters,
    x: &Array1<D>,
    temperature: D,
) -> D {
    let m = &parameters.m;
    let n = m.len();
    let dimensionless_bond_length = 1.0;
    let dimensionless_bond_length_squared = dimensionless_bond_length.powi(2);
    let orientational_average = 0.0006736 * dimensionless_bond_length
        + 0.7443 * dimensionless_bond_length_squared
        + 0.01482 * dimensionless_bond_length * dimensionless_bond_length_squared
        - 0.06578 * dimensionless_bond_length_squared.powi(2);
    let mut b20 = D::zero();
    for i in 0..n {
        let x_i = x[i];
        let m_i = m[i];
        let gamma_i = 0.2079 + 0.6388 / m_i.powi(3);
        for j in 0..n {
            let gamma_j = 0.2079 + 0.6388 / m[j].powi(3);
            let m_ij = 0.5 * (m_i + m[j]);
            let d_ij = diameter_bh_chain_ij(m_i, m[j], parameters.eps_k_ij[[i, j]], temperature);
            b20 += x_i
                * x[j]
                * d_ij.powi(3)
                * parameters.sigma_ij[[i, j]].powi(3)
                * 4.0
                * FRAC_PI_6
                * (1.0
                    + (m_ij - 1.0)
                        * (1.5 * dimensionless_bond_length
                            - (1.0 / 8.0)
                                * dimensionless_bond_length
                                * dimensionless_bond_length_squared)
                    + 0.5
                        * orientational_average
                        * (m_i - 1.0).powf(1.0 - 0.5 * gamma_i * dimensionless_bond_length)
                        * (m[j] - 1.0).powf(1.0 - 0.5 * gamma_j * dimensionless_bond_length));
        }
    }
    b20
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::uvtheory::parameters::utils::test_parameters;
    use feos_core::parameter::{Identifier, Parameter, PureRecord};
    use ndarray::arr1;

    #[test]
    fn test_g_mspt() {
        let moles = arr1(&[2.0]);

        let reduced_temperature = 2.0;
        let reduced_density = 0.6;
        let reduced_volume = moles[0] / reduced_density;

        let p = test_parameters(1.0, 12.0, 6.0, 1.0, 1.0);
        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());

        let d = diameter_bh(&p, state.temperature);
        let g_mspt = g_mspt(
            state.partial_density.sum(),
            &state.molefracs,
            &p.m,
            &d,
            0,
            p.sigma[0] / d[0],
        );
        assert_eq!(g_mspt, 2.0827648438059994);
    }

    #[test]
    fn test_a_chain() {
        let moles = arr1(&[2.0]);

        let reduced_temperature = 2.0;
        let reduced_density = 0.6;
        let reduced_volume = moles[0] / reduced_density;

        let p = test_parameters(2.0, 12.0, 6.0, 1.0, 1.0);
        let chain = ChainBH {
            parameters: Arc::new(p.clone()),
        };
        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());

        let a_chain = chain.helmholtz_energy(&state) / moles.sum();
        dbg!(&a_chain);
        assert_eq!(a_chain, 2.0827648438059994);
    }

    #[test]
    fn test_g_mspt_mixture() {
        let moles = arr1(&[0.6, 0.4]) * 2.0;

        let reduced_temperature = 2.0;
        let reduced_density = 0.6;
        let reduced_volume = moles.sum() / reduced_density;

        let p = UVParameters::new_binary(
            vec![
                PureRecord::new(
                    Identifier::default(),
                    1.0,
                    UVRecord::new(
                        2.0, 12.0, 6.0, 1.0, 1.0, None, None, None, None, None, None, None,
                    ),
                ),
                PureRecord::new(
                    Identifier::default(),
                    1.0,
                    UVRecord::new(
                        1.0, 12.0, 6.0, 1.0, 1.0, None, None, None, None, None, None, None,
                    ),
                ),
            ],
            None,
        )
        .unwrap();
        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());

        let d = diameter_bh(&p, state.temperature);
        let g_mspt_0 = g_mspt(
            state.partial_density.sum(),
            &state.molefracs,
            &p.m,
            &d,
            0,
            p.sigma[0] / d[0],
        );
        let g_mspt_1 = g_mspt(
            state.partial_density.sum(),
            &state.molefracs,
            &p.m,
            &d,
            1,
            p.sigma[1] / d[1],
        );
        dbg!(&g_mspt_0);
        assert_eq!(g_mspt_0, g_mspt_1 * 0.9);
    }

    #[test]
    fn test_g_mspt_mixture2() {
        let moles = arr1(&[0.6, 0.4]) * 2.0;

        let reduced_temperature = 2.0;
        let reduced_density = 0.2;
        let reduced_volume = moles.sum() / reduced_density;

        let p = UVParameters::new_binary(
            vec![
                PureRecord::new(
                    Identifier::default(),
                    1.0,
                    UVRecord::new(
                        2.0, 12.0, 6.0, 1.0, 1.0, None, None, None, None, None, None, None,
                    ),
                ),
                PureRecord::new(
                    Identifier::default(),
                    1.0,
                    UVRecord::new(
                        1.0, 12.0, 6.0, 2.0, 6.0, None, None, None, None, None, None, None,
                    ),
                ),
            ],
            None,
        )
        .unwrap();
        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());

        let d = diameter_bh(&p, state.temperature);
        dbg!(&d);
        let g_mspt_0 = g_mspt(
            state.partial_density.sum(),
            &state.molefracs,
            &p.m,
            &d,
            0,
            p.sigma[0] / d[0],
        );
        let g_mspt_1 = g_mspt(
            state.partial_density.sum(),
            &state.molefracs,
            &p.m,
            &d,
            1,
            p.sigma[1] / d[1],
        );
        assert_eq!(g_mspt_0, g_mspt_1);
    }

    #[test]
    fn test_a_chain_mixture() {
        let moles = arr1(&[0.6, 0.4]) * 2.0;

        let reduced_temperature = 2.0;
        let reduced_density = 0.2;
        let reduced_volume = moles.sum() / reduced_density;

        let p = UVParameters::new_binary(
            vec![
                PureRecord::new(
                    Identifier::default(),
                    1.0,
                    UVRecord::new(
                        2.0, 12.0, 6.0, 1.0, 1.0, None, None, None, None, None, None, None,
                    ),
                ),
                PureRecord::new(
                    Identifier::default(),
                    1.0,
                    UVRecord::new(
                        1.0, 12.0, 6.0, 2.0, 6.0, None, None, None, None, None, None, None,
                    ),
                ),
            ],
            None,
        )
        .unwrap();
        let chain = ChainBH {
            parameters: Arc::new(p.clone()),
        };
        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());

        let a = chain.helmholtz_energy(&state) / moles.sum();
        dbg!(&a);
        assert_eq!(a, 1.0);
    }

    #[test]
    fn test_b20_lj_chain() {
        let moles = arr1(&[2.0]);

        let reduced_temperature = 4.0;
        let reduced_density = 1.0;
        let reduced_volume = moles[0] / reduced_density;

        let p = test_parameters(3.0, 12.0, 6.0, 1.0, 1.0);
        let pt = ChainBH {
            parameters: Arc::new(p.clone()),
        };
        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());
        let b20 = b20_lj_chain(&p, &state.molefracs, state.temperature);
        dbg!(&b20);
        assert_eq!(b20, -0.0611105573289734);
    }
}
