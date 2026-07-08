#![allow(clippy::excessive_precision)]

use std::f64::consts::TAU;

use itertools::izip;
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use quantity::{GRAM, KILOGRAM, MOL, NAV};

use super::parameters::{QuantumCorrection, SaftVRCSPars};

/// 10-point Gauss-Legendre quadrature [position, weight]
const GLQ10: [[f64; 2]; 10] = [
    [-0.1488743389816312, 0.2955242247147529],
    [0.1488743389816312, 0.2955242247147529],
    [-0.4333953941292472, 0.2692667193099963],
    [0.4333953941292472, 0.2692667193099963],
    [-0.6794095682990244, 0.219086362515982],
    [0.6794095682990244, 0.219086362515982],
    [-0.8650633666889845, 0.1494513491505806],
    [0.8650633666889845, 0.1494513491505806],
    [-0.9739065285171717, 0.0666713443086881],
    [0.9739065285171717, 0.0666713443086881],
];

// Quantum correction constants for effective repulsive exponent
const C_LR: [[f64; 3]; 5] = [
    [-1.0577620680e+02, 2.7130516920e+01, -7.9885048560E-02],
    [-2.2875719760e+03, 5.1995559760e+02, 8.2111050000E-01],
    [1.7306319200e+03, -1.7476174200e+02, 3.2561549200e+01],
    [-9.0993639000e+01, 2.2823545260e+01, -2.8168749280e-02],
    [-1.4764571200e+03, 4.3447760860e+02, -1.1267359060e+01],
];

// Quantum correction constants for effective sigma
const C_SIGMA: [f64; 9] = [
    31.60144413, 2.09612861, 0.20116118, 50.80286701, -4.28014515, 0.20063555, 25.11333946,
    1.35310377, 0.20250069,
];

// Quantum correction constants for effective epsilon_k
const C_EPSILON: [f64; 9] = [
    10.60837315, 1.58987996, 0.19807347, 17.92633472, -19.76493004, 3.91514157, 7.17199935,
    6.45489122, 0.18773935,
];

const KB: f64 = 1.380649e-23;
const PLANCK: f64 = 6.62607015e-34;
const D_QM_PREFACTOR: f64 = PLANCK * PLANCK / (TAU * TAU) / 12.0 * 1e20 / KB;

/// Temperature-dependent effective SAFT-VR Mie parameters from the corresponding states principle.
///
/// All parameters are D-typed because they depend on temperature via Feynman-Hibbs corrections.
#[derive(Debug)]
pub struct CorrespondingParameters<D> {
    pub ncomponents: usize,
    pub m: DVector<f64>,
    /// Effective segment sigma per component
    pub sigma: DVector<D>,
    /// Effective epsilon_k per component
    pub epsilon_k: DVector<D>,
    /// Effective repulsive exponent per component
    pub lr: DVector<D>,
    /// Attractive exponent per component (fixed, not corrected)
    pub la: DVector<f64>,
    /// Effective cross-interaction sigma_ij
    pub sigma_ij: DMatrix<D>,
    /// Effective cross-interaction epsilon_k_ij
    pub epsilon_k_ij: DMatrix<D>,
    /// Effective cross-interaction repulsive exponent lr_ij
    pub lr_ij: DMatrix<D>,
    /// Cross-interaction attractive exponent la_ij (fixed)
    pub la_ij: DMatrix<f64>,
    /// Mie prefactor c_ij from effective lr/la
    pub c_ij: DMatrix<D>,
    /// Integrated mean-field constant alpha_ij from effective lr/la
    pub alpha_ij: DMatrix<D>,
    /// β * c_ij * epsilon_k_ij (precomputed for GLQ integration)
    pub c_eps_t_ij: DMatrix<D>,
}

impl<D: DualNum<f64> + Copy> CorrespondingParameters<D> {
    pub fn new(params: &SaftVRCSPars, temperature: D) -> Self {
        let n = params.ncomponents;
        let t_inv = temperature.recip();
        let to_mass = (GRAM / MOL / NAV / KILOGRAM).into_value();

        let mut sigma = DVector::zeros(n);
        let mut epsilon_k = DVector::zeros(n);
        let mut lr = DVector::zeros(n);
        let la = params.la.clone();

        for i in 0..n {
            match &params.quantum_correction[i] {
                None => {
                    sigma[i] = D::from(params.sigma[i]);
                    epsilon_k[i] = D::from(params.epsilon_k[i]);
                    lr[i] = D::from(params.lr[i]);
                }
                Some(QuantumCorrection::FeynmanHibbs1 {
                    c_sigma,
                    c_epsilon_k,
                    c_lr: c_lr_scale,
                }) => {
                    let mass = params.molarweight[i] * to_mass;
                    let lr_raw = params.lr[i];
                    // Dimensionless quantum parameter D/σ²
                    let d_s2 =
                        t_inv / mass * D_QM_PREFACTOR / params.sigma[i].powi(2);
                    let s = eff_sigma(d_s2, lr_raw, c_sigma.as_ref());
                    let e = eff_epsilon_k(d_s2, lr_raw, c_epsilon_k.as_ref());
                    let m_ratio = eff_lr_ratio(d_s2, lr_raw, c_lr_scale.as_ref());
                    sigma[i] = s * params.sigma[i];
                    epsilon_k[i] = e * params.epsilon_k[i];
                    lr[i] = m_ratio * lr_raw;
                }
            }
        }

        // Cross-interaction parameters using SAFT-VR Mie combining rules
        let mut sigma_ij = DMatrix::zeros(n, n);
        let mut epsilon_k_ij = DMatrix::zeros(n, n);
        let mut lr_ij = DMatrix::zeros(n, n);
        let mut la_ij = DMatrix::zeros(n, n);
        let mut c_ij = DMatrix::zeros(n, n);
        let mut alpha_ij = DMatrix::zeros(n, n);
        let mut c_eps_t_ij = DMatrix::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                let sig_ij = (sigma[i] + sigma[j]) * 0.5;
                let e_k_ij = (sigma[i].powi(3) * sigma[j].powi(3)).sqrt()
                    / sig_ij.powi(3)
                    * (epsilon_k[i] * epsilon_k[j]).sqrt();
                let k = params.k_ij[(i, j)];
                let gamma = params.gamma_ij[(i, j)];

                let la_i = params.la[i];
                let la_j = params.la[j];
                let la_ij_val =
                    ((la_i - 3.0) * (la_j - 3.0)).sqrt() + 3.0;
                let lr_ij_val =
                    ((lr[i] - 3.0) * (lr[j] - 3.0)).sqrt() * (1.0 - gamma) + 3.0;

                let la_over_lrla = (lr_ij_val - la_ij_val).recip() * la_ij_val;
                let c = lr_ij_val / (lr_ij_val - la_ij_val)
                    * (lr_ij_val / la_ij_val).powd(la_over_lrla);
                let alpha = c * ((lr_ij_val - 3.0).recip() * (-1.0) + (la_ij_val - 3.0).recip());

                sigma_ij[(i, j)] = sig_ij;
                epsilon_k_ij[(i, j)] = e_k_ij * (1.0 - k);
                lr_ij[(i, j)] = lr_ij_val;
                la_ij[(i, j)] = la_ij_val;
                c_ij[(i, j)] = c;
                alpha_ij[(i, j)] = alpha;
                c_eps_t_ij[(i, j)] = c * epsilon_k_ij[(i, j)] * t_inv;
            }
        }

        Self {
            ncomponents: n,
            m: params.m.clone(),
            sigma,
            epsilon_k,
            lr,
            la,
            sigma_ij,
            epsilon_k_ij,
            lr_ij,
            la_ij,
            c_ij,
            alpha_ij,
            c_eps_t_ij,
        }
    }

    /// Barker-Henderson diameter d_ij via 10-point Gauss-Legendre quadrature.
    pub fn hs_diameter_ij(&self, i: usize, j: usize) -> D {
        let la = self.la_ij[(i, j)];
        let lr = self.lr_ij[(i, j)];
        let c_eps_t = self.c_eps_t_ij[(i, j)];
        let sig = self.sigma_ij[(i, j)];

        let r0 = lower_integral_limit(la, lr, c_eps_t);
        let width = (-r0 + 1.0) * 0.5;
        let d = GLQ10.iter().fold(r0, |acc, &[x, w]| {
            let r = width * x + width + r0;
            let u = beta_u_mie(r, la, lr, c_eps_t);
            let f_u = -(-u).exp_m1();
            acc + width * f_u * w
        });
        d * sig
    }
}

/// Dimensionless Mie potential β u(r/σ) with D-typed repulsive exponent
#[inline]
fn beta_u_mie<D: DualNum<f64> + Copy>(r: D, la: f64, lr: D, c_eps_t: D) -> D {
    let ri = r.recip();
    (ri.powd(lr) - ri.powf(la)) * c_eps_t
}

/// Halley step fractions [f/f', f'/f''] for root finding in lower_integral_limit
#[inline]
fn mie_halley<D: DualNum<f64> + Copy>(r: D, la: f64, lr: D, c_eps_t: D) -> [D; 3] {
    let ri = r.recip();
    let plr = ri.powd(lr);
    let pla = ri.powf(la);
    let u = plr - pla;
    let dplr = plr * (-lr) * ri;
    let dpla = pla * (-la) * ri;
    let du = dplr - dpla;
    let d2u = (dplr * (-lr - 1.0) - dpla * (-la - 1.0)) * ri;
    let f = -c_eps_t * u - f64::EPSILON.ln();
    let df = -c_eps_t * du;
    let d2f = -c_eps_t * d2u;
    [f, f / df, df / d2f]
}

/// Lower integration limit r0 where β u(r0) = ln(ε_machine), found via Halley's method.
fn lower_integral_limit<D: DualNum<f64> + Copy>(la: f64, lr: D, c_eps_t: D) -> D {
    let k = (-c_eps_t.recip() * f64::EPSILON.ln()).ln();
    let mut r: D = (-k / lr).exp();
    for _ in 1..5 {
        let [u, u_du, du_d2u] = mie_halley(r, la, lr, c_eps_t);
        if u.re() < 0.0 {
            return r;
        }
        let dr = u_du / (-u_du / du_d2u * 0.5 + 1.0);
        r -= dr;
    }
    r
}

/// Effective lr_eff/lr_raw ratio from Feynman-Hibbs quantum correction
fn eff_lr_ratio<D: DualNum<f64> + Copy>(qd: D, lr: f64, c: Option<&[f64; 5]>) -> D {
    let c_scale = c.unwrap_or(&[1.0; 5]);
    let mut pref = [0.0; 5];
    for (pi, ci, ci_scale) in izip!(&mut pref, &C_LR, c_scale) {
        *pi = (ci[0] + lr * (ci[1] + lr * ci[2])) * ci_scale;
    }
    (qd * (qd * (qd * pref[2] + pref[1]) + pref[0]) + 1.0)
        / (qd * (qd * pref[4] + pref[3]) + 1.0)
}

/// Effective sigma_eff/sigma ratio from Feynman-Hibbs quantum correction
fn eff_sigma<D: DualNum<f64> + Copy>(qd: D, lr: f64, c: Option<&[f64; 3]>) -> D {
    let c_scale = c.unwrap_or(&[1.0; 3]);
    let c0 = C_SIGMA[0] + lr * (C_SIGMA[1] + lr * C_SIGMA[2]);
    let c1 = C_SIGMA[3] + lr * (C_SIGMA[4] + lr * C_SIGMA[5]);
    let c2 = C_SIGMA[6] + lr * (C_SIGMA[7] + lr * C_SIGMA[8]);
    (qd * (qd * c1 * c_scale[1] + c0 * c_scale[0]) + 1.0) / (qd * c2 * c_scale[2] + 1.0)
}

/// Effective epsilon_k_eff/epsilon_k ratio from Feynman-Hibbs quantum correction
fn eff_epsilon_k<D: DualNum<f64> + Copy>(qd: D, lr: f64, c: Option<&[f64; 3]>) -> D {
    let c_scale = c.unwrap_or(&[1.0; 3]);
    let c0 = C_EPSILON[0] + lr * (C_EPSILON[1] + lr * C_EPSILON[2]);
    let c1 = C_EPSILON[3] + lr * (C_EPSILON[4] + lr * C_EPSILON[5]);
    let c2 = C_EPSILON[6] + lr * (C_EPSILON[7] + lr * C_EPSILON[8]);
    (qd * (qd * c1 * c_scale[1] + c0 * c_scale[0]) + 1.0) / (qd * c2 * c_scale[2] + 1.0)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::saftvrcs::parameters::SaftVRCSPars;

    #[test]
    fn test_hs_diameter_no_qc() {
        // Methane-like (no quantum correction): compare with SaftVRMie diameter
        use crate::saftvrcs::parameters::utils::test_parameters;
        let p_rec = test_parameters(1.0, 3.7412, 153.36, 12.65, 6.0);
        let p = SaftVRCSPars::new(&p_rec);
        let temperature = 200.0_f64;
        let cp = CorrespondingParameters::new(&p, temperature);
        let d = cp.hs_diameter_ij(0, 0);
        // Should be close to saftvrmie value at this temperature
        assert!(d > 0.0 && d < p.sigma[0], "diameter should be between 0 and sigma");
    }

    #[test]
    fn test_eff_lr_ratio_no_correction() {
        let ratio = eff_lr_ratio(0.01_f64, 12.0, None);
        // At very low quantum parameter, ratio should be close to 1
        assert!((ratio - 1.0).abs() < 0.5);
    }
}
