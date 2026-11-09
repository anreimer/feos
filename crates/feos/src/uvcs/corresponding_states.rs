use std::{collections::HashMap, f64::consts::TAU};

use super::parameters::{CombiningRule, QuantumCorrection, UVCSPars};
use itertools::izip;
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use quantity::{GRAM, KILOGRAM, MOL, NAV};

const C: [[f64; 3]; 5] = [
    [-1.0577620680e+02, 2.7130516920e+01, -7.9885048560E-02],
    [-2.2875719760e+03, 5.1995559760e+02, 8.2111050000E-01],
    [1.7306319200e+03, -1.7476174200e+02, 3.2561549200e+01],
    [-9.0993639000e+01, 2.2823545260e+01, -2.8168749280e-02],
    [-1.4764571200e+03, 4.3447760860e+02, -1.1267359060e+01],
];

/// Parameters for effective sigma
const CS: [f64; 9] = [
    31.60144413,
    2.09612861,
    0.20116118,
    50.80286701,
    -4.28014515,
    0.20063555,
    25.11333946,
    1.35310377,
    0.20250069,
];

const CE: [f64; 9] = [
    10.60837315,
    1.58987996,
    0.19807347,
    17.92633472,
    -19.76493004,
    3.91514157,
    7.17199935,
    6.45489122,
    0.18773935,
];

/// Parameters according to corresponding states principle
#[derive(Debug)]
pub struct CorrespondingParameters<D> {
    pub ncomponents: usize,
    pub rep: DVector<D>,
    pub att: DVector<D>,
    pub sigma: DVector<D>,
    pub epsilon_k: DVector<D>,
    pub rep_ij: DMatrix<D>,
    pub att_ij: DMatrix<D>,
    pub sigma_ij: DMatrix<D>,
    pub eps_k_ij: DMatrix<D>,
}

impl<D: DualNum<f64> + Copy> CorrespondingParameters<D> {
    pub fn new(parameters: &UVCSPars, temperature: D) -> Self {
        let p = &parameters;
        let n = p.ncomponents;
        let to_mass_per_molecule = (GRAM / MOL / NAV / KILOGRAM).into_value();
        let mass = p.molarweight.map(|mw| mw * to_mass_per_molecule);
        let classical = |i: usize| [p.rep[i], p.sigma[i], p.epsilon_k[i]];
        let zeta: Vec<_> = p.quantum_correction.iter().map(prefactors).collect();

        // Every component is mapped onto effective parameters [rep, sigma, epsilon_k];
        // missing prefactors take their default values.
        let mut rep = DVector::zeros(n);
        let mut sigma = DVector::zeros(n);
        let mut epsilon_k = DVector::zeros(n);
        for i in 0..n {
            assert_eq!(p.att[i], 6.0);
            [rep[i], sigma[i], epsilon_k[i]] =
                effective_parameters(temperature, mass[i], classical(i), &zeta[i]);
        }
        let att = p.att.map(|a| D::one() * a);

        let mut rep_ij = DMatrix::zeros(n, n);
        let mut att_ij = DMatrix::zeros(n, n);
        let mut sigma_ij = DMatrix::zeros(n, n);
        let mut eps_k_ij = DMatrix::zeros(n, n);
        for i in 0..n {
            rep_ij[(i, i)] = rep[i];
            att_ij[(i, i)] = att[i];
            sigma_ij[(i, i)] = sigma[i];
            eps_k_ij[(i, i)] = epsilon_k[i];
            for j in i + 1..n {
                let [r, s, e] = match p.combining_rule {
                    CombiningRule::Classical => {
                        // m_ij = 2 mu_ij with the reduced mass mu_ij of the pair
                        let m_ij = 2.0 * mass[i] * mass[j] / (mass[i] + mass[j]);
                        let zeta_ij: [[f64; 5]; 3] = std::array::from_fn(|k| {
                            std::array::from_fn(|l| 0.5 * (zeta[i][k][l] + zeta[j][k][l]))
                        });
                        let classical_ij = combine(p, i, j, classical(i), classical(j));
                        effective_parameters(temperature, m_ij, classical_ij, &zeta_ij)
                    }
                    CombiningRule::Effective => combine(
                        p,
                        i,
                        j,
                        [rep[i], sigma[i], epsilon_k[i]],
                        [rep[j], sigma[j], epsilon_k[j]],
                    ),
                };
                rep_ij[(i, j)] = r;
                rep_ij[(j, i)] = r;
                att_ij[(i, j)] = (att[i] + att[j]) * 0.5;
                att_ij[(j, i)] = att_ij[(i, j)];
                sigma_ij[(i, j)] = s;
                sigma_ij[(j, i)] = s;
                eps_k_ij[(i, j)] = e;
                eps_k_ij[(j, i)] = e;
            }
        }
        Self {
            ncomponents: n,
            rep,
            att,
            sigma,
            epsilon_k,
            rep_ij,
            att_ij,
            sigma_ij,
            eps_k_ij,
        }
    }
}

/// Prefactors [c_rep, c_sigma, c_epsilon_k] of a component, defaults where not given
/// (also for components without a quantum correction record).
fn prefactors(qc: &Option<QuantumCorrection>) -> [[f64; 5]; 3] {
    const DEFAULT_C_REP: [f64; 5] = [1.0; 5];
    const DEFAULT_C_SIGMA_EPSILON: [f64; 5] = [1.0, 1.0, 1.0, 0.0, 0.0];
    match qc {
        Some(QuantumCorrection::FeynmanHibbs1 {
            c_sigma,
            c_epsilon_k,
            c_rep,
        }) => [
            c_rep.unwrap_or(DEFAULT_C_REP),
            c_sigma.unwrap_or(DEFAULT_C_SIGMA_EPSILON),
            c_epsilon_k.unwrap_or(DEFAULT_C_SIGMA_EPSILON),
        ],
        None => [DEFAULT_C_REP, DEFAULT_C_SIGMA_EPSILON, DEFAULT_C_SIGMA_EPSILON],
    }
}

/// Effective [rep, sigma, epsilon_k] from the classical [rep, sigma, epsilon_k],
/// the particle mass and the prefactors [c_rep, c_sigma, c_epsilon_k].
fn effective_parameters<D: DualNum<f64> + Copy>(
    temperature: D,
    mass: f64,
    [rep, sigma, epsilon_k]: [f64; 3],
    [c_rep, c_sigma, c_epsilon_k]: &[[f64; 5]; 3],
) -> [D; 3] {
    let d_s2 = temperature.recip() / mass * D_QM_PREFACTOR / sigma.powi(2);
    [
        effective_repulsive_exponent_ratio(d_s2, rep, Some(c_rep)) * rep,
        effective_sigma(d_s2, rep, Some(c_sigma)) * sigma,
        effective_epsilon_k(d_s2, rep, Some(c_epsilon_k)) * epsilon_k,
    ]
}

/// Unlike [rep_ij, sigma_ij, epsilon_k_ij] from the [rep, sigma, epsilon_k] of i and j:
/// rep_ij = (rep_i + rep_j) / 2, sigma_ij = (sigma_i + sigma_j) / 2 * (1 - l_ij),
/// epsilon_k_ij = sqrt(epsilon_k_i * epsilon_k_j) * (1 - k_ij).
fn combine<T: DualNum<f64> + Copy>(
    p: &UVCSPars,
    i: usize,
    j: usize,
    [rep_i, sigma_i, eps_i]: [T; 3],
    [rep_j, sigma_j, eps_j]: [T; 3],
) -> [T; 3] {
    [
        (rep_i + rep_j) * 0.5,
        (sigma_i + sigma_j) * 0.5 * (1.0 - p.l_ij[(i, j)]),
        (eps_i * eps_j).sqrt() * (1.0 - p.k_ij[(i, j)]),
    ]
}

/// Calculate m_eff/m given D(T)/sig^2 and m.
fn effective_repulsive_exponent_ratio<D: DualNum<f64> + Copy>(
    qd: D,
    m: f64,
    c: Option<&[f64; 5]>,
) -> D {
    let c_scale = c.unwrap_or(&[1.0; 5]);
    let mut pref = [0.0; 5];
    for (pi, ci, c_scale_i) in izip!(&mut pref, &C, c_scale) {
        *pi = (ci[0] + m * (ci[1] + m * ci[2])) * c_scale_i;
    }
    (qd * (qd * ((qd * pref[2]) + pref[1]) + pref[0]) + 1.0)
        / (qd * ((qd * pref[4]) + pref[3]) + 1.0)
}

fn effective_sigma<D: DualNum<f64> + Copy>(qd: D, m: f64, c: Option<&[f64; 5]>) -> D {
    let c_scale = c.unwrap_or(&[1.0, 1.0, 1.0, 0.0, 0.0]);
    let c0 = CS[0] + m * (CS[1] + (m * CS[2]));
    let c1 = CS[3] + m * (CS[4] + (m * CS[5]));
    let c2 = CS[6] + m * (CS[7] + (m * CS[8]));

    (qd * (qd * c1 * c_scale[1] + c0 * c_scale[0] + qd.powi(2) * c_scale[3]) + 1.0) / (qd * c2 * c_scale[2] + qd.powi(2) * c_scale[4]  + 1.0)
}

fn effective_epsilon_k<D: DualNum<f64> + Copy>(qd: D, m: f64, c: Option<&[f64; 5]>) -> D {
    //let c_scale = c.unwrap_or(&[1.0; 3]);
    let c_scale = c.unwrap_or(&[1.0, 1.0, 1.0, 0.0, 0.0]);
    let c0 = CE[0] + m * (CE[1] + (m * CE[2]));
    let c1 = CE[3] + m * (CE[4] + (m * CE[5]));
    let c2 = CE[6] + m * (CE[7] + (m * CE[8]));

    (qd * (qd * c1 * c_scale[1] + c0 * c_scale[0] + qd.powi(2) * c_scale[3]) + 1.0) / (qd * c2 * c_scale[2] + qd.powi(2) * c_scale[4] + 1.0)
}

const KB: f64 = 1.380649e-23;
const PLANCK: f64 = 6.62607015e-34;
const D_QM_PREFACTOR: f64 = PLANCK * PLANCK / (TAU * TAU) / 12.0 * 1e20 / KB;

pub struct MieFeynmanHibbs1 {
    pub sigma: f64,
    pub epsilon_k: f64,
    pub lr: f64,
    pub la: f64,
}

impl MieFeynmanHibbs1 {
    pub fn u_du_d2u2<D: DualNum<f64> + Copy>(&self, r: D, mass: f64, temperature: D) -> [D; 3] {
        let s = self.sigma;
        let eps = self.epsilon_k;
        let lr = self.lr;
        let la = self.la;
        let c = lr / (lr - la) * (lr / la).powf(la / (lr - la));

        let q1r = lr * (lr - 1.0);
        let q1a = la * (la - 1.0);
        let d = temperature.recip() / mass * D_QM_PREFACTOR;
        let mut u = r.powf(lr).recip() * s.powf(lr) - r.powf(la).recip() * s.powf(la);
        let mut u_r = -r.powf(lr + 1.0).recip() * lr * s.powf(lr)
            + r.powf(la + 1.0).recip() * la * s.powf(la);
        let mut u_rr = r.powf(lr + 2.0).recip() * lr * (lr + 1.0) * s.powf(lr)
            - r.powf(la + 2.0).recip() * la * (la + 1.0) * s.powf(la);

        // 1st order
        u += d
            * (r.powf(lr + 2.0).recip() * q1r * s.powf(lr)
                - r.powf(la + 2.0).recip() * q1a * s.powf(la));
        u_r += d
            * (r.powf(lr + 3.0).recip() * -q1r * (lr + 2.0) * s.powf(lr)
                + r.powf(la + 3.0).recip() * q1a * (la + 2.0) * s.powf(la));
        u_rr += d
            * (r.powf(lr + 4.0).recip() * q1r * (lr + 2.0) * (lr + 3.0) * s.powf(lr)
                - r.powf(la + 4.0).recip() * q1a * (la + 2.0) * (la + 3.0) * s.powf(la));

        u *= c * eps;
        u_r *= c * eps;
        u_rr *= c * eps;
        [u, u_r, u_rr]
    }

    // pub fn calc_sigma_eff_ij<D: DualNum<f64> + Copy>(
    //     &self,
    //     i: usize,
    //     j: usize,
    //     temperature: D,
    // ) -> D {
    //     let mut r = D::one() * self.sigma_ij[[i, j]];
    //     let mut u_vec = [D::zero(), D::zero(), D::zero()];
    //     for _k in 1..20 {
    //         u_vec = self.qmie_potential_ij(i, j, r, temperature);
    //         if u_vec[0].re().abs() < 1.0e-12 {
    //             break;
    //         }
    //         r += -u_vec[0] / u_vec[1];
    //     }
    //     if u_vec[0].re().abs() > 1.0e-12 {
    //         println!("calc_sigma_eff_ij calculation failed");
    //     }
    //     r
    // }

    pub fn effective_parameters<D: DualNum<f64> + Copy>(
        &self,
        temperature: D,
        mass: f64,
    ) -> [D; 3] {
        let mut r = D::one() * self.sigma;
        let mut u_vec = [D::zero(), D::zero(), D::zero()];

        // sigma
        for _k in 1..20 {
            u_vec = self.u_du_d2u2(r, mass, temperature);
            if u_vec[0].re().abs() < 1.0e-12 {
                break;
            }
            r += -u_vec[0] / u_vec[1];
        }
        if u_vec[0].re().abs() > 1.0e-12 {
            println!("calc_sigma_eff_ij calculation failed");
        }
        let sigma_eff = r;

        // epsilon
        let mut r = sigma_eff;
        for _k in 1..20 {
            u_vec = self.u_du_d2u2(r, mass, temperature);
            if u_vec[1].re().abs() < 1.0e-12 {
                break;
            }
            r += -u_vec[1] / u_vec[2];
        }
        if u_vec[1].re().abs() > 1.0e-12 {
            println!("calc_epsilon_k_eff_ij calculation failed");
        }

        // d
        let d = temperature.recip() / mass * D_QM_PREFACTOR;
        [sigma_eff, -u_vec[0], d / self.sigma.powi(2)]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::uvcs::parameters::{UVCSParameters, UVCSRecord};
    use feos_core::parameter::{Identifier, PureRecord};

    #[test]
    fn test_effective_parameters() {
        let fh = MieFeynmanHibbs1 {
            sigma: 2.7443,
            epsilon_k: 5.4195,
            lr: 9.0,
            la: 6.0,
        };
        let molarweight = 4.002601643881807;
        let to_mass_per_molecule = (GRAM / MOL / NAV / KILOGRAM).into_value();
        let [sigma, eps, d] = fh.effective_parameters(20.0, molarweight * to_mass_per_molecule);
        assert!((sigma - 2.92483368).abs() / 2.92483368 < 1e-5);
        assert!((eps - 4.50862751).abs() / 4.50862751 < 1e-5);
        assert!((d - 0.00670507).abs() / 0.00670507 < 1e-5)
    }

    fn fh1_record(molarweight: f64, sigma: f64, c: [f64; 5]) -> PureRecord<UVCSRecord, ()> {
        let qc = QuantumCorrection::FeynmanHibbs1 {
            c_sigma: Some(c),
            c_epsilon_k: Some(c),
            c_rep: Some(c),
        };
        PureRecord::new(
            Identifier::default(),
            molarweight,
            UVCSRecord::new(12.0, 6.0, sigma, 10.0, Some(qc)),
        )
    }

    #[test]
    fn test_classical_combining_rule() {
        // Same classical parameters, different masses and prefactors: the unlike pair
        // must equal a pure fluid with m_ij = 2 m_i m_j / (m_i + m_j) and mean prefactors.
        let (m1, m2) = (4.0, 2.0);
        let c1 = [1.1, 0.9, 1.0, 0.1, 0.0];
        let c2 = [0.9, 1.3, 1.2, 0.0, 0.2];
        let c12 = std::array::from_fn(|l| 0.5 * (c1[l] + c2[l]));
        let binary = UVCSParameters::new_binary(
            [fh1_record(m1, 2.8, c1), fh1_record(m2, 2.8, c2)],
            None,
            vec![],
        )
        .unwrap();
        let pure =
            UVCSParameters::new_pure(fh1_record(2.0 * m1 * m2 / (m1 + m2), 2.8, c12)).unwrap();
        let temperature = 10.0;
        let rules = |combining_rule| UVCSPars::with_combining_rule(&binary, combining_rule);

        let cp = CorrespondingParameters::new(&rules(CombiningRule::Classical), temperature);
        let cp_ref = CorrespondingParameters::new(&UVCSPars::new(&pure), temperature);
        for (x, x_ref) in [
            (cp.sigma_ij[(0, 1)], cp_ref.sigma[0]),
            (cp.eps_k_ij[(0, 1)], cp_ref.epsilon_k[0]),
            (cp.rep_ij[(0, 1)], cp_ref.rep[0]),
        ] {
            assert!((x / x_ref - 1.0).abs() < 1e-12, "{x} != {x_ref}");
        }

        // like pairs are the same for both combining rules, unlike pairs differ
        let cp_eff = CorrespondingParameters::new(&rules(CombiningRule::Effective), temperature);
        for i in 0..2 {
            assert_eq!(cp.sigma_ij[(i, i)], cp_eff.sigma_ij[(i, i)]);
            assert_eq!(cp.eps_k_ij[(i, i)], cp_eff.eps_k_ij[(i, i)]);
            assert_eq!(cp.rep_ij[(i, i)], cp_eff.rep_ij[(i, i)]);
        }
        assert!((cp.eps_k_ij[(0, 1)] / cp_eff.eps_k_ij[(0, 1)] - 1.0).abs() > 1e-6);
    }

    #[test]
    fn test_arithmetic_exponents() {
        // effective exponents of a quantum pair: rep_ij and att_ij are arithmetic means
        let record = |molarweight, rep| {
            PureRecord::new(
                Identifier::default(),
                molarweight,
                UVCSRecord::new(rep, 6.0, 2.8, 10.0, None),
            )
        };
        let binary =
            UVCSParameters::new_binary([record(4.0, 12.0), record(2.0, 24.0)], None, vec![])
                .unwrap();
        let temperature = 10.0;

        // effective route: arithmetic mean of the effective pure-component exponents
        let p = UVCSPars::with_combining_rule(&binary, CombiningRule::Effective);
        let cp = CorrespondingParameters::new(&p, temperature);
        let rep_mean = (cp.rep[0] + cp.rep[1]) * 0.5;
        assert!((cp.rep_ij[(0, 1)] / rep_mean - 1.0).abs() < 1e-14);
        assert_eq!(cp.rep_ij[(0, 1)], cp.rep_ij[(1, 0)]);
        assert_eq!(cp.att_ij[(0, 1)], 6.0);

        // classical route (default): arithmetic mean of the classical exponents (18),
        // mapped at the pair mass m_ij = 2 m_i m_j / (m_i + m_j)
        let cp = CorrespondingParameters::new(&UVCSPars::new(&binary), temperature);
        let pair = UVCSParameters::new_pure(record(2.0 * 4.0 * 2.0 / 6.0, 18.0)).unwrap();
        let cp_pair = CorrespondingParameters::new(&UVCSPars::new(&pair), temperature);
        assert!((cp.rep_ij[(0, 1)] / cp_pair.rep[0] - 1.0).abs() < 1e-14);
        assert_eq!(cp.att_ij[(0, 1)], 6.0);
    }

    #[test]
    fn test_no_quantum_correction_record_is_mapped() {
        // A record without quantum correction is mapped with the default prefactors.
        let record = |qc| {
            let pr = PureRecord::new(
                Identifier::default(),
                4.0,
                UVCSRecord::new(12.0, 6.0, 2.8, 10.0, qc),
            );
            UVCSPars::new(&UVCSParameters::new_pure(pr).unwrap())
        };
        let none = record(None);
        let default = record(Some(QuantumCorrection::FeynmanHibbs1 {
            c_sigma: None,
            c_epsilon_k: None,
            c_rep: None,
        }));
        let cp_none = CorrespondingParameters::new(&none, 10.0);
        let cp_default = CorrespondingParameters::new(&default, 10.0);
        assert_eq!(cp_none.sigma[0], cp_default.sigma[0]);
        assert_eq!(cp_none.epsilon_k[0], cp_default.epsilon_k[0]);
        assert_eq!(cp_none.rep[0], cp_default.rep[0]);
        assert!((cp_none.sigma[0] / none.sigma[0] - 1.0).abs() > 1e-3);
    }

    #[test]
    fn test_effective_repulsive_exponent_ratio() {
        let m = effective_repulsive_exponent_ratio(10.0, 12.0, None);
        assert!(f64::abs(m - 22.18560064715075) / 22.18560064715075 < 1e-12);

        let m = effective_repulsive_exponent_ratio(1.234, 9.123, None);
        assert!(f64::abs(m - 3.7406349183375567) / 3.7406349183375567 < 1e-12);

        let m =
            effective_repulsive_exponent_ratio(1.234, 9.123, Some(&[1.01, 0.98, 0.99, 1.02, 2.0]));
        assert!(f64::abs(m - 1.8976602346883111) / 1.8976602346883111 < 1e-12)
    }

    #[test]
    fn test_effective_epsilon_k() {
        let e = effective_epsilon_k(1.234, 9.123, None);
        dbg!(e);
        assert!(f64::abs(e - 2.9592337917650835) / 2.9592337917650835 < 1e-12);

        let e = effective_epsilon_k(1.234, 9.123, Some(&[1.01, 0.98, 0.99, 0.0, 0.0]));
        dbg!(e);
        assert!(f64::abs(e - 2.94452907265039) / 2.94452907265039 < 1e-12)
    }

    #[test]
    fn test_effective_sigma() {
        let s = effective_sigma(1.234, 9.123, None);
        dbg!(s);
        assert!(f64::abs(s - 1.8756435576601465) / 1.8756435576601465 < 1e-12);

        let s = effective_sigma(1.234, 9.123, Some(&[1.01, 0.98, 0.99, 0.0, 0.0]));
        dbg!(s);
        assert!(f64::abs(s - 1.8938029821536044) / 1.8938029821536044 < 1e-12)
    }
}
