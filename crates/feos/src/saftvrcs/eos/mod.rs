#![allow(clippy::needless_range_loop)]

use super::corresponding_states::CorrespondingParameters;
use super::parameters::{CombiningRule, SaftVRCSParameters, SaftVRCSPars};
use crate::hard_sphere::HardSphere;
use feos_core::{Molarweight, ResidualDyn, StateHD, Subset};
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use quantity::MolarWeight;
use std::f64::consts::FRAC_PI_6;

pub(crate) mod dispersion;
mod non_additive_hs;
use dispersion::{
    Properties, helmholtz_energy_density_disp, helmholtz_energy_density_disp_chain,
};
use non_additive_hs::helmholtz_energy_density_non_additive_hs;

/// Customization options for SAFT-VR-CS.
#[derive(Copy, Clone)]
pub struct SaftVRCSOptions {
    pub max_eta: f64,
    pub combining_rule: CombiningRule,
    /// Non-additive hard spheres (as in SAFT-VRQ Mie): unlike diameters d_ij from the
    /// unlike pair potential in the dispersion term plus a hard-sphere correction term.
    /// If false, d_ij = (d_ii + d_jj) / 2.
    pub non_additive_hs: bool,
}

impl Default for SaftVRCSOptions {
    fn default() -> Self {
        Self {
            max_eta: 0.5,
            combining_rule: CombiningRule::default(),
            non_additive_hs: true,
        }
    }
}

/// SAFT-VR Mie equation of state with Feynman-Hibbs quantum effective parameters.
pub struct SaftVRCSMie {
    pub parameters: SaftVRCSParameters,
    pub params: SaftVRCSPars,
    pub chain: bool,
    pub non_additive_hard_sphere: bool,
    pub options: SaftVRCSOptions,
}

impl SaftVRCSMie {
    pub fn new(parameters: SaftVRCSParameters) -> Self {
        Self::with_options(parameters, SaftVRCSOptions::default())
    }

    pub fn with_options(parameters: SaftVRCSParameters, options: SaftVRCSOptions) -> Self {
        let params =
            SaftVRCSPars::new(&parameters).with_combining_rule(options.combining_rule);
        let chain = params.m.iter().any(|&m| m > 1.0);
        let non_additive_hard_sphere = params.ncomponents > 1 && options.non_additive_hs;
        Self {
            parameters,
            params,
            chain,
            non_additive_hard_sphere,
            options,
        }
    }
}

impl ResidualDyn for SaftVRCSMie {
    fn components(&self) -> usize {
        self.params.m.len()
    }

    fn compute_max_density<D: DualNum<f64> + Copy>(&self, molefracs: &DVector<D>) -> D {
        let msigma3 = self
            .params
            .m
            .component_mul(&self.params.sigma.map(|v| v.powi(3)));
        (msigma3.map(D::from).dot(molefracs) * FRAC_PI_6).recip() * self.options.max_eta
    }

    fn reduced_helmholtz_energy_density_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(&'static str, D)> {
        // Effective (temperature-dependent) parameters from quantum corrections
        let cp = CorrespondingParameters::new(&self.params, state.temperature);
        let n = cp.ncomponents;

        // BH hard-sphere diameters via GLQ integration of the effective Mie potentials;
        // unlike d_ij from the unlike pair potential (non-additive) or the arithmetic mean
        let d: DVector<D> = DVector::from_fn(n, |i, _| cp.hs_diameter_ij(i, i));
        let mut d_ij = DMatrix::from_diagonal(&d);
        for i in 0..n {
            for j in i + 1..n {
                d_ij[(i, j)] = if self.non_additive_hard_sphere {
                    cp.hs_diameter_ij(i, j)
                } else {
                    (d[i] + d[j]) * 0.5
                };
                d_ij[(j, i)] = d_ij[(i, j)];
            }
        }

        // Hard sphere (BMCSL) via the generic HardSphere implementation.
        // We supply &self.params which implements HardSphereProperties via hs_diameter(T).
        let (a_hs, _, _) =
            HardSphere.helmholtz_energy_density_and_properties(&self.params, state);

        // Dispersion (+ chain) using effective parameters
        let props = Properties::new(&cp, state, &d_ij);
        let mut a = vec![("Hard Sphere", a_hs)];
        if self.non_additive_hard_sphere {
            a.push((
                "Non additive Hard Sphere",
                helmholtz_energy_density_non_additive_hs(&cp.m, &state.partial_density, &d_ij),
            ));
        }
        if self.chain {
            a.push((
                "Dispersion + Chain",
                helmholtz_energy_density_disp_chain(&cp, &props, state),
            ));
        } else {
            a.push((
                "Dispersion",
                helmholtz_energy_density_disp(&cp, &props, state),
            ));
        }
        a
    }
}

impl Subset for SaftVRCSMie {
    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(self.parameters.subset(component_list), self.options)
    }
}

impl Molarweight for SaftVRCSMie {
    fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
        self.parameters.molar_weight.clone()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::saftvrcs::parameters::utils::test_parameters;
    use crate::saftvrcs::parameters::{QuantumCorrection, SaftVRCSRecord};
    use feos_core::parameter::{Identifier, PureRecord};
    use feos_core::{FeosResult, State};
    use nalgebra::dvector;
    use quantity::{ANGSTROM, KELVIN, MOL, NAV, RGAS};
    use std::sync::Arc;
    use typenum::P3;

    fn qc_record(molarweight: f64, sigma: f64, epsilon_k: f64, lr: f64) -> PureRecord<SaftVRCSRecord, ()> {
        let qc = QuantumCorrection::FeynmanHibbs1 {
            c_sigma: None,
            c_epsilon_k: None,
            c_lr: None,
        };
        PureRecord::new(
            Identifier::default(),
            molarweight,
            SaftVRCSRecord::new(1.0, sigma, epsilon_k, lr, 6.0, Some(qc)),
        )
    }

    fn contributions(eos: &SaftVRCSMie, temperature: f64, density: f64, x: f64) -> Vec<(&'static str, f64)> {
        let state = StateHD::new(temperature, 1.0 / density, &dvector![x, 1.0 - x]);
        eos.reduced_helmholtz_energy_density_contributions(&state)
    }

    #[test]
    fn non_additive_hs_identical_components() {
        // two identical components: d_ij is additive, the option must not change anything
        let record = || qc_record(4.0, 2.64, 9.56, 11.8);
        let parameters =
            || SaftVRCSParameters::new_binary([record(), record()], None, vec![]).unwrap();
        let with_option = |non_additive_hs| {
            let options = SaftVRCSOptions {
                non_additive_hs,
                ..Default::default()
            };
            SaftVRCSMie::with_options(parameters(), options)
        };
        let on = contributions(&with_option(true), 4.0, 0.02, 0.3);
        let off = contributions(&with_option(false), 4.0, 0.02, 0.3);
        assert_eq!(on[1].0, "Non additive Hard Sphere");
        assert!(on[1].1.abs() < 1e-14);
        let total = |c: &[(&str, f64)]| c.iter().map(|(_, a)| a).sum::<f64>();
        assert!((total(&on) / total(&off) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn non_additive_hs_helium_hydrogen() {
        // He/H2-like binary: the unlike BH diameter differs from the arithmetic mean
        let parameters = || {
            SaftVRCSParameters::new_binary(
                [qc_record(4.0026, 2.64, 9.56, 11.8), qc_record(2.0157, 3.05, 30.2, 9.9)],
                None,
                vec![],
            )
            .unwrap()
        };
        for combining_rule in [CombiningRule::Effective, CombiningRule::Classical] {
            let params = SaftVRCSPars::new(&parameters()).with_combining_rule(combining_rule);
            let cp = CorrespondingParameters::new(&params, 20.0);
            let d_add = 0.5 * (cp.hs_diameter_ij(0, 0) + cp.hs_diameter_ij(1, 1));
            assert!((cp.hs_diameter_ij(0, 1) / d_add - 1.0).abs() > 1e-4);

            let options = SaftVRCSOptions {
                combining_rule,
                ..Default::default()
            };
            let eos = SaftVRCSMie::with_options(parameters(), options);
            let a = contributions(&eos, 20.0, 0.03, 0.1);
            assert_eq!(a.len(), 3);
            assert!(a[1].1.abs() > 1e-8);
        }
    }

    #[test]
    fn helmholtz_energy_methane_like() -> FeosResult<()> {
        // At T = eps_k (strong dispersion) and moderate density, a_res should be clearly negative.
        let sig = 3.7412_f64;
        let eps_k = 153.36_f64;
        let p = test_parameters(1.0, sig, eps_k, 12.65, 6.0);
        let eos = Arc::new(SaftVRCSMie::new(p));

        let temperature = eps_k * KELVIN;
        let moles = dvector![1.0] * MOL;
        let volume = (sig * ANGSTROM).powi::<P3>() / 0.3 * NAV * 1.0 * MOL;
        let s = State::new_nvt(&eos, temperature, volume, &moles)?;
        let a = (s.residual_molar_helmholtz_energy() / (RGAS * temperature)).into_value();
        assert!(a < -0.1, "expected clearly negative reduced Helmholtz energy, got {a}");
        Ok(())
    }
}
