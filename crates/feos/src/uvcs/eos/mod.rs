#![allow(clippy::excessive_precision)]
#![allow(clippy::needless_range_loop)]

use super::parameters::{CombiningRule, UVCSParameters, UVCSPars};
use feos_core::{Molarweight, ResidualDyn, Subset};
use super::corresponding_states::CorrespondingParameters;
use nalgebra::DVector;
use quantity::MolarWeight;
use std::f64::consts::FRAC_PI_6;

mod attractive_perturbation;
mod hard_sphere;
mod non_additive_hs;
mod reference_perturbation;
use attractive_perturbation::attractive_perturbation_helmholtz_energy_density;
use hard_sphere::{diameter_wca_ij, hard_sphere_helmholtz_energy_density};
use non_additive_hs::non_additive_hard_sphere_helmholtz_energy_density;
use reference_perturbation::reference_perturbation_helmholtz_energy_density;

/// Options for the uv-CS theory.
#[derive(Debug, Clone, Copy)]
pub struct UVCSOptions {
    pub max_eta: f64,
    /// Route to the effective unlike parameters, see `CombiningRule` (default: classical).
    pub combining_rule: CombiningRule,
    /// Non-additive hard spheres (as in SAFT-VRQ Mie): unlike WCA diameters d_ij from the
    /// unlike pair potential in the reference perturbation plus a first-order correction
    /// of the additive BMCSL hard-sphere term. If false, d_ij = (d_ii + d_jj) / 2.
    pub non_additive_hs: bool,
}

impl Default for UVCSOptions {
    fn default() -> Self {
        Self {
            max_eta: 0.5,
            combining_rule: CombiningRule::default(),
            non_additive_hs: false,
        }
    }
}

/// uv-theory equation of state
pub struct UVCSTheory {
    pub parameters: UVCSParameters,
    pub params: UVCSPars,
    pub non_additive_hard_sphere: bool,
    options: UVCSOptions,
}

impl UVCSTheory {
    /// uv-theory with default options (WCA).
    pub fn new(parameters: UVCSParameters) -> Self {
        Self::with_options(parameters, UVCSOptions::default())
    }

    /// uv-theory with provided options.
    pub fn with_options(parameters: UVCSParameters, options: UVCSOptions) -> Self {
        let params = UVCSPars::with_combining_rule(&parameters, options.combining_rule);
        let non_additive_hard_sphere = params.ncomponents > 1 && options.non_additive_hs;
        Self {
            parameters,
            params,
            non_additive_hard_sphere,
            options,
        }
    }
}

impl Subset for UVCSTheory {
    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(self.parameters.subset(component_list), self.options)
    }
}

impl ResidualDyn for UVCSTheory {
    fn components(&self) -> usize {
        self.parameters.pure.len()
    }

    fn compute_max_density<D: num_dual::DualNum<f64> + Copy>(&self, molefracs: &DVector<D>) -> D {
        (self.params.sigma.map(|v| D::from(v.powi(3))).dot(molefracs) * FRAC_PI_6).recip()
            * self.options.max_eta
    }

    fn reduced_helmholtz_energy_density_contributions<D: num_dual::DualNum<f64> + Copy>(
        &self,
        state: &feos_core::StateHD<D>,
    ) -> Vec<(&'static str, D)> {
        let parameters = CorrespondingParameters::new(&self.params, state.temperature);
        let d_ij = diameter_wca_ij(&parameters, state.temperature, self.non_additive_hard_sphere);
        let mut v = vec![(
            "Hard Sphere (WCA)",
            hard_sphere_helmholtz_energy_density(&parameters, state),
        )];
        if self.non_additive_hard_sphere {
            v.push((
                "Non additive Hard Sphere (WCA)",
                non_additive_hard_sphere_helmholtz_energy_density(state, &d_ij),
            ));
        }
        v.push((
            "Reference Perturbation (WCA)",
            reference_perturbation_helmholtz_energy_density(&parameters, state, &d_ij),
        ));
        v.push((
            "Attractive Perturbation (WCA)",
            attractive_perturbation_helmholtz_energy_density(&parameters, state),
        ));
        v
    }
}

impl Molarweight for UVCSTheory {
    fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
        self.parameters.molar_weight.clone()
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use super::*;

    use crate::uvcs::parameters::utils::{CLASSICAL_LIMIT_MOLARWEIGHT, test_parameters_mixture};
    use crate::uvcs::parameters::*;
    use approx::assert_relative_eq;
    use feos_core::{
        FeosResult, State,
        parameter::{Identifier, PureRecord},
    };
    use nalgebra::dvector;
    use feos_core::StateHD;
    use quantity::{ANGSTROM, KELVIN, MOL, NAV, RGAS};
    use typenum::P3;

    fn record(molarweight: f64, rep: f64, sigma: f64, epsilon_k: f64) -> PureRecord<UVCSRecord, ()> {
        PureRecord::new(
            Identifier::default(),
            molarweight,
            UVCSRecord::new(rep, 6.0, sigma, epsilon_k, None),
        )
    }

    fn contributions(
        eos: &UVCSTheory,
        temperature: f64,
        density: f64,
        x: f64,
    ) -> Vec<(&'static str, f64)> {
        let state = StateHD::new(temperature, 1.0 / density, &dvector![x, 1.0 - x]);
        eos.reduced_helmholtz_energy_density_contributions(&state)
    }

    fn with_non_additive_hs(parameters: UVCSParameters, non_additive_hs: bool) -> UVCSTheory {
        let options = UVCSOptions {
            non_additive_hs,
            ..Default::default()
        };
        UVCSTheory::with_options(parameters, options)
    }

    #[test]
    fn non_additive_hs_identical_components() {
        // two identical components: d_ij is additive, the option must not change anything
        let parameters = || {
            UVCSParameters::new_binary(
                [record(2.0157, 10.0, 3.05, 30.2), record(2.0157, 10.0, 3.05, 30.2)],
                None,
                vec![],
            )
            .unwrap()
        };
        let on = contributions(&with_non_additive_hs(parameters(), true), 20.0, 0.02, 0.3);
        let off = contributions(&with_non_additive_hs(parameters(), false), 20.0, 0.02, 0.3);
        assert_eq!(on.len(), 4);
        assert_eq!(off.len(), 3);
        assert_eq!(on[1].0, "Non additive Hard Sphere (WCA)");
        assert!(on[1].1.abs() < 1e-14);
        let total = |c: &[(&str, f64)]| c.iter().map(|(_, a)| a).sum::<f64>();
        assert_relative_eq!(total(&on), total(&off), max_relative = 1e-12);
    }

    #[test]
    fn non_additive_hs_helium_hydrogen() {
        // He/H2-like binary: the unlike WCA diameter differs from the arithmetic mean
        let parameters = || {
            UVCSParameters::new_binary(
                [record(4.0026, 12.0, 2.64, 9.56), record(2.0157, 10.0, 3.05, 30.2)],
                None,
                vec![],
            )
            .unwrap()
        };
        let t = 20.0;
        let cp = CorrespondingParameters::new(&UVCSPars::new(&parameters()), t);
        let d_add = diameter_wca_ij(&cp, t, false);
        let d_non_add = diameter_wca_ij(&cp, t, true);
        for i in 0..2 {
            assert_eq!(d_add[(i, i)], d_non_add[(i, i)]);
        }
        assert!((d_non_add[(0, 1)] / d_add[(0, 1)] - 1.0).abs() > 1e-4);
        assert_eq!(d_non_add[(0, 1)], d_non_add[(1, 0)]);

        let on = contributions(&with_non_additive_hs(parameters(), true), t, 0.02, 0.1);
        let off = contributions(&with_non_additive_hs(parameters(), false), t, 0.02, 0.1);
        assert!(on[1].1.abs() > 1e-8);
        // like-pair terms are unaffected
        assert_eq!(on[0], off[0]);
        assert_eq!(on[3], off[2]);
        assert!((on[2].1 / off[1].1 - 1.0).abs() > 1e-6);
    }

    #[test]
    fn helmholtz_energy_pure_wca() -> FeosResult<()> {
        let sig = 3.7039;
        let eps_k = 150.03;
        let pure_record = PureRecord::new(
            Identifier::default(),
            CLASSICAL_LIMIT_MOLARWEIGHT,
            UVCSRecord {
                rep: 24.0,
                att: 6.0,
                sigma: sig,
                epsilon_k: eps_k,
                quantum_correction: None,
            },
        );
        let parameters = UVCSParameters::new_pure(pure_record)?;
        let eos = Arc::new(UVCSTheory::new(parameters));

        let reduced_temperature = 4.0;
        let reduced_density = 1.0;
        let temperature = reduced_temperature * eps_k * KELVIN;
        let moles = dvector![2.0] * MOL;
        let volume = (sig * ANGSTROM).powi::<P3>() / reduced_density * NAV * 2.0 * MOL;
        let s = State::new_nvt(&eos, temperature, volume, &moles).unwrap();
        let a = (s.residual_molar_helmholtz_energy() / (RGAS * temperature)).into_value();
        assert_relative_eq!(a, 2.972986567516, max_relative = 1e-12); //wca
        Ok(())
    }

    #[test]
    fn helmholtz_energy_wca_mixture() -> FeosResult<()> {
        let _p = test_parameters_mixture(
            dvector![12.0, 12.0],
            dvector![6.0, 6.0],
            dvector![1.0, 1.0],
            dvector![1.0, 0.5],
        );
        let p = UVCSPars::new(&_p);

        // state
        let reduced_temperature = 1.0;
        let t_x = reduced_temperature * p.epsilon_k[0] * KELVIN;
        let reduced_density = 0.9;
        let moles = dvector![0.4, 0.6] * MOL;
        let total_moles = moles.sum();
        let volume = (p.sigma[0] * ANGSTROM).powi::<P3>() / reduced_density * NAV * total_moles;

        // EoS
        let eos_wca = Arc::new(UVCSTheory::new(_p));
        let state_wca = State::new_nvt(&eos_wca, t_x, volume, &moles).unwrap();
        let a_wca = (state_wca.residual_helmholtz_energy() / (RGAS * t_x * state_wca.total_moles))
            .into_value();

        assert_relative_eq!(a_wca, -0.597791038364405, max_relative = 1e-5);
        Ok(())
    }

    #[test]
    fn helmholtz_energy_wca_mixture_different_sigma() -> FeosResult<()> {
        let _p = test_parameters_mixture(
            dvector![12.0, 12.0],
            dvector![6.0, 6.0],
            dvector![1.0, 2.0],
            dvector![1.0, 0.5],
        );
        let p = UVCSPars::new(&_p);

        // state
        let reduced_temperature = 1.5;
        let t_x = reduced_temperature * p.epsilon_k[0] * KELVIN;
        let sigma_x_3 = (0.4 + 0.6 * 8.0) * ANGSTROM.powi::<P3>();
        let density = 0.52000000000000002 / sigma_x_3;
        let moles = dvector![0.4, 0.6] * MOL;
        let total_moles = moles.sum();
        let volume = NAV * total_moles / density;

        // EoS
        let eos_wca = Arc::new(UVCSTheory::new(_p));
        let state_wca = State::new_nvt(&eos_wca, t_x, volume, &moles).unwrap();
        let a_wca = (state_wca.residual_molar_helmholtz_energy() / (RGAS * t_x)).into_value();
        // d*_x from Eq. (36) (upstream uv-theory with Eq. 38: -0.034206207363139396)
        assert_relative_eq!(a_wca, -0.04479571669110117, max_relative = 1e-5);
        Ok(())
    }
}
