#![allow(clippy::needless_range_loop)]

use super::corresponding_states::CorrespondingParameters;
use super::parameters::{SaftVRCSParameters, SaftVRCSPars};
use crate::hard_sphere::HardSphere;
use feos_core::{Molarweight, ResidualDyn, StateHD, Subset};
use nalgebra::DVector;
use num_dual::DualNum;
use quantity::MolarWeight;
use std::f64::consts::FRAC_PI_6;

pub(crate) mod dispersion;
use dispersion::{
    Properties, helmholtz_energy_density_disp, helmholtz_energy_density_disp_chain,
};

/// Customization options for SAFT-VR-CS.
#[derive(Copy, Clone)]
pub struct SaftVRCSOptions {
    pub max_eta: f64,
}

impl Default for SaftVRCSOptions {
    fn default() -> Self {
        Self { max_eta: 0.5 }
    }
}

/// SAFT-VR Mie equation of state with Feynman-Hibbs quantum effective parameters.
pub struct SaftVRCSMie {
    pub parameters: SaftVRCSParameters,
    pub params: SaftVRCSPars,
    pub chain: bool,
    pub options: SaftVRCSOptions,
}

impl SaftVRCSMie {
    pub fn new(parameters: SaftVRCSParameters) -> Self {
        Self::with_options(parameters, SaftVRCSOptions::default())
    }

    pub fn with_options(parameters: SaftVRCSParameters, options: SaftVRCSOptions) -> Self {
        let params = SaftVRCSPars::new(&parameters);
        let chain = params.m.iter().any(|&m| m > 1.0);
        Self {
            parameters,
            params,
            chain,
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

        // BH hard-sphere diameter via GLQ integration of the effective Mie potential
        let d: DVector<D> = DVector::from_fn(n, |i, _| cp.hs_diameter_ij(i, i));

        // Hard sphere (BMCSL) via the generic HardSphere implementation.
        // We supply &self.params which implements HardSphereProperties via hs_diameter(T).
        let (a_hs, _, _) =
            HardSphere.helmholtz_energy_density_and_properties(&self.params, state);

        // Dispersion (+ chain) using effective parameters
        let props = Properties::new(&cp, state, &d);
        let mut a = vec![("Hard Sphere", a_hs)];
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
    use feos_core::{FeosResult, State};
    use nalgebra::dvector;
    use quantity::{ANGSTROM, KELVIN, MOL, NAV, RGAS};
    use std::sync::Arc;
    use typenum::P3;

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
