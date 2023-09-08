#![allow(clippy::excessive_precision)]
#![allow(clippy::needless_range_loop)]

use super::parameters::UVParameters;
use crate::association::Association;
use feos_core::si::{MolarWeight, GRAM, MOL};
use feos_core::{parameter::Parameter, Components, EosError, EosResult, HelmholtzEnergy, Residual};
use ndarray::Array1;
use std::f64::consts::FRAC_PI_6;
use std::sync::Arc;
use typenum::P3;
pub(crate) mod attractive_perturbation_bh;
pub(crate) mod attractive_perturbation_uvb3;
pub(crate) mod attractive_perturbation_wca;
pub(crate) mod chain_bh_tptv;
pub(crate) mod chain_bh_tpty;
pub(crate) mod hard_sphere_bh;
pub(crate) mod hard_sphere_wca;
pub(crate) mod polar;
pub(crate) mod reference_perturbation_bh;
pub(crate) mod reference_perturbation_uvb3;
pub(crate) mod reference_perturbation_wca;
pub(crate) mod ufraction;
use attractive_perturbation_bh::AttractivePerturbationBH;
use attractive_perturbation_uvb3::AttractivePerturbationUVB3;
use attractive_perturbation_wca::AttractivePerturbationWCA;
use chain_bh_tptv::ChainBhTptv;
use chain_bh_tpty::ChainBH;
use hard_sphere_bh::HardSphereBH;
use hard_sphere_wca::HardSphereWCA;
//pub use polar::DQVariants;
use polar::{Dipole, DipoleQuadrupole, Quadrupole};
use reference_perturbation_bh::ReferencePerturbationBH;
use reference_perturbation_uvb3::ReferencePerturbationUVB3;
use reference_perturbation_wca::ReferencePerturbationWCA;

/// Type of Combination Rule.
#[derive(Clone)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
pub enum CombinationRule {
    ArithmeticPhi,
    GeometricPhi,
    GeometricPsi,
    OneFluidPsi,
}

/// Type of perturbation.
#[derive(Clone)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
pub enum Perturbation {
    BarkerHenderson,
    WeeksChandlerAndersen,
}

/// Order of the highest virial coefficient included in the model.
#[derive(Clone)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
pub enum VirialOrder {
    Second,
    Third,
}

/// Configuration options for uv-theory
#[derive(Clone)]
pub struct UVTheoryOptions {
    pub max_eta: f64,
    pub perturbation: Perturbation,
    pub virial_order: VirialOrder,
    pub combination_rule: CombinationRule,
    pub max_iter_cross_assoc: usize,
    pub tol_cross_assoc: f64,
    //pub dq_variant: DQVariants,
}

impl Default for UVTheoryOptions {
    fn default() -> Self {
        Self {
            max_eta: 0.5,
            perturbation: Perturbation::WeeksChandlerAndersen,
            virial_order: VirialOrder::Second,
            combination_rule: CombinationRule::ArithmeticPhi,
            max_iter_cross_assoc: 50,
            tol_cross_assoc: 1e-10,
            //dq_variant: DQVariants::DQ35,
        }
    }
}

/// uv-theory equation of state
pub struct UVTheory {
    parameters: Arc<UVParameters>,
    options: UVTheoryOptions,
    contributions: Vec<Box<dyn HelmholtzEnergy>>,
}

impl UVTheory {
    /// uv-theory with default options (WCA).
    pub fn new(parameters: Arc<UVParameters>) -> EosResult<Self> {
        Self::with_options(parameters, UVTheoryOptions::default())
    }

    /// uv-theory with provided options.
    pub fn with_options(
        parameters: Arc<UVParameters>,
        options: UVTheoryOptions,
    ) -> EosResult<Self> {
        let mut contributions: Vec<Box<dyn HelmholtzEnergy>> = Vec::with_capacity(8);

        match options.perturbation {
            Perturbation::BarkerHenderson => match options.virial_order {
                VirialOrder::Second => {
                    contributions.push(Box::new(HardSphereBH {
                        parameters: parameters.clone(),
                    }));
                    contributions.push(Box::new(ReferencePerturbationBH {
                        parameters: parameters.clone(),
                    }));
                    contributions.push(Box::new(AttractivePerturbationBH {
                        parameters: parameters.clone(),
                        combination_rule: options.combination_rule.clone(),
                    }));
                    if parameters.m.iter().any(|&mi| mi > 1.0) {
                        contributions.push(Box::new(ChainBhTptv {
                            parameters: parameters.clone(),
                        }));
                    }
                }
                VirialOrder::Third => {
                    return Err(EosError::Error(
                        "Third virial coefficient is not implemented for Barker-Henderson"
                            .to_string(),
                    ))
                }
            },
            Perturbation::WeeksChandlerAndersen => {
                contributions.push(Box::new(HardSphereWCA {
                    parameters: parameters.clone(),
                }));
                match options.virial_order {
                    VirialOrder::Second => {
                        contributions.push(Box::new(ReferencePerturbationWCA {
                            parameters: parameters.clone(),
                        }));
                        contributions.push(Box::new(AttractivePerturbationWCA {
                            parameters: parameters.clone(),
                            combination_rule: options.combination_rule.clone(),
                        }));
                    }
                    VirialOrder::Third => {
                        if parameters.sigma.len() > 1 {
                            return Err(EosError::Error(
                                "Third virial coefficient is not implemented for mixtures!"
                                    .to_string(),
                            ));
                        }
                        if parameters.att[0] != 6.0 {
                            return Err(EosError::Error(
                                "Third virial coefficient is not implemented for attractive exponents other than 6!"
                                    .to_string(),
                            ));
                        }
                        contributions.push(Box::new(ReferencePerturbationUVB3 {
                            parameters: parameters.clone(),
                        }));
                        contributions.push(Box::new(AttractivePerturbationUVB3 {
                            parameters: parameters.clone(),
                        }));
                    }
                }
            }
        }
        if !parameters.association.is_empty() {
            contributions.push(Box::new(Association::new(
                &parameters,
                &parameters.association,
                options.max_iter_cross_assoc,
                options.tol_cross_assoc,
            )));
        };
        if parameters.ndipole > 0 {
            contributions.push(Box::new(Dipole {
                parameters: parameters.clone(),
            }));
        };
        if parameters.nquadpole > 0 {
            contributions.push(Box::new(Quadrupole {
                parameters: parameters.clone(),
            }));
        };
        if parameters.ndipole > 0 && parameters.nquadpole > 0 {
            contributions.push(Box::new(DipoleQuadrupole {
                parameters: parameters.clone(),
                //variant: options.dq_variant,
            }));
        };
        Ok(Self {
            parameters,
            options,
            contributions,
        })
    }
}

impl Components for UVTheory {
    fn components(&self) -> usize {
        self.parameters.pure_records.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(
            Arc::new(self.parameters.subset(component_list)),
            self.options.clone(),
        )
        .expect("Not defined for mixture")
    }
}

impl Residual for UVTheory {
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * &self.parameters.m * self.parameters.sigma.mapv(|v| v.powi(3)) * moles)
                .sum()
    }

    fn contributions(&self) -> &[Box<dyn HelmholtzEnergy>] {
        &self.contributions
    }
    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        self.parameters.molarweight.clone() * GRAM / MOL
    }
}

// impl MolarWeight for UVTheory {
//     fn molar_weight(&self) -> SIArray1 {
//         self.parameters.molarweight.clone() * GRAM / MOL
//     }
// }

#[cfg(test)]
mod test {
    use super::*;

    use crate::uvtheory::parameters::utils::{test_parameters, test_parameters_mixture};
    use crate::uvtheory::parameters::*;
    use approx::assert_relative_eq;
    use feos_core::parameter::{Identifier, Parameter, PureRecord};
    use feos_core::si::{ANGSTROM, KELVIN, MOL, NAV, RGAS};
    use feos_core::State;
    use ndarray::arr1;
    use typenum::P3;

    #[test]
    fn helmholtz_energy_pure_bh_contributions() -> EosResult<()> {
        let p = test_parameters(1.0, 12.0, 6.0, 1.5, 1.5);
        let options = UVTheoryOptions {
            max_eta: 0.5,
            perturbation: Perturbation::BarkerHenderson,
            virial_order: VirialOrder::Second,
            combination_rule: CombinationRule::OneFluidPsi,
            max_iter_cross_assoc: 50,
            tol_cross_assoc: 1e-10,
            //dq_variant: DQVariants::DQ35,
        };
        let eos = Arc::new(UVTheory::with_options(Arc::new(p.clone()), options)?);

        let moles = arr1(&[2.0]);
        let reduced_temperature = 2.0; // p.epsilon_k[0];
        let reduced_density = 0.02; //* p.sigma[0].powi(3) * p.m[0];
        let reduced_volume = moles[0] / reduced_density;

        let m = moles / NAV;
        let s = State::new_nvt(
            &eos,
            reduced_temperature * p.epsilon_k[0] * KELVIN,
            reduced_volume * (p.sigma[0] * ANGSTROM).powi::<P3>(),
            &m,
        )?;
        let contributions = s.residual_helmholtz_energy_contributions();

        println!(
            "T* = {}, rho* = {}, m = {}, sigma = {}, epsilon_k = {}",
            reduced_temperature, reduced_density, p.m[0], p.sigma[0], p.epsilon_k[0]
        );
        for (name, value) in contributions.iter() {
            let a_red = (value / (RGAS * s.temperature * s.total_moles)).into_value(); //value.to_reduced(RGAS * s.temperature * s.total_moles)?;
            println!("{:<30}: A / NkT = {:>.10}", &name, a_red);
            // dbg!(&name, value);
        }
        println!(
            "{:<30}: A / NkT = {:>.10}",
            "Total",
            (s.residual_helmholtz_energy() / (RGAS * s.temperature * s.total_moles)).into_value() // s.residual_helmholtz_energy()
                                                                                                  //     .to_reduced(RGAS * s.temperature * s.total_moles)?
        );

        let a = (s.residual_helmholtz_energy() / s.total_moles / RGAS / s.temperature).into_value();
        assert_relative_eq!(a, -0.025994839368858807, max_relative = 1e-12); //wca
                                                                             //assert_relative_eq!(atot, -0.0259948394, max_relative = 1e-12);
        Ok(())
    }

    #[test]
    fn helmholtz_energy_mix_bh_contributions() -> EosResult<()> {
        let p = test_parameters_mixture(
            arr1(&[8.0, 4.0]),
            arr1(&[12.0, 12.0]),
            arr1(&[6.0, 6.0]),
            arr1(&[1.0, 2.0]),
            arr1(&[1.0, 2.0]),
        );
        let options = UVTheoryOptions {
            max_eta: 0.5,
            perturbation: Perturbation::BarkerHenderson,
            virial_order: VirialOrder::Second,
            combination_rule: CombinationRule::OneFluidPsi,
            max_iter_cross_assoc: 50,
            tol_cross_assoc: 1e-10,
            //dq_variant: DQVariants::DQ35,
        };
        let eos = Arc::new(UVTheory::with_options(Arc::new(p.clone()), options)?);

        let moles = arr1(&[2.0, 2.0]);
        let reduced_temperature = 2.0;
        let reduced_density = 0.02;
        let reduced_volume = moles.sum() / reduced_density;

        let m = moles / NAV;
        let s = State::new_nvt(
            &eos,
            reduced_temperature * p.epsilon_k[0] * KELVIN,
            reduced_volume * (p.sigma[0] * ANGSTROM).powi::<P3>(),
            &m,
        )?;

        let contributions = s.residual_helmholtz_energy_contributions();

        println!(
            "T* = {}, rho* = {}, m = {}, sigma = {}, epsilon_k = {}",
            reduced_temperature, reduced_density, p.m[0], p.sigma[0], p.epsilon_k[0]
        );
        for (name, value) in contributions.iter() {
            let a_red = (value / (RGAS * s.temperature * s.total_moles)).into_value();
            println!("{:<30}: A / NkT = {:>.10}", &name, a_red);
            // dbg!(&name, value);
        }
        println!(
            "{:<30}: A / NkT = {:>.10}",
            "Total",
            (s.residual_helmholtz_energy() / (RGAS * s.temperature * s.total_moles)).into_value()
        );
        assert!(1 == 2);
        Ok(())
    }

    #[test]
    fn helmholtz_energy_pure_wca() -> EosResult<()> {
        let sig = 3.7039;
        let eps_k = 150.03;
        let parameters = UVParameters::new_simple(1.0, 24.0, 6.0, sig, eps_k);
        let eos = Arc::new(UVTheory::new(Arc::new(parameters))?);

        let reduced_temperature = 4.0;
        let reduced_density = 1.0;
        let temperature = reduced_temperature * eps_k * KELVIN;
        let moles = arr1(&[2.0]) * MOL;
        let volume = (sig * ANGSTROM).powi::<P3>() / reduced_density * NAV * 2.0 * MOL;
        let s = State::new_nvt(&eos, temperature, volume, &moles).unwrap();
        let a =
            ((s.residual_helmholtz_energy() / s.total_moles) / (RGAS * temperature)).into_value();
        assert_relative_eq!(a, 2.972986567516, max_relative = 1e-12); //wca
        Ok(())
    }

    #[test]
    fn helmholtz_energy_pure_bh() -> EosResult<()> {
        let eps_k = 150.03;
        let sig = 3.7039;
        let rep = 24.0;
        let att = 6.0;
        let parameters = UVParameters::new_simple(1.0, rep, att, sig, eps_k);
        let options = UVTheoryOptions {
            max_eta: 0.5,
            perturbation: Perturbation::BarkerHenderson,
            virial_order: VirialOrder::Second,
            combination_rule: CombinationRule::OneFluidPsi,
            max_iter_cross_assoc: 50,
            tol_cross_assoc: 1e-10,
            //dq_variant: DQVariants::DQ35,
        };
        let eos = Arc::new(UVTheory::with_options(Arc::new(parameters), options)?);

        let reduced_temperature = 4.0;
        let reduced_density = 1.0;
        let temperature = reduced_temperature * eps_k * KELVIN;
        let moles = arr1(&[2.0]) * MOL;
        let volume = (sig * ANGSTROM).powi::<P3>() / reduced_density * NAV * 2.0 * MOL;
        let s = State::new_nvt(&eos, temperature, volume, &moles).unwrap();

        let a = (s.residual_helmholtz_energy() / s.total_moles / RGAS / temperature).into_value();

        //assert_relative_eq!(a, 2.993577305779432, max_relative = 1e-12); //Monomer Verison
        assert_relative_eq!(a, 2.8835102086093167, max_relative = 1e-12);
        Ok(())
    }

    #[test]
    fn helmholtz_energy_pure_uvb3() -> EosResult<()> {
        let eps_k = 150.03;
        let sig = 3.7039;
        let rep = 12.0;
        let att = 6.0;
        let parameters = UVParameters::new_simple(1.0, rep, att, sig, eps_k);
        let options = UVTheoryOptions {
            max_eta: 0.5,
            perturbation: Perturbation::WeeksChandlerAndersen,
            virial_order: VirialOrder::Third,
            combination_rule: CombinationRule::OneFluidPsi,
            max_iter_cross_assoc: 50,
            tol_cross_assoc: 1e-10,
            //dq_variant: DQVariants::DQ35,
        };
        let eos = Arc::new(UVTheory::with_options(Arc::new(parameters), options)?);

        let reduced_temperature = 4.0;
        let reduced_density = 0.5;
        let temperature = reduced_temperature * eps_k * KELVIN;
        let moles = arr1(&[2.0]) * MOL;
        let volume = (sig * ANGSTROM).powi::<P3>() / reduced_density * NAV * 2.0 * MOL;
        let s = State::new_nvt(&eos, temperature, volume, &moles).unwrap();
        let a = (s.residual_helmholtz_energy() / (RGAS * temperature * s.total_moles)).into_value();
        dbg!(a);
        assert_relative_eq!(a, 0.37659379124271003, max_relative = 1e-12);
        Ok(())
    }

    #[test]
    fn helmholtz_energy_mixtures_bh() -> EosResult<()> {
        // Mixture of equal components --> result must be the same as for pure fluid ///
        // component 1
        let rep1 = 24.0;
        let eps_k1 = 150.03;
        let sig1 = 3.7039;
        let r1 = UVRecord::new(
            1.0, rep1, 6.0, sig1, eps_k1, None, None, None, None, None, None, None,
        );
        let i = Identifier::new(None, None, None, None, None, None);
        // compontent 2
        let rep2 = 24.0;
        let eps_k2 = 150.03;
        let sig2 = 3.7039;
        let r2 = UVRecord::new(
            1.0, rep2, 6.0, sig2, eps_k2, None, None, None, None, None, None, None,
        );
        let j = Identifier::new(None, None, None, None, None, None);
        //////////////

        let pr1 = PureRecord::new(i, 1.0, r1);
        let pr2 = PureRecord::new(j, 1.0, r2);
        let pure_records = vec![pr1, pr2];
        let uv_parameters = UVParameters::new_binary(pure_records, None).unwrap();
        // state
        let reduced_temperature = 4.0;
        let eps_k_x = (eps_k1 + eps_k2) / 2.0; // Check rule!!
        let t_x = reduced_temperature * eps_k_x * KELVIN;
        let sig_x = (sig1 + sig2) / 2.0; // Check rule!!
        let reduced_density = 1.0;
        let moles = arr1(&[1.7, 0.3]) * MOL;
        let total_moles = moles.sum();
        let volume = (sig_x * ANGSTROM).powi::<P3>() / reduced_density * NAV * total_moles;

        // EoS
        let options = UVTheoryOptions {
            max_eta: 0.5,
            perturbation: Perturbation::BarkerHenderson,
            virial_order: VirialOrder::Second,
            combination_rule: CombinationRule::ArithmeticPhi,
            max_iter_cross_assoc: 50,
            tol_cross_assoc: 1e-10,
            //dq_variant: DQVariants::DQ35,
        };

        let eos_bh = Arc::new(UVTheory::with_options(Arc::new(uv_parameters), options)?);

        let state_bh = State::new_nvt(&eos_bh, t_x, volume, &moles).unwrap();
        let a_bh = (state_bh.residual_helmholtz_energy() / (RGAS * t_x * state_bh.total_moles))
            .into_value();

        assert_relative_eq!(a_bh, 2.993577305779432, max_relative = 1e-12);
        Ok(())
    }

    #[test]
    fn helmholtz_energy_wca_mixture() -> EosResult<()> {
        let p = test_parameters_mixture(
            arr1(&[1.0, 1.0]),
            arr1(&[12.0, 12.0]),
            arr1(&[6.0, 6.0]),
            arr1(&[1.0, 1.0]),
            arr1(&[1.0, 0.5]),
        );

        // state
        let reduced_temperature = 1.0;
        let t_x = reduced_temperature * p.epsilon_k[0] * KELVIN;
        let reduced_density = 0.9;
        let moles = arr1(&[0.4, 0.6]) * MOL;
        let total_moles = moles.sum();
        let volume = (p.sigma[0] * ANGSTROM).powi::<P3>() / reduced_density * NAV * total_moles;

        // EoS
        let eos_wca = Arc::new(UVTheory::new(Arc::new(p))?);
        let state_wca = State::new_nvt(&eos_wca, t_x, volume, &moles).unwrap();
        // let a_wca = state_wca
        //     .residual_helmholtz_energy()
        //     .to_reduced(RGAS * t_x * state_wca.total_moles)
        //     .unwrap();
        let a_wca = (state_wca.residual_molar_helmholtz_energy() / (RGAS * state_wca.temperature))
            .into_value();
        assert_relative_eq!(a_wca, -0.597791038364405, max_relative = 1e-5);
        Ok(())
    }

    // #[test]
    // fn helmholtz_energy_wca_mixture_different_sigma() -> EosResult<()> {
    //     let p = test_parameters_mixture(
    //         arr1(&[1.0, 1.0]),
    //         arr1(&[12.0, 12.0]),
    //         arr1(&[6.0, 6.0]),
    //         arr1(&[1.0, 2.0]),
    //         arr1(&[1.0, 0.5]),
    //     );

    //     // state
    //     let reduced_temperature = 1.5;
    //     let t_x = reduced_temperature * p.epsilon_k[0] * KELVIN;
    //     let sigma_x_3 = (0.4 + 0.6 * 8.0) * ANGSTROM.powi(3);
    //     let density = 0.52000000000000002 / sigma_x_3;
    //     let moles = arr1(&[0.4, 0.6]) * MOL;
    //     let total_moles = moles.sum();
    //     let volume = NAV * total_moles / density;

    //     // EoS
    //     let eos_wca = Arc::new(UVTheory::new(Arc::new(p))?);
    //     let state_wca = State::new_nvt(&eos_wca, t_x, volume, &moles).unwrap();
    //     let a_wca = state_wca
    //         .residual_helmholtz_energy()
    //         .to_reduced(RGAS * t_x * state_wca.total_moles)
    //         .unwrap();
    //     assert_relative_eq!(a_wca, -0.034206207363139396, max_relative = 1e-5);
    //     Ok(())
    // }
    #[test]
    fn helmholtz_energy_wca_mixture_different_sigma() -> EosResult<()> {
        let p = test_parameters_mixture(
            arr1(&[1.0, 1.0]),
            arr1(&[12.0, 12.0]),
            arr1(&[6.0, 6.0]),
            arr1(&[1.0, 2.0]),
            arr1(&[1.0, 0.5]),
        );

        // state
        let reduced_temperature = 1.5;
        let t_x = reduced_temperature * p.epsilon_k[0] * KELVIN;
        let sigma_x_3 = (0.4 + 0.6 * 8.0) * ANGSTROM.powi::<P3>();
        let density = 0.52000000000000002 / sigma_x_3;
        let moles = arr1(&[0.4, 0.6]) * MOL;
        let total_moles = moles.sum();
        let volume = NAV * total_moles / density;

        // EoS
        let eos_wca = Arc::new(UVTheory::new(Arc::new(p))?);
        let state_wca = State::new_nvt(&eos_wca, t_x, volume, &moles).unwrap();
        let a_wca = (state_wca.residual_molar_helmholtz_energy() / (RGAS * t_x)).into_value();
        assert_relative_eq!(a_wca, -0.034206207363139396, max_relative = 1e-5);
        Ok(())
    }
}
