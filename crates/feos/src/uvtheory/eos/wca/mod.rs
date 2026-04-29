use crate::hard_sphere::HardSphere;
use crate::uvtheory::parameters::UVTheoryPars;
use feos_core::StateHD;
use feos_core::parameter::{AssociationParameters, AssociationSite, BinaryParameters};
mod attractive_perturbation;
mod attractive_perturbation_uvb3;
mod hard_sphere;
mod reference_perturbation;
mod reference_perturbation_uvb3;
use crate::uvtheory::parameters::UVTheoryAssociationRecord;
use attractive_perturbation::AttractivePerturbation;
use attractive_perturbation_uvb3::AttractivePerturbationB3;
use num_dual::DualNum;
use reference_perturbation::ReferencePerturbation;
use reference_perturbation_uvb3::ReferencePerturbationB3;

pub struct WeeksChandlerAndersen;

impl WeeksChandlerAndersen {
    pub fn residual_helmholtz_energy_contributions<D: DualNum<f64> + Copy>(
        &self,
        parameters: &UVTheoryPars,
        assoc_params: &AssociationParameters<UVTheoryAssociationRecord>,
        state: &StateHD<D>,
    ) -> Vec<(&'static str, D)> {
        vec![
            (
                "Hard Sphere (WCA)",
                HardSphere.helmholtz_energy_density(parameters, state),
            ),
            (
                "Reference Perturbation (WCA)",
                ReferencePerturbation.helmholtz_energy_density(parameters, state),
            ),
            (
                "Attractive Perturbation (WCA)",
                AttractivePerturbation.helmholtz_energy_density(parameters, assoc_params, state),
            ),
        ]
    }
}

pub struct WeeksChandlerAndersenB3;

impl WeeksChandlerAndersenB3 {
    pub fn residual_helmholtz_energy_contributions<D: DualNum<f64> + Copy>(
        &self,
        parameters: &UVTheoryPars,
        state: &StateHD<D>,
    ) -> Vec<(&'static str, D)> {
        vec![
            (
                "Hard Sphere (WCA)",
                HardSphere.helmholtz_energy_density(parameters, state),
            ),
            (
                "Reference Perturbation (WCA B3)",
                ReferencePerturbationB3.helmholtz_energy_density(parameters, state),
            ),
            (
                "Attractive Perturbation (WCA B3)",
                AttractivePerturbationB3.helmholtz_energy_density(parameters, state),
            ),
        ]
    }
}
