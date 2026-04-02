use super::PyEquationOfState;
use crate::ideal_gas::IdealGasModel;
use crate::parameter::PyParameters;
use crate::residual::ResidualModel;
use feos::uvtheory::{Perturbation, UVTheory, UVTheoryOptions};
use feos_core::{EquationOfState, ResidualDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

#[pymethods]
impl PyEquationOfState {
    /// UV-Theory equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : UVTheoryParameters
    ///     The parameters of the UV-theory equation of state to use.
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// perturbation : "BH" | "WCA" | "WCA_B3", optional
    ///     Division type of the Mie potential. Defaults to "WCA".
    /// max_iter_cross_assoc : unsigned integer, optional
    ///     Maximum number of iterations for cross association. Defaults to 50.
    /// tol_cross_assoc : float, optional
    ///     Tolerance for convergence of cross association. Defaults to 1e-10.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The UV-Theory equation of state that can be used to compute thermodynamic
    ///     states.
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, perturbation="WCA", max_iter_cross_assoc=50, tol_cross_assoc=1e-10),
        text_signature = r#"(parameters, max_eta=0.5, perturbation="WCA", max_iter_cross_assoc=50, tol_cross_assoc=1e-10)"#
    )]
    fn uvtheory(parameters: PyParameters, max_eta: f64, perturbation: &str,  max_iter_cross_assoc: usize,
        tol_cross_assoc: f64) -> PyResult<Self> {
        let perturbation = match perturbation {
            "BH" => Perturbation::BarkerHenderson,
            "WCA" => Perturbation::WeeksChandlerAndersen,
            "WCA_B3" => Perturbation::WeeksChandlerAndersenB3,
            _ => {
                return Err(PyErr::new::<PyValueError, _>(
                    r#"perturbation must be "BH", "WCA" or "WCA_B3""#.to_string(),
                ))
            }
        };
        let options = UVTheoryOptions {
            max_eta,
            perturbation,
            max_iter_cross_assoc,
            tol_cross_assoc
        };
        let residual =
            ResidualModel::UVTheory(UVTheory::with_options(parameters.try_convert()?, options));
        let ideal_gas = vec![IdealGasModel::NoModel; residual.components()];
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }
}
