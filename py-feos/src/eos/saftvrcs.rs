use super::PyEquationOfState;
use crate::ideal_gas::IdealGasModel;
use crate::parameter::PyParameters;
use crate::residual::ResidualModel;
use feos::saftvrcs::{
    CombiningRule, QuantumCorrection, SaftVRCSBinaryRecord, SaftVRCSMie, SaftVRCSOptions, SaftVRCSRecord,
};
use feos_core::{EquationOfState, ResidualDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

/// Feynman-Hibbs quantum correction for SAFT-VR-CS.
///
/// Parameters
/// ----------
/// c_sigma : list[float] of length 3, optional
///     Scaling coefficients for the effective sigma correction.
/// c_epsilon_k : list[float] of length 3, optional
///     Scaling coefficients for the effective epsilon_k correction.
/// c_lr : list[float] of length 5, optional
///     Scaling coefficients for the effective repulsive exponent correction.
#[pyclass(name = "SaftVRCSQuantumCorrection")]
#[derive(Clone)]
pub struct PySaftVRCSQuantumCorrection(QuantumCorrection);

#[pymethods]
impl PySaftVRCSQuantumCorrection {
    #[staticmethod]
    #[pyo3(signature = (c_sigma=None, c_epsilon_k=None, c_lr=None))]
    fn feynman_hibbs1(
        c_sigma: Option<[f64; 5]>,
        c_epsilon_k: Option<[f64; 5]>,
        c_lr: Option<[f64; 5]>,
    ) -> Self {
        Self(QuantumCorrection::FeynmanHibbs1 {
            c_sigma,
            c_epsilon_k,
            c_lr,
        })
    }

    fn __repr__(&self) -> String {
        self.0.to_string()
    }
}

/// Pure-component parameters for SAFT-VR-CS.
///
/// Parameters
/// ----------
/// m : float
///     Segment number (chain length).
/// sigma : float
///     Segment diameter in Angstrom.
/// epsilon_k : float
///     Dispersion energy ε/k_B in Kelvin.
/// lr : float
///     Repulsive Mie exponent.
/// la : float
///     Attractive Mie exponent.
/// quantum_correction : SaftVRCSQuantumCorrection, optional
///     Feynman-Hibbs quantum correction parameters. ``None`` (default) uses
///     the default prefactors; every component is mapped onto effective parameters.
#[pyclass(name = "SaftVRCSRecord")]
#[derive(Clone)]
pub struct PySaftVRCSRecord(SaftVRCSRecord);

#[pymethods]
impl PySaftVRCSRecord {
    #[new]
    #[pyo3(signature = (m, sigma, epsilon_k, lr, la, quantum_correction=None))]
    fn new(
        m: f64,
        sigma: f64,
        epsilon_k: f64,
        lr: f64,
        la: f64,
        quantum_correction: Option<PySaftVRCSQuantumCorrection>,
    ) -> Self {
        Self(SaftVRCSRecord::new(
            m,
            sigma,
            epsilon_k,
            lr,
            la,
            quantum_correction.map(|qc| qc.0),
        ))
    }

    fn __repr__(&self) -> String {
        self.0.to_string()
    }

    #[getter]
    fn get_m(&self) -> f64 {
        self.0.m
    }
    #[getter]
    fn get_sigma(&self) -> f64 {
        self.0.sigma
    }
    #[getter]
    fn get_epsilon_k(&self) -> f64 {
        self.0.epsilon_k
    }
    #[getter]
    fn get_lr(&self) -> f64 {
        self.0.lr
    }
    #[getter]
    fn get_la(&self) -> f64 {
        self.0.la
    }
    #[getter]
    fn get_quantum_correction(&self) -> Option<PySaftVRCSQuantumCorrection> {
        self.0
            .quantum_correction
            .as_ref()
            .map(|qc| PySaftVRCSQuantumCorrection(qc.clone()))
    }
}

/// Binary interaction parameters for SAFT-VR-CS.
///
/// Parameters
/// ----------
/// k_ij : float
///     Binary dispersion energy interaction parameter.
/// gamma_ij : float
///     Binary repulsive exponent interaction parameter.
#[pyclass(name = "SaftVRCSBinaryRecord")]
#[derive(Clone)]
pub struct PySaftVRCSBinaryRecord(SaftVRCSBinaryRecord);

#[pymethods]
impl PySaftVRCSBinaryRecord {
    #[new]
    #[pyo3(signature = (k_ij=0.0, gamma_ij=0.0))]
    fn new(k_ij: f64, gamma_ij: f64) -> Self {
        Self(SaftVRCSBinaryRecord { k_ij, gamma_ij })
    }

    fn __repr__(&self) -> String {
        self.0.to_string()
    }

    #[getter]
    fn get_k_ij(&self) -> f64 {
        self.0.k_ij
    }
    #[getter]
    fn get_gamma_ij(&self) -> f64 {
        self.0.gamma_ij
    }
    #[setter]
    fn set_k_ij(&mut self, k_ij: f64) {
        self.0.k_ij = k_ij
    }
    #[setter]
    fn set_gamma_ij(&mut self, gamma_ij: f64) {
        self.0.gamma_ij = gamma_ij
    }
}

#[pymethods]
impl PyEquationOfState {
    /// SAFT-VR Mie equation of state with Feynman-Hibbs quantum effective parameters.
    ///
    /// Parameters
    /// ----------
    /// parameters : Parameters
    ///     The SAFT-VR-CS parameters. Build using ``Parameters`` with
    ///     ``SaftVRCSRecord`` (and optionally ``SaftVRCSBinaryRecord``) as model records.
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// combining_rule : str, optional
    ///     Route to the effective unlike parameters. ``"effective"`` (default) combines
    ///     the effective pure-component parameters. ``"classical"`` combines the
    ///     classical parameters and maps the pair at m_ij = 2 m_i m_j / (m_i + m_j)
    ///     with prefactors (zeta_i + zeta_j) / 2.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, combining_rule="effective"),
        text_signature = r#"(parameters, max_eta=0.5, combining_rule="effective")"#
    )]
    fn saftvrcs_mie(
        parameters: PyParameters,
        max_eta: f64,
        combining_rule: &str,
    ) -> PyResult<Self> {
        let combining_rule = match combining_rule.to_lowercase().as_str() {
            "effective" => CombiningRule::Effective,
            "classical" => CombiningRule::Classical,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "combining_rule must be 'effective' or 'classical', got '{combining_rule}'"
                )));
            }
        };
        let options = SaftVRCSOptions {
            max_eta,
            combining_rule,
        };
        let residual = ResidualModel::SaftVRCSMie(SaftVRCSMie::with_options(
            parameters.try_convert()?,
            options,
        ));
        let ideal_gas = vec![IdealGasModel::NoModel; residual.components()];
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }
}
