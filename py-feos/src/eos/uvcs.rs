use super::PyEquationOfState;
use crate::ideal_gas::IdealGasModel;
use crate::parameter::PyParameters;
use crate::residual::ResidualModel;
use feos::uvcs::UVCSTheory;
use feos_core::{EquationOfState, ResidualDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;
use feos::uvcs::{QuantumCorrection, UVCSBinaryRecord, UVCSPars, UVCSRecord};

#[pyclass(name = "QuantumCorrection")]
#[derive(Clone)]
pub struct PyQuantumCorrection(QuantumCorrection);

#[pymethods]
impl PyQuantumCorrection {
    #[staticmethod]
    #[pyo3(signature = (c_sigma=None, c_epsilon_k=None, c_rep=None))]
    fn feynman_hibbs1(
        c_sigma: Option<[f64; 3]>,
        c_epsilon_k: Option<[f64; 3]>,
        c_rep: Option<[f64; 5]>,
    ) -> Self {
        Self(QuantumCorrection::FeynmanHibbs1 {
            c_sigma,
            c_epsilon_k,
            c_rep,
        })
    }
}


/// Create a set of UV Theory parameters from records.
#[pyclass(name = "UVCSRecord")]
#[derive(Clone)]
pub struct PyUVCSRecord(UVCSRecord);

#[pymethods]
impl PyUVCSRecord {
    #[new]
    #[pyo3(signature = (rep, att, sigma, epsilon_k, quantum_correction=None))]
    fn new(
        rep: f64,
        att: f64,
        sigma: f64,
        epsilon_k: f64,
        quantum_correction: Option<PyQuantumCorrection>,
    ) -> Self {
        Self(UVCSRecord::new(
            rep,
            att,
            sigma,
            epsilon_k,
            quantum_correction.map(|qc| qc.0),
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }

    #[getter]
    fn get_quantum_correction(&self) -> Option<PyQuantumCorrection> {
        if let Some(qc) = self.0.quantum_correction.as_ref() {
            Some(PyQuantumCorrection(qc.clone()))
        } else {
            None
        }
    }
}

/// Create a binary record from k_ij and l_ij values.
#[pyclass(name = "UVCSBinaryRecord")]
#[derive(Clone)]
pub struct PyUVCSBinaryRecord(UVCSBinaryRecord);

#[pymethods]
impl PyUVCSBinaryRecord {
    #[new]
    #[pyo3(text_signature = "(k_ij, l_ij)")]
    fn new(k_ij: f64, l_ij: f64) -> Self {
        Self(UVCSBinaryRecord { k_ij, l_ij })
    }

    #[getter]
    fn get_k_ij(&self) -> f64 {
        self.0.k_ij
    }

    #[getter]
    fn get_l_ij(&self) -> f64 {
        self.0.l_ij
    }

    #[setter]
    fn set_k_ij(&mut self, k_ij: f64) {
        self.0.k_ij = k_ij
    }

    #[setter]
    fn set_l_ij(&mut self, l_ij: f64) {
        self.0.l_ij = l_ij
    }
}

// impl_json_handling!(PyUVCSRecord);
// impl_binary_record!(UVCSBinaryRecord, PyUVCSBinaryRecord);

#[pyclass(name = "UVCSParameters")]
#[derive(Clone)]
pub struct PyUVCSParameters(pub Arc<UVCSParameters>);

#[pymethods]
impl PyUVCSParameters {
    /// Create a set of UV Theory parameters from lists.
    ///
    /// Parameters
    /// ----------
    /// rep : List[float]
    ///     repulsive exponents
    /// att : List[float]
    ///     attractive exponents
    /// sigma : List[float]
    ///     Mie diameter in units of Angstrom
    /// epsilon_k : List[float]
    ///     Mie energy parameter in units of Kelvin
    ///
    /// Returns
    /// -------
    /// UVCSParameters
    #[pyo3(text_signature = "(rep, att, sigma, epsilon_k)")]
    #[staticmethod]
    fn from_lists(
        rep: Vec<f64>,
        att: Vec<f64>,
        sigma: Vec<f64>,
        epsilon_k: Vec<f64>,
        quantum_correction: Vec<Option<PyQuantumCorrection>>,
    ) -> PyResult<Self> {
        let n = rep.len();
        let pure_records = (0..n)
            .map(|i| {
                let identifier = Identifier::new(
                    Some(format!("{}", i).as_str()),
                    None,
                    None,
                    None,
                    None,
                    None,
                );
                let model_record = UVCSRecord::new(
                    rep[i],
                    att[i],
                    sigma[i],
                    epsilon_k[i],
                    quantum_correction[i].as_ref().map(|qc| qc.0.clone()),
                );
                PureRecord::new(identifier, 1.0, model_record)
            })
            .collect();
        Ok(Self(Arc::new(UVCSParameters::from_records(
            pure_records,
            None,
        )?)))
    }

/// Create UV Theory parameters for pure substance.
    ///
    /// Parameters
    /// ----------
    /// rep : float
    ///     repulsive exponents
    /// att : float
    ///     attractive exponents
    /// sigma : float
    ///     Mie diameter in units of Angstrom
    /// epsilon_k : float
    ///     Mie energy parameter in units of Kelvin
    ///
    /// Returns
    /// -------
    /// UVCSParameters
    ///
    /// # Info
    ///
    /// Molar weight is one. No ideal gas contribution is considered.
    #[pyo3(text_signature = "(rep, att, sigma, epsilon_k)")]
    #[staticmethod]
    fn new_simple(rep: f64, att: f64, sigma: f64, epsilon_k: f64) -> PyResult<Self> {
        Ok(Self(Arc::new(UVCSParameters::new_simple(
            rep, att, sigma, epsilon_k,
        )?)))
    }

    /// Print effective parameters
    fn print_effective_parameters(&self, temperature: f64) -> String {
        self.0.print_effective_parameters(temperature)
    }

    fn _repr_markdown_(&self) -> String {
        self.0.to_markdown()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

// impl_pure_record!(UVCSRecord, PyUVCSRecord);
// impl_parameter!(
//     UVCSParameters,
//     PyUVCSParameters,
//     PyUVCSRecord,
//     PyUVCSBinaryRecord
// );


#[pymethods]
impl PyEquationOfState {
    /// UV-CS-Theory equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : UVTheoryParameters
    ///     The parameters of the UV-theory equation of state to use.
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The UV-CS-Theory equation of state that can be used to compute thermodynamic
    ///     states.
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5),
        text_signature = r#"(parameters, max_eta=0.5)"#
    )]
    fn uvcstheory(parameters: PyParameters, max_eta: f64) -> PyResult<Self> {
        let residual =
            ResidualModel::UVCSTheory(UVCSTheory::with_options(parameters.try_convert()?, max_eta));
        let ideal_gas = vec![IdealGasModel::NoModel; residual.components()];
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }
}