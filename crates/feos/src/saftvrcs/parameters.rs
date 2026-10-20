use crate::hard_sphere::{HardSphereProperties, MonomerShape};
use feos_core::parameter::Parameters;
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use quantity::{GRAM, MOL};
use serde::{Deserialize, Serialize};

/// Feynman-Hibbs quantum correction options
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum QuantumCorrection {
    FeynmanHibbs1 {
        #[serde(skip_serializing_if = "Option::is_none")]
        c_sigma: Option<[f64; 5]>,
        #[serde(skip_serializing_if = "Option::is_none")]
        c_epsilon_k: Option<[f64; 5]>,
        #[serde(skip_serializing_if = "Option::is_none")]
        c_lr: Option<[f64; 5]>,
    },
}

impl std::fmt::Display for QuantumCorrection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FeynmanHibbs1 {
                c_sigma,
                c_epsilon_k,
                c_lr,
            } => write!(
                f,
                "FeynmanHibbs1(c_sigma={:?}, c_epsilon_k={:?}, c_lr={:?})",
                c_sigma.unwrap_or([1.0, 1.0, 1.0, 0.0, 0.0]),
                c_epsilon_k.unwrap_or([1.0, 1.0, 1.0, 0.0, 0.0]),
                c_lr.unwrap_or([1.0; 5]),
            ),
        }
    }
}

/// Default prefactors of the effective sigma and epsilon_k functions.
pub(crate) const DEFAULT_C_SIGMA_EPSILON: [f64; 5] = [1.0, 1.0, 1.0, 0.0, 0.0];
/// Default prefactors of the effective repulsive exponent function.
pub(crate) const DEFAULT_C_LR: [f64; 5] = [1.0; 5];

/// Route to the effective unlike parameters sigma_ij, epsilon_k_ij and lr_ij.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CombiningRule {
    /// Combine the effective pure-component parameters (each evaluated at its own mass)
    /// with the SAFT-VR Mie combining rules.
    #[default]
    Effective,
    /// Combine the classical parameters with the SAFT-VR Mie combining rules, then map
    /// the pair onto effective parameters at the pair mass m_ij = 2 m_i m_j / (m_i + m_j)
    /// with prefactors zeta_ij = (zeta_i + zeta_j) / 2.
    Classical,
}

/// SAFT-VR-CS pure-component parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SaftVRCSRecord {
    /// Segment number
    pub m: f64,
    /// Segment diameter in Angstrom
    pub sigma: f64,
    /// Dispersion energy in Kelvin
    pub epsilon_k: f64,
    /// Repulsive Mie exponent
    pub lr: f64,
    /// Attractive Mie exponent
    pub la: f64,
    /// Prefactors of the effective-parameter functions; `None` uses the defaults
    /// (every component is mapped onto effective parameters).
    #[serde(skip_serializing_if = "Option::is_none", flatten)]
    pub quantum_correction: Option<QuantumCorrection>,
}

impl SaftVRCSRecord {
    pub fn new(
        m: f64,
        sigma: f64,
        epsilon_k: f64,
        lr: f64,
        la: f64,
        quantum_correction: Option<QuantumCorrection>,
    ) -> Self {
        Self {
            m,
            sigma,
            epsilon_k,
            lr,
            la,
            quantum_correction,
        }
    }
}

impl std::fmt::Display for SaftVRCSRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SaftVRCSRecord(m={}, sigma={}, epsilon_k={}, lr={}, la={}",
            self.m, self.sigma, self.epsilon_k, self.lr, self.la
        )?;
        if let Some(qc) = &self.quantum_correction {
            write!(f, ", quantum_correction={}", qc)?;
        }
        write!(f, ")")
    }
}

/// SAFT-VR-CS binary interaction parameters
#[derive(Serialize, Deserialize, Clone, Default, Debug)]
pub struct SaftVRCSBinaryRecord {
    /// Binary dispersion energy interaction parameter
    #[serde(default)]
    pub k_ij: f64,
    /// Binary repulsive exponent interaction parameter
    #[serde(default)]
    pub gamma_ij: f64,
    /// Binary segment diameter interaction parameter (as in SAFT-VRQ Mie):
    /// sigma_ij = (1 - l_ij) (sigma_i + sigma_j) / 2
    #[serde(default)]
    pub l_ij: f64,
}

impl std::fmt::Display for SaftVRCSBinaryRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SaftVRCSBinaryRecord(k_ij={}, gamma_ij={}, l_ij={})",
            self.k_ij, self.gamma_ij, self.l_ij
        )
    }
}

/// Parameter set for SAFT-VR-CS (SAFT-VR Mie with effective parameters)
pub type SaftVRCSParameters = Parameters<SaftVRCSRecord, SaftVRCSBinaryRecord, ()>;

/// Raw (temperature-independent) SAFT-VR-CS parameters in accessible format
pub struct SaftVRCSPars {
    pub ncomponents: usize,
    pub m: DVector<f64>,
    pub sigma: DVector<f64>,
    pub epsilon_k: DVector<f64>,
    pub lr: DVector<f64>,
    pub la: DVector<f64>,
    pub k_ij: DMatrix<f64>,
    pub gamma_ij: DMatrix<f64>,
    pub l_ij: DMatrix<f64>,
    pub molarweight: DVector<f64>,
    pub quantum_correction: Vec<Option<QuantumCorrection>>,
    pub combining_rule: CombiningRule,
}

impl SaftVRCSPars {
    pub fn new(parameters: &SaftVRCSParameters) -> Self {
        let ncomponents = parameters.pure.len();
        let [m, sigma, epsilon_k] = parameters.collate(|pr| [pr.m, pr.sigma, pr.epsilon_k]);
        let [lr, la] = parameters.collate(|pr| [pr.lr, pr.la]);
        let [k_ij, gamma_ij, l_ij] =
            parameters.collate_binary(|b| [b.k_ij, b.gamma_ij, b.l_ij]);
        let molarweight = parameters.molar_weight.clone().convert_into(GRAM / MOL);
        let quantum_correction = parameters
            .pure
            .iter()
            .map(|pr| pr.model_record.quantum_correction.clone())
            .collect();

        Self {
            ncomponents,
            m,
            sigma,
            epsilon_k,
            lr,
            la,
            k_ij,
            gamma_ij,
            l_ij,
            molarweight,
            quantum_correction,
            combining_rule: CombiningRule::default(),
        }
    }

    pub fn with_combining_rule(mut self, combining_rule: CombiningRule) -> Self {
        self.combining_rule = combining_rule;
        self
    }
}

impl HardSphereProperties for SaftVRCSPars {
    fn monomer_shape<N: DualNum<f64> + Copy>(&self, _: N) -> MonomerShape<'_, N> {
        MonomerShape::NonSpherical(self.m.map(N::from))
    }

    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> DVector<D> {
        use super::corresponding_states::CorrespondingParameters;
        let cp = CorrespondingParameters::new(self, temperature);
        DVector::from_fn(self.ncomponents, |i, _| cp.hs_diameter_ij(i, i))
    }
}

impl std::fmt::Display for SaftVRCSPars {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SaftVRCSPars(")?;
        write!(f, "\n\tmolarweight={}", self.molarweight)?;
        write!(f, "\n\tm={}", self.m)?;
        write!(f, "\n\tsigma={}", self.sigma)?;
        write!(f, "\n\tepsilon_k={}", self.epsilon_k)?;
        write!(f, "\n\tlr={}", self.lr)?;
        write!(f, "\n\tla={}", self.la)?;
        write!(f, "\n)")
    }
}

#[cfg(test)]
pub mod utils {
    use super::*;
    use feos_core::parameter::{Identifier, PureRecord};
    use quantity::{KILOGRAM, NAV};

    pub fn test_parameters(
        m: f64,
        sigma: f64,
        epsilon_k: f64,
        lr: f64,
        la: f64,
    ) -> SaftVRCSParameters {
        let pr = PureRecord::new(
            Identifier::default(),
            m * 12.011,
            SaftVRCSRecord::new(m, sigma, epsilon_k, lr, la, None),
        );
        SaftVRCSParameters::new_pure(pr).unwrap()
    }

    #[allow(dead_code)]
    pub fn helium_parameters() -> SaftVRCSParameters {
        let to_mass_per_molecule = (GRAM / MOL / NAV / KILOGRAM).into_value();
        let molarweight = 4.002601643881807;
        let _ = molarweight * to_mass_per_molecule;
        let pr = PureRecord::new(
            Identifier::new(Some("He"), None, None, None, None, None),
            molarweight,
            SaftVRCSRecord::new(
                1.0,
                2.6494,
                4.4483,
                14.953,
                6.0,
                Some(QuantumCorrection::FeynmanHibbs1 {
                    c_sigma: None,
                    c_epsilon_k: None,
                    c_lr: None,
                }),
            ),
        );
        SaftVRCSParameters::new_pure(pr).unwrap()
    }
}
