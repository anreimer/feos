use crate::association::{AssociationParameters, AssociationRecord, BinaryAssociationRecord};
use feos_core::parameter::Identifier;
use feos_core::parameter::{Parameter, PureRecord};
use lazy_static::lazy_static;
use ndarray::concatenate;
use ndarray::prelude::*;
use ndarray::{Array, Array1, Array2};
use num_dual::DualNum;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fmt::Write;

use crate::hard_sphere::{HardSphereProperties, MonomerShape};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NoRecord;

impl fmt::Display for NoRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "")
    }
}

/// uv-theory parameters for a pure substance
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct UVRecord {
    m: f64,
    rep: f64,
    att: f64,
    sigma: f64,
    epsilon_k: f64,
    /// Association parameters
    #[serde(flatten)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub association_record: Option<AssociationRecord>,
}

impl UVRecord {
    /// Single substance record for uv-theory
    pub fn new(
        m: f64,
        rep: f64,
        att: f64,
        sigma: f64,
        epsilon_k: f64,
        kappa_ab: Option<f64>,
        epsilon_k_ab: Option<f64>,
        na: Option<f64>,
        nb: Option<f64>,
        nc: Option<f64>,
    ) -> UVRecord {
        let association_record = if kappa_ab.is_none()
            && epsilon_k_ab.is_none()
            && na.is_none()
            && nb.is_none()
            && nc.is_none()
        {
            None
        } else {
            Some(AssociationRecord::new(
                kappa_ab.unwrap_or_default(),
                epsilon_k_ab.unwrap_or_default(),
                na.unwrap_or_default(),
                nb.unwrap_or_default(),
                nc.unwrap_or_default(),
            ))
        };
        UVRecord {
            m,
            rep,
            att,
            sigma,
            epsilon_k,
            association_record,
        }
    }
}

impl std::fmt::Display for UVRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UVRecord(m={}", self.m)?;
        write!(f, ", rep={}", self.rep)?;
        write!(f, ", att={}", self.att)?;
        write!(f, ", sigma={}", self.sigma)?;
        write!(f, ", epsilon_k={}", self.epsilon_k)?;
        if let Some(n) = &self.association_record {
            write!(f, ", association_record={}", n)?;
        }
        write!(f, ")")
    }
}

/// Binary interaction parameters
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct UVBinaryRecord {
    /// Binary dispersion interaction parameter
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub k_ij: f64,
    /// Binary association parameters
    #[serde(flatten)]
    association: Option<BinaryAssociationRecord>,
}

impl From<f64> for UVBinaryRecord {
    fn from(k_ij: f64) -> Self {
        Self {
            k_ij,
            association: None,
        }
    }
}

impl From<UVBinaryRecord> for f64 {
    fn from(binary_record: UVBinaryRecord) -> Self {
        binary_record.k_ij
    }
}

impl UVBinaryRecord {
    pub fn new(k_ij: Option<f64>, kappa_ab: Option<f64>, epsilon_k_ab: Option<f64>) -> Self {
        let k_ij = k_ij.unwrap_or_default();
        let association = if kappa_ab.is_none() && epsilon_k_ab.is_none() {
            None
        } else {
            Some(BinaryAssociationRecord::new(kappa_ab, epsilon_k_ab, None))
        };
        Self { k_ij, association }
    }
}

impl std::fmt::Display for UVBinaryRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut tokens = vec![];
        if !self.k_ij.is_zero() {
            tokens.push(format!("k_ij={}", self.k_ij));
        }
        if let Some(association) = self.association {
            if let Some(kappa_ab) = association.kappa_ab {
                tokens.push(format!("kappa_ab={}", kappa_ab));
            }
            if let Some(epsilon_k_ab) = association.epsilon_k_ab {
                tokens.push(format!("epsilon_k_ab={}", epsilon_k_ab));
            }
        }
        write!(f, "UVBinaryRecord({})", tokens.join(", "))
    }
}
lazy_static! {
/// Constants for BH temperature dependent HS diameter.
    static ref CD_BH: Array2<f64> = arr2(&[
        [0.0, 1.09360455168912E-02, 0.0],
        [-2.00897880971934E-01, -1.27074910870683E-02, 0.0],
        [
            1.40422470174053E-02,
            7.35946850956932E-02,
            1.28463973950737E-02,
        ],
        [
            3.71527116894441E-03,
            5.05384813757953E-03,
            4.91003312452622E-02,
        ],
    ]);
}

#[inline]
pub fn mie_prefactor<D: DualNum<f64> + Copy>(rep: D, att: D) -> D {
    rep / (rep - att) * (rep / att).powd(att / (rep - att))
}

#[inline]
pub fn mean_field_constant_f64(rep: f64, att: f64, x: f64) -> f64 {
    mie_prefactor(rep, att) * (x.powd(-att + 3.0) / (att - 3.0) - x.powd(-rep + 3.0) / (rep - 3.0))
}

#[inline]
pub fn mean_field_constant<D: DualNum<f64> + Copy>(rep: D, att: D, x: D) -> D {
    mie_prefactor(rep, att) * (x.powd(-att + 3.0) / (att - 3.0) - x.powd(-rep + 3.0) / (rep - 3.0))
}

/// Parameters for all substances for uv-theory equation of state and Helmholtz energy functional
#[derive(Clone)] //, Serialize, Deserialize)]
pub struct UVParameters {
    pub ncomponents: usize,
    pub m: Array1<f64>,
    pub rep: Array1<f64>,
    pub att: Array1<f64>,
    pub sigma: Array1<f64>,
    pub epsilon_k: Array1<f64>,
    pub association: AssociationParameters,
    pub molarweight: Array1<f64>,
    pub k_ij: Option<Array2<f64>>,
    pub rep_ij: Array2<f64>,
    pub att_ij: Array2<f64>,
    pub sigma_ij: Array2<f64>,
    pub eps_k_ij: Array2<f64>,
    pub cd_bh_pure: Vec<Array1<f64>>,
    pub cd_bh_binary: Array2<Array1<f64>>,
    pub pure_records: Vec<PureRecord<UVRecord>>,
    pub binary_records: Option<Array2<UVBinaryRecord>>,
}

impl Parameter for UVParameters {
    type Pure = UVRecord;
    type Binary = UVBinaryRecord;

    fn from_records(
        pure_records: Vec<PureRecord<Self::Pure>>,
        binary_records: Option<Array2<Self::Binary>>,
    ) -> Self {
        let n = pure_records.len();

        let mut molarweight = Array::zeros(n);
        let mut m = Array::zeros(n);
        let mut rep = Array::zeros(n);
        let mut att = Array::zeros(n);
        let mut sigma = Array::zeros(n);
        let mut epsilon_k = Array::zeros(n);
        let mut component_index = HashMap::with_capacity(n);
        let mut association_records = Vec::with_capacity(n);

        for (i, record) in pure_records.iter().enumerate() {
            component_index.insert(record.identifier.clone(), i);
            let r = &record.model_record;
            m[i] = r.m;
            rep[i] = r.rep;
            att[i] = r.att;
            sigma[i] = r.sigma;
            epsilon_k[i] = r.epsilon_k;
            association_records.push(r.association_record.into_iter().collect());

            // construction of molar weights for GC methods, see Builder
            molarweight[i] = record.molarweight;
        }

        let mut rep_ij = Array2::zeros((n, n));
        let mut att_ij = Array2::zeros((n, n));
        let mut sigma_ij = Array2::zeros((n, n));
        let mut eps_k_ij = Array2::zeros((n, n));
        let k_ij = binary_records.as_ref().map(|br| br.map(|br| br.k_ij));
        let binary_association: Vec<_> = binary_records
            .iter()
            .flat_map(|r| {
                r.indexed_iter()
                    .filter_map(|(i, record)| record.association.map(|r| (i, r)))
            })
            .collect();
        let association =
            AssociationParameters::new(&association_records, &sigma, &binary_association, None);
        for i in 0..n {
            rep_ij[[i, i]] = rep[i];
            att_ij[[i, i]] = att[i];
            sigma_ij[[i, i]] = sigma[i];
            eps_k_ij[[i, i]] = epsilon_k[i];
            for j in i + 1..n {
                rep_ij[[i, j]] = (rep[i] * rep[j]).sqrt();
                rep_ij[[j, i]] = rep_ij[[i, j]];
                att_ij[[i, j]] = (att[i] * att[j]).sqrt();
                att_ij[[j, i]] = att_ij[[i, j]];
                sigma_ij[[i, j]] = 0.5 * (sigma[i] + sigma[j]);
                sigma_ij[[j, i]] = sigma_ij[[i, j]];
                eps_k_ij[[i, j]] = (1.0 - k_ij.as_ref().map_or(0.0, |k_ij| k_ij[[i, j]]))
                    * (epsilon_k[i] * epsilon_k[j]).sqrt();
                eps_k_ij[[j, i]] = eps_k_ij[[i, j]];
            }
        }

        // BH temperature dependent HS diameter, eq. 21
        let cd_bh_pure: Vec<Array1<f64>> = rep.iter().map(|&mi| bh_coefficients(mi, 6.0)).collect();
        let cd_bh_binary =
            Array2::from_shape_fn((n, n), |(i, j)| bh_coefficients(rep_ij[[i, j]], 6.0));

        Self {
            ncomponents: n,
            m,
            rep,
            att,
            sigma,
            epsilon_k,
            association,
            molarweight,
            k_ij,
            rep_ij,
            att_ij,
            sigma_ij,
            eps_k_ij,
            cd_bh_pure,
            cd_bh_binary,
            pure_records,
            binary_records,
        }
    }

    fn records(&self) -> (&[PureRecord<UVRecord>], Option<&Array2<UVBinaryRecord>>) {
        (&self.pure_records, self.binary_records.as_ref())
    }
}

impl UVParameters {
    /// Parameters for a single substance with molar weight one and no (default) ideal gas contributions.
    pub fn new_simple(m: f64, rep: f64, att: f64, sigma: f64, epsilon_k: f64) -> Self {
        let model_record =
            UVRecord::new(m, rep, att, sigma, epsilon_k, None, None, None, None, None);
        let pure_record = PureRecord::new(Identifier::default(), 1.0, model_record);
        Self::new_pure(pure_record)
    }

    /// Parameters for a single substance with molar weight mw and association.
    pub fn new_simple_assoc(
        m: f64,
        rep: f64,
        att: f64,
        sigma: f64,
        epsilon_k: f64,
        kappa_ab: f64,
        epsilon_k_ab: f64,
        na: f64,
        nb: f64,
        nc: f64,
        mw: f64,
    ) -> Self {
        let model_record = UVRecord::new(
            m,
            rep,
            att,
            sigma,
            epsilon_k,
            Some(kappa_ab),
            Some(epsilon_k_ab),
            Some(na),
            Some(nb),
            Some(nc),
        );
        let pure_record = PureRecord::new(Identifier::default(), mw, model_record);
        Self::new_pure(pure_record)
    }

    /// Markdown representation of parameters.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();
        let o = &mut output;
        write!(
            o,
            "|component|molarweight|$m$|$\\sigma$|$\\varepsilon$|$\\mu$|$Q$|$\\kappa_{{AB}}$|$\\varepsilon_{{AB}}$|$N_A$|$N_B$|$N_C$|\n|-|-|-|-|-|-|-|-|-|-|-|-|"
        )
        .unwrap();
        for (i, record) in self.pure_records.iter().enumerate() {
            let component = record.identifier.name.clone();
            let component = component.unwrap_or(format!("Component {}", i + 1));
            let association = record
                .model_record
                .association_record
                .unwrap_or_else(|| AssociationRecord::new(0.0, 0.0, 0.0, 0.0, 0.0));
            write!(
                o,
                "\n|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|",
                component,
                record.molarweight,
                record.model_record.m,
                record.model_record.sigma,
                record.model_record.epsilon_k,
                association.kappa_ab,
                association.epsilon_k_ab,
                association.na,
                association.nb,
                association.nc
            )
            .unwrap();
        }

        output
    }
}

impl HardSphereProperties for UVParameters {
    fn monomer_shape<N: DualNum<f64>>(&self, _: N) -> MonomerShape<N> {
        MonomerShape::NonSpherical(self.m.mapv(N::from))
    }

    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> Array1<D> {
        self.cd_bh_pure
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let t = temperature / self.epsilon_k[i];
                let d = t.powf(0.25) * c[1] + t.powf(0.75) * c[2] + t.powf(1.25) * c[3];
                (t * c[0] + d * (t + 1.0).ln() + t.powi(2) * c[4] + 1.0).powf(-0.5 / self.rep[i])
                    * self.sigma[i]
            })
            .collect()
    }
}

fn bh_coefficients(rep: f64, att: f64) -> Array1<f64> {
    let inv_a76 = 1.0 / mean_field_constant(7.0, att, 1.0);
    let am6 = mean_field_constant(rep, att, 1.0);
    let alpha = 1.0 / am6 - inv_a76;
    let c0 = arr1(&[-2.0 * rep / ((att - rep) * mie_prefactor(rep, att))]);
    concatenate![Axis(0), c0, CD_BH.dot(&arr1(&[1.0, alpha, alpha * alpha]))]
}

#[cfg(test)]
pub mod utils {
    use super::*;
    use feos_core::parameter::{Identifier, PureRecord};
    use std::f64;

    pub fn test_parameters(m: f64, rep: f64, att: f64, sigma: f64, epsilon: f64) -> UVParameters {
        let identifier = Identifier::new(Some("1"), None, None, None, None, None);
        let model_record = UVRecord::new(m, rep, att, sigma, epsilon, None, None, None, None, None);
        let pr = PureRecord::new(identifier, 1.0, model_record);
        UVParameters::new_pure(pr)
    }

    pub fn test_parameters_mixture(
        m: Array1<f64>,
        rep: Array1<f64>,
        att: Array1<f64>,
        sigma: Array1<f64>,
        epsilon: Array1<f64>,
    ) -> UVParameters {
        let identifier = Identifier::new(Some("1"), None, None, None, None, None);
        let model_record = UVRecord::new(
            m[0], rep[0], att[0], sigma[0], epsilon[0], None, None, None, None, None,
        );
        let pr1 = PureRecord::new(identifier, 1.0, model_record);
        //
        let identifier2 = Identifier::new(Some("1"), None, None, None, None, None);
        let model_record2 = UVRecord::new(
            m[0], rep[1], att[1], sigma[1], epsilon[1], None, None, None, None, None,
        );
        let pr2 = PureRecord::new(identifier2, 1.0, model_record2);
        let pure_records = vec![pr1, pr2];
        UVParameters::new_binary(pure_records, None)
    }

    pub fn methane_parameters(rep: f64, att: f64) -> UVParameters {
        let identifier = Identifier::new(Some("1"), None, None, None, None, None);
        let model_record =
            UVRecord::new(1.0, rep, att, 3.7039, 150.03, None, None, None, None, None);
        let pr = PureRecord::new(identifier, 1.0, model_record);
        UVParameters::new_pure(pr)
    }
}
