//! SAFT-VR Mie with corresponding states (effective Feynman-Hibbs quantum parameters).
//!
//! SAFT-VR-CS applies temperature-dependent Feynman-Hibbs quantum corrections to the
//! SAFT-VR Mie segment parameters (σ_eff, ε_eff, λ_r_eff) and then evaluates the
//! full SAFT-VR Mie EoS (BH hard sphere + dispersion + chain) with those effective
//! parameters.
mod corresponding_states;
mod eos;
mod parameters;

pub use eos::{SaftVRCSMie, SaftVRCSOptions};
pub use parameters::{
    QuantumCorrection, SaftVRCSBinaryRecord, SaftVRCSPars, SaftVRCSRecord,
    SaftVRCSParameters,
};
