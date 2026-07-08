#![allow(clippy::excessive_precision)]

use crate::saftvrcs::corresponding_states::CorrespondingParameters;
use feos_core::StateHD;
use nalgebra::{DMatrix, DVector};
use num_dual::{Dual, DualNum};
use num_traits::Zero;
use std::f64::consts::{FRAC_PI_6, PI};

/// Pre-computed density-dependent properties used in the dispersion integrals.
#[derive(Debug)]
pub struct Properties<D> {
    pub diameter: DVector<D>,
    pub segment_density: D,
    pub segment_molefracs: DVector<D>,
    pub mean_segment_number: D,
    pub zeta_x: D,
    pub zeta_x_bar: D,
    /// k-values for the HS pair correlation function at contact
    pub k0: [D; 4],
}

impl<D: DualNum<f64> + Copy + Zero> Properties<D> {
    pub fn new(
        cp: &CorrespondingParameters<D>,
        state: &StateHD<D>,
        diameter: &DVector<D>,
    ) -> Self {
        let n = cp.ncomponents;
        let x = &state.molefracs;

        let mean_segment_number: D = x.component_mul(&cp.m.map(D::from)).sum();
        let xs = x.component_mul(&cp.m.map(D::from)) / mean_segment_number;
        let segment_density = state
            .partial_density
            .component_mul(&cp.m.map(D::from))
            .sum();

        let d_ij = DMatrix::from_fn(n, n, |i, j| (diameter[i] + diameter[j]) * 0.5);
        let d3_ij = d_ij.map(|d| d.powi(3));

        let mut zeta_x = D::zero();
        let mut zeta_x_bar = D::zero();
        for i in 0..n {
            zeta_x += xs[i].powi(2) * d3_ij[(i, i)];
            zeta_x_bar += xs[i].powi(2) * cp.sigma_ij[(i, i)].powi(3);
            for j in i + 1..n {
                zeta_x += xs[i] * xs[j] * d3_ij[(i, j)] * 2.0;
                zeta_x_bar += xs[i] * xs[j] * cp.sigma_ij[(i, j)].powi(3) * 2.0;
            }
        }
        zeta_x *= segment_density * FRAC_PI_6;
        zeta_x_bar *= segment_density * FRAC_PI_6;

        let frac_1mzeta3 = (-zeta_x + 1.0).powi(3).recip();
        let z = zeta_x;
        let z2 = z * z;
        let z3 = z2 * z;
        let k0 = -(-z + 1.0).ln()
            + z * (-z * 39.0 + z2 * 9.0 - z3 * 2.0 + 42.0) * frac_1mzeta3 / 6.0;
        let k1 = z * frac_1mzeta3 * 0.5 * (z3 + z * 6.0 - 12.0);
        let k2 = -z2 * 3.0 / 8.0 * frac_1mzeta3 * (-zeta_x + 1.0);
        let k3 = z * frac_1mzeta3 / 6.0 * (-z3 + z * 3.0 + 3.0);

        Self {
            diameter: diameter.clone(),
            segment_density,
            segment_molefracs: xs,
            mean_segment_number,
            zeta_x,
            zeta_x_bar,
            k0: [k0, k1, k2, k3],
        }
    }
}

/// Phi coefficients for a3 (Lafitte 2013 Table III, 0-indexed)
pub(super) const PHI: [[f64; 7]; 6] = [
    [7.5365557, -37.60463, 71.745953, -46.83552, -2.467982, -0.50272, 8.0956883],
    [-359.44, 1825.6, -3168.0, 1884.2, -0.82376, -3.1935, 3.709],
    [1550.9, -5070.1, 6534.6, -3288.7, -2.7171, 2.0883, 0.0],
    [-1.19932, 9.063632, -17.9482, 11.34027, 20.52142, -56.6377, 40.53683],
    [-1911.28, 21390.175, -51320.7, 37064.54, 1103.742, -3264.61, 2556.181],
    [9236.9, -129430.0, 357230.0, -315530.0, 1390.2, -4518.2, 4241.6],
];

/// C coefficients for zeta_eff (Lafitte 2013 Table II)
const C: [[f64; 4]; 4] = [
    [0.81096, 1.7888, -37.578, 92.284],
    [1.0205, -19.341, 151.26, -463.50],
    [-1.9057, 22.845, -228.14, 973.92],
    [1.0885, -6.1962, 106.98, -677.64],
];

/// ζ_eff — D-typed lambda (for temperature-dependent repulsive exponent lr)
#[inline]
pub(super) fn zeta_eff_d<D: DualNum<f64> + Copy>(zeta: D, lambda: D) -> D {
    let li = lambda.recip();
    let li2 = li * li;
    let li3 = li * li2;
    let c = [
        li * C[0][1] + li2 * C[0][2] + li3 * C[0][3] + C[0][0],
        li * C[1][1] + li2 * C[1][2] + li3 * C[1][3] + C[1][0],
        li * C[2][1] + li2 * C[2][2] + li3 * C[2][3] + C[2][0],
        li * C[3][1] + li2 * C[3][2] + li3 * C[3][3] + C[3][0],
    ];
    zeta * (zeta * (zeta * (zeta * c[3] + c[2]) + c[1]) + c[0])
}

/// ζ_eff — f64 lambda (fast path for the fixed attractive exponent la)
#[inline]
fn zeta_eff_f<D: DualNum<f64> + Copy>(zeta: D, lambda: f64) -> D {
    let li = 1.0 / lambda;
    let li2 = li * li;
    let li3 = li * li2;
    let c = [
        li * C[0][1] + li2 * C[0][2] + li3 * C[0][3] + C[0][0],
        li * C[1][1] + li2 * C[1][2] + li3 * C[1][3] + C[1][0],
        li * C[2][1] + li2 * C[2][2] + li3 * C[2][3] + C[2][0],
        li * C[3][1] + li2 * C[3][2] + li3 * C[3][3] + C[3][0],
    ];
    zeta * (zeta * (zeta * (zeta * c[3] + c[2]) + c[1]) + c[0])
}

#[inline]
fn a1s_d<D: DualNum<f64> + Copy>(zeta_x: D, lambda: D) -> D {
    let ze = zeta_eff_d(zeta_x, lambda);
    -(-ze * 0.5 + 1.0) / ((-ze + 1.0).powi(3) * (lambda - 3.0))
}

#[inline]
fn a1s_f<D: DualNum<f64> + Copy>(zeta_x: D, lambda: f64) -> D {
    let ze = zeta_eff_f(zeta_x, lambda);
    -(-ze * 0.5 + 1.0) / ((-ze + 1.0).powi(3) * (lambda - 3.0))
}

/// b_ij correction — D-typed lambda
#[inline]
fn b_d<D: DualNum<f64> + Copy>(zeta_x: D, x0: D, lambda: D) -> D {
    let lm3 = lambda - 3.0;
    let lm4 = lambda - 4.0;
    let x0_3ml = x0.powd(-lm3); // x0^(3-λ) = x0^(-(λ-3))
    let x0_4ml = x0.powd(-lm4); // x0^(4-λ)
    let i = -(x0_3ml - 1.0) / lm3;
    let j = -(x0_4ml * lm3 - x0_3ml * lm4 - 1.0) / (lm3 * lm4);
    ((-zeta_x * 0.5 + 1.0) * i - zeta_x * (zeta_x + 1.0) * 4.5 * j) * (-zeta_x + 1.0).powi(-3)
}

/// b_ij correction — f64 lambda (fast path for la)
#[inline]
fn b_f<D: DualNum<f64> + Copy>(zeta_x: D, x0: D, lambda: f64) -> D {
    let x0_3ml = x0.powf(3.0 - lambda);
    let i = -(x0_3ml - 1.0) / (lambda - 3.0);
    let j = -(x0.powf(4.0 - lambda) * (lambda - 3.0) - x0_3ml * (lambda - 4.0) - 1.0)
        / ((lambda - 3.0) * (lambda - 4.0));
    ((-zeta_x * 0.5 + 1.0) * i - zeta_x * (zeta_x + 1.0) * 4.5 * j) * (-zeta_x + 1.0).powi(-3)
}

/// x0^λ (a1s + b) — D-typed lambda (for lr and combined terms)
#[inline]
fn a1sb_d<D: DualNum<f64> + Copy>(zeta_x: D, x0: D, lambda: D) -> D {
    x0.powd(lambda) * (a1s_d(zeta_x, lambda) + b_d(zeta_x, x0, lambda))
}

/// x0^λ (a1s + b) — f64 lambda (avoids expensive powd for fixed la)
#[inline]
fn a1sb_f<D: DualNum<f64> + Copy>(zeta_x: D, x0: D, lambda: f64) -> D {
    x0.powf(lambda) * (a1s_f(zeta_x, lambda) + b_f(zeta_x, x0, lambda))
}

/// f_k(α) rational function — D-typed alpha (Lafitte Table IV, 0-indexed)
#[inline]
fn f_d<D: DualNum<f64> + Copy>(k: usize, alpha: D) -> D {
    let a2 = alpha * alpha;
    let a3 = a2 * alpha;
    let phi = PHI[k];
    (alpha * phi[1] + a2 * phi[2] + a3 * phi[3] + phi[0])
        / (alpha * phi[4] + a2 * phi[5] + a3 * phi[6] + 1.0)
}

/// First + second + third order perturbation dispersion (no chain).
pub fn helmholtz_energy_density_disp<D: DualNum<f64> + Copy>(
    cp: &CorrespondingParameters<D>,
    props: &Properties<D>,
    state: &StateHD<D>,
) -> D {
    let n = cp.ncomponents;
    let xs = &props.segment_molefracs;
    let t_inv = state.temperature.recip();
    let zeta_x = props.zeta_x;
    let k_hs = (zeta_x - 1.0).powi(4)
        / ((zeta_x + zeta_x.powi(2) - zeta_x.powi(3)) * 4.0 + zeta_x.powi(4) + 1.0);
    let zx = props.zeta_x_bar;
    let zx5 = zx.powi(5);
    let zx8 = zx.powi(8);

    let mut a1 = D::zero();
    let mut a2 = D::zero();
    let mut a3 = D::zero();

    for i in 0..n {
        let eps = cp.epsilon_k[i];
        let sig = cp.sigma[i];
        let la = cp.la[i];      // f64
        let lr = cp.lr[i];      // D
        let c = cp.c_ij[(i, i)];   // D
        let alpha = cp.alpha_ij[(i, i)];  // D
        let di = props.diameter[i];
        let x0 = di.recip() * sig;
        let pref = props.segment_density * di.powi(3) * eps * c * (2.0 * PI);

        let t_la = a1sb_f(zeta_x, x0, la);
        let t_lr = a1sb_d(zeta_x, x0, lr);
        let t_2la = a1sb_f(zeta_x, x0, 2.0 * la);
        let t_lalr = a1sb_d(zeta_x, x0, lr + la);
        let t_2lr = a1sb_d(zeta_x, x0, lr * 2.0);

        let a1_ii = pref * (t_la - t_lr);
        let a2_ii = pref * eps * c * k_hs * 0.5 * (t_2la - t_lalr * 2.0 + t_2lr);
        let a3_ii = -zx * f_d(3, alpha) * (zx * (zx * f_d(5, alpha) + f_d(4, alpha))).exp()
            * eps.powi(3);
        let xi_ii = zx * f_d(0, alpha) + zx5 * f_d(1, alpha) + zx8 * f_d(2, alpha);

        let xs2 = xs[i] * xs[i];
        a1 += a1_ii * xs2;
        a2 += a2_ii * xs2 * (xi_ii + 1.0);
        a3 += a3_ii * xs2;

        for j in i + 1..n {
            let eps = cp.epsilon_k_ij[(i, j)];
            let sig = cp.sigma_ij[(i, j)];
            let la = cp.la_ij[(i, j)];  // f64
            let lr = cp.lr_ij[(i, j)];  // D
            let c = cp.c_ij[(i, j)];
            let alpha = cp.alpha_ij[(i, j)];
            let dij = (di + props.diameter[j]) * 0.5;
            let x0 = dij.recip() * sig;
            let pref = props.segment_density * dij.powi(3) * eps * c * (2.0 * PI);

            let t_la = a1sb_f(zeta_x, x0, la);
            let t_lr = a1sb_d(zeta_x, x0, lr);
            let t_2la = a1sb_f(zeta_x, x0, 2.0 * la);
            let t_lalr = a1sb_d(zeta_x, x0, lr + la);
            let t_2lr = a1sb_d(zeta_x, x0, lr * 2.0);

            let a1_ij = pref * (t_la - t_lr);
            let a2_ij = pref * eps * c * k_hs * 0.5 * (t_2la - t_lalr * 2.0 + t_2lr);
            let a3_ij = -zx * f_d(3, alpha) * (zx * (zx * f_d(5, alpha) + f_d(4, alpha))).exp()
                * eps.powi(3);
            let xi_ij = zx * f_d(0, alpha) + zx5 * f_d(1, alpha) + zx8 * f_d(2, alpha);

            let xs_ij = xs[i] * xs[j];
            a1 += a1_ij * xs_ij * 2.0;
            a2 += a2_ij * xs_ij * (xi_ij + 1.0) * 2.0;
            a3 += a3_ij * xs_ij * 2.0;
        }
    }
    state.partial_density.sum()
        * props.mean_segment_number
        * (a1 * t_inv + a2 * t_inv.powi(2) + a3 * t_inv.powi(3))
}

/// Dispersion + chain Helmholtz energy density.
///
/// Uses `Dual<D>` to differentiate a1 and a2 w.r.t. ρ_s analytically for the chain term.
/// Key: effective parameters (eps, lr, c, sigma from cp) are wrapped in `Dual::from_re`
/// since they depend on T but not on ρ_s.
pub fn helmholtz_energy_density_disp_chain<D: DualNum<f64> + Copy>(
    cp: &CorrespondingParameters<D>,
    props: &Properties<D>,
    state: &StateHD<D>,
) -> D {
    let n = cp.ncomponents;
    let kv = &props.k0;
    let xs = &props.segment_molefracs;
    let t_inv = state.temperature.recip();

    // Differentiate w.r.t. segment density ρ_s using Dual<D>
    let rho_s_d = Dual::from_re(props.segment_density).derivative();
    let zeta_x = props.zeta_x;
    let zeta_x_d = if props.segment_density.is_zero() {
        rho_s_d * 0.0
    } else {
        Dual::from_re(zeta_x / props.segment_density) * rho_s_d
    };
    let k_hs_d = (zeta_x_d - 1.0).powi(4)
        / ((zeta_x_d + zeta_x_d.powi(2) - zeta_x_d.powi(3)) * 4.0
            + zeta_x_d.powi(4)
            + 1.0);
    let k_hs = k_hs_d.re; // real part for terms not needing d/dρ_s

    let zx = props.zeta_x_bar;
    let zx5 = zx.powi(5);
    let zx8 = zx.powi(8);

    let mut a1 = D::zero();
    let mut a2 = D::zero();
    let mut a3 = D::zero();
    let mut a_chain = D::zero();

    for i in 0..n {
        let m_i = cp.m[i];
        let eps = cp.epsilon_k[i];       // D (temperature-dependent effective)
        let sig = cp.sigma[i];           // D
        let la = cp.la[i];               // f64 (fixed)
        let lr = cp.lr[i];               // D (temperature-dependent effective)
        let c = cp.c_ij[(i, i)];         // D
        let alpha = cp.alpha_ij[(i, i)]; // D
        let di = props.diameter[i];      // D

        let lr_d = Dual::from_re(lr);
        let d3_d = Dual::from_re(di.powi(3));
        let x0_d = Dual::from_re(di.recip() * sig);

        // pref = 2π ρ_s d³ c ε  (all factors: Dual<D>)
        let pref_d = rho_s_d * d3_d * Dual::from_re(eps * c * (2.0 * PI));

        let t_la_d = a1sb_f(zeta_x_d, x0_d, la);
        let t_lr_d = a1sb_d(zeta_x_d, x0_d, lr_d);
        let t_2la_d = a1sb_f(zeta_x_d, x0_d, 2.0 * la);
        let t_lalr_d = a1sb_d(zeta_x_d, x0_d, lr_d + la);
        let t_2lr_d = a1sb_d(zeta_x_d, x0_d, lr_d * 2.0);

        let a1_ii = pref_d * (t_la_d - t_lr_d);
        let a2_ii =
            pref_d * Dual::from_re(eps * c) * k_hs_d * 0.5 * (t_2la_d - t_lalr_d * 2.0 + t_2lr_d);

        let a3_ii = -zx * f_d(3, alpha) * (zx * (zx * f_d(5, alpha) + f_d(4, alpha))).exp()
            * eps.powi(3);
        let xi_ii = zx * f_d(0, alpha) + zx5 * f_d(1, alpha) + zx8 * f_d(2, alpha);

        let xs2 = xs[i] * xs[i];
        a1 += a1_ii.re * xs2;
        a2 += a2_ii.re * xs2 * (xi_ii + 1.0);
        a3 += a3_ii * xs2;

        // Chain contribution: da1/dρ_s and da2/dρ_s from dual parts
        let x0 = x0_d.re; // D
        let pref = d3_d.re * eps * c * (2.0 * PI); // D: d³ * ε * c (no ρ_s factor)
        let g_hs = (kv[0] + kv[1] * x0 + kv[2] * x0.powi(2) + kv[3] * x0.powi(3)).exp();
        // g1 = d(a1_ii/pref)/dρ_s * 3  — uses eps part of a1_ii
        let g1 = a1_ii.eps * 3.0 / pref - (t_la_d.re * la - t_lr_d.re * lr) * c;
        let g2_mca = a2_ii.eps * 3.0 / pref / eps
            - (t_2lr_d.re * lr - t_lalr_d.re * (lr + la) + t_2la_d.re * la) * k_hs * c.powi(2);
        let beta_eps = t_inv * eps;
        let gamma = zx
            * beta_eps.exp_m1()
            * 10.0
            * ((-(alpha - 0.57) * 10.0).tanh() + 1.0)
            * (-zx * 6.7 - zx.powi(2) * 8.0).exp();
        let g2 = g2_mca * (gamma + 1.0);
        let ln_g = g_hs.ln() + (beta_eps * g1 + beta_eps.powi(2) * g2) / g_hs;
        a_chain += -state.molefracs[i] * (m_i - 1.0) * ln_g;

        for j in i + 1..n {
            let eps = cp.epsilon_k_ij[(i, j)];
            let sig = cp.sigma_ij[(i, j)];
            let la = cp.la_ij[(i, j)]; // f64
            let lr = cp.lr_ij[(i, j)]; // D
            let c = cp.c_ij[(i, j)];
            let alpha = cp.alpha_ij[(i, j)];
            let dij = (di + props.diameter[j]) * 0.5;
            let x0 = dij.recip() * sig;
            let pref = props.segment_density * dij.powi(3) * eps * c * (2.0 * PI);

            let t_la = a1sb_f(zeta_x, x0, la);
            let t_lr = a1sb_d(zeta_x, x0, lr);
            let t_2la = a1sb_f(zeta_x, x0, 2.0 * la);
            let t_lalr = a1sb_d(zeta_x, x0, lr + la);
            let t_2lr = a1sb_d(zeta_x, x0, lr * 2.0);

            let a1_ij = pref * (t_la - t_lr);
            let a2_ij = pref * eps * c * k_hs * 0.5 * (t_2la - t_lalr * 2.0 + t_2lr);
            let a3_ij = -zx * f_d(3, alpha) * (zx * (zx * f_d(5, alpha) + f_d(4, alpha))).exp()
                * eps.powi(3);
            let xi_ij = zx * f_d(0, alpha) + zx5 * f_d(1, alpha) + zx8 * f_d(2, alpha);

            let xs_ij = xs[i] * xs[j];
            a1 += a1_ij * xs_ij * 2.0;
            a2 += a2_ij * xs_ij * (xi_ij + 1.0) * 2.0;
            a3 += a3_ij * xs_ij * 2.0;
        }
    }
    state.partial_density.sum()
        * (props.mean_segment_number
            * (a1 * t_inv + a2 * t_inv.powi(2) + a3 * t_inv.powi(3))
            + a_chain)
}
