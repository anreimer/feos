import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fitting $uv$-CS-theory to helium with the new correction Ansatz

    Marimo port of `parameter_adjustment.ipynb` / `Helium_Parameters.ipynb`: the
    base Mie potential (`sigma`, `epsilon_k`, `rep`, `att`) stays fixed at the
    Thijs-optimized values, and only the Feynman-Hibbs-1 correction coefficients
    `c_sigma`, `c_epsilon_k`, `c_rep` (5 slots each, $\xi_0 .. \xi_4$) are fit.

    ## Effective coefficients

    $$
    \frac{\sigma^{\text{eff}}}{\sigma} =
    \frac{1 + \xi_0 C_0(\sigma)\,D^{*} + \xi_1 C_1(\sigma)\,D^{*2} + \xi_3 D^{*2}}
         {1 + \xi_2 C_2(\sigma)\,D^{*} + \xi_4 D^{*2}}
    \qquad
    \frac{\varepsilon^{\text{eff}}}{\varepsilon} =
    \frac{1 + \xi_0 C_0(\varepsilon)\,D^{*} + \xi_1 C_1(\varepsilon)\,D^{*2} + \xi_3 D^{*2}}
         {1 + \xi_2 C_2(\varepsilon)\,D^{*} + \xi_4 D^{*2}}
    $$

    $$
    \frac{\nu^{\text{eff}}}{\nu} =
    \frac{1 + \xi_0 C_0(\nu)\,D^{*} + \xi_1 C_1(\nu)\,D^{*2} + \xi_2 C_2(\nu)\,D^{*3}}
         {1 + \xi_3 C_3(\nu)\,D^{*} + \xi_4 C_4(\nu)\,D^{*2}}
    $$

    with $D^* = \hbar^2/(12 k_B T m \sigma^2)$.

    Earlier versions of this notebook fixed $\xi_0=\xi_2=1$ per property (9
    free parameters total), following `parameter_adjustment.ipynb`'s
    `eos_from_parameters` layout. That turned out to be *less* expressive
    than Thijs's own baseline in the directions that matter: Thijs fits
    $\xi_0,\xi_1,\xi_2$ **freely** for every property (all three carry a
    genuine $C_i(m)$ prefactor, so they're well-scaled and directly
    comparable to Thijs's values like `c_sigma = [4.15, 17.13, 4.65, ...]`),
    while pinning them at 1 left only $\xi_1$ as a real degree of freedom per
    property — $\xi_3,\xi_4$ (the bare $D^{*2}$ terms with no $C_i(m)$
    prefactor) need values of order 10-100 just to matter and are regularized
    toward 0 below anyway (see further down). So fixing $\xi_0=\xi_2=1$ threw
    away exactly the freedom Thijs relies on.

    All **15 slots are now free** ($\xi_0..\xi_4$ for each of `c_sigma`,
    `c_epsilon_k`, `c_rep`) — a strict superset of Thijs's parameterization,
    so the fit can only match or beat it, never do worse from capacity alone.

    `parameter_adjustment.ipynb`'s `cost` only matched vapor pressure and its
    `least_squares` call returned the initial guess unchanged (`popt.x == p0`) —
    a sign the objective wasn't actually constraining the fit. This notebook
    follows `fit_saft_vrcs_marimo.py`'s more robust recipe instead: liquid
    *and vapor* density (liquid weighted 2x) + vapor pressure along the
    sub-critical VLE curve, density along the isotherms, and the critical
    point, each residual group scaled by $1/\sqrt{n_\mathrm{group}}$ so no
    single group's size dominates or vanishes in the sum of squares. (Vapor
    density used to be purely emergent — never in the objective — which is
    likely why it was the weakest metric against NIST; it's now fit
    directly.)
    """)
    return


@app.cell
def _():
    import feos
    from feos import EquationOfState, Parameters, PhaseDiagram, PhaseEquilibrium, State
    import si_units as si
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.optimize import least_squares
    import seaborn as sns

    sns.set_context("notebook")
    sns.set_palette("Dark2")
    sns.set_style("ticks")
    colors = sns.palettes.color_palette("Dark2", 8)
    return (
        EquationOfState,
        Parameters,
        PhaseDiagram,
        PhaseEquilibrium,
        State,
        colors,
        feos,
        least_squares,
        np,
        pd,
        plt,
        si,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Helium configuration

    - **base**: fixed Mie potential (`sigma`, `epsilon_k`, `rep`, `att=6`),
      the Thijs-optimized values also used as the `uvcs` baseline's base.
    - **vrqmie**: SAFT-VRQ-Mie baseline (Aasen et al., FH order 1).
    - **uvcs**: Thijs's $uv$-CS-theory fit (3-slot `c_sigma`/`c_epsilon_k`,
      padded to 5 with zeros for the effective-parameter formulas below).
    - **t_min**: low-temperature VLE cutoff below which the phase-equilibrium
      solver becomes unreliable.
    - **pc_extrap_n**: number of highest-temperature rows of the *unfiltered*
      VLE table used to linearly extrapolate `p_c` to `tc` (NIST's table
      stops just short of `Tc`).
    """)
    return


@app.cell
def _():
    gas = dict(
        name="helium",
        molarweight=4.002602,
        tc=5.1953,
        base=dict(sigma=2.6953, epsilon_k=5.2850, rep=8.2761, att=6.0),
        vrqmie=dict(m=1.0, sigma=2.7443, epsilon_k=5.4195, lr=9, la=6, fh=1),
        uvcs=dict(
            sigma=2.6953,
            epsilon_k=5.2850,
            rep=8.2761,
            att=6.0,
            c_sigma=[4.1529, 17.130, 4.6523, 0.0, 0.0],
            c_epsilon_k=[0.25290, 0.0, 0.42998, 0.0, 0.0],
            c_rep=[0.88165, 1.0523, 2.7524, 0.97560, 0.82524],
        ),
        vle_path="data/helium_data/nist_vle.txt",
        isotherms_path="data/helium_data/nist_isotherms.txt",
        t_min=2.7,
        pc_extrap_n=15,
        # 15 free slots: [xi0..xi4] for c_sigma, then c_epsilon_k, then c_rep.
        # Starts at the literature FH1 neutral default for each property
        # ([1,1,1,0,0] for sigma/epsilon, [1,1,1,1,1] for rep, since every
        # rep slot carries a C_i(m) prefactor) rather than seeded with
        # Thijs's values, so convergence to a good fit demonstrates the
        # objective/parameterization works on its own.
        par0=[1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    return (gas,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reference data (NIST) and baseline models
    """)
    return


@app.cell
def _(gas, np, pd):
    vle_full = pd.read_csv(gas["vle_path"], sep="\t")
    vle = vle_full[
        (vle_full["Temperature (K)"] > gas["t_min"]) & (vle_full["Temperature (K)"] < 0.9 * gas["tc"])
    ]
    isotherms = pd.read_csv(gas["isotherms_path"], sep="\t")

    # NIST's table stops just short of Tc, so the critical pressure isn't
    # tabulated directly; extrapolate psat(T) linearly from the highest-T
    # rows of the *unfiltered* table (near Tc, psat(T) is close to linear).
    _diam_rows = vle_full.tail(gas["pc_extrap_n"])
    _p_fit = np.polyfit(_diam_rows["Temperature (K)"], _diam_rows["Pressure (MPa)"], 1)
    pc_nist = np.polyval(_p_fit, gas["tc"])
    return isotherms, pc_nist, vle


@app.cell
def _(EquationOfState, Parameters, PhaseDiagram, feos, gas, si):
    vrq_pr = feos.PureRecord(
        identifier=feos.Identifier(gas["name"]), molarweight=gas["molarweight"], **gas["vrqmie"]
    )
    vrq_model = EquationOfState.saftvrqmie(Parameters.new_pure(vrq_pr))
    df_vrq = PhaseDiagram.pure(vrq_model, min_temperature=2 * si.KELVIN, npoints=250).to_dict(
        feos.Contributions.Residual
    )

    uvcs_pr = feos.PureRecord(
        identifier=feos.Identifier(gas["name"]), molarweight=gas["molarweight"], **gas["uvcs"]
    )
    cs_model = EquationOfState.uvcstheory(Parameters.new_pure(uvcs_pr))
    df_cs = PhaseDiagram.pure(cs_model, min_temperature=2 * si.KELVIN, npoints=250).to_dict(
        feos.Contributions.Residual
    )
    return cs_model, df_cs, df_vrq, vrq_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effective-parameter formulas

    Ported 1:1 from `crates/feos/src/uvcs/corresponding_states.rs`
    (`effective_sigma`/`effective_epsilon_k`/`effective_repulsive_exponent_ratio`
    — same polynomial constants, same rational-function form, same $D^*$
    prefactor), so both the fit's low-$D^*$ anchor below and the
    evaluation plot further down re-evaluate the existing Rust formulas
    rather than approximating them.
    """)
    return


@app.cell
def _():
    _C = [
        [-1.0577620680e02, 2.7130516920e01, -7.9885048560e-02],
        [-2.2875719760e03, 5.1995559760e02, 8.2111050000e-01],
        [1.7306319200e03, -1.7476174200e02, 3.2561549200e01],
        [-9.0993639000e01, 2.2823545260e01, -2.8168749280e-02],
        [-1.4764571200e03, 4.3447760860e02, -1.1267359060e01],
    ]
    _CS = [
        31.60144413, 2.09612861, 0.20116118,
        50.80286701, -4.28014515, 0.20063555,
        25.11333946, 1.35310377, 0.20250069,
    ]
    _CE = [
        10.60837315, 1.58987996, 0.19807347,
        17.92633472, -19.76493004, 3.91514157,
        7.17199935, 6.45489122, 0.18773935,
    ]

    def effective_rep_ratio(qd, m, c_scale=None):
        c_scale = c_scale if c_scale is not None else [1.0] * 5
        pref = [(_C[i][0] + m * (_C[i][1] + m * _C[i][2])) * c_scale[i] for i in range(5)]
        num = qd * (qd * (qd * pref[2] + pref[1]) + pref[0]) + 1.0
        den = qd * (qd * pref[4] + pref[3]) + 1.0
        return num / den

    def effective_sigma(qd, m, c_scale=None):
        c_scale = c_scale if c_scale is not None else [1.0, 1.0, 1.0, 0.0, 0.0]
        c0 = _CS[0] + m * (_CS[1] + m * _CS[2])
        c1 = _CS[3] + m * (_CS[4] + m * _CS[5])
        c2 = _CS[6] + m * (_CS[7] + m * _CS[8])
        num = qd * (qd * c1 * c_scale[1] + c0 * c_scale[0] + qd**2 * c_scale[3]) + 1.0
        den = qd * c2 * c_scale[2] + qd**2 * c_scale[4] + 1.0
        return num / den

    def effective_epsilon_k(qd, m, c_scale=None):
        c_scale = c_scale if c_scale is not None else [1.0, 1.0, 1.0, 0.0, 0.0]
        c0 = _CE[0] + m * (_CE[1] + m * _CE[2])
        c1 = _CE[3] + m * (_CE[4] + m * _CE[5])
        c2 = _CE[6] + m * (_CE[7] + m * _CE[8])
        num = qd * (qd * c1 * c_scale[1] + c0 * c_scale[0]) + qd**2 * c_scale[3] + 1.0
        den = qd * c2 * c_scale[2] + qd**2 * c_scale[4] + 1.0
        return num / den

    # D* = hbar^2 / (12 kB T m sigma^2), same prefactor as
    # corresponding_states.rs's D_QM_PREFACTOR (SI constants, sigma in
    # Angstrom via the 1e20 factor).
    _KB = 1.380649e-23
    _PLANCK = 6.62607015e-34
    _NAV = 6.02214076e23
    _TAU = 6.283185307179586
    _D_QM_PREFACTOR = _PLANCK**2 / (_TAU**2) / 12.0 * 1e20 / _KB

    def d_star_from_temperature(temp_k, gas_config):
        mass = gas_config["molarweight"] / 1000.0 / _NAV
        return (1.0 / temp_k) / mass * _D_QM_PREFACTOR / gas_config["base"]["sigma"] ** 2

    return d_star_from_temperature, effective_epsilon_k, effective_rep_ratio, effective_sigma


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## $uv$-CS fit with the new Ansatz

    `eos_from_parameters` builds the 5-slot `c_sigma`/`c_epsilon_k`/`c_rep`
    vectors directly from the 15 free coefficients (`pars[0:5]`,
    `pars[5:10]`, `pars[10:15]`) — all slots free, matching Thijs's own
    parameterization plus the two new bare-$D^{*2}$ slots per property.
    `cost` returns the concatenated relative-residual vector that
    `least_squares` minimizes.
    """)
    return


@app.cell
def _(EquationOfState, Parameters, feos, gas):
    def eos_from_parameters(pars):
        c_sigma = list(pars[0:5])
        c_epsilon_k = list(pars[5:10])
        c_rep = list(pars[10:15])

        params = dict(gas["base"], c_sigma=c_sigma, c_epsilon_k=c_epsilon_k, c_rep=c_rep)
        pr = feos.PureRecord(
            identifier=feos.Identifier(gas["name"]), molarweight=gas["molarweight"], **params
        )
        return EquationOfState.uvcstheory(Parameters.new_pure(pr))

    return (eos_from_parameters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each deviation group (liquid density, vapor pressure, isotherm density,
    critical point) is scaled by $1/\sqrt{n_\mathrm{group}}$, not $1/n_\mathrm{group}$
    — dividing by `n` before squaring would let a group's contribution to the
    sum-of-squares shrink as it grows, hiding a real mismatch behind group size
    instead of penalizing it. The critical-point term matters beyond its 2
    residuals: without it, nothing in the sub-critical VLE objective stops the
    optimizer from overshooting `Tc`/`p_c` while still threading the fitted
    points.

    **Regularization on the bare-$D^{*2}$ slots only.** `c_sigma[3]`,
    `c_sigma[4]`, `c_epsilon_k[3]`, `c_epsilon_k[4]`, `c_rep[3]`, `c_rep[4]`
    multiply a *bare* $D^{*2}$ term with no $C_i(m)$ polynomial prefactor
    (see the formulas above), unlike `[0]`, `[1]`, `[2]` of every property.
    Since $D^*$ tops out around 0.05 in the fitted range, $D^{*2}$ peaks near
    0.0025, so those six coefficients must reach values of order 10-100 just
    to compete with the well-scaled slots — and a hard `bounds=(...)` box
    just gets hit as a wall, wherever it's drawn (tried both `(-8, 8)` and
    `(-100, 100)`; both pinned). `[0]`, `[1]`, `[2]` get **no** regularization
    now — they're well-scaled, genuine free parameters exactly like Thijs's
    (whose fitted values, e.g. `c_sigma = [4.15, 17.13, 4.65, ...]`, are
    nowhere near a "neutral" 1 — regularizing them toward 1 would only fight
    the fit).

    `cost` appends a small L2 term pulling *only* the six bare-$D^{*2}$ slots
    back toward 0, scaled like the other groups. This lets the fit use them
    *if* they earn their keep against real data, but stops the optimizer
    from trading a large parameter excursion for a marginal cost improvement
    in what's otherwise a nearly flat direction — an interior optimum instead
    of a box edge.

    **Low-$D^*$ anchor for all three properties.** The fitted VLE + isotherm
    data only covers temperatures up to $T_c$, i.e. $D^* \gtrsim D^*(T_c)$ —
    below that, nothing constrains `c_sigma`/`c_epsilon_k`/`c_rep`'s shape,
    and (now that `[0]`/`[2]` are free too) all three are free to drift
    arbitrarily far from the literature FH1 curve there, not just `c_rep` as
    before. `cost` adds a residual per property comparing
    `effective_sigma`/`effective_epsilon_k`/`effective_rep_ratio` under the
    fitted coefficients against each property's own unfitted curve
    (`[1,1,1,0,0]` for $\sigma$/$\varepsilon$, `[1]*5` for $\nu$) at a handful
    of $D^*$ points between 0 and $D^*(T_c)$, scaled by $1/\sqrt{n}$ like the
    other groups **and** an explicit `anchor_weight = 0.3` — chosen (for
    `c_rep` alone, previously) to noticeably narrow the low-$D^*$ gap to FH1
    without dominating the fit of actual VLE/isotherm data (full 1x weight
    dragged every data-fit group down with it: cost 0.00077 vs. 0.00021
    without any anchor). Raise it for a tighter FH1 match at the cost of fit
    quality, or lower it to relax back toward the unconstrained fit.
    """)
    return


@app.cell
def _(
    PhaseEquilibrium,
    State,
    d_star_from_temperature,
    effective_epsilon_k,
    effective_rep_ratio,
    effective_sigma,
    eos_from_parameters,
    feos,
    gas,
    isotherms,
    np,
    pc_nist,
    si,
    vle,
):
    def cost(pars):
        eos = eos_from_parameters(pars)

        error_rhol = []
        error_rhov = []
        error_ps = []
        n_vle = len(vle)
        for _, row in vle.iterrows():
            temp = row["Temperature (K)"]
            try:
                feos_vle = PhaseEquilibrium.pure(eos, temperature_or_pressure=temp * si.KELVIN)
                rho_l = row["Density (l, kg/m3)"]
                rho_l_feos = feos_vle.liquid.mass_density() / si.KILOGRAM * si.METER**3
                error_rhol.append(2 * (rho_l_feos - rho_l) / rho_l / np.sqrt(n_vle))

                rho_v = row["Density (v, kg/m3)"]
                rho_v_feos = feos_vle.vapor.mass_density() / si.KILOGRAM * si.METER**3
                error_rhov.append((rho_v_feos - rho_v) / rho_v / np.sqrt(n_vle))

                pexp = row["Pressure (MPa)"]
                pfeos = feos_vle.vapor.pressure() / (si.MEGA * si.PASCAL)
                error_ps.append((pfeos - pexp) / pexp / np.sqrt(n_vle))
            except Exception:
                error_rhol.append(3)
                error_rhov.append(3)
                error_ps.append(3)

        err_rho = []
        n_iso = len(isotherms)
        for _, row in isotherms.iterrows():
            p_exp = row["Pressure (MPa)"]
            t_exp = row["Temperature (K)"]
            rho_exp = row["Density (l, kg/m3)"]
            try:
                s = feos.State(
                    eos,
                    temperature=t_exp * si.KELVIN,
                    pressure=p_exp * si.MEGA * si.PASCAL,
                )
                rho_feos = s.mass_density() / (si.KILOGRAM / si.METER**3)
                err_rho.append((rho_feos - rho_exp) / rho_exp / np.sqrt(n_iso))
            except Exception:
                err_rho.append(3)

        n_crit = 2
        try:
            crit_state = State.critical_point_pure(eos)[0]
            tc_feos = crit_state.temperature / si.KELVIN
            pc_feos = crit_state.pressure() * 1e-6 / si.PASCAL
            err_crit = [
                (tc_feos - gas["tc"]) / gas["tc"] / np.sqrt(n_crit),
                (pc_feos - pc_nist) / pc_nist / np.sqrt(n_crit),
            ]
        except Exception:
            err_crit = [3 / np.sqrt(n_crit)] * n_crit

        # Pull only the six bare-D*^2 slots (index 3, 4 of each property)
        # back toward 0; [0]/[1]/[2] get no regularization now — they're
        # meant to be free, well-scaled parameters like Thijs's (see
        # markdown above).
        reg_idx = [3, 4, 8, 9, 13, 14]
        reg_lambda = 0.01
        err_reg = reg_lambda * np.array([pars[i] for i in reg_idx]) / np.sqrt(len(reg_idx))

        # Anchor all three properties' effective-parameter ratios to the
        # literature FH1 curve below D*(Tc), the unconstrained
        # high-temperature extrapolation region with no VLE or isotherm data
        # behind it (see markdown above).
        c_sigma = pars[0:5]
        c_epsilon_k = pars[5:10]
        c_rep = pars[10:15]
        d_star_tc = d_star_from_temperature(gas["tc"], gas)
        d_star_anchor = np.linspace(0.0, d_star_tc, 6)
        rep_base = gas["base"]["rep"]
        anchor_weight = 0.3
        n_anchor = len(d_star_anchor)

        sigma_fitted = effective_sigma(d_star_anchor, rep_base, c_sigma)
        sigma_unfitted = effective_sigma(d_star_anchor, rep_base, [1.0, 1.0, 1.0, 0.0, 0.0])
        err_sigma_anchor = anchor_weight * (sigma_fitted - sigma_unfitted) / np.sqrt(n_anchor)

        epsilon_fitted = effective_epsilon_k(d_star_anchor, rep_base, c_epsilon_k)
        epsilon_unfitted = effective_epsilon_k(d_star_anchor, rep_base, [1.0, 1.0, 1.0, 0.0, 0.0])
        err_epsilon_anchor = anchor_weight * (epsilon_fitted - epsilon_unfitted) / np.sqrt(n_anchor)

        rep_fitted = effective_rep_ratio(d_star_anchor, rep_base, c_rep)
        rep_unfitted = effective_rep_ratio(d_star_anchor, rep_base, [1.0] * 5)
        err_rep_anchor = anchor_weight * (rep_fitted - rep_unfitted) / np.sqrt(n_anchor)

        return np.concatenate([
            error_rhol, error_rhov, error_ps, err_rho, err_crit, err_reg,
            err_sigma_anchor, err_epsilon_anchor, err_rep_anchor,
        ])

    return (cost,)


@app.cell
def _(cost, gas, least_squares, mo):
    with mo.status.spinner(title="Fitting uv-CS c-parameters for helium..."):
        # Loose safety-net bounds only, to keep the numerical Jacobian well
        # behaved; the L2 regularization term in `cost` does the real work of
        # keeping the fit away from degenerate blow-ups (see markdown above).
        # max_nfev raised from 300 (9 params) since 15 free params need more
        # function evaluations per Jacobian step to converge.
        popt = least_squares(cost, gas["par0"], bounds=(-50, 50), max_nfev=500)
    popt
    return (popt,)


@app.cell
def _(popt):
    popt.x
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fitted model vs. baselines
    """)
    return


@app.cell
def _(PhaseDiagram, eos_from_parameters, feos, popt, si):
    cs_model_new = eos_from_parameters(popt.x)
    df_cs_new = PhaseDiagram.pure(cs_model_new, min_temperature=2 * si.KELVIN, npoints=250).to_dict(
        feos.Contributions.Residual
    )
    return cs_model_new, df_cs_new


@app.cell
def _(
    colors,
    cs_model,
    cs_model_new,
    df_cs,
    df_cs_new,
    df_vrq,
    feos,
    gas,
    isotherms,
    np,
    plt,
    si,
    vle,
    vrq_model,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- vapor pressure vs 1/T ---
    axes[0].plot(
        1 / vle["Temperature (K)"],
        vle["Pressure (MPa)"] * (si.MEGA * si.PASCAL / si.BAR),
        "o", mec="k", mfc="None", label="NIST",
    )
    axes[0].plot(1 / np.array(df_vrq["temperature"]), np.array(df_vrq["pressure"]) * (si.PASCAL / si.BAR), "--", color=colors[1], label="SAFT-VRQ-Mie")
    axes[0].plot(1 / np.array(df_cs["temperature"]), np.array(df_cs["pressure"]) * (si.PASCAL / si.BAR), "-", color=colors[0], label="$uv$-CS (Thijs)")
    axes[0].plot(1 / np.array(df_cs_new["temperature"]), np.array(df_cs_new["pressure"]) * (si.PASCAL / si.BAR), "-", color=colors[2], label="$uv$-CS (new Ansatz)")
    axes[0].set_xlabel("$1/T$ / K$^{-1}$")
    axes[0].set_ylabel("$p$ / bar")
    axes[0].set_yscale("log")
    axes[0].set_xlim(0.2, 0.45)
    axes[0].legend(fontsize=8, frameon=False)

    # --- VLE envelope, density vs temperature ---
    axes[1].plot(vle["Density (l, kg/m3)"], vle["Temperature (K)"], "o", mec="k", mfc="None", ms=6)
    axes[1].plot(vle["Density (v, kg/m3)"], vle["Temperature (K)"], "o", mec="k", mfc="None", ms=6)
    axes[1].plot(df_vrq["mass density liquid"], df_vrq["temperature"], "--", color=colors[1], label="SAFT-VRQ-Mie")
    axes[1].plot(df_vrq["mass density vapor"], df_vrq["temperature"], "--", color=colors[1])
    axes[1].plot(df_cs["mass density liquid"], df_cs["temperature"], "-", color=colors[0], label="$uv$-CS (Thijs)")
    axes[1].plot(df_cs["mass density vapor"], df_cs["temperature"], "-", color=colors[0])
    axes[1].plot(df_cs_new["mass density liquid"], df_cs_new["temperature"], "-", color=colors[2], label="$uv$-CS (new Ansatz)")
    axes[1].plot(df_cs_new["mass density vapor"], df_cs_new["temperature"], "-", color=colors[2])
    axes[1].set_xlabel(r"$\rho$ / (kg/m³)")
    axes[1].set_ylabel("$T$ / K")
    axes[1].set_ylim(2.5, 5.5)
    axes[1].legend(loc="best", frameon=False)

    # --- isotherms, pressure vs density ---
    for _temp, _group in isotherms.groupby("Temperature (K)"):
        axes[2].plot(
            _group["Density (l, kg/m3)"],
            np.array(_group["Pressure (MPa)"]) * (si.MEGA * si.PASCAL) / si.BAR,
            "o", mec="k", mfc="None", ms=4, label=f"{_temp} K",
        )

    # Overlay each model's p(rho) at the isotherm temperatures: Mie EoS
    # residual pressure + ideal-gas term, since these are single-phase
    # states, not saturation points from PhaseDiagram.
    for _temp in sorted(isotherms["Temperature (K)"].unique()):
        _rho_min = isotherms[isotherms["Temperature (K)"] == _temp]["Density (l, kg/m3)"].min()
        _rho_max = isotherms[isotherms["Temperature (K)"] == _temp]["Density (l, kg/m3)"].max()
        _densities = np.linspace(_rho_min * 0.8, _rho_max * 1.2, 150)

        _valid_densities, _p_vrq, _p_cs, _p_cs_new = [], [], [], []
        for _rho in _densities:
            try:
                _moldens = (_rho / gas["molarweight"]) * si.MOL / si.METER**3 * 1000
                _pig = _moldens * si.RGAS * _temp * si.KELVIN

                _s_vrq = feos.State(vrq_model, temperature=_temp * si.KELVIN, density=_moldens)
                _s_cs = feos.State(cs_model, temperature=_temp * si.KELVIN, density=_moldens)
                _s_cs_new = feos.State(cs_model_new, temperature=_temp * si.KELVIN, density=_moldens)

                _p_vrq.append(_s_vrq.pressure(feos.Contributions.Residual) / si.BAR + _pig / si.BAR)
                _p_cs.append(_s_cs.pressure(feos.Contributions.Residual) / si.BAR + _pig / si.BAR)
                _p_cs_new.append(_s_cs_new.pressure(feos.Contributions.Residual) / si.BAR + _pig / si.BAR)
                _valid_densities.append(_rho)
            except Exception:
                continue

        if _valid_densities:
            axes[2].plot(_valid_densities, _p_vrq, "--", lw=1.2, color=colors[1])
            axes[2].plot(_valid_densities, _p_cs, "-", lw=1.2, color=colors[0])
            axes[2].plot(_valid_densities, _p_cs_new, "-", lw=1.2, color=colors[2])

    axes[2].set_xlabel(r"$\rho$ / (kg/m³)")
    axes[2].set_ylabel("$p$ / bar")
    axes[2].legend(fontsize=7, loc="best")
    axes[2].set_ylim(-5, 50)
    axes[2].set_xlim(-2, 180)

    plt.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fitted coefficients
    """)
    return


@app.cell
def _(mo, pd, popt):
    fit_summary = pd.DataFrame(
        [dict(coefficient=f"c_sigma[{i}]", value=popt.x[i]) for i in range(5)]
        + [dict(coefficient=f"c_epsilon_k[{i}]", value=popt.x[5 + i]) for i in range(5)]
        + [dict(coefficient=f"c_rep[{i}]", value=popt.x[10 + i]) for i in range(5)]
    )
    mo.vstack(
        [
            mo.ui.table(fit_summary, label="Fitted coefficients (all 15 slots free)"),
            mo.md(f"`success={popt.success}`, `cost={popt.cost:.6f}`, `nfev={popt.nfev}`, `status={popt.status}`"),
        ]
    )
    return


@app.cell
def _(State, cs_model_new, df_cs_new, gas, mo, np, pc_nist, pd, si, vle):
    def _aad(model_col, nist_col, unit_factor=1.0):
        order = np.argsort(df_cs_new["temperature"])
        t_model = np.array(df_cs_new["temperature"])[order]
        y_model = np.array(df_cs_new[model_col])[order] * unit_factor

        t_nist = vle["Temperature (K)"].to_numpy()
        y_nist = vle[nist_col].to_numpy()

        mask = (t_nist >= t_model.min()) & (t_nist <= t_model.max())
        y_interp = np.interp(t_nist[mask], t_model, y_model)
        return 100 * np.mean(np.abs((y_interp - y_nist[mask]) / y_nist[mask]))

    _crit = State.critical_point_pure(cs_model_new)[0]
    _tc_fit = _crit.temperature / si.KELVIN
    _pc_fit = _crit.pressure() * 1e-6 / si.PASCAL

    fit_quality = pd.DataFrame(
        [
            dict(quantity="T_c / K", target=gas["tc"], fit=_tc_fit, pct_dev=100 * abs(_tc_fit - gas["tc"]) / gas["tc"]),
            dict(quantity="p_c / MPa (extrapolated target)", target=pc_nist, fit=_pc_fit, pct_dev=100 * abs(_pc_fit - pc_nist) / pc_nist),
            dict(quantity="AAD p_sat (%)", target=0.0, fit=_aad("pressure", "Pressure (MPa)", unit_factor=1e-6), pct_dev=np.nan),
            dict(quantity="AAD rho_liq (%)", target=0.0, fit=_aad("mass density liquid", "Density (l, kg/m3)"), pct_dev=np.nan),
            dict(quantity="AAD rho_vap (%)", target=0.0, fit=_aad("mass density vapor", "Density (v, kg/m3)"), pct_dev=np.nan),
        ]
    )
    mo.ui.table(fit_quality, label="Fit quality vs. NIST (helium)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effective potential parameters vs. quantum parameter $D^*$

    $\sigma_\mathrm{eff}/\sigma$, $\varepsilon_\mathrm{eff}/\varepsilon$, and
    $\nu_\mathrm{eff}/\nu$ as functions of $D^*$, ported 1:1 from
    `crates/feos/src/uvcs/corresponding_states.rs`
    (`effective_sigma`/`effective_epsilon_k`/`effective_repulsive_exponent_ratio`
    — same polynomial constants, same rational-function form) so this only
    re-evaluates the existing Rust formulas in Python rather than
    approximating them. Compares three curves, all evaluated at the same
    fixed base `rep=8.2761`:

    - **unfitted (literature FH1)**: `c_scale=1` on every slot that
      multiplies a genuine $C_i(m)$ polynomial, `0` on the slots that only
      add a bare $D^{*2}$ term with no such polynomial — i.e. `feos`'s own
      default when `c_sigma`/`c_epsilon_k`/`c_rep` are left as `None`
      (`[1,1,1,0,0]` for $\sigma$/$\varepsilon$, `[1,1,1,1,1]` for $\nu$,
      since every $\nu$ slot has a $C_i(m)$ term).
    - **Thijs**: the baseline $uv$-CS fit's correction coefficients.
    - **new Ansatz**: this notebook's fitted coefficients.

    A vertical line marks $D^*(T_c)$: the fitted VLE/isotherm data only spans
    temperatures up to $T_c$, i.e. $D^* \gtrsim D^*(T_c) \approx 0.027$ for
    helium — everything to the *left* of that line is pure extrapolation with
    zero data constraining it, which is exactly why `cost` anchors all three
    properties there (see the fit section above).
    """)
    return


@app.cell
def _(
    d_star_from_temperature,
    effective_epsilon_k,
    effective_rep_ratio,
    effective_sigma,
    gas,
    np,
    popt,
):
    d_star = np.linspace(0.0, 0.05, 200)
    d_star_tc = d_star_from_temperature(gas["tc"], gas)
    _rep = gas["base"]["rep"]

    # "Unfitted": c_scale = 1 on every slot that multiplies a genuine C_i(m)
    # polynomial, 0 on the slots that only add a bare qd**2 term with no such
    # polynomial (slots 3, 4 of sigma/epsilon). This is the literature FH1
    # formula with no correction applied — feos's own default when c_sigma /
    # c_epsilon_k / c_rep are left as None. For c_rep, all 5 slots have a
    # C_i(m) polynomial (the `C` table), so its unfitted vector is all ones.
    _c_sigma_unfitted = [1, 1, 1, 0, 0]
    _c_epsilon_unfitted = [1, 1, 1, 0, 0]
    _c_rep_unfitted = [1, 1, 1, 1, 1]

    _c_sigma_thijs = gas["uvcs"]["c_sigma"]
    _c_epsilon_thijs = gas["uvcs"]["c_epsilon_k"]
    _c_rep_thijs = gas["uvcs"]["c_rep"]

    _c_sigma_new = list(popt.x[0:5])
    _c_epsilon_new = list(popt.x[5:10])
    _c_rep_new = list(popt.x[10:15])

    sigma_ratio_unfitted = effective_sigma(d_star, _rep, _c_sigma_unfitted)
    epsilon_ratio_unfitted = effective_epsilon_k(d_star, _rep, _c_epsilon_unfitted)
    rep_ratio_unfitted = effective_rep_ratio(d_star, _rep, _c_rep_unfitted)

    sigma_ratio_thijs = effective_sigma(d_star, _rep, _c_sigma_thijs)
    epsilon_ratio_thijs = effective_epsilon_k(d_star, _rep, _c_epsilon_thijs)
    rep_ratio_thijs = effective_rep_ratio(d_star, _rep, _c_rep_thijs)

    sigma_ratio_new = effective_sigma(d_star, _rep, _c_sigma_new)
    epsilon_ratio_new = effective_epsilon_k(d_star, _rep, _c_epsilon_new)
    rep_ratio_new = effective_rep_ratio(d_star, _rep, _c_rep_new)
    return (
        d_star,
        d_star_tc,
        epsilon_ratio_new,
        epsilon_ratio_thijs,
        epsilon_ratio_unfitted,
        rep_ratio_new,
        rep_ratio_thijs,
        rep_ratio_unfitted,
        sigma_ratio_new,
        sigma_ratio_thijs,
        sigma_ratio_unfitted,
    )


@app.cell
def _(
    d_star,
    d_star_tc,
    epsilon_ratio_new,
    epsilon_ratio_thijs,
    epsilon_ratio_unfitted,
    plt,
    rep_ratio_new,
    rep_ratio_thijs,
    rep_ratio_unfitted,
    sigma_ratio_new,
    sigma_ratio_thijs,
    sigma_ratio_unfitted,
):
    fig2, axes2 = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

    axes2[0].plot(d_star, rep_ratio_unfitted, ":", color="k", label="unfitted (literature FH1)")
    axes2[0].plot(d_star, rep_ratio_thijs, "-", color="tab:blue", label="Thijs")
    axes2[0].plot(d_star, rep_ratio_new, "--", color="tab:orange", label="new Ansatz")
    axes2[0].axvline(d_star_tc, color="gray", lw=0.8, ls="-.", label="$D^*(T_c)$")
    axes2[0].set_ylabel(r"$\nu_\mathrm{eff}/\nu$")
    axes2[0].legend(frameon=False, fontsize=8)

    axes2[1].plot(d_star, sigma_ratio_unfitted, ":", color="k")
    axes2[1].plot(d_star, sigma_ratio_thijs, "-", color="tab:blue")
    axes2[1].plot(d_star, sigma_ratio_new, "--", color="tab:orange")
    axes2[1].axvline(d_star_tc, color="gray", lw=0.8, ls="-.")
    axes2[1].set_ylabel(r"$\sigma_\mathrm{eff}/\sigma$")

    axes2[2].plot(d_star, epsilon_ratio_unfitted, ":", color="k")
    axes2[2].plot(d_star, epsilon_ratio_thijs, "-", color="tab:blue")
    axes2[2].plot(d_star, epsilon_ratio_new, "--", color="tab:orange")
    axes2[2].axvline(d_star_tc, color="gray", lw=0.8, ls="-.")
    axes2[2].set_ylabel(r"$\varepsilon_\mathrm{eff}/\varepsilon$")
    axes2[2].set_xlabel("$D^*$")

    plt.tight_layout()
    fig2
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
