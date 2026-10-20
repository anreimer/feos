import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fitting SAFT-VR-CS to helium with the fully-free Ansatz

    Port of `../uvcs/helium_adjustment.py`'s fitting scheme to SAFT-VR-CS.
    `crates/feos/src/saftvrcs/corresponding_states.rs`'s Feynman-Hibbs
    quantum correction is now structurally identical to $uv$-CS-theory's:
    `c_sigma`, `c_epsilon_k`, `c_lr` are each 5-slot ($\xi_0..\xi_4$), with
    the same rational-polynomial-in-$D^*$ form and even the same
    correspondence-state polynomial constants (`C_SIGMA`/`C_EPSILON`/`C_LR`
    match `uvcs`'s `CS`/`CE`/`C` tables exactly). So the same lesson applies
    directly: fixing $\xi_0=\xi_2=1$ per property (as an early version of the
    $uv$-CS notebook did) throws away real degrees of freedom, since
    $\xi_0,\xi_1,\xi_2$ all carry a genuine $C_i(m)$ prefactor and are
    well-scaled, while only $\xi_3,\xi_4$ (the bare $D^{*2}$ terms) are
    structurally weak.

    All **15 slots are free** here from the start (`c_sigma`/`c_epsilon_k`/
    `c_lr`, 5 each). The base Mie potential (`m`, `sigma`, `epsilon_k`, `lr`,
    `la`) stays fixed at `fit_saft_vrcs_marimo.py`'s refined `sina`-style base
    rather than the raw Aasen FH1 literature values.
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

    - **base**: fixed Mie potential (`m=1`, `sigma`, `epsilon_k`, `lr`,
      `la=6`) — the refined `sina`-style base from `fit_saft_vrcs_marimo.py`,
      not the raw Aasen Table II / VRQ-Mie values.
    - **vrqmie**: SAFT-VRQ-Mie baseline (Aasen et al., FH order 1).
    - **uvcs**: Thijs's $uv$-CS-theory fit, kept as a cross-theory reference
      line in the comparison plots.
    - **t_min**/**pc_extrap_n**: same NIST VLE cutoff and critical-pressure
      extrapolation window as the $uv$-CS notebook.
    """)
    return


@app.cell
def _():
    gas = dict(
        name="helium",
        molarweight=4.002602,
        tc=5.1953,
        base=dict(m=1.0, sigma=2.674056, epsilon_k=9.466831, lr=11.981772, la=6.0),
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
        # 15 free slots: [xi0..xi4] for c_sigma, then c_epsilon_k, then c_lr.
        # Starts at the literature FH1 neutral default per property
        # ([1,1,1,0,0] for sigma/epsilon, [1,1,1,1,1] for lr, since every
        # lr slot carries a C_i(m) prefactor).
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

    Ported 1:1 from `crates/feos/src/saftvrcs/corresponding_states.rs`
    (`eff_sigma`/`eff_epsilon_k`/`eff_lr_ratio` — same polynomial constants,
    same rational-function form as $uv$-CS-theory's equivalents), so both the
    fit's low-$D^*$ anchor below and the evaluation plot further down
    re-evaluate the existing Rust formulas rather than approximating them.
    """)
    return


@app.cell
def _():
    _C_LR = [
        [-1.0577620680e02, 2.7130516920e01, -7.9885048560e-02],
        [-2.2875719760e03, 5.1995559760e02, 8.2111050000e-01],
        [1.7306319200e03, -1.7476174200e02, 3.2561549200e01],
        [-9.0993639000e01, 2.2823545260e01, -2.8168749280e-02],
        [-1.4764571200e03, 4.3447760860e02, -1.1267359060e01],
    ]
    _C_SIGMA = [
        31.60144413, 2.09612861, 0.20116118,
        50.80286701, -4.28014515, 0.20063555,
        25.11333946, 1.35310377, 0.20250069,
    ]
    _C_EPSILON = [
        10.60837315, 1.58987996, 0.19807347,
        17.92633472, -19.76493004, 3.91514157,
        7.17199935, 6.45489122, 0.18773935,
    ]

    def effective_lr_ratio(qd, lr, c_scale=None):
        c_scale = c_scale if c_scale is not None else [1.0] * 5
        pref = [(_C_LR[i][0] + lr * (_C_LR[i][1] + lr * _C_LR[i][2])) * c_scale[i] for i in range(5)]
        num = qd * (qd * (qd * pref[2] + pref[1]) + pref[0]) + 1.0
        den = qd * (qd * pref[4] + pref[3]) + 1.0
        return num / den

    def effective_sigma(qd, lr, c_scale=None):
        c_scale = c_scale if c_scale is not None else [1.0, 1.0, 1.0, 0.0, 0.0]
        c0 = _C_SIGMA[0] + lr * (_C_SIGMA[1] + lr * _C_SIGMA[2])
        c1 = _C_SIGMA[3] + lr * (_C_SIGMA[4] + lr * _C_SIGMA[5])
        c2 = _C_SIGMA[6] + lr * (_C_SIGMA[7] + lr * _C_SIGMA[8])
        num = qd * (qd * c1 * c_scale[1] + c0 * c_scale[0] + qd**2 * c_scale[3]) + 1.0
        den = qd * c2 * c_scale[2] + qd**2 * c_scale[4] + 1.0
        return num / den

    def effective_epsilon_k(qd, lr, c_scale=None):
        c_scale = c_scale if c_scale is not None else [1.0, 1.0, 1.0, 0.0, 0.0]
        c0 = _C_EPSILON[0] + lr * (_C_EPSILON[1] + lr * _C_EPSILON[2])
        c1 = _C_EPSILON[3] + lr * (_C_EPSILON[4] + lr * _C_EPSILON[5])
        c2 = _C_EPSILON[6] + lr * (_C_EPSILON[7] + lr * _C_EPSILON[8])
        num = qd * (qd * c1 * c_scale[1] + c0 * c_scale[0]) + qd**2 * c_scale[3] + 1.0
        den = qd * c2 * c_scale[2] + qd**2 * c_scale[4] + 1.0
        return num / den

    # D* = hbar^2 / (12 kB T m sigma^2), same prefactor as
    # corresponding_states.rs's D_QM_PREFACTOR.
    _KB = 1.380649e-23
    _PLANCK = 6.62607015e-34
    _NAV = 6.02214076e23
    _TAU = 6.283185307179586
    _D_QM_PREFACTOR = _PLANCK**2 / (_TAU**2) / 12.0 * 1e20 / _KB

    def d_star_from_temperature(temp_k, gas_config):
        mass = gas_config["molarweight"] / 1000.0 / _NAV
        return (1.0 / temp_k) / mass * _D_QM_PREFACTOR / gas_config["base"]["sigma"] ** 2

    return d_star_from_temperature, effective_epsilon_k, effective_lr_ratio, effective_sigma


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SAFT-VR-CS fit with the fully-free Ansatz

    `eos_from_parameters` builds the 5-slot `c_sigma`/`c_epsilon_k`/`c_lr`
    vectors directly from the 15 free coefficients (`pars[0:5]`,
    `pars[5:10]`, `pars[10:15]`) — all slots free. `cost` returns the
    concatenated relative-residual vector that `least_squares` minimizes.
    """)
    return


@app.cell
def _(EquationOfState, Parameters, feos, gas):
    def eos_from_parameters(pars):
        c_sigma = list(pars[0:5])
        c_epsilon_k = list(pars[5:10])
        c_lr = list(pars[10:15])

        params = dict(gas["base"], c_sigma=c_sigma, c_epsilon_k=c_epsilon_k, c_lr=c_lr)
        pr = feos.PureRecord(
            identifier=feos.Identifier(gas["name"]), molarweight=gas["molarweight"], **params
        )
        return EquationOfState.saftvrcs_mie(Parameters.new_pure(pr))

    return (eos_from_parameters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each deviation group (liquid density, vapor density, vapor pressure,
    isotherm density, critical point) is scaled by $1/\sqrt{n_\mathrm{group}}$
    so no single group's size dominates or vanishes in the sum of squares.
    Liquid density is weighted 2x; vapor density is fit directly (not left
    emergent) since that was the weakest metric in the $uv$-CS notebook when
    it wasn't directly optimized.

    **Regularization on the bare-$D^{*2}$ slots only.** `c_sigma[3]`,
    `c_sigma[4]`, `c_epsilon_k[3]`, `c_epsilon_k[4]`, `c_lr[3]`, `c_lr[4]`
    multiply a bare $D^{*2}$ term with no $C_i(m)$ prefactor — a small L2
    term pulls just those six back toward 0. `[0]`/`[1]`/`[2]` of every
    property get no regularization; they're meant to be free, well-scaled
    parameters.

    **Low-$D^*$ anchor for all three properties.** Below $D^*(T_c)$ there's
    no VLE or isotherm data at all (real helium doesn't exist there), so
    `cost` adds a residual per property comparing the fitted effective-ratio
    against each property's own unfitted (literature FH1) curve at a handful
    of $D^*$ points between 0 and $D^*(T_c)$, weighted at `0.3` (matching the
    $uv$-CS notebook's tuned value) so it nudges toward FH1 without
    dominating the fit of actual VLE/isotherm data.
    """)
    return


@app.cell
def _(
    PhaseEquilibrium,
    State,
    d_star_from_temperature,
    effective_epsilon_k,
    effective_lr_ratio,
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

        reg_idx = [3, 4, 8, 9, 13, 14]
        reg_lambda = 0.01
        err_reg = reg_lambda * np.array([pars[i] for i in reg_idx]) / np.sqrt(len(reg_idx))

        c_sigma = pars[0:5]
        c_epsilon_k = pars[5:10]
        c_lr = pars[10:15]
        d_star_tc = d_star_from_temperature(gas["tc"], gas)
        d_star_anchor = np.linspace(0.0, d_star_tc, 6)
        lr_base = gas["base"]["lr"]
        anchor_weight = 0.3
        n_anchor = len(d_star_anchor)

        sigma_fitted = effective_sigma(d_star_anchor, lr_base, c_sigma)
        sigma_unfitted = effective_sigma(d_star_anchor, lr_base, [1.0, 1.0, 1.0, 0.0, 0.0])
        err_sigma_anchor = anchor_weight * (sigma_fitted - sigma_unfitted) / np.sqrt(n_anchor)

        epsilon_fitted = effective_epsilon_k(d_star_anchor, lr_base, c_epsilon_k)
        epsilon_unfitted = effective_epsilon_k(d_star_anchor, lr_base, [1.0, 1.0, 1.0, 0.0, 0.0])
        err_epsilon_anchor = anchor_weight * (epsilon_fitted - epsilon_unfitted) / np.sqrt(n_anchor)

        lr_fitted = effective_lr_ratio(d_star_anchor, lr_base, c_lr)
        lr_unfitted = effective_lr_ratio(d_star_anchor, lr_base, [1.0] * 5)
        err_lr_anchor = anchor_weight * (lr_fitted - lr_unfitted) / np.sqrt(n_anchor)

        return np.concatenate([
            error_rhol, error_rhov, error_ps, err_rho, err_crit, err_reg,
            err_sigma_anchor, err_epsilon_anchor, err_lr_anchor,
        ])

    return (cost,)


@app.cell
def _(cost, gas, least_squares, mo):
    with mo.status.spinner(title="Fitting SAFT-VR-CS c-parameters for helium..."):
        popt = least_squares(cost, gas["par0"], bounds=(-50, 50), max_nfev=500)
    popt
    return (popt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fitted model vs. baselines
    """)
    return


@app.cell
def _(PhaseDiagram, eos_from_parameters, feos, popt, si):
    vrcs_model = eos_from_parameters(popt.x)
    df_vrcs = PhaseDiagram.pure(vrcs_model, min_temperature=2 * si.KELVIN, npoints=250).to_dict(
        feos.Contributions.Residual
    )
    return df_vrcs, vrcs_model


@app.cell
def _(
    colors,
    cs_model,
    df_cs,
    df_vrcs,
    df_vrq,
    feos,
    gas,
    isotherms,
    np,
    plt,
    si,
    vle,
    vrcs_model,
    vrq_model,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].plot(
        1 / vle["Temperature (K)"],
        vle["Pressure (MPa)"] * (si.MEGA * si.PASCAL / si.BAR),
        "o", mec="k", mfc="None", label="NIST",
    )
    axes[0].plot(1 / np.array(df_vrq["temperature"]), np.array(df_vrq["pressure"]) * (si.PASCAL / si.BAR), "--", color=colors[1], label="SAFT-VRQ-Mie")
    axes[0].plot(1 / np.array(df_cs["temperature"]), np.array(df_cs["pressure"]) * (si.PASCAL / si.BAR), "-", color=colors[0], label="$uv$-CS (Thijs)")
    axes[0].plot(1 / np.array(df_vrcs["temperature"]), np.array(df_vrcs["pressure"]) * (si.PASCAL / si.BAR), "-", color=colors[2], label="SAFT-VR-CS (fit)")
    axes[0].set_xlabel("$1/T$ / K$^{-1}$")
    axes[0].set_ylabel("$p$ / bar")
    axes[0].set_yscale("log")
    axes[0].set_xlim(0.2, 0.45)
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].plot(vle["Density (l, kg/m3)"], vle["Temperature (K)"], "o", mec="k", mfc="None", ms=6)
    axes[1].plot(vle["Density (v, kg/m3)"], vle["Temperature (K)"], "o", mec="k", mfc="None", ms=6)
    axes[1].plot(df_vrq["mass density liquid"], df_vrq["temperature"], "--", color=colors[1], label="SAFT-VRQ-Mie")
    axes[1].plot(df_vrq["mass density vapor"], df_vrq["temperature"], "--", color=colors[1])
    axes[1].plot(df_cs["mass density liquid"], df_cs["temperature"], "-", color=colors[0], label="$uv$-CS (Thijs)")
    axes[1].plot(df_cs["mass density vapor"], df_cs["temperature"], "-", color=colors[0])
    axes[1].plot(df_vrcs["mass density liquid"], df_vrcs["temperature"], "-", color=colors[2], label="SAFT-VR-CS (fit)")
    axes[1].plot(df_vrcs["mass density vapor"], df_vrcs["temperature"], "-", color=colors[2])
    axes[1].set_xlabel(r"$\rho$ / (kg/m³)")
    axes[1].set_ylabel("$T$ / K")
    axes[1].set_ylim(2.5, 5.5)
    axes[1].legend(loc="best", frameon=False)

    for _temp, _group in isotherms.groupby("Temperature (K)"):
        axes[2].plot(
            _group["Density (l, kg/m3)"],
            np.array(_group["Pressure (MPa)"]) * (si.MEGA * si.PASCAL) / si.BAR,
            "o", mec="k", mfc="None", ms=4, label=f"{_temp} K",
        )

    for _temp in sorted(isotherms["Temperature (K)"].unique()):
        _rho_min = isotherms[isotherms["Temperature (K)"] == _temp]["Density (l, kg/m3)"].min()
        _rho_max = isotherms[isotherms["Temperature (K)"] == _temp]["Density (l, kg/m3)"].max()
        _densities = np.linspace(_rho_min * 0.8, _rho_max * 1.2, 150)

        _valid_densities, _p_vrq, _p_cs, _p_vrcs = [], [], [], []
        for _rho in _densities:
            try:
                _moldens = (_rho / gas["molarweight"]) * si.MOL / si.METER**3 * 1000
                _pig = _moldens * si.RGAS * _temp * si.KELVIN

                _s_vrq = feos.State(vrq_model, temperature=_temp * si.KELVIN, density=_moldens)
                _s_cs = feos.State(cs_model, temperature=_temp * si.KELVIN, density=_moldens)
                _s_vrcs = feos.State(vrcs_model, temperature=_temp * si.KELVIN, density=_moldens)

                _p_vrq.append(_s_vrq.pressure(feos.Contributions.Residual) / si.BAR + _pig / si.BAR)
                _p_cs.append(_s_cs.pressure(feos.Contributions.Residual) / si.BAR + _pig / si.BAR)
                _p_vrcs.append(_s_vrcs.pressure(feos.Contributions.Residual) / si.BAR + _pig / si.BAR)
                _valid_densities.append(_rho)
            except Exception:
                continue

        if _valid_densities:
            axes[2].plot(_valid_densities, _p_vrq, "--", lw=1.2, color=colors[1])
            axes[2].plot(_valid_densities, _p_cs, "-", lw=1.2, color=colors[0])
            axes[2].plot(_valid_densities, _p_vrcs, "-", lw=1.2, color=colors[2])

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
        + [dict(coefficient=f"c_lr[{i}]", value=popt.x[10 + i]) for i in range(5)]
    )
    mo.vstack(
        [
            mo.ui.table(fit_summary, label="Fitted coefficients (all 15 slots free)"),
            mo.md(f"`success={popt.success}`, `cost={popt.cost:.6f}`, `nfev={popt.nfev}`, `status={popt.status}`"),
        ]
    )
    return


@app.cell
def _(State, df_vrcs, gas, mo, np, pc_nist, pd, si, vle, vrcs_model):
    def _aad(model_col, nist_col, unit_factor=1.0):
        order = np.argsort(df_vrcs["temperature"])
        t_model = np.array(df_vrcs["temperature"])[order]
        y_model = np.array(df_vrcs[model_col])[order] * unit_factor

        t_nist = vle["Temperature (K)"].to_numpy()
        y_nist = vle[nist_col].to_numpy()

        mask = (t_nist >= t_model.min()) & (t_nist <= t_model.max())
        y_interp = np.interp(t_nist[mask], t_model, y_model)
        return 100 * np.mean(np.abs((y_interp - y_nist[mask]) / y_nist[mask]))

    _crit = State.critical_point_pure(vrcs_model)[0]
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
    mo.ui.table(fit_quality, label="Fit quality vs. NIST (helium, SAFT-VR-CS)")
    return


if __name__ == "__main__":
    app.run()
