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
    # Fitting SAFT-VR-CS to helium and hydrogen

    Marimo port of `fit_saft_vrcs.ipynb`: fixes the base Mie potential
    (`m`, `sigma`, `epsilon_k`, `lr`, `la`) at helium's `sina` values (or
    the hydrogen equivalent) and fits a *reduced* subset of the 11
    Feynman-Hibbs correction coefficients — `c_sigma[1]`, `c_epsilon_k[1]`,
    `c_lr[1]`, `c_lr[2]`, `c_lr[4]` — with the rest pinned at 1, exactly as
    the notebook's `eos_from_parameter` does. An earlier version of this
    notebook tried fitting all 11 unconstrained; even with `/sqrt(n)`
    group weighting and `bounds=(-8, 8)`, that overshot $T_c$ by ~60% and
    missed vapor pressure by ~85% — the reduced 5-parameter problem is
    less prone to this kind of runaway.

    The objective (`cost`) matches the notebook too: it calls
    `PhaseEquilibrium.pure` per VLE point (not the faster
    `PhaseDiagram.pure`), fits liquid density (weighted 2x) and vapor
    pressure along the sub-critical VLE curve, plus density along the
    pressure isotherms where available. Use the gas selector below to
    switch between helium and hydrogen — hydrogen has no isotherm table,
    so that term drops out automatically.
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
    ## Gas selection
    """)
    return


@app.cell
def _(mo):
    gas_choice = mo.ui.dropdown(options=["helium", "hydrogen"], value="helium", label="Gas to fit")
    gas_choice
    return (gas_choice,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-gas configuration

    - **base**: fixed Mie potential parameters (`la` is always 6).
      Helium's `sina` base is a refined literature set (not the raw Aasen
      Table II values); no equivalent refined base is on hand for hydrogen,
      so hydrogen falls back to the Aasen FH=1 literature parameters.
    - **vrqmie**: SAFT-VRQ-Mie baseline (Aasen et al., FH order 1).
    - **uvcs**: $uv$-CS-theory baseline, parameters already optimized to
      each gas's VLE (+ isotherms for helium).
    - **t_min**: low-temperature VLE cutoff below which the phase-equilibrium
      solver becomes unreliable (helium only; hydrogen's table already
      starts well clear of that regime).
    - **pc_extrap_n**: number of highest-temperature rows of the *unfiltered*
      VLE table used to linearly extrapolate the critical pressure to `tc`
      (NIST's tables stop just short of $T_c$, so $p_c$ isn't tabulated
      directly). Matched to each table's spacing: 15 rows for helium's
      0.05 K grid (~0.75 K window), 8 rows for hydrogen's coarser 0.5 K
      grid (~3.5 K window).
    """)
    return


@app.cell
def _(gas_choice):
    gas_configs = {
        "helium": dict(
            name="helium",
            molarweight=4.002602,
            tc=5.1953,
            base=dict(m=1.0, sigma=2.674056, epsilon_k=9.466831, lr=11.981772, la=6),
            vrqmie=dict(m=1.0, sigma=2.7443, epsilon_k=5.4195, lr=9, la=6, fh=1),
            uvcs=dict(
                sigma=2.6953,
                epsilon_k=5.2850,
                rep=8.2761,
                att=6.0,
                c_sigma=[4.1529, 17.130, 4.6523],
                c_epsilon_k=[0.25290, 0.0, 0.42998],
                c_rep=[0.88165, 1.0523, 2.7524, 0.97560, 0.82524],
            ),
            vle_path="data/helium_data/nist_vle.txt",
            isotherms_path="data/helium_data/nist_isotherms.txt",
            t_min=2.7,
            pc_extrap_n=15,
            par0=[0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        "hydrogen": dict(
            name="hydrogen",
            molarweight=2.0157309551872,
            tc=33.145,
            base=dict(m=1.0, sigma=3.0243, epsilon_k=26.706, lr=9.0, la=6),
            vrqmie=dict(m=1.0, sigma=3.0243, epsilon_k=26.706, lr=9, la=6, fh=1),
            uvcs=dict(
                sigma=3.0156,
                epsilon_k=23.8730,
                rep=8.101,
                att=6.0,
                c_sigma=[1.43832767e00, 6.45545710e-04, 1.50887940e00],
                c_epsilon_k=[1.0, 1.0, 1.0],
                c_rep=[2.76585117e-01, 1.78562579e01, 1.82868514e02, 1.26035205e-04, 2.88248756e01],
            ),
            vle_path="data/hydrogen_data/nist_vle.txt",
            isotherms_path=None,
            t_min=0.0,
            pc_extrap_n=8,
            par0=[0.0, 0.0, 0.0, 0.0, 0.0],
        ),
    }
    gas = gas_configs[gas_choice.value]
    return gas, gas_configs


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

    if gas["isotherms_path"] is not None:
        isotherms = pd.read_csv(gas["isotherms_path"], sep="\t")
    else:
        isotherms = None

    # NIST's tables stop just short of Tc, so the critical pressure isn't
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
    ## SAFT-VR-CS fit

    `eos_from_parameter` builds the SAFT-VR-CS model from the 5 free
    coefficients; `cost` returns the concatenated relative-residual vector
    that `least_squares` minimizes. Click the button to run — the fit
    calls `PhaseEquilibrium.pure` once per VLE point, so it's slower than
    the group-weighted `PhaseDiagram.pure` fit in `saftvrcs_comparison.py`.
    """)
    return


@app.cell
def _(EquationOfState, Parameters, feos, gas):
    def eos_from_parameter(pars):
        c_sigma = [1, pars[0], 1]
        c_epsilon_k = [1, pars[1], 1]
        c_lr = [1, pars[2], pars[3], 1, pars[4]]

        params = dict(gas["base"], c_sigma=c_sigma, c_epsilon_k=c_epsilon_k, c_lr=c_lr)
        pr = feos.PureRecord(
            identifier=feos.Identifier(gas["name"]), molarweight=gas["molarweight"], **params
        )
        return EquationOfState.saftvrcs_mie(Parameters.new_pure(pr))

    return (eos_from_parameter,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each deviation group (liquid density, vapor pressure, isotherm density,
    critical point) is scaled by $1/\sqrt{n_\mathrm{group}}$, not
    $1/n_\mathrm{group}$: the original notebook's `/n` scaling divides
    relative residuals by group size *before* squaring, so a group's
    contribution to the sum-of-squares shrinks like $1/n$ as it grows —
    with `PhaseEquilibrium.pure` inherently a fickle solver near the
    critical point, this let `least_squares` report `success=True` at a
    shallow, deflated minimum while still being ~30% off on liquid density.
    `/sqrt(n)` keeps each group's contribution to the objective independent
    of how many points it has (matching `saftvrcs_comparison.py`), so an
    actual mismatch shows up as real cost instead of vanishing into the
    group-size denominator.

    The critical-point group (`Tc`, `pc`) matters more than its 2 residuals
    suggest: without it, nothing in the VLE-only objective penalizes the
    optimizer for chasing the sub-critical points at the cost of
    overshooting $T_c$/$p_c$ — which is exactly what an earlier version of
    this cell did (Tc off by ~15%, ~60% AAD on vapor pressure, ~40% on
    liquid density from a `par0` fit without this term). Adding it, matching
    `saftvrcs_comparison.py`, anchors the whole envelope and lets the fit
    converge to <2% AAD on psat/liquid density from a plain `par0 = [0, 0,
    0, 0, 0]` start.
    """)
    return


@app.cell
def _(
    PhaseEquilibrium,
    State,
    eos_from_parameter,
    feos,
    gas,
    isotherms,
    np,
    pc_nist,
    si,
    vle,
):
    def cost(pars):
        eos = eos_from_parameter(pars)

        error_rhol = []
        error_ps = []
        n_vle = len(vle)
        for _, row in vle.iterrows():
            temp = row["Temperature (K)"]
            try:
                feos_vle = PhaseEquilibrium.pure(eos, temperature_or_pressure=temp * si.KELVIN)
                rho_l = row["Density (l, kg/m3)"]
                rho_l_feos = feos_vle.liquid.mass_density() / si.KILOGRAM * si.METER**3
                error_rhol.append(2 * (rho_l_feos - rho_l) / rho_l / np.sqrt(n_vle))

                pexp = row["Pressure (MPa)"]
                pfeos = feos_vle.vapor.pressure() / (si.MEGA * si.PASCAL)
                error_ps.append((pfeos - pexp) / pexp / np.sqrt(n_vle))
            except Exception:
                error_rhol.append(3)
                error_ps.append(3)

        err_rho = []
        if isotherms is not None:
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

        # Critical point (Tc, pc): without this, the optimizer is free to
        # chase the sub-critical VLE points at the cost of overshooting the
        # critical point, since nothing in the VLE-only residuals penalizes
        # that (see note above).
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

        return np.concatenate([error_rhol, error_ps, err_rho, err_crit])

    return (cost,)


@app.cell
def _(mo):
    fit_button = mo.ui.run_button(label="Run SAFT-VR-CS fit (~1-2 min)")
    fit_button
    return (fit_button,)


@app.cell
def _(cost, fit_button, gas, least_squares, mo):
    mo.stop(not fit_button.value, mo.md("*Click the button above to run the fit.*"))

    with mo.status.spinner(title=f"Fitting SAFT-VR-CS c-parameters for {gas['name']}..."):
        # Unbounded, an 11-parameter fit against ~40-300 residuals can drive
        # c_lr into the thousands: a degenerate potential shape that still
        # threads the fitted VLE points but has no coherent phase envelope
        # elsewhere. Bounds matching saftvrcs_comparison.py's working fit
        # keep the correction physically a correction, not a blow-up.
        popt = least_squares(cost, gas["par0"], bounds=(-8, 8), max_nfev=300)
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
def _(PhaseDiagram, eos_from_parameter, feos, popt, si):
    vrcs_model = eos_from_parameter(popt.x)
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
    _ncols = 3 if isotherms is not None else 2
    fig, axes = plt.subplots(1, _ncols, figsize=(6 * _ncols, 6))

    # --- vapor pressure vs 1/T ---
    axes[0].plot(
        1 / vle["Temperature (K)"],
        vle["Pressure (MPa)"] * (si.MEGA * si.PASCAL / si.BAR),
        "o", mec="k", mfc="None", label="NIST",
    )
    axes[0].plot(1 / np.array(df_vrq["temperature"]), np.array(df_vrq["pressure"]) * (si.PASCAL / si.BAR), "--", color=colors[1], label="SAFT-VRQ-Mie")
    axes[0].plot(1 / np.array(df_cs["temperature"]), np.array(df_cs["pressure"]) * (si.PASCAL / si.BAR), "-", color=colors[0], label="$uv$-CS")
    axes[0].plot(1 / np.array(df_vrcs["temperature"]), np.array(df_vrcs["pressure"]) * (si.PASCAL / si.BAR), "r", label="SAFT-VR-CS (fit)")
    axes[0].set_xlabel("$1/T$ / K$^{-1}$")
    axes[0].set_ylabel("$p$ / bar")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_title(gas["name"])

    # --- VLE envelope, density vs temperature ---
    axes[1].plot(vle["Density (l, kg/m3)"], vle["Temperature (K)"], "o", mec="k", mfc="None", ms=6)
    axes[1].plot(vle["Density (v, kg/m3)"], vle["Temperature (K)"], "o", mec="k", mfc="None", ms=6)
    axes[1].plot(df_vrq["mass density liquid"], df_vrq["temperature"], "--", color=colors[1], label="SAFT-VRQ-Mie")
    axes[1].plot(df_vrq["mass density vapor"], df_vrq["temperature"], "--", color=colors[1])
    axes[1].plot(df_cs["mass density liquid"], df_cs["temperature"], "-", color=colors[0], label="$uv$-CS")
    axes[1].plot(df_cs["mass density vapor"], df_cs["temperature"], "-", color=colors[0])
    axes[1].plot(df_vrcs["mass density liquid"], df_vrcs["temperature"], "r", label="SAFT-VR-CS (fit)")
    axes[1].plot(df_vrcs["mass density vapor"], df_vrcs["temperature"], "r")
    axes[1].set_xlabel(r"$\rho$ / (kg/m³)")
    axes[1].set_ylabel("$T$ / K")
    axes[1].legend(loc="best", frameon=False)

    # --- isotherms, pressure vs density (helium only) ---
    if isotherms is not None:
        for _temp, _group in isotherms.groupby("Temperature (K)"):
            axes[2].plot(
                _group["Density (l, kg/m3)"],
                np.array(_group["Pressure (MPa)"]) * (si.MEGA * si.PASCAL) / si.BAR,
                "o", mec="k", mfc="None", ms=4, label=f"{_temp} K",
            )

        # Overlay each model's p(rho) at the isotherm temperatures. Mie EoS
        # residual pressure + ideal-gas term, since these are single-phase
        # states, not saturation points from PhaseDiagram.
        for _temp in sorted(isotherms["Temperature (K)"].unique()):
            _rho_min = isotherms[isotherms["Temperature (K)"] == _temp]["Density (l, kg/m3)"].min()
            _rho_max = isotherms[isotherms["Temperature (K)"] == _temp]["Density (l, kg/m3)"].max()
            _densities = np.linspace(_rho_min * 0.8, _rho_max * 1.2, 100)

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
                axes[2].plot(_valid_densities, _p_vrcs, "r", lw=1.2)

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
        [
            dict(coefficient="c_sigma[1]", value=popt.x[0]),
            dict(coefficient="c_epsilon_k[1]", value=popt.x[1]),
            dict(coefficient="c_lr[1]", value=popt.x[2]),
            dict(coefficient="c_lr[2]", value=popt.x[3]),
            dict(coefficient="c_lr[4]", value=popt.x[4]),
        ]
    )
    mo.vstack(
        [
            mo.ui.table(fit_summary, label="Fitted coefficients (c_sigma/c_epsilon_k/c_lr others fixed at 1)"),
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
    mo.ui.table(fit_quality, label=f"Fit quality vs. NIST ({gas['name']})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effective potential parameters vs. quantum parameter $D^*$

    Reproduces the style of Aasen et al.'s Fig. S6: the Feynman-Hibbs-1
    correction ratios $\sigma_\mathrm{eff}/\sigma$, $\varepsilon_\mathrm{eff}/\varepsilon$,
    and $\lambda_{r,\mathrm{eff}}/\lambda_r$ as functions of the dimensionless
    quantum parameter $D^* = \hbar^2/(12 k_B T m \sigma^2)$ (larger $D^*$ =
    lower temperature / lighter mass = more quantum).

    "Mie-FH1" evaluates the literature correction formula with unit scaling
    coefficients (`c_sigma = c_epsilon_k = c_lr = 1`) at each gas's Aasen
    FH1 base `lr`; "Optimal" applies this notebook's fitted `c_sigma`,
    `c_epsilon_k`, `c_lr` at the `sina`-style base `lr` used for the fit.
    For hydrogen these two bases coincide (no refined base is on hand), so
    both curves start from the same `lr` and only diverge because of the
    fitted correction.

    The correction functions themselves (`eff_sigma`, `eff_epsilon_k`,
    `eff_lr_ratio`) are ported 1:1 from
    `crates/feos/src/saftvrcs/corresponding_states.rs` — same polynomial
    constants, same rational-function form — so this only re-evaluates the
    existing Rust formulas in Python rather than approximating them.
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

    def eff_lr_ratio(d_star, lr, c_scale=None):
        c_scale = c_scale if c_scale is not None else [1.0] * 5
        pref = [(_C_LR[i][0] + lr * (_C_LR[i][1] + lr * _C_LR[i][2])) * c_scale[i] for i in range(5)]
        num = d_star * (d_star * (d_star * pref[2] + pref[1]) + pref[0]) + 1.0
        den = d_star * (d_star * pref[4] + pref[3]) + 1.0
        return num / den

    def eff_sigma(d_star, lr, c_scale=None):
        c_scale = c_scale if c_scale is not None else [1.0] * 3
        c0 = _C_SIGMA[0] + lr * (_C_SIGMA[1] + lr * _C_SIGMA[2])
        c1 = _C_SIGMA[3] + lr * (_C_SIGMA[4] + lr * _C_SIGMA[5])
        c2 = _C_SIGMA[6] + lr * (_C_SIGMA[7] + lr * _C_SIGMA[8])
        return (d_star * (d_star * c1 * c_scale[1] + c0 * c_scale[0]) + 1.0) / (d_star * c2 * c_scale[2] + 1.0)

    def eff_epsilon_k(d_star, lr, c_scale=None):
        c_scale = c_scale if c_scale is not None else [1.0] * 3
        c0 = _C_EPSILON[0] + lr * (_C_EPSILON[1] + lr * _C_EPSILON[2])
        c1 = _C_EPSILON[3] + lr * (_C_EPSILON[4] + lr * _C_EPSILON[5])
        c2 = _C_EPSILON[6] + lr * (_C_EPSILON[7] + lr * _C_EPSILON[8])
        return (d_star * (d_star * c1 * c_scale[1] + c0 * c_scale[0]) + 1.0) / (d_star * c2 * c_scale[2] + 1.0)

    return eff_epsilon_k, eff_lr_ratio, eff_sigma


@app.cell
def _(eff_epsilon_k, eff_lr_ratio, eff_sigma, gas, np, popt):
    d_star = np.linspace(0.0, 0.05, 200)

    _lr_fh1 = gas["vrqmie"]["lr"]
    _lr_opt = gas["base"]["lr"]
    _c_sigma_opt = [1, popt.x[0], 1]
    _c_epsilon_opt = [1, popt.x[1], 1]
    _c_lr_opt = [1, popt.x[2], popt.x[3], 1, popt.x[4]]

    lr_ratio_fh1 = eff_lr_ratio(d_star, _lr_fh1)
    sigma_ratio_fh1 = eff_sigma(d_star, _lr_fh1)
    epsilon_ratio_fh1 = eff_epsilon_k(d_star, _lr_fh1)

    lr_ratio_opt = eff_lr_ratio(d_star, _lr_opt, _c_lr_opt)
    sigma_ratio_opt = eff_sigma(d_star, _lr_opt, _c_sigma_opt)
    epsilon_ratio_opt = eff_epsilon_k(d_star, _lr_opt, _c_epsilon_opt)
    return (
        d_star,
        epsilon_ratio_fh1,
        epsilon_ratio_opt,
        lr_ratio_fh1,
        lr_ratio_opt,
        sigma_ratio_fh1,
        sigma_ratio_opt,
    )


@app.cell
def _(
    d_star,
    epsilon_ratio_fh1,
    epsilon_ratio_opt,
    gas,
    lr_ratio_fh1,
    lr_ratio_opt,
    plt,
    sigma_ratio_fh1,
    sigma_ratio_opt,
):
    fig3, axes3 = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

    axes3[0].plot(d_star, lr_ratio_fh1, "-", color="tab:blue", label=f"{gas['name'].capitalize()}, Mie-FH1")
    axes3[0].plot(d_star, lr_ratio_opt, "--", color="tab:orange", label=f"{gas['name'].capitalize()}, Optimal")
    axes3[0].set_ylabel(r"$\nu_\mathrm{eff}/\nu$")
    axes3[0].legend(frameon=False)

    axes3[1].plot(d_star, sigma_ratio_fh1, "-", color="tab:blue")
    axes3[1].plot(d_star, sigma_ratio_opt, "--", color="tab:orange")
    axes3[1].set_ylabel(r"$\sigma_\mathrm{eff}/\sigma$")

    axes3[2].plot(d_star, epsilon_ratio_fh1, "-", color="tab:blue")
    axes3[2].plot(d_star, epsilon_ratio_opt, "--", color="tab:orange")
    axes3[2].set_ylabel(r"$\varepsilon_\mathrm{eff}/\varepsilon$")
    axes3[2].set_xlabel("$D^*$")

    plt.tight_layout()
    fig3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Helium-hydrogen mixture vs. GEMC data

    Compares the SAFT-VR-CS pure-component models against **Gibbs Ensemble
    Monte Carlo (GEMC) VLE data** for the He/H2 binary
    (`data/helium_hydrogen_gemc_vle.csv`: temperature, pressure, and both
    phases' helium mole fraction at 4 isotherms).

    This needs *both* gases' fitted models at once, which the single
    `gas_choice` dropdown above doesn't give directly (it only fits one gas
    at a time). So this section refits both components independently with
    a self-contained, faster version of the same objective — `PhaseDiagram.pure`
    instead of `PhaseEquilibrium.pure`, no isotherm term — which converges
    in ~1s per gas instead of the ~1-2 min above (validated to reach the
    same quality: <2% AAD on psat/liquid density for both gases).

    The two pure fits are then combined into a binary mixture via
    `Parameters.new_binary([he_pr, h2_pr], k_ij=...)`. No literature `k_ij`
    is on hand for this pair, so it defaults to 0 (no correction) —
    adjustable below to see how much a single binary parameter can close
    the gap.

    For each GEMC point, `PhaseEquilibrium.bubble_point` is called at the
    experimental $(T, x_\mathrm{He}^\mathrm{liquid})$ to predict $p$ and
    $y_\mathrm{He}^\mathrm{vapor}$ — a direct tie-line comparison, and more
    robust than tracing the full binary envelope with `PhaseDiagram.binary_vle`
    (which fails to converge at the lowest GEMC temperature, 20.4 K, where
    helium-rich mixtures are already supercritical).
    """)
    return


@app.cell
def _(
    EquationOfState,
    Parameters,
    PhaseDiagram,
    State,
    feos,
    least_squares,
    np,
    pd,
    si,
):
    def fit_mixture_component(gas_config):
        vle_full = pd.read_csv(gas_config["vle_path"], sep="\t")
        vle = vle_full[
            (vle_full["Temperature (K)"] > gas_config["t_min"])
            & (vle_full["Temperature (K)"] < 0.9 * gas_config["tc"])
        ]
        _diam_rows = vle_full.tail(gas_config["pc_extrap_n"])
        _p_fit = np.polyfit(_diam_rows["Temperature (K)"], _diam_rows["Pressure (MPa)"], 1)
        pc_nist = np.polyval(_p_fit, gas_config["tc"])

        def pure_record(pars):
            c_sigma = [1, pars[0], 1]
            c_epsilon_k = [1, pars[1], 1]
            c_lr = [1, pars[2], pars[3], 1, pars[4]]
            params = dict(gas_config["base"], c_sigma=c_sigma, c_epsilon_k=c_epsilon_k, c_lr=c_lr)
            return feos.PureRecord(
                identifier=feos.Identifier(gas_config["name"]),
                molarweight=gas_config["molarweight"],
                **params,
            )

        t_nist = vle["Temperature (K)"].to_numpy()
        psat_nist = vle["Pressure (MPa)"].to_numpy()
        rhol_nist = vle["Density (l, kg/m3)"].to_numpy()
        n_vle = len(t_nist)

        def cost(pars):
            eos = EquationOfState.saftvrcs_mie(Parameters.new_pure(pure_record(pars)))
            try:
                df = PhaseDiagram.pure(
                    eos, min_temperature=t_nist.min() * si.KELVIN, npoints=60
                ).to_dict(feos.Contributions.Residual)
            except Exception:
                return np.full(2 * n_vle + 2, 3.0)

            order = np.argsort(df["temperature"])
            t_model = np.array(df["temperature"])[order]
            p_model = np.array(df["pressure"])[order] * 1e-6
            rl_model = np.array(df["mass density liquid"])[order]

            p_i = np.interp(t_nist, t_model, p_model)
            rl_i = np.interp(t_nist, t_model, rl_model)
            res_p = (p_i - psat_nist) / psat_nist / np.sqrt(n_vle)
            res_rl = 2 * (rl_i - rhol_nist) / rhol_nist / np.sqrt(n_vle)

            try:
                crit = State.critical_point_pure(eos)[0]
                tc_m = crit.temperature / si.KELVIN
                pc_m = crit.pressure() / si.MEGA / si.PASCAL
                res_crit = np.array(
                    [(tc_m - gas_config["tc"]) / gas_config["tc"], (pc_m - pc_nist) / pc_nist]
                ) / np.sqrt(2)
            except Exception:
                res_crit = np.full(2, 3.0) / np.sqrt(2)

            result = np.concatenate([res_p, res_rl, res_crit])
            return np.where(np.isfinite(result), result, 3.0)

        popt = least_squares(cost, gas_config["par0"], bounds=(-8, 8), max_nfev=200, diff_step=1e-3)
        return pure_record(popt.x), popt

    return (fit_mixture_component,)


@app.cell
def _(mo):
    mix_fit_button = mo.ui.run_button(label="Fit He + H2 pure components for mixture (~few sec)")
    mix_fit_button
    return (mix_fit_button,)


@app.cell
def _(fit_mixture_component, gas_configs, mix_fit_button, mo):
    mo.stop(
        not mix_fit_button.value,
        mo.md("*Click the button above to fit both pure components for the mixture comparison.*"),
    )

    with mo.status.spinner(title="Fitting helium and hydrogen SAFT-VR-CS components..."):
        he_pr, he_mix_popt = fit_mixture_component(gas_configs["helium"])
        h2_pr, h2_mix_popt = fit_mixture_component(gas_configs["hydrogen"])

    mo.md(
        f"Helium: `success={he_mix_popt.success}`, `cost={he_mix_popt.cost:.6f}`. "
        f"Hydrogen: `success={h2_mix_popt.success}`, `cost={h2_mix_popt.cost:.6f}`."
    )
    return h2_pr, he_pr


@app.cell
def _(mo):
    kij_slider = mo.ui.slider(
        -0.2, 0.2, step=0.005, value=0.0, label="k_ij (binary interaction parameter, no literature value on hand)"
    )
    kij_slider
    return (kij_slider,)


@app.cell
def _(EquationOfState, Parameters, h2_pr, he_pr, kij_slider):
    mix_eos = EquationOfState.saftvrcs_mie(Parameters.new_binary([he_pr, h2_pr], k_ij=kij_slider.value))
    return (mix_eos,)


@app.cell
def _(PhaseEquilibrium, mix_eos, np, pd, si):
    gemc = pd.read_csv("data/helium_hydrogen_gemc_vle.csv", sep=r"\s+")

    _p_model, _y_model = [], []
    for _, _row in gemc.iterrows():
        try:
            _vle = PhaseEquilibrium.bubble_point(
                mix_eos, _row["temperature[K]"] * si.KELVIN, np.array([_row["x_he"], 1 - _row["x_he"]])
            )
            _p_model.append(_vle.vapor.pressure() / si.BAR)
            _y_model.append(_vle.vapor.molefracs[0])
        except Exception:
            _p_model.append(np.nan)
            _y_model.append(np.nan)

    gemc = gemc.assign(p_model=_p_model, y_model=_y_model)
    return (gemc,)


@app.cell
def _(PhaseDiagram, mix_eos, si):
    PhaseDiagram.binary_vle(mix_eos, 20.4*si.KELVIN)
    return


@app.cell
def _(gemc, mo, np, pd):
    _valid = gemc.dropna(subset=["p_model", "y_model"])
    mix_aad = pd.DataFrame(
        [
            dict(
                quantity="pressure (%)",
                AAD=100 * np.mean(np.abs((_valid["p_model"] - _valid["pressure[bar]"]) / _valid["pressure[bar]"])),
            ),
            dict(
                quantity="y_he (%)",
                AAD=100 * np.mean(np.abs((_valid["y_model"] - _valid["y_he"]) / _valid["y_he"])),
            ),
        ]
    )
    mo.ui.table(mix_aad, label=f"Mixture fit quality vs. GEMC ({len(_valid)}/{len(gemc)} points converged)")
    return


@app.cell
def _(colors, gemc, plt):
    _temps = sorted(gemc["temperature[K]"].unique())
    fig4, axes4 = plt.subplots(1, len(_temps), figsize=(4.5 * len(_temps), 4.5), sharey=True)

    for _i, _t in enumerate(_temps):
        _ax = axes4[_i]
        _group = gemc[gemc["temperature[K]"] == _t].sort_values("x_he")

        _ax.plot(_group["x_he"], _group["pressure[bar]"], "o", mec="k", mfc="None", label="GEMC (liquid)")
        _ax.plot(_group["y_he"], _group["pressure[bar]"], "s", mec="k", mfc="None", label="GEMC (vapor)")
        _ax.plot(_group["x_he"], _group["p_model"], "-", color=colors[3], label="SAFT-VR-CS (liquid)")
        _ax.plot(_group["y_model"], _group["p_model"], "--", color=colors[3], label="SAFT-VR-CS (vapor)")

        _ax.set_xlabel(r"$x_\mathrm{He}, y_\mathrm{He}$")
        _ax.set_title(f"{_t} K")
        if _i == 0:
            _ax.set_ylabel("$p$ / bar")
            _ax.legend(fontsize=7, frameon=False)

    plt.tight_layout()
    fig4
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
