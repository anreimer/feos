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
    # SAFT-VR-CS for Helium: evaluation against SAFT-VRQ-Mie and $uv$-CS-theory

    Helium-4 below ~0.9 $T_c$ is a good stress test for quantum-corrected
    equations of state: the mass is small enough that Feynman-Hibbs
    corrections matter a lot for the VLE envelope and isotherms.

    This notebook builds three models for helium and compares them against
    NIST reference data:

    - **SAFT-VRQ-Mie** (Aasen et al.), literature parameters, Feynman-Hibbs order 1.
    - **$uv$-CS-theory**, parameters optimized to helium VLE + isotherms.
    - **SAFT-VR-CS** (this implementation), same Mie-potential parameters,
      with sliders below so you can see how sensitive the fit is to each one.

    Reproduces the comparison in Fig. 4 of *J. Chem. Phys.* 162, 031101 (2025);
    doi: 10.1063/5.0243474.
    """)
    return


@app.cell
def _():
    import feos
    from feos import EquationOfState, Parameters, PhaseDiagram, State
    import si_units as si
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("notebook")
    sns.set_palette("Dark2")
    sns.set_style("ticks")
    colors = sns.palettes.color_palette("Dark2", 8)
    return (
        EquationOfState,
        Parameters,
        PhaseDiagram,
        State,
        colors,
        feos,
        np,
        pd,
        plt,
        si,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reference data (NIST)
    """)
    return


@app.cell
def _(np, pd):
    MOLARWEIGHT_HE = 4.002602  # g/mol
    TC_NIST = 5.1953  # K, NIST critical temperature for helium-4

    # Finer-resolution NIST VLE table (0.02 K spacing vs. 0.05 K previously),
    # merged from separate liquid/vapor NIST exports into one file with the
    # densities converted from mol/l to kg/m3.
    vle_full = pd.read_csv("data/helium_data/nist_vle_fine.txt", sep="\t")
    _min_t_vle = 2.7268  # K, matches the low-T cutoff used previously
    vle = vle_full[
        (vle_full["Temperature (K)"] >= _min_t_vle) & (vle_full["Temperature (K)"] < 0.9 * TC_NIST)
    ]

    # NIST's tables stop just short of Tc, so the critical density isn't
    # tabulated directly. Extrapolate the rectilinear diameter
    # (rho_l + rho_v)/2, which is linear in T close to Tc, from the 15
    # highest-temperature rows of the *unfiltered* VLE table.
    _diam_rows = vle_full.tail(15)
    _rho_diam = (_diam_rows["Density (l, kg/m3)"] + _diam_rows["Density (v, kg/m3)"]) / 2
    _diam_fit = np.polyfit(_diam_rows["Temperature (K)"], _rho_diam, 1)
    RHOC_NIST = np.polyval(_diam_fit, TC_NIST)  # kg/m^3

    # Same idea for the critical pressure: linear extrapolation of psat(T)
    # from the same near-critical rows (the extrapolation distance is only
    # ~0.02 K, so linear is accurate here).
    _p_fit = np.polyfit(_diam_rows["Temperature (K)"], _diam_rows["Pressure (MPa)"], 1)
    PC_NIST = np.polyval(_p_fit, TC_NIST)  # MPa

    isotherms_full = pd.read_csv("data/helium_data/nist_isotherms.txt", sep="\t")
    isotherms = isotherms_full.loc[isotherms_full["Pressure (MPa)"] < 5]
    return MOLARWEIGHT_HE, PC_NIST, RHOC_NIST, TC_NIST, isotherms, vle


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Models

    ### SAFT-VRQ-Mie (fixed, literature parameters)
    """)
    return


@app.cell
def _(EquationOfState, MOLARWEIGHT_HE, Parameters, PhaseDiagram, feos, si):
    vrqmie_params = dict(m=1.0, sigma=2.7443, epsilon_k=5.4195, lr=9, la=6, fh=1)
    vrq_pr = feos.PureRecord(
        identifier=feos.Identifier("helium"),
        molarweight=MOLARWEIGHT_HE,
        **vrqmie_params,
    )
    vrq_model = EquationOfState.saftvrqmie(Parameters.new_pure(vrq_pr))
    df_vrq = PhaseDiagram.pure(
        vrq_model, min_temperature=2 * si.KELVIN, npoints=250
    ).to_dict(feos.Contributions.Residual)
    return df_vrq, vrq_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### $uv$-CS-theory (optimized parameters from Thijs)
    """)
    return


@app.cell
def _(EquationOfState, MOLARWEIGHT_HE, Parameters, PhaseDiagram, feos, si):
    uvcs_params = dict(
        sigma=2.6953,
        epsilon_k=5.2850,
        rep=8.2761,
        att=6.0,
        c_sigma=[4.1529, 17.130, 4.6523],
        c_epsilon_k=[0.25290, 0.0, 0.42998],
        c_rep=[0.88165, 1.0523, 2.7524, 0.97560, 0.82524],
    )
    uvcs_pr = feos.PureRecord(
        identifier=feos.Identifier("helium"),
        molarweight=MOLARWEIGHT_HE,
        **uvcs_params,
    )
    cs_model = EquationOfState.uvcstheory(Parameters.new_pure(uvcs_pr))
    df_cs = PhaseDiagram.pure(
        cs_model, min_temperature=2 * si.KELVIN, npoints=250
    ).to_dict(feos.Contributions.Residual)
    return cs_model, df_cs, uvcs_params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SAFT-VR-CS (this implementation, interactive)

    Starts from the same Feynman-Hibbs-corrected Mie parameters as
    $uv$-CS-theory above. Tune them and re-submit to see how the chain
    formalism responds relative to the non-chain $uv$-CS reference.
    """)
    return


@app.cell
def _(mo, uvcs_params):
    vrcs_form = mo.ui.array(
        [
            mo.ui.number(0.8, 1.5, 0.01, value=1.0, label="m (chain length)"),
            mo.ui.number(2.0, 3.5, 0.0001, value=uvcs_params["sigma"], label="σ / Å"),
            mo.ui.number(3.0, 8.0, 0.0001, value=uvcs_params["epsilon_k"], label="ε/k / K"),
            mo.ui.number(6.0, 14.0, 0.0001, value=uvcs_params["rep"], label="λr"),
            mo.ui.number(5.0, 7.0, 0.1, value=uvcs_params["att"], label="λa"),
        ]
    ).form(label="chain + Mie parameters")

    vrcs_c_sigma_form = mo.ui.array(
        [mo.ui.number(-5.0, 25.0, 0.0001, value=v) for v in uvcs_params["c_sigma"]]
    ).form(label="c_sigma")

    vrcs_c_epsilon_form = mo.ui.array(
        [mo.ui.number(-5.0, 5.0, 0.0001, value=v) for v in uvcs_params["c_epsilon_k"]]
    ).form(label="c_epsilon_k")

    vrcs_c_lr_form = mo.ui.array(
        [mo.ui.number(-5.0, 5.0, 0.0001, value=v) for v in uvcs_params["c_rep"]]
    ).form(label="c_lr")

    mo.hstack(
        [vrcs_form, vrcs_c_sigma_form, vrcs_c_epsilon_form, vrcs_c_lr_form],
        justify="start",
        gap=2,
    )
    return vrcs_c_epsilon_form, vrcs_c_lr_form, vrcs_c_sigma_form, vrcs_form


@app.cell
def _(
    EquationOfState,
    MOLARWEIGHT_HE,
    Parameters,
    PhaseDiagram,
    feos,
    mo,
    si,
    uvcs_params,
    vrcs_c_epsilon_form,
    vrcs_c_lr_form,
    vrcs_c_sigma_form,
    vrcs_form,
):
    # form.value is None until the user submits at least once; fall back to
    # the uv-CS-optimized defaults shown in the sliders above so the notebook
    # produces a result on first load.
    _m, _sigma, _epsilon_k, _lr, _la = vrcs_form.value or (
        1.0,
        uvcs_params["sigma"],
        uvcs_params["epsilon_k"],
        uvcs_params["rep"],
        uvcs_params["att"],
    )

    vrcs_params = dict(
        m=_m,
        sigma=_sigma,
        epsilon_k=_epsilon_k,
        lr=_lr,
        la=_la,
        c_sigma=list(vrcs_c_sigma_form.value or uvcs_params["c_sigma"]),
        c_epsilon_k=list(vrcs_c_epsilon_form.value or uvcs_params["c_epsilon_k"]),
        c_lr=list(vrcs_c_lr_form.value or uvcs_params["c_rep"]),
    )
    vrcs_pr = feos.PureRecord(
        identifier=feos.Identifier("helium"),
        molarweight=MOLARWEIGHT_HE,
        **vrcs_params,
    )
    vrcs_model = EquationOfState.saftvrcs_mie(Parameters.new_pure(vrcs_pr))

    try:
        df_vrcs = PhaseDiagram.pure(
            vrcs_model, min_temperature=2 * si.KELVIN, npoints=250
        ).to_dict(feos.Contributions.Residual)
        vrcs_error = None
    except Exception as exc:  # phase diagram solver can fail for bad params
        df_vrcs = None
        vrcs_error = str(exc)

    mo.stop(
        vrcs_error is not None,
        mo.md(f"**SAFT-VR-CS phase diagram failed for these parameters:** {vrcs_error}"),
    )
    return df_vrcs, vrcs_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Critical point comparison
    """)
    return


@app.cell
def _(
    PC_NIST,
    RHOC_NIST,
    State,
    TC_NIST,
    cs_model,
    mo,
    pd,
    si,
    vrcs_model,
    vrq_model,
):
    def _crit_row(label, model):
        state = State.critical_point_pure(model)[0]
        return dict(
            model=label,
            T_c=state.temperature / si.KELVIN,
            p_c=state.pressure() * 1e-6 / si.PASCAL,
            rho_c=state.mass_density() / (si.KILOGRAM / si.METER**3),
        )

    # p_c / rho_c aren't tabulated by NIST directly; PC_NIST / RHOC_NIST are
    # the near-critical extrapolations computed in the data-loading cell above.
    crit_rows = [dict(model="NIST", T_c=TC_NIST, p_c=PC_NIST, rho_c=RHOC_NIST)]
    for _label, _model in [
        ("SAFT-VRQ-Mie", vrq_model),
        ("uv-CS-theory", cs_model),
        ("SAFT-VR-CS", vrcs_model),
    ]:
        crit_rows.append(_crit_row(_label, _model))

    crit_df = pd.DataFrame(crit_rows)
    crit_table = mo.ui.table(crit_df, label="Critical point: T_c / K, p_c / MPa, rho_c / (kg/m^3)")
    crit_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## VLE envelope, vapor pressure, and isotherms
    """)
    return


@app.cell
def _(
    MOLARWEIGHT_HE,
    PC_NIST,
    RHOC_NIST,
    TC_NIST,
    colors,
    cs_model,
    df_cs,
    df_vrcs,
    df_vrq,
    feos,
    isotherms,
    np,
    plt,
    si,
    vle,
    vrcs_model,
    vrq_model,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 3.5))

    # --- Subplot 1: vapor pressure vs 1/T ---
    axes[0].plot(
        1 / vle["Temperature (K)"],
        vle["Pressure (MPa)"] * (si.MEGA * si.PASCAL / si.BAR),
        "o",
        mec="k",
        mfc="None",
        label="NIST",
    )
    axes[0].plot(1 / np.array(df_vrq["temperature"]), np.array(df_vrq["pressure"]) * (si.PASCAL / si.BAR), "--", color=colors[1], label="SAFT-VRQ-Mie")
    axes[0].plot(1 / np.array(df_cs["temperature"]), np.array(df_cs["pressure"]) * (si.PASCAL / si.BAR), "-", color=colors[0], label="$uv$-CS")
    axes[0].plot(1 / np.array(df_vrcs["temperature"]), np.array(df_vrcs["pressure"]) * (si.PASCAL / si.BAR), ":", color=colors[2], label="SAFT-VR-CS")
    axes[0].plot(
        1 / TC_NIST, PC_NIST * (si.MEGA * si.PASCAL / si.BAR),
        "*", ms=14, mec="k", mfc="gold", zorder=5, label="NIST crit. point",
    )
    axes[0].set_xlabel("$1/T$ / K$^{-1}$")
    axes[0].set_ylabel("$p$ / bar")
    axes[0].set_yscale("log")
    axes[0].set_xlim(0.2, 0.45)
    axes[0].legend(fontsize=8, frameon=False)

    # --- Subplot 2: VLE envelope, density vs temperature ---
    axes[1].plot(vle["Density (l, kg/m3)"], vle["Temperature (K)"], "o", mec="k", mfc="None")
    axes[1].plot(vle["Density (v, kg/m3)"], vle["Temperature (K)"], "o", mec="k", mfc="None")
    axes[1].plot(df_vrq["mass density liquid"], df_vrq["temperature"], "--", color=colors[1], label="SAFT-VRQ-Mie")
    axes[1].plot(df_vrq["mass density vapor"], df_vrq["temperature"], "--", color=colors[1])
    axes[1].plot(df_cs["mass density liquid"], df_cs["temperature"], "-", color=colors[0], label="$uv$-CS")
    axes[1].plot(df_cs["mass density vapor"], df_cs["temperature"], "-", color=colors[0])
    axes[1].plot(df_vrcs["mass density liquid"], df_vrcs["temperature"], ":", color=colors[2], label="SAFT-VR-CS")
    axes[1].plot(df_vrcs["mass density vapor"], df_vrcs["temperature"], ":", color=colors[2])
    axes[1].plot(
        RHOC_NIST, TC_NIST,
        "*", ms=14, mec="k", mfc="gold", zorder=5, label="NIST crit. point",
    )
    axes[1].set_xlabel(r"$\rho$ / (kg/m³)")
    axes[1].set_ylabel("$T$ / K")
    axes[1].set_ylim(2.5, 5.5)
    axes[1].legend(loc="best", frameon=False)

    # --- Subplot 3: isotherms, pressure vs density ---
    for _temp, _group in isotherms.groupby("Temperature (K)"):
        axes[2].plot(
            _group["Density (l, kg/m3)"],
            np.array(_group["Pressure (MPa)"]) * (si.MEGA * si.PASCAL) / si.BAR,
            "o",
            mec="k",
            mfc="None",
            ms=4,
            label=f"{_temp} K",
        )

    _isotherm_temps = sorted(isotherms["Temperature (K)"].unique())
    for _i, _temp in enumerate(_isotherm_temps):
        _rho_min = isotherms[isotherms["Temperature (K)"] == _temp]["Density (l, kg/m3)"].min()
        _rho_max = isotherms[isotherms["Temperature (K)"] == _temp]["Density (l, kg/m3)"].max()
        _densities = np.linspace(_rho_min * 0.8, _rho_max * 1.2, 200)

        _valid_densities = []
        _p_vrq, _p_cs, _p_vrcs = [], [], []
        for _rho in _densities:
            try:
                _moldens = (_rho / MOLARWEIGHT_HE) * si.MOL / si.METER**3 * 1000
                _pig = _moldens * si.RGAS * _temp * si.KELVIN

                _state_vrq = feos.State(vrq_model, temperature=_temp * si.KELVIN, density=_moldens)
                _state_cs = feos.State(cs_model, temperature=_temp * si.KELVIN, density=_moldens)
                _state_vrcs = feos.State(vrcs_model, temperature=_temp * si.KELVIN, density=_moldens)

                _p_vrq.append(_state_vrq.pressure(feos.Contributions.Residual) / si.BAR + _pig / si.BAR)
                _p_cs.append(_state_cs.pressure(feos.Contributions.Residual) / si.BAR + _pig / si.BAR)
                _p_vrcs.append(_state_vrcs.pressure(feos.Contributions.Residual) / si.BAR + _pig / si.BAR)
                _valid_densities.append(_rho)
            except Exception:
                continue

        if _valid_densities:
            _color = colors[_i % len(colors)]
            axes[2].plot(_valid_densities, _p_vrq, "--", lw=1.2, color=colors[1])
            axes[2].plot(_valid_densities, _p_cs, "-", lw=1.2, color=colors[0])
            axes[2].plot(_valid_densities, _p_vrcs, ":", lw=1.2, color=colors[2])

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
    ## Deviations from NIST along the VLE curve

    %AAD computed by interpolating each model's VLE curve onto the NIST
    temperature grid shown above.
    """)
    return


@app.cell
def _(df_cs, df_vrcs, df_vrq, mo, np, pd, vle):
    def _aad(model_df, model_col, nist_col, unit_factor=1.0):
        order = np.argsort(model_df["temperature"])
        t_model = np.array(model_df["temperature"])[order]
        y_model = np.array(model_df[model_col])[order] * unit_factor

        t_nist = vle["Temperature (K)"].to_numpy()
        y_nist = vle[nist_col].to_numpy()

        mask = (t_nist >= t_model.min()) & (t_nist <= t_model.max())
        y_interp = np.interp(t_nist[mask], t_model, y_model)
        return 100 * np.mean(np.abs((y_interp - y_nist[mask]) / y_nist[mask]))

    _rows = []
    for _label, _df in [
        ("SAFT-VRQ-Mie", df_vrq),
        ("uv-CS-theory", df_cs),
        ("SAFT-VR-CS", df_vrcs),
    ]:
        _rows.append(
            dict(
                model=_label,
                AAD_psat=_aad(_df, "pressure", "Pressure (MPa)", unit_factor=1e-6),
                AAD_rho_liq=_aad(_df, "mass density liquid", "Density (l, kg/m3)"),
                AAD_rho_vap=_aad(_df, "mass density vapor", "Density (v, kg/m3)"),
            )
        )

    dev_df = pd.DataFrame(_rows)
    mo.ui.table(dev_df, label="%AAD vs NIST over the VLE range shown above")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit SAFT-VR-CS FH-correction coefficients from an Aasen base

    Rather than borrowing $uv$-CS's optimized parameters, fix $m$, $\sigma$,
    $\varepsilon/k_B$, and $\lambda_r$ at one of the **Mie parameter sets
    for helium from Aasen et al., Table II** (FH=0: classical, no quantum
    correction in the base potential; FH=1: Feynman-Hibbs order-1
    corrected), with $\lambda_a=6$ in both cases, and fit only the 11
    Feynman-Hibbs correction coefficients (`c_sigma`, `c_epsilon_k`,
    `c_lr`) to five deviation groups using `scipy.optimize.least_squares`:
    saturation pressure, liquid density, and vapor density along the NIST
    VLE curve above, the NIST critical point ($T_c$, $p_c$), and the NIST
    pressure isotherms. Each group is weighted *equally* in the objective
    (its residuals are scaled by $1/\sqrt{n_\mathrm{group}}$) regardless of
    how many points it contributes — otherwise the 2-point critical-point
    group is drowned out by the ~300 VLE/isotherm points and the fit
    overshoots $T_c$/$p_c$ to chase the sub-critical data instead.

    The natural initial guess differs by base: all coefficients at 0
    (no extra correction) reproduces $T_c$ well for the **FH=0** base,
    while all coefficients at 1 is the better start for the **FH=1**
    base — that's what the original example notebook's "ailo" parameters
    used pre-fit. The cell below picks the matching default automatically.
    """)
    return


@app.cell
def _(mo):
    fh_choice = mo.ui.dropdown(
        options=["FH0", "FH1"],
        value="FH1",
        label="Aasen base parameters (FH order)",
    )
    fh_choice
    return (fh_choice,)


@app.cell
def _(fh_choice, np):
    AASEN_HE_FH0 = dict(m=1.0, sigma=3.3530, epsilon_k=4.44, lr=14.84, la=6.0)
    AASEN_HE_FH1 = dict(m=1.0, sigma=2.7443, epsilon_k=5.4195, lr=9.0, la=6.0)

    if fh_choice.value == "FH0":
        aasen_base = AASEN_HE_FH0
        fit_x0 = np.zeros(11)
    else:
        aasen_base = AASEN_HE_FH1
        fit_x0 = np.ones(11)
    return aasen_base, fit_x0


@app.cell
def _(mo):
    fit_button = mo.ui.run_button(label="Run SAFT-VR-CS c-parameter fit (~60-120s)")
    fit_button
    return (fit_button,)


@app.cell
def _(
    EquationOfState,
    MOLARWEIGHT_HE,
    PC_NIST,
    Parameters,
    PhaseDiagram,
    State,
    TC_NIST,
    aasen_base,
    feos,
    fit_button,
    fit_x0,
    isotherms,
    mo,
    np,
    si,
    vle,
):
    mo.stop(not fit_button.value, mo.md("*Click the button above to run the fit.*"))

    from scipy.optimize import least_squares

    _t_nist = vle["Temperature (K)"].to_numpy()
    _psat_nist = vle["Pressure (MPa)"].to_numpy()
    _rho_l_nist = vle["Density (l, kg/m3)"].to_numpy()
    _rho_v_nist = vle["Density (v, kg/m3)"].to_numpy()

    _t_iso = isotherms["Temperature (K)"].to_numpy()
    _rho_iso = isotherms["Density (l, kg/m3)"].to_numpy()
    _p_iso_nist = isotherms["Pressure (MPa)"].to_numpy() * (si.MEGA * si.PASCAL / si.BAR)

    # Five deviation groups, weighted *equally* regardless of how many
    # points each one has: dividing a group's residuals by sqrt(n_group)
    # makes its contribution to the sum-of-squares equal to its own mean
    # squared relative deviation, so a 2-point group (Tc, pc) carries the
    # same weight as e.g. the 84-point isotherm group instead of being
    # drowned out by it.
    _n_vle = len(_t_nist)
    _n_iso = len(_t_iso)
    _n_crit = 2
    _n_residuals = 3 * _n_vle + _n_crit + _n_iso

    def _build_model(c_vec):
        params = dict(
            aasen_base,
            c_sigma=list(c_vec[0:3]),
            c_epsilon_k=list(c_vec[3:6]),
            c_lr=list(c_vec[6:11]),
        )
        pr = feos.PureRecord(
            identifier=feos.Identifier("helium"),
            molarweight=MOLARWEIGHT_HE,
            **params,
        )
        return EquationOfState.saftvrcs_mie(Parameters.new_pure(pr))

    def _residuals(c_vec, npoints=25):
        try:
            model = _build_model(c_vec)
            df = PhaseDiagram.pure(
                model, min_temperature=2 * si.KELVIN, npoints=npoints
            ).to_dict(feos.Contributions.Residual)
        except Exception:
            return np.full(_n_residuals, 1.0)

        order = np.argsort(df["temperature"])
        t_model = np.array(df["temperature"])[order]
        p_model = np.array(df["pressure"])[order] * 1e-6
        rl_model = np.array(df["mass density liquid"])[order]
        rv_model = np.array(df["mass density vapor"])[order]

        # np.interp clamps to the boundary value outside t_model's range,
        # which keeps residuals smooth even when a candidate's computed
        # range doesn't yet reach the full NIST temperature range.
        p_i = np.interp(_t_nist, t_model, p_model)
        rl_i = np.interp(_t_nist, t_model, rl_model)
        rv_i = np.interp(_t_nist, t_model, rv_model)

        res_p = (p_i - _psat_nist) / _psat_nist / np.sqrt(_n_vle)
        res_rl = (rl_i - _rho_l_nist) / _rho_l_nist / np.sqrt(_n_vle)
        res_rv = (rv_i - _rho_v_nist) / _rho_v_nist / np.sqrt(_n_vle)

        # Critical point (Tc, pc): without this, the optimizer is free to
        # chase the sub-critical VLE points at the cost of overshooting the
        # critical point, since nothing in the VLE-only residuals penalizes
        # that.
        try:
            _crit_state = State.critical_point_pure(model)[0]
            _tc_model = _crit_state.temperature / si.KELVIN
            _pc_model = _crit_state.pressure() * 1e-6 / si.PASCAL
            res_crit = (
                np.array(
                    [
                        (_tc_model - TC_NIST) / TC_NIST,
                        (_pc_model - PC_NIST) / PC_NIST,
                    ]
                )
                / np.sqrt(_n_crit)
            )
        except Exception:
            res_crit = np.full(_n_crit, 1.0) / np.sqrt(_n_crit)

        # Pressure isotherms: p(rho) at fixed T, mostly supercritical,
        # anchors the single-phase density dependence away from the VLE
        # curve.
        res_iso = np.full(_n_iso, 1.0)
        for _i in range(_n_iso):
            try:
                _moldens = (_rho_iso[_i] / MOLARWEIGHT_HE) * si.MOL / si.METER**3 * 1000
                _pig = _moldens * si.RGAS * _t_iso[_i] * si.KELVIN
                _state = feos.State(model, temperature=_t_iso[_i] * si.KELVIN, density=_moldens)
                _p_model_bar = _state.pressure(feos.Contributions.Residual) / si.BAR + _pig / si.BAR
                res_iso[_i] = (_p_model_bar - _p_iso_nist[_i]) / _p_iso_nist[_i]
            except Exception:
                pass
        res_iso = res_iso / np.sqrt(_n_iso)

        result = np.concatenate([res_p, res_rl, res_rv, res_crit, res_iso])
        # feos can return NaN pressures for out-of-range candidates without
        # raising, which SciPy's SVD-based trust-region step can't handle.
        return np.where(np.isfinite(result), result, 1.0)

    with mo.status.spinner(title="Fitting SAFT-VR-CS c-parameters..."):
        fit_result = least_squares(
            _residuals, fit_x0, bounds=(-8, 8), max_nfev=300,
            xtol=1e-10, ftol=1e-10, gtol=1e-10, diff_step=1e-3,
        )

    fit_c_sigma = list(fit_result.x[0:3])
    fit_c_epsilon_k = list(fit_result.x[3:6])
    fit_c_lr = list(fit_result.x[6:11])

    vrcs_fit_params = dict(
        aasen_base,
        c_sigma=fit_c_sigma,
        c_epsilon_k=fit_c_epsilon_k,
        c_lr=fit_c_lr,
    )
    vrcs_fit_pr = feos.PureRecord(
        identifier=feos.Identifier("helium"),
        molarweight=MOLARWEIGHT_HE,
        **vrcs_fit_params,
    )
    vrcs_fit_model = EquationOfState.saftvrcs_mie(Parameters.new_pure(vrcs_fit_pr))
    df_vrcs_fit = PhaseDiagram.pure(
        vrcs_fit_model, min_temperature=2.5 * si.KELVIN, npoints=250
    ).to_dict(feos.Contributions.Residual)
    return (
        df_vrcs_fit,
        fit_c_epsilon_k,
        fit_c_lr,
        fit_c_sigma,
        fit_result,
        vrcs_fit_model,
    )


@app.cell
def _(fit_c_epsilon_k, fit_c_lr, fit_c_sigma, fit_result, mo, pd):
    fit_summary = pd.DataFrame(
        [
            dict(coefficient="c_sigma", values=[round(v, 4) for v in fit_c_sigma]),
            dict(coefficient="c_epsilon_k", values=[round(v, 4) for v in fit_c_epsilon_k]),
            dict(coefficient="c_lr", values=[round(v, 4) for v in fit_c_lr]),
        ]
    )
    mo.vstack(
        [
            mo.ui.table(fit_summary, label="Fitted coefficients"),
            mo.md(
                f"`success={fit_result.success}`, `cost={fit_result.cost:.4f}`, "
                f"`nfev={fit_result.nfev}`, `status={fit_result.status}`"
            ),
        ]
    )
    return


@app.cell
def _(
    MOLARWEIGHT_HE,
    PC_NIST,
    RHOC_NIST,
    State,
    TC_NIST,
    feos,
    isotherms,
    mo,
    np,
    pd,
    si,
    vrcs_fit_model,
):
    _crit_state = State.critical_point_pure(vrcs_fit_model)[0]
    _tc_fit = _crit_state.temperature / si.KELVIN
    _pc_fit = _crit_state.pressure() * 1e-6 / si.PASCAL
    _rhoc_fit = _crit_state.mass_density() / (si.KILOGRAM / si.METER**3)

    _iso_errs = []
    for _t, _rho, _p_nist in zip(
        isotherms["Temperature (K)"], isotherms["Density (l, kg/m3)"], isotherms["Pressure (MPa)"]
    ):
        try:
            _moldens = (_rho / MOLARWEIGHT_HE) * si.MOL / si.METER**3 * 1000
            _pig = _moldens * si.RGAS * _t * si.KELVIN
            _state = feos.State(vrcs_fit_model, temperature=_t * si.KELVIN, density=_moldens)
            _p_model_bar = _state.pressure(feos.Contributions.Residual) / si.BAR + _pig / si.BAR
            _p_nist_bar = _p_nist * (si.MEGA * si.PASCAL / si.BAR)
            _iso_errs.append(100 * abs(_p_model_bar - _p_nist_bar) / _p_nist_bar)
        except Exception:
            continue

    crit_iso_df = pd.DataFrame(
        [
            dict(
                quantity="T_c / K (fit target)",
                NIST=TC_NIST,
                fit=_tc_fit,
                pct_dev=100 * abs(_tc_fit - TC_NIST) / TC_NIST,
            ),
            dict(
                quantity="p_c / MPa (fit target)",
                NIST=PC_NIST,
                fit=_pc_fit,
                pct_dev=100 * abs(_pc_fit - PC_NIST) / PC_NIST,
            ),
            dict(
                quantity="rho_c / (kg/m^3) (not fit, informational)",
                NIST=RHOC_NIST,
                fit=_rhoc_fit,
                pct_dev=100 * abs(_rhoc_fit - RHOC_NIST) / RHOC_NIST,
            ),
            dict(
                quantity="isotherm pressure AAD (%)",
                NIST=0.0,
                fit=np.mean(_iso_errs) if _iso_errs else float("nan"),
                pct_dev=float("nan"),
            ),
        ]
    )
    mo.ui.table(crit_iso_df, label="Fitted SAFT-VR-CS: critical point and isotherm fit quality")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fitted SAFT-VR-CS vs. NIST, $uv$-CS, and the manually-tuned SAFT-VR-CS
    """)
    return


@app.cell
def _(
    PC_NIST,
    RHOC_NIST,
    State,
    TC_NIST,
    colors,
    df_cs,
    df_vrcs_fit,
    fh_choice,
    np,
    plt,
    si,
    vle,
    vrcs_fit_model,
):
    _fit_label = f"SAFT-VR-CS (fit, Aasen {fh_choice.value})"

    _fit_crit = State.critical_point_pure(vrcs_fit_model)[0]
    _tc_fit = _fit_crit.temperature / si.KELVIN
    _pc_fit = _fit_crit.pressure() * 1e-6 / si.PASCAL
    _rhoc_fit = _fit_crit.mass_density() / (si.KILOGRAM / si.METER**3)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5.5))

    axes2[0].plot(
        1 / vle["Temperature (K)"],
        vle["Pressure (MPa)"] * (si.MEGA * si.PASCAL / si.BAR),
        "o", mec="k", mfc="None", label="NIST",
    )
    axes2[0].plot(1 / np.array(df_cs["temperature"]), np.array(df_cs["pressure"]) * (si.PASCAL / si.BAR), "-", color=colors[0], label="$uv$-CS")
    #axes2[0].plot(1 / np.array(df_vrcs["temperature"]), np.array(df_vrcs["pressure"]) * (si.PASCAL / si.BAR), ":", color=colors[2], label="SAFT-VR-CS (manual)")
    axes2[0].plot(1 / np.array(df_vrcs_fit["temperature"]), np.array(df_vrcs_fit["pressure"]) * (si.PASCAL / si.BAR), "-.", color=colors[3], label=_fit_label)
    axes2[0].plot(
        1 / TC_NIST, PC_NIST * (si.MEGA * si.PASCAL / si.BAR),
        "*", ms=14, mec="k", mfc="gold", zorder=5, label="NIST crit. point",
    )
    axes2[0].plot(
        1 / _tc_fit, _pc_fit * (si.MEGA * si.PASCAL / si.BAR),
        "*", ms=14, mec="k", mfc=colors[3], zorder=5, label=f"{_fit_label} crit. point",
    )
    axes2[0].set_xlabel("$1/T$ / K$^{-1}$")
    axes2[0].set_ylabel("$p$ / bar")
    axes2[0].set_yscale("log")
    axes2[0].set_xlim(0.18, 0.45)
    axes2[0].legend(fontsize=8, frameon=False)

    axes2[1].plot(vle["Density (l, kg/m3)"], vle["Temperature (K)"], "o", mec="k", mfc="None")
    axes2[1].plot(vle["Density (v, kg/m3)"], vle["Temperature (K)"], "o", mec="k", mfc="None")
    axes2[1].plot(df_cs["mass density liquid"], df_cs["temperature"], "-", color=colors[0], label="$uv$-CS")
    axes2[1].plot(df_cs["mass density vapor"], df_cs["temperature"], "-", color=colors[0])
    #axes2[1].plot(df_vrcs["mass density liquid"], df_vrcs["temperature"], ":", color=colors[2], label="SAFT-VR-CS (manual)")
    #axes2[1].plot(df_vrcs["mass density vapor"], df_vrcs["temperature"], ":", color=colors[2])
    axes2[1].plot(df_vrcs_fit["mass density liquid"], df_vrcs_fit["temperature"], "-.", color=colors[3], label=_fit_label)
    axes2[1].plot(df_vrcs_fit["mass density vapor"], df_vrcs_fit["temperature"], "-.", color=colors[3])
    axes2[1].plot(
        RHOC_NIST, TC_NIST,
        "*", ms=14, mec="k", mfc="gold", zorder=5, label="NIST crit. point",
    )
    #axes2[1].plot(
    #    _rhoc_fit, _tc_fit,
    #    "*", ms=14, mec="k", mfc=colors[3], zorder=5, label=f"{_fit_label} crit. point",
    #)
    axes2[1].set_xlabel(r"$\rho$ / (kg/m³)")
    axes2[1].set_ylabel("$T$ / K")
    axes2[1].set_ylim(2.0, 5.5)
    axes2[1].legend(loc="best", frameon=False)

    plt.tight_layout()
    fig2
    return


@app.cell
def _(df_cs, df_vrcs, df_vrcs_fit, df_vrq, fh_choice, mo, np, pd, vle):
    def _aad2(model_df, model_col, nist_col, unit_factor=1.0):
        order = np.argsort(model_df["temperature"])
        t_model = np.array(model_df["temperature"])[order]
        y_model = np.array(model_df[model_col])[order] * unit_factor

        t_nist = vle["Temperature (K)"].to_numpy()
        y_nist = vle[nist_col].to_numpy()

        mask = (t_nist >= t_model.min()) & (t_nist <= t_model.max())
        y_interp = np.interp(t_nist[mask], t_model, y_model)
        return 100 * np.mean(np.abs((y_interp - y_nist[mask]) / y_nist[mask]))

    _rows2 = []
    for _label2, _df2 in [
        ("SAFT-VRQ-Mie", df_vrq),
        ("uv-CS-theory", df_cs),
        ("SAFT-VR-CS (manual)", df_vrcs),
        (f"SAFT-VR-CS (fit, Aasen {fh_choice.value})", df_vrcs_fit),
    ]:
        _rows2.append(
            dict(
                model=_label2,
                AAD_psat=_aad2(_df2, "pressure", "Pressure (MPa)", unit_factor=1e-6),
                AAD_rho_liq=_aad2(_df2, "mass density liquid", "Density (l, kg/m3)"),
                AAD_rho_vap=_aad2(_df2, "mass density vapor", "Density (v, kg/m3)"),
            )
        )

    dev_df2 = pd.DataFrame(_rows2)
    mo.ui.table(dev_df2, label="%AAD vs NIST, including the fitted SAFT-VR-CS")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Notes / next steps

    - The manually-tuned SAFT-VR-CS section above starts from the
      $uv$-CS-optimized values reused verbatim (as in the original example
      notebook); the fit section fixes classical Aasen FH=0 parameters and
      fits only the FH-correction coefficients — compare the two %AAD
      tables to see which strategy generalizes better.
    - Consider extending to a classical (non-quantum) fluid for contrast, and/or
      adding PC-SAFT / SAFT-VR-Mie as classical baselines.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SAFT-VR-CS for Hydrogen: evaluation against SAFT-VRQ-Mie and $uv$-CS-theory

    Normal hydrogen (3:1 ortho:para) is a second quantum-fluid stress test:
    lighter than helium in absolute mass, but with a critical temperature
    (~33 K) far enough above the NIST table's low-T cutoff that
    Feynman-Hibbs corrections matter less at the extremes than for helium's
    2 K isotherms — still, they're far from negligible for the VLE curve.

    This section repeats the helium comparison above for hydrogen:

    - **SAFT-VRQ-Mie** (Aasen et al.), literature FH=1 parameters.
    - **$uv$-CS-theory**, parameters optimized to hydrogen VLE.
    - **SAFT-VR-CS** (this implementation), same Mie-potential parameters,
      with sliders below, plus a fit of the FH-correction coefficients.

    Two differences from the helium workflow, driven by what data/parameters
    are available:

    - NIST only provides a coarse (0.5 K spacing) VLE table for hydrogen
      here, and no pressure-isotherm table, so the fit below only has the
      VLE + critical-point deviation groups (no isotherm group).
    - Only a first-order Feynman-Hibbs (FH=1) literature base is on hand for
      hydrogen (`feos/parameters/saftvrqmie/aasen2019.json`); there's no
      classical FH=0 base wired up, so there's no FH0/FH1 selector like the
      helium section has.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reference data (NIST)
    """)
    return


@app.cell
def _(np, pd):
    MOLARWEIGHT_H2 = 2.0157309551872  # g/mol, matches feos/parameters/saftvrqmie/aasen2019.json
    TC_NIST_H2 = 33.145  # K, NIST/Leachman et al. (2009) critical temperature for normal hydrogen

    vle_h2 = pd.read_csv("data/hydrogen_data/nist_vle.txt", sep="\t")

    # This table only goes up to 33.000 K (0.5 K spacing, vs. helium's 0.02 K
    # fine grid), 0.145 K short of Tc. Same rectilinear-diameter / linear-psat
    # extrapolation as helium, but over the last 8 rows (3.5 K) since that's
    # the closest data we have to the critical point — noisier than helium's
    # extrapolation given the coarser spacing.
    _diam_rows = vle_h2.tail(8)
    _rho_diam = (_diam_rows["Density (l, kg/m3)"] + _diam_rows["Density (v, kg/m3)"]) / 2
    _diam_fit = np.polyfit(_diam_rows["Temperature (K)"], _rho_diam, 1)
    RHOC_NIST_H2 = np.polyval(_diam_fit, TC_NIST_H2)  # kg/m^3

    _p_fit = np.polyfit(_diam_rows["Temperature (K)"], _diam_rows["Pressure (MPa)"], 1)
    PC_NIST_H2 = np.polyval(_p_fit, TC_NIST_H2)  # MPa
    return MOLARWEIGHT_H2, PC_NIST_H2, RHOC_NIST_H2, TC_NIST_H2, vle_h2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Models

    ### SAFT-VRQ-Mie (fixed, literature parameters)
    """)
    return


@app.cell
def _(EquationOfState, MOLARWEIGHT_H2, Parameters, PhaseDiagram, feos, si):
    vrqmie_params_h2 = dict(m=1.0, sigma=3.0243, epsilon_k=26.706, lr=9, la=6, fh=1)
    vrq_pr_h2 = feos.PureRecord(
        identifier=feos.Identifier("hydrogen"),
        molarweight=MOLARWEIGHT_H2,
        **vrqmie_params_h2,
    )
    vrq_model_h2 = EquationOfState.saftvrqmie(Parameters.new_pure(vrq_pr_h2))
    df_vrq_h2 = PhaseDiagram.pure(
        vrq_model_h2, min_temperature=14 * si.KELVIN, npoints=250
    ).to_dict(feos.Contributions.Residual)
    return df_vrq_h2, vrq_model_h2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### $uv$-CS-theory (optimized parameters from Thijs)
    """)
    return


@app.cell
def _(EquationOfState, MOLARWEIGHT_H2, Parameters, PhaseDiagram, feos, si):
    uvcs_params_h2 = dict(
        sigma=3.0156,
        epsilon_k=23.8730,
        rep=8.101,
        att=6.0,
        c_sigma=[1.43832767e00, 6.45545710e-04, 1.50887940e00],
        c_epsilon_k=[1.0, 1.0, 1.0],
        c_rep=[2.76585117e-01, 1.78562579e01, 1.82868514e02, 1.26035205e-04, 2.88248756e01],
    )
    uvcs_pr_h2 = feos.PureRecord(
        identifier=feos.Identifier("hydrogen"),
        molarweight=MOLARWEIGHT_H2,
        **uvcs_params_h2,
    )
    cs_model_h2 = EquationOfState.uvcstheory(Parameters.new_pure(uvcs_pr_h2))
    df_cs_h2 = PhaseDiagram.pure(
        cs_model_h2, min_temperature=14 * si.KELVIN, npoints=250
    ).to_dict(feos.Contributions.Residual)
    return cs_model_h2, df_cs_h2, uvcs_params_h2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SAFT-VR-CS (this implementation, interactive)

    Starts from the same Feynman-Hibbs-corrected Mie parameters as
    $uv$-CS-theory above. Tune them and re-submit to see how the chain
    formalism responds relative to the non-chain $uv$-CS reference.
    """)
    return


@app.cell
def _(mo, uvcs_params_h2):
    vrcs_form_h2 = mo.ui.array(
        [
            mo.ui.number(0.8, 1.5, 0.01, value=1.0, label="m (chain length)"),
            mo.ui.number(2.0, 3.5, 0.0001, value=uvcs_params_h2["sigma"], label="σ / Å"),
            mo.ui.number(10.0, 40.0, 0.0001, value=uvcs_params_h2["epsilon_k"], label="ε/k / K"),
            mo.ui.number(6.0, 14.0, 0.0001, value=uvcs_params_h2["rep"], label="λr"),
            mo.ui.number(5.0, 7.0, 0.1, value=uvcs_params_h2["att"], label="λa"),
        ]
    ).form(label="chain + Mie parameters")

    vrcs_c_sigma_form_h2 = mo.ui.array(
        [mo.ui.number(-5.0, 25.0, 0.0001, value=v) for v in uvcs_params_h2["c_sigma"]]
    ).form(label="c_sigma")

    vrcs_c_epsilon_form_h2 = mo.ui.array(
        [mo.ui.number(-5.0, 5.0, 0.0001, value=v) for v in uvcs_params_h2["c_epsilon_k"]]
    ).form(label="c_epsilon_k")

    vrcs_c_lr_form_h2 = mo.ui.array(
        [mo.ui.number(-5.0, 200.0, 0.0001, value=v) for v in uvcs_params_h2["c_rep"]]
    ).form(label="c_lr")

    mo.hstack(
        [vrcs_form_h2, vrcs_c_sigma_form_h2, vrcs_c_epsilon_form_h2, vrcs_c_lr_form_h2],
        justify="start",
        gap=2,
    )
    return vrcs_c_epsilon_form_h2, vrcs_c_lr_form_h2, vrcs_c_sigma_form_h2, vrcs_form_h2


@app.cell
def _(
    EquationOfState,
    MOLARWEIGHT_H2,
    Parameters,
    PhaseDiagram,
    feos,
    mo,
    si,
    uvcs_params_h2,
    vrcs_c_epsilon_form_h2,
    vrcs_c_lr_form_h2,
    vrcs_c_sigma_form_h2,
    vrcs_form_h2,
):
    _m, _sigma, _epsilon_k, _lr, _la = vrcs_form_h2.value or (
        1.0,
        uvcs_params_h2["sigma"],
        uvcs_params_h2["epsilon_k"],
        uvcs_params_h2["rep"],
        uvcs_params_h2["att"],
    )

    vrcs_params_h2 = dict(
        m=_m,
        sigma=_sigma,
        epsilon_k=_epsilon_k,
        lr=_lr,
        la=_la,
        c_sigma=list(vrcs_c_sigma_form_h2.value or uvcs_params_h2["c_sigma"]),
        c_epsilon_k=list(vrcs_c_epsilon_form_h2.value or uvcs_params_h2["c_epsilon_k"]),
        c_lr=list(vrcs_c_lr_form_h2.value or uvcs_params_h2["c_rep"]),
    )
    vrcs_pr_h2 = feos.PureRecord(
        identifier=feos.Identifier("hydrogen"),
        molarweight=MOLARWEIGHT_H2,
        **vrcs_params_h2,
    )
    vrcs_model_h2 = EquationOfState.saftvrcs_mie(Parameters.new_pure(vrcs_pr_h2))

    try:
        df_vrcs_h2 = PhaseDiagram.pure(
            vrcs_model_h2, min_temperature=14 * si.KELVIN, npoints=250
        ).to_dict(feos.Contributions.Residual)
        vrcs_error_h2 = None
    except Exception as exc:  # phase diagram solver can fail for bad params
        df_vrcs_h2 = None
        vrcs_error_h2 = str(exc)

    mo.stop(
        vrcs_error_h2 is not None,
        mo.md(f"**SAFT-VR-CS phase diagram failed for these parameters:** {vrcs_error_h2}"),
    )
    return df_vrcs_h2, vrcs_model_h2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Critical point comparison
    """)
    return


@app.cell
def _(
    PC_NIST_H2,
    RHOC_NIST_H2,
    State,
    TC_NIST_H2,
    cs_model_h2,
    mo,
    pd,
    si,
    vrcs_model_h2,
    vrq_model_h2,
):
    def _crit_row(label, model):
        state = State.critical_point_pure(model)[0]
        return dict(
            model=label,
            T_c=state.temperature / si.KELVIN,
            p_c=state.pressure() * 1e-6 / si.PASCAL,
            rho_c=state.mass_density() / (si.KILOGRAM / si.METER**3),
        )

    # p_c / rho_c aren't tabulated by NIST directly; PC_NIST_H2 / RHOC_NIST_H2
    # are the near-critical extrapolations computed in the data-loading cell.
    crit_rows_h2 = [dict(model="NIST", T_c=TC_NIST_H2, p_c=PC_NIST_H2, rho_c=RHOC_NIST_H2)]
    for _label, _model in [
        ("SAFT-VRQ-Mie", vrq_model_h2),
        ("uv-CS-theory", cs_model_h2),
        ("SAFT-VR-CS", vrcs_model_h2),
    ]:
        crit_rows_h2.append(_crit_row(_label, _model))

    crit_df_h2 = pd.DataFrame(crit_rows_h2)
    crit_table_h2 = mo.ui.table(crit_df_h2, label="Critical point: T_c / K, p_c / MPa, rho_c / (kg/m^3)")
    crit_table_h2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## VLE envelope and vapor pressure

    No NIST pressure-isotherm table is bundled for hydrogen (unlike
    helium), so there's no third isotherm panel here.
    """)
    return


@app.cell
def _(
    PC_NIST_H2,
    RHOC_NIST_H2,
    TC_NIST_H2,
    colors,
    cs_model_h2,
    df_cs_h2,
    df_vrcs_h2,
    df_vrq_h2,
    np,
    plt,
    si,
    vle_h2,
):
    fig_h2, axes_h2 = plt.subplots(1, 2, figsize=(12, 3.5))

    # --- Subplot 1: vapor pressure vs 1/T ---
    axes_h2[0].plot(
        1 / vle_h2["Temperature (K)"],
        vle_h2["Pressure (MPa)"] * (si.MEGA * si.PASCAL / si.BAR),
        "o",
        mec="k",
        mfc="None",
        label="NIST",
    )
    axes_h2[0].plot(1 / np.array(df_vrq_h2["temperature"]), np.array(df_vrq_h2["pressure"]) * (si.PASCAL / si.BAR), "--", color=colors[1], label="SAFT-VRQ-Mie")
    axes_h2[0].plot(1 / np.array(df_cs_h2["temperature"]), np.array(df_cs_h2["pressure"]) * (si.PASCAL / si.BAR), "-", color=colors[0], label="$uv$-CS")
    axes_h2[0].plot(1 / np.array(df_vrcs_h2["temperature"]), np.array(df_vrcs_h2["pressure"]) * (si.PASCAL / si.BAR), ":", color=colors[2], label="SAFT-VR-CS")
    axes_h2[0].plot(
        1 / TC_NIST_H2, PC_NIST_H2 * (si.MEGA * si.PASCAL / si.BAR),
        "*", ms=14, mec="k", mfc="gold", zorder=5, label="NIST crit. point",
    )
    axes_h2[0].set_xlabel("$1/T$ / K$^{-1}$")
    axes_h2[0].set_ylabel("$p$ / bar")
    axes_h2[0].set_yscale("log")
    axes_h2[0].legend(fontsize=8, frameon=False)

    # --- Subplot 2: VLE envelope, density vs temperature ---
    axes_h2[1].plot(vle_h2["Density (l, kg/m3)"], vle_h2["Temperature (K)"], "o", mec="k", mfc="None")
    axes_h2[1].plot(vle_h2["Density (v, kg/m3)"], vle_h2["Temperature (K)"], "o", mec="k", mfc="None")
    axes_h2[1].plot(df_vrq_h2["mass density liquid"], df_vrq_h2["temperature"], "--", color=colors[1], label="SAFT-VRQ-Mie")
    axes_h2[1].plot(df_vrq_h2["mass density vapor"], df_vrq_h2["temperature"], "--", color=colors[1])
    axes_h2[1].plot(df_cs_h2["mass density liquid"], df_cs_h2["temperature"], "-", color=colors[0], label="$uv$-CS")
    axes_h2[1].plot(df_cs_h2["mass density vapor"], df_cs_h2["temperature"], "-", color=colors[0])
    axes_h2[1].plot(df_vrcs_h2["mass density liquid"], df_vrcs_h2["temperature"], ":", color=colors[2], label="SAFT-VR-CS")
    axes_h2[1].plot(df_vrcs_h2["mass density vapor"], df_vrcs_h2["temperature"], ":", color=colors[2])
    axes_h2[1].plot(
        RHOC_NIST_H2, TC_NIST_H2,
        "*", ms=14, mec="k", mfc="gold", zorder=5, label="NIST crit. point",
    )
    axes_h2[1].set_xlabel(r"$\rho$ / (kg/m³)")
    axes_h2[1].set_ylabel("$T$ / K")
    axes_h2[1].legend(loc="best", frameon=False)

    plt.tight_layout()
    fig_h2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Deviations from NIST along the VLE curve

    %AAD computed by interpolating each model's VLE curve onto the NIST
    temperature grid shown above.
    """)
    return


@app.cell
def _(df_cs_h2, df_vrcs_h2, df_vrq_h2, mo, np, pd, vle_h2):
    def _aad(model_df, model_col, nist_col, unit_factor=1.0):
        order = np.argsort(model_df["temperature"])
        t_model = np.array(model_df["temperature"])[order]
        y_model = np.array(model_df[model_col])[order] * unit_factor

        t_nist = vle_h2["Temperature (K)"].to_numpy()
        y_nist = vle_h2[nist_col].to_numpy()

        mask = (t_nist >= t_model.min()) & (t_nist <= t_model.max())
        y_interp = np.interp(t_nist[mask], t_model, y_model)
        return 100 * np.mean(np.abs((y_interp - y_nist[mask]) / y_nist[mask]))

    _rows = []
    for _label, _df in [
        ("SAFT-VRQ-Mie", df_vrq_h2),
        ("uv-CS-theory", df_cs_h2),
        ("SAFT-VR-CS", df_vrcs_h2),
    ]:
        _rows.append(
            dict(
                model=_label,
                AAD_psat=_aad(_df, "pressure", "Pressure (MPa)", unit_factor=1e-6),
                AAD_rho_liq=_aad(_df, "mass density liquid", "Density (l, kg/m3)"),
                AAD_rho_vap=_aad(_df, "mass density vapor", "Density (v, kg/m3)"),
            )
        )

    dev_df_h2 = pd.DataFrame(_rows)
    mo.ui.table(dev_df_h2, label="%AAD vs NIST over the VLE range shown above")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit SAFT-VR-CS FH-correction coefficients from an Aasen FH1 base

    Same idea as the helium fit above: fix $m$, $\sigma$, $\varepsilon/k_B$,
    $\lambda_r$, $\lambda_a$ at the **Aasen et al. FH=1 literature Mie
    parameters for hydrogen** (`feos/parameters/saftvrqmie/aasen2019.json`)
    and fit only the 11 Feynman-Hibbs correction coefficients (`c_sigma`,
    `c_epsilon_k`, `c_lr`) with `scipy.optimize.least_squares`.

    Two differences from the helium fit, both because less data/literature
    is available here:

    - There's no FH0 literature base on hand for hydrogen, so unlike helium
      there's no FH0/FH1 selector — this always starts from the FH1 base.
    - There's no NIST pressure-isotherm table for hydrogen, so the
      objective only has two deviation groups (saturation pressure / liquid
      density / vapor density along VLE, and the critical point
      $T_c$/$p_c$), not three. Each group is still weighted equally via
      $1/\sqrt{n_\mathrm{group}}$ scaling, so the 2-point critical-point
      group isn't drowned out by the ~40-point VLE group.

    Initial guess: all 11 coefficients at 1, matching the FH1 convention
    used for helium's FH1 base.
    """)
    return


@app.cell
def _(np):
    AASEN_H2_FH1 = dict(m=1.0, sigma=3.0243, epsilon_k=26.706, lr=9.0, la=6.0)
    fit_x0_h2 = np.ones(11)
    return AASEN_H2_FH1, fit_x0_h2


@app.cell
def _(mo):
    fit_button_h2 = mo.ui.run_button(label="Run SAFT-VR-CS c-parameter fit for hydrogen (~30-60s)")
    fit_button_h2
    return (fit_button_h2,)


@app.cell
def _(
    AASEN_H2_FH1,
    EquationOfState,
    MOLARWEIGHT_H2,
    PC_NIST_H2,
    Parameters,
    PhaseDiagram,
    State,
    TC_NIST_H2,
    feos,
    fit_button_h2,
    fit_x0_h2,
    mo,
    np,
    si,
    vle_h2,
):
    mo.stop(not fit_button_h2.value, mo.md("*Click the button above to run the fit.*"))

    from scipy.optimize import least_squares as _least_squares

    _t_nist = vle_h2["Temperature (K)"].to_numpy()
    _psat_nist = vle_h2["Pressure (MPa)"].to_numpy()
    _rho_l_nist = vle_h2["Density (l, kg/m3)"].to_numpy()
    _rho_v_nist = vle_h2["Density (v, kg/m3)"].to_numpy()

    # Two deviation groups (no isotherm table for hydrogen), weighted
    # *equally* regardless of how many points each one has — see the
    # helium fit cell above for why.
    _n_vle = len(_t_nist)
    _n_crit = 2
    _n_residuals = 3 * _n_vle + _n_crit

    def _build_model(c_vec):
        params = dict(
            AASEN_H2_FH1,
            c_sigma=list(c_vec[0:3]),
            c_epsilon_k=list(c_vec[3:6]),
            c_lr=list(c_vec[6:11]),
        )
        pr = feos.PureRecord(
            identifier=feos.Identifier("hydrogen"),
            molarweight=MOLARWEIGHT_H2,
            **params,
        )
        return EquationOfState.saftvrcs_mie(Parameters.new_pure(pr))

    def _residuals(c_vec, npoints=25):
        try:
            model = _build_model(c_vec)
            df = PhaseDiagram.pure(
                model, min_temperature=14 * si.KELVIN, npoints=npoints
            ).to_dict(feos.Contributions.Residual)
        except Exception:
            return np.full(_n_residuals, 1.0)

        order = np.argsort(df["temperature"])
        t_model = np.array(df["temperature"])[order]
        p_model = np.array(df["pressure"])[order] * 1e-6
        rl_model = np.array(df["mass density liquid"])[order]
        rv_model = np.array(df["mass density vapor"])[order]

        p_i = np.interp(_t_nist, t_model, p_model)
        rl_i = np.interp(_t_nist, t_model, rl_model)
        rv_i = np.interp(_t_nist, t_model, rv_model)

        res_p = (p_i - _psat_nist) / _psat_nist / np.sqrt(_n_vle)
        res_rl = (rl_i - _rho_l_nist) / _rho_l_nist / np.sqrt(_n_vle)
        res_rv = (rv_i - _rho_v_nist) / _rho_v_nist / np.sqrt(_n_vle)

        try:
            _crit_state = State.critical_point_pure(model)[0]
            _tc_model = _crit_state.temperature / si.KELVIN
            _pc_model = _crit_state.pressure() * 1e-6 / si.PASCAL
            res_crit = (
                np.array(
                    [
                        (_tc_model - TC_NIST_H2) / TC_NIST_H2,
                        (_pc_model - PC_NIST_H2) / PC_NIST_H2,
                    ]
                )
                / np.sqrt(_n_crit)
            )
        except Exception:
            res_crit = np.full(_n_crit, 1.0) / np.sqrt(_n_crit)

        result = np.concatenate([res_p, res_rl, res_rv, res_crit])
        return np.where(np.isfinite(result), result, 1.0)

    with mo.status.spinner(title="Fitting SAFT-VR-CS c-parameters for hydrogen..."):
        fit_result_h2 = _least_squares(
            _residuals, fit_x0_h2, bounds=(-8, 8), max_nfev=300,
            xtol=1e-10, ftol=1e-10, gtol=1e-10, diff_step=1e-3,
        )

    fit_c_sigma_h2 = list(fit_result_h2.x[0:3])
    fit_c_epsilon_k_h2 = list(fit_result_h2.x[3:6])
    fit_c_lr_h2 = list(fit_result_h2.x[6:11])

    vrcs_fit_params_h2 = dict(
        AASEN_H2_FH1,
        c_sigma=fit_c_sigma_h2,
        c_epsilon_k=fit_c_epsilon_k_h2,
        c_lr=fit_c_lr_h2,
    )
    vrcs_fit_pr_h2 = feos.PureRecord(
        identifier=feos.Identifier("hydrogen"),
        molarweight=MOLARWEIGHT_H2,
        **vrcs_fit_params_h2,
    )
    vrcs_fit_model_h2 = EquationOfState.saftvrcs_mie(Parameters.new_pure(vrcs_fit_pr_h2))
    df_vrcs_fit_h2 = PhaseDiagram.pure(
        vrcs_fit_model_h2, min_temperature=14 * si.KELVIN, npoints=250
    ).to_dict(feos.Contributions.Residual)
    return (
        df_vrcs_fit_h2,
        fit_c_epsilon_k_h2,
        fit_c_lr_h2,
        fit_c_sigma_h2,
        fit_result_h2,
        vrcs_fit_model_h2,
    )


@app.cell
def _(fit_c_epsilon_k_h2, fit_c_lr_h2, fit_c_sigma_h2, fit_result_h2, mo, pd):
    fit_summary_h2 = pd.DataFrame(
        [
            dict(coefficient="c_sigma", values=[round(v, 4) for v in fit_c_sigma_h2]),
            dict(coefficient="c_epsilon_k", values=[round(v, 4) for v in fit_c_epsilon_k_h2]),
            dict(coefficient="c_lr", values=[round(v, 4) for v in fit_c_lr_h2]),
        ]
    )
    mo.vstack(
        [
            mo.ui.table(fit_summary_h2, label="Fitted coefficients"),
            mo.md(
                f"`success={fit_result_h2.success}`, `cost={fit_result_h2.cost:.4f}`, "
                f"`nfev={fit_result_h2.nfev}`, `status={fit_result_h2.status}`"
            ),
        ]
    )
    return


@app.cell
def _(
    PC_NIST_H2,
    RHOC_NIST_H2,
    State,
    TC_NIST_H2,
    mo,
    pd,
    si,
    vrcs_fit_model_h2,
):
    _crit_state = State.critical_point_pure(vrcs_fit_model_h2)[0]
    _tc_fit = _crit_state.temperature / si.KELVIN
    _pc_fit = _crit_state.pressure() * 1e-6 / si.PASCAL
    _rhoc_fit = _crit_state.mass_density() / (si.KILOGRAM / si.METER**3)

    crit_fit_df_h2 = pd.DataFrame(
        [
            dict(
                quantity="T_c / K (fit target)",
                NIST=TC_NIST_H2,
                fit=_tc_fit,
                pct_dev=100 * abs(_tc_fit - TC_NIST_H2) / TC_NIST_H2,
            ),
            dict(
                quantity="p_c / MPa (fit target)",
                NIST=PC_NIST_H2,
                fit=_pc_fit,
                pct_dev=100 * abs(_pc_fit - PC_NIST_H2) / PC_NIST_H2,
            ),
            dict(
                quantity="rho_c / (kg/m^3) (not fit, informational)",
                NIST=RHOC_NIST_H2,
                fit=_rhoc_fit,
                pct_dev=100 * abs(_rhoc_fit - RHOC_NIST_H2) / RHOC_NIST_H2,
            ),
        ]
    )
    mo.ui.table(crit_fit_df_h2, label="Fitted SAFT-VR-CS: critical point fit quality")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fitted SAFT-VR-CS vs. NIST, $uv$-CS, and the manually-tuned SAFT-VR-CS
    """)
    return


@app.cell
def _(
    PC_NIST_H2,
    RHOC_NIST_H2,
    State,
    TC_NIST_H2,
    colors,
    df_cs_h2,
    df_vrcs_fit_h2,
    np,
    plt,
    si,
    vle_h2,
    vrcs_fit_model_h2,
):
    _fit_crit = State.critical_point_pure(vrcs_fit_model_h2)[0]
    _tc_fit = _fit_crit.temperature / si.KELVIN
    _pc_fit = _fit_crit.pressure() * 1e-6 / si.PASCAL

    fig2_h2, axes2_h2 = plt.subplots(1, 2, figsize=(12, 5.5))

    axes2_h2[0].plot(
        1 / vle_h2["Temperature (K)"],
        vle_h2["Pressure (MPa)"] * (si.MEGA * si.PASCAL / si.BAR),
        "o", mec="k", mfc="None", label="NIST",
    )
    axes2_h2[0].plot(1 / np.array(df_cs_h2["temperature"]), np.array(df_cs_h2["pressure"]) * (si.PASCAL / si.BAR), "-", color=colors[0], label="$uv$-CS")
    axes2_h2[0].plot(1 / np.array(df_vrcs_fit_h2["temperature"]), np.array(df_vrcs_fit_h2["pressure"]) * (si.PASCAL / si.BAR), "-.", color=colors[3], label="SAFT-VR-CS (fit, Aasen FH1)")
    axes2_h2[0].plot(
        1 / TC_NIST_H2, PC_NIST_H2 * (si.MEGA * si.PASCAL / si.BAR),
        "*", ms=14, mec="k", mfc="gold", zorder=5, label="NIST crit. point",
    )
    axes2_h2[0].plot(
        1 / _tc_fit, _pc_fit * (si.MEGA * si.PASCAL / si.BAR),
        "*", ms=14, mec="k", mfc=colors[3], zorder=5, label="fit crit. point",
    )
    axes2_h2[0].set_xlabel("$1/T$ / K$^{-1}$")
    axes2_h2[0].set_ylabel("$p$ / bar")
    axes2_h2[0].set_yscale("log")
    axes2_h2[0].legend(fontsize=8, frameon=False)

    axes2_h2[1].plot(vle_h2["Density (l, kg/m3)"], vle_h2["Temperature (K)"], "o", mec="k", mfc="None")
    axes2_h2[1].plot(vle_h2["Density (v, kg/m3)"], vle_h2["Temperature (K)"], "o", mec="k", mfc="None")
    axes2_h2[1].plot(df_cs_h2["mass density liquid"], df_cs_h2["temperature"], "-", color=colors[0], label="$uv$-CS")
    axes2_h2[1].plot(df_cs_h2["mass density vapor"], df_cs_h2["temperature"], "-", color=colors[0])
    axes2_h2[1].plot(df_vrcs_fit_h2["mass density liquid"], df_vrcs_fit_h2["temperature"], "-.", color=colors[3], label="SAFT-VR-CS (fit, Aasen FH1)")
    axes2_h2[1].plot(df_vrcs_fit_h2["mass density vapor"], df_vrcs_fit_h2["temperature"], "-.", color=colors[3])
    axes2_h2[1].plot(
        RHOC_NIST_H2, TC_NIST_H2,
        "*", ms=14, mec="k", mfc="gold", zorder=5, label="NIST crit. point",
    )
    axes2_h2[1].set_xlabel(r"$\rho$ / (kg/m³)")
    axes2_h2[1].set_ylabel("$T$ / K")
    axes2_h2[1].legend(loc="best", frameon=False)

    plt.tight_layout()
    fig2_h2
    return


@app.cell
def _(df_cs_h2, df_vrcs_fit_h2, df_vrcs_h2, df_vrq_h2, mo, np, pd, vle_h2):
    def _aad2(model_df, model_col, nist_col, unit_factor=1.0):
        order = np.argsort(model_df["temperature"])
        t_model = np.array(model_df["temperature"])[order]
        y_model = np.array(model_df[model_col])[order] * unit_factor

        t_nist = vle_h2["Temperature (K)"].to_numpy()
        y_nist = vle_h2[nist_col].to_numpy()

        mask = (t_nist >= t_model.min()) & (t_nist <= t_model.max())
        y_interp = np.interp(t_nist[mask], t_model, y_model)
        return 100 * np.mean(np.abs((y_interp - y_nist[mask]) / y_nist[mask]))

    _rows2 = []
    for _label2, _df2 in [
        ("SAFT-VRQ-Mie", df_vrq_h2),
        ("uv-CS-theory", df_cs_h2),
        ("SAFT-VR-CS (manual)", df_vrcs_h2),
        ("SAFT-VR-CS (fit, Aasen FH1)", df_vrcs_fit_h2),
    ]:
        _rows2.append(
            dict(
                model=_label2,
                AAD_psat=_aad2(_df2, "pressure", "Pressure (MPa)", unit_factor=1e-6),
                AAD_rho_liq=_aad2(_df2, "mass density liquid", "Density (l, kg/m3)"),
                AAD_rho_vap=_aad2(_df2, "mass density vapor", "Density (v, kg/m3)"),
            )
        )

    dev_df2_h2 = pd.DataFrame(_rows2)
    mo.ui.table(dev_df2_h2, label="%AAD vs NIST, including the fitted SAFT-VR-CS")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Notes / next steps (hydrogen)

    - Unlike helium, there's no NIST pressure-isotherm table bundled here and
      no classical (FH0) Aasen literature base on hand, so the fit only uses
      the VLE + critical-point deviation groups and always starts from the
      FH1 base — adding either would strengthen the fit the same way it did
      for helium.
    - `PC_NIST_H2` / `RHOC_NIST_H2` are extrapolated from the last 8 rows of
      a coarse 0.5 K-spaced VLE table, 0.145 K short of $T_c$ — noisier than
      helium's fine-grained extrapolation, so treat them as approximate.
    """)
    return


if __name__ == "__main__":
    app.run()
