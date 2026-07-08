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
def _(pd):
    MOLARWEIGHT_HE = 4.002602  # g/mol
    TC_NIST = 5.1953  # K, NIST critical temperature for helium-4

    vle_full = pd.read_csv("data/helium_data/nist_vle.txt", sep="\t")
    vle = vle_full[vle_full["Temperature (K)"] < 0.9 * TC_NIST][11:]

    isotherms_full = pd.read_csv("data/helium_data/nist_isotherms.txt", sep="\t")
    isotherms = isotherms_full.loc[isotherms_full["Pressure (MPa)"] < 5]
    return MOLARWEIGHT_HE, TC_NIST, isotherms, vle


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
    ### $uv$-CS-theory (fixed, optimized parameters)
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
def _(State, TC_NIST, cs_model, mo, pd, si, vrcs_model, vrq_model):
    def _crit_row(label, model):
        state = State.critical_point_pure(model)[0]
        return dict(
            model=label,
            T_c=state.temperature / si.KELVIN,
            rho_c=state.mass_density() / (si.KILOGRAM / si.METER**3),
        )

    crit_rows = [dict(model="NIST", T_c=TC_NIST, rho_c=float("nan"))]
    for _label, _model in [
        ("SAFT-VRQ-Mie", vrq_model),
        ("uv-CS-theory", cs_model),
        ("SAFT-VR-CS", vrcs_model),
    ]:
        crit_rows.append(_crit_row(_label, _model))

    crit_df = pd.DataFrame(crit_rows)
    crit_table = mo.ui.table(crit_df, label="Critical point: T_c / K, rho_c / (kg/m^3)")
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
    `c_lr`) to the NIST VLE data above (vapor pressure + liquid/vapor
    density) using `scipy.optimize.least_squares`.

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
    fit_button = mo.ui.run_button(label="Run SAFT-VR-CS c-parameter fit (~30-60s)")
    fit_button
    return (fit_button,)


@app.cell
def _(
    EquationOfState,
    MOLARWEIGHT_HE,
    Parameters,
    PhaseDiagram,
    aasen_base,
    feos,
    fit_button,
    fit_x0,
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
            return np.full(3 * len(_t_nist), 1.0)

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

        res_p = (p_i - _psat_nist) / _psat_nist
        res_rl = (rl_i - _rho_l_nist) / _rho_l_nist
        res_rv = (rv_i - _rho_v_nist) / _rho_v_nist
        return np.concatenate([res_p, res_rl, res_rv])

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
        vrcs_fit_model, min_temperature=2 * si.KELVIN, npoints=250
    ).to_dict(feos.Contributions.Residual)
    return df_vrcs_fit, fit_c_epsilon_k, fit_c_lr, fit_c_sigma, fit_result


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fitted SAFT-VR-CS vs. NIST, $uv$-CS, and the manually-tuned SAFT-VR-CS
    """)
    return


@app.cell
def _(colors, df_cs, df_vrcs, df_vrcs_fit, fh_choice, np, plt, si, vle):
    _fit_label = f"SAFT-VR-CS (fit, Aasen {fh_choice.value})"

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5.5))

    axes2[0].plot(
        1 / vle["Temperature (K)"],
        vle["Pressure (MPa)"] * (si.MEGA * si.PASCAL / si.BAR),
        "o", mec="k", mfc="None", label="NIST",
    )
    axes2[0].plot(1 / np.array(df_cs["temperature"]), np.array(df_cs["pressure"]) * (si.PASCAL / si.BAR), "-", color=colors[0], label="$uv$-CS")
    axes2[0].plot(1 / np.array(df_vrcs["temperature"]), np.array(df_vrcs["pressure"]) * (si.PASCAL / si.BAR), ":", color=colors[2], label="SAFT-VR-CS (manual)")
    axes2[0].plot(1 / np.array(df_vrcs_fit["temperature"]), np.array(df_vrcs_fit["pressure"]) * (si.PASCAL / si.BAR), "-.", color=colors[3], label=_fit_label)
    axes2[0].set_xlabel("$1/T$ / K$^{-1}$")
    axes2[0].set_ylabel("$p$ / bar")
    axes2[0].set_yscale("log")
    axes2[0].set_xlim(0.18, 0.45)
    axes2[0].legend(fontsize=8, frameon=False)

    axes2[1].plot(vle["Density (l, kg/m3)"], vle["Temperature (K)"], "o", mec="k", mfc="None")
    axes2[1].plot(vle["Density (v, kg/m3)"], vle["Temperature (K)"], "o", mec="k", mfc="None")
    axes2[1].plot(df_cs["mass density liquid"], df_cs["temperature"], "-", color=colors[0], label="$uv$-CS")
    axes2[1].plot(df_cs["mass density vapor"], df_cs["temperature"], "-", color=colors[0])
    axes2[1].plot(df_vrcs["mass density liquid"], df_vrcs["temperature"], ":", color=colors[2], label="SAFT-VR-CS (manual)")
    axes2[1].plot(df_vrcs["mass density vapor"], df_vrcs["temperature"], ":", color=colors[2])
    axes2[1].plot(df_vrcs_fit["mass density liquid"], df_vrcs_fit["temperature"], "-.", color=colors[3], label=_fit_label)
    axes2[1].plot(df_vrcs_fit["mass density vapor"], df_vrcs_fit["temperature"], "-.", color=colors[3])
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


if __name__ == "__main__":
    app.run()
