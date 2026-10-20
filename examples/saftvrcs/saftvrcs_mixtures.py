import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SAFT-VR-CS: helium / hydrogen mixtures

    Test of the SAFT-VR-CS combining rules for the effective (corresponding-states) parameters.

    1. Pure helium and hydrogen with the Ziegler parameters (`parameters/saftvrcs/saft_vr_cs_params.json`) vs. NIST.
    2. He/H$_2$ mixture with two routes to the unlike parameters, vs. experimental data (DDB) and GEMC:
        - `effective`: combine the effective pure-component parameters (each at its own mass) with the SAFT-VR Mie rules.
        - `classical`: combine the classical parameters with the SAFT-VR Mie rules, then map the pair at the
          pair mass $m_{ij} = 2 m_i m_j / (m_i + m_j)$ with prefactors $\zeta_{ij} = (\zeta_i + \zeta_j)/2$.
    """)
    return


@app.cell
def _():
    import json

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import si_units as si
    import feos
    from feos import (
        EquationOfState,
        Parameters,
        PhaseDiagram,
        PhaseEquilibrium,
        State,
    )
    from scipy.optimize import minimize_scalar

    DATA = mo.notebook_dir() / "data"
    PARAMETERS = mo.notebook_dir().parents[1] / "parameters"
    PARAMETER_FILE = PARAMETERS / "saftvrcs" / "saft_vr_cs_params.json"
    VRQ_PATH = PARAMETERS / "saftvrqmie"
    return (
        DATA,
        EquationOfState,
        PARAMETER_FILE,
        Parameters,
        PhaseDiagram,
        PhaseEquilibrium,
        State,
        VRQ_PATH,
        feos,
        json,
        minimize_scalar,
        mo,
        np,
        pd,
        plt,
        si,
    )


@app.cell
def _(PARAMETER_FILE, feos, json, pd):
    MW = dict(Helium=4.002602, Hydrogen=2.0157309551872)  # g/mol, normal hydrogen
    COMPONENTS = list(MW)

    with open(PARAMETER_FILE) as _f:
        ziegler = {
            name: _p["parameters"]
            for name, _p in json.load(_f).items()
            if name in MW
        }


    def record(name):
        """SAFT-VR-CS pure record from the Ziegler parameters (Csigma k -> c_sigma[k], etc.)."""
        p = ziegler[name]
        return feos.PureRecord(
            identifier=feos.Identifier(name=name),
            molarweight=MW[name],
            m=1.0,
            sigma=p["sigma"],
            epsilon_k=p["epsilon"],
            lr=p["nu_eff"],
            la=6.0,
            c_sigma=[p[f"Csigma{k}"] for k in range(5)],
            c_epsilon_k=[p[f"Ceps{k}"] for k in range(5)],
            c_lr=[p[f"Crep{k}"] for k in range(5)],
        )


    pd.DataFrame(ziegler)
    return COMPONENTS, record


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effective parameters in SAFT-VR-CS

    Every component is mapped onto a classical Mie potential with temperature-dependent effective
    parameters $\sigma_\mathrm{eff}$, $\varepsilon_\mathrm{eff}$, $\lambda_{r,\mathrm{eff}}$; the attractive
    exponent stays $\lambda_a = 6$. The hard-sphere diameter (Barker-Henderson), the Mie prefactor $C$ and the
    dispersion terms of SAFT-VR Mie are evaluated with these effective parameters.

    ### Like pairs ($ii$)

    Reduced quantum parameter with the particle mass $m_i$ (segment number $= 1$):

    $$D^*_i = \frac{\hbar^2}{12\, m_i\, k_\mathrm{B} T\, \sigma_i^2}$$

    $$\frac{\sigma_{\mathrm{eff},i}}{\sigma_i} = F^\sigma\!\left(D^*_i, \lambda_{r,i}; \zeta^\sigma_i\right), \qquad
    \frac{\varepsilon_{\mathrm{eff},i}}{\varepsilon_i} = F^\varepsilon\!\left(D^*_i, \lambda_{r,i}; \zeta^\varepsilon_i\right), \qquad
    \frac{\lambda_{r,\mathrm{eff},i}}{\lambda_{r,i}} = F^\lambda\!\left(D^*_i, \lambda_{r,i}; \zeta^\lambda_i\right)$$

    with ($X = \sigma, \varepsilon$; coefficients $c^X_k(\lambda_r) = a^X_k + b^X_k \lambda_r + d^X_k \lambda_r^2$)

    $$F^X = \frac{1 + \zeta^X_0 c^X_0 D^* + \zeta^X_1 c^X_1 D^{*2} + \zeta^X_3 D^{*3}}{1 + \zeta^X_2 c^X_2 D^* + \zeta^X_4 D^{*2}},
    \qquad
    F^\lambda = \frac{1 + \zeta^\lambda_0 c^\lambda_0 D^* + \zeta^\lambda_1 c^\lambda_1 D^{*2} + \zeta^\lambda_2 c^\lambda_2 D^{*3}}{1 + \zeta^\lambda_3 c^\lambda_3 D^* + \zeta^\lambda_4 c^\lambda_4 D^{*2}}$$

    The $\zeta$ are the component-specific prefactors (`Csigma k`, `Ceps k`, `Crep k` in the parameter file).
    Defaults: $\zeta^{\sigma,\varepsilon} = (1, 1, 1, 0, 0)$, $\zeta^\lambda = (1, 1, 1, 1, 1)$.

    ### Unlike pairs ($ij$), route `effective`

    SAFT-VR Mie combining rules applied to the effective pure-component parameters (each at its own mass $m_i$):

    $$\sigma_{\mathrm{eff},ij} = (1 - l_{ij}) \frac{\sigma_{\mathrm{eff},i} + \sigma_{\mathrm{eff},j}}{2}, \qquad
    \varepsilon_{\mathrm{eff},ij} = (1 - k_{ij}) \frac{\sqrt{\sigma_{\mathrm{eff},i}^3 \sigma_{\mathrm{eff},j}^3}}{\sigma_{\mathrm{eff},ij}^3} \sqrt{\varepsilon_{\mathrm{eff},i}\, \varepsilon_{\mathrm{eff},j}}, \qquad
    \lambda_{r,\mathrm{eff},ij} = 3 + (1 - \gamma_{ij}) \sqrt{(\lambda_{r,\mathrm{eff},i} - 3)(\lambda_{r,\mathrm{eff},j} - 3)}$$

    ### Unlike pairs ($ij$), route `classical`

    Notation: $\nu \equiv \lambda_r$ (`nu_eff` in the parameter file).

    **Step 1: classical unlike parameters** (SAFT-VR Mie combining rules on the classical pure-component parameters)

    $$\sigma_{ij} = (1 - l_{ij}) \frac{\sigma_i + \sigma_j}{2}, \qquad
    \varepsilon_{ij} = (1 - k_{ij}) \frac{\sqrt{\sigma_i^3 \sigma_j^3}}{\sigma_{ij}^3} \sqrt{\varepsilon_i \varepsilon_j}, \qquad
    \nu_{ij} = 3 + (1 - \gamma_{ij}) \sqrt{(\nu_i - 3)(\nu_j - 3)}$$

    **Step 2: quantum parameter and prefactors of the pair**

    $$D_{ij} = \frac{\hbar^2}{12\, m_{ij}\, k_\mathrm{B} T} = \frac{D_i + D_j}{2}
    \quad \text{with} \quad m_{ij} = \frac{2 m_i m_j}{m_i + m_j}, \qquad
    D^*_{ij} = \frac{D_{ij}}{\sigma_{ij}^2}, \qquad
    \zeta^X_{ij} = \frac{\zeta^X_i + \zeta^X_j}{2} \quad (X = \sigma, \varepsilon, \nu;\ \text{element by element})$$

    **Step 3: effective unlike parameters** (the like-pair functions $F^X$, evaluated with the pair quantities)

    $$\sigma_{\mathrm{eff},ij} = \sigma_{ij}\, F^\sigma\!\left(D^*_{ij}, \nu_{ij}; \zeta^\sigma_{ij}\right), \qquad
    \varepsilon_{\mathrm{eff},ij} = \varepsilon_{ij}\, F^\varepsilon\!\left(D^*_{ij}, \nu_{ij}; \zeta^\varepsilon_{ij}\right), \qquad
    \nu_{\mathrm{eff},ij} = \nu_{ij}\, F^\nu\!\left(D^*_{ij}, \nu_{ij}; \zeta^\nu_{ij}\right)$$

    written out for $\sigma$:

    $$\frac{\sigma_{\mathrm{eff},ij}}{\sigma_{ij}} =
    \frac{1 + \zeta^\sigma_{ij,0}\, c^\sigma_0(\nu_{ij})\, D^*_{ij} + \zeta^\sigma_{ij,1}\, c^\sigma_1(\nu_{ij})\, D^{*2}_{ij} + \zeta^\sigma_{ij,3}\, D^{*3}_{ij}}
    {1 + \zeta^\sigma_{ij,2}\, c^\sigma_2(\nu_{ij})\, D^*_{ij} + \zeta^\sigma_{ij,4}\, D^{*2}_{ij}}$$

    What enters the functions $F^X$:

    - $D^*_{ij}$, i.e. $T$, the pair mass $m_{ij}$ and the classical $\sigma_{ij}$ (including $l_{ij}$);
    - the classical $\nu_{ij}$ (through the coefficients $c^X_k(\nu_{ij})$; $\gamma_{ij}$ therefore acts on all three effective parameters);
    - the mean prefactors $\zeta^X_{ij}$.

    $\varepsilon_{ij}$ does **not** enter $F^X$: the functions depend on $D^*$ and $\nu$ only. $\varepsilon_{ij}$ (and with it $k_{ij}$)
    only scales $\varepsilon_{\mathrm{eff},ij}$. The attractive exponent stays classical, $\lambda_{a,ij} = 6$.

    Binary parameters: $k_{ij}$ on $\varepsilon_{ij}$, $\gamma_{ij}$ on $\nu_{ij}$, $l_{ij}$ on $\sigma_{ij}$. As in SAFT-VRQ Mie,
    $\sigma_{ij}$ including $l_{ij}$ is used in the $\varepsilon_{ij}$ combining rule, so $l_{ij}$ also changes $\varepsilon_{ij}$
    (and $D^*_{ij}$ in the `classical` route).

    For $i = j$ both routes reduce to the like-pair equations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pure components vs. NIST
    """)
    return


@app.cell
def _(
    COMPONENTS,
    DATA,
    EquationOfState,
    Parameters,
    PhaseDiagram,
    State,
    VRQ_PATH,
    feos,
    pd,
    record,
    si,
):
    def read_nist(fname):
        d = pd.read_csv(fname, sep="\t")
        return pd.DataFrame(
            {
                "T": d["Temperature (K)"],
                "p": 10 * d["Pressure (MPa)"],  # bar
                "rho_l": d["Density (l, kg/m3)"],
                "rho_v": d["Density (v, kg/m3)"],
            }
        )


    def vrq_parameters(names, binary=False):
        """SAFT-VRQ Mie (FH1) parameters: Aasen et al. (2019) pure, Aasen et al. (2020) binary."""
        files = [str(VRQ_PATH / "aasen2019.json")]
        if binary:
            files.append(str(VRQ_PATH / "aasen2020_binary.json"))
        return Parameters.from_json(
            [n.lower() for n in names],
            *files,
            identifier_option=feos.IdentifierOption.Name,
        )


    nist = {
        "Helium": read_nist(DATA / "helium_data" / "nist_vle.txt"),
        "Hydrogen": read_nist(DATA / "hydrogen_data" / "nist_vle.txt"),
    }
    CS, VRQ = "SAFT-VR-CS (Ziegler)", "SAFT-VRQ Mie (Aasen 2019)"
    pure_models = {
        name: {
            CS: EquationOfState.saftvrcs_mie(Parameters.new_pure(record(name))),
            VRQ: EquationOfState.saftvrqmie(vrq_parameters([name])),
        }
        for name in COMPONENTS
    }
    pure_styles = {CS: dict(color="C0", ls="-"), VRQ: dict(color="k", ls="-.")}
    T_MIN_VRQ = 2.7  # K, SAFT-VRQ Mie does not converge for helium below ~2.7 K
    pure_diagrams = {
        (name, label): PhaseDiagram.pure(
            eos,
            max(nist[name]["T"].min(), T_MIN_VRQ if label == VRQ else 0.0)
            * si.KELVIN,
            200,
        )
        for name, _models in pure_models.items()
        for label, eos in _models.items()
    }
    critical_points = pd.DataFrame(
        {
            f"{name}, {label}": {
                "T_c / K": (_cp := State.critical_point(eos)).temperature
                / si.KELVIN,
                "p_c / bar": _cp.pressure() / si.BAR,
                "rho_c / (kg/m3)": _cp.mass_density()
                / (si.KILOGRAM / si.METER**3),
            }
            for name, _models in pure_models.items()
            for label, eos in _models.items()
        }
    )
    critical_points
    return nist, pure_diagrams, pure_models, pure_styles, vrq_parameters


@app.cell
def _(COMPONENTS, nist, plt, pure_diagrams, pure_models, pure_styles, si):
    _fig, _axes = plt.subplots(2, 2, figsize=(10, 8))
    _rho_unit = si.KILOGRAM / si.METER**3
    for (_ax_p, _ax_rho), name in zip(_axes, COMPONENTS):
        _ref = nist[name]
        for label in pure_models[name]:
            _dia = pure_diagrams[name, label]
            _T = _dia.vapor.temperature / si.KELVIN
            _ax_p.plot(
                _T, _dia.vapor.pressure / si.BAR, label=label, **pure_styles[label]
            )
            _ax_rho.plot(
                _dia.liquid.mass_density / _rho_unit,
                _T,
                label=label,
                **pure_styles[label]
            )
            _ax_rho.plot(
                _dia.vapor.mass_density / _rho_unit, _T, **pure_styles[label]
            )
        _ax_p.plot(
            _ref["T"], _ref["p"], "o", mfc="none", mec="k", ms=5, label="NIST"
        )
        _ax_rho.plot(
            _ref["rho_l"], _ref["T"], "o", mfc="none", mec="k", ms=5, label="NIST"
        )
        _ax_rho.plot(_ref["rho_v"], _ref["T"], "o", mfc="none", mec="k", ms=5)
        _ax_p.set(xlabel="$T$ / K", ylabel="$p_\\mathrm{sat}$ / bar", title=name)
        _ax_rho.set(xlabel="$\\rho$ / (kg/m$^3$)", ylabel="$T$ / K", title=name)
        _ax_p.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(PhaseEquilibrium, nist, pd, pure_models, si):
    def pure_deviations(eos, ref):
        """Relative deviations (%) at the NIST temperatures."""
        rows = []
        _rho_unit = si.KILOGRAM / si.METER**3
        for T, p, rho_l, rho_v in ref[["T", "p", "rho_l", "rho_v"]].itertuples(
            index=False
        ):
            try:
                vle = PhaseEquilibrium.pure(eos, T * si.KELVIN)
            except KeyboardInterrupt:
                raise
            except BaseException:  # also catches Rust panics (PanicException)
                continue
            rows.append(
                dict(
                    T=T,
                    p=100 * (vle.vapor.pressure() / si.BAR / p - 1),
                    rho_l=100
                    * (vle.liquid.mass_density() / _rho_unit / rho_l - 1),
                    rho_v=100 * (vle.vapor.mass_density() / _rho_unit / rho_v - 1),
                )
            )
        return pd.DataFrame(rows)


    pure_dev = {
        (name, label): pure_deviations(eos, nist[name])
        for name, _models in pure_models.items()
        for label, eos in _models.items()
    }
    pd.DataFrame(
        {
            f"{name}, {label}": {
                "AARD p_sat / %": d["p"].abs().mean(),
                "AARD rho_l / %": d["rho_l"].abs().mean(),
                "AARD rho_v / %": d["rho_v"].abs().mean(),
                "n converged": len(d),
                "n NIST": len(nist[name]),
            }
            for (name, label), d in pure_dev.items()
        }
    ).round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## He/H$_2$ mixture

    Binary parameters from the pure records above, $k_{ij} = \gamma_{ij} = 0$. Component 0 is helium.
    Experimental data: DDB export (`examples/uvcs/data/helium_hydrogen_ddb_vle.csv`, $T < 33$ K) and GEMC.
    """)
    return


@app.cell
def _(DATA, mo, pd):
    exp = pd.read_csv(
        mo.notebook_dir().parent / "uvcs" / "data" / "helium_hydrogen_ddb_vle.csv"
    ).rename(columns={"temperature_K": "T", "pressure_bar": "p"})
    exp = exp[exp["T"] < 33.0]
    gemc = pd.read_csv(DATA / "helium_hydrogen_gemc_vle.csv", sep=r"\s+").rename(
        columns={"temperature[K]": "T", "pressure[bar]": "p"}
    )
    references = sorted(exp["reference"].unique())
    markers = dict(zip(references, "sD^v<>ph"))
    exp.groupby("reference").agg(
        n=("p", "size"), T_min=("T", "min"), T_max=("T", "max"), p_max=("p", "max")
    )
    return exp, gemc, markers


@app.cell
def _(COMPONENTS, EquationOfState, Parameters, record, vrq_parameters):
    def saftvrcs_mixture(combining_rule, k_ij=0.0):
        """SAFT-VR-CS He/H2 (component 0: helium) with the Ziegler pure parameters."""
        _params = Parameters.new_binary(
            [record("Helium"), record("Hydrogen")], k_ij=k_ij
        )
        return EquationOfState.saftvrcs_mie(_params, combining_rule=combining_rule)


    # SAFT-VRQ Mie with the published binary parameters (k_ij = 0.08, l_ij = -0.05) and without (k_ij = l_ij = 0)
    vrq_mixture = EquationOfState.saftvrqmie(
        vrq_parameters(COMPONENTS, binary=True)
    )
    vrq_mixture_predictive = EquationOfState.saftvrqmie(vrq_parameters(COMPONENTS))
    return saftvrcs_mixture, vrq_mixture, vrq_mixture_predictive


@app.cell
def _(PhaseEquilibrium, np, pd, si):
    def _equilibrium(eos, T, z, p_exp, bubble):
        """Bubble (z = x_He) or dew (z = y_He) point; returns (p, other composition) or None."""
        for p_init in [
            None,
            p_exp * si.BAR,
        ]:  # feos' own initial guess first, then the experimental p
            try:
                if bubble:
                    vle = PhaseEquilibrium.bubble_point(
                        eos, T * si.KELVIN, np.array([z, 1 - z]), tp_init=p_init
                    )
                    w = vle.vapor.molefracs[0]
                else:
                    vle = PhaseEquilibrium.dew_point(
                        eos, T * si.KELVIN, np.array([z, 1 - z]), tp_init=p_init
                    )
                    w = vle.liquid.molefracs[0]
                if abs(w - z) > 1e-6:  # not the trivial solution
                    return vle.vapor.pressure() / si.BAR, w
            except KeyboardInterrupt:
                raise
            except BaseException:  # also catches Rust panics (PanicException)
                pass
        return None


    def mixture_deviations(eos, d):
        """Bubble points at the experimental (T, x_He); dew points at (T, y_He) where x_He is missing."""
        rows = []
        for T, p, x, y, ref in d[
            ["T", "p", "x_he", "y_he", "reference"]
        ].itertuples(index=False):
            bubble = not np.isnan(x)
            res = _equilibrium(eos, T, x if bubble else y, p, bubble)
            p_calc, w_calc = res if res is not None else (np.nan, np.nan)
            rows.append(
                dict(
                    reference=ref,
                    T=T,
                    p=p,
                    kind="bubble" if bubble else "dew",
                    p_calc=p_calc,
                    # deviation of the other phase composition (y_He for bubble, x_He for dew points)
                    dz=w_calc - (y if bubble else x),
                )
            )
        return pd.DataFrame(rows)


    def summarize(d):
        return pd.Series(
            {
                "AARD p / %": 100 * (d["p_calc"] / d["p"] - 1).abs().mean(),
                "AAD y or x": d["dz"].abs().mean(),
                "n": len(d),
            }
        )

    return mixture_deviations, summarize


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fit of $k_{ij}$ for the `classical` route

    Objective: AARD of the bubble pressures at the experimental $(T, x_\mathrm{He})$ (DDB, all references);
    points without a converged bubble point count as 100 %. A scan showed the minimum between
    $k_{ij} = 0.25$ and $0.3$; for $k_{ij} \gtrsim 0.33$ single points diverge.
    """)
    return


@app.cell
def _(exp, minimize_scalar, mixture_deviations, mo, saftvrcs_mixture):
    exp_bubble = exp[exp["x_he"].notna()]


    def kij_objective(k_ij):
        _d = mixture_deviations(saftvrcs_mixture("classical", k_ij), exp_bubble)
        return (_d["p_calc"] / _d["p"] - 1).abs().fillna(1.0).mean()


    with mo.persistent_cache("kij_fit_classical"):
        kij_fit_result = minimize_scalar(
            kij_objective,
            bounds=(0.2, 0.33),
            method="bounded",
            options=dict(xatol=1e-3),
        )
    k_ij_fit = float(kij_fit_result.x)
    mo.md(
        f"$k_{{ij}}$ = {k_ij_fit:.4f}, objective {100 * kij_fit_result.fun:.1f} % "
        f"({kij_fit_result.nfev} evaluations)"
    )
    return (k_ij_fit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Models in the comparison

    | model | binary parameters | |
    |---|---|---|
    | SAFT-VR-CS, `effective` | $k_{ij} = \gamma_{ij} = l_{ij} = 0$ | predictive |
    | SAFT-VR-CS, `classical` | $k_{ij} = \gamma_{ij} = l_{ij} = 0$ | predictive |
    | SAFT-VR-CS, `classical`, fitted $k_{ij}$ | $k_{ij}$ fitted above, $\gamma_{ij} = l_{ij} = 0$ | 1 binary parameter |
    | SAFT-VRQ Mie, $k_{ij} = l_{ij} = 0$ | none | predictive |
    | SAFT-VRQ Mie (Aasen 2020) | $k_{ij} = 0.08$, $l_{ij} = -0.05$ | 2 binary parameters |

    SAFT-VRQ Mie applies the Feynman-Hibbs correction directly to the unlike pair (reduced mass, combined
    classical $\sigma_{ij}$, $\varepsilon_{ij}$, $\lambda_{ij}$), which corresponds to the `classical` route of SAFT-VR-CS.
    Its $l_{ij}$ acts on $\sigma_{ij}$ (and through the combining rule on $\varepsilon_{ij}$); SAFT-VR-CS has the same $l_{ij}$ plus $\gamma_{ij}$ on $\lambda_{r,ij}$, which SAFT-VRQ Mie does not have.
    Only the predictive rows compare the models on equal terms; the fitted rows show what one or two binary
    parameters achieve.
    """)
    return


@app.cell
def _(k_ij_fit, saftvrcs_mixture, vrq_mixture, vrq_mixture_predictive):
    models_cs = {
        "SAFT-VR-CS, effective": saftvrcs_mixture("effective"),
        "SAFT-VR-CS, classical": saftvrcs_mixture("classical"),
        f"SAFT-VR-CS, classical, $k_{{ij}}$ = {k_ij_fit:.3f}": saftvrcs_mixture(
            "classical", k_ij_fit
        ),
    }
    models_vrq = {
        "SAFT-VRQ Mie, $k_{ij} = l_{ij} = 0$": vrq_mixture_predictive,
        "SAFT-VRQ Mie (Aasen 2020)": vrq_mixture,
    }
    models = models_cs | models_vrq
    styles = dict(
        zip(
            models,
            [
                dict(color="C0", ls="-"),
                dict(color="C3", ls="--"),
                dict(color="C2", ls="-"),
                dict(color="k", ls=":"),
                dict(color="k", ls="-."),
            ],
        )
    )
    return models, models_cs, models_vrq, styles


@app.cell
def _(PhaseEquilibrium, np, pd, si):
    def bubble_curve(eos, T, dx0=0.01, dx_min=1e-5, x_max=0.95):
        """Bubble-point curve from x_He = 1e-3 up to the mixture critical point (continuation in x)."""
        rows = []
        x, dx = 1e-3, 1e-3
        while x < x_max and dx >= dx_min:
            p_init = y_init = None
            if len(rows) >= 2:  # linear extrapolation of p and y in x
                (x1, y1, p1), (x2, y2, p2) = rows[-2], rows[-1]
                s = (x - x2) / (x2 - x1)
                p_init = (p2 + s * (p2 - p1)) * si.BAR
                yi = min(max(y2 + s * (y2 - y1), 1e-8), 1 - 1e-8)
                y_init = np.array([yi, 1 - yi])
            elif rows:
                p_init = rows[-1][2] * si.BAR
            try:
                vle = PhaseEquilibrium.bubble_point(
                    eos,
                    T * si.KELVIN,
                    np.array([x, 1 - x]),
                    tp_init=p_init,
                    vapor_molefracs=y_init,
                )
                y = vle.vapor.molefracs[0]
                p = vle.vapor.pressure() / si.BAR
                if y - x < 1e-4 * x:  # trivial solution or past the critical point
                    raise RuntimeError
                if (
                    rows and p < rows[-1][2]
                ):  # jump to another branch: p must increase
                    raise RuntimeError
                if len(rows) >= 2 and abs(p * si.BAR / p_init - 1) > 0.1:
                    raise RuntimeError
            except KeyboardInterrupt:
                raise
            except BaseException:  # also catches Rust panics (PanicException)
                if not rows:
                    break
                x = rows[-1][0] + dx / 2
                dx /= 2
                continue
            rows.append((x, y, p))
            dx = min(
                2 * dx, dx0
            )  # small steps at high dilution, where y changes quickly
            x += dx
        return pd.DataFrame(rows, columns=["x_he", "y_he", "p_bar"])

    return (bubble_curve,)


@app.cell
def _(bubble_curve, models_cs):
    temperatures_gemc = [20.4, 26.0, 29.0, 31.5]
    temperatures_exp = [15.5, 17.4, 23.0, 27.8]
    curves_cs = {
        (label, T): bubble_curve(eos, T)
        for label, eos in models_cs.items()
        for T in temperatures_gemc + temperatures_exp
    }
    return curves_cs, temperatures_exp, temperatures_gemc


@app.cell
def _(
    bubble_curve,
    curves_cs,
    mo,
    models_vrq,
    temperatures_exp,
    temperatures_gemc,
):
    # SAFT-VRQ Mie is slow (nested iterations for sigma_eff, eps_eff and d_BH in every evaluation) -> cached
    with mo.persistent_cache("mixture_curves_vrq"):
        curves_vrq = {
            (label, T): bubble_curve(eos, T)
            for label, eos in models_vrq.items()
            for T in temperatures_gemc + temperatures_exp
        }
    curves = curves_cs | curves_vrq
    return (curves,)


@app.cell
def _(curves, exp, gemc, markers, models, plt, styles):
    P_MAX = 200.0  # bar, upper limit of the plots


    def plot_isotherm(ax, T):
        _e = exp[(exp["T"] - T).abs() < 0.06]
        for ref, g in _e.groupby("reference"):
            kw = dict(ls="", marker=markers[ref], ms=4, color="C7", alpha=0.7)
            ax.plot(g["x_he"], g["p"], label=ref, **kw)
            ax.plot(g["y_he"], g["p"], **kw)
        _g = gemc[gemc["T"] == T]
        if len(_g):
            ax.plot(_g["x_he"], _g["p"], "o", mfc="w", mec="k", label="GEMC")
            ax.plot(_g["y_he"], _g["p"], "o", mfc="w", mec="k")
        p_top = [_e["p"].max() if len(_e) else 0, _g["p"].max() if len(_g) else 0]
        for rule in models:
            df = curves[rule, T]
            if len(df):
                ax.plot(
                    df["x_he"],
                    df["p_bar"],
                    label=rule,
                    **styles[rule],
                )
                ax.plot(df["y_he"], df["p_bar"], **styles[rule])
                p_top.append(df["p_bar"].max())
        ax.set_title(f"T = {T} K")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, min(1.1 * max(p_top), P_MAX))


    def plot_isotherms(temps):
        fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True)
        for ax, T in zip(axes.flat, temps):
            plot_isotherm(ax, T)
        for ax in axes[1]:
            ax.set_xlabel(r"$x_\mathrm{He}, y_\mathrm{He}$")
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$p$ / bar")
        handles = {}
        for ax in axes.flat:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(l, h)
        fig.legend(
            handles.values(),
            handles.keys(),
            loc="lower center",
            ncol=4,
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        return fig

    return (plot_isotherms,)


@app.cell
def _(plot_isotherms, temperatures_gemc):
    plot_isotherms(temperatures_gemc)
    return


@app.cell
def _(plot_isotherms, temperatures_exp):
    plot_isotherms(temperatures_exp)
    return


@app.cell
def _(exp, mixture_deviations, models_cs):
    mixture_dev_cs = {
        label: mixture_deviations(eos, exp) for label, eos in models_cs.items()
    }
    return (mixture_dev_cs,)


@app.cell
def _(exp, mixture_deviations, mo, models_vrq):
    with mo.persistent_cache("mixture_deviations_vrq"):
        mixture_dev_vrq = {
            label: mixture_deviations(eos, exp)
            for label, eos in models_vrq.items()
        }
    return (mixture_dev_vrq,)


@app.cell
def _(exp, mixture_dev_cs, mixture_dev_vrq, mo, np, pd, summarize):
    mixture_dev = mixture_dev_cs | mixture_dev_vrq
    # compare the models on the points where all of them converge
    converged_all = np.logical_and.reduce(
        [d["p_calc"].notna().to_numpy() for d in mixture_dev.values()]
    )
    _counts = ", ".join(
        f"{label}: {d['p_calc'].notna().sum()}" for label, d in mixture_dev.items()
    )
    mo.vstack(
        [
            mo.md(
                f"Converged of {len(exp)} points: {_counts}; all models: {converged_all.sum()}. "
                "Dew points above the model's mixture critical pressure have no solution."
            ),
            pd.concat(
                {
                    label: pd.concat(
                        [
                            d[converged_all]
                            .groupby(["kind", "reference"])
                            .apply(summarize, include_groups=False),
                            summarize(d[converged_all]).to_frame(("all", "")).T,
                        ]
                    )
                    for label, d in mixture_dev.items()
                },
                axis=1,
            )
            .round(3)
            .pipe(
                lambda t: t.set_axis(
                    [f"{label}: {m}" for label, m in t.columns], axis=1
                )
            )
            .rename_axis(["kind", "reference"])
            .reset_index(),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Findings (2026-10-20)

    **Pure fluids** (vs. NIST): SAFT-VR-CS with the Ziegler parameters reproduces $p_\mathrm{sat}$ within about 2 %
    and $\rho_l$ within 1 % (AARD) for both helium and hydrogen; larger deviations only close to $T_c$.
    SAFT-VRQ Mie (Aasen 2019) is clearly worse for helium (AARD $p_\mathrm{sat}$ 37 %), similar for hydrogen (6 %).

    **Mixture, predictive** ($k_{ij} = 0$; AARD $p$ on the 473 data points where all models converge):

    - SAFT-VR-CS `effective` 62.6 %, `classical` 61.3 %, SAFT-VRQ Mie ($k_{ij} = l_{ij} = 0$) 55.9 %.
    - All predictive models dissolve far too much helium in liquid hydrogen (bubble pressures too low).
    - The `classical` route raises the bubble pressures compared to `effective`; the effect is large near the
      mixture critical region at low $T$ (15.5 K, 17.4 K) but small on the data points, most of which are at low $x_\mathrm{He}$.
      It moves SAFT-VR-CS towards SAFT-VRQ Mie, which treats the unlike pair in the same way.

    **Mixture, fitted binary parameters:**

    - SAFT-VR-CS `classical` with fitted $k_{ij} = 0.257$: AARD $p$ 36.6 %, AAD of the coexisting composition 0.047.
    - SAFT-VRQ Mie (Aasen 2020, $k_{ij} = 0.08$, $l_{ij} = -0.05$): AARD $p$ 28.7 %, 0.034.
    - The fitted $k_{ij}$ of SAFT-VR-CS is much larger than that of SAFT-VRQ Mie. One candidate reason (not checked):
      the Ziegler helium $\varepsilon/k_\mathrm{B} = 9.56$ K is much larger than Aasen's 5.42 K, so the
      geometric-mean unlike energy starts higher. With $k_{ij} = 0.257$ the two-phase region opens up at high
      pressure (no closed loop below 200 bar at $T \le 29$ K) and is too wide at 27.8 K and 31.5 K; a single $k_{ij}$ cannot
      match low and high temperatures at the same time.
    """)
    return


if __name__ == "__main__":
    app.run()
