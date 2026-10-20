import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SAFT-VRQ Mie: additive vs. non-additive hard spheres for He/H$_2$

    Impact of the non-additive hard-sphere correction on the He/H$_2$ phase behaviour with SAFT-VRQ Mie (FH1).

    - **non-additive** (default, `inc_nonadd_term=True`): BMCSL reference with the like diameters $d_{ii}$ plus the
      first-order non-additive correction for $d_{ij} \neq (d_{ii} + d_{jj})/2$, where $d_{ij}$ is the Barker-Henderson
      diameter of the effective unlike pair potential.
    - **additive** (`inc_nonadd_term=False`): BMCSL reference only.

    Parameters: Aasen et al. (2019) pure, with $k_{ij} = l_{ij} = 0$ (predictive) and with the Aasen et al. (2020)
    binary parameters ($k_{ij} = 0.08$, $l_{ij} = -0.05$). Component 0 is helium.

    Note: the Aasen et al. (2020) $k_{ij}$ and $l_{ij}$ were fitted with the non-additive hard-sphere correction.
    The "additive HS, Aasen 2020" model therefore uses binary parameters that were not fitted for it, so its comparison
    with the non-additive model is somewhat unfair: it shows how much the published fit depends on the correction term,
    not the best additive SAFT-VRQ Mie result.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hard-sphere diameters $d_{ij}$ in the dispersion term

    Checked in the feos source (`crates/feos/src`):

    | model | $d_{ij}$ ($i \neq j$) in the dispersion term | non-additive HS correction |
    |---|---|---|
    | SAFT-VRQ Mie (`saftvrqmie`) | **always non-additive**: Barker-Henderson diameter of the effective unlike FH-Mie potential | optional (`inc_nonadd_term`, default on) |
    | SAFT-VR Mie (`saftvrmie`) | **always additive**: $d_{ij} = (d_{ii} + d_{jj})/2$ | none |
    | SAFT-VR-CS (`saftvrcs_mie`) | non-additive or additive, set by `non_additive_hs` | coupled to `non_additive_hs` |

    - SAFT-VRQ Mie: `TemperatureDependentProperties` computes `hs_diameter_ij(i, j)` for all pairs by integrating
      the unlike potential `qmie_potential_ij` (`saftvrqmie/eos/hard_sphere.rs`, `saftvrqmie/eos/mod.rs`). The dispersion
      term uses this matrix in $\zeta_x = \frac{\pi}{6} \rho_s \sum_{ij} x_{s,i} x_{s,j} d_{ij}^3$ and in $x_{0,ij} = \sigma_{ij}/d_{ij}$
      (`saftvrqmie/eos/dispersion.rs`). `inc_nonadd_term=False` only drops the `Non additive Hard Sphere` contribution;
      the dispersion term is unchanged.
    - The BMCSL hard-sphere term of SAFT-VRQ Mie uses only the like diameters $d_{ii}$ (additive) in both cases.
    - SAFT-VR Mie: `d_ij = (diameter[i] + diameter[j]) * 0.5` in `saftvrmie/eos/dispersion.rs`.

    In this notebook, "additive HS" for SAFT-VRQ Mie therefore means an additive hard-sphere **reference** only, not a
    fully additive model as in SAFT-VR Mie.
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
    from scipy.optimize import minimize

    DATA = mo.notebook_dir() / "data"
    PARAMETERS = mo.notebook_dir().parents[1] / "parameters"
    PARAMETER_FILE = PARAMETERS / "saftvrcs" / "saft_vr_cs_params.json"
    VRQ_PATH = PARAMETERS / "saftvrqmie"
    return (
        DATA,
        EquationOfState,
        PARAMETER_FILE,
        Parameters,
        PhaseEquilibrium,
        State,
        VRQ_PATH,
        feos,
        json,
        minimize,
        mo,
        np,
        pd,
        plt,
        si,
    )


@app.cell
def _(EquationOfState, Parameters, State, VRQ_PATH, feos, np, pd, si):
    COMPONENTS = ["Helium", "Hydrogen"]  # component 0: helium


    def vrq_parameters(binary=False):
        """SAFT-VRQ Mie (FH1) He/H2: Aasen et al. (2019) pure, Aasen et al. (2020) binary."""
        files = [str(VRQ_PATH / "aasen2019.json")]
        if binary:
            files.append(str(VRQ_PATH / "aasen2020_binary.json"))
        return Parameters.from_json(
            [n.lower() for n in COMPONENTS],
            *files,
            identifier_option=feos.IdentifierOption.Name,
        )


    HS_LABELS = {True: "non-additive HS", False: "additive HS"}
    BINARY_LABELS = {False: "$k_{ij} = l_{ij} = 0$", True: "Aasen 2020"}
    models = {
        f"{HS_LABELS[non_additive]}, {BINARY_LABELS[binary]}": EquationOfState.saftvrqmie(
            vrq_parameters(binary), inc_nonadd_term=non_additive
        )
        for binary in BINARY_LABELS
        for non_additive in HS_LABELS
    }
    # color: binary parameters, line style: hard-sphere reference
    styles = {
        f"{HS_LABELS[non_additive]}, {BINARY_LABELS[binary]}": dict(
            color=color, ls="-" if non_additive else "--"
        )
        for binary, color in zip(BINARY_LABELS, ["k", "C3"])
        for non_additive in HS_LABELS
    }

    # sanity check: residual Helmholtz energy contributions at a liquid-like state (20.4 K, 35 kmol/m3, x_He = 0.05)
    _T = 20.4 * si.KELVIN
    _states = {
        label: State(
            eos,
            _T,
            density=35 * si.KILO * si.MOL / si.METER**3,
            molefracs=np.array([0.05, 0.95]),
        )
        for label, eos in models.items()
    }
    pd.DataFrame(
        {
            label: {
                name: a / (si.RGAS * _T)
                for name, a in s.residual_molar_helmholtz_energy_contributions()
            }
            for label, s in _states.items()
        }
    ).rename_axis("a_res / RT").round(5)
    return BINARY_LABELS, COMPONENTS, HS_LABELS, models, styles


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## He/H$_2$ VLE

    Bubble-point curves (continuation in $x_\mathrm{He}$ up to the mixture critical point) for the four models.
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
def _(bubble_curve, mo, models):
    temperatures_gemc = [20.4, 26.0, 29.0, 31.5]
    temperatures_exp = [15.5, 17.4, 23.0, 27.8]
    # SAFT-VRQ Mie is slow (nested iterations for sigma_eff, eps_eff and d_BH in every evaluation) -> cached
    with mo.persistent_cache("mixture_curves_vrq_hs"):
        curves = {
            (label, T): bubble_curve(eos, T)
            for label, eos in models.items()
            for T in temperatures_gemc + temperatures_exp
        }
    return curves, temperatures_exp, temperatures_gemc


@app.cell
def _(exp, gemc, markers, plt):
    P_MAX = 200.0  # bar, upper limit of the plots


    def plot_isotherm(ax, T, curves, styles):
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
        for label, style in styles.items():
            df = curves[label, T]
            if len(df):
                ax.plot(df["x_he"], df["p_bar"], label=label, **style)
                ax.plot(df["y_he"], df["p_bar"], **style)
                p_top.append(df["p_bar"].max())
        ax.set_title(f"T = {T} K")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, min(1.1 * max(p_top), P_MAX))


    def plot_isotherms(temps, curves, styles):
        """2x2 isotherms with data; curves[label, T] are bubble curves, styles[label] the line styles."""
        fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True)
        for ax, T in zip(axes.flat, temps):
            plot_isotherm(ax, T, curves, styles)
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
def _(curves, plot_isotherms, styles, temperatures_gemc):
    plot_isotherms(temperatures_gemc, curves, styles)
    return


@app.cell
def _(curves, plot_isotherms, styles, temperatures_exp):
    plot_isotherms(temperatures_exp, curves, styles)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SAFT-VR-CS: additive vs. non-additive hard spheres

    SAFT-VR-CS with the Ziegler parameters (`parameters/saftvrcs/saft_vr_cs_params.json`), $k_{ij} = \gamma_{ij} = 0$,
    `classical` route to the unlike parameters (see `saftvrcs_mixtures.py`):

    1. SAFT-VR Mie combining rules on the **classical** pure-component parameters: $\sigma_{ij}$, $\varepsilon_{ij}$, $\lambda_{r,ij}$.
    2. Pair quantities: $D^*_{ij} = \hbar^2 / (12\, m_{ij}\, k_\mathrm{B} T\, \sigma_{ij}^2)$ with $m_{ij} = 2 m_i m_j / (m_i + m_j)$,
       prefactors $\zeta^X_{ij} = (\zeta^X_i + \zeta^X_j)/2$.
    3. Effective unlike parameters from the ansatz functions: $X_{\mathrm{eff},ij} = X_{ij}\, F^X(D^*_{ij}, \lambda_{r,ij}; \zeta^X_{ij})$.

    This route corresponds to SAFT-VRQ Mie, where the Feynman-Hibbs correction acts on the combined classical unlike pair.

    - **non-additive** (`non_additive_hs=True`, default): $d_{ij}$ from the effective unlike pair potential in the
      dispersion term plus the non-additive hard-sphere correction, as in SAFT-VRQ Mie.
    - **additive** (`non_additive_hs=False`): $d_{ij} = (d_{ii} + d_{jj})/2$ in the dispersion term and no correction,
      i.e. fully additive as in SAFT-VR Mie.

    Unlike SAFT-VRQ Mie, the switch here also changes the dispersion term.
    """)
    return


@app.cell
def _(
    COMPONENTS,
    EquationOfState,
    HS_LABELS,
    PARAMETER_FILE,
    Parameters,
    State,
    feos,
    json,
    np,
    pd,
    si,
):
    MW = dict(Helium=4.002602, Hydrogen=2.0157309551872)  # g/mol, normal hydrogen

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


    def saftvrcs_mixture(non_additive_hs, k_ij=0.0, l_ij=0.0):
        """SAFT-VR-CS He/H2 (component 0: helium), Ziegler pure parameters, `classical` combining route."""
        _params = Parameters.new_binary(
            [record(n) for n in COMPONENTS], k_ij=k_ij, l_ij=l_ij
        )
        return EquationOfState.saftvrcs_mie(
            _params, combining_rule="classical", non_additive_hs=non_additive_hs
        )


    models_cs = {
        f"SAFT-VR-CS, {HS_LABELS[non_additive]}": saftvrcs_mixture(non_additive)
        for non_additive in HS_LABELS
    }
    # line style: hard-sphere treatment (as for SAFT-VRQ Mie)
    styles_cs = {
        f"SAFT-VR-CS, {HS_LABELS[non_additive]}": dict(
            color="C0", ls="-" if non_additive else "--"
        )
        for non_additive in HS_LABELS
    }

    # sanity check: residual Helmholtz energy contributions at the same liquid-like state as above
    _T = 20.4 * si.KELVIN
    _states = {
        label: State(
            eos,
            _T,
            density=35 * si.KILO * si.MOL / si.METER**3,
            molefracs=np.array([0.05, 0.95]),
        )
        for label, eos in models_cs.items()
    }
    pd.DataFrame(
        {
            label: {
                name: a / (si.RGAS * _T)
                for name, a in s.residual_molar_helmholtz_energy_contributions()
            }
            for label, s in _states.items()
        }
    ).rename_axis("a_res / RT").round(5)
    return models_cs, saftvrcs_mixture, styles_cs


@app.cell
def _(bubble_curve, models_cs, temperatures_exp, temperatures_gemc):
    curves_cs = {
        (label, T): bubble_curve(eos, T)
        for label, eos in models_cs.items()
        for T in temperatures_gemc + temperatures_exp
    }
    return (curves_cs,)


@app.cell
def _(curves_cs, plot_isotherms, styles_cs, temperatures_gemc):
    plot_isotherms(temperatures_gemc, curves_cs, styles_cs)
    return


@app.cell
def _(curves_cs, plot_isotherms, styles_cs, temperatures_exp):
    plot_isotherms(temperatures_exp, curves_cs, styles_cs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fit of $k_{ij}$ and $l_{ij}$ for SAFT-VR-CS

    Both binary parameters of the classical unlike pair, $\sigma_{ij} = (1 - l_{ij})(\sigma_i + \sigma_j)/2$ and
    $\varepsilon_{ij} = (1 - k_{ij})\, \sigma_i^{3/2} \sigma_j^{3/2} / \sigma_{ij}^3\, \sqrt{\varepsilon_i \varepsilon_j}$, are fitted
    separately for non-additive and additive hard spheres ($\gamma_{ij} = 0$, `classical` route).

    - Objective: AARD of the bubble pressures at the experimental $(T, x_\mathrm{He})$ (DDB, all references);
      points without a converged bubble point count as 100 %.
    - Nelder-Mead, starting at the SAFT-VRQ Mie values of Aasen et al. (2020), $k_{ij} = 0.08$, $l_{ij} = -0.05$.
    - About 7 s per objective evaluation, so each fit takes several minutes; the result is cached (`kij_lij_fit_cs`).
    """)
    return


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

    return (mixture_deviations,)


@app.cell
def _(HS_LABELS, exp, minimize, mixture_deviations, mo, pd, saftvrcs_mixture):
    exp_bubble = exp[exp["x_he"].notna()]
    KIJ_LIJ_START = [0.08, -0.05]  # SAFT-VRQ Mie, Aasen et al. (2020)


    def kij_lij_objective(x, non_additive_hs):
        """AARD of the bubble pressures; points without a converged bubble point count as 100 %."""
        _eos = saftvrcs_mixture(non_additive_hs, k_ij=x[0], l_ij=x[1])
        _d = mixture_deviations(_eos, exp_bubble)
        return (_d["p_calc"] / _d["p"] - 1).abs().fillna(1.0).mean()


    with mo.persistent_cache("kij_lij_fit_cs"):
        kij_lij_fits = {
            non_additive: minimize(
                kij_lij_objective,
                KIJ_LIJ_START,
                args=(non_additive,),
                method="Nelder-Mead",
                options=dict(
                    initial_simplex=[KIJ_LIJ_START, [0.13, -0.05], [0.08, 0.0]],
                    xatol=1e-3,
                    fatol=1e-4,
                ),
            )
            for non_additive in HS_LABELS
        }
    pd.DataFrame(
        {
            HS_LABELS[non_additive]: {
                "k_ij": res.x[0],
                "l_ij": res.x[1],
                "AARD p_bubble / %": 100 * res.fun,
                "evaluations": res.nfev,
                "converged": res.success,
            }
            for non_additive, res in kij_lij_fits.items()
        }
    ).T
    return (kij_lij_fits,)


@app.cell
def _(
    BINARY_LABELS,
    HS_LABELS,
    bubble_curve,
    curves,
    kij_lij_fits,
    saftvrcs_mixture,
    temperatures_exp,
    temperatures_gemc,
):
    VRQ_AASEN = (
        f"{HS_LABELS[True]}, {BINARY_LABELS[True]}"  # SAFT-VRQ Mie as published
    )
    models_cs_fit = {
        f"SAFT-VR-CS, {HS_LABELS[non_additive]}, fitted": saftvrcs_mixture(
            non_additive, k_ij=res.x[0], l_ij=res.x[1]
        )
        for non_additive, res in kij_lij_fits.items()
    }
    curves_cs_fit = {
        (label, T): bubble_curve(eos, T)
        for label, eos in models_cs_fit.items()
        for T in temperatures_gemc + temperatures_exp
    }
    # fitted SAFT-VR-CS vs. SAFT-VRQ Mie (Aasen 2020)
    curves_fit = curves_cs_fit | {
        (f"SAFT-VRQ Mie, {VRQ_AASEN}", T): curves[VRQ_AASEN, T]
        for T in temperatures_gemc + temperatures_exp
    }
    styles_fit = {
        f"SAFT-VR-CS, {HS_LABELS[non_additive]}, fitted": dict(
            color="C2", ls="-" if non_additive else "--"
        )
        for non_additive in HS_LABELS
    } | {f"SAFT-VRQ Mie, {VRQ_AASEN}": dict(color="C3", ls="-")}
    return curves_fit, styles_fit


@app.cell
def _(curves_fit, plot_isotherms, styles_fit, temperatures_gemc):
    plot_isotherms(temperatures_gemc, curves_fit, styles_fit)
    return


@app.cell
def _(curves_fit, plot_isotherms, styles_fit, temperatures_exp):
    plot_isotherms(temperatures_exp, curves_fit, styles_fit)
    return


if __name__ == "__main__":
    app.run()
