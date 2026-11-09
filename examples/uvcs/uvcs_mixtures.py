import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # uv-CS: helium / hydrogen mixtures

    Comparison of the mixture options of the uv-CS theory (`EquationOfState.uvcstheory`) for He/H$_2$,
    with SAFT-VR-CS and SAFT-VRQ Mie as reference models (see `examples/saftvrcs/saftvrcs_mixtures.py`
    and `examples/saftvrcs/hs_additivity_impact.py`).

    1. Pure helium and hydrogen with the uv-CS parameters of Ziegler (2026) vs. NIST.
    2. Cross second virial coefficient $B_{12}(T)$ for every option.
    3. He/H$_2$ phase equilibria vs. experimental data (DDB) and GEMC simulations.
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

    DATA = mo.notebook_dir() / "data"
    DATA_SAFTVRCS = mo.notebook_dir().parent / "saftvrcs" / "data"
    PARAMETERS = mo.notebook_dir().parents[1] / "parameters"
    VRQ_PATH = PARAMETERS / "saftvrqmie"
    SAFTVRCS_PARAMETER_FILE = PARAMETERS / "saftvrcs" / "saft_vr_cs_params.json"
    return (
        DATA,
        DATA_SAFTVRCS,
        EquationOfState,
        Parameters,
        PhaseDiagram,
        PhaseEquilibrium,
        SAFTVRCS_PARAMETER_FILE,
        State,
        VRQ_PATH,
        feos,
        json,
        mo,
        np,
        pd,
        plt,
        si,
    )


@app.cell
def _(feos, mo):
    MW = dict(Helium=4.002602, Hydrogen=2.0157309551872)  # g/mol, normal hydrogen
    COMPONENTS = list(MW)

    # uv-CS parameters, Ziegler (2026); c_* index k == xi_k
    ZIEGLER2026 = dict(
        Helium=dict(
            sigma=2.685576,
            epsilon_k=8.538732,
            rep=10.788977,
            att=6.0,
            c_rep=[1.0, 1.151765, -6.107946, 1.0, 1.0],
            c_sigma=[1.0, 2.730451, 1.0, -257.843813, 0.0],
            c_epsilon_k=[1.0, 1.250274, 1.0, -1428.749516, 2051.147273],
        ),
        Hydrogen=dict(
            sigma=3.025765,
            epsilon_k=31.690343,
            rep=10.689166,
            att=6.0,
            c_rep=[1.0, 1.347585, 1.0, 1.0, 1.0],
            c_sigma=[1.0, -2.789573, 1.0, 0.0, 0.0],
            c_epsilon_k=[1.0, -0.620551, 1.0, 0.0, 0.0],
        ),
    )


    def uvcs_record(name):
        """uv-CS pure record with the Ziegler (2026) parameters."""
        return feos.PureRecord(
            feos.Identifier(name=name), MW[name], **ZIEGLER2026[name]
        )


    def _parameter_table():
        """Markdown table of the uv-CS pure-component parameters (c_* index k == xi_k)."""
        names = list(ZIEGLER2026)
        rows = [
            ("$M$ / (g/mol)", lambda n: MW[n]),
            ("$\\sigma$ / Å", lambda n: ZIEGLER2026[n]["sigma"]),
            (
                "$\\varepsilon/k_\\mathrm{B}$ / K",
                lambda n: ZIEGLER2026[n]["epsilon_k"],
            ),
            ("$\\nu$", lambda n: ZIEGLER2026[n]["rep"]),
            ("$n$", lambda n: ZIEGLER2026[n]["att"]),
        ]
        for key, sym in [
            ("c_rep", "\\nu"),
            ("c_sigma", "\\sigma"),
            ("c_epsilon_k", "\\varepsilon"),
        ]:
            rows += [
                (
                    f"$\\xi_{k}^{{{sym}}}$ (`{key}[{k}]`)",
                    lambda n, key=key, k=k: ZIEGLER2026[n][key][k],
                )
                for k in range(5)
            ]
        lines = [
            "| parameter | " + " | ".join(names) + " |",
            "|---|" + "---:|" * len(names),
        ]
        lines += [
            f"| {label} | " + " | ".join(f"{f(n):.10g}" for n in names) + " |"
            for label, f in rows
        ]
        return "\n".join(lines)


    mo.md(
        "### Pure-component parameters\n\n"
        "uv-CS parameters of Ziegler (2026), used for all uv-CS models in this notebook "
        "(normal hydrogen). The $\\xi_k$ are the prefactors of the correction functions "
        "($\\xi^\\nu_k$: `c_rep`, $\\xi^\\sigma_k$: `c_sigma`, $\\xi^\\varepsilon_k$: `c_epsilon_k`).\n\n"
        + _parameter_table()
        + "\n\nReference models: SAFT-VR-CS with the Ziegler parameters "
        "(`parameters/saftvrcs/saft_vr_cs_params.json`), SAFT-VRQ Mie (FH1) with Aasen et al. (2019) "
        "pure and Aasen et al. (2020) binary parameters (`parameters/saftvrqmie/`)."
    )
    return COMPONENTS, MW, uvcs_record


@app.cell
def _(MW, Parameters, SAFTVRCS_PARAMETER_FILE, VRQ_PATH, feos, json):
    with open(SAFTVRCS_PARAMETER_FILE) as _f:
        _ziegler_vrcs = {
            n: _p["parameters"] for n, _p in json.load(_f).items() if n in MW
        }


    def saftvrcs_record(name):
        """SAFT-VR-CS pure record with the Ziegler parameters (Csigma k -> c_sigma[k], etc.)."""
        p = _ziegler_vrcs[name]
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

    return saftvrcs_record, vrq_parameters


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pure components vs. NIST

    The mixture options do not change the pure fluids. uv-CS (Ziegler 2026) and SAFT-VR-CS (Ziegler) for reference.
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
    pd,
    saftvrcs_record,
    si,
    uvcs_record,
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


    nist = {
        "Helium": read_nist(DATA / "helium_data" / "nist_vle.txt"),
        "Hydrogen": read_nist(DATA / "hydrogen_data" / "nist_vle.txt"),
    }
    UVCS, VRCS = "uv-CS (Ziegler 2026)", "SAFT-VR-CS (Ziegler)"
    pure_models = {
        name: {
            UVCS: EquationOfState.uvcstheory(
                Parameters.new_pure(uvcs_record(name))
            ),
            VRCS: EquationOfState.saftvrcs_mie(
                Parameters.new_pure(saftvrcs_record(name))
            ),
        }
        for name in COMPONENTS
    }
    pure_styles = {UVCS: dict(color="C0", ls="-"), VRCS: dict(color="C1", ls="--")}
    pure_diagrams = {
        (name, label): PhaseDiagram.pure(
            eos, nist[name]["T"].min() * si.KELVIN, 200
        )
        for name, _models in pure_models.items()
        for label, eos in _models.items()
    }
    pd.DataFrame(
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
    ).round(3)
    return nist, pure_diagrams, pure_models, pure_styles


@app.cell
def _(COMPONENTS, nist, plt, pure_diagrams, pure_models, pure_styles, si):
    _fig, _axes = plt.subplots(2, 2, figsize=(10, 8))
    _rho_unit = si.KILOGRAM / si.METER**3
    for (_ax_p, _ax_rho), _name in zip(_axes, COMPONENTS):
        _ref = nist[_name]
        for _label in pure_models[_name]:
            _dia = pure_diagrams[_name, _label]
            _T = _dia.vapor.temperature / si.KELVIN
            _ax_p.plot(
                _T,
                _dia.vapor.pressure / si.BAR,
                label=_label,
                **pure_styles[_label]
            )
            _ax_rho.plot(
                _dia.liquid.mass_density / _rho_unit,
                _T,
                label=_label,
                **pure_styles[_label]
            )
            _ax_rho.plot(
                _dia.vapor.mass_density / _rho_unit, _T, **pure_styles[_label]
            )
        _kw = dict(ls="", marker="o", mfc="none", mec="k", ms=5)
        _ax_p.plot(_ref["T"], _ref["p"], label="NIST", **_kw)
        _ax_rho.plot(_ref["rho_l"], _ref["T"], label="NIST", **_kw)
        _ax_rho.plot(_ref["rho_v"], _ref["T"], **_kw)
        _ax_p.set(xlabel="$T$ / K", ylabel="$p_\\mathrm{sat}$ / bar", title=_name)
        _ax_rho.set(xlabel="$\\rho$ / (kg/m$^3$)", ylabel="$T$ / K", title=_name)
        _ax_p.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## He/H$_2$ mixture: uv-CS options

    Every component is mapped onto a classical Mie $\nu$-6 potential with temperature-dependent effective
    parameters $\sigma^\mathrm{eff}$, $\varepsilon^\mathrm{eff}$, $\nu^\mathrm{eff}$ (corrections in
    $D^* = \hbar^2 / (12 m k_\mathrm{B} T \sigma^2)$). Combining rules for the unlike pair (fixed, no option):

    $$\nu_{ij} = \frac{\nu_i + \nu_j}{2}, \qquad n_{ij} = 6, \qquad
    \sigma_{ij} = \frac{\sigma_i + \sigma_j}{2}\,(1 - l_{ij}), \qquad
    \varepsilon_{ij} = \sqrt{\varepsilon_i \varepsilon_j}\,(1 - k_{ij})$$

    Options of `EquationOfState.uvcstheory(parameters, max_eta=0.5, combining_rule="classical", non_additive_hs=False)`:

    | option | value | meaning |
    |---|---|---|
    | `combining_rule` | **`"classical"` (default)** | apply the combining rules to the classical parameters, then map the pair onto effective parameters at the pair mass $m_{ij} = 2 m_i m_j/(m_i + m_j)$ with prefactors $\zeta_{ij} = (\zeta_i + \zeta_j)/2$ |
    | | `"effective"` | apply the combining rules to the effective pure-component parameters, each evaluated at its own mass $m_i$ |
    | `non_additive_hs` | **`False` (default)** | additive hard spheres, $d_{ij} = (d_{ii} + d_{jj})/2$ |
    | | `True` | $d_{ij}$ = WCA diameter of the unlike pair potential in $\Delta a_0$, plus a first-order correction of the BMCSL term |
    | `max_eta` | **`0.5` (default)** | maximum packing fraction (not varied here) |

    Binary parameters: $k_{ij} = l_{ij} = 0$ (default of `Parameters.new_binary`) for all models below unless stated
    otherwise. Component 0 is helium.
    """)
    return


@app.cell
def _(EquationOfState, Parameters, uvcs_record):
    def uvcs_mixture(k_ij=0.0, l_ij=0.0, **options):
        """uv-CS He/H2 (component 0: helium) with the Ziegler (2026) pure parameters."""
        _params = Parameters.new_binary(
            [uvcs_record("Helium"), uvcs_record("Hydrogen")], k_ij=k_ij, l_ij=l_ij
        )
        return EquationOfState.uvcstheory(_params, **options)


    # defaults of EquationOfState.uvcstheory (see the table above)
    UVCS_DEFAULTS = dict(combining_rule="classical", non_additive_hs=False)
    UVCS_OPTIONS = {
        "uv-CS, classical, additive HS (default)": dict(),
        "uv-CS, classical, non-additive HS": dict(non_additive_hs=True),
        "uv-CS, effective, additive HS": dict(combining_rule="effective"),
        "uv-CS, effective, non-additive HS": dict(
            combining_rule="effective", non_additive_hs=True
        ),
    }
    UVCS_DEFAULT_LABEL = "uv-CS, classical, additive HS (default)"
    models_uvcs = {
        label: uvcs_mixture(**opt) for label, opt in UVCS_OPTIONS.items()
    }
    return UVCS_DEFAULTS, UVCS_DEFAULT_LABEL, UVCS_OPTIONS, models_uvcs


@app.cell
def _(
    COMPONENTS,
    EquationOfState,
    Parameters,
    saftvrcs_record,
    vrq_parameters,
):
    # reference models, separate cell so that their (slow) cached results survive changes of the uv-CS models
    REFERENCE_MODELS = {
        "SAFT-VR-CS, classical": dict(
            eos="SAFT-VR-CS (`saftvrcs_mie`)",
            options='`combining_rule="classical"`, `non_additive_hs=True` (default of `saftvrcs_mie`)',
            parameters="Ziegler, `parameters/saftvrcs/saft_vr_cs_params.json`",
            binary="$k_{ij} = l_{ij} = \\gamma_{ij} = 0$",
        ),
        "SAFT-VRQ Mie, $k_{ij} = l_{ij} = 0$": dict(
            eos="SAFT-VRQ Mie (`saftvrqmie`, FH1)",
            options="defaults",
            parameters="Aasen et al. (2019), `parameters/saftvrqmie/aasen2019.json`",
            binary="$k_{ij} = l_{ij} = 0$",
        ),
        "SAFT-VRQ Mie (Aasen 2020)": dict(
            eos="SAFT-VRQ Mie (`saftvrqmie`, FH1)",
            options="defaults",
            parameters="Aasen et al. (2019) + `aasen2020_binary.json`",
            binary="$k_{ij} = 0.08$, $l_{ij} = -0.05$ (fitted)",
        ),
    }
    models_ref = {
        "SAFT-VR-CS, classical": EquationOfState.saftvrcs_mie(
            Parameters.new_binary(
                [saftvrcs_record("Helium"), saftvrcs_record("Hydrogen")]
            ),
            combining_rule="classical",
        ),
        "SAFT-VRQ Mie, $k_{ij} = l_{ij} = 0$": EquationOfState.saftvrqmie(
            vrq_parameters(COMPONENTS)
        ),
        "SAFT-VRQ Mie (Aasen 2020)": EquationOfState.saftvrqmie(
            vrq_parameters(COMPONENTS, binary=True)
        ),
    }
    return REFERENCE_MODELS, models_ref


@app.cell
def _(
    REFERENCE_MODELS,
    UVCS_DEFAULTS,
    UVCS_OPTIONS,
    mo,
    models_ref,
    models_uvcs,
):
    models = models_uvcs | models_ref
    # color: route (classical red, effective blue), line style: hard spheres (additive solid, non-additive dashed)
    styles = {
        label: dict(
            color="C3"
            if (UVCS_DEFAULTS | opt)["combining_rule"] == "classical"
            else "C0",
            ls="--" if (UVCS_DEFAULTS | opt)["non_additive_hs"] else "-",
        )
        for label, opt in UVCS_OPTIONS.items()
    } | dict(
        zip(
            models_ref,
            [
                dict(color="C4", ls="-."),
                dict(color="k", ls=":"),
                dict(color="k", ls="-."),
            ],
        )
    )


    def _fmt(key, value, opt):
        text = f'`"{value}"`' if isinstance(value, str) else f"`{value}`"
        return text + (" (default)" if key not in opt else "")


    _rows = [
        f"| {label} | uv-CS (`uvcstheory`) | "
        + ", ".join(
            f"`{k}`: {_fmt(k, v, opt)}" for k, v in (UVCS_DEFAULTS | opt).items()
        )
        + " | Ziegler (2026), table above | $k_{ij} = l_{ij} = 0$ |"
        for label, opt in UVCS_OPTIONS.items()
    ] + [
        f"| {label} | {r['eos']} | {r['options']} | {r['parameters']} | {r['binary']} |"
        for label, r in REFERENCE_MODELS.items()
    ]
    mo.md(
        "### Models in the comparison\n\n"
        "| model | equation of state | options | pure-component parameters | binary parameters |\n"
        "|---|---|---|---|---|\n" + "\n".join(_rows)
    )
    return models, styles


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cross second virial coefficient

    $B_{12}$ isolates the unlike interaction: $B(x) = x_1^2 B_{11} + 2 x_1 x_2 B_{12} + x_2^2 B_{22}$, so
    $B_{12} = 2 B(x_1 = 1/2) - (B_{11} + B_{22})/2$. With additive hard spheres, $B_{12}$ of uv-CS is the exact
    $B_2$ of the effective unlike Mie pair; with `non_additive_hs` it is exact only to first order in $d_{ij} - \bar d_{ij}$.
    """)
    return


@app.cell
def _(UVCS_DEFAULT_LABEL, models, np, pd, plt, si, styles):
    def cross_virial(eos, T):
        """B_12 in cm^3/mol at temperature T (K)."""
        _b = lambda x: eos.second_virial_coefficient(
            T * si.KELVIN, np.array([x, 1 - x])
        ) / ((si.CENTI * si.METER) ** 3 / si.MOL)
        return 2 * _b(0.5) - 0.5 * (_b(1.0) + _b(0.0))


    temperatures_b12 = np.geomspace(5.0, 300.0, 60)
    b12 = pd.DataFrame(
        {
            label: [cross_virial(eos, T) for T in temperatures_b12]
            for label, eos in models.items()
        },
        index=pd.Index(temperatures_b12, name="T / K"),
    )

    _fig, (_ax, _ax_rel) = plt.subplots(1, 2, figsize=(12, 4.5))
    _default = b12[UVCS_DEFAULT_LABEL]
    for _label in models:
        _ax.plot(b12.index, b12[_label], label=_label, **styles[_label])
        _ax_rel.plot(b12.index, b12[_label] - _default, **styles[_label])
    _ax.set(
        xscale="log",
        ylim=(-270, 30),
        xlabel="$T$ / K",
        ylabel="$B_{12}$ / (cm$^3$/mol)",
    )
    _ax_rel.set(
        xscale="log",
        xlabel="$T$ / K",
        ylabel="$B_{12} - B_{12}^\\mathrm{default}$ / (cm$^3$/mol)",
    )
    _ax.axhline(0, color="0.8", lw=0.8)
    _ax_rel.axhline(0, color="0.8", lw=0.8)
    _fig.legend(
        *_ax.get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8
    )
    _fig.tight_layout(rect=(0, 0.15, 1, 1))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Phase equilibria

    Experimental data: DDB export (`data/helium_hydrogen_ddb_vle.csv`, $T < 33$ K) and GEMC simulations
    (`../saftvrcs/data/helium_hydrogen_gemc_vle.csv`). Isotherms are traced by bubble-point continuation in
    $x_\mathrm{He}$ from (almost) pure hydrogen up to the mixture critical point (same algorithm as in the
    SAFT-VR-CS notebooks).
    """)
    return


@app.cell
def _(DATA, DATA_SAFTVRCS, pd):
    exp = pd.read_csv(DATA / "helium_hydrogen_ddb_vle.csv").rename(
        columns={"temperature_K": "T", "pressure_bar": "p"}
    )
    exp = exp[exp["T"] < 33.0]
    gemc = pd.read_csv(
        DATA_SAFTVRCS / "helium_hydrogen_gemc_vle.csv", sep=r"\s+"
    ).rename(columns={"temperature[K]": "T", "pressure[bar]": "p"})
    markers = dict(zip(sorted(exp["reference"].unique()), "sD^v<>ph"))
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
def _(bubble_curve, mo, models_ref, models_uvcs):
    temperatures_gemc = [20.4, 26.0, 29.0, 31.5]
    temperatures_exp = [15.5, 17.4, 23.0, 27.8]
    # cached on disk; reference models separately (SAFT-VRQ Mie is slow)
    with mo.persistent_cache("mixture_curves_uvcs"):
        curves_uvcs = {
            (label, T): bubble_curve(eos, T)
            for label, eos in models_uvcs.items()
            for T in temperatures_gemc + temperatures_exp
        }
    with mo.persistent_cache("mixture_curves_ref"):
        curves_ref = {
            (label, T): bubble_curve(eos, T)
            for label, eos in models_ref.items()
            for T in temperatures_gemc + temperatures_exp
        }
    curves = curves_uvcs | curves_ref
    return curves, temperatures_exp, temperatures_gemc


@app.cell
def _(curves, exp, gemc, markers, plt, styles):
    P_MAX = 200.0  # bar, upper limit of the plots


    def plot_isotherm(ax, T, labels):
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
        for label in labels:
            df = curves[label, T]
            if len(df):
                ax.plot(df["x_he"], df["p_bar"], label=label, **styles[label])
                ax.plot(df["y_he"], df["p_bar"], **styles[label])
                p_top.append(df["p_bar"].max())
        ax.set_title(f"T = {T} K")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, min(1.1 * max(p_top), P_MAX))


    def plot_isotherms(temps, labels):
        fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True)
        for ax, T in zip(axes.flat, temps):
            plot_isotherm(ax, T, labels)
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
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        return fig

    return (plot_isotherms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### uv-CS options
    """)
    return


@app.cell
def _(models_uvcs, plot_isotherms, temperatures_gemc):
    plot_isotherms(temperatures_gemc, list(models_uvcs))
    return


@app.cell
def _(models_uvcs, plot_isotherms, temperatures_exp):
    plot_isotherms(temperatures_exp, list(models_uvcs))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### uv-CS vs. SAFT-VR-CS and SAFT-VRQ Mie
    """)
    return


@app.cell
def _(UVCS_DEFAULT_LABEL, models_ref, plot_isotherms, temperatures_gemc):
    plot_isotherms(
        temperatures_gemc,
        [UVCS_DEFAULT_LABEL, "uv-CS, effective, additive HS", *models_ref],
    )
    return


@app.cell
def _(UVCS_DEFAULT_LABEL, models_ref, plot_isotherms, temperatures_exp):
    plot_isotherms(
        temperatures_exp,
        [UVCS_DEFAULT_LABEL, "uv-CS, effective, additive HS", *models_ref],
    )
    return


if __name__ == "__main__":
    app.run()
