import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # SAFT-VRQ Mie: BMCSL vs. Santos hard-sphere reference

    Both models treat the hard-sphere reference as **non-additive**, with the Barker-Henderson diameters $d_{ij}$
    of the unlike Feynman-Hibbs-Mie potentials, $d_{ij} \neq (d_{ii} + d_{jj})/2$:

    - **BMCSL + correction** (`hs_reference="bmcsl"`, default): additive BMCSL mixture of the like diameters $d_{ii}$
      plus the correction that is first order in $d_{ij} - (d_{ii} + d_{jj})/2$.
    - **Santos** (`hs_reference="santos"`): non-additive hard-sphere mixture of Santos et al., as used by
      Aasen et al., J. Chem. Phys. 152, 074507 (2020), Eqs. (14)-(24). It reproduces the exact second and third virial
      coefficients of the non-additive hard-sphere mixture.

    The dispersion term is identical in both models. Parameters: Aasen et al. (2019) pure (FH1) and Aasen et al. (2020)
    binary parameters (fitted with the Santos reference). Component 0 is helium in all mixtures.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import si_units as si
    import feos
    from feos import EquationOfState, Parameters, PhaseEquilibrium, State
    from scipy.integrate import quad
    from scipy.optimize import brentq

    EXAMPLES = mo.notebook_dir().parent
    VRQ_PATH = mo.notebook_dir().parents[1] / "parameters" / "saftvrqmie"
    return (
        EXAMPLES,
        EquationOfState,
        Parameters,
        PhaseEquilibrium,
        State,
        VRQ_PATH,
        feos,
        mo,
        np,
        pd,
        plt,
        quad,
        si,
    )


@app.cell
def _(EquationOfState, Parameters, State, VRQ_PATH, feos, np, pd, si):
    MIXTURES = {"He/H2": ["helium", "hydrogen"], "He/Ne": ["helium", "neon"]}


    def vrq_parameters(components):
        """SAFT-VRQ Mie (FH1): Aasen et al. (2019) pure, Aasen et al. (2020) binary parameters."""
        return Parameters.from_json(
            components,
            str(VRQ_PATH / "aasen2019.json"),
            str(VRQ_PATH / "aasen2020_binary.json"),
            identifier_option=feos.IdentifierOption.Name,
        )


    HS_REFERENCES = {"BMCSL + correction": "bmcsl", "Santos": "santos"}
    models = {
        (mixture, label): EquationOfState.saftvrqmie(
            vrq_parameters(components), hs_reference=hs
        )
        for mixture, components in MIXTURES.items()
        for label, hs in HS_REFERENCES.items()
    }
    styles = {
        "BMCSL + correction": dict(color="C0", ls="-"),
        "Santos": dict(color="C3", ls="--"),
    }

    # sanity check: residual Helmholtz energy contributions at a liquid-like state (x_He = 0.05)
    _T = 25.0 * si.KELVIN
    _rows = {}
    for (_mixture, _label), _eos in models.items():
        _s = State(
            _eos,
            _T,
            density=35 * si.KILO * si.MOL / si.METER**3,
            molefracs=np.array([0.05, 0.95]),
        )
        _rows[_mixture, _label] = {
            name: a / (si.RGAS * _T)
            for name, a in _s.residual_molar_helmholtz_energy_contributions()
        }
    pd.DataFrame(_rows).rename_axis("a_res / RT").round(5)
    return MIXTURES, models, styles


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cross second virial coefficients $B_{12}$

    $B_{12}$ of the EoS from the mixture virial coefficient, $B_{12} = 2 B(x_1 = 1/2) - (B_{11} + B_{22})/2$.
    The pure-component $B_{ii}$ are identical in both models, since the Santos model reduces to Carnahan-Starling (= BMCSL)
    for pure fluids.

    Reference: the exact $B_{12}$ of the unlike Feynman-Hibbs-Mie (FH1) potential, Aasen et al. (2020), Eq. (50),
    $B_{12} = -2 \pi N_\mathrm{A} \int_0^\infty \left[\exp(-\beta u_{12}(r)) - 1\right] r^2 \mathrm{d}r$,
    with the same combining rules and binary parameters as in the EoS.
    """)
    return


@app.cell
def _(VRQ_PATH, np, quad):
    import json

    _KB = 1.380649e-23
    _HBAR = 6.62607015e-34 / (2 * np.pi)
    _NA = 6.02214076e23


    def unlike_potential(components):
        """FH1-Mie potential u_12(r, T) / k_B in K (r in Angstrom) with SAFT-VRQ Mie combining rules."""
        _pure = {
            r["identifier"]["name"]: r
            for r in json.loads((VRQ_PATH / "aasen2019.json").read_text())
        }
        _binary = json.loads((VRQ_PATH / "aasen2020_binary.json").read_text())
        p1, p2 = (_pure[c] for c in components)
        k_ij = l_ij = 0.0
        for b in _binary:
            if {b["id1"]["name"], b["id2"]["name"]} == set(components):
                k_ij, l_ij = b.get("k_ij", 0.0), b.get("l_ij", 0.0)
        sigma = (1 - l_ij) * (p1["sigma"] + p2["sigma"]) / 2
        eps = (
            (1 - k_ij)
            * np.sqrt(p1["sigma"] ** 3 * p2["sigma"] ** 3)
            / sigma**3
            * np.sqrt(p1["epsilon_k"] * p2["epsilon_k"])
        )
        lr = 3 + np.sqrt((p1["lr"] - 3) * (p2["lr"] - 3))
        la = 3 + np.sqrt((p1["la"] - 3) * (p2["la"] - 3))
        c = lr / (lr - la) * (lr / la) ** (la / (lr - la))
        m_inv = (1 / p1["molarweight"] + 1 / p2["molarweight"]) * 1e3 * _NA  # 1/kg

        def u(r, T):
            D = _HBAR**2 * m_inv / (24 * _KB * T) * 1e20  # Angstrom^2
            s_r, s_a = (sigma / r) ** lr, (sigma / r) ** la
            return (
                c
                * eps
                * (
                    s_r
                    - s_a
                    + D / r**2 * (lr * (lr - 1) * s_r - la * (la - 1) * s_a)
                )
            )

        return u, sigma


    def b12_exact(components, temperatures):
        """B_12 in cm3/mol from the FH1-Mie potential."""
        u, sigma = unlike_potential(components)
        _b = []
        for T in temperatures:
            f = lambda r: (np.exp(-u(r, T) / T) - 1) * r**2
            _i = (
                quad(f, 1e-3, 2 * sigma, limit=200)[0]
                + quad(f, 2 * sigma, np.inf, limit=200)[0]
            )
            _b.append(-2 * np.pi * _i * _NA * 1e-24)
        return np.array(_b)

    return (b12_exact,)


@app.cell
def _(MIXTURES, b12_exact, models, np, pd, si):
    temperatures_virial = np.geomspace(12.0, 300.0, 40)
    _CM3_MOL = (si.CENTI * si.METER) ** 3 / si.MOL


    def b12_eos(eos, temperatures):
        """B_12 in cm3/mol from the mixture second virial coefficient."""
        _b = []
        for T in temperatures:
            _B = [
                eos.second_virial_coefficient(T * si.KELVIN, np.array([x, 1 - x]))
                / _CM3_MOL
                for x in (1.0, 0.0, 0.5)
            ]
            _b.append(2 * _B[2] - (_B[0] + _B[1]) / 2)
        return np.array(_b)


    b12 = {
        mixture: {
            "exact (FH1-Mie potential)": b12_exact(components, temperatures_virial)
        }
        for mixture, components in MIXTURES.items()
    }
    for (_mixture, _label), _eos in models.items():
        b12[_mixture][_label] = b12_eos(_eos, temperatures_virial)
    pd.DataFrame(
        {
            (mixture, label): [
                v[np.argmin(abs(temperatures_virial - T))]
                for T in (15, 30, 100, 300)
            ]
            for mixture, d in b12.items()
            for label, v in d.items()
        },
        index=[f"T = {T} K" for T in (15, 30, 100, 300)],
    ).round(3)
    return b12, temperatures_virial


@app.cell
def _(b12, plt, styles, temperatures_virial):
    _fig, _axes = plt.subplots(
        2, 2, figsize=(11, 7), sharex=True, height_ratios=[2, 1]
    )
    for _j, (_mixture, _d) in enumerate(b12.items()):
        _ref = _d["exact (FH1-Mie potential)"]
        _axes[0, _j].plot(
            temperatures_virial,
            _ref,
            "k-",
            lw=3,
            alpha=0.3,
            label="exact (FH1-Mie potential)",
        )
        for _label, _style in styles.items():
            _axes[0, _j].plot(
                temperatures_virial, _d[_label], label=_label, **_style
            )
            _axes[1, _j].plot(
                temperatures_virial, _d[_label] - _ref, label=_label, **_style
            )
        _axes[0, _j].set_title(_mixture)
        _axes[0, _j].axhline(0, color="k", lw=0.5)
        _axes[1, _j].axhline(0, color="k", lw=0.5)
        _axes[1, _j].set_xlabel("$T$ / K")
        _axes[1, _j].set_xscale("log")
    _axes[0, 0].set_ylabel(r"$B_{12}$ / cm$^3$ mol$^{-1}$")
    _axes[1, 0].set_ylabel(r"$B_{12} - B_{12}^\mathrm{exact}$ / cm$^3$ mol$^{-1}$")
    _axes[0, 0].legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At zero density, the two references differ only in the hard-sphere part of $B_{12}$: Santos gives the exact
    $\frac{2\pi}{3} N_\mathrm{A} d_{12}^3$, BMCSL + correction its linearization around $\bar d_{12} = (d_{11} + d_{22})/2$,
    $\frac{2\pi}{3} N_\mathrm{A} [\bar d_{12}^3 + 3 \bar d_{12}^2 (d_{12} - \bar d_{12})]$. The difference is second order in the
    non-additivity and therefore small (largest for He/H$_2$). The deviation from the exact $B_{12}$ of the FH1-Mie potential
    is dominated by the perturbation (dispersion) terms, not by the hard-sphere reference.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vapor-liquid equilibria

    Bubble-point curves by continuation in $x_\mathrm{He}$ up to the mixture critical point or $p_\mathrm{max}$.

    - He/H$_2$: experimental data from the DDB export (`examples/uvcs/data/helium_hydrogen_ddb_vle.csv`) and
      GEMC simulations of the FH1-Mie force field (`examples/saftvrcs/data/helium_hydrogen_gemc_vle.csv`).
    - He/Ne: model comparison only (no data in the repository yet).
    """)
    return


@app.cell
def _(EXAMPLES, pd):
    exp_he_h2 = pd.read_csv(
        EXAMPLES / "uvcs" / "data" / "helium_hydrogen_ddb_vle.csv"
    ).rename(columns={"temperature_K": "T", "pressure_bar": "p"})
    gemc_he_h2 = pd.read_csv(
        EXAMPLES / "saftvrcs" / "data" / "helium_hydrogen_gemc_vle.csv", sep=r"\s+"
    ).rename(columns={"temperature[K]": "T", "pressure[bar]": "p"})
    markers = dict(zip(sorted(exp_he_h2["reference"].unique()), "sD^v<>ph"))
    exp_he_h2.groupby("reference").agg(
        n=("p", "size"), T_min=("T", "min"), T_max=("T", "max"), p_max=("p", "max")
    )
    return exp_he_h2, gemc_he_h2, markers


@app.cell
def _(PhaseEquilibrium, np, pd, si):
    def bubble_curve(eos, T, p_max=500.0, dx0=0.01, dx_min=1e-5, x_max=0.95):
        """Bubble-point curve from x_He = 1e-3 up to the mixture critical point or p_max (continuation in x)."""
        rows = []
        x, dx = 1e-3, 1e-3
        while x < x_max and dx >= dx_min and (not rows or rows[-1][2] < p_max):
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
    temperatures_vle = {
        "He/H2": [15.5, 17.4, 23.0, 27.8, 20.4, 26.0, 29.0, 31.5],  # DDB, GEMC
        "He/Ne": [26.0, 30.0, 35.0, 40.0],
    }
    P_MAX = {"He/H2": 200.0, "He/Ne": 500.0}  # bar
    with mo.persistent_cache("vle_curves_hs_reference"):
        curves = {
            (mixture, label, T): bubble_curve(
                models[mixture, label], T, p_max=P_MAX[mixture]
            )
            for mixture, label in models
            for T in temperatures_vle[mixture]
        }
    return P_MAX, curves, temperatures_vle


@app.cell
def _(P_MAX, curves, exp_he_h2, gemc_he_h2, markers, plt, styles):
    def plot_isotherms(mixture, temps):
        """2x2 isotherms with the bubble/dew curves of both hard-sphere references and the He/H2 data."""
        fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True)
        for ax, T in zip(axes.flat, temps):
            p_top = []
            if mixture == "He/H2":
                _e = exp_he_h2[(exp_he_h2["T"] - T).abs() < 0.06]
                for ref, g in _e.groupby("reference"):
                    kw = dict(
                        ls="", marker=markers[ref], ms=4, color="C7", alpha=0.7
                    )
                    ax.plot(g["x_he"], g["p"], label=ref, **kw)
                    ax.plot(g["y_he"], g["p"], **kw)
                _g = gemc_he_h2[gemc_he_h2["T"] == T]
                ax.plot(_g["x_he"], _g["p"], "o", mfc="w", mec="k", label="GEMC")
                ax.plot(_g["y_he"], _g["p"], "o", mfc="w", mec="k")
                p_top += [
                    _e["p"].max() if len(_e) else 0,
                    _g["p"].max() if len(_g) else 0,
                ]
            for label, style in styles.items():
                df = curves[mixture, label, T]
                ax.plot(df["x_he"], df["p_bar"], label=label, **style)
                ax.plot(df["y_he"], df["p_bar"], **style)
                p_top.append(df["p_bar"].max() if len(df) else 0)
            ax.set_title(f"{mixture}, T = {T} K")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, min(1.1 * max(p_top), P_MAX[mixture]))
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
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        return fig

    return (plot_isotherms,)


@app.cell
def _(plot_isotherms, temperatures_vle):
    plot_isotherms("He/H2", temperatures_vle["He/H2"][:4])
    return


@app.cell
def _(plot_isotherms, temperatures_vle):
    plot_isotherms("He/H2", temperatures_vle["He/H2"][4:])
    return


@app.cell
def _(plot_isotherms, temperatures_vle):
    plot_isotherms("He/Ne", temperatures_vle["He/Ne"])
    return


if __name__ == "__main__":
    app.run()
