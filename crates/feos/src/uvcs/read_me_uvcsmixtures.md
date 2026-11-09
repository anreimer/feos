# uv-CS mixtures: changes of 2026-11-03

Changes to the mixture expressions of the uv-CS theory (`crates/feos/src/uvcs`),
following the open issues in *Defining Mixture Expressions for the uv-Theory*
(`POSTDOC/uv_mixtures/main.pdf`, version of 21.10.2026). Equation numbers below refer
to that document.

| # | Topic | Before | Now |
|---|---|---|---|
| 1 | Exponent combining rule (Eq. 6) | `rep_ij` geometric or SAFT-VR Mie (option), `att_ij` geometric | `rep_ij`, `att_ij` arithmetic means, no option |
| 2 | Hard-sphere diameter `d_ij` | always additive | option `non_additive_hs` (default: additive) |
| 3 | `r*_m,ij` and `tau_ij` (Eq. 19, 21) | `tau_ij` from the mean of the pure-component `r*_m,ii` | `tau_ij = r*_m,ij - d*_ij` from the pair exponents |
| 4 | One-fluid `d*_x` (Eq. 36, 38) | Eq. 38: `sum_i x_i d_ii^3 / sum_i x_i sigma_ii^3` | Eq. 36: `sum_i x_i (d_ii / sigma_ii)^3` |
| 5 | One-fluid `nu_x`, `n_x` (Eq. 37) | double sums | same result, cheaper evaluation |
| 6 | Unit tests | 5 tests failed | all 21 uvcs tests pass |

Items 1, 3 and 4 change results for mixtures. Item 2 changes results only if switched on.
Pure fluids are not affected by any of the changes.

---

## 1. Arithmetic combining rule for the exponents

```
rep_ij = (rep_ii + rep_jj) / 2
att_ij = (att_ii + att_jj) / 2        (att = 6 for all uv-CS components)
```

as chosen in Eq. 6 and Eq. 73 of the write-up. In the `effective` route the mean is
taken over the effective (temperature-dependent) exponents of the pure components, in
the `classical` route over the classical exponents before the pair is mapped at `m_ij`.

The option for the repulsive exponent is removed:

- Rust: enum `RepCombiningRule`, the field `UVCSOptions::rep_combining_rule` and the
  corresponding argument of `UVCSPars::with_combining_rules` (now 4 arguments) are gone.
- Python: the keyword `rep_combining_rule` of `EquationOfState.uvcstheory` is gone
  (passing it raises a `TypeError`).

## 1a. Options reduced to combining route and hard spheres (2026-11-09)

Only two mixture options are left:

| option (Rust `UVCSOptions` / Python `uvcstheory`) | values | default |
|---|---|---|
| `combining_rule` | `classical`, `effective` | **`classical`** |
| `non_additive_hs` | `false`, `true` | **`false`** (additive `d_ij`) |

Both routes use `rep_ij = (rep_i + rep_j)/2`, `sigma_ij = (sigma_i + sigma_j)/2 (1 - l_ij)`,
`eps_ij = sqrt(eps_i eps_j) (1 - k_ij)`. `classical` combines the classical parameters and
maps the pair at `m_ij = 2 m_i m_j / (m_i + m_j)` with `zeta_ij = (zeta_i + zeta_j)/2`;
`effective` combines the effective pure-component parameters, each at its own mass.

Removed: `EpsilonCombiningRule` (SAFT-VR Mie energy rule; geometric is now fixed),
`EffectiveMassRule` (evaluation of the pure components at the pair mass), the
corresponding fields of `UVCSOptions`/`UVCSPars` and Python keywords
(`epsilon_combining_rule`, `effective_mass_rule`). `UVCSPars::with_combining_rules` is now
`UVCSPars::with_combining_rule(parameters, combining_rule)`. The default route changed from
`effective` to `classical`, so default mixture results change.

## 2. Non-additive hard spheres (option)

New option `non_additive_hs` (Rust: `UVCSOptions::non_additive_hs`, Python:
`uvcstheory(..., non_additive_hs=False)`), analogous to SAFT-VRQ Mie (BMCSL + first-order
correction; the Santos hard-sphere mixture is **not** used). Only active for mixtures.

- **Unlike diameter.** With the option, the unlike WCA diameter `d_ij` is evaluated from
  the unlike pair potential (`rep_ij`, `att_ij`, `sigma_ij`, `eps_k_ij`) with the same
  WCA correlation as the pure-component diameters. Without the option,
  `d_ij = (d_ii + d_jj) / 2`. Function: `diameter_wca_ij` in `eos/hard_sphere.rs`.
- **Where `d_ij` enters.** Everywhere the additive mean appeared before: the
  `(r_m,ij^3 - d_ij^3)` term of `Delta a_0` (Eq. 26) and `tau_ij` (Eq. 19).
- **Hard-sphere term.** The BMCSL term keeps the pure-component `d_ii`. With the option, a
  first-order correction is added (new file `eos/non_additive_hs.rs`, contribution
  `"Non additive Hard Sphere (WCA)"`):

  ```
  a / V = -2 pi sum_ij rho_i rho_j d_add,ij^2 g_hs,ij (d_add,ij - d_ij),   d_add,ij = (d_ii + d_jj) / 2
  ```

  with the BMCSL contact value `g_hs,ij` of the additive mixture.
- **Not changed.** The packing fraction `eta` and the one-fluid `d*_x` use the
  pure-component `d_ii` only.

Default is `false`, so existing results are unchanged unless the option is switched on.

**Known limitation.** Because the correction is first order in `(d_ij - d_add,ij)`, the
second virial coefficient is no longer exactly `B_20 = 2 pi / 3 sum_ij x_i x_j q_ij^3`
(Eq. 28): the error is second order, e.g. 2e-6 (relative, `B_12`) for `l_ij = 0` and
1.6e-3 for `l_ij = 0.02`. SAFT-VRQ Mie has the same limitation.

## 3. `r*_m,ij` from the pair exponents everywhere (`tau_ij`)

`r*_m,ij` is never the mean of the pure-component values but always the property of the
pair potential, `r*_m,ij = (rep_ij / att_ij)^(1 / (rep_ij - att_ij))`.
This was already the case for the explicit `r_m,ij` in `Delta a_0` and `Delta B_2`. The only
remaining place with pure-component values was the length scale in `eta_A,ij` and
`eta_B,ij`, which now follows Eq. 19 instead of Eq. 21:

```
before:  tau_ij = ((r*_m,ii sigma_ii - d_ii) + (r*_m,jj sigma_jj - d_jj)) / (2 sigma_ij)
now:     tau_ij = r*_m,ij - d_ij / sigma_ij
```

(`dimensionless_length_scale` in `eos/hard_sphere.rs`). Both are identical if all
components have the same exponents. The commented-out arithmetic-mean `rs_ij` in
`eos/reference_perturbation.rs` was removed.

`tau_ij` (with `d_ij`) is now passed to `packing_fraction_a`/`packing_fraction_b`, and
`reference_perturbation_helmholtz_energy_density` takes the matrix `d_ij` as an
additional argument.

## 4. One-fluid hard-sphere diameter `d*_x` (Eq. 36)

```
before (Eq. 38):  d*_x^3 = sum_i x_i d_ii^3 / sum_i x_i sigma_ii^3
now    (Eq. 36):  d*_x^3 = sum_i x_i (d_ii / sigma_ii)^3
```

Both agree if all `sigma_ii` are equal. `d*_x` is only used dimensionless, in
`tau_x = r*_m,x - d*_x` of the correlation integral `I^u` (Eq. 35), so no dimensional
one-fluid diameter is needed. It affects `Delta a_1^u` at finite density only; `B_2` and the
u-fraction term are unchanged.

Example (test mixture `sigma = (1, 2)`, `eps = (1, 0.5)`, `x = (0.4, 0.6)`, `T* = 1.5`):
`d*_x` = 0.95197 (Eq. 38) → 0.96284 (Eq. 36), both checked by hand with the WCA correlation.

## 5. One-fluid exponents `nu_x`, `n_x` (efficiency only)

`nu_x = sum_ij x_i x_j nu_ij` stays the exact quadratic form. It is **not** replaced by
`sum_i x_i nu_ii`, because the two differ for `combining_rule="classical"` (there `nu_ij`
is evaluated at the pair mass `m_ij`).

- The double loop over pairs in `one_fluid_properties` runs over `j >= i` only (unlike
  pairs counted twice), which also applies to `sum x_i x_j sigma_ij^3` and `epsilon_x`.
- `n_x = (sum_i x_i att_i) (sum_j x_j)`, which equals the double sum for arithmetic
  `att_ij` also for mole fractions that do not sum to one (feos-core evaluates such
  states in derivatives with respect to single `x_i`).

## 6. Unit tests

The five tests that failed before were broken test setups, not model errors:

- the pure tests passed moles (`dvector![2.0]`) where `StateHD::new` expects mole fractions;
- the mixture tests divided by the total number of moles instead of the density
  (factors 1/0.9 and 10).

With the setup of the upstream `uvtheory` tests (normalized mole fractions, molar volume
`1/rho`, per-particle values divided by the density) they reproduce the upstream reference
values. The two tests with `sigma = (1, 2)` have new reference values because of item 4 (the
upstream values are kept in comments). New tests: arithmetic exponents, pair `tau_ij`,
non-additive hard spheres (sign/linearity, identical components, He/H2).

## Verification

- **Exact pair second virial coefficient** (classical limit, additive hard spheres): `B_12` of
  the mixture equals `B_2` of a pure Mie fluid with the pair parameters to 1e-16, also with
  `k_ij`, `l_ij` ≠ 0; `B_2(x)` is quadratic in `x` to machine precision.
- **Classical route, quantum regime** (He/H2, 15–50 K): `B_12` equals `B_2` of the pure pair
  fluid mapped at `m_ij` to 1e-14.
- **Identical components** (quantum, `nu = 10.3`, 25 K, up to 40 kmol/m³): a binary of two
  identical components reproduces the pure fluid to < 1e-15 for both combining routes, with
  and without non-additive hard spheres.

## Still open

### Model decisions (change results)

**A. One-fluid `sigma_x` (Eq. 36, 39, 40).**
Currently `sigma_x^3 = sum_i x_i sigma_ii^3` (Eq. 36, pure-component values, no `l_ij`),
while `epsilon_x` and the prefactor of `Delta a_1^u` and `Delta B_21^u` are weighted with the
vdW1 sum `sum_ij x_i x_j sigma_ij^3`. The two definitions of the mixture size are therefore
inconsistent as soon as `sigma_ii` differ or `l_ij` ≠ 0.
`sigma_x` enters the one-fluid density `rho*_x = rho sigma_x^3` in `I^u` (Eq. 35) and in the
u-fraction `phi^u` (Eq. 50). Options:
- keep Eq. 36;
- vdW1, `sigma_x^3 = sum_ij x_i x_j sigma_ij^3` (Eq. 39), consistent with `epsilon_x` and
  includes `l_ij`;
- linear, `sigma_x = sum_i x_i sigma_ii` (Eq. 40).

Note: `d*_x` (Eq. 36, item 4) is reduced with the pure `sigma_ii`, not with `sigma_x`, so it
does not change with this decision.

**B. `Delta a_1^u` without the one-fluid approximation (Eq. 46–48).**
Currently one correlation integral `I^u(rho*_x, T*_x, nu_x, d*_x)` of an effective pure fluid
for all pairs (Eq. 30). Alternative: one integral per pair,
`I^u(rho*_ij, T*_ij, nu_ij, d*_ij)` with pair quantities `q*_ij`, `r*_m,ij`, `alpha_ij`,
`C_k(nu_ij, tau_ij)`, analogous to `Delta a_0`. Then `Delta B_21^u` splits naturally into pair
contributions (Eq. 48). To be decided: the pair density,
- `rho*_ij = rho sigma_ij^3`, or
- `rho*_ij = 6 eta / (pi d*_ij^3)` (pure fluid of pair `ij` at the packing fraction of the
  mixture, consistent with `eta_A,ij`, `eta_B,ij`).

Both reduce to the pure fluid for identical components. Larger structural change;
would make the decisions on `sigma_x`, `d*_x` and `nu_x` obsolete for `Delta a_1^u`.

**C. Pair-specific u-fractions (Section 3.4.1).**
Currently one u-fraction `phi^u(rho*_x, nu_x)` for the whole mixture. Alternative: pair form
`Delta a^phi = rho sum_ij x_i x_j (1 - phi^u_ij)(Delta B_2,ij - Delta B_21,ij^u)` (Eq. 61) with
`phi^u_ij` from
1. the one-fluid approximation (identical to now),
2. arithmetic mean of the pure values,
3. geometric mean,
4. geometric mean of `1 - phi^u`,
5. pair inputs `phi^u(T*_ij, rho*_ij, nu_ij)` (natural together with B).

For options 2–4 the pure-component values need a density: option A, `rho*_ii = rho sigma_ii^3`,
or option B, the one-fluid `rho*_x`. Depends on B (pair split of `Delta B_21^u`, pair density).

**D. Second virial coefficient with non-additive hard spheres.**
With `non_additive_hs = true`, `B_20` is exact only to first order in `d_ij - d_add,ij`
(1.6e-3 relative error in `B_12` for `l_ij = 0.02`). Options: accept (as in SAFT-VRQ Mie), or
make the hard-sphere `B_2` exact, e.g. by correcting the `B_2` of the BMCSL term with the
non-additive `d_ij`. Only relevant if the option is used.

Suggested order: A, then B, then C (C builds on the pair densities of B); D independently.

### Robustness (no change of normal results)

- `compute_max_density` uses the classical `sigma` instead of the effective one. For quantum
  fluids `sigma_eff > sigma`, so the packing fraction at the maximum density can exceed
  `max_eta`. Not specific to mixtures.
- The one-fluid rules use `state.molefracs` as given. feos-core always normalizes them for
  the Helmholtz energy, pressure, chemical potentials and phase equilibria. Only
  `chemical_potential_contributions` passes unnormalized mole fractions; that function does
  not add up to the total chemical potential for other models either (checked with PeTS), so
  it is a feos-core issue. Normalizing `x` inside uv-CS would be a one-line change.

### Housekeeping

- `examples/uvcs/uvcs_combination_rules.ipynb` updated: the mixture section now compares
  only the epsilon rules (geometric vs. SAFT-VR Mie) with arithmetic exponents; stored
  outputs are stale until the notebook is re-run.
- Write-up (`main.typ`): the boxes at Eq. 6 (geometric exponents), Eq. 21 (`tau_ij` from
  pure values), Eq. 22 (mean `r*_m,ij`) and Eq. 38 (`d*_x`) no longer describe uv-CS; the
  box at Eq. 37 is now identical to the linear rule of Eq. 41; Section 4.2 (non-additive
  hard spheres) is empty and could take the content of item 2.
- None of the changes of 2026-11-03 are committed yet.

## Changed files

| File | Change |
|---|---|
| `uvcs/parameters.rs` | `RepCombiningRule`, `EpsilonCombiningRule`, `EffectiveMassRule` removed; default `CombiningRule::Classical` |
| `uvcs/corresponding_states.rs` | arithmetic `rep_ij`, `att_ij`; geometric `eps_ij` only |
| `uvcs/eos/mod.rs` | option `non_additive_hs`, `d_ij` passed to the reference perturbation |
| `uvcs/eos/hard_sphere.rs` | `diameter_wca_ij`, pair `tau_ij` |
| `uvcs/eos/reference_perturbation.rs` | uses `d_ij` matrix, test setup fixed |
| `uvcs/eos/attractive_perturbation.rs` | `d*_x` from Eq. 36, cheaper one-fluid sums, test setup fixed |
| `uvcs/eos/non_additive_hs.rs` | new: non-additive hard-sphere correction |
| `uvcs/mod.rs` | exports |
| `py-feos/src/eos/uvcs.rs` | `non_additive_hs` added; `rep_combining_rule`, `epsilon_combining_rule`, `effective_mass_rule` removed; default `combining_rule="classical"` |
