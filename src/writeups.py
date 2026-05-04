"""Generate Markdown and standalone MathJax HTML research notes."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Iterable


def _fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, float):
        if abs(value) >= 1000 or (0 < abs(value) < 0.001):
            return f"{value:.{digits}e}"
        return f"{value:.{digits}f}"
    return str(value)


def _trim_rows(rows: list[dict[str, object]], columns: list[str], limit: int | None = None) -> list[dict[str, object]]:
    data = rows if limit is None else rows[:limit]
    return [{col: row.get(col, "") for col in columns} for row in data]


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "_No rows generated._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        vals = [str(row.get(col, "")).replace("\n", " ") for col in columns]
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def html_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "<p><em>No rows generated.</em></p>"
    parts = ["<table>", "<thead><tr>"]
    for col in columns:
        parts.append(f"<th>{html.escape(col)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for col in columns:
            parts.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _format_numeric_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    formatted = []
    for row in rows:
        out = {}
        for key, value in row.items():
            if isinstance(value, float):
                out[key] = _fmt(value)
            elif isinstance(value, (dict, list, tuple)):
                out[key] = json.dumps(value)
            else:
                out[key] = value
        formatted.append(out)
    return formatted


def _best_rows(model_rows: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, object] | None]:
    ordered = sorted(model_rows, key=lambda row: float(row["bic"]))
    best = ordered[0]
    best_single = None
    for row in ordered:
        if row["model"] != "Sparse nonnegative mixture":
            best_single = row
            break
    return best, best_single


def build_markdown(
    summary_rows: list[dict[str, object]],
    model_rows: list[dict[str, object]],
    sparse_rows: list[dict[str, object]],
) -> str:
    best, best_single = _best_rows(model_rows)
    summary_display = _format_numeric_rows(
        _trim_rows(
            summary_rows,
            [
                "component",
                "relative_rmse_delta_g",
                "corr_delta_g",
                "delta_g_positive_fraction",
                "mass_monotonic_fraction",
                "rho_nonnegative_fraction",
            ],
        )
    )
    model_display = _format_numeric_rows(
        _trim_rows(
            model_rows,
            [
                "model",
                "weighted_mse",
                "bic",
                "complexity",
                "params",
                "delta_g_positive_fraction",
                "mass_monotonic_fraction",
                "rho_nonnegative_fraction",
                "pathological_oscillations",
            ],
        )
    )
    sparse_display = _format_numeric_rows(_trim_rows(sparse_rows, ["basis", "normalized_weight", "contribution_rms"]))

    best_single_text = best_single["model"] if best_single else "not available"

    return f"""# AI Search over Geometric Residuals in the Missing-Mass Problem

**Author:** J. R. Landers  
**Date:** May 2026

## Abstract

This note turns the geometric-residual framing of the missing-mass problem into a small reproducible computational prototype. The prototype generates toy rotation-curve data, recovers the weak-field geometric residual, fits several candidate latent physical generators, compares single-source and mixed-source explanations, and adds a conceptual second observable based on gravitational slip. The goal is not to solve dark matter. It is to formalize a toy inverse-problem scaffold that can be made progressively more realistic.

## Conceptual Starting Point

The starting point is the distinction between the geometry predicted by observed baryons and the geometry implied by observations. In relativistic language, define the Einstein-tensor geometric residual

$$
\\Delta G_{{\\mu\\nu}}
=
G_{{\\mu\\nu}}[g^{{\\rm obs}}]
-
G_{{\\mu\\nu}}[g^{{\\rm bar}}].
$$

If the Einstein equation is kept fixed, this residual can be represented as an effective missing stress-energy tensor,

$$
T_{{\\mu\\nu}}^{{\\rm miss}}
=
\\frac{{c^4}}{{8\\pi G}}\\Delta G_{{\\mu\\nu}}.
$$

The phrase "effective" matters. A geometric residual can be caused by particle-like dark matter, field-like sources, modified geometry laws, baryonic modeling errors, environmental effects, or observational systematics. The prototype below treats these possibilities as candidate latent physical generators of one observed residual.

## Weak-Field Galaxy Reduction

For the toy galaxy experiments, circular speed is converted into acceleration by

$$
g(r)=\\frac{{v^2(r)}}{{r}}.
$$

The practical weak-field geometric residual is

$$
\\Delta g(r)
=
g_{{\\rm obs}}(r)-g_{{\\rm bar}}(r)
=
\\frac{{v_{{\\rm obs}}^2(r)-v_{{\\rm bar}}^2(r)}}{{r}}.
$$

The potential residual also defines an effective density proxy,

$$
\\rho_{{\\rm eff}}(r)
=
\\frac{{1}}{{4\\pi G}}\\nabla^2\\Delta\\Phi.
$$

Using a spherical toy proxy,

$$
g(r) = \\frac{{G M(<r)}}{{r^2}},
\\qquad
M_{{\\rm eff}}(<r)
=
\\frac{{r^2\\Delta g(r)}}{{G}},
$$

and

$$
\\rho_{{\\rm eff}}(r)
=
\\frac{{1}}{{4\\pi r^2}}\\frac{{dM_{{\\rm eff}}}}{{dr}}.
$$

This is not a realistic disk-galaxy inversion. It is a controlled spherical proxy for testing how an acceleration residual can be recovered and decomposed.

## AI-Assisted Inverse Problem

The inverse problem is to infer one or more latent generators whose induced residual matches observations:

$$
\\Delta G_{{\\mu\\nu}}^{{\\rm obs}}
\\approx
\\sum_k \\Delta G_{{\\mu\\nu}}^{{(k)}}.
$$

In this toy reduction, that becomes a search over residual acceleration functions:

$$
\\Delta g(r) \\approx \\sum_k w_k f_k(r,g_{{\\rm bar}},\\rho_{{\\rm bar}},\\nabla\\rho_{{\\rm bar}},\\ldots).
$$

The basis-library fit in this prototype is deliberately modest: it uses nonnegative sparse combinations of NFW-like, cored/isothermal-like, MOND-like, soliton-like, and baryon-coupled radial functions. It is not physics discovery. It is a scaffold for later symbolic regression or differentiable search over covariant structures such as

$$
\\Delta \\mathcal{{E}}_{{\\mu\\nu}}
=
F_{{\\mu\\nu}}
\\left(
g_{{\\mu\\nu}},
R_{{\\mu\\nu}},
R,
T_{{\\mu\\nu}}^{{\\rm bar}},
\\nabla_\\alpha T_{{\\mu\\nu}}^{{\\rm bar}},
\\phi,
A_\\mu,
\\ldots
\\right).
$$

## Experimental Design

The synthetic baryonic profile is

$$
M_{{\\rm bar}}(<r) = M_b\\left[1-e^{{-r/R_d}}(1+r/R_d)\\right],
$$

with baryonic acceleration

$$
g_{{\\rm bar}}(r)=\\frac{{G M_{{\\rm bar}}(<r)}}{{r^2}}.
$$

Five hidden/residual generators are used: an NFW-like halo, a cored/isothermal-like halo, a MOND-like acceleration residual, a soliton/cored scalar-field-inspired component, and a mixed NFW+MOND component. Gaussian velocity noise is added with a fixed random seed.

The inversion stage recovers $\\Delta g$, $M_{{\\rm eff}}$, and $\\rho_{{\\rm eff}}$ from noisy $v_{{\\rm obs}}(r)$ and known $g_{{\\rm bar}}(r)$. The model-comparison stage fits candidate generators and reports weighted MSE, AIC/BIC-like scores, and simple physical validity checks: positivity of $\\Delta g$, monotonicity of $M_{{\\rm eff}}$, nonnegativity of $\\rho_{{\\rm eff}}$, and absence of strong oscillatory pathologies.

The multi-probe experiment adds a conceptual lensing/slip proxy. Dynamics probes the potential $\\Phi$, while lensing responds to $\\Phi+\\Psi$. A particle-like model is assigned $\\eta=\\Psi/\\Phi\\approx 1$, while a toy modified-gravity twin is assigned $\\eta(r)\\ne 1$. This is not physical lensing. It only demonstrates how a second projection can break a rotation-curve degeneracy.

## Results

### Residual Recovery

![Synthetic rotation curve](../figures/rotation_curve.png)

![Acceleration residual](../figures/acceleration_residual.png)

![Effective mass and density](../figures/effective_mass_density.png)

The synthetic inversion recovers the broad geometric residual in all generated cases. Noise is amplified when converting velocity to acceleration and especially when differentiating $M_{{\\rm eff}}$ to estimate $\\rho_{{\\rm eff}}$, which is why the density proxy is the least stable derived quantity.

{markdown_table(summary_display, list(summary_display[0].keys()) if summary_display else [])}

### Candidate Generator Degeneracy

![Model comparison](../figures/model_comparison.png)

The mixed-source example illustrates the core inverse-problem issue: several candidate generators can produce broadly similar rotation residuals. The best overall BIC-like fit in this run is **{best['model']}**, while the best single-family fit is **{best_single_text}**. The sparse mixture is allowed to represent the missing residual as a superposition of latent mechanisms, so it can improve the fit when the truth is mixed.

{markdown_table(model_display, list(model_display[0].keys()) if model_display else [])}

![Mixed source fit](../figures/mixed_source_fit.png)

This supports the interpretation that the residual decomposition is itself another inverse problem. The missing residual need not be one source; it may be a superposition of particle-like, field-like, baryon-coupled, modified-geometry, and systematic contributions.

### Multi-Probe Toy Constraint

![Lensing proxy degeneracy](../figures/lensing_proxy_degeneracy.png)

The top panel uses two nearly identical rotation residuals. The lower panels assign different slip behavior and therefore different lensing proxies. This demonstrates the logic of joint constraints without pretending to compute real lensing observables.

### Sparse Basis Search

![Sparse basis search](../figures/sparse_basis_search.png)

The sparse basis fit selected the following terms:

{markdown_table(sparse_display, list(sparse_display[0].keys()) if sparse_display else [])}

This is an AI/symbolic-regression-inspired scaffold: the basis functions are hand supplied, and the search is a sparse nonnegative linear fit. A more serious version would learn across many galaxies and search over physically constrained residual structures rather than only radial profile shapes.

## Most Promising Directions

1. Replace synthetic curves with real rotation-curve data, for example SPARC-like data.
2. Use disk geometry rather than the spherical proxy used here.
3. Jointly fit rotation, lensing, stellar kinematics, gas morphology, and environmental observables.
4. Learn residual representations across galaxy populations rather than one galaxy at a time.
5. Search over physically constrained Lagrangians or field equations instead of only profile templates.
6. Enforce conservation constraints such as

$$
\\nabla^\\mu T_{{\\mu\\nu}}^{{\\rm eff}}=0.
$$

7. Add priors for positivity, stability, monotonicity, and cosmological consistency.
8. Use symbolic regression, differentiable programming, and neural operators to propose candidate field equations or effective stress-energy structures.

## Limitations

This prototype uses toy units and synthetic data. The spherical approximation is not a disk-galaxy inversion. No real galaxy data are fit. No true general-relativistic metric reconstruction is attempted. The lensing proxy is conceptual, not physical lensing. The model families are simple radial templates, not full particle, field, or modified-gravity theories. The sparse basis search is not true discovery of new physics. Finally, the inverse problem is fundamentally degenerate: rotation curves alone do not uniquely identify the latent physical generator of a geometric residual.

## Conclusion

The missing-mass problem can be studied as an inverse problem over geometric residuals: infer the simplest physically valid generator, or mixture of generators, whose induced curvature residual matches observations across probes.

## References

1. Rubin, V. C., Ford, W. K., Jr., and Thonnard, N. (1980). Extended rotation curves of spiral galaxies.
2. Lelli, F., McGaugh, S. S., and Schombert, J. M. (2016). SPARC database and mass models for disk galaxies.
3. McGaugh, S. S., Lelli, F., and Schombert, J. M. (2016). Radial acceleration relation in rotationally supported galaxies.
4. Bertone, G., and Hooper, D. (2018). History and status of dark matter.
5. Mistele, T., McGaugh, S., Lelli, F., Schombert, J., and Li, P. (2024). Weak-lensing constraints related to extended flat circular velocities.
"""


def build_html(
    summary_rows: list[dict[str, object]],
    model_rows: list[dict[str, object]],
    sparse_rows: list[dict[str, object]],
) -> str:
    best, best_single = _best_rows(model_rows)
    summary_display = _format_numeric_rows(
        _trim_rows(
            summary_rows,
            [
                "component",
                "relative_rmse_delta_g",
                "corr_delta_g",
                "delta_g_positive_fraction",
                "mass_monotonic_fraction",
                "rho_nonnegative_fraction",
            ],
        )
    )
    model_display = _format_numeric_rows(
        _trim_rows(
            model_rows,
            [
                "model",
                "weighted_mse",
                "bic",
                "complexity",
                "params",
                "delta_g_positive_fraction",
                "mass_monotonic_fraction",
                "rho_nonnegative_fraction",
                "pathological_oscillations",
            ],
        )
    )
    sparse_display = _format_numeric_rows(_trim_rows(sparse_rows, ["basis", "normalized_weight", "contribution_rms"]))
    best_single_text = html.escape(str(best_single["model"] if best_single else "not available"))
    best_text = html.escape(str(best["model"]))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Search over Geometric Residuals in the Missing-Mass Problem</title>
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <style>
    body {{
      margin: 0;
      background: #f6f7f8;
      color: #1f252b;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.58;
    }}
    main {{
      max-width: 900px;
      margin: 0 auto;
      padding: 42px 22px 72px;
      background: #ffffff;
    }}
    h1, h2, h3 {{ line-height: 1.2; color: #111820; }}
    h1 {{ margin-bottom: 0.2rem; }}
    .meta {{ color: #52606d; margin-bottom: 2rem; }}
    img {{
      max-width: 100%;
      height: auto;
      border: 1px solid #d8dde3;
      margin: 0.75rem 0 1.25rem;
      background: #fff;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 1rem 0 1.5rem;
      font-size: 0.92rem;
    }}
    th, td {{
      border: 1px solid #d8dde3;
      padding: 0.45rem 0.55rem;
      vertical-align: top;
    }}
    th {{ background: #eef2f5; text-align: left; }}
    code {{ background: #eef2f5; padding: 0.1rem 0.25rem; }}
    .note {{ color: #46525f; }}
  </style>
</head>
<body>
<main>
<h1>AI Search over Geometric Residuals in the Missing-Mass Problem</h1>
<p class="meta"><strong>Author:</strong> J. R. Landers<br><strong>Date:</strong> May 2026</p>

<h2>Abstract</h2>
<p>This standalone note turns the geometric-residual framing of the missing-mass problem into a small reproducible computational prototype. It generates toy rotation curves, recovers the weak-field geometric residual, compares candidate latent physical generators, and demonstrates why multi-probe constraints matter. It is explicitly a toy framework, not a solution to dark matter.</p>

<h2>Conceptual Starting Point</h2>
<p>The central object is the difference between the geometry implied by observations and the geometry predicted by visible matter:</p>
\\[
\\Delta G_{{\\mu\\nu}}
=
G_{{\\mu\\nu}}[g^{{\\rm obs}}]
-
G_{{\\mu\\nu}}[g^{{\\rm bar}}].
\\]
<p>If Einstein's equation is retained, the same geometric residual can be represented as an effective missing stress-energy tensor:</p>
\\[
T_{{\\mu\\nu}}^{{\\rm miss}}
=
\\frac{{c^4}}{{8\\pi G}}\\Delta G_{{\\mu\\nu}}.
\\]
<p>This effective object may correspond to particle-like dark matter, field-like sources, modified geometry laws, baryonic errors, environmental effects, systematics, or mixtures of these.</p>

<h2>Weak-Field Galaxy Reduction</h2>
<p>For toy rotation curves, the practical residual is</p>
\\[
\\Delta g(r)
=
g_{{\\rm obs}}(r)-g_{{\\rm bar}}(r)
=
\\frac{{v_{{\\rm obs}}^2(r)-v_{{\\rm bar}}^2(r)}}{{r}}.
\\]
<p>The potential residual defines the effective density proxy</p>
\\[
\\rho_{{\\rm eff}}(r)=
\\frac{{1}}{{4\\pi G}}\\nabla^2\\Delta\\Phi.
\\]
<p>Under a spherical proxy,</p>
\\[
M_{{\\rm eff}}(<r)=\\frac{{r^2\\Delta g(r)}}{{G}},
\\qquad
\\rho_{{\\rm eff}}(r)=
\\frac{{1}}{{4\\pi r^2}}\\frac{{dM_{{\\rm eff}}}}{{dr}}.
\\]
<p class="note">This spherical inversion is intentionally simplified and is not a realistic disk-galaxy mass reconstruction.</p>

<h2>AI-Assisted Inverse Problem</h2>
<p>The prototype frames the residual as a latent-generator decomposition:</p>
\\[
\\Delta G_{{\\mu\\nu}}^{{\\rm obs}}
\\approx
\\sum_k \\Delta G_{{\\mu\\nu}}^{{(k)}}.
\\]
<p>In the toy radial setting, the search is</p>
\\[
\\Delta g(r) \\approx \\sum_k w_k f_k(r,g_{{\\rm bar}},\\rho_{{\\rm bar}},\\nabla\\rho_{{\\rm bar}},\\ldots).
\\]
<p>A future covariant search could instead target structures like</p>
\\[
\\Delta \\mathcal{{E}}_{{\\mu\\nu}}
=
F_{{\\mu\\nu}}
\\left(
g_{{\\mu\\nu}},R_{{\\mu\\nu}},R,T_{{\\mu\\nu}}^{{\\rm bar}},
\\nabla_\\alpha T_{{\\mu\\nu}}^{{\\rm bar}},\\phi,A_\\mu,\\ldots
\\right).
\\]

<h2>Experimental Design</h2>
<p>The synthetic baryonic mass profile is</p>
\\[
M_{{\\rm bar}}(<r) = M_b\\left[1-e^{{-r/R_d}}(1+r/R_d)\\right],
\\quad
g_{{\\rm bar}}(r)=\\frac{{G M_{{\\rm bar}}(<r)}}{{r^2}}.
\\]
<p>The hidden-source library contains NFW-like, cored/isothermal-like, MOND-like, soliton/cored scalar-inspired, and mixed residual profiles. Model comparison uses weighted MSE and AIC/BIC-like scores. Physical validity checks track positivity, monotonic effective mass, nonnegative density proxy, and oscillatory behavior. A conceptual lensing/slip proxy is added to show how another observable projection can break a rotation-only degeneracy.</p>

<h2>Results</h2>
<h3>Residual Recovery</h3>
<img src="../figures/rotation_curve.png" alt="Synthetic rotation curve">
<img src="../figures/acceleration_residual.png" alt="Acceleration residual">
<img src="../figures/effective_mass_density.png" alt="Effective mass and density">
<p>The broad geometric residual is recovered in all toy cases. Density recovery is noisier because it differentiates the inferred cumulative effective mass.</p>
{html_table(summary_display, list(summary_display[0].keys()) if summary_display else [])}

<h3>Candidate Generator Degeneracy</h3>
<img src="../figures/model_comparison.png" alt="Model comparison">
<p>The best overall BIC-like fit is <strong>{best_text}</strong>. The best single-family fit is <strong>{best_single_text}</strong>. The mixed example illustrates another inverse problem: the missing residual can be a superposition of latent mechanisms.</p>
{html_table(model_display, list(model_display[0].keys()) if model_display else [])}
<img src="../figures/mixed_source_fit.png" alt="Mixed source fit">

<h3>Multi-Probe Toy Constraint</h3>
<img src="../figures/lensing_proxy_degeneracy.png" alt="Lensing proxy degeneracy">
<p>Two nearly identical rotation residuals are assigned different gravitational slip behavior. The resulting lensing proxies differ, demonstrating the conceptual value of multi-probe constraints. This is not a physical lensing calculation.</p>

<h3>Sparse Basis Search</h3>
<img src="../figures/sparse_basis_search.png" alt="Sparse basis search">
{html_table(sparse_display, list(sparse_display[0].keys()) if sparse_display else [])}

<h2>Most Promising Directions</h2>
<ol>
  <li>Use real rotation-curve data, for example SPARC-like data.</li>
  <li>Replace the spherical proxy with disk geometry.</li>
  <li>Jointly fit rotation and lensing observables.</li>
  <li>Learn residual representations across galaxy populations.</li>
  <li>Search over physically constrained Lagrangians and field equations.</li>
  <li>Enforce conservation, \\(\\nabla^\\mu T_{{\\mu\\nu}}^{{\\rm eff}}=0\\).</li>
  <li>Add priors for positivity, stability, monotonicity, and cosmological consistency.</li>
  <li>Use symbolic regression, differentiable programming, and neural operators to propose residual field structures.</li>
</ol>

<h2>Limitations</h2>
<p>The prototype uses toy units, synthetic data, and a spherical approximation. It fits no real galaxy data, performs no true GR metric reconstruction, and does not discover particles. The lensing proxy is conceptual, not physical lensing. The sparse basis search is only a scaffold. The inverse problem remains degenerate.</p>

<h2>Conclusion</h2>
<p>The missing-mass problem can be studied as an inverse problem over geometric residuals: infer the simplest physically valid generator, or mixture of generators, whose induced curvature residual matches observations across probes.</p>

<h2>References</h2>
<ol>
  <li>Rubin, V. C., Ford, W. K., Jr., and Thonnard, N. (1980). Extended rotation curves of spiral galaxies.</li>
  <li>Lelli, F., McGaugh, S. S., and Schombert, J. M. (2016). SPARC database and disk-galaxy mass models.</li>
  <li>McGaugh, S. S., Lelli, F., and Schombert, J. M. (2016). Radial acceleration relation.</li>
  <li>Bertone, G., and Hooper, D. (2018). History and status of dark matter.</li>
  <li>Mistele, T., McGaugh, S., Lelli, F., Schombert, J., and Li, P. (2024). Weak-lensing constraints related to extended flat circular velocities.</li>
</ol>
</main>
</body>
</html>
"""


def write_research_notes(
    root: str | Path,
    summary_rows: list[dict[str, object]],
    model_rows: list[dict[str, object]],
    sparse_rows: list[dict[str, object]],
) -> tuple[Path, Path]:
    root = Path(root)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    md_path = reports / "geometric_residual_ai_experiment.md"
    html_path = reports / "geometric_residual_ai_experiment.html"
    md_path.write_text(build_markdown(summary_rows, model_rows, sparse_rows), encoding="utf-8")
    html_path.write_text(build_html(summary_rows, model_rows, sparse_rows), encoding="utf-8")
    return md_path, html_path


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
