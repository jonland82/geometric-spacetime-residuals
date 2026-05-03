"""Write Markdown and MathJax HTML reports for the warp grid search."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if abs(value) >= 1000 or (0 < abs(value) < 0.001):
            return f"{value:.4e}"
        return f"{value:.4f}"
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def _display_frame(frame: pd.DataFrame, columns: list[str], limit: int) -> list[dict[str, str]]:
    rows = []
    for _, row in frame.head(limit).iterrows():
        rows.append({col: _fmt(row[col]) for col in columns if col in row})
    return rows


def markdown_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "_No rows generated._"
    columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row[col] for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def html_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "<p><em>No rows generated.</em></p>"
    columns = list(rows[0].keys())
    parts = ["<table>", "<thead><tr>"]
    for col in columns:
        parts.append(f"<th>{html.escape(col)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for col in columns:
            parts.append(f"<td>{html.escape(row[col])}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def build_tables(top: pd.DataFrame, families: pd.DataFrame, basis: pd.DataFrame) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    top_rows = _display_frame(
        top,
        [
            "family",
            "eta_mode",
            "total_score",
            "rotation_mse",
            "lensing_mse",
            "flatness_metric",
            "weak_field_max",
            "params",
        ],
        10,
    )
    family_rows = _display_frame(
        families,
        [
            "family",
            "count",
            "best_total_score",
            "best_rotation_mse",
            "best_lensing_mse",
            "best_eta_mode",
            "best_flatness_metric",
            "best_weak_field_max",
        ],
        8,
    )
    basis_rows = _display_frame(basis, ["center", "width", "weight"], 12) if not basis.empty else []
    return top_rows, family_rows, basis_rows


def build_markdown(top: pd.DataFrame, families: pd.DataFrame, basis: pd.DataFrame) -> str:
    top_rows, family_rows, basis_rows = build_tables(top, families, basis)
    best = top.iloc[0]
    best_family = str(best["family"])
    best_eta = str(best["eta_mode"])
    best_score = _fmt(float(best["total_score"]))

    return f"""# Geometry-First Grid Search over Weak-Field Spacetime Warps

**Author:** J. R. Landers  
**Date:** May 2026

## Abstract

This research note builds a toy computational experiment for exploring the missing-mass problem without starting from named dark-matter or modified-gravity theories. The experiment searches directly over parameterized weak-field metric perturbations, or spacetime warps, and asks which geometric deformations reproduce a synthetic galaxy-like rotation residual. The result is a geometry-first scaffold: candidate warps are ranked by their rotation-curve fit, weak-field consistency, smoothness, effective density behavior, and a conceptual lensing/slip proxy. The model is intentionally synthetic and radial; it is not a solution to dark matter.

## Conceptual Motivation

The missing-mass problem can be reframed as a search for geometric perturbations that transform a baryon-predicted metric into an observation-compatible metric:

\\[
g_{{\\mu\\nu}}^{{\\rm trial}}
=
g_{{\\mu\\nu}}^{{\\rm bar}}
+
h_{{\\mu\\nu}}(\\theta).
\\]

The question is not initially "which named theory is correct?" The question is more primitive: what kinds of metric perturbations numerically generate the residual, and what effective physical structure would those perturbations imply?

## Weak-Field Metric Ansatz

The prototype uses a static weak-field radial ansatz:

\\[
ds^2
=
-\\left(1+\\frac{{2\\Phi(r)}}{{c^2}}\\right)c^2dt^2
+
\\left(1-\\frac{{2\\Psi(r)}}{{c^2}}\\right)
\\left(dr^2+r^2d\\Omega^2\\right).
\\]

The baryonic metric potentials are perturbed as

\\[
\\Phi(r)=\\Phi_{{\\rm bar}}(r)+\\delta\\Phi(r;\\theta),
\\qquad
\\Psi(r)=\\Psi_{{\\rm bar}}(r)+\\delta\\Psi(r;\\theta).
\\]

Rotation curves mostly constrain

\\[
v^2(r)=r\\frac{{d\\Phi}}{{dr}},
\\qquad
\\Delta g(r)=\\frac{{d}}{{dr}}\\delta\\Phi(r),
\\]

while a lensing-like projection is sensitive to a combination closer to

\\[
L(r)\\propto \\Phi(r)+\\Psi(r).
\\]

The potential zero point is arbitrary in this toy calculation; derivatives and relative profiles carry the information.

## Warp Search Space

The grid search explores direct geometric perturbation families:

\\[
\\delta\\Phi(r)=A\\log(1+r/r_0),
\\]

\\[
\\delta\\Phi(r)=A(r/r_0)^\\alpha,
\\]

\\[
\\delta\\Phi(r)=A(1-e^{{-r/r_0}}),
\\qquad
\\delta\\Phi(r)=A\\frac{{r^n}}{{r^n+r_0^n}},
\\]

\\[
\\delta\\Phi(r)=A\\exp\\left[-\\frac{{(r-r_c)^2}}{{2\\sigma^2}}\\right],
\\]

plus a nonnegative radial basis warp,

\\[
\\frac{{d}}{{dr}}\\delta\\Phi(r)
\\approx
\\sum_k a_k\\exp\\left[-\\frac{{(r-c_k)^2}}{{2s^2}}\\right],
\\qquad
a_k\\ge 0.
\\]

The second potential is parameterized by a slip-like relation,

\\[
\\delta\\Psi(r)=\\eta(r;\\lambda)\\delta\\Phi(r),
\\]

with no-slip, constant-slip, inner-slip, and outer-slip variants.

## Grid-Search Objective

Each trial warp is scored by

\\[
\\mathcal{{J}}(\\theta)
=
\\sum_i
\\frac{{[v_{{\\rm trial}}(r_i;\\theta)-v_{{\\rm obs}}(r_i)]^2}}{{\\sigma_i^2}}
+
\\lambda_1 S_{{\\rm smooth}}
+
\\lambda_2 P_{{\\rm path}}
+
\\lambda_3 P_{{\\rm weak}}
+
\\lambda_4 C(\\theta)
+
\\lambda_5 P_{{\\rm lens}}.
\\]

The path penalty tracks negative residual acceleration, nonmonotone effective mass, negative effective density, and oscillatory behavior. Weak-field validity is tracked using \\(\\max |\\Phi|/c^2\\). The lensing term is a conceptual proxy, not a physical lensing calculation.

The effective density proxy is

\\[
\\rho_{{\\rm eff}}(r)
=
\\frac{{1}}{{4\\pi G r^2}}
\\frac{{d}}{{dr}}
\\left[
r^2\\frac{{d}}{{dr}}\\delta\\Phi(r)
\\right].
\\]

## Numerical Experiments

The baryonic cumulative mass profile is

\\[
M_{{\\rm bar}}(<r)
=
M_b\\left[1-e^{{-r/R_d}}(1+r/R_d)\\right],
\\qquad
g_{{\\rm bar}}(r)=\\frac{{GM_{{\\rm bar}}(<r)}}{{r^2}}.
\\]

The synthetic observed curve is built by adding an empirical flat-speed component,

\\[
v_{{\\rm obs}}^2(r)
=
v_{{\\rm bar}}^2(r)
+
\\left[v_f(1-e^{{-r/r_f}})\\right]^2.
\\]

This target makes the required outer residual approximately

\\[
\\Delta g_{{\\rm target}}(r)\\sim \\frac{{v_f^2}}{{r}},
\\]

which is why logarithmic potential warps are expected to be competitive.

## Results

The best-ranked trial in this run is **{best_family}** with slip mode **{best_eta}** and total score **{best_score}**.

### Rotation and Residual Fit

![Warp rotation fit](../figures/warp_rotation_fit.png)

![Warp residual acceleration](../figures/warp_residual_acceleration.png)

The best direct warp reproduces the smooth flat-curve residual without naming a physical source. The important output is not a theory label; it is the shape of \\(\\delta\\Phi\\), its derivative, and the effective density structure implied by the warp.

### Metric Potentials and Slip

![Warp potentials](../figures/warp_potentials.png)

![Warp lensing proxy](../figures/warp_lensing_proxy.png)

Rotation fixes \\(d\\Phi/dr\\), but it does not uniquely determine \\(\\Psi\\). The slip proxy demonstrates how two perturbations with similar rotation behavior can separate under a second projection.

### Effective Curvature/Density Proxy

![Warp effective profiles](../figures/warp_effective_profiles.png)

Successful long-range warps imply an extended effective density proxy. In this toy target, the outer flat rotation curve pushes the search toward perturbations whose derivative falls roughly like \\(1/r\\), corresponding to a logarithmic potential over the searched radial range.

### Parameter Regions

![Log warp heatmap](../figures/warp_log_parameter_heatmap.png)

![Warp search scatter](../figures/warp_parameter_scatter.png)

![Warp family scores](../figures/warp_family_scores.png)

The heatmap shows the region of the logarithmic family that works. The scatter plot compares fit quality and outer flatness across families.

### Basis Warp

![Warp basis components](../figures/warp_basis_components.png)

The radial basis warp is not a named physical theory; it is an agnostic function approximator for \\(d\\delta\\Phi/dr\\). It provides a useful check on whether a simple analytic warp is missing structure.

### Top-Ranked Candidates

{markdown_table(top_rows)}

### Family Summary

{markdown_table(family_rows)}

### Selected Basis Terms

{markdown_table(basis_rows)}

## Interpretation

The successful warps tend to be smooth and long-range. The flat target curve favors perturbations with \\(d\\delta\\Phi/dr\\sim 1/r\\), so the corresponding potential looks logarithmic across the outer radial range. Saturating and localized bump perturbations usually struggle because their derivatives either decay too quickly or change sign. Power-law perturbations can work when their exponent is small enough to mimic logarithmic growth. The basis warp can approximate the same behavior by combining several positive radial components.

The best candidates do not require gravitational slip for the synthetic no-slip lensing proxy used here. However, rotation alone cannot rule out slip-like \\(\\delta\\Psi\\) behavior, which is why multi-probe constraints are central in any more serious extension.

## Limitations

This is a weak-field toy model with a radial ansatz and synthetic target data. It is not a full Einstein-tensor calculation, not a real galaxy fit, and not a proof of new physics. Spherical symmetry is assumed. Disk geometry is ignored. Coordinate and gauge issues are suppressed by the chosen ansatz. The lensing/slip observable is conceptual. The effective density proxy is a diagnostic of the warp, not a confirmed matter density.

## Most Promising Next Directions

1. Fit real SPARC-like rotation curves.
2. Replace the spherical/radial ansatz with disk geometry.
3. Compute full metric-perturbation curvature diagnostics.
4. Use automatic differentiation over the metric ansatz.
5. Use symbolic regression over warp functions.
6. Enforce covariant conservation constraints.
7. Jointly fit rotation and lensing data.
8. Classify learned warp families across galaxy populations.

## Conclusion

The purpose is not to assume a named theory, but to numerically explore the space of possible weak-field geometric deformations and analyze which kinds of spacetime warps reproduce the observed residual dynamics.

## References

1. Rubin, V. C., Ford, W. K., Jr., and Thonnard, N. (1980). Extended rotation curves of spiral galaxies.
2. Lelli, F., McGaugh, S. S., and Schombert, J. M. (2016). SPARC database and disk-galaxy mass models.
3. McGaugh, S. S., Lelli, F., and Schombert, J. M. (2016). Radial acceleration relation.
4. Bertone, G., and Hooper, D. (2018). History and status of dark matter.
5. Mistele, T., McGaugh, S., Lelli, F., Schombert, J., and Li, P. (2024). Weak-lensing constraints related to extended flat circular velocities.
"""


def build_html(top: pd.DataFrame, families: pd.DataFrame, basis: pd.DataFrame) -> str:
    top_rows, family_rows, basis_rows = build_tables(top, families, basis)
    best = top.iloc[0]
    best_family = html.escape(str(best["family"]))
    best_eta = html.escape(str(best["eta_mode"]))
    best_score = html.escape(_fmt(float(best["total_score"])))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Geometry-First Grid Search over Weak-Field Spacetime Warps</title>
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
      font-size: 0.88rem;
    }}
    th, td {{
      border: 1px solid #d8dde3;
      padding: 0.45rem 0.55rem;
      vertical-align: top;
    }}
    th {{ background: #eef2f5; text-align: left; }}
    .note {{ color: #46525f; }}
  </style>
</head>
<body>
<main>
<h1>Geometry-First Grid Search over Weak-Field Spacetime Warps</h1>
<p class="meta"><strong>Author:</strong> J. R. Landers<br><strong>Date:</strong> May 2026</p>

<h2>Abstract</h2>
<p>This standalone report searches directly over weak-field spacetime warps rather than over named dark-matter or modified-gravity theories. It is a synthetic radial prototype for asking what kinds of metric perturbations reproduce a galaxy-like dynamical residual.</p>

<h2>Conceptual Motivation</h2>
\\[
g_{{\\mu\\nu}}^{{\\rm trial}}
=
g_{{\\mu\\nu}}^{{\\rm bar}}
+
h_{{\\mu\\nu}}(\\theta).
\\]
<p>The experiment asks what geometric perturbations numerically generate the residual, and what effective physical structure those perturbations imply.</p>

<h2>Weak-Field Metric Ansatz</h2>
\\[
ds^2
=
-\\left(1+\\frac{{2\\Phi(r)}}{{c^2}}\\right)c^2dt^2
+
\\left(1-\\frac{{2\\Psi(r)}}{{c^2}}\\right)
\\left(dr^2+r^2d\\Omega^2\\right).
\\]
\\[
\\Phi=\\Phi_{{\\rm bar}}+\\delta\\Phi,
\\qquad
\\Psi=\\Psi_{{\\rm bar}}+\\delta\\Psi.
\\]
<p>Rotation probes \\(d\\Phi/dr\\), while the toy lensing/slip proxy responds to \\(\\Phi+\\Psi\\).</p>

<h2>Warp Search Space</h2>
<p>The grid includes logarithmic, power-law, exponential, rational saturating, Gaussian bump, and nonnegative radial basis perturbations. Slip is parameterized as</p>
\\[
\\delta\\Psi(r)=\\eta(r;\\lambda)\\delta\\Phi(r).
\\]

<h2>Objective and Diagnostics</h2>
\\[
\\mathcal{{J}}(\\theta)
=
\\chi^2_{{\\rm rot}}
+
\\lambda_1 S_{{\\rm smooth}}
+
\\lambda_2 P_{{\\rm path}}
+
\\lambda_3 P_{{\\rm weak}}
+
\\lambda_4 C(\\theta)
+
\\lambda_5 P_{{\\rm lens}}.
\\]
\\[
\\rho_{{\\rm eff}}(r)
=
\\frac{{1}}{{4\\pi G r^2}}
\\frac{{d}}{{dr}}
\\left[
r^2\\frac{{d}}{{dr}}\\delta\\Phi(r)
\\right].
\\]

<h2>Results</h2>
<p>The best-ranked trial is <strong>{best_family}</strong> with slip mode <strong>{best_eta}</strong> and score <strong>{best_score}</strong>.</p>

<h3>Rotation and Residual Fit</h3>
<img src="../figures/warp_rotation_fit.png" alt="Warp rotation fit">
<img src="../figures/warp_residual_acceleration.png" alt="Warp residual acceleration">

<h3>Metric Potentials and Slip</h3>
<img src="../figures/warp_potentials.png" alt="Warp potentials">
<img src="../figures/warp_lensing_proxy.png" alt="Warp lensing proxy">

<h3>Effective Density Proxy</h3>
<img src="../figures/warp_effective_profiles.png" alt="Warp effective profiles">

<h3>Parameter Regions</h3>
<img src="../figures/warp_log_parameter_heatmap.png" alt="Log warp heatmap">
<img src="../figures/warp_parameter_scatter.png" alt="Warp parameter scatter">
<img src="../figures/warp_family_scores.png" alt="Warp family scores">

<h3>Basis Warp</h3>
<img src="../figures/warp_basis_components.png" alt="Warp basis components">

<h3>Top-Ranked Candidates</h3>
{html_table(top_rows)}

<h3>Family Summary</h3>
{html_table(family_rows)}

<h3>Selected Basis Terms</h3>
{html_table(basis_rows)}

<h2>Interpretation</h2>
<p>The successful warps are smooth and long-range. The flat target curve favors \\(d\\delta\\Phi/dr\\sim 1/r\\), so the corresponding perturbation is approximately logarithmic over the outer radial range. Localized bumps and rapidly saturating perturbations generally struggle or become pathologic.</p>

<h2>Limitations</h2>
<p>This is a weak-field radial toy model with synthetic data. It is not a full Einstein-tensor calculation, not a real galaxy fit, and not proof of new physics. Coordinate and gauge issues are suppressed by the ansatz, and the lensing proxy is conceptual.</p>

<h2>Most Promising Next Directions</h2>
<ol>
  <li>Fit real SPARC-like rotation curves.</li>
  <li>Use disk geometry.</li>
  <li>Compute full metric-perturbation curvature diagnostics.</li>
  <li>Use automatic differentiation over the metric ansatz.</li>
  <li>Use symbolic regression over warp functions.</li>
  <li>Enforce covariant conservation.</li>
  <li>Jointly fit rotation and lensing.</li>
  <li>Classify learned warp families across galaxies.</li>
</ol>

<h2>Conclusion</h2>
<p>The purpose is not to assume a named theory, but to numerically explore the space of possible weak-field geometric deformations and analyze which kinds of spacetime warps reproduce the observed residual dynamics.</p>

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


def write_reports(root: str | Path, top: pd.DataFrame, families: pd.DataFrame, basis: pd.DataFrame) -> tuple[Path, Path]:
    root = Path(root)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    md_path = reports / "geometric_warp_grid_search.md"
    html_path = reports / "geometric_warp_grid_search.html"
    md_path.write_text(build_markdown(top, families, basis), encoding="utf-8")
    html_path.write_text(build_html(top, families, basis), encoding="utf-8")
    return md_path, html_path
