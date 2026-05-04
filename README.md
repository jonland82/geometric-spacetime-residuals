# Geometric Spacetime Residuals

**Project site:** <https://jonland82.github.io/geometric-spacetime-residuals/>

This repository contains small reproducible experiments and short papers for a
geometric-residual framing of the missing-mass problem. The central object is
the mismatch between the geometry implied by observations and the geometry
predicted from observed baryons.

## Context and Contributions

Galaxy rotation curves define an inverse problem: observed baryons predict one
weak-field acceleration profile, while the measured dynamics imply another.
Rather than treating the mismatch only as a parameter-fitting residual for a
single named model, this repo studies the leftover field itself. The guiding
question is whether geometric residuals are structured, compressible, and
learnable across a population of galaxies.

The work develops three connected ideas:

- A **geometric residual** view of missing mass, where
  $\Delta g(R) = [V_{\rm obs}^2(R)-V_{\rm bar}^2(R)]/R$ is treated as an
  inferred field rather than just an error term.
- A **residual-of-residual** diagnostic: fit a physical baseline such as NFW,
  subtract it, and analyze the coherent structure that remains.
- A **statistical-learning layer** for real SPARC galaxies, using masked smooth
  dictionary learning to find stable population modes in those leftover fields.

The main current result is the SPARC residual dictionary-learning paper in
`papers/sparc_residual_dictionary_learning/`. It shows that residual profiles
from 131 primary-quality SPARC galaxies are strongly compressible, that the
first signed residual-of-residual modes are bootstrap-stable, and that learned
coordinates align with independent central-overshoot diagnostics. The claim is
model criticism, not theory selection: fit the baseline, learn what remains,
and test whether the leftover geometry is coherent.

## Repository Layout

- `src/`: shared Python modules for toy profiles, weak-field inversion,
  fitting, plotting, warp search, and report generation.
- `figures/`: generated figures for the synthetic/toy experiments.
- `results/`: generated CSV outputs for the synthetic/toy experiments.
- `reports/`: generated Markdown/HTML reports from the root runners.
- `papers/`: standalone manuscripts and compiled PDFs.
- `experiments/`: self-contained real-data or specialized experiments.

## Experiments

### Geometric Residual AI Experiment

Fits named toy residual families as a scaffold for decomposing a geometric
residual into candidate latent generators.

```bash
python run_experiments.py
```

Primary outputs:

- `reports/geometric_residual_ai_experiment.md`
- `reports/geometric_residual_ai_experiment.html`
- `results/synthetic_experiment_summary.csv`
- `results/model_comparison_metrics.csv`
- `results/sparse_basis_weights.csv`

### Geometry-First Warp Grid Search

Searches directly over weak-field metric perturbations,

$$
g_{\mu\nu}^{\rm trial}
=
g_{\mu\nu}^{\rm bar}
+
h_{\mu\nu}(\theta),
$$

using radial perturbations of $\delta\Phi(r)$ and $\delta\Psi(r)$, rather
than starting from named physical theories.

```bash
python run_warp_grid_search.py
```

Primary outputs:

- `reports/geometric_warp_grid_search.md`
- `reports/geometric_warp_grid_search.html`
- `results/warp_grid_search_all.csv`
- `results/warp_top_candidates.csv`
- `results/warp_family_summary.csv`
- `results/warp_basis_weights.csv`
- `figures/warp_rotation_fit.png`
- `figures/warp_potentials.png`
- `figures/warp_residual_acceleration.png`
- `figures/warp_effective_profiles.png`
- `figures/warp_lensing_proxy.png`
- `figures/warp_log_parameter_heatmap.png`
- `figures/warp_parameter_scatter.png`
- `figures/warp_family_scores.png`
- `figures/warp_basis_components.png`

### Synthetic NFW Residual-of-Residual Test

Tests pure NFW against the synthetic flat-curve target and computes the
leftover residual field.

```bash
python run_nfw_residual_experiment.py
```

Primary outputs:

- `figures/nfw_residual_of_residual.png`
- `results/nfw_residual_experiment_summary.csv`
- `results/nfw_residual_of_residual_profiles.csv`
- `papers/nfw_residual_of_residual/nfw_residual_of_residual_JRLanders.tex`
- `papers/nfw_residual_of_residual/nfw_residual_of_residual_JRLanders.pdf`

### SPARC NFW Residual-of-Residual Experiment

Applies the residual-of-residual diagnostic to public SPARC rotation curves.
This experiment is self-contained under:

- `experiments/sparc_nfw_residuals/`

Run it with:

```bash
python experiments/sparc_nfw_residuals/run_sparc_nfw_residuals.py
```

It downloads missing SPARC source tables, fits NFW and cored/isothermal
residual families, and writes its own results, figures, and paper in the
experiment folder.

### SPARC Residual Dictionary Learning

Learns population-level residual dictionaries from the SPARC residual profiles
produced by the NFW residual-of-residual experiment.

```bash
python experiments/sparc_residual_dictionary/run_sparc_residual_dictionary.py
```

This experiment is self-contained under:

- `experiments/sparc_residual_dictionary/`

It interpolates primary-quality SPARC residual profiles onto a common
normalized radius grid, learns nonnegative and signed residual dictionaries,
evaluates held-out radial-point reconstruction, bootstraps mode stability, and
writes a Markdown report in the experiment folder.

## Papers

- `papers/geometric_residual_missing_mass/`: original geometric-residual and
  weak-field warp-search manuscript.
- `papers/nfw_residual_of_residual/`: synthetic NFW residual-of-residual note.
- `papers/sparc_nfw_residuals/`: real-data SPARC residual-of-residual note.
- `papers/sparc_residual_dictionary_learning/`: stat.ML-style conference paper
  on masked dictionary learning for SPARC residual-of-residual fields.

## Scope

The synthetic experiments use toy units, synthetic data, simplified
radial/spherical proxies, and conceptual lensing/slip observables. The SPARC
experiment uses real rotation-curve data but fixed mass-to-light ratios and a
simplified residual-space fitting pipeline. These are diagnostic prototypes,
not full GR metric reconstructions, particle-discovery pipelines, or final
evidence for a new physical theory.
