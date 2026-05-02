# Dark Matter Exploration Prototypes

This workspace contains two small reproducible toy prototypes for the geometric-residual framing of the missing-mass problem.

## 1. Geometric Residual AI Experiment

Runs named toy residual families as a first-pass scaffold for decomposing a **geometric residual** into candidate latent generators.

```bash
python run_experiments.py
```

Primary outputs:

- `geometric_residual_ai_experiment.md`
- `geometric_residual_ai_experiment.html`
- `results/synthetic_experiment_summary.csv`
- `results/model_comparison_metrics.csv`
- `results/sparse_basis_weights.csv`

## 2. Geometry-First Warp Grid Search

Runs the theory-agnostic weak-field metric perturbation search. This is the cleaner geometry-first experiment: it searches directly over parameterized spacetime warps,

\[
g_{\mu\nu}^{\rm trial}
=
g_{\mu\nu}^{\rm bar}
+
h_{\mu\nu}(\theta),
\]

using radial perturbations of \(\delta\Phi(r)\) and \(\delta\Psi(r)\), rather than starting from named physical theories.

```bash
python run_warp_grid_search.py
```

Primary outputs:

- `geometric_warp_grid_search.md`
- `geometric_warp_grid_search.html`
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

## Scope

Both experiments use toy units, synthetic data, simplified radial/spherical proxies, and conceptual lensing/slip observables. They are not realistic disk-galaxy inversions, full GR metric reconstructions, particle-discovery pipelines, or evidence for a new physical theory.

