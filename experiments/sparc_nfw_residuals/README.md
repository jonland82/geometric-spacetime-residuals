# SPARC NFW Residual-of-Residual Experiment

This folder applies the NFW residual-of-residual diagnostic to public SPARC
rotation-curve data. It is the real-data counterpart to the synthetic NFW
experiment in `run_nfw_residual_experiment.py`.

## Data

Raw inputs are downloaded into `data/raw/`:

- `MassModels_Lelli2016c.mrt`: SPARC rotation curves and baryonic velocity
  contributions.
- `SPARC_Lelli2016c.mrt`: SPARC galaxy sample metadata from the Zenodo mirror.

Primary source: <https://astroweb.cwru.edu/SPARC/>

Zenodo mirror: <https://zenodo.org/records/16284118>

The SPARC mass-model table provides radius, observed circular velocity,
velocity uncertainty, gas contribution, stellar disk contribution, bulge
contribution, and surface brightness. `Vdisk` and `Vbul` are tabulated for
stellar mass-to-light ratio `M/L = 1` at 3.6 micron.

## Method

The observed residual acceleration is computed as

```text
Delta g_obs(R) = [Vobs^2 - Vbar^2] / R
```

with fixed fiducial stellar mass-to-light ratios

```text
Upsilon_disk = 0.5
Upsilon_bulge = 0.7
```

and

```text
Vbar^2 = sign(Vgas) Vgas^2 + Upsilon_disk Vdisk^2 + Upsilon_bulge Vbul^2.
```

For each usable galaxy, the script fits:

- full-range NFW residual
- outer-range NFW residual
- full-range cored/isothermal residual
- full-range logarithmic-tail residual

The central diagnostic is

```text
epsilon_g(R) = Delta g_obs(R) - Delta g_NFW_outer(R).
```

When the outer NFW fit matches the outer residual but produces a negative inner
`epsilon_g`, the interpretation is central overshoot: the NFW source required by
the outer region is too large in the inner region.

## Outputs

Results:

- `results/galaxy_fit_summary.csv`: one row per galaxy with fit metrics and
  residual classifications.
- `results/residual_profiles.csv`: pointwise residual profiles and model curves.
- `results/population_summary.csv`: population-level summary statistics.
- `results/source_and_assumptions.json`: source URLs and fixed assumptions.

Figures:

- `figures/example_residual_profiles.png`
- `figures/population_fit_comparison.png`
- `figures/central_overshoot_distribution.png`

## Headline Results

The run parsed 3,391 rotation-curve points for 175 SPARC galaxies. Of these,
165 had enough points for the fit pipeline. The primary-quality sample contains
131 galaxies with SPARC quality flag 1 or 2, at least 8 points, and inclination
above 30 degrees when available. The `primary_flat` subset contains 113 of
those with measured `Vflat > 0`.

Population summary:

| sample | galaxies | median NFW rel. RMSE | median cored rel. RMSE | cored beats NFW | central overshoot after outer NFW | median inner gap norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all_ok | 165 | 0.337 | 0.203 | 0.788 | 0.521 | -0.449 |
| primary_quality | 131 | 0.334 | 0.206 | 0.802 | 0.519 | -0.469 |
| primary_flat | 113 | 0.334 | 0.211 | 0.805 | 0.504 | -0.449 |

The real-data result broadly echoes the toy result:

1. A cored/isothermal residual shape fits the baryon-subtracted SPARC residuals
   better than pure NFW for about 80% of the primary-quality sample.
2. When NFW is fit only to the outer region, about half of the primary-quality
   galaxies show a structured negative inner residual.
3. Full-range NFW fits often become too shallow in the outer region; this
   happens for about 75% of the primary-quality sample under the current slope
   diagnostic.

## Interpretation

This does not falsify cold dark matter. The analysis uses fixed stellar
mass-to-light ratios, no distance or inclination marginalization, no
cosmological concentration prior, no baryonic feedback model, and a spherical
effective-density proxy. It is a controlled residual-space diagnostic, not a
full halo inference.

What it does show is that the residual-of-residual object is meaningful on real
data. The leftover field after NFW fitting is not just numerical clutter; it
forms repeatable patterns that can be classified across galaxies.

## Reproduce

From the repository root:

```powershell
python .\experiments\sparc_nfw_residuals\run_sparc_nfw_residuals.py
```

The script downloads missing raw SPARC files, reruns the fits, and rewrites the
CSV and figure outputs.

## Next Steps

- Add fitted or marginalized stellar mass-to-light ratios.
- Fit NFW with cosmological mass-concentration priors.
- Add Burkert, pISO, Einasto, DC14, and coreNFW comparisons.
- Replace the spherical density proxy with disk-aware diagnostics.
- Bootstrap or MCMC the residual-of-residual classifications.
- Compare against the published SPARC halo-fit catalog by Li et al. (2020).
