"""Run all geometric residual toy experiments and write outputs.

Usage:
    python run_experiments.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src import fitting, inversion, plotting, profiles, writeups


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
SEED = 314159


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key, "")) for key in fieldnames})


def make_noisy_observation(
    r: np.ndarray,
    g_bar: np.ndarray,
    delta_g_true: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Create noisy circular-speed data from a known residual."""
    g_obs_true = g_bar + delta_g_true
    v_obs_true = profiles.velocity_from_acceleration(r, g_obs_true)
    v_bar = profiles.velocity_from_acceleration(r, g_bar)
    v_hidden = profiles.velocity_from_acceleration(r, delta_g_true)

    # Controlled low-amplitude Gaussian velocity noise. Keeping this modest lets
    # the mixed-source inverse problem show up without hiding it under noise.
    sigma_v = 0.3 + 0.003 * v_obs_true
    v_obs_noisy = np.clip(v_obs_true + rng.normal(0.0, sigma_v), 1.0, None)

    # Error propagation for g=v^2/r. This is approximate and intentionally simple.
    sigma_delta = np.maximum(2.0 * v_obs_true * sigma_v / np.maximum(r, profiles.EPS), 0.1)
    return {
        "g_obs_true": g_obs_true,
        "v_obs_true": v_obs_true,
        "v_bar": v_bar,
        "v_hidden": v_hidden,
        "sigma_v": sigma_v,
        "v_obs_noisy": v_obs_noisy,
        "sigma_delta": sigma_delta,
    }


def scenario_residuals(r: np.ndarray, g_bar: np.ndarray) -> dict[str, np.ndarray]:
    """Generate the hidden residual families used in the experiments."""
    return {
        "nfw_like": profiles.nfw_residual(r, mass_scale=22.0, r_s=12.0),
        "cored_isothermal": profiles.isothermal_residual(r, v0=124.0, r_c=4.2),
        "mond_like": profiles.mond_residual(r, g_bar=g_bar, a0=1000.0),
        "soliton_cored": profiles.soliton_residual(r, rho0=0.32, r_c=3.1),
        "mixed_nfw_mond": profiles.mixed_residual(
            r,
            g_bar=g_bar,
            nfw_weight=0.45,
            mond_weight=0.55,
            nfw_mass_scale=22.0,
            nfw_r_s=12.0,
            mond_a0=2200.0,
        ),
    }


def summarize_recovery(
    component: str,
    r: np.ndarray,
    delta_true: np.ndarray,
    delta_est: np.ndarray,
) -> dict[str, Any]:
    err = delta_est - delta_true
    rmse = float(np.sqrt(np.mean(err**2)))
    rel_rmse = float(rmse / max(np.sqrt(np.mean(delta_true**2)), 1.0e-12))
    corr = float(np.corrcoef(delta_true, delta_est)[0, 1])
    checks = inversion.physical_checks(r, delta_est)
    row: dict[str, Any] = {
        "component": component,
        "rmse_delta_g": rmse,
        "relative_rmse_delta_g": rel_rmse,
        "corr_delta_g": corr,
    }
    row.update(checks)
    return row


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    plotting.setup_matplotlib()
    rng = np.random.default_rng(SEED)

    r = profiles.radial_grid()
    m_bar = profiles.baryonic_mass(r, m_b=5.0, r_d=3.0)
    g_bar = profiles.acceleration_from_mass(r, m_bar)
    hidden = scenario_residuals(r, g_bar)

    summary_rows: list[dict[str, Any]] = []
    scenario_data: dict[str, dict[str, np.ndarray]] = {}

    for component, delta_true in hidden.items():
        obs = make_noisy_observation(r, g_bar, delta_true, rng)
        delta_est, m_eff, rho_eff = inversion.invert_effective_profiles(
            r,
            obs["v_obs_noisy"],
            g_bar,
            smooth_window=7,
        )
        scenario_data[component] = {
            **obs,
            "delta_true": delta_true,
            "delta_est": delta_est,
            "m_eff_est": m_eff,
            "rho_eff_est": rho_eff,
            "m_eff_true": inversion.effective_mass(r, delta_true),
            "rho_eff_true": inversion.effective_density(
                r,
                inversion.effective_mass(r, delta_true),
                smooth_window=1,
            ),
        }
        summary_rows.append(summarize_recovery(component, r, delta_true, delta_est))

    main_case = scenario_data["mixed_nfw_mond"]
    fits = fitting.fit_candidate_models(
        r,
        main_case["delta_est"],
        main_case["sigma_delta"],
        g_bar,
        include_sparse=True,
    )
    model_rows = fitting.fits_to_rows(fits)
    for row in model_rows:
        row["scenario"] = "mixed_nfw_mond"

    sparse_fit = next(fit for fit in fits if fit.name == "Sparse nonnegative mixture")
    single_fits = [fit for fit in fits if fit.name != "Sparse nonnegative mixture"]
    best_single = min(single_fits, key=lambda item: item.bic)
    sparse_info = fitting.fit_sparse_basis(
        r,
        main_case["delta_est"],
        main_case["sigma_delta"],
        g_bar,
        max_terms=4,
    )
    sparse_rows = list(sparse_info["weights_table"])

    # Multi-probe toy: keep nearly the same rotation residual but change slip.
    particle_delta = sparse_fit.y_pred
    modified_delta = particle_delta * (1.0 + 0.018 * np.sin(r / 4.0))
    eta_modified = 1.0 + 0.35 * np.exp(-r / 12.0)
    particle_proxy = inversion.lensing_proxy(r, particle_delta, eta=1.0)
    modified_proxy = inversion.lensing_proxy(r, modified_delta, eta=eta_modified)

    plotting.save_rotation_curve(
        FIGURES / "rotation_curve.png",
        r,
        main_case["v_bar"],
        main_case["v_obs_true"],
        main_case["v_obs_noisy"],
        main_case["v_hidden"],
        main_case["sigma_v"],
    )
    plotting.save_acceleration_residual(
        FIGURES / "acceleration_residual.png",
        r,
        main_case["delta_true"],
        main_case["delta_est"],
        main_case["sigma_delta"],
    )
    plotting.save_effective_mass_density(
        FIGURES / "effective_mass_density.png",
        r,
        main_case["m_eff_true"],
        main_case["m_eff_est"],
        main_case["rho_eff_true"],
        main_case["rho_eff_est"],
    )
    plotting.save_model_comparison(
        FIGURES / "model_comparison.png",
        fits,
        r,
        main_case["delta_est"],
    )
    plotting.save_mixed_source_fit(
        FIGURES / "mixed_source_fit.png",
        r,
        main_case["delta_true"],
        main_case["delta_est"],
        best_single,
        sparse_fit,
    )
    plotting.save_lensing_proxy_degeneracy(
        FIGURES / "lensing_proxy_degeneracy.png",
        r,
        particle_delta,
        modified_delta,
        particle_proxy,
        modified_proxy,
        eta_modified,
    )
    plotting.save_sparse_basis_search(
        FIGURES / "sparse_basis_search.png",
        r,
        main_case["delta_est"],
        sparse_info,
    )

    write_csv(RESULTS / "synthetic_experiment_summary.csv", summary_rows)
    write_csv(RESULTS / "model_comparison_metrics.csv", model_rows)
    write_csv(RESULTS / "sparse_basis_weights.csv", sparse_rows)
    md_path, html_path = writeups.write_research_notes(ROOT, summary_rows, model_rows, sparse_rows)

    best_overall = min(fits, key=lambda item: item.bic)
    print("Geometric residual toy experiments completed.")
    print(f"Random seed: {SEED}")
    print("Created writeups:")
    print(f"  - {md_path.name}")
    print(f"  - {html_path.name}")
    print("Created result CSVs:")
    print("  - results/synthetic_experiment_summary.csv")
    print("  - results/model_comparison_metrics.csv")
    print("  - results/sparse_basis_weights.csv")
    print("Created figures:")
    for name in [
        "rotation_curve.png",
        "acceleration_residual.png",
        "effective_mass_density.png",
        "model_comparison.png",
        "mixed_source_fit.png",
        "lensing_proxy_degeneracy.png",
        "sparse_basis_search.png",
    ]:
        print(f"  - figures/{name}")
    print("Key result highlights:")
    print(f"  - Best overall mixed-source fit by BIC-like score: {best_overall.name}")
    print(f"  - Best single-family fit by BIC-like score: {best_single.name}")
    print("  - Sparse selected terms: " + ", ".join(sparse_info["selected_labels"]))
    print("  - Lensing/slip proxy is conceptual, not a physical lensing calculation.")


if __name__ == "__main__":
    main()
