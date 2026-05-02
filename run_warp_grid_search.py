"""Run the geometry-first weak-field spacetime-warp grid search.

Usage:
    python run_warp_grid_search.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import warp_plotting, warp_search, warp_writeups


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    warp_plotting.setup_matplotlib()

    context = warp_search.make_target_context()
    frame, basis_weights = warp_search.run_grid_search(context)
    family_summary = warp_search.summarize_families(frame)
    top = frame.head(30).copy()
    basis_frame = pd.DataFrame(basis_weights)

    warp_search.dataframe_for_csv(frame).to_csv(RESULTS / "warp_grid_search_all.csv", index=False)
    warp_search.dataframe_for_csv(top).to_csv(RESULTS / "warp_top_candidates.csv", index=False)
    warp_search.dataframe_for_csv(family_summary).to_csv(RESULTS / "warp_family_summary.csv", index=False)
    basis_frame.to_csv(RESULTS / "warp_basis_weights.csv", index=False)

    best = frame.iloc[0]
    best_arrays = warp_search.reconstruct_candidate(best, context)

    family_curves: list[tuple[str, object]] = []
    for _, fam_row in family_summary.head(4).iterrows():
        family = fam_row["family"]
        candidate = frame[frame["family"] == family].iloc[0]
        arrays = warp_search.reconstruct_candidate(candidate, context)
        if family != best["family"]:
            family_curves.append((str(family), arrays["v_trial"]))

    slip_examples = []
    for cfg in warp_search.eta_configurations(context.r):
        if cfg["eta_mode"] in {"constant_high", "outer_slip"}:
            eta = cfg["eta"]
            proxy = best_arrays["delta_phi"] + eta * best_arrays["delta_phi"]
            slip_examples.append((str(cfg["eta_mode"]), proxy))

    warp_plotting.save_rotation_fit(FIGURES / "warp_rotation_fit.png", context, best_arrays, family_curves)
    warp_plotting.save_warp_potentials(FIGURES / "warp_potentials.png", context, best_arrays)
    warp_plotting.save_residual_acceleration(FIGURES / "warp_residual_acceleration.png", context, best_arrays)
    warp_plotting.save_effective_profiles(FIGURES / "warp_effective_profiles.png", context, best_arrays)
    warp_plotting.save_lensing_proxy(FIGURES / "warp_lensing_proxy.png", context, best_arrays, slip_examples)
    warp_plotting.save_log_heatmap(FIGURES / "warp_log_parameter_heatmap.png", frame)
    warp_plotting.save_parameter_scatter(FIGURES / "warp_parameter_scatter.png", frame)
    warp_plotting.save_family_scores(FIGURES / "warp_family_scores.png", family_summary)
    warp_plotting.save_basis_components(FIGURES / "warp_basis_components.png", context, basis_weights)

    md_path, html_path = warp_writeups.write_reports(ROOT, top, family_summary, basis_frame)

    print("Geometry-first weak-field warp grid search completed.")
    print(f"Evaluated {len(frame)} trial warp/slip combinations.")
    print("Created writeups:")
    print(f"  - {md_path.name}")
    print(f"  - {html_path.name}")
    print("Created result CSVs:")
    print("  - results/warp_grid_search_all.csv")
    print("  - results/warp_top_candidates.csv")
    print("  - results/warp_family_summary.csv")
    print("  - results/warp_basis_weights.csv")
    print("Created figures:")
    for name in [
        "warp_rotation_fit.png",
        "warp_potentials.png",
        "warp_residual_acceleration.png",
        "warp_effective_profiles.png",
        "warp_lensing_proxy.png",
        "warp_log_parameter_heatmap.png",
        "warp_parameter_scatter.png",
        "warp_family_scores.png",
        "warp_basis_components.png",
    ]:
        print(f"  - figures/{name}")
    print("Top-ranked warp candidates:")
    for idx, row in frame.head(5).iterrows():
        print(
            f"  {idx + 1}. {row['family']} / {row['eta_mode']} "
            f"score={row['total_score']:.4f}, rotation_mse={row['rotation_mse']:.4f}, "
            f"params={row['params']}"
        )


if __name__ == "__main__":
    main()

