"""Focused NFW residual-of-residual experiment.

This script compares a pure NFW/CDM residual against the geometry-first
flat-curve target used by the warp search. It asks a narrow question:

    after the best NFW fit, what residual geometry is still left?

The calculation stays in the repository's weak-field toy units:
- r is kpc-like,
- acceleration is (km/s)^2 / kpc,
- mass is in 1e10 solar masses.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from src import inversion, plotting, profiles, warp_search

try:
    from scipy.optimize import curve_fit
except Exception:  # pragma: no cover - only used without scipy.
    curve_fit = None


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
EPS = 1.0e-12


ArrayModel = Callable[..., np.ndarray]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    param_names: tuple[str, ...]
    p0: tuple[float, ...]
    bounds: tuple[tuple[float, ...], tuple[float, ...]]
    func: ArrayModel


def safe_sigma(y: np.ndarray) -> np.ndarray:
    """A stable relative error model for deterministic synthetic residuals."""
    y = np.asarray(y, dtype=float)
    floor = 0.03 * max(float(np.max(np.abs(y))), 1.0)
    return 0.05 * np.maximum(np.abs(y), floor) + floor


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    if len(y) > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def fit_model(
    spec: ModelSpec,
    r: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float], str]:
    """Fit a model on a radial mask and return full-grid prediction."""
    p0 = np.asarray(spec.p0, dtype=float)
    lower = np.asarray(spec.bounds[0], dtype=float)
    upper = np.asarray(spec.bounds[1], dtype=float)
    message = "ok"

    if curve_fit is not None:
        try:
            popt, _ = curve_fit(
                spec.func,
                r[mask],
                y[mask],
                p0=p0,
                sigma=sigma[mask],
                absolute_sigma=False,
                bounds=(lower, upper),
                maxfev=100_000,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback.
            popt = p0
            message = f"fit failed; used p0 ({exc})"
    else:  # pragma: no cover - exercised only without scipy.
        popt = p0
        message = "scipy unavailable; used p0"

    pred = np.asarray(spec.func(r, *popt), dtype=float)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    params = {name: float(value) for name, value in zip(spec.param_names, popt)}
    return pred, params, message


def log_slope(r: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.gradient(
        np.log(np.maximum(np.asarray(y, dtype=float), EPS)),
        np.log(np.maximum(np.asarray(r, dtype=float), EPS)),
        edge_order=2,
    )


def summarize_fit(
    model: str,
    window: str,
    r: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    sigma: np.ndarray,
    fit_mask: np.ndarray,
    eval_mask: np.ndarray,
    params: dict[str, float],
    message: str,
) -> dict[str, object]:
    err = pred - target
    fit_err = err[fit_mask]
    eval_err = err[eval_mask]
    denom = np.maximum(np.abs(target), 0.03 * float(np.max(np.abs(target))))
    slope = log_slope(r, pred)
    slope_target = log_slope(r, target)
    speed = profiles.velocity_from_acceleration(r, pred)
    target_speed = profiles.velocity_from_acceleration(r, target)
    outer = r >= 10.0
    gap = target - pred
    gap_mass = inversion.effective_mass(r, gap)
    gap_density = inversion.effective_density(r, gap_mass, smooth_window=5)

    return {
        "model": model,
        "fit_window": window,
        "params": json.dumps(params),
        "message": message,
        "fit_weighted_mse": float(np.mean((fit_err / sigma[fit_mask]) ** 2)),
        "full_weighted_mse": float(np.mean((err / sigma) ** 2)),
        "full_relative_rmse": float(
            np.sqrt(np.mean(err**2)) / max(np.sqrt(np.mean(target**2)), EPS)
        ),
        "eval_relative_rmse": float(
            np.sqrt(np.mean(eval_err**2)) / max(np.sqrt(np.mean(target[eval_mask] ** 2)), EPS)
        ),
        "full_mean_abs_fractional_error": float(np.mean(np.abs(err) / denom)),
        "eval_mean_abs_fractional_error": float(np.mean(np.abs(eval_err) / denom[eval_mask])),
        "outer_speed_mean": float(np.mean(speed[outer])),
        "outer_speed_flatness": float(np.std(speed[outer]) / max(np.mean(speed[outer]), EPS)),
        "target_outer_speed_mean": float(np.mean(target_speed[outer])),
        "target_outer_speed_flatness": float(
            np.std(target_speed[outer]) / max(np.mean(target_speed[outer]), EPS)
        ),
        "pred_slope_mean_10_40": float(np.mean(slope[outer])),
        "target_slope_mean_10_40": float(np.mean(slope_target[outer])),
        "pred_slope_near_minus_one_fraction_10_40": float(
            np.mean((slope[outer] > -1.15) & (slope[outer] < -0.85))
        ),
        "target_slope_near_minus_one_fraction_10_40": float(
            np.mean((slope_target[outer] > -1.15) & (slope_target[outer] < -0.85))
        ),
        "gap_positive_fraction": float(np.mean(gap > 0.0)),
        "gap_mass_at_rmax": float(gap_mass[-1]),
        "gap_density_nonnegative_fraction": float(np.mean(gap_density >= 0.0)),
        "gap_density_min": float(np.min(gap_density)),
        "gap_density_max": float(np.max(gap_density)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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
        writer.writerows(rows)


def save_diagnostic_figure(
    path: Path,
    r: np.ndarray,
    target: np.ndarray,
    curves: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target_speed = profiles.velocity_from_acceleration(r, target)
    target_density = inversion.effective_density(r, inversion.effective_mass(r, target), smooth_window=5)

    nfw_outer = curves["NFW fit on 10-40 kpc"]
    nfw_full = curves["NFW fit on 0.3-40 kpc"]
    iso_full = curves["cored/isothermal fit on 0.3-40 kpc"]
    gap_outer = target - nfw_outer
    gap_density_outer = inversion.effective_density(
        r,
        inversion.effective_mass(r, gap_outer),
        smooth_window=5,
    )

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.6), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(r, target, color="#111111", lw=2.2, label="target residual")
    ax.plot(r, nfw_full, color="#D95D39", lw=1.8, label="NFW full fit")
    ax.plot(r, nfw_outer, color="#1B998B", lw=1.8, label="NFW outer fit")
    ax.plot(r, iso_full, color="#5B5F97", lw=1.6, label="cored/isothermal full fit")
    ax.set_title("Residual acceleration")
    ax.set_xlabel("r")
    ax.set_ylabel("Delta g")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.plot(r, target_speed, color="#111111", lw=2.2, label="target")
    for label, curve in curves.items():
        if "NFW" in label or "cored" in label:
            ax.plot(r, profiles.velocity_from_acceleration(r, curve), lw=1.5, label=label)
    ax.set_title("Equivalent hidden speed")
    ax.set_xlabel("r")
    ax.set_ylabel("sqrt(r Delta g)")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 2]
    ax.plot(r, log_slope(r, target), color="#111111", lw=2.2, label="target")
    ax.plot(r, log_slope(r, nfw_full), color="#D95D39", lw=1.8, label="NFW full fit")
    ax.plot(r, log_slope(r, nfw_outer), color="#1B998B", lw=1.8, label="NFW outer fit")
    ax.axhline(-1.0, color="#555555", lw=1.0, ls="--", label="flat-curve slope")
    ax.set_ylim(-1.8, 0.6)
    ax.set_title("Local acceleration slope")
    ax.set_xlabel("r")
    ax.set_ylabel("d ln Delta g / d ln r")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.axhline(0.0, color="#555555", lw=1.0)
    ax.plot(r, target - nfw_full, color="#D95D39", lw=1.8, label="target - NFW full")
    ax.plot(r, gap_outer, color="#1B998B", lw=1.8, label="target - NFW outer")
    ax.set_title("Residual of the residual")
    ax.set_xlabel("r")
    ax.set_ylabel("Delta g gap")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    ax.axhline(0.0, color="#555555", lw=1.0)
    ax.plot(r, target_density, color="#111111", lw=2.0, label="target rho_eff")
    ax.plot(r, gap_density_outer, color="#1B998B", lw=1.8, label="gap rho_eff after NFW outer")
    ax.set_title("Effective density proxy")
    ax.set_xlabel("r")
    ax.set_ylabel("rho_eff")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 2]
    phi_target = cumulative_trapezoid(target, r)
    phi_nfw_outer = cumulative_trapezoid(nfw_outer, r)
    phi_gap_outer = phi_target - phi_nfw_outer
    ax.plot(r, phi_target, color="#111111", lw=2.2, label="target delta Phi")
    ax.plot(r, phi_nfw_outer, color="#1B998B", lw=1.8, label="NFW outer delta Phi")
    ax.plot(r, phi_gap_outer, color="#D95D39", lw=1.8, label="leftover delta Phi")
    ax.set_title("Potential-level comparison")
    ax.set_xlabel("r")
    ax.set_ylabel("integral Delta g dr")
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    plotting.setup_matplotlib()

    context = warp_search.make_target_context()
    r = context.r
    target = context.delta_g_target
    sigma = safe_sigma(target)

    windows = {
        "0.3-40 kpc": np.ones_like(r, dtype=bool),
        "1-15 kpc": (r >= 1.0) & (r <= 15.0),
        "3-30 kpc": (r >= 3.0) & (r <= 30.0),
        "10-40 kpc": (r >= 10.0) & (r <= 40.0),
    }
    eval_mask = windows["0.3-40 kpc"]

    specs = [
        ModelSpec(
            name="NFW",
            param_names=("mass_scale", "r_s"),
            p0=(80.0, 15.0),
            bounds=((0.01, 0.5), (2_000.0, 250.0)),
            func=lambda rr, mass_scale, r_s: profiles.nfw_residual(
                rr,
                mass_scale=mass_scale,
                r_s=r_s,
            ),
        ),
        ModelSpec(
            name="cored/isothermal",
            param_names=("v0", "r_c"),
            p0=(190.0, 4.5),
            bounds=((10.0, 0.1), (400.0, 60.0)),
            func=lambda rr, v0, r_c: profiles.isothermal_residual(rr, v0=v0, r_c=r_c),
        ),
        ModelSpec(
            name="log-potential tail",
            param_names=("A", "r0"),
            p0=(45_000.0, 12.0),
            bounds=((1.0, 0.05), (200_000.0, 200.0)),
            func=lambda rr, amplitude, r0: amplitude / (rr + r0),
        ),
    ]

    summary_rows: list[dict[str, object]] = []
    named_curves: dict[str, np.ndarray] = {}

    for spec in specs:
        for window_name, mask in windows.items():
            pred, params, message = fit_model(spec, r, target, sigma, mask)
            row = summarize_fit(
                spec.name,
                window_name,
                r,
                target,
                pred,
                sigma,
                mask,
                eval_mask,
                params,
                message,
            )
            summary_rows.append(row)
            if window_name in {"0.3-40 kpc", "10-40 kpc"}:
                named_curves[f"{spec.name} fit on {window_name}"] = pred

    nfw_full = named_curves["NFW fit on 0.3-40 kpc"]
    nfw_outer = named_curves["NFW fit on 10-40 kpc"]
    iso_full = named_curves["cored/isothermal fit on 0.3-40 kpc"]
    gap_full = target - nfw_full
    gap_outer = target - nfw_outer

    profile_rows = []
    target_density = inversion.effective_density(r, inversion.effective_mass(r, target), smooth_window=5)
    nfw_outer_density = inversion.effective_density(
        r,
        inversion.effective_mass(r, nfw_outer),
        smooth_window=5,
    )
    gap_outer_density = inversion.effective_density(
        r,
        inversion.effective_mass(r, gap_outer),
        smooth_window=5,
    )
    for idx in range(len(r)):
        profile_rows.append(
            {
                "r": float(r[idx]),
                "target_delta_g": float(target[idx]),
                "nfw_full_delta_g": float(nfw_full[idx]),
                "nfw_outer_delta_g": float(nfw_outer[idx]),
                "cored_isothermal_full_delta_g": float(iso_full[idx]),
                "gap_full_delta_g": float(gap_full[idx]),
                "gap_outer_delta_g": float(gap_outer[idx]),
                "target_speed": float(profiles.velocity_from_acceleration(r[idx], target[idx])),
                "nfw_outer_speed": float(profiles.velocity_from_acceleration(r[idx], nfw_outer[idx])),
                "target_slope": float(log_slope(r, target)[idx]),
                "nfw_outer_slope": float(log_slope(r, nfw_outer)[idx]),
                "target_density_proxy": float(target_density[idx]),
                "nfw_outer_density_proxy": float(nfw_outer_density[idx]),
                "gap_outer_density_proxy": float(gap_outer_density[idx]),
            }
        )

    write_csv(RESULTS / "nfw_residual_experiment_summary.csv", summary_rows)
    write_csv(RESULTS / "nfw_residual_of_residual_profiles.csv", profile_rows)
    save_diagnostic_figure(FIGURES / "nfw_residual_of_residual.png", r, target, named_curves)

    best_nfw_full = next(
        row for row in summary_rows if row["model"] == "NFW" and row["fit_window"] == "0.3-40 kpc"
    )
    best_nfw_outer = next(
        row for row in summary_rows if row["model"] == "NFW" and row["fit_window"] == "10-40 kpc"
    )
    best_iso_full = next(
        row
        for row in summary_rows
        if row["model"] == "cored/isothermal" and row["fit_window"] == "0.3-40 kpc"
    )

    print("NFW residual-of-residual experiment completed.")
    print("Created result CSVs:")
    print("  - results/nfw_residual_experiment_summary.csv")
    print("  - results/nfw_residual_of_residual_profiles.csv")
    print("Created figure:")
    print("  - figures/nfw_residual_of_residual.png")
    print("Key fit rows:")
    for row in (best_nfw_full, best_nfw_outer, best_iso_full):
        print(
            f"  - {row['model']} fit {row['fit_window']}: "
            f"rel_rmse={row['full_relative_rmse']:.4f}, "
            f"eval_frac_err={row['eval_mean_abs_fractional_error']:.4f}, "
            f"outer_flatness={row['outer_speed_flatness']:.4f}, "
            f"params={row['params']}"
        )


if __name__ == "__main__":
    main()
