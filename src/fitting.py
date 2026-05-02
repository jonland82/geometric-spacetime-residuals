"""Candidate-generator fitting for geometric residual profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import inversion, profiles

try:  # scipy is allowed but the prototype should degrade gracefully.
    from scipy.optimize import curve_fit, nnls
except Exception:  # pragma: no cover - exercised only without scipy.
    curve_fit = None
    nnls = None


ArrayFunc = Callable[..., np.ndarray]


@dataclass
class CandidateFit:
    name: str
    params: dict[str, float | str]
    y_pred: np.ndarray
    chi2: float
    weighted_mse: float
    aic: float
    bic: float
    complexity: int
    success: bool
    message: str
    checks: dict[str, float | bool]


def _safe_sigma(sigma: np.ndarray | None, y: np.ndarray) -> np.ndarray:
    if sigma is None:
        floor = 0.05 * max(float(np.nanmax(np.abs(y))), 1.0)
        return np.full_like(y, floor, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    floor = 0.01 * max(float(np.nanmax(np.abs(y))), 1.0)
    return np.maximum(sigma, floor)


def _metrics(y: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray, complexity: int) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sigma = _safe_sigma(sigma, y)
    resid = y - y_pred
    weighted = resid / sigma
    chi2 = float(np.sum(weighted**2))
    weighted_mse = float(np.mean(weighted**2))
    n = len(y)
    rss = max(chi2, 1.0e-12)
    aic = float(n * np.log(rss / n) + 2 * complexity)
    bic = float(n * np.log(rss / n) + complexity * np.log(n))
    return {"chi2": chi2, "weighted_mse": weighted_mse, "aic": aic, "bic": bic}


def _fit_parametric(
    name: str,
    func: ArrayFunc,
    param_names: list[str],
    p0: list[float],
    bounds: tuple[list[float], list[float]],
    r: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
) -> CandidateFit:
    success = True
    message = "ok"
    popt = np.asarray(p0, dtype=float)

    if curve_fit is not None:
        try:
            popt, _ = curve_fit(
                func,
                r,
                y,
                p0=p0,
                sigma=sigma,
                absolute_sigma=False,
                bounds=bounds,
                maxfev=30000,
            )
        except Exception as exc:
            success = False
            message = f"fit failed; used initial guess ({exc})"
    else:
        success = False
        message = "scipy unavailable; used initial guess"

    y_pred = np.asarray(func(r, *popt), dtype=float)
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
    met = _metrics(y, y_pred, sigma, complexity=len(popt))
    checks = inversion.physical_checks(r, y_pred)
    params = {key: float(value) for key, value in zip(param_names, popt)}
    return CandidateFit(
        name=name,
        params=params,
        y_pred=y_pred,
        chi2=met["chi2"],
        weighted_mse=met["weighted_mse"],
        aic=met["aic"],
        bic=met["bic"],
        complexity=len(popt),
        success=success,
        message=message,
        checks=checks,
    )


def fit_candidate_models(
    r: np.ndarray,
    delta_g_est: np.ndarray,
    sigma_delta_g: np.ndarray,
    g_bar: np.ndarray,
    include_sparse: bool = True,
) -> list[CandidateFit]:
    """Fit single-generator candidate families and an optional sparse mixture."""
    r = np.asarray(r, dtype=float)
    y = np.asarray(delta_g_est, dtype=float)
    sigma = _safe_sigma(sigma_delta_g, y)
    g_bar = np.asarray(g_bar, dtype=float)

    candidates: list[CandidateFit] = []
    candidates.append(
        _fit_parametric(
            "NFW-like halo",
            lambda rr, mass_scale, r_s: profiles.nfw_residual(rr, mass_scale=mass_scale, r_s=r_s),
            ["mass_scale", "r_s"],
            [20.0, 12.0],
            ([0.1, 1.0], [120.0, 45.0]),
            r,
            y,
            sigma,
        )
    )
    candidates.append(
        _fit_parametric(
            "Cored/isothermal halo",
            lambda rr, v0, r_c: profiles.isothermal_residual(rr, v0=v0, r_c=r_c),
            ["v0", "r_c"],
            [120.0, 4.0],
            ([10.0, 0.2], [280.0, 25.0]),
            r,
            y,
            sigma,
        )
    )
    candidates.append(
        _fit_parametric(
            "MOND-like relation",
            lambda rr, a0: profiles.mond_residual(rr, g_bar=g_bar, a0=a0),
            ["a0"],
            [1000.0],
            ([10.0], [6000.0]),
            r,
            y,
            sigma,
        )
    )
    candidates.append(
        _fit_parametric(
            "Soliton/cored scalar",
            lambda rr, rho0, r_c: profiles.soliton_residual(rr, rho0=rho0, r_c=r_c),
            ["rho0", "r_c"],
            [0.35, 3.0],
            ([1.0e-4, 0.3], [8.0, 12.0]),
            r,
            y,
            sigma,
        )
    )

    if include_sparse:
        sparse = fit_sparse_basis(r, y, sigma, g_bar, max_terms=4)
        candidates.append(
            CandidateFit(
                name="Sparse nonnegative mixture",
                params={"terms": "; ".join(sparse["selected_labels"])},
                y_pred=sparse["y_pred"],
                chi2=sparse["metrics"]["chi2"],
                weighted_mse=sparse["metrics"]["weighted_mse"],
                aic=sparse["metrics"]["aic"],
                bic=sparse["metrics"]["bic"],
                complexity=len(sparse["selected_labels"]),
                success=True,
                message="greedy NNLS basis selection",
                checks=inversion.physical_checks(r, sparse["y_pred"]),
            )
        )

    return sorted(candidates, key=lambda item: item.bic)


def build_basis_library(r: np.ndarray, g_bar: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Construct an AI/symbolic-regression-inspired residual basis library."""
    r = np.asarray(r, dtype=float)
    g_bar = np.asarray(g_bar, dtype=float)
    median_g = max(float(np.median(np.maximum(g_bar, 0.0))), 1.0)
    labels: list[str] = []
    columns: list[np.ndarray] = []

    for r_s in (6.0, 12.0, 24.0):
        labels.append(f"NFW shape rs={r_s:g}")
        columns.append(profiles.nfw_residual(r, mass_scale=1.0, r_s=r_s))

    for r_c in (2.0, 5.0, 10.0):
        labels.append(f"Iso shape rc={r_c:g}")
        columns.append(profiles.isothermal_residual(r, v0=1.0, r_c=r_c))

    for a0 in (400.0, 1000.0, 2200.0):
        labels.append(f"MOND residual a0={a0:g}")
        columns.append(profiles.mond_residual(r, g_bar=g_bar, a0=a0))

    for r_c in (1.5, 3.0, 6.0):
        labels.append(f"Soliton shape rc={r_c:g}")
        columns.append(profiles.soliton_residual(r, rho0=1.0, r_c=r_c))

    labels.extend(["Baryon-coupled gbar", "sqrt(gbar) shape", "1/(r+Rd) tail"])
    columns.extend(
        [
            np.maximum(g_bar, 0.0),
            np.sqrt(np.maximum(g_bar, 0.0) * median_g),
            median_g / (r + 3.0),
        ]
    )

    basis = np.column_stack(columns)
    basis = np.nan_to_num(basis, nan=0.0, posinf=0.0, neginf=0.0)
    return labels, basis


def _solve_nnls(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Nonnegative least squares with a numpy fallback."""
    if design.size == 0:
        return np.array([])
    if nnls is not None:
        weights, _ = nnls(design, y)
        return weights
    weights, *_ = np.linalg.lstsq(design, y, rcond=None)
    return np.maximum(weights, 0.0)


def fit_sparse_basis(
    r: np.ndarray,
    delta_g_est: np.ndarray,
    sigma_delta_g: np.ndarray,
    g_bar: np.ndarray,
    max_terms: int = 4,
) -> dict[str, object]:
    """Greedy sparse nonnegative basis search.

    Columns are normalized so the selected weights have acceleration units.
    Selection minimizes BIC on the weighted residual.
    """
    r = np.asarray(r, dtype=float)
    y = np.asarray(delta_g_est, dtype=float)
    sigma = _safe_sigma(sigma_delta_g, y)
    labels, basis = build_basis_library(r, g_bar)
    scales = np.maximum(np.max(np.abs(basis), axis=0), 1.0e-12)
    basis_norm = basis / scales
    weighted_y = y / sigma
    weighted_basis = basis_norm / sigma[:, None]

    selected: list[int] = []
    remaining = list(range(len(labels)))
    best_bic = np.inf
    best_weights = np.array([])

    for _ in range(max_terms):
        best_trial = None
        for idx in remaining:
            trial = selected + [idx]
            design = weighted_basis[:, trial]
            weights = _solve_nnls(design, weighted_y)
            pred = basis_norm[:, trial] @ weights
            met = _metrics(y, pred, sigma, complexity=len(trial))
            if best_trial is None or met["bic"] < best_trial["bic"]:
                best_trial = {"idx": idx, "weights": weights, "bic": met["bic"], "metrics": met}
        if best_trial is None:
            break
        if best_trial["bic"] < best_bic - 1.0:
            selected.append(int(best_trial["idx"]))
            remaining.remove(int(best_trial["idx"]))
            best_bic = float(best_trial["bic"])
            best_weights = np.asarray(best_trial["weights"], dtype=float)
        else:
            break

    if not selected:
        idx = int(np.argmax(np.maximum(np.sum(basis_norm * y[:, None], axis=0), 0.0)))
        selected = [idx]
        best_weights = _solve_nnls(weighted_basis[:, selected], weighted_y)

    design = weighted_basis[:, selected]
    best_weights = _solve_nnls(design, weighted_y)
    y_pred = basis_norm[:, selected] @ best_weights
    met = _metrics(y, y_pred, sigma, complexity=len(selected))
    selected_labels = [labels[i] for i in selected]
    contributions = basis_norm[:, selected] * best_weights
    rms_contrib = np.sqrt(np.mean(contributions**2, axis=0))

    weights_table = [
        {
            "basis": label,
            "normalized_weight": float(weight),
            "contribution_rms": float(rms),
        }
        for label, weight, rms in zip(selected_labels, best_weights, rms_contrib)
    ]

    return {
        "selected_labels": selected_labels,
        "weights": best_weights,
        "weights_table": weights_table,
        "y_pred": y_pred,
        "contributions": contributions,
        "metrics": met,
        "all_labels": labels,
    }


def fits_to_rows(fits: list[CandidateFit]) -> list[dict[str, object]]:
    """Convert fit objects to CSV-friendly dictionaries."""
    rows: list[dict[str, object]] = []
    for fit in fits:
        row: dict[str, object] = {
            "model": fit.name,
            "chi2": fit.chi2,
            "weighted_mse": fit.weighted_mse,
            "aic": fit.aic,
            "bic": fit.bic,
            "complexity": fit.complexity,
            "success": fit.success,
            "message": fit.message,
            "params": fit.params,
        }
        row.update(fit.checks)
        rows.append(row)
    return rows

