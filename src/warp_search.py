"""Geometry-first weak-field metric-warp grid search.

This module deliberately searches over neutral perturbations of the weak-field
metric potentials rather than over named dark-matter or modified-gravity models.
The potential units are the same toy astrophysical units used elsewhere in this
repository: Phi has units of (km/s)^2, r is kpc-like, and dPhi/dr has units of
(km/s)^2/kpc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from typing import Iterator

import numpy as np
import pandas as pd

from . import profiles

try:
    from scipy.optimize import nnls
except Exception:  # pragma: no cover - used only without scipy.
    nnls = None


C_KMS = 299_792.458
EPS = 1.0e-12


@dataclass
class WarpContext:
    r: np.ndarray
    g_bar: np.ndarray
    phi_bar: np.ndarray
    v_bar: np.ndarray
    v_obs: np.ndarray
    sigma_v: np.ndarray
    delta_g_target: np.ndarray
    delta_phi_target: np.ndarray
    lensing_target: np.ndarray
    sigma_lensing: float


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Small cumulative trapezoid helper."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(y)
    if len(y) > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def make_target_context(
    r_min: float = 0.3,
    r_max: float = 40.0,
    n: int = 220,
    m_b: float = 4.0,
    r_d: float = 3.0,
    v_flat: float = 190.0,
    r_flat: float = 4.5,
) -> WarpContext:
    """Construct a baryon-plus-flat-residual toy galaxy target."""
    r = profiles.radial_grid(r_min=r_min, r_max=r_max, n=n)
    m_bar = profiles.baryonic_mass(r, m_b=m_b, r_d=r_d)
    g_bar = profiles.acceleration_from_mass(r, m_bar)
    v_bar = profiles.velocity_from_acceleration(r, g_bar)

    # The extra speed component approaches v_flat, so the implied potential
    # perturbation approaches a logarithmic radial growth at large radius.
    v_extra = v_flat * (1.0 - np.exp(-r / r_flat))
    v_obs = np.sqrt(v_bar**2 + v_extra**2)
    delta_g_target = np.maximum(v_obs**2 / np.maximum(r, EPS) - g_bar, 0.0)
    phi_bar = cumulative_trapezoid(g_bar, r)
    delta_phi_target = cumulative_trapezoid(delta_g_target, r)

    sigma_v = 3.0 + 0.015 * v_obs
    lensing_target = 2.0 * delta_phi_target
    sigma_lensing = 0.08 * max(float(np.max(np.abs(lensing_target))), 1.0)

    return WarpContext(
        r=r,
        g_bar=g_bar,
        phi_bar=phi_bar,
        v_bar=v_bar,
        v_obs=v_obs,
        sigma_v=sigma_v,
        delta_g_target=delta_g_target,
        delta_phi_target=delta_phi_target,
        lensing_target=lensing_target,
        sigma_lensing=sigma_lensing,
    )


def eta_configurations(r: np.ndarray) -> list[dict[str, object]]:
    """Slip-like deltaPsi/deltaPhi parameterizations."""
    r = np.asarray(r, dtype=float)
    return [
        {"eta_mode": "no_slip", "eta_params": {"eta": 1.0}, "eta": np.ones_like(r), "eta_complexity": 0},
        {"eta_mode": "constant_low", "eta_params": {"eta": 0.75}, "eta": 0.75 * np.ones_like(r), "eta_complexity": 1},
        {"eta_mode": "constant_high", "eta_params": {"eta": 1.25}, "eta": 1.25 * np.ones_like(r), "eta_complexity": 1},
        {
            "eta_mode": "inner_slip",
            "eta_params": {"beta": 0.35, "r_eta": 8.0},
            "eta": 1.0 + 0.35 * np.exp(-r / 8.0),
            "eta_complexity": 2,
        },
        {
            "eta_mode": "outer_slip",
            "eta_params": {"beta": 0.35, "r_eta": 12.0},
            "eta": 1.0 + 0.35 * (1.0 - np.exp(-r / 12.0)),
            "eta_complexity": 2,
        },
    ]


def log_warp(r: np.ndarray, amplitude: float, r0: float) -> tuple[np.ndarray, np.ndarray]:
    phi = amplitude * np.log1p(r / r0)
    delta_g = amplitude / (r + r0)
    return phi - phi[0], delta_g


def power_warp(r: np.ndarray, amplitude: float, r0: float, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.maximum(r / r0, EPS)
    phi = amplitude * x**alpha
    delta_g = amplitude * alpha * x ** (alpha - 1.0) / r0
    return phi - phi[0], delta_g


def exponential_warp(r: np.ndarray, amplitude: float, r0: float) -> tuple[np.ndarray, np.ndarray]:
    phi = amplitude * (1.0 - np.exp(-r / r0))
    delta_g = amplitude * np.exp(-r / r0) / r0
    return phi - phi[0], delta_g


def rational_warp(r: np.ndarray, amplitude: float, r0: float, n: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.maximum(r / r0, EPS) ** n
    phi = amplitude * x / (1.0 + x)
    delta_g = amplitude * n * x / (np.maximum(r, EPS) * (1.0 + x) ** 2)
    return phi - phi[0], delta_g


def gaussian_bump_warp(
    r: np.ndarray,
    amplitude: float,
    r_c: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    phi = amplitude * np.exp(-0.5 * ((r - r_c) / sigma) ** 2)
    delta_g = phi * (-(r - r_c) / max(sigma**2, EPS))
    return phi - phi[0], delta_g


def shape_from_params(
    family: str,
    params: dict[str, float | int | str],
    r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild a warp shape from stored parameter metadata."""
    if family == "log_warp":
        return log_warp(r, float(params["A"]), float(params["r0"]))
    if family == "power_law_warp":
        return power_warp(r, float(params["A"]), float(params["r0"]), float(params["alpha"]))
    if family == "exponential_saturating_warp":
        return exponential_warp(r, float(params["A"]), float(params["r0"]))
    if family == "rational_saturating_warp":
        return rational_warp(r, float(params["A"]), float(params["r0"]), float(params["n"]))
    if family == "gaussian_bump_warp":
        return gaussian_bump_warp(r, float(params["A"]), float(params["r_c"]), float(params["sigma"]))
    raise ValueError(f"Unknown warp family: {family}")


def iter_parametric_shapes(r: np.ndarray) -> Iterator[dict[str, object]]:
    """Yield direct metric-potential perturbation candidates."""
    for amplitude, r0 in product(np.linspace(22_000.0, 70_000.0, 36), np.geomspace(0.8, 30.0, 28)):
        phi, delta_g = log_warp(r, amplitude, r0)
        yield {
            "family": "log_warp",
            "params": {"A": float(amplitude), "r0": float(r0)},
            "delta_phi": phi,
            "delta_g": delta_g,
            "complexity": 2,
        }

    for amplitude, r0, alpha in product(
        np.linspace(25_000.0, 180_000.0, 18),
        np.geomspace(1.0, 15.0, 10),
        [0.15, 0.25, 0.40, 0.60, 0.80, 1.00, 1.25],
    ):
        phi, delta_g = power_warp(r, amplitude, r0, alpha)
        yield {
            "family": "power_law_warp",
            "params": {"A": float(amplitude), "r0": float(r0), "alpha": float(alpha)},
            "delta_phi": phi,
            "delta_g": delta_g,
            "complexity": 3,
        }

    for amplitude, r0 in product(np.linspace(10_000.0, 300_000.0, 16), np.geomspace(4.0, 85.0, 8)):
        phi, delta_g = exponential_warp(r, amplitude, r0)
        yield {
            "family": "exponential_saturating_warp",
            "params": {"A": float(amplitude), "r0": float(r0)},
            "delta_phi": phi,
            "delta_g": delta_g,
            "complexity": 2,
        }

    for amplitude, r0, n in product(
        np.linspace(20_000.0, 140_000.0, 16),
        np.geomspace(2.0, 28.0, 8),
        [1.0, 2.0, 3.0, 4.0],
    ):
        phi, delta_g = rational_warp(r, amplitude, r0, n)
        yield {
            "family": "rational_saturating_warp",
            "params": {"A": float(amplitude), "r0": float(r0), "n": float(n)},
            "delta_phi": phi,
            "delta_g": delta_g,
            "complexity": 3,
        }

    amplitudes = np.concatenate([np.linspace(-160_000.0, -25_000.0, 7), np.linspace(25_000.0, 160_000.0, 7)])
    for amplitude, r_c, sigma in product(amplitudes, [4.0, 8.0, 12.0, 18.0, 25.0, 32.0], [2.0, 4.0, 7.0, 11.0]):
        phi, delta_g = gaussian_bump_warp(r, amplitude, r_c, sigma)
        yield {
            "family": "gaussian_bump_warp",
            "params": {"A": float(amplitude), "r_c": float(r_c), "sigma": float(sigma)},
            "delta_phi": phi,
            "delta_g": delta_g,
            "complexity": 3,
        }


def effective_profiles(r: np.ndarray, delta_g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m_eff = r**2 * delta_g / profiles.G_TOY
    dmass_dr = np.gradient(m_eff, r, edge_order=2)
    rho_eff = dmass_dr / (4.0 * np.pi * np.maximum(r, EPS) ** 2)
    return m_eff, rho_eff


def normalized_proxy(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    shifted = y - y[0]
    scale = np.max(np.abs(shifted))
    if scale <= 0:
        return np.zeros_like(shifted)
    return shifted / scale


def _oscillation_score(y: np.ndarray) -> float:
    second = np.diff(y, n=2)
    if len(second) < 2:
        return 0.0
    return float(np.mean(np.diff(np.signbit(second)) != 0))


def score_warp(
    context: WarpContext,
    family: str,
    params: dict[str, float | int | str],
    delta_phi: np.ndarray,
    delta_g: np.ndarray,
    eta_config: dict[str, object],
    complexity: int,
) -> dict[str, object]:
    r = context.r
    eta = np.asarray(eta_config["eta"], dtype=float)
    delta_psi = eta * delta_phi
    g_trial = context.g_bar + delta_g
    v_trial = profiles.velocity_from_acceleration(r, g_trial)

    sigma_delta = np.maximum(2.0 * context.v_obs * context.sigma_v / np.maximum(r, EPS), 1.0)
    rotation_mse = float(np.mean(((v_trial - context.v_obs) / context.sigma_v) ** 2))
    residual_mse = float(np.mean(((delta_g - context.delta_g_target) / sigma_delta) ** 2))

    lensing_proxy = delta_phi + delta_psi
    lensing_mse = float(np.mean(((lensing_proxy - context.lensing_target) / context.sigma_lensing) ** 2))

    m_eff, rho_eff = effective_profiles(r, delta_g)
    delta_positive_fraction = float(np.mean(delta_g >= -1.0e-9))
    mass_monotonic_fraction = float(np.mean(np.diff(m_eff) >= -1.0e-7))
    rho_scale = max(float(np.nanmax(np.abs(rho_eff))), 1.0e-12)
    rho_nonnegative_fraction = float(np.mean(rho_eff >= -1.0e-3 * rho_scale))

    dg_scale = max(float(np.sqrt(np.mean(context.delta_g_target**2))), 1.0)
    r_span = max(float(r[-1] - r[0]), 1.0)
    d_delta_g = np.gradient(delta_g, r, edge_order=2)
    smoothness = float(np.mean((d_delta_g * r_span / dg_scale) ** 2))
    oscillation_score = _oscillation_score(delta_g)

    weak_field_max = float(np.max(np.abs(context.phi_bar + delta_phi)) / C_KMS**2)
    weak_penalty = 0.0
    if weak_field_max > 1.0e-5:
        weak_penalty = float(5_000.0 * ((weak_field_max / 1.0e-5) - 1.0) ** 2)

    path_penalty = (
        300.0 * (1.0 - delta_positive_fraction)
        + 300.0 * (1.0 - mass_monotonic_fraction)
        + 300.0 * (1.0 - rho_nonnegative_fraction)
        + 30.0 * oscillation_score
    )
    total_complexity = int(complexity + int(eta_config["eta_complexity"]))
    complexity_penalty = 0.9 * total_complexity
    total_score = float(
        rotation_mse
        + 0.15 * residual_mse
        + 4.0 * lensing_mse
        + 0.08 * np.log1p(smoothness)
        + path_penalty
        + weak_penalty
        + complexity_penalty
    )

    outer = r > (0.55 * r[-1])
    outer_v = v_trial[outer]
    flatness_metric = float(np.std(outer_v) / max(np.mean(outer_v), 1.0e-12))
    outer_delta_speed = np.sqrt(np.maximum(r[outer] * np.maximum(delta_g[outer], 0.0), 0.0))
    warp_flatness_metric = float(np.std(outer_delta_speed) / max(np.mean(outer_delta_speed), 1.0e-12))

    return {
        "family": family,
        "eta_mode": eta_config["eta_mode"],
        "params": params,
        "eta_params": eta_config["eta_params"],
        "rotation_mse": rotation_mse,
        "residual_mse": residual_mse,
        "lensing_mse": lensing_mse,
        "smoothness": smoothness,
        "path_penalty": path_penalty,
        "weak_field_max": weak_field_max,
        "complexity": total_complexity,
        "total_score": total_score,
        "flatness_metric": flatness_metric,
        "warp_flatness_metric": warp_flatness_metric,
        "delta_g_positive_fraction": delta_positive_fraction,
        "mass_monotonic_fraction": mass_monotonic_fraction,
        "rho_nonnegative_fraction": rho_nonnegative_fraction,
        "oscillation_score": oscillation_score,
        "eta_mean": float(np.mean(eta)),
        "eta_span": float(np.max(eta) - np.min(eta)),
    }


def _solve_nnls(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    if nnls is not None:
        weights, _ = nnls(design, y)
        return weights
    weights, *_ = np.linalg.lstsq(design, y, rcond=None)
    return np.maximum(weights, 0.0)


def radial_basis_warp(context: WarpContext) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Fit a nonnegative basis of cumulative Gaussian warp functions."""
    r = context.r
    centers = np.linspace(2.0, 36.0, 10)
    width = 5.0
    accel_basis = np.column_stack([np.exp(-0.5 * ((r - center) / width) ** 2) for center in centers])
    weighted_design = accel_basis / np.maximum(context.delta_g_target * 0.08, 25.0)[:, None]
    weighted_target = context.delta_g_target / np.maximum(context.delta_g_target * 0.08, 25.0)
    weights = _solve_nnls(weighted_design, weighted_target)
    delta_g = accel_basis @ weights
    delta_phi = cumulative_trapezoid(delta_g, r)
    threshold = 0.01 * max(float(np.max(weights)), 1.0)
    weight_rows = [
        {"center": float(center), "width": width, "weight": float(weight)}
        for center, weight in zip(centers, weights)
        if weight > threshold
    ]
    return delta_phi, delta_g, weight_rows


def run_grid_search(context: WarpContext) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    """Evaluate all warp candidates and return a metrics table."""
    rows: list[dict[str, object]] = []
    eta_configs = eta_configurations(context.r)
    for shape in iter_parametric_shapes(context.r):
        for eta_config in eta_configs:
            rows.append(
                score_warp(
                    context,
                    str(shape["family"]),
                    dict(shape["params"]),
                    np.asarray(shape["delta_phi"], dtype=float),
                    np.asarray(shape["delta_g"], dtype=float),
                    eta_config,
                    int(shape["complexity"]),
                )
            )

    basis_phi, basis_delta_g, basis_weights = radial_basis_warp(context)
    basis_complexity = max(len(basis_weights), 1)
    basis_params = {"terms": basis_complexity, "basis": "cumulative_gaussian", "width": 5.0}
    for eta_config in eta_configs:
        rows.append(
            score_warp(
                context,
                "radial_basis_warp",
                basis_params,
                basis_phi,
                basis_delta_g,
                eta_config,
                basis_complexity,
            )
        )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values("total_score", ascending=True).reset_index(drop=True)
    return frame, basis_weights


def reconstruct_candidate(row: pd.Series | dict[str, object], context: WarpContext) -> dict[str, np.ndarray]:
    """Reconstruct arrays for a grid row."""
    family = str(row["family"])
    params = row["params"]
    if isinstance(params, str):
        params = json.loads(params)
    if family == "radial_basis_warp":
        delta_phi, delta_g, _ = radial_basis_warp(context)
    else:
        delta_phi, delta_g = shape_from_params(family, params, context.r)

    eta_mode = str(row["eta_mode"])
    eta = next(cfg["eta"] for cfg in eta_configurations(context.r) if cfg["eta_mode"] == eta_mode)
    eta = np.asarray(eta, dtype=float)
    delta_psi = eta * delta_phi
    v_trial = profiles.velocity_from_acceleration(context.r, context.g_bar + delta_g)
    m_eff, rho_eff = effective_profiles(context.r, delta_g)
    return {
        "delta_phi": delta_phi,
        "delta_g": delta_g,
        "delta_psi": delta_psi,
        "eta": eta,
        "v_trial": v_trial,
        "m_eff": m_eff,
        "rho_eff": rho_eff,
        "lensing_proxy": delta_phi + delta_psi,
    }


def summarize_families(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a compact family-level summary table."""
    grouped = []
    for family, group in frame.groupby("family"):
        best = group.sort_values("total_score").iloc[0]
        grouped.append(
            {
                "family": family,
                "count": int(len(group)),
                "best_total_score": float(best["total_score"]),
                "best_rotation_mse": float(best["rotation_mse"]),
                "best_lensing_mse": float(best["lensing_mse"]),
                "best_eta_mode": best["eta_mode"],
                "best_params": best["params"],
                "best_flatness_metric": float(best["flatness_metric"]),
                "best_weak_field_max": float(best["weak_field_max"]),
            }
        )
    return pd.DataFrame(grouped).sort_values("best_total_score", ascending=True).reset_index(drop=True)


def dataframe_for_csv(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert dict-valued columns to JSON strings for stable CSV output."""
    out = frame.copy()
    for col in out.columns:
        out[col] = out[col].apply(lambda value: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value)
    return out
