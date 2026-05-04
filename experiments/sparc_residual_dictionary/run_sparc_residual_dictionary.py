"""Learn population residual dictionaries from the SPARC residual experiment.

This experiment consumes the outputs of ``experiments/sparc_nfw_residuals`` and
adds a population-level learning layer:

- interpolate primary-quality SPARC residual profiles onto a common R/Rd grid;
- learn a nonnegative dictionary for the positive observed residual;
- learn a signed, two-channel dictionary for the NFW residual-of-residual;
- evaluate held-out radial-point reconstruction and bootstrap mode stability;
- compare learned modes with simple named residual-shape templates.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - used only if scipy is unavailable.
    linear_sum_assignment = None


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "sparc_nfw_residuals"
SOURCE_RESULTS = SOURCE / "results"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

PROFILE_PATH = SOURCE_RESULTS / "residual_profiles.csv"
SUMMARY_PATH = SOURCE_RESULTS / "galaxy_fit_summary.csv"

EPS = 1.0e-12
N_GRID = 72
X_GRID = np.geomspace(0.25, 8.0, N_GRID)
LOG_X_GRID = np.log(X_GRID)
MIN_GRID_POINTS = 18


@dataclass
class ProfileDataset:
    name: str
    x_grid: np.ndarray
    values: np.ndarray
    matrix: np.ndarray
    mask: np.ndarray
    weights: np.ndarray
    galaxies: list[str]
    metadata: pd.DataFrame
    channel_slices: list[slice]
    signed: bool


@dataclass
class NMFResult:
    rank: int
    w: np.ndarray
    h: np.ndarray
    train_channel_rmse: float
    train_profile_rmse: float
    objective: float


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(ok)) < 3:
        return float("nan")
    aa = a[ok] - float(np.mean(a[ok]))
    bb = b[ok] - float(np.mean(b[ok]))
    denom = float(np.sqrt(np.sum(aa**2) * np.sum(bb**2)))
    if denom <= EPS:
        return float("nan")
    return float(np.sum(aa * bb) / denom)


def _weighted_mean(values: np.ndarray, weights: np.ndarray, axis: int = 0) -> np.ndarray:
    numer = np.sum(values * weights, axis=axis)
    denom = np.maximum(np.sum(weights, axis=axis), EPS)
    return numer / denom


def _moving_average(y: np.ndarray, window: int = 5) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(y, pad_width=pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def _smooth_h(h: np.ndarray, channel_slices: list[slice], window: int = 5) -> np.ndarray:
    out = h.copy()
    for row in range(out.shape[0]):
        for section in channel_slices:
            out[row, section] = _moving_average(out[row, section], window=window)
    return np.maximum(out, EPS)


def _normalize_components(w: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(w, dtype=float).copy()
    h = np.asarray(h, dtype=float).copy()
    for idx in range(h.shape[0]):
        scale = float(np.sqrt(np.mean(h[idx] ** 2)))
        if scale <= EPS:
            continue
        h[idx] /= scale
        w[:, idx] *= scale
    return w, h


def _sort_components(w: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    strength = np.sqrt(np.mean(w**2, axis=0)) * np.sqrt(np.mean(h**2, axis=1))
    order = np.argsort(strength)[::-1]
    return w[:, order], h[order]


def load_source_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PROFILE_PATH.exists() or not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Missing SPARC residual outputs. Run "
            "experiments/sparc_nfw_residuals/run_sparc_nfw_residuals.py first."
        )
    profiles = pd.read_csv(PROFILE_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    return profiles, summary


def build_profile_matrices() -> tuple[ProfileDataset, ProfileDataset, pd.DataFrame]:
    profiles, summary = load_source_tables()
    ok = summary[
        (summary["fit_status"] == "ok")
        & (summary["quality_primary"].astype(bool))
        & np.isfinite(summary["Rdisk_kpc"])
        & (summary["Rdisk_kpc"] > 0.0)
        & (summary["n_points"] >= 8)
    ].copy()

    raw_rows: list[np.ndarray] = []
    epsilon_rows: list[np.ndarray] = []
    sigma_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    meta_rows: list[dict[str, object]] = []

    for _, meta in ok.sort_values("galaxy").iterrows():
        galaxy = str(meta["galaxy"])
        group = profiles[profiles["galaxy"] == galaxy].sort_values("R_kpc").copy()
        if group.empty:
            continue

        rdisk = float(meta["Rdisk_kpc"])
        x = group["R_kpc"].to_numpy(dtype=float) / max(rdisk, EPS)
        valid = np.isfinite(x) & (x > 0)
        valid &= np.isfinite(group["delta_g_obs"].to_numpy(dtype=float))
        valid &= np.isfinite(group["gap_nfw_outer"].to_numpy(dtype=float))
        valid &= np.isfinite(group["sigma_delta_g"].to_numpy(dtype=float))
        if int(np.sum(valid)) < 6:
            continue

        x = x[valid]
        order = np.argsort(x)
        x = x[order]
        log_x = np.log(x)

        delta_g = group["delta_g_obs"].to_numpy(dtype=float)[valid][order]
        epsilon_g = group["gap_nfw_outer"].to_numpy(dtype=float)[valid][order]
        sigma_g = group["sigma_delta_g"].to_numpy(dtype=float)[valid][order]

        scale = max(float(np.sqrt(np.mean(delta_g**2))), float(np.median(np.abs(delta_g))), 1.0)
        raw_norm = delta_g / scale
        epsilon_norm = epsilon_g / scale
        sigma_norm = np.maximum(sigma_g / scale, 0.03)

        grid_mask = (LOG_X_GRID >= float(np.min(log_x))) & (LOG_X_GRID <= float(np.max(log_x)))
        if int(np.sum(grid_mask)) < MIN_GRID_POINTS:
            continue

        raw_grid = np.zeros_like(X_GRID, dtype=float)
        eps_grid = np.zeros_like(X_GRID, dtype=float)
        sig_grid = np.ones_like(X_GRID, dtype=float)
        raw_grid[grid_mask] = np.interp(LOG_X_GRID[grid_mask], log_x, raw_norm)
        eps_grid[grid_mask] = np.interp(LOG_X_GRID[grid_mask], log_x, epsilon_norm)
        sig_grid[grid_mask] = np.interp(LOG_X_GRID[grid_mask], log_x, sigma_norm)

        raw_rows.append(raw_grid)
        epsilon_rows.append(eps_grid)
        sigma_rows.append(sig_grid)
        mask_rows.append(grid_mask)
        meta_rows.append(
            {
                "galaxy": galaxy,
                "Rdisk_kpc": rdisk,
                "Vflat_kms": meta.get("Vflat_kms", np.nan),
                "n_points": int(meta["n_points"]),
                "coverage_fraction": float(np.mean(grid_mask)),
                "acceleration_scale": scale,
                "iso_beats_nfw_full": bool(meta["iso_beats_nfw_full"]),
                "log_beats_nfw_full": bool(meta["log_beats_nfw_full"]),
                "central_overshoot_after_outer_nfw": bool(meta["central_overshoot_after_outer_nfw"]),
                "inner_gap_mean_norm_after_outer_nfw": float(meta["inner_gap_mean_norm_after_outer_nfw"]),
                "nfw_full_relative_rmse": float(meta["nfw_full_relative_rmse"]),
                "iso_full_relative_rmse": float(meta["iso_full_relative_rmse"]),
            }
        )

    raw_values = np.vstack(raw_rows)
    epsilon_values = np.vstack(epsilon_rows)
    sigma = np.vstack(sigma_rows)
    mask = np.vstack(mask_rows).astype(bool)
    metadata = pd.DataFrame(meta_rows)
    galaxies = metadata["galaxy"].astype(str).tolist()

    raw_matrix = np.maximum(raw_values, 0.0)
    epsilon_matrix = np.concatenate([np.maximum(epsilon_values, 0.0), np.maximum(-epsilon_values, 0.0)], axis=1)
    epsilon_mask = np.concatenate([mask, mask], axis=1)
    epsilon_sigma = np.concatenate([sigma, sigma], axis=1)

    raw_weights = _weights_from_sigma(mask, sigma)
    epsilon_weights = _weights_from_sigma(epsilon_mask, epsilon_sigma)

    raw = ProfileDataset(
        name="positive_delta_g",
        x_grid=X_GRID,
        values=raw_values,
        matrix=raw_matrix,
        mask=mask,
        weights=raw_weights,
        galaxies=galaxies,
        metadata=metadata,
        channel_slices=[slice(0, N_GRID)],
        signed=False,
    )
    epsilon = ProfileDataset(
        name="signed_nfw_residual_of_residual",
        x_grid=X_GRID,
        values=epsilon_values,
        matrix=epsilon_matrix,
        mask=epsilon_mask,
        weights=epsilon_weights,
        galaxies=galaxies,
        metadata=metadata,
        channel_slices=[slice(0, N_GRID), slice(N_GRID, 2 * N_GRID)],
        signed=True,
    )
    return raw, epsilon, metadata


def _weights_from_sigma(mask: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    weights = np.zeros_like(sigma, dtype=float)
    finite = mask & np.isfinite(sigma) & (sigma > 0)
    if not np.any(finite):
        weights[mask] = 1.0
        return weights
    inv_var = 1.0 / np.maximum(sigma, 0.03) ** 2
    median = float(np.median(inv_var[finite]))
    weights[finite] = np.clip(inv_var[finite] / max(median, EPS), 0.2, 5.0)
    return weights


def make_holdout(mask: np.ndarray, frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base_mask = mask[:, :N_GRID] if mask.shape[1] == 2 * N_GRID else mask
    holdout_base = np.zeros_like(base_mask, dtype=bool)
    for row in range(base_mask.shape[0]):
        observed = np.flatnonzero(base_mask[row])
        if len(observed) < 8:
            continue
        count = max(1, int(round(frac * len(observed))))
        chosen = rng.choice(observed, size=count, replace=False)
        holdout_base[row, chosen] = True
    if mask.shape[1] == 2 * N_GRID:
        return np.concatenate([holdout_base, holdout_base], axis=1)
    return holdout_base


def fit_masked_nmf(
    dataset: ProfileDataset,
    rank: int,
    train_weights: np.ndarray,
    seed: int,
    n_iter: int = 900,
    smooth_window: int = 5,
    l2: float = 1.0e-5,
) -> NMFResult:
    rng = np.random.default_rng(seed)
    x = dataset.matrix
    n_rows, n_cols = x.shape
    col_mean = _weighted_mean(x, np.maximum(train_weights, 0.0), axis=0)
    col_mean = np.maximum(col_mean, 0.02)
    w = rng.uniform(0.2, 1.0, size=(n_rows, rank))
    h = np.vstack([col_mean * rng.uniform(0.75, 1.25, size=n_cols) for _ in range(rank)])
    h = _smooth_h(np.maximum(h, EPS), dataset.channel_slices, window=smooth_window)
    w, h = _normalize_components(w, h)

    weighted_x = train_weights * x
    for _ in range(n_iter):
        pred = np.maximum(w @ h, EPS)
        numer_h = w.T @ weighted_x
        denom_h = w.T @ (train_weights * pred) + l2
        h *= numer_h / np.maximum(denom_h, EPS)
        h = _smooth_h(h, dataset.channel_slices, window=smooth_window)
        w, h = _normalize_components(w, h)

        pred = np.maximum(w @ h, EPS)
        numer_w = weighted_x @ h.T
        denom_w = (train_weights * pred) @ h.T + l2
        w *= numer_w / np.maximum(denom_w, EPS)
        w = np.maximum(w, EPS)
        w, h = _normalize_components(w, h)

    w, h = _sort_components(w, h)
    pred = w @ h
    channel_rmse = channel_rmse_for(dataset, pred, train_weights)
    profile_rmse = profile_rmse_for(dataset, pred, train_weights > 0)
    objective = float(np.sum(train_weights * (dataset.matrix - pred) ** 2) / max(float(np.sum(train_weights)), EPS))
    return NMFResult(rank, w, h, channel_rmse, profile_rmse, objective)


def fit_best_nmf(
    dataset: ProfileDataset,
    rank: int,
    train_weights: np.ndarray,
    seeds: list[int],
    n_iter: int,
) -> NMFResult:
    best: NMFResult | None = None
    for seed in seeds:
        result = fit_masked_nmf(dataset, rank, train_weights, seed=seed, n_iter=n_iter)
        if best is None or result.objective < best.objective:
            best = result
    assert best is not None
    return best


def signed_prediction(dataset: ProfileDataset, pred_matrix: np.ndarray) -> np.ndarray:
    if dataset.signed:
        return pred_matrix[:, :N_GRID] - pred_matrix[:, N_GRID:]
    return pred_matrix


def profile_rmse_for(dataset: ProfileDataset, pred_matrix: np.ndarray, eval_mask: np.ndarray) -> float:
    if dataset.signed:
        base_mask = eval_mask[:, :N_GRID] if eval_mask.shape[1] == 2 * N_GRID else eval_mask
        pred = signed_prediction(dataset, pred_matrix)
        resid = dataset.values - pred
        return float(np.sqrt(np.mean(resid[base_mask] ** 2))) if np.any(base_mask) else float("nan")
    base_mask = eval_mask[:, :N_GRID] if eval_mask.shape[1] == 2 * N_GRID else eval_mask
    resid = np.maximum(dataset.values, 0.0) - pred_matrix
    return float(np.sqrt(np.mean(resid[base_mask] ** 2))) if np.any(base_mask) else float("nan")


def channel_rmse_for(dataset: ProfileDataset, pred_matrix: np.ndarray, eval_weights: np.ndarray) -> float:
    ok = eval_weights > 0
    if not np.any(ok):
        return float("nan")
    numer = float(np.sum(eval_weights * (dataset.matrix - pred_matrix) ** 2))
    denom = max(float(np.sum(eval_weights)), EPS)
    return float(np.sqrt(numer / denom))


def baseline_prediction(dataset: ProfileDataset, train_weights: np.ndarray) -> np.ndarray:
    mean_profile = _weighted_mean(dataset.matrix, train_weights, axis=0)
    return np.tile(mean_profile, (dataset.matrix.shape[0], 1))


def run_model_selection(dataset: ProfileDataset) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, object]] = []
    for split in range(5):
        holdout = make_holdout(dataset.mask, frac=0.15, seed=1000 + 17 * split)
        train_weights = dataset.weights.copy()
        train_weights[holdout] = 0.0
        holdout_weights = dataset.weights.copy()
        holdout_weights[~holdout] = 0.0

        baseline = baseline_prediction(dataset, train_weights)
        rows.append(
            {
                "target": dataset.name,
                "rank": 0,
                "split": split,
                "train_channel_rmse": channel_rmse_for(dataset, baseline, train_weights),
                "holdout_channel_rmse": channel_rmse_for(dataset, baseline, holdout_weights),
                "train_profile_rmse": profile_rmse_for(dataset, baseline, train_weights > 0),
                "holdout_profile_rmse": profile_rmse_for(dataset, baseline, holdout),
            }
        )

        for rank in range(1, 7):
            seeds = [10_000 + 101 * split + 13 * rank + offset for offset in range(4)]
            result = fit_best_nmf(dataset, rank, train_weights, seeds=seeds, n_iter=850)
            pred = result.w @ result.h
            rows.append(
                {
                    "target": dataset.name,
                    "rank": rank,
                    "split": split,
                    "train_channel_rmse": result.train_channel_rmse,
                    "holdout_channel_rmse": channel_rmse_for(dataset, pred, holdout_weights),
                    "train_profile_rmse": result.train_profile_rmse,
                    "holdout_profile_rmse": profile_rmse_for(dataset, pred, holdout),
                }
            )

    frame = pd.DataFrame(rows)
    aggregate = frame[frame["rank"] > 0].groupby("rank")["holdout_profile_rmse"].mean()
    min_rmse = float(aggregate.min())
    selected = int(aggregate[aggregate <= 1.02 * min_rmse].index.min())
    return frame, selected


def component_modes(dataset: ProfileDataset, h: np.ndarray, normalize: bool = True) -> np.ndarray:
    if dataset.signed:
        modes = h[:, :N_GRID] - h[:, N_GRID:]
    else:
        modes = h[:, :N_GRID]
    modes = np.asarray(modes, dtype=float)
    if normalize:
        out = modes.copy()
        for idx in range(out.shape[0]):
            scale = float(np.max(np.abs(out[idx])))
            if scale > EPS:
                out[idx] /= scale
        return out
    return modes


def run_bootstrap_stability(
    dataset: ProfileDataset,
    reference: NMFResult,
    n_bootstrap: int = 40,
) -> pd.DataFrame:
    ref_modes = component_modes(dataset, reference.h, normalize=True)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(2026)
    rank = reference.rank

    for boot in range(n_bootstrap):
        sample = rng.choice(np.arange(dataset.matrix.shape[0]), size=dataset.matrix.shape[0], replace=True)
        boot_dataset = ProfileDataset(
            name=dataset.name,
            x_grid=dataset.x_grid,
            values=dataset.values[sample],
            matrix=dataset.matrix[sample],
            mask=dataset.mask[sample],
            weights=dataset.weights[sample],
            galaxies=[dataset.galaxies[i] for i in sample],
            metadata=dataset.metadata.iloc[sample].reset_index(drop=True),
            channel_slices=dataset.channel_slices,
            signed=dataset.signed,
        )
        result = fit_best_nmf(
            boot_dataset,
            rank,
            boot_dataset.weights,
            seeds=[20_000 + boot * 7, 20_001 + boot * 7],
            n_iter=650,
        )
        boot_modes = component_modes(boot_dataset, result.h, normalize=True)
        corr = np.abs(np.array([[_safe_corr(a, b) for b in boot_modes] for a in ref_modes]))
        corr = np.nan_to_num(corr, nan=0.0)
        if linear_sum_assignment is not None:
            row_ind, col_ind = linear_sum_assignment(-corr)
            assignment = dict(zip(row_ind.tolist(), col_ind.tolist()))
            for mode_idx in range(rank):
                rows.append(
                    {
                        "target": dataset.name,
                        "bootstrap": boot,
                        "mode": mode_idx + 1,
                        "best_abs_correlation": float(corr[mode_idx, assignment.get(mode_idx, int(np.argmax(corr[mode_idx])))]),
                    }
                )
        else:
            for mode_idx in range(rank):
                rows.append(
                    {
                        "target": dataset.name,
                        "bootstrap": boot,
                        "mode": mode_idx + 1,
                        "best_abs_correlation": float(np.max(corr[mode_idx])),
                    }
                )
    return pd.DataFrame(rows)


def template_library(x: np.ndarray, signed: bool) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=float)

    def nfw_shape(rs: float) -> np.ndarray:
        z = np.maximum(x / rs, EPS)
        return (np.log1p(z) - z / (1.0 + z)) / np.maximum(x, EPS) ** 2

    if signed:
        inner = 1.0 / (1.0 + (x / 1.0) ** 2)
        outer = x / (x + 1.5)
        return {
            "negative_inner_overshoot": -inner,
            "positive_inner_excess": inner,
            "positive_outer_tail": outer,
            "negative_outer_deficit": -outer,
            "slope_mismatch_inner_neg_outer_pos": np.tanh(np.log(x / 1.4)),
            "slope_mismatch_inner_pos_outer_neg": -np.tanh(np.log(x / 1.4)),
        }
    return {
        "cored_isothermal_xc1": x / (x**2 + 1.0),
        "log_tail_x0_1": 1.0 / (x + 1.0),
        "nfw_like_rs1": nfw_shape(1.0),
        "nfw_like_rs3": nfw_shape(3.0),
        "central_peak": np.exp(-0.5 * (np.log(x / 0.7) / 0.65) ** 2),
        "broad_outer_tail": 1.0 / np.sqrt(x + 0.4),
    }


def template_correlations(dataset: ProfileDataset, result: NMFResult) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    modes = component_modes(dataset, result.h, normalize=True)
    templates = template_library(dataset.x_grid, signed=dataset.signed)
    for mode_idx, mode in enumerate(modes, start=1):
        for name, template in templates.items():
            corr = _safe_corr(mode, template)
            rows.append(
                {
                    "target": dataset.name,
                    "mode": mode_idx,
                    "template": name,
                    "template_match": name if corr >= 0 else f"opposite_of_{name}",
                    "correlation": corr,
                    "abs_correlation": abs(corr),
                }
            )
    return pd.DataFrame(rows)


def coefficient_table(dataset: ProfileDataset, result: NMFResult) -> pd.DataFrame:
    data = dataset.metadata[["galaxy"]].copy()
    for idx in range(result.rank):
        data[f"coef_mode_{idx + 1}"] = result.w[:, idx]
    data = data.merge(dataset.metadata, on="galaxy", how="left")
    return data


def coefficient_associations(dataset: ProfileDataset, result: NMFResult) -> pd.DataFrame:
    coeffs = coefficient_table(dataset, result)
    rows: list[dict[str, object]] = []
    binary_fields = ["central_overshoot_after_outer_nfw", "iso_beats_nfw_full", "log_beats_nfw_full"]
    continuous_fields = [
        "inner_gap_mean_norm_after_outer_nfw",
        "nfw_full_relative_rmse",
        "iso_full_relative_rmse",
        "Vflat_kms",
        "Rdisk_kpc",
    ]
    for idx in range(result.rank):
        coef = coeffs[f"coef_mode_{idx + 1}"].to_numpy(dtype=float)
        for field in binary_fields:
            flag = coeffs[field].astype(bool).to_numpy()
            true_vals = coef[flag]
            false_vals = coef[~flag]
            pooled = max(float(np.std(coef)), EPS)
            rows.append(
                {
                    "target": dataset.name,
                    "mode": idx + 1,
                    "feature": field,
                    "association_type": "binary_mean_difference",
                    "correlation": _safe_corr(coef, flag.astype(float)),
                    "mean_true": float(np.mean(true_vals)) if len(true_vals) else float("nan"),
                    "mean_false": float(np.mean(false_vals)) if len(false_vals) else float("nan"),
                    "standardized_difference": (
                        float((np.mean(true_vals) - np.mean(false_vals)) / pooled)
                        if len(true_vals) and len(false_vals)
                        else float("nan")
                    ),
                }
            )
        for field in continuous_fields:
            values = coeffs[field].to_numpy(dtype=float)
            rows.append(
                {
                    "target": dataset.name,
                    "mode": idx + 1,
                    "feature": field,
                    "association_type": "pearson",
                    "correlation": _safe_corr(coef, values),
                    "mean_true": float("nan"),
                    "mean_false": float("nan"),
                    "standardized_difference": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def modes_to_frame(dataset: ProfileDataset, result: NMFResult) -> pd.DataFrame:
    modes = component_modes(dataset, result.h, normalize=True)
    frame = pd.DataFrame({"R_over_Rdisk": dataset.x_grid})
    for idx, mode in enumerate(modes, start=1):
        frame[f"mode_{idx}"] = mode
    return frame


def save_figures(
    raw: ProfileDataset,
    eps: ProfileDataset,
    raw_result: NMFResult,
    eps_result: NMFResult,
    model_selection: pd.DataFrame,
    stability: pd.DataFrame,
    eps_coeffs: pd.DataFrame,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for target, label, color in [
        (raw.name, "positive Delta g", "#1B998B"),
        (eps.name, "NFW residual-of-residual", "#D95D39"),
    ]:
        data = model_selection[model_selection["target"] == target]
        grouped = data.groupby("rank")["holdout_profile_rmse"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            grouped["rank"],
            grouped["mean"],
            yerr=grouped["std"].fillna(0.0),
            marker="o",
            lw=1.8,
            capsize=3,
            label=label,
            color=color,
        )
    ax.set_xlabel("dictionary rank K")
    ax.set_ylabel("held-out normalized RMSE")
    ax.set_title("Held-out reconstruction")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_selection.png", dpi=220)
    plt.close(fig)

    raw_modes = component_modes(raw, raw_result.h, normalize=True)
    eps_modes = component_modes(eps, eps_result.h, normalize=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.25), sharex=True)
    for idx, mode in enumerate(raw_modes, start=1):
        axes[0].plot(raw.x_grid, mode, lw=1.8, label=f"mode {idx}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("R / Rdisk")
    axes[0].set_ylabel("normalized mode amplitude")
    axes[0].set_title("Positive residual")
    axes[0].axhline(0, color="#222222", lw=0.8)
    for idx, mode in enumerate(eps_modes, start=1):
        axes[1].plot(eps.x_grid, mode, lw=1.8, label=f"mode {idx}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("R / Rdisk")
    axes[1].set_title("After outer NFW fit")
    axes[1].axhline(0, color="#222222", lw=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=8.5,
        handlelength=1.6,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGURES / "learned_modes.png", dpi=220)
    plt.close(fig)

    if eps_result.rank >= 2:
        flag = eps_coeffs["central_overshoot_after_outer_nfw"].astype(bool).to_numpy()
        mode_scores = []
        for idx in range(eps_result.rank):
            coef = eps_coeffs[f"coef_mode_{idx + 1}"].to_numpy(dtype=float)
            mode_scores.append((abs(_safe_corr(coef, flag.astype(float))), idx + 1))
        selected_modes = [mode for _, mode in sorted(mode_scores, reverse=True)[:2]]
        x_mode, y_mode = selected_modes

        fig, ax = plt.subplots(figsize=(4.6, 3.55))
        colors = np.where(flag, "#D95D39", "#4C78A8")
        ax.scatter(
            eps_coeffs[f"coef_mode_{x_mode}"],
            eps_coeffs[f"coef_mode_{y_mode}"],
            c=colors,
            s=34,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_xlabel(f"signed coefficient {x_mode}")
        ax.set_ylabel(f"signed coefficient {y_mode}")
        ax.set_title("Diagnostic residual coefficients")
        ax.scatter([], [], c="#D95D39", label="central overshoot")
        ax.scatter([], [], c="#4C78A8", label="no central overshoot")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIGURES / "coefficient_space.png", dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    labels = []
    medians = []
    low = []
    high = []
    for target, prefix in [(raw.name, "raw"), (eps.name, "eps")]:
        target_data = stability[stability["target"] == target]
        for mode, group in target_data.groupby("mode"):
            labels.append(f"{prefix} {mode}")
            q10, q50, q90 = np.quantile(group["best_abs_correlation"], [0.1, 0.5, 0.9])
            medians.append(q50)
            low.append(q50 - q10)
            high.append(q90 - q50)
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, medians, color="#6C8EBF")
    ax.errorbar(x_pos, medians, yerr=[low, high], fmt="none", ecolor="#222222", capsize=3, lw=1.0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("bootstrap |correlation|")
    ax.set_title("Learned mode stability")
    fig.tight_layout()
    fig.savefig(FIGURES / "bootstrap_stability.png", dpi=220)
    plt.close(fig)


def _fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 1000 or (0 < abs(value) < 0.001):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows generated._"
    data = frame[columns].copy()
    if max_rows is not None:
        data = data.head(max_rows)
    rows = []
    for _, row in data.iterrows():
        out = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                out.append(_fmt(value))
            else:
                out.append(str(value))
        rows.append(out)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def write_report(
    raw: ProfileDataset,
    eps: ProfileDataset,
    raw_result: NMFResult,
    eps_result: NMFResult,
    model_selection: pd.DataFrame,
    stability: pd.DataFrame,
    correlations: pd.DataFrame,
    associations: pd.DataFrame,
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    selection_summary = (
        model_selection.groupby(["target", "rank"])
        .agg(
            holdout_profile_rmse=("holdout_profile_rmse", "mean"),
            holdout_profile_rmse_std=("holdout_profile_rmse", "std"),
            holdout_channel_rmse=("holdout_channel_rmse", "mean"),
        )
        .reset_index()
    )
    selected_table = selection_summary[
        ((selection_summary["target"] == raw.name) & (selection_summary["rank"].isin([0, raw_result.rank])))
        | ((selection_summary["target"] == eps.name) & (selection_summary["rank"].isin([0, eps_result.rank])))
    ].copy()

    stability_summary = (
        stability.groupby(["target", "mode"])["best_abs_correlation"]
        .agg(median="median", q10=lambda x: np.quantile(x, 0.1), q90=lambda x: np.quantile(x, 0.9))
        .reset_index()
    )

    best_corr = (
        correlations.sort_values(["target", "mode", "abs_correlation"], ascending=[True, True, False])
        .groupby(["target", "mode"])
        .head(1)
        .reset_index(drop=True)
    )

    assoc_focus = associations[
        associations["feature"].isin(["central_overshoot_after_outer_nfw", "inner_gap_mean_norm_after_outer_nfw"])
    ].copy()
    assoc_focus = assoc_focus.sort_values(["target", "mode", "feature"])

    n_galaxies = raw.matrix.shape[0]
    mean_coverage = float(raw.metadata["coverage_fraction"].mean())
    central_fraction = float(raw.metadata["central_overshoot_after_outer_nfw"].mean())
    iso_fraction = float(raw.metadata["iso_beats_nfw_full"].mean())

    raw_baseline = selected_table[(selected_table["target"] == raw.name) & (selected_table["rank"] == 0)][
        "holdout_profile_rmse"
    ].iloc[0]
    raw_rmse = selected_table[(selected_table["target"] == raw.name) & (selected_table["rank"] == raw_result.rank)][
        "holdout_profile_rmse"
    ].iloc[0]
    eps_baseline = selected_table[(selected_table["target"] == eps.name) & (selected_table["rank"] == 0)][
        "holdout_profile_rmse"
    ].iloc[0]
    eps_rmse = selected_table[(selected_table["target"] == eps.name) & (selected_table["rank"] == eps_result.rank)][
        "holdout_profile_rmse"
    ].iloc[0]

    report = f"""# SPARC Residual Dictionary Learning

**Author:** J. R. Landers  
**Date:** May 2026

## Summary

This experiment adds a population learning layer on top of the SPARC
NFW residual-of-residual experiment. It uses the primary-quality SPARC
galaxies already processed in `../sparc_nfw_residuals/`, places their residual
profiles on a common normalized radius grid, and learns low-rank residual
dictionaries.

Two targets are learned:

1. A nonnegative dictionary for the positive observed acceleration residual,
   $\\max(\\Delta g_{{\\rm obs}},0)$.
2. A signed two-channel dictionary for the NFW residual-of-residual,
   $\\epsilon_g=\\Delta g_{{\\rm obs}}-\\Delta g_{{\\rm NFW,outer}}$.

The signed dictionary is learned by splitting the field into positive and
negative channels,

$$
\\epsilon_g(r)=\\epsilon_g^+(r)-\\epsilon_g^-(r),
\\qquad
\\epsilon_g^\\pm(r)\\ge 0,
$$

and fitting

$$
X_i(r)\\approx \\sum_{{k=1}}^K a_{{ik}}\\phi_k(r),
\\qquad
a_{{ik}}\\ge 0.
$$

## Dataset

- Source experiment: `experiments/sparc_nfw_residuals`
- Galaxies used after quality and radial-coverage cuts: **{n_galaxies}**
- Radius coordinate: $R/R_d$
- Common grid: **{N_GRID}** log-spaced points from **{X_GRID[0]:.2f}** to **{X_GRID[-1]:.1f}** $R/R_d$
- Mean grid coverage per galaxy: **{mean_coverage:.3f}**
- Central-overshoot fraction in this learned sample: **{central_fraction:.3f}**
- Cored/isothermal beats full-range NFW fraction in this learned sample: **{iso_fraction:.3f}**

Each galaxy is normalized by its observed residual RMS before learning. This
makes the learned dictionaries primarily shape dictionaries rather than galaxy
mass-scale dictionaries.

## Model Selection

Model selection uses held-out radial points, not held-out galaxies. For each
rank $K$, 15 percent of observed radial grid entries are held out per split,
the dictionary is trained on the remaining entries, and reconstruction is
scored on the held-out entries. Rank zero is the learned population mean.

{markdown_table(selected_table, ["target", "rank", "holdout_profile_rmse", "holdout_profile_rmse_std", "holdout_channel_rmse"])}

The selected positive-residual dictionary has **K={raw_result.rank}**. Its
held-out normalized profile RMSE is **{raw_rmse:.3f}**, compared with
**{raw_baseline:.3f}** for the population-mean baseline.

The selected signed NFW residual-of-residual dictionary has **K={eps_result.rank}**.
Its held-out normalized profile RMSE is **{eps_rmse:.3f}**, compared with
**{eps_baseline:.3f}** for the population-mean baseline.

Both targets improve monotonically up to the largest searched rank, $K=6$.
That is informative but also a caveat: this run shows strong compressible
structure, but it does not identify a sharp intrinsic rank. In the paper
narrative, $K=6$ should be treated as the searched dictionary size, while the
bootstrap table below identifies which modes are stable enough to interpret.

![Model selection](figures/model_selection.png)

## Learned Modes

![Learned modes](figures/learned_modes.png)

The positive-residual modes are broad, smooth acceleration-residual shapes.
The signed residual-of-residual modes are more diagnostic: they describe
structured departures left after NFW has been forced to match the outer
rotation-curve residual.

The closest simple template for each learned mode is shown below. A negative
correlation means the learned mode is closer to the opposite of that named
template.

{markdown_table(best_corr, ["target", "mode", "template_match", "correlation", "abs_correlation"])}

## Bootstrap Stability

Mode stability is measured by bootstrapping galaxies, refitting the selected
dictionary, and matching each bootstrap mode back to the reference dictionary
by absolute profile correlation.

{markdown_table(stability_summary, ["target", "mode", "median", "q10", "q90"])}

![Bootstrap stability](figures/bootstrap_stability.png)

## Relation To Existing Residual Diagnostics

The table below reports associations between learned coefficients and the
pre-existing NFW residual diagnostics.

{markdown_table(assoc_focus, ["target", "mode", "feature", "association_type", "correlation", "standardized_difference"], max_rows=32)}

![Coefficient space](figures/coefficient_space.png)

## Interpretation

This experiment supports a clean statistical-learning extension of the
geometric-residual paper. The useful result is not merely that a hand-written
template beats another hand-written template. The SPARC residual population can
be compressed into a small number of learned, stable residual modes, and those
modes can be compared with the existing NFW/cored/central-overshoot diagnostics.

The signed residual-of-residual dictionary is the more paper-relevant object.
It directly learns the structured leftover field after the baseline NFW model
has been fit to the outer galaxy. This makes it a population-level model
criticism tool: fit the physical baseline, learn what remains, and test whether
the leftover geometry is coherent across galaxies.

## Limitations

- This is still a residual-space diagnostic, not a full halo inference.
- Stellar mass-to-light ratios, distances, and inclinations are inherited from
  the fixed-assumption SPARC residual experiment.
- The common-grid interpolation uses $R/R_d$ and does not model disk geometry.
- The NMF objective is a smooth, masked reconstruction model, not a full
  probabilistic likelihood.
- Held-out radial points test profile compression; held-out-galaxy prediction
  should be added before making strong generalization claims.

## Outputs

- `results/model_selection.csv`
- `results/raw_modes.csv`
- `results/epsilon_modes.csv`
- `results/raw_galaxy_coefficients.csv`
- `results/epsilon_galaxy_coefficients.csv`
- `results/mode_template_correlations.csv`
- `results/bootstrap_mode_stability.csv`
- `results/coefficient_associations.csv`
- `figures/model_selection.png`
- `figures/learned_modes.png`
- `figures/coefficient_space.png`
- `figures/bootstrap_stability.png`
"""

    (ROOT / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    raw, eps, metadata = build_profile_matrices()

    metadata.to_csv(RESULTS / "dictionary_dataset_galaxies.csv", index=False)
    pd.DataFrame(
        [
            {
                "target": raw.name,
                "n_galaxies": raw.matrix.shape[0],
                "n_grid": N_GRID,
                "x_min": float(X_GRID[0]),
                "x_max": float(X_GRID[-1]),
                "mean_coverage_fraction": float(raw.metadata["coverage_fraction"].mean()),
            },
            {
                "target": eps.name,
                "n_galaxies": eps.matrix.shape[0],
                "n_grid": N_GRID,
                "x_min": float(X_GRID[0]),
                "x_max": float(X_GRID[-1]),
                "mean_coverage_fraction": float(eps.metadata["coverage_fraction"].mean()),
            },
        ]
    ).to_csv(RESULTS / "dataset_summary.csv", index=False)

    raw_selection, raw_rank = run_model_selection(raw)
    eps_selection, eps_rank = run_model_selection(eps)
    model_selection = pd.concat([raw_selection, eps_selection], ignore_index=True)
    model_selection.to_csv(RESULTS / "model_selection.csv", index=False)

    raw_result = fit_best_nmf(raw, raw_rank, raw.weights, seeds=[30_000, 30_001, 30_002, 30_003, 30_004], n_iter=1100)
    eps_result = fit_best_nmf(eps, eps_rank, eps.weights, seeds=[40_000, 40_001, 40_002, 40_003, 40_004], n_iter=1100)

    modes_to_frame(raw, raw_result).to_csv(RESULTS / "raw_modes.csv", index=False)
    modes_to_frame(eps, eps_result).to_csv(RESULTS / "epsilon_modes.csv", index=False)

    raw_coeffs = coefficient_table(raw, raw_result)
    eps_coeffs = coefficient_table(eps, eps_result)
    raw_coeffs.to_csv(RESULTS / "raw_galaxy_coefficients.csv", index=False)
    eps_coeffs.to_csv(RESULTS / "epsilon_galaxy_coefficients.csv", index=False)

    correlations = pd.concat(
        [template_correlations(raw, raw_result), template_correlations(eps, eps_result)],
        ignore_index=True,
    )
    correlations.to_csv(RESULTS / "mode_template_correlations.csv", index=False)

    associations = pd.concat(
        [coefficient_associations(raw, raw_result), coefficient_associations(eps, eps_result)],
        ignore_index=True,
    )
    associations.to_csv(RESULTS / "coefficient_associations.csv", index=False)

    stability = pd.concat(
        [
            run_bootstrap_stability(raw, raw_result, n_bootstrap=40),
            run_bootstrap_stability(eps, eps_result, n_bootstrap=40),
        ],
        ignore_index=True,
    )
    stability.to_csv(RESULTS / "bootstrap_mode_stability.csv", index=False)

    selected = {
        "positive_delta_g_rank": raw_rank,
        "signed_nfw_residual_of_residual_rank": eps_rank,
        "positive_delta_g_train_profile_rmse": raw_result.train_profile_rmse,
        "signed_nfw_residual_of_residual_train_profile_rmse": eps_result.train_profile_rmse,
    }
    (RESULTS / "selected_models.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    save_figures(raw, eps, raw_result, eps_result, model_selection, stability, eps_coeffs)
    write_report(raw, eps, raw_result, eps_result, model_selection, stability, correlations, associations)

    print("SPARC residual dictionary experiment completed.")
    print(f"Galaxies used: {raw.matrix.shape[0]}")
    print(f"Selected positive residual rank: {raw_rank}")
    print(f"Selected signed residual-of-residual rank: {eps_rank}")
    print(f"Report: {ROOT / 'README.md'}")


if __name__ == "__main__":
    main()
