"""Run a real-data SPARC NFW residual-of-residual experiment.

This experiment applies the residual-of-residual diagnostic to public SPARC
rotation-curve data. It is deliberately conservative and reproducible:

- raw SPARC MRT files are downloaded if missing;
- stellar mass-to-light ratios are fixed to common fiducial values;
- NFW and cored/isothermal residual families are fit per galaxy;
- the leftover epsilon_g = Delta g_obs - Delta g_candidate is summarized.

The output is a set of CSV files and figures under this experiment folder.
"""

from __future__ import annotations

import csv
import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - used only if scipy is unavailable.
    least_squares = None


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

MASS_MODELS_URL = "https://astroweb.cwru.edu/SPARC/MassModels_Lelli2016c.mrt"
SAMPLE_TABLE_URL = "https://zenodo.org/records/16284118/files/SPARC_Lelli2016c.mrt?download=1"

MASS_MODELS_PATH = RAW / "MassModels_Lelli2016c.mrt"
SAMPLE_TABLE_PATH = RAW / "SPARC_Lelli2016c.mrt"

G_KPC_KMS2_PER_MSUN = 4.30091e-6
EPS = 1.0e-12

UPSILON_DISK = 0.5
UPSILON_BULGE = 0.7


@dataclass(frozen=True)
class FitResult:
    success: bool
    params: dict[str, float]
    prediction: np.ndarray
    weighted_mse_fit: float
    weighted_mse_full: float
    relative_rmse_full: float
    mean_abs_fractional_error_full: float
    message: str


def ensure_inputs() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    targets = [
        (MASS_MODELS_URL, MASS_MODELS_PATH),
        (SAMPLE_TABLE_URL, SAMPLE_TABLE_PATH),
    ]
    for url, path in targets:
        if path.exists() and path.stat().st_size > 0:
            continue
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def read_mass_models(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.split()
            if len(parts) < 10 or not _looks_numeric(parts[1]):
                continue
            try:
                rows.append(
                    {
                        "galaxy": parts[0],
                        "D_Mpc": float(parts[1]),
                        "R_kpc": float(parts[2]),
                        "Vobs_kms": float(parts[3]),
                        "e_Vobs_kms": float(parts[4]),
                        "Vgas_kms": float(parts[5]),
                        "Vdisk_kms": float(parts[6]),
                        "Vbul_kms": float(parts[7]),
                        "SBdisk": float(parts[8]),
                        "SBbul": float(parts[9]),
                    }
                )
            except ValueError:
                continue
    frame = pd.DataFrame(rows)
    return frame.sort_values(["galaxy", "R_kpc"]).reset_index(drop=True)


def read_sample_table(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.split()
            if len(parts) < 18:
                continue
            try:
                row = {
                    "galaxy": parts[0],
                    "T": int(parts[1]),
                    "D_Mpc": float(parts[2]),
                    "e_D_Mpc": float(parts[3]),
                    "distance_method": int(parts[4]),
                    "Inc_deg": float(parts[5]),
                    "e_Inc_deg": float(parts[6]),
                    "L36_1e9Lsun": float(parts[7]),
                    "e_L36_1e9Lsun": float(parts[8]),
                    "Reff_kpc": float(parts[9]),
                    "SBeff": float(parts[10]),
                    "Rdisk_kpc": float(parts[11]),
                    "SBdisk0": float(parts[12]),
                    "MHI_1e9Msun": float(parts[13]),
                    "RHI_kpc": float(parts[14]),
                    "Vflat_kms": float(parts[15]),
                    "e_Vflat_kms": float(parts[16]),
                    "Q": int(parts[17]),
                    "reference": " ".join(parts[18:]) if len(parts) > 18 else "",
                }
                rows.append(row)
            except ValueError:
                continue
    return pd.DataFrame(rows)


def baryonic_speed_squared(group: pd.DataFrame) -> np.ndarray:
    vgas = group["Vgas_kms"].to_numpy(dtype=float)
    vdisk = group["Vdisk_kms"].to_numpy(dtype=float)
    vbul = group["Vbul_kms"].to_numpy(dtype=float)

    # SPARC permits negative Vgas where the gas contribution is outward in the
    # disk decomposition. Velocity contributions add in quadrature with sign.
    gas_term = np.sign(vgas) * vgas**2
    disk_term = UPSILON_DISK * vdisk**2
    bulge_term = UPSILON_BULGE * vbul**2
    return gas_term + disk_term + bulge_term


def acceleration_residual(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = group["R_kpc"].to_numpy(dtype=float)
    vobs = group["Vobs_kms"].to_numpy(dtype=float)
    evobs = group["e_Vobs_kms"].to_numpy(dtype=float)
    vbar2 = baryonic_speed_squared(group)
    delta_g = (vobs**2 - vbar2) / np.maximum(r, EPS)

    formal_sigma = 2.0 * np.abs(vobs) * np.maximum(evobs, 0.0) / np.maximum(r, EPS)
    systematic_floor = 0.05 * vobs**2 / np.maximum(r, EPS)
    sigma = np.sqrt(formal_sigma**2 + systematic_floor**2)
    sigma = np.maximum(sigma, 0.03 * max(float(np.sqrt(np.mean(delta_g**2))), 1.0))
    return r, delta_g, sigma, vbar2


def nfw_shape(r: np.ndarray, r_s: float) -> np.ndarray:
    x = np.maximum(r / max(r_s, EPS), EPS)
    return np.log1p(x) - x / (1.0 + x)


def nfw_acceleration(r: np.ndarray, log_amp: float, log_r_s: float) -> np.ndarray:
    amp = math.exp(float(log_amp))
    r_s = math.exp(float(log_r_s))
    return amp * nfw_shape(r, r_s) / np.maximum(r, EPS) ** 2


def cored_isothermal_acceleration(r: np.ndarray, log_v0: float, log_r_c: float) -> np.ndarray:
    v0 = math.exp(float(log_v0))
    r_c = math.exp(float(log_r_c))
    return v0**2 * r / (r**2 + r_c**2)


def log_tail_acceleration(r: np.ndarray, log_amp: float, log_r0: float) -> np.ndarray:
    amp = math.exp(float(log_amp))
    r0 = math.exp(float(log_r0))
    return amp / (r + r0)


def fit_positive_model(
    name: str,
    model: Callable[..., np.ndarray],
    param_names: tuple[str, str],
    starts: list[tuple[float, float]],
    bounds: tuple[tuple[float, float], tuple[float, float]],
    r: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    fit_mask: np.ndarray,
) -> FitResult:
    if least_squares is None:
        pred = np.zeros_like(y)
        return FitResult(False, {}, pred, np.inf, np.inf, np.inf, np.inf, "scipy unavailable")

    if int(np.sum(fit_mask)) < 4:
        pred = np.zeros_like(y)
        return FitResult(False, {}, pred, np.inf, np.inf, np.inf, np.inf, "too few fit points")

    lower = np.asarray(bounds[0], dtype=float)
    upper = np.asarray(bounds[1], dtype=float)
    best = None

    def residual(theta: np.ndarray) -> np.ndarray:
        pred_fit = model(r[fit_mask], float(theta[0]), float(theta[1]))
        return (pred_fit - y[fit_mask]) / sigma[fit_mask]

    for start in starts:
        x0 = np.clip(np.asarray(start, dtype=float), lower + 1.0e-9, upper - 1.0e-9)
        try:
            result = least_squares(
                residual,
                x0=x0,
                bounds=(lower, upper),
                max_nfev=50_000,
            )
        except Exception:
            continue
        score = float(np.mean(result.fun**2))
        if best is None or score < best[0]:
            best = (score, result)

    if best is None:
        pred = np.zeros_like(y)
        return FitResult(False, {}, pred, np.inf, np.inf, np.inf, np.inf, "fit failed")

    _, result = best
    pred = model(r, float(result.x[0]), float(result.x[1]))
    err = pred - y
    scale = max(float(np.sqrt(np.mean(y**2))), EPS)
    denom = np.maximum(np.abs(y), 0.05 * scale)
    params = {param_names[0]: float(math.exp(result.x[0])), param_names[1]: float(math.exp(result.x[1]))}
    return FitResult(
        success=bool(result.success),
        params=params,
        prediction=np.asarray(pred, dtype=float),
        weighted_mse_fit=float(np.mean(((pred[fit_mask] - y[fit_mask]) / sigma[fit_mask]) ** 2)),
        weighted_mse_full=float(np.mean(((pred - y) / sigma) ** 2)),
        relative_rmse_full=float(np.sqrt(np.mean(err**2)) / scale),
        mean_abs_fractional_error_full=float(np.mean(np.abs(err) / denom)),
        message=f"{name}: {result.message}",
    )


def moving_average(y: np.ndarray, window: int = 5) -> np.ndarray:
    if len(y) < 3 or window <= 1:
        return y.copy()
    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if window < 3:
        return y.copy()
    pad = window // 2
    padded = np.pad(y, pad_width=pad, mode="reflect")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def effective_density_proxy(r: np.ndarray, delta_g: np.ndarray) -> np.ndarray:
    m_eff = r**2 * delta_g / G_KPC_KMS2_PER_MSUN
    smoothed = moving_average(m_eff, window=5)
    if len(r) < 3:
        return np.full_like(r, np.nan)
    dm_dr = np.gradient(smoothed, r, edge_order=2)
    return dm_dr / (4.0 * np.pi * np.maximum(r, EPS) ** 2)


def slope_loglog(r: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(r) < 3:
        return np.full_like(r, np.nan)
    clipped = np.maximum(y, EPS)
    return np.gradient(np.log(clipped), np.log(np.maximum(r, EPS)), edge_order=2)


def fit_galaxy(group: pd.DataFrame, meta: dict[str, object] | None) -> tuple[dict[str, object], pd.DataFrame]:
    galaxy = str(group["galaxy"].iloc[0])
    r, delta_g, sigma, vbar2 = acceleration_residual(group)
    order = np.argsort(r)
    r = r[order]
    delta_g = delta_g[order]
    sigma = sigma[order]
    vbar2 = vbar2[order]
    vobs = group["Vobs_kms"].to_numpy(dtype=float)[order]
    evobs = group["e_Vobs_kms"].to_numpy(dtype=float)[order]

    valid = np.isfinite(r) & np.isfinite(delta_g) & np.isfinite(sigma) & (r > 0.0) & (sigma > 0.0)
    r = r[valid]
    delta_g = delta_g[valid]
    sigma = sigma[valid]
    vbar2 = vbar2[valid]
    vobs = vobs[valid]
    evobs = evobs[valid]

    n = len(r)
    if n < 6 or float(np.sqrt(np.mean(delta_g**2))) <= 0.0:
        row = {
            "galaxy": galaxy,
            "n_points": n,
            "fit_status": "skipped",
            "skip_reason": "too few points or zero residual scale",
        }
        return row, pd.DataFrame()

    rdisk = float(meta.get("Rdisk_kpc", np.nan)) if meta else np.nan
    qflag = int(meta.get("Q", 0)) if meta else 0
    inc = float(meta.get("Inc_deg", np.nan)) if meta else np.nan
    vflat = float(meta.get("Vflat_kms", np.nan)) if meta else np.nan

    radial_break = np.nan
    if np.isfinite(rdisk) and rdisk > 0.0:
        radial_break = 2.2 * rdisk
    if not np.isfinite(radial_break) or radial_break <= 0.0:
        radial_break = float(np.median(r))
    radial_break = min(max(radial_break, float(np.percentile(r, 35))), float(np.percentile(r, 70)))

    outer_mask = r >= radial_break
    inner_mask = r <= radial_break
    if int(np.sum(outer_mask)) < 4:
        cutoff = np.partition(r, max(n - 4, 0))[max(n - 4, 0)]
        outer_mask = r >= cutoff
    if int(np.sum(inner_mask)) < 3:
        cutoff = np.partition(r, min(2, n - 1))[min(2, n - 1)]
        inner_mask = r <= cutoff
    full_mask = np.ones_like(r, dtype=bool)

    rmax = max(float(np.max(r)), 1.0)
    y_scale = max(float(np.sqrt(np.mean(delta_g**2))), 1.0)
    amp_guesses = [
        max(y_scale * rmax**2 * factor, 1.0)
        for factor in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    ]
    rs_guesses = [max(float(np.median(r)) * factor, 0.05) for factor in (0.2, 0.5, 1.0, 2.0, 5.0)]
    nfw_starts = [(math.log(a), math.log(rs)) for a in amp_guesses for rs in rs_guesses]
    nfw_bounds = ((math.log(1.0e-6), math.log(0.03)), (math.log(1.0e12), math.log(1.0e4)))

    speed_guesses = [
        max(float(np.nanmedian(np.abs(vobs))) * factor, 1.0)
        for factor in (0.5, 0.8, 1.0, 1.3, 1.8)
    ]
    rc_guesses = [max(float(np.median(r)) * factor, 0.03) for factor in (0.1, 0.3, 0.7, 1.5, 3.0)]
    iso_starts = [(math.log(v), math.log(rc)) for v in speed_guesses for rc in rc_guesses]
    iso_bounds = ((math.log(0.1), math.log(0.01)), (math.log(1.0e4), math.log(1.0e4)))

    log_amp_guesses = [max(y_scale * max(float(np.median(r)), 1.0) * factor, 1.0) for factor in (0.3, 1.0, 3.0, 10.0)]
    log_starts = [(math.log(a), math.log(r0)) for a in log_amp_guesses for r0 in rc_guesses]
    log_bounds = ((math.log(1.0e-6), math.log(0.001)), (math.log(1.0e12), math.log(1.0e4)))

    nfw_full = fit_positive_model(
        "NFW full",
        nfw_acceleration,
        ("nfw_amp", "nfw_r_s_kpc"),
        nfw_starts,
        nfw_bounds,
        r,
        delta_g,
        sigma,
        full_mask,
    )
    nfw_outer = fit_positive_model(
        "NFW outer",
        nfw_acceleration,
        ("nfw_outer_amp", "nfw_outer_r_s_kpc"),
        nfw_starts,
        nfw_bounds,
        r,
        delta_g,
        sigma,
        outer_mask,
    )
    iso_full = fit_positive_model(
        "cored/isothermal full",
        cored_isothermal_acceleration,
        ("iso_v0_kms", "iso_r_c_kpc"),
        iso_starts,
        iso_bounds,
        r,
        delta_g,
        sigma,
        full_mask,
    )
    log_full = fit_positive_model(
        "log-tail full",
        log_tail_acceleration,
        ("log_amp", "log_r0_kpc"),
        log_starts,
        log_bounds,
        r,
        delta_g,
        sigma,
        full_mask,
    )

    gap_outer = delta_g - nfw_outer.prediction
    gap_full = delta_g - nfw_full.prediction
    inner_gap = gap_outer[inner_mask]
    inner_scale = max(float(np.sqrt(np.mean(delta_g[inner_mask] ** 2))), 1.0)
    inner_gap_mean_norm = float(np.mean(inner_gap) / inner_scale)
    inner_gap_negative_fraction = float(np.mean(inner_gap < 0.0))
    central_overshoot = bool(inner_gap_negative_fraction >= 0.67 and inner_gap_mean_norm <= -0.25)

    outer_slope_nfw_full = slope_loglog(r, nfw_full.prediction)[outer_mask]
    outer_slope_nfw_outer = slope_loglog(r, nfw_outer.prediction)[outer_mask]
    outer_slope_target = slope_loglog(r, np.maximum(delta_g, EPS))[outer_mask]
    nfw_full_shallow_outer = bool(np.nanmean(outer_slope_nfw_full) > -0.7)

    gap_density_outer = effective_density_proxy(r, gap_outer)
    inner_gap_density = gap_density_outer[inner_mask]

    quality_primary = bool(qflag in (1, 2) and n >= 8 and (not np.isfinite(inc) or inc >= 30.0))
    row: dict[str, object] = {
        "galaxy": galaxy,
        "fit_status": "ok",
        "n_points": n,
        "Q": qflag,
        "Inc_deg": inc,
        "Rdisk_kpc": rdisk,
        "Vflat_kms": vflat,
        "Rmax_kpc": float(np.max(r)),
        "radial_break_kpc": float(radial_break),
        "n_inner": int(np.sum(inner_mask)),
        "n_outer": int(np.sum(outer_mask)),
        "quality_primary": quality_primary,
        "upsilon_disk": UPSILON_DISK,
        "upsilon_bulge": UPSILON_BULGE,
        "nfw_full_success": nfw_full.success,
        "nfw_full_params": json.dumps(nfw_full.params),
        "nfw_full_weighted_mse": nfw_full.weighted_mse_full,
        "nfw_full_relative_rmse": nfw_full.relative_rmse_full,
        "nfw_full_mean_abs_fractional_error": nfw_full.mean_abs_fractional_error_full,
        "nfw_outer_success": nfw_outer.success,
        "nfw_outer_params": json.dumps(nfw_outer.params),
        "nfw_outer_fit_weighted_mse": nfw_outer.weighted_mse_fit,
        "nfw_outer_full_weighted_mse": nfw_outer.weighted_mse_full,
        "nfw_outer_relative_rmse": nfw_outer.relative_rmse_full,
        "nfw_outer_mean_abs_fractional_error": nfw_outer.mean_abs_fractional_error_full,
        "iso_full_success": iso_full.success,
        "iso_full_params": json.dumps(iso_full.params),
        "iso_full_weighted_mse": iso_full.weighted_mse_full,
        "iso_full_relative_rmse": iso_full.relative_rmse_full,
        "iso_full_mean_abs_fractional_error": iso_full.mean_abs_fractional_error_full,
        "log_full_success": log_full.success,
        "log_full_params": json.dumps(log_full.params),
        "log_full_weighted_mse": log_full.weighted_mse_full,
        "log_full_relative_rmse": log_full.relative_rmse_full,
        "log_full_mean_abs_fractional_error": log_full.mean_abs_fractional_error_full,
        "iso_beats_nfw_full": bool(iso_full.weighted_mse_full < nfw_full.weighted_mse_full),
        "log_beats_nfw_full": bool(log_full.weighted_mse_full < nfw_full.weighted_mse_full),
        "inner_gap_mean_norm_after_outer_nfw": inner_gap_mean_norm,
        "inner_gap_negative_fraction_after_outer_nfw": inner_gap_negative_fraction,
        "central_overshoot_after_outer_nfw": central_overshoot,
        "inner_gap_density_mean": float(np.nanmean(inner_gap_density)),
        "inner_gap_density_negative_fraction": float(np.mean(inner_gap_density < 0.0)),
        "nfw_full_outer_slope_mean": float(np.nanmean(outer_slope_nfw_full)),
        "nfw_outer_outer_slope_mean": float(np.nanmean(outer_slope_nfw_outer)),
        "target_outer_slope_mean": float(np.nanmean(outer_slope_target)),
        "nfw_full_shallow_outer": nfw_full_shallow_outer,
    }

    profile = pd.DataFrame(
        {
            "galaxy": galaxy,
            "R_kpc": r,
            "Vobs_kms": vobs,
            "e_Vobs_kms": evobs,
            "Vbar2_kms2": vbar2,
            "delta_g_obs": delta_g,
            "sigma_delta_g": sigma,
            "nfw_full_delta_g": nfw_full.prediction,
            "nfw_outer_delta_g": nfw_outer.prediction,
            "iso_full_delta_g": iso_full.prediction,
            "log_full_delta_g": log_full.prediction,
            "gap_nfw_full": gap_full,
            "gap_nfw_outer": gap_outer,
            "gap_density_proxy_nfw_outer": gap_density_outer,
            "is_inner": inner_mask,
            "is_outer": outer_mask,
        }
    )
    return row, profile


def write_population_summary(summary: pd.DataFrame) -> pd.DataFrame:
    ok = summary[summary["fit_status"] == "ok"].copy()
    primary = ok[ok["quality_primary"]].copy()
    primary_flat = primary[primary["Vflat_kms"] > 0.0].copy()

    def safe_fraction(series: pd.Series) -> float:
        return float(np.mean(series.astype(bool))) if len(series) else float("nan")

    def metric_rows(label: str, frame: pd.DataFrame) -> dict[str, object]:
        return {
            "sample": label,
            "n_galaxies": int(len(frame)),
            "median_nfw_full_relative_rmse": float(frame["nfw_full_relative_rmse"].median()),
            "median_iso_full_relative_rmse": float(frame["iso_full_relative_rmse"].median()),
            "median_nfw_full_weighted_mse": float(frame["nfw_full_weighted_mse"].median()),
            "median_iso_full_weighted_mse": float(frame["iso_full_weighted_mse"].median()),
            "fraction_iso_beats_nfw_full": safe_fraction(frame["iso_beats_nfw_full"]),
            "fraction_log_beats_nfw_full": safe_fraction(frame["log_beats_nfw_full"]),
            "fraction_central_overshoot_after_outer_nfw": safe_fraction(
                frame["central_overshoot_after_outer_nfw"]
            ),
            "median_inner_gap_mean_norm_after_outer_nfw": float(
                frame["inner_gap_mean_norm_after_outer_nfw"].median()
            ),
            "fraction_nfw_full_shallow_outer": safe_fraction(frame["nfw_full_shallow_outer"]),
        }

    rows = [
        metric_rows("all_ok", ok),
        metric_rows("primary_quality", primary),
        metric_rows("primary_flat", primary_flat),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "population_summary.csv", index=False)
    return out


def save_example_profiles(summary: pd.DataFrame, profiles: pd.DataFrame) -> None:
    ok = summary[
        (summary["fit_status"] == "ok") & (summary["quality_primary"]) & (summary["Vflat_kms"] > 0.0)
    ].copy()
    if ok.empty:
        ok = summary[(summary["fit_status"] == "ok") & (summary["quality_primary"])].copy()
    if ok.empty:
        ok = summary[summary["fit_status"] == "ok"].copy()
    ok = ok.sort_values("inner_gap_mean_norm_after_outer_nfw").head(6)
    galaxies = list(ok["galaxy"])
    if not galaxies:
        return

    ncols = 2
    nrows = int(math.ceil(len(galaxies) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.3 * nrows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, galaxy in zip(axes_arr, galaxies):
        data = profiles[profiles["galaxy"] == galaxy].sort_values("R_kpc")
        row = summary[summary["galaxy"] == galaxy].iloc[0]
        ax.axhline(0.0, color="#666666", lw=0.8)
        ax.errorbar(
            data["R_kpc"],
            data["delta_g_obs"],
            yerr=data["sigma_delta_g"],
            fmt="o",
            ms=3.2,
            color="#111111",
            ecolor="#BBBBBB",
            label="SPARC residual",
        )
        ax.plot(data["R_kpc"], data["nfw_full_delta_g"], color="#D95D39", lw=1.7, label="NFW full")
        ax.plot(data["R_kpc"], data["nfw_outer_delta_g"], color="#1B998B", lw=1.7, label="NFW outer")
        ax.plot(data["R_kpc"], data["iso_full_delta_g"], color="#5B5F97", lw=1.5, label="cored/isothermal")
        ax.fill_between(
            data["R_kpc"],
            data["gap_nfw_outer"],
            0,
            color="#1B998B",
            alpha=0.12,
            label="target - NFW outer",
        )
        ax.axvline(float(row["radial_break_kpc"]), color="#666666", lw=0.8, ls="--")
        ax.set_title(
            f"{galaxy}: inner gap norm {float(row['inner_gap_mean_norm_after_outer_nfw']):.2f}"
        )
        ax.set_xlabel("R [kpc]")
        ax.set_ylabel("Delta g [(km/s)^2/kpc]")
        ax.legend(frameon=False, fontsize=7)

    for ax in axes_arr[len(galaxies) :]:
        ax.axis("off")

    fig.savefig(FIGURES / "example_residual_profiles.png", dpi=180)
    plt.close(fig)


def save_population_figures(summary: pd.DataFrame) -> None:
    ok = summary[summary["fit_status"] == "ok"].copy()
    primary = ok[ok["quality_primary"]].copy()
    plot_frame = primary if len(primary) >= 10 else ok

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.scatter(
        plot_frame["nfw_full_relative_rmse"],
        plot_frame["iso_full_relative_rmse"],
        c=plot_frame["central_overshoot_after_outer_nfw"].astype(int),
        cmap="viridis",
        s=34,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
    )
    lim = max(
        float(plot_frame["nfw_full_relative_rmse"].quantile(0.95)),
        float(plot_frame["iso_full_relative_rmse"].quantile(0.95)),
        0.2,
    )
    ax.plot([0, lim], [0, lim], color="#555555", ls="--", lw=1.0)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("NFW relative RMSE")
    ax.set_ylabel("Cored/isothermal relative RMSE")
    ax.set_title("Full-range residual fit comparison")

    ax = axes[1]
    ratio = plot_frame["iso_full_weighted_mse"] / np.maximum(plot_frame["nfw_full_weighted_mse"], EPS)
    ax.hist(np.clip(np.log10(ratio), -2.5, 2.5), bins=28, color="#5B5F97", alpha=0.85)
    ax.axvline(0.0, color="#111111", ls="--", lw=1.0)
    ax.set_xlabel("log10(cored/isothermal weighted MSE / NFW weighted MSE)")
    ax.set_ylabel("galaxies")
    ax.set_title("Negative values favor cored/isothermal")
    fig.savefig(FIGURES / "population_fit_comparison.png", dpi=180)
    plt.close(fig)

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

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.25), constrained_layout=True)
    ax = axes[0]
    ax.hist(
        plot_frame["inner_gap_mean_norm_after_outer_nfw"],
        bins=30,
        color="#1B998B",
        alpha=0.85,
    )
    ax.axvline(0.0, color="#111111", ls="--", lw=1.0)
    ax.axvline(-0.25, color="#D95D39", ls="--", lw=1.0)
    ax.set_xlabel("mean inner gap / inner RMS")
    ax.set_ylabel("galaxies")
    ax.set_title("After outer NFW fit")

    ax = axes[1]
    ax.scatter(
        plot_frame["Vflat_kms"],
        plot_frame["inner_gap_mean_norm_after_outer_nfw"],
        c=plot_frame["Q"],
        cmap="plasma_r",
        s=36,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axhline(0.0, color="#111111", ls="--", lw=1.0)
    ax.axhline(-0.25, color="#D95D39", ls="--", lw=1.0)
    ax.set_xlabel("SPARC Vflat [km/s]")
    ax.set_ylabel("mean inner gap / inner RMS")
    ax.set_title("By galaxy scale")
    fig.savefig(FIGURES / "central_overshoot_distribution.png", dpi=220)
    plt.close(fig)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    ensure_inputs()

    mass_models = read_mass_models(MASS_MODELS_PATH)
    sample = read_sample_table(SAMPLE_TABLE_PATH)
    metadata = {row["galaxy"]: row for row in sample.to_dict(orient="records")}

    summary_rows: list[dict[str, object]] = []
    profile_frames: list[pd.DataFrame] = []

    for _, group in mass_models.groupby("galaxy", sort=True):
        galaxy = str(group["galaxy"].iloc[0])
        row, profile = fit_galaxy(group, metadata.get(galaxy))
        summary_rows.append(row)
        if not profile.empty:
            profile_frames.append(profile)

    summary = pd.DataFrame(summary_rows)
    profiles = pd.concat(profile_frames, ignore_index=True) if profile_frames else pd.DataFrame()

    summary.to_csv(RESULTS / "galaxy_fit_summary.csv", index=False)
    profiles.to_csv(RESULTS / "residual_profiles.csv", index=False)
    population = write_population_summary(summary)
    save_example_profiles(summary, profiles)
    save_population_figures(summary)

    source_info = {
        "mass_models_url": MASS_MODELS_URL,
        "sample_table_url": SAMPLE_TABLE_URL,
        "upsilon_disk": UPSILON_DISK,
        "upsilon_bulge": UPSILON_BULGE,
        "notes": [
            "Vgas is included with sign as recommended for SPARC velocity contributions.",
            "Vdisk and Vbul are scaled from M/L=1 using fixed fiducial mass-to-light ratios.",
            "Effective density proxies are spherical diagnostics, not disk inversions.",
        ],
    }
    (RESULTS / "source_and_assumptions.json").write_text(
        json.dumps(source_info, indent=2),
        encoding="utf-8",
    )

    print("SPARC NFW residual experiment completed.")
    print(f"Parsed {len(mass_models)} rotation-curve points for {mass_models['galaxy'].nunique()} galaxies.")
    print(f"Fit status counts: {summary['fit_status'].value_counts().to_dict()}")
    print("Population summary:")
    print(population.to_string(index=False))
    print("Created outputs:")
    for rel in [
        "results/galaxy_fit_summary.csv",
        "results/residual_profiles.csv",
        "results/population_summary.csv",
        "results/source_and_assumptions.json",
        "figures/example_residual_profiles.png",
        "figures/population_fit_comparison.png",
        "figures/central_overshoot_distribution.png",
    ]:
        print(f"  - {rel}")


if __name__ == "__main__":
    main()
