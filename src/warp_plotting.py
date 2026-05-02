"""Plotting utilities for the geometry-first warp grid search."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import warp_search


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8.0, 5.0),
            "figure.dpi": 120,
            "savefig.dpi": 170,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 2.0,
        }
    )


def _path(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def save_rotation_fit(path: str | Path, context, best: dict[str, np.ndarray], top_family_curves: list[tuple[str, np.ndarray]]) -> Path:
    path = _path(path)
    fig, ax = plt.subplots()
    r = context.r
    ax.plot(r, context.v_bar, label="Baryon-only curve", color="#276FBF")
    ax.plot(r, context.v_obs, label="Synthetic target", color="#111111", lw=2.4)
    ax.plot(r, best["v_trial"], label="Best warp", color="#D95D39", lw=2.3)
    for label, curve in top_family_curves:
        ax.plot(r, curve, lw=1.2, alpha=0.7, label=label)
    ax.set_title("Rotation Curve from Direct Metric Warp")
    ax.set_xlabel("Radius r [toy kpc]")
    ax.set_ylabel("Circular speed v [km/s]")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_warp_potentials(path: str | Path, context, best: dict[str, np.ndarray]) -> Path:
    path = _path(path)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    r = context.r
    axes[0].plot(r, context.delta_phi_target, label=r"Target $\delta\Phi$", color="#111111")
    axes[0].plot(r, best["delta_phi"], label=r"Best $\delta\Phi$", color="#D95D39")
    axes[0].plot(r, best["delta_psi"], label=r"Best $\delta\Psi$", color="#4C78A8")
    axes[0].set_title("Best-Fit Weak-Field Warp Potentials")
    axes[0].set_ylabel(r"Potential perturbation [(km/s)$^2$]")
    axes[0].legend(loc="best")

    axes[1].plot(r, best["eta"], label=r"$\eta(r)=\delta\Psi/\delta\Phi$", color="#1B998B")
    axes[1].axhline(1.0, color="#666666", lw=1.0)
    axes[1].set_xlabel("Radius r [toy kpc]")
    axes[1].set_ylabel("Slip parameter")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_residual_acceleration(path: str | Path, context, best: dict[str, np.ndarray]) -> Path:
    path = _path(path)
    fig, ax = plt.subplots()
    r = context.r
    ax.plot(r, context.delta_g_target, label=r"Target $\Delta g=d\delta\Phi/dr$", color="#111111")
    ax.plot(r, best["delta_g"], label="Best warp residual", color="#D95D39")
    ax.set_title("Acceleration Residual from Metric Perturbation")
    ax.set_xlabel("Radius r [toy kpc]")
    ax.set_ylabel(r"$\Delta g$ [(km/s)$^2$/kpc]")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_effective_profiles(path: str | Path, context, best: dict[str, np.ndarray]) -> Path:
    path = _path(path)
    target_m, target_rho = warp_search.effective_profiles(context.r, context.delta_g_target)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    r = context.r
    axes[0].plot(r, target_m, label="Target proxy", color="#111111")
    axes[0].plot(r, best["m_eff"], label="Best warp proxy", color="#D95D39")
    axes[0].set_title("Effective Mass and Density Implied by Warp")
    axes[0].set_ylabel(r"$M_{\rm eff}(<r)$ [$10^{10}M_\odot$]")
    axes[0].legend(loc="best")

    axes[1].plot(r, target_rho, label="Target proxy", color="#111111")
    axes[1].plot(r, best["rho_eff"], label="Best warp proxy", color="#D95D39")
    axes[1].axhline(0.0, color="#666666", lw=1.0)
    axes[1].set_xlabel("Radius r [toy kpc]")
    axes[1].set_ylabel(r"$\rho_{\rm eff}$ [$10^{10}M_\odot$/kpc$^3$]")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_lensing_proxy(path: str | Path, context, best: dict[str, np.ndarray], slip_examples: list[tuple[str, np.ndarray]]) -> Path:
    path = _path(path)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    r = context.r
    axes[0].plot(r, warp_search.normalized_proxy(context.lensing_target), label="Target residual proxy", color="#111111")
    axes[0].plot(r, warp_search.normalized_proxy(best["lensing_proxy"]), label="Best warp proxy", color="#D95D39")
    for label, proxy in slip_examples:
        axes[0].plot(r, warp_search.normalized_proxy(proxy), lw=1.2, alpha=0.8, label=label)
    axes[0].set_title("Conceptual Lensing/Slip Proxy")
    axes[0].set_ylabel("Normalized residual proxy")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(r, best["delta_phi"], label=r"$\delta\Phi$", color="#D95D39")
    axes[1].plot(r, best["delta_psi"], label=r"$\delta\Psi$", color="#4C78A8")
    axes[1].set_xlabel("Radius r [toy kpc]")
    axes[1].set_ylabel(r"Potential perturbation [(km/s)$^2$]")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_log_heatmap(path: str | Path, frame: pd.DataFrame) -> Path:
    path = _path(path)
    subset = frame[(frame["family"] == "log_warp") & (frame["eta_mode"] == "no_slip")].copy()
    subset["A"] = subset["params"].apply(lambda item: item["A"] if isinstance(item, dict) else np.nan)
    subset["r0"] = subset["params"].apply(lambda item: item["r0"] if isinstance(item, dict) else np.nan)
    pivot = subset.pivot_table(index="r0", columns="A", values="total_score", aggfunc="min")
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    image = ax.imshow(
        np.log10(pivot.values),
        origin="lower",
        aspect="auto",
        extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
        cmap="viridis_r",
    )
    ax.set_title("Log-Warp Parameter Region")
    ax.set_xlabel(r"Amplitude $A$ [(km/s)$^2$]")
    ax.set_ylabel(r"Scale $r_0$ [toy kpc]")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(r"$\log_{10}$ total score")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_family_scores(path: str | Path, family_summary: pd.DataFrame) -> Path:
    path = _path(path)
    ordered = family_summary.sort_values("best_total_score", ascending=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.barh(np.arange(len(ordered)), ordered["best_total_score"], color="#4C78A8")
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels(ordered["family"])
    ax.invert_yaxis()
    ax.set_title("Best Score by Warp Family")
    ax.set_xlabel("Best total score, lower is better")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_parameter_scatter(path: str | Path, frame: pd.DataFrame) -> Path:
    path = _path(path)
    subset = frame[frame["eta_mode"] == "no_slip"].copy()
    subset = subset.sort_values("total_score").head(4000)
    families = list(dict.fromkeys(subset["family"]))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for idx, family in enumerate(families):
        group = subset[subset["family"] == family]
        ax.scatter(
            group["flatness_metric"],
            group["rotation_mse"],
            s=12,
            alpha=0.55,
            label=family,
            color=cmap(idx % 10),
        )
    ax.set_yscale("log")
    ax.set_title("Working Regions in Warp Search")
    ax.set_xlabel("Outer rotation flatness metric")
    ax.set_ylabel("Rotation weighted MSE")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_basis_components(path: str | Path, context, basis_weights: list[dict[str, float]]) -> Path:
    path = _path(path)
    r = context.r
    width = basis_weights[0]["width"] if basis_weights else 5.0
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    total = np.zeros_like(r)
    labels = []
    weights = []
    for row in basis_weights:
        component = row["weight"] * np.exp(-0.5 * ((r - row["center"]) / width) ** 2)
        total += component
        axes[0].plot(r, component, lw=1.2, alpha=0.85, label=f"c={row['center']:.1f}")
        labels.append(f"{row['center']:.1f}")
        weights.append(row["weight"])
    axes[0].plot(r, context.delta_g_target, color="#111111", lw=2.2, label="Target")
    axes[0].plot(r, total, color="#D95D39", lw=2.0, label="Basis sum")
    axes[0].set_title("Radial Basis Warp Acceleration")
    axes[0].set_xlabel("Radius r [toy kpc]")
    axes[0].set_ylabel(r"$d\delta\Phi/dr$")
    axes[0].legend(loc="best", fontsize=7)

    axes[1].barh(np.arange(len(weights)), weights, color="#4C78A8")
    axes[1].set_yticks(np.arange(len(weights)))
    axes[1].set_yticklabels(labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Nonnegative basis weight")
    axes[1].set_title("Selected Basis Centers")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path

