"""Matplotlib figures for the geometric residual prototype."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _prepare_path(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


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


def save_rotation_curve(
    path: str | Path,
    r: np.ndarray,
    v_bar: np.ndarray,
    v_obs_true: np.ndarray,
    v_obs_noisy: np.ndarray,
    v_hidden: np.ndarray,
    sigma_v: np.ndarray,
) -> Path:
    path = _prepare_path(path)
    fig, ax = plt.subplots()
    ax.plot(r, v_bar, label="Baryons only", color="#276FBF")
    ax.plot(r, v_obs_true, label="Observed truth", color="#1B998B")
    ax.plot(r, v_hidden, label=r"Hidden contribution $\sqrt{r\Delta g}$", color="#D95D39")
    ax.errorbar(
        r[::5],
        v_obs_noisy[::5],
        yerr=sigma_v[::5],
        fmt="o",
        ms=3.5,
        color="#222222",
        alpha=0.75,
        label="Noisy samples",
    )
    ax.set_title("Synthetic Rotation Curve")
    ax.set_xlabel("Radius r [toy kpc]")
    ax.set_ylabel("Circular speed v [km/s]")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_acceleration_residual(
    path: str | Path,
    r: np.ndarray,
    delta_true: np.ndarray,
    delta_est: np.ndarray,
    sigma_delta: np.ndarray,
) -> Path:
    path = _prepare_path(path)
    fig, ax = plt.subplots()
    ax.plot(r, delta_true, label="True geometric residual", color="#1B998B")
    ax.errorbar(
        r[::5],
        delta_est[::5],
        yerr=sigma_delta[::5],
        fmt="o",
        ms=3.5,
        color="#222222",
        alpha=0.7,
        label="Recovered from noisy v(r)",
    )
    ax.set_title("Recovered Acceleration Residual")
    ax.set_xlabel("Radius r [toy kpc]")
    ax.set_ylabel(r"$\Delta g$ [(km/s)$^2$/kpc]")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_effective_mass_density(
    path: str | Path,
    r: np.ndarray,
    m_true: np.ndarray,
    m_est: np.ndarray,
    rho_true: np.ndarray,
    rho_est: np.ndarray,
) -> Path:
    path = _prepare_path(path)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    axes[0].plot(r, m_true, label="True proxy", color="#1B998B")
    axes[0].plot(r, m_est, label="Recovered proxy", color="#222222", alpha=0.78)
    axes[0].set_title("Toy Effective Missing Mass and Density")
    axes[0].set_ylabel(r"$M_{\rm eff}(<r)$ [$10^{10}M_\odot$]")
    axes[0].legend(loc="best")

    axes[1].plot(r, rho_true, label="True proxy", color="#1B998B")
    axes[1].plot(r, rho_est, label="Recovered proxy", color="#222222", alpha=0.78)
    axes[1].axhline(0.0, color="#666666", lw=1.0)
    axes[1].set_xlabel("Radius r [toy kpc]")
    axes[1].set_ylabel(r"$\rho_{\rm eff}$ [$10^{10}M_\odot$/kpc$^3$]")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_model_comparison(
    path: str | Path,
    fits: list,
    r: np.ndarray,
    delta_est: np.ndarray,
) -> Path:
    path = _prepare_path(path)
    names = [fit.name for fit in fits]
    bic = np.array([fit.bic for fit in fits], dtype=float)
    order = np.argsort(bic)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    short_names = [names[i].replace(" halo", "").replace(" relation", "") for i in order]
    axes[0].barh(np.arange(len(order)), bic[order], color="#4C78A8")
    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels(short_names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("BIC-like score, lower is better")
    axes[0].set_title("Candidate Generator Comparison")

    axes[1].plot(r, delta_est, color="#222222", alpha=0.5, label="Recovered residual")
    for fit in fits[:4]:
        axes[1].plot(r, fit.y_pred, label=fit.name)
    axes[1].set_title("Residual Fits")
    axes[1].set_xlabel("Radius r [toy kpc]")
    axes[1].set_ylabel(r"$\Delta g$ [(km/s)$^2$/kpc]")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_mixed_source_fit(
    path: str | Path,
    r: np.ndarray,
    delta_true: np.ndarray,
    delta_est: np.ndarray,
    best_single,
    sparse_fit,
) -> Path:
    path = _prepare_path(path)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    axes[0].plot(r, delta_true, label="True mixed source", color="#1B998B")
    axes[0].scatter(r[::4], delta_est[::4], label="Recovered samples", color="#222222", s=14, alpha=0.65)
    axes[0].plot(r, best_single.y_pred, label=f"Best single: {best_single.name}", color="#D95D39")
    axes[0].plot(r, sparse_fit.y_pred, label="Sparse nonnegative mixture", color="#4C78A8")
    axes[0].set_title("Mixed Hidden Source Fit")
    axes[0].set_ylabel(r"$\Delta g$ [(km/s)$^2$/kpc]")
    axes[0].legend(loc="best")

    axes[1].plot(r, delta_est - best_single.y_pred, label="Recovered - best single", color="#D95D39")
    axes[1].plot(r, delta_est - sparse_fit.y_pred, label="Recovered - sparse mixture", color="#4C78A8")
    axes[1].axhline(0.0, color="#666666", lw=1.0)
    axes[1].set_xlabel("Radius r [toy kpc]")
    axes[1].set_ylabel("Fit residual")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_lensing_proxy_degeneracy(
    path: str | Path,
    r: np.ndarray,
    particle_delta: np.ndarray,
    modified_delta: np.ndarray,
    particle_proxy: np.ndarray,
    modified_proxy: np.ndarray,
    eta_modified: np.ndarray,
) -> Path:
    path = _prepare_path(path)
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.5), sharex=True)
    axes[0].plot(r, particle_delta, label="Particle-like residual", color="#1B998B")
    axes[0].plot(r, modified_delta, "--", label="Modified-gravity slip twin", color="#D95D39")
    axes[0].set_title("Toy Multi-Probe Degeneracy Break")
    axes[0].set_ylabel(r"$\Delta g$ [(km/s)$^2$/kpc]")
    axes[0].legend(loc="best")

    axes[1].plot(r, np.ones_like(r), label=r"Particle-like $\eta=1$", color="#1B998B")
    axes[1].plot(r, eta_modified, label=r"Toy slip $\eta(r)\ne 1$", color="#D95D39")
    axes[1].set_ylabel(r"Slip $\eta=\Psi/\Phi$")
    axes[1].legend(loc="best")

    axes[2].plot(r, particle_proxy, label="Particle-like lensing proxy", color="#1B998B")
    axes[2].plot(r, modified_proxy, label="Slip-model lensing proxy", color="#D95D39")
    axes[2].set_xlabel("Radius r [toy kpc]")
    axes[2].set_ylabel("Normalized lensing proxy")
    axes[2].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_sparse_basis_search(
    path: str | Path,
    r: np.ndarray,
    delta_est: np.ndarray,
    sparse_info: dict[str, object],
) -> Path:
    path = _prepare_path(path)
    labels = list(sparse_info["selected_labels"])
    contributions = np.asarray(sparse_info["contributions"], dtype=float)
    y_pred = np.asarray(sparse_info["y_pred"], dtype=float)
    weights_table = list(sparse_info["weights_table"])
    weights = np.array([row["normalized_weight"] for row in weights_table], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    axes[0].plot(r, delta_est, label="Recovered residual", color="#222222", alpha=0.55)
    axes[0].plot(r, y_pred, label="Sparse basis fit", color="#4C78A8", lw=2.4)
    for idx, label in enumerate(labels):
        axes[0].plot(r, contributions[:, idx], lw=1.3, alpha=0.85, label=label)
    axes[0].set_title("Sparse Basis Decomposition")
    axes[0].set_xlabel("Radius r [toy kpc]")
    axes[0].set_ylabel(r"$\Delta g$ contribution")
    axes[0].legend(loc="best", fontsize=7)

    axes[1].barh(np.arange(len(labels)), weights, color="#4C78A8")
    axes[1].set_yticks(np.arange(len(labels)))
    axes[1].set_yticklabels(labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Nonnegative normalized weight")
    axes[1].set_title("Selected Basis Terms")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path

