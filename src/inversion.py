"""Weak-field residual inversion utilities."""

from __future__ import annotations

import numpy as np

from .profiles import EPS, G_TOY


def estimate_acceleration_from_velocity(r: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Estimate g=v^2/r from a circular-speed curve."""
    r = np.asarray(r, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    return velocity**2 / np.maximum(r, EPS)


def recover_residual_acceleration(
    r: np.ndarray,
    v_obs: np.ndarray,
    g_bar: np.ndarray,
) -> np.ndarray:
    """Recover Delta g = g_obs - g_bar from observed velocity and baryons."""
    return estimate_acceleration_from_velocity(r, v_obs) - np.asarray(g_bar, dtype=float)


def effective_mass(r: np.ndarray, delta_g: np.ndarray, g_const: float = G_TOY) -> np.ndarray:
    """Toy spherical effective missing mass M_eff(<r)=r^2 Delta g/G."""
    r = np.asarray(r, dtype=float)
    delta_g = np.asarray(delta_g, dtype=float)
    return r**2 * delta_g / g_const


def moving_average(y: np.ndarray, window: int = 7) -> np.ndarray:
    """Reflect-padded moving average for derivative stabilization."""
    y = np.asarray(y, dtype=float)
    if window <= 1:
        return y.copy()
    window = int(window)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(y, pad_width=pad, mode="reflect")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def effective_density(
    r: np.ndarray,
    m_eff: np.ndarray,
    smooth_window: int = 7,
) -> np.ndarray:
    """Toy spherical density proxy rho_eff=(1/4 pi r^2) dM_eff/dr."""
    r = np.asarray(r, dtype=float)
    m_eff = np.asarray(m_eff, dtype=float)
    mass_for_derivative = moving_average(m_eff, smooth_window)
    dmass_dr = np.gradient(mass_for_derivative, r, edge_order=2)
    return dmass_dr / (4.0 * np.pi * np.maximum(r, EPS) ** 2)


def invert_effective_profiles(
    r: np.ndarray,
    v_obs: np.ndarray,
    g_bar: np.ndarray,
    smooth_window: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Delta g, M_eff, and rho_eff from noisy rotation data."""
    delta_g = recover_residual_acceleration(r, v_obs, g_bar)
    m_eff = effective_mass(r, delta_g)
    rho_eff = effective_density(r, m_eff, smooth_window=smooth_window)
    return delta_g, m_eff, rho_eff


def physical_checks(r: np.ndarray, delta_g: np.ndarray) -> dict[str, float | bool]:
    """Simple toy validity checks for an inferred residual profile."""
    r = np.asarray(r, dtype=float)
    delta_g = np.asarray(delta_g, dtype=float)
    m_eff = effective_mass(r, delta_g)
    rho_eff = effective_density(r, m_eff, smooth_window=5)
    dmass = np.diff(m_eff)
    second = np.diff(delta_g, n=2)
    sign_changes = np.sum(np.diff(np.signbit(second)) != 0) if len(second) > 1 else 0
    oscillation_score = float(sign_changes / max(len(second), 1))

    return {
        "delta_g_positive_fraction": float(np.mean(delta_g > -1.0e-9)),
        "mass_monotonic_fraction": float(np.mean(dmass >= -1.0e-6)) if len(dmass) else 1.0,
        "rho_nonnegative_fraction": float(np.mean(rho_eff >= -1.0e-9)),
        "oscillation_score": oscillation_score,
        "mostly_positive_delta_g": bool(np.mean(delta_g > -1.0e-9) > 0.95),
        "mostly_monotonic_mass": bool((np.mean(dmass >= -1.0e-6) if len(dmass) else 1.0) > 0.90),
        "mostly_nonnegative_density": bool(np.mean(rho_eff >= -1.0e-9) > 0.85),
        "pathological_oscillations": bool(oscillation_score > 0.35),
    }


def lensing_proxy(
    r: np.ndarray,
    delta_g: np.ndarray,
    eta: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Conceptual lensing/slip proxy based on an integral over effective mass.

    Dynamics in the weak-field limit mostly probes Phi. Lensing probes Phi+Psi.
    We therefore scale the effective mass by (1+eta)/2 before integrating. This
    is not a physical lensing calculation; it is only a second projection that
    can respond differently to candidate latent generators.
    """
    r = np.asarray(r, dtype=float)
    m_eff = effective_mass(r, delta_g)
    eta_arr = np.asarray(eta, dtype=float) + np.zeros_like(r)
    lensing_weight = 0.5 * (1.0 + eta_arr)
    integrand = lensing_weight * m_eff / np.maximum(r, EPS) ** 2
    proxy = np.zeros_like(r)
    if len(r) > 1:
        proxy[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r))
    max_abs = np.max(np.abs(proxy))
    if max_abs > 0:
        proxy = proxy / max_abs
    return proxy

