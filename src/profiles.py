"""Toy baryonic and hidden-source profiles for geometric residual experiments.

The units are deliberately simple astrophysical toy units:
- radius r is kpc-like,
- mass is in 1e10 solar masses,
- velocity is km/s,
- acceleration is (km/s)^2 / kpc.

The numerical constants make the rotation curves look galaxy-like, but these
profiles are not intended as precision disk-galaxy models.
"""

from __future__ import annotations

import numpy as np

G_TOY = 4.30091e4  # kpc (km/s)^2 per 1e10 Msun
EPS = 1.0e-12


def radial_grid(r_min: float = 0.25, r_max: float = 35.0, n: int = 180) -> np.ndarray:
    """Return a radial grid that avoids r=0 singularities."""
    return np.linspace(r_min, r_max, n)


def baryonic_mass(r: np.ndarray, m_b: float = 5.0, r_d: float = 3.0) -> np.ndarray:
    """Exponential-disk-inspired cumulative baryonic mass profile."""
    r = np.asarray(r, dtype=float)
    x = np.maximum(r / r_d, 0.0)
    return m_b * (1.0 - np.exp(-x) * (1.0 + x))


def baryonic_density_proxy(r: np.ndarray, m_b: float = 5.0, r_d: float = 3.0) -> np.ndarray:
    """Spherical density proxy implied by the toy cumulative baryonic mass."""
    r = np.asarray(r, dtype=float)
    mass = baryonic_mass(r, m_b=m_b, r_d=r_d)
    dmass_dr = np.gradient(mass, r, edge_order=2)
    return dmass_dr / (4.0 * np.pi * np.maximum(r, EPS) ** 2)


def acceleration_from_mass(r: np.ndarray, mass: np.ndarray, g_const: float = G_TOY) -> np.ndarray:
    """Spherical acceleration g=G M(<r)/r^2."""
    r = np.asarray(r, dtype=float)
    mass = np.asarray(mass, dtype=float)
    return g_const * mass / np.maximum(r, EPS) ** 2


def velocity_from_acceleration(r: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
    """Circular velocity v=sqrt(r g), clipped to avoid noise-induced negatives."""
    r = np.asarray(r, dtype=float)
    acceleration = np.asarray(acceleration, dtype=float)
    return np.sqrt(np.maximum(r * acceleration, 0.0))


def nfw_shape_mass(r: np.ndarray, r_s: float = 12.0) -> np.ndarray:
    """Dimensionless NFW-like cumulative mass shape."""
    r = np.asarray(r, dtype=float)
    x = np.maximum(r / max(r_s, EPS), 0.0)
    return np.log1p(x) - x / (1.0 + x)


def nfw_residual(r: np.ndarray, mass_scale: float = 22.0, r_s: float = 12.0) -> np.ndarray:
    """NFW-like residual acceleration from a scaled cumulative mass shape."""
    mass = mass_scale * nfw_shape_mass(r, r_s=r_s)
    return acceleration_from_mass(r, mass)


def isothermal_residual(r: np.ndarray, v0: float = 125.0, r_c: float = 4.0) -> np.ndarray:
    """Cored/isothermal-like residual acceleration.

    v_iso^2 = v0^2 r^2 / (r^2 + r_c^2), so g_iso = v_iso^2 / r.
    """
    r = np.asarray(r, dtype=float)
    return (v0**2) * r / (r**2 + max(r_c, EPS) ** 2)


def mond_residual(r: np.ndarray, g_bar: np.ndarray, a0: float = 1000.0) -> np.ndarray:
    """Simple MOND-like residual acceleration from a baryonic acceleration."""
    del r  # The local toy form depends on g_bar, not directly on radius.
    g_bar = np.asarray(g_bar, dtype=float)
    g_total = 0.5 * (g_bar + np.sqrt(g_bar**2 + 4.0 * a0 * np.maximum(g_bar, 0.0)))
    return np.maximum(g_total - g_bar, 0.0)


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Small local cumulative trapezoid helper to avoid requiring scipy here."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(y)
    if len(y) > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def soliton_density(r: np.ndarray, rho0: float = 0.35, r_c: float = 3.0) -> np.ndarray:
    """Cored scalar-field/soliton-inspired density shape."""
    r = np.asarray(r, dtype=float)
    x2 = (r / max(r_c, EPS)) ** 2
    return rho0 * (1.0 + x2) ** -8


def soliton_mass(r: np.ndarray, rho0: float = 0.35, r_c: float = 3.0) -> np.ndarray:
    """Numerically integrate the soliton-inspired density to M(<r)."""
    r = np.asarray(r, dtype=float)
    r_ext = np.concatenate(([0.0], r))
    rho_ext = soliton_density(r_ext, rho0=rho0, r_c=r_c)
    integrand = 4.0 * np.pi * r_ext**2 * rho_ext
    return _cumulative_trapezoid(integrand, r_ext)[1:]


def soliton_residual(r: np.ndarray, rho0: float = 0.35, r_c: float = 3.0) -> np.ndarray:
    """Acceleration from the soliton-inspired cored density."""
    return acceleration_from_mass(r, soliton_mass(r, rho0=rho0, r_c=r_c))


def mixed_residual(
    r: np.ndarray,
    g_bar: np.ndarray,
    nfw_weight: float = 0.65,
    mond_weight: float = 0.35,
    nfw_mass_scale: float = 22.0,
    nfw_r_s: float = 12.0,
    mond_a0: float = 1000.0,
) -> np.ndarray:
    """Two-source residual used to demonstrate latent-generator degeneracy."""
    nfw = nfw_residual(r, mass_scale=nfw_mass_scale, r_s=nfw_r_s)
    mond = mond_residual(r, g_bar=g_bar, a0=mond_a0)
    return nfw_weight * nfw + mond_weight * mond

