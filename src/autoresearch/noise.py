"""Start noise of the best-of-R score: bootstrap over the starts of one sensor_layout, τ(R) and the choice of R."""

from __future__ import annotations

import numpy as np

R_GRID = (1, 2, 4, 8)
N_DRAWS = 1000
EPS_GAIN = 2 * np.log(1.10)  # one of s₁, s₂ changes by 10 % at fixed ρ: Δ log det C = 2 ln 1.10


def best_of_r(phi_train, phi_val, valid, R, n_draws=N_DRAWS, seed=0) -> np.ndarray:
    """Φ_val of the valid start with the lowest Φ_train in each of n_draws draws of R starts with replacement;
    NaN for a draw without a valid start."""
    phi_train, phi_val, valid = (np.asarray(a) for a in (phi_train, phi_val, valid))
    idx = np.random.default_rng(seed).integers(len(phi_train), size=(n_draws, R))
    pt = np.where(valid[idx], phi_train[idx], np.inf)
    best = phi_val[idx[np.arange(n_draws), np.argmin(pt, axis=1)]]
    return np.where(np.isfinite(pt.min(axis=1)), best, np.nan)


def tau(sigma):
    """Threshold of one comparison of two independent scores with standard deviation σ: 2 √2 σ."""
    return 2 * np.sqrt(2) * np.asarray(sigma)


def choose_R(tau_of_R: dict, eps=EPS_GAIN) -> int:
    """Smallest R with τ(R) <= ε; the largest R if none."""
    ok = [R for R in sorted(tau_of_R) if tau_of_R[R] <= eps]
    return ok[0] if ok else max(tau_of_R)
