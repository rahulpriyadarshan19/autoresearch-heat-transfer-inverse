"""Start noise of the best-of-R score: bootstrap over the starts of one sensor_layout, τ(R), the choice of R, the
best-of-n bias c(n), B_max, and the calibration read by the harness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

R_GRID = (1, 2, 4, 8)
N_DRAWS = 1000
EPS_GAIN = 2 * np.log(1.10)  # one of s₁, s₂ changes by 10 % at fixed ρ: Δ log det C = 2 ln 1.10
F_HYP, B_CAP = 0.05, 32  # one hypothesis uses at most 5 % of the session time; 32 holds all 2-face splits of 31
_X = np.linspace(-12.0, 12.0, 24001)


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


def expected_max(n: int) -> float:
    """c(n) = E[max of n independent standard normal values]; c(n) = 0 for n <= 1."""
    if n <= 1:
        return 0.0
    return float(np.trapezoid(_X * n * norm.pdf(_X) * norm.cdf(_X) ** (n - 1), _X))


def b_max(t_session, R, t_start, f=F_HYP, cap=B_CAP) -> int:
    """min(cap, ⌊f T_session / (R t̄_start)⌋), at least 1."""
    return int(min(cap, max(1, f * t_session // (R * t_start))))


def load_calibration(path, R=None, eps=EPS_GAIN) -> dict:
    """Session noise constants from the calibration.json of case2_ar_smoke_noise (file or its folder): R (the
    calibrated choice unless given), σ_R, τ = τ(R), ε, ε_eff = max(τ, ε), N, t̄_start, maxiter, kill time, run id."""
    path = Path(path)
    path = path / "calibration.json" if path.is_dir() else path
    cal = json.loads(path.read_text())
    R = int(cal["R"] if R is None else R)
    if str(R) not in cal["sigma_R"]:
        raise ValueError(f"R = {R} is not in the calibration grid {sorted(cal['sigma_R'], key=int)}")
    tau_R = float(cal["tau_R"][str(R)])
    ns = {sum(int(c) for c in cfg.split("-")) for cfg in cal["configs"]}
    if len(ns) != 1:
        raise ValueError(f"calibration sensor_layouts have different N: {sorted(ns)}")
    return dict(id=path.parent.name, file=str(path), R=R, sigma_R=float(cal["sigma_R"][str(R)]), tau=tau_R,
                eps=float(eps), eps_eff=max(tau_R, float(eps)), N=ns.pop(), t_start=float(cal["t_start_mean"]),
                maxiter=int(cal["maxiter"]), kill_time=float(cal["kill_time"]))
