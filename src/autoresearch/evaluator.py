"""Frozen evaluator: score and Laplace metrics on the held-out bank, and the hard constraint checks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from heat_inverse.cases import case2

from . import problem as P

TOL_SPACING = 5e-5  # m; the soft penalty leaves the smallest spacing a few µm below D_MIN
TOL_FACE = 1e-9  # m


@jax.jit
def _metrics(X, B):
    A = P.information(X, B)
    det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] ** 2
    C11, C22, C12 = A[:, 1, 1] / det, A[:, 0, 0] / det, -A[:, 0, 1] / det
    s1, s2 = jnp.sqrt(C11), jnp.sqrt(C22)
    return -jnp.mean(jnp.log(det)), jnp.mean(s1), jnp.mean(s2), jnp.mean(C12 / (s1 * s2))


def min_spacing(X) -> float:
    """Smallest distance between two sensors, m."""
    X = np.asarray(X)
    if len(X) < 2:
        return float("inf")
    return float(np.linalg.norm(X[:, None] - X[None], axis=-1)[np.triu_indices(len(X), 1)].min())


def on_faces(X, faces) -> bool:
    """Every sensor on its named face only, and inside the margins along both in-plane axes."""
    X = np.asarray(X)
    if any(a != (f,) for a, f in zip(case2.faces_at(X, tol=TOL_FACE), faces)):
        return False
    ip = P.IN_PLANE[[P.FACE[f].axis for f in faces]]
    x = X[np.arange(len(X))[:, None], ip]
    return bool(np.all(x >= P.M_LO[ip] - TOL_FACE) and np.all(x <= P.L3[ip] - P.M_HI[ip] + TOL_FACE))


def evaluate(X, faces, B_val) -> dict:
    """Score Φ_val, bank-mean s₁, s₂, ρ, and the feasibility of the final design."""
    score, s1, s2, rho = (float(v) for v in _metrics(jnp.asarray(X), B_val))
    d = min_spacing(X)
    on = on_faces(X, faces)
    return dict(score=score, s1=s1, s2=s2, rho=rho, min_spacing_mm=1e3 * d, on_faces=on,
                feasible=bool(on and d >= P.D_MIN - TOL_SPACING))
