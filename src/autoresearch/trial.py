"""One trial: L-BFGS-B over the in-plane coordinates of every sensor, on the training bank."""

from __future__ import annotations

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from . import problem as P

FTOL = 2.220446049250313e-09  # SciPy L-BFGS-B default, used by run_dopt_case2.py
GTOL = 1e-05  # SciPy L-BFGS-B default


def face_axes(faces):
    """(N, 2) in-plane axes of each sensor's face."""
    return P.IN_PLANE[[P.FACE[f].axis for f in faces]]


def to_X(u, ip, X0):
    """Unit in-plane coordinates -> (N, 3) in m; the normal coordinate of each sensor is kept from X0."""
    rows = jnp.arange(len(ip))[:, None]
    return jnp.asarray(X0).at[rows, ip].set(u.reshape(-1, 2) * jnp.asarray(P.L3)[ip])


phi_u = jax.jit(lambda u, B, ip, X0: P.phi(to_X(u, ip, X0), B))
loss_and_grad = jax.jit(jax.value_and_grad(
    lambda u, B, ip, X0: P.phi(to_X(u, ip, X0), B) + P.MU * P.spacing(to_X(u, ip, X0))))


def bounds_and_start(X0, faces):
    """In-plane axes, unit bounds inside the margins, and the start u0: X0 clipped to them."""
    ip = face_axes(faces)
    lo, hi = (P.M_LO / P.L3)[ip].ravel(), 1 - (P.M_HI / P.L3)[ip].ravel()
    u0 = np.clip((X0 / P.L3)[np.arange(len(X0))[:, None], ip].ravel(), lo, hi)
    return ip, lo, hi, u0


class TrialResult(NamedTuple):
    X_start: np.ndarray
    X: np.ndarray
    X_snapshot: np.ndarray | None  # design after snapshot_iter iterations, None if the run stopped earlier
    faces: np.ndarray
    nit: int
    converged: bool
    killed: bool
    message: str
    loss: np.ndarray  # training loss after each iteration
    t_iter: np.ndarray  # s since the start, after each iteration
    wall_time: float


def optimise(X0, faces, B_train, maxiter, kill_time=None, snapshot_iter=None) -> TrialResult:
    """L-BFGS-B from X0; stops at convergence, maxiter, or after kill_time s (killed)."""
    ip, lo, hi, u0 = bounds_and_start(X0, faces)
    loss, t_iter, snap, killed = [], [], [None], [False]
    t0 = time.perf_counter()

    def fun(u):
        v, dv = loss_and_grad(jnp.asarray(u), B_train, ip, X0)
        return float(v), np.asarray(dv, dtype=float)

    def callback(intermediate_result):
        loss.append(float(intermediate_result.fun))
        t_iter.append(time.perf_counter() - t0)
        if snapshot_iter is not None and len(loss) == snapshot_iter:
            snap[0] = np.array(intermediate_result.x)
        if kill_time is not None and t_iter[-1] > kill_time:
            killed[0] = True
            raise StopIteration

    r = minimize(fun, u0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)), callback=callback,
                 options=dict(maxiter=maxiter, ftol=FTOL, gtol=GTOL))
    wall = time.perf_counter() - t0
    X = lambda u: np.asarray(to_X(jnp.asarray(u), ip, X0))  # noqa: E731
    return TrialResult(X_start=X(u0), X=X(r.x), X_snapshot=None if snap[0] is None else X(snap[0]),
                       faces=np.asarray(faces), nit=int(r.nit), converged=bool(r.success) and not killed[0],
                       killed=killed[0], message=str(r.message), loss=np.array(loss), t_iter=np.array(t_iter),
                       wall_time=wall)
