"""Batched box-constrained L-BFGS on the GPU: independent starts of trial.loss in one vectorised loop."""

from __future__ import annotations

import functools
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import problem as P
from . import trial

M_HIST = 10  # SciPy L-BFGS-B default maxcor
MAX_LS = 20  # SciPy L-BFGS-B default maxls
C_ARMIJO = 1e-3  # sufficient-decrease constant of the L-BFGS-B line search (dcsrch ftol)
EPS_ACTIVE = 1e-3  # unit coordinates; capped by the projected-gradient norm, so the ε-active set shrinks to the bounds
EPS_MACH = float(np.finfo(np.float64).eps)
CHUNK = 100  # iterations per device call; the host checks the kill time between calls

RUNNING, CONV_F, CONV_G, MAXITER, LS_FAIL, KILLED, PAD = range(7)
STATUS = ("running", "conv_ftol", "conv_pgtol", "maxiter", "ls_fail", "killed", "pad")


class State(NamedTuple):
    u: jax.Array
    f: jax.Array
    g: jax.Array
    S: jax.Array  # (M_HIST, n) steps, circular buffer
    Y: jax.Array  # (M_HIST, n) gradient changes, circular buffer
    rho: jax.Array
    n_mem: jax.Array
    head: jax.Array  # next slot of the buffer
    nit: jax.Array
    nfev: jax.Array
    status: jax.Array
    k: jax.Array  # iterations in the current chunk
    hist: jax.Array  # loss after each iteration, NaN after the last
    lo: jax.Array
    hi: jax.Array
    ip: jax.Array
    X0: jax.Array


_vg = jax.value_and_grad(trial.loss)
loss_and_grad_batch = jax.jit(jax.vmap(_vg, in_axes=(0, None, 0, 0)))
phi_batch = jax.jit(jax.vmap(P.phi, in_axes=(0, None)))
_to_X = jax.jit(jax.vmap(trial.to_X))


def _int(v):
    return jnp.asarray(v, dtype=int)


def _pg_norm(u, g, lo, hi):
    """Infinity norm of the projected gradient (the pgtol test of L-BFGS-B)."""
    return jnp.max(jnp.abs(jnp.clip(u - g, lo, hi) - u))


def _init(u, lo, hi, ip, X0, pad, B, maxiter):
    f, g = _vg(u, B, ip, X0)
    status = jnp.where(pad, PAD, jnp.where(_pg_norm(u, g, lo, hi) <= trial.GTOL, CONV_G, RUNNING))
    zero, mem = _int(0), jnp.zeros((M_HIST, u.shape[0]))
    return State(u=u, f=f, g=g, S=mem, Y=mem, rho=jnp.zeros(M_HIST), n_mem=zero, head=zero, nit=zero,
                 nfev=_int(1), status=_int(status), k=zero, hist=jnp.full(maxiter + 1, jnp.nan).at[0].set(f),
                 lo=lo, hi=hi, ip=ip, X0=X0)


def _two_loop(q, s):
    """H q for the L-BFGS inverse-Hessian approximation with H0 = γ I, γ = sᵀy / yᵀy of the newest pair."""
    def back(i, c):
        q, a = c
        j = (s.head - 1 - i) % M_HIST
        ai = jnp.where(i < s.n_mem, s.rho[j] * (s.S[j] @ q), 0.0)
        return q - ai * s.Y[j], a.at[j].set(ai)

    q, a = jax.lax.fori_loop(0, M_HIST, back, (q, jnp.zeros(M_HIST)))
    j = (s.head - 1) % M_HIST
    gamma = jnp.where(s.n_mem > 0, (s.S[j] @ s.Y[j]) / (s.Y[j] @ s.Y[j]), 1.0)

    def forward(i, r):
        j = (s.head - s.n_mem + i) % M_HIST
        return r + jnp.where(i < s.n_mem, a[j] - s.rho[j] * (s.Y[j] @ r), 0.0) * s.S[j]

    return jax.lax.fori_loop(0, M_HIST, forward, gamma * q), gamma


def _direction(s):
    """Two-metric projection (Bertsekas 1982): L-BFGS on the free variables, γ-scaled steepest descent on the
    ε-active ones; -g if the result is not a descent direction."""
    eps = jnp.minimum(EPS_ACTIVE, _pg_norm(s.u, s.g, s.lo, s.hi))
    active = ((s.u <= s.lo + eps) & (s.g > 0)) | ((s.u >= s.hi - eps) & (s.g < 0))
    r, gamma = _two_loop(jnp.where(active, 0.0, s.g), s)
    d = jnp.where(active, -gamma * s.g, -r)
    descent = s.g @ d < 0
    return jnp.where(descent, d, -s.g), descent


def _line_search(s, d, B):
    """Halving on the projected path P(u + α d) from α = 1 (the L-BFGS-B first trial on a fully boxed problem)
    until the Armijo condition holds -> (u, f, g, accepted, evaluations)."""
    running = s.status == RUNNING

    def point(a):
        u = jnp.clip(s.u + a * d, s.lo, s.hi)
        f, g = _vg(u, B, s.ip, s.X0)
        return u, f, g, jnp.isfinite(f) & (f <= s.f + C_ARMIJO * (s.g @ (u - s.u)))

    def body(c):
        a, *_, n = c
        return (0.5 * a, *point(0.5 * a), n + 1)

    # a stopped start must not keep the vectorised loop alive
    c = jax.lax.while_loop(lambda c: running & ~c[4] & (c[5] < MAX_LS), body,
                           (jnp.asarray(1.0), *point(1.0), _int(1)))
    return c[1:]


def _iterate(s, B, maxiter):
    d, descent = _direction(s)
    n_mem = jnp.where(descent, s.n_mem, 0)
    u, f, g, ok, nev = _line_search(s, d, B)
    # as L-BFGS-B: a failed search with curvature pairs drops them and retries along -g
    retry = ~ok & (n_mem > 0)
    sv, yv = u - s.u, g - s.g
    sy = sv @ yv
    keep = ok & (sy > EPS_MACH * -(s.g @ sv))
    S = jnp.where(keep, s.S.at[s.head].set(sv), s.S)
    Y = jnp.where(keep, s.Y.at[s.head].set(yv), s.Y)
    rho = jnp.where(keep, s.rho.at[s.head].set(1.0 / sy), s.rho)
    head = jnp.where(keep, (s.head + 1) % M_HIST, s.head)
    n_mem = jnp.where(keep, jnp.minimum(n_mem + 1, M_HIST), jnp.where(retry, 0, n_mem))
    u, f, g = jnp.where(ok, u, s.u), jnp.where(ok, f, s.f), jnp.where(ok, g, s.g)
    nit = s.nit + ok
    conv_g = ok & (_pg_norm(u, g, s.lo, s.hi) <= trial.GTOL)
    conv_f = ok & (s.f - f <= trial.FTOL * jnp.maximum(jnp.maximum(jnp.abs(s.f), jnp.abs(f)), 1.0))
    status = jnp.select([conv_g, conv_f, ~ok & ~retry, nit >= maxiter], [CONV_G, CONV_F, LS_FAIL, MAXITER],
                        RUNNING)
    return s._replace(u=u, f=f, g=g, S=S, Y=Y, rho=rho, n_mem=_int(n_mem), head=_int(head), nit=_int(nit),
                      nfev=s.nfev + nev, status=_int(status), k=s.k + 1, hist=s.hist.at[nit].set(f))


def _chunk(s, B, n, maxiter):
    s = s._replace(k=jnp.zeros_like(s.k))
    return jax.lax.while_loop(lambda s: (s.status == RUNNING) & (s.k < n), lambda s: _iterate(s, B, maxiter), s)


@functools.lru_cache(maxsize=None)
def _compiled(maxiter: int):
    init = jax.jit(jax.vmap(functools.partial(_init, maxiter=maxiter), in_axes=(0, 0, 0, 0, 0, 0, None)))
    chunk = jax.jit(jax.vmap(functools.partial(_chunk, maxiter=maxiter), in_axes=(0, None, None)))
    return init, chunk


def stack_starts(starts, size):
    """[(X0, faces), ...] -> device arrays (u0, lo, hi, ip, X0, pad), padded to size with copies of the first start."""
    rows = [trial.bounds_and_start(np.asarray(X0, dtype=float), faces) for X0, faces in starts]
    idx = list(range(len(starts))) + [0] * (size - len(starts))
    ip, lo, hi, u0 = (np.stack([rows[i][k] for i in idx]) for k in range(4))
    X0 = np.stack([np.asarray(starts[i][0], dtype=float) for i in idx])
    return tuple(jnp.asarray(a) for a in (u0, lo, hi, ip, X0, np.arange(size) >= len(starts)))


class BatchResult(NamedTuple):
    X_start: np.ndarray  # (b, N, 3)
    X: np.ndarray
    faces: np.ndarray  # (b, N)
    loss: np.ndarray  # final training loss, penalty included
    nit: np.ndarray
    nfev: np.ndarray
    status: np.ndarray  # names from STATUS
    converged: np.ndarray
    hist: np.ndarray  # (b, maxiter + 1) loss at the start and after each iteration, NaN after the last
    wall_time: float  # s, the compile of a new padded size included
    size: int  # padded batch size


def optimise(starts, B_train, maxiter, kill_time=None, pad_to=None) -> BatchResult:
    """All starts [(X0, faces), ...] (equal N) in one vectorised run, padded to pad_to. Each start stops on its own
    pgtol, ftol, maxiter or line-search test; starts still running after kill_time s get status killed."""
    b = len(starts)
    size = max(b, pad_to or b)
    init, chunk = _compiled(maxiter)
    t0 = time.perf_counter()
    args = stack_starts(starts, size)
    s = init(*args, B_train)
    while True:
        s = chunk(s, B_train, CHUNK)
        status = np.asarray(s.status)
        if not (status == RUNNING).any():
            break
        if kill_time is not None and time.perf_counter() - t0 > kill_time:
            status = np.where(status == RUNNING, KILLED, status)
            break
    wall = time.perf_counter() - t0
    status = status[:b]
    return BatchResult(X_start=np.asarray(_to_X(args[0], args[3], args[4]))[:b],
                       X=np.asarray(_to_X(s.u, s.ip, s.X0))[:b], faces=np.stack([np.asarray(f) for _, f in starts]),
                       loss=np.asarray(s.f)[:b], nit=np.asarray(s.nit)[:b], nfev=np.asarray(s.nfev)[:b],
                       status=np.array(STATUS)[status], converged=np.isin(status, (CONV_F, CONV_G)),
                       hist=np.asarray(s.hist)[:b], wall_time=wall, size=size)


def iteration_time(starts, B_train, n_iter, maxiter) -> float:
    """s per loop iteration of one batch from its start designs, compile excluded."""
    init, chunk = _compiled(maxiter)
    s = init(*stack_starts(starts, len(starts)), B_train)
    jax.block_until_ready(chunk(s, B_train, 1))
    t0 = time.perf_counter()
    s = jax.block_until_ready(chunk(s, B_train, n_iter))
    return (time.perf_counter() - t0) / max(int(np.asarray(s.k).max()), 1)
