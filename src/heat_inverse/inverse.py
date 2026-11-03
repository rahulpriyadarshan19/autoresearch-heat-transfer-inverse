"""Reduced-space forward map for inferring conductivities.

THE REDUCED-SPACE IDEA. Equation (1) is bilinear in (T, k): with T = Phi beta,
the PDE term k_i (Phi_ii beta) is a product of two unknowns, so beta and k
cannot be solved for jointly by linear algebra. But for FIXED k the problem in
beta is an ordinary linear forward solve. So rather than searching the joint
space, we define

    beta(k) = the exact null-space-projected solution at that k

and search only over k, scoring each candidate by how well Phi_data beta(k)
reproduces the observations. The physics is satisfied exactly at every trial k;
only the data misfit is being minimised. This is the classical Golub-Pereyra
variable-projection formulation, and it is what makes a 1-3 parameter problem
out of one that otherwise couples thousands of coefficients.

WHY THIS MODULE EXISTS SEPARATELY FROM ``solver.py``. ``solve_forward`` is the
readable reference: one call, one solve, full diagnostics. It is also nearly
unbatchable -- profiled at 6.05 ms/particle under vmap against 8.89 ms for a
single solve, because jnp.linalg.svd and jnp.linalg.lstsq both dispatch to
LAPACK and batched LAPACK on CPU is a serial loop over the batch. An ensemble
method calls the forward map J times per iteration, so that cost is the whole
runtime.

``make_beta_of_k`` is the same mathematics restructured for the ensemble:

  1. HOIST THE CONSTRAINT SVD. For Case 1.1 the boundary rows do not depend on
     k, so beta_star and the null-space basis W are computed once at setup
     rather than J times per iteration.

  2. QR INSTEAD OF LSTSQ. lstsq is SVD-based, for robustness to rank
     deficiency -- but after projection onto the null space of H_c the matrix
     H_f W is full column rank BY CONSTRUCTION: the deficiency lived in H_c and
     the projection removed it. A QR plus one triangular solve answers the same
     least-squares question. Measured identical to lstsq to the last digit at
     every degree tested.

  3. HOIST THE QR TOO, where k is a common scalar factor. For isotropic k,
     H_f(k) = k*(Phi_xx + Phi_yy) = k*A, and row-normalisation divides k back
     out, so the normalised projected matrix does not depend on k at all.

Measured at J = 128, degree 12, all three variants agreeing to 8.022e-11 MAE:

    reference solver under vmap     774 ms      6.05 ms/particle
    hoisted SVD, still lstsq        583 ms      4.55 ms/particle
    hoisted SVD + hoisted QR       0.57 ms     0.004 ms/particle

CHOLESKY ON THE NORMAL EQUATIONS WAS MEASURED AND REJECTED. It is faster still
(0.19 ms/particle in the orthotropic case) but squares the condition number.
At degree 16 -- the accuracy optimum found in phase 1 -- kappa(M^T M) reaches
2.8e16 and the MAE degrades from 1.1e-14 to 5.1e-09; by degree 20 it is 9.6e-06.
QR costs a little more and does not do that.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

from . import config  # noqa: F401  (enables x64 before any array is made)

import jax
import jax.numpy as jnp
import numpy as np

from .solver import svd


class Observations(NamedTuple):
    """Synthetic temperature measurements at sensor locations.

    Attributes:
        x:      (n_obs, d) sensor coordinates.
        T:      (n_obs,) observed temperatures, noise included.
        T_clean:(n_obs,) the noise-free truth, kept for diagnostics only --
                never used by the inference.
        sigma:  observation noise standard deviation.
        Phi:    (n_obs, N*) basis values at the sensors, precomputed because
                the forward map needs them on every particle of every step.
    """

    x: jnp.ndarray
    T: jnp.ndarray
    T_clean: jnp.ndarray
    sigma: float
    Phi: jnp.ndarray


def make_observations(
    basis,
    x_obs,
    exact_fn,
    sigma_frac: float,
    seed: int,
    *,
    sigma_mode: str = "range",
    T_scale: float | None = None,
    sigma_abs: float | None = None,
) -> Observations:
    """Sample noisy temperatures from an exact solution at sensor locations.

    ``sigma_frac`` is a fraction of a temperature SCALE, chosen by
    ``sigma_mode``. Expressing it relatively keeps the noise meaningful when
    the same code is pointed at a case whose temperatures are O(300 K) rather
    than O(1).

      "range"    -- the temperature range over the SENSORS. Depends on where
                    the sensors happen to fall, so moving them changes sigma.
      "max_abs"  -- the absolute magnitude of the field, max|T|, which is how
                    the paper states its 1% noise. Pass ``T_scale`` (max|T|
                    over the DOMAIN) to make sigma independent of the sensor
                    layout; without it max|T| over the sensors is used.

    ``sigma_abs``, when given, sets sigma directly in kelvin and overrides
    ``sigma_frac`` and ``sigma_mode`` -- for a noise level stated as a sensor
    accuracy rather than as a fraction of the field.
    """
    x_obs = jnp.asarray(x_obs)
    T_clean = exact_fn(x_obs)
    if sigma_abs is not None:
        scale, sigma_frac = 1.0, float(sigma_abs)
    elif sigma_mode == "range":
        scale = float(jnp.max(T_clean) - jnp.min(T_clean))
    elif sigma_mode == "max_abs":
        scale = float(jnp.max(jnp.abs(T_clean))) if T_scale is None else float(T_scale)
    else:
        raise ValueError(f"unknown sigma_mode {sigma_mode!r}")
    sigma = sigma_frac * scale

    rng = np.random.default_rng(seed)
    noise = jnp.asarray(rng.normal(0.0, sigma, size=T_clean.shape))

    return Observations(
        x=x_obs,
        T=T_clean + noise,
        T_clean=T_clean,
        sigma=sigma,
        Phi=basis.features(x_obs),
    )


def _row_normalise(H, K):
    """Scale each row to unit L2 norm, applying the same factor to the RHS."""
    norms = jnp.linalg.norm(H, axis=1, keepdims=True)
    norms = jnp.where(norms < 1e-300, 1.0, norms)
    return H / norms, K / norms.reshape(-1)


def _nullspace(H_c, K_c, rcond: float):
    """Particular solution and null-space basis of the constraint system.

    Both come from ONE SVD with ONE rank cut, so W is exactly orthogonal to the
    row space of H_c and the correction W eta cannot contaminate the
    constraints. Unlike the jitted path in ``solver.py`` this slices by a
    concrete rank -- it runs once at setup, outside any trace, so the shapes
    are free to depend on the data and W carries only the genuine null
    directions.
    """
    U, S, Vt = svd(H_c, full_matrices=True)
    rank = int(jnp.sum(S > rcond * S[0]))
    S_r = S[:rank]
    beta_star = Vt[:rank].T @ ((U[:, :rank].T @ K_c) / S_r)
    W = Vt[rank:].T  # (N*, N* - rank)
    return beta_star, W, rank


def make_beta_of_k(
    basis,
    x_f,
    rhs_f,
    constraints,
    *,
    rcond: float = config.RCOND,
    k_is_common_factor: bool = False,
) -> tuple[Callable, int]:
    """Build a fast, vmap-friendly map k -> beta.

    Args:
        basis:      a ``TensorBasis``.
        x_f:        (n_f, d) interior collocation points.
        rhs_f:      (n_f,) source term at those points.
        constraints: assembled boundary rows, k-independent.
        rcond:      relative rank threshold for the constraint SVD.
        k_is_common_factor: True when every Laplacian block carries the SAME
            coefficient (isotropic k), which lets the QR be hoisted as well.
            Setting it wrongly is caught by the reference check in the driver.

    Returns:
        (beta_of_k, rank). ``beta_of_k`` maps a parameter vector to the
        coefficient vector, is pure, and contains no host callbacks -- it is
        traced by ``jax.vmap`` over the whole ensemble.

    Case 2's Robin conditions make H_c depend on k, so its SVD cannot be
    hoisted; passing k-dependent constraints here would silently give the
    wrong answer. Use ``make_beta_of_k_varying`` for that case.
    """
    x_f = jnp.asarray(x_f)
    rhs_f = jnp.asarray(rhs_f).reshape(-1)

    H_c, K_c = _row_normalise(constraints.H, constraints.K)
    beta_star, W, rank = _nullspace(H_c, K_c, rcond)

    if rank >= basis.n_star:
        raise ValueError(
            f"constraints exhaust the basis: rank {rank} >= N* {basis.n_star}"
        )

    # Projected Laplacian blocks: k-independent, so built once.
    blocks = basis.laplacian_blocks(x_f)
    PB = [blk @ W for blk in blocks]  # (n_f, m) each
    pb = [blk @ beta_star for blk in blocks]  # (n_f,) each

    if k_is_common_factor:
        # H_f(k) = k * sum_i Phi_ii, so the row norms carry the same k and it
        # cancels: the normalised projected matrix is independent of k, and so
        # is its QR. Only the right-hand side varies per particle.
        A = sum(blocks)
        nr = jnp.linalg.norm(A, axis=1, keepdims=True)
        nr = jnp.where(nr < 1e-300, 1.0, nr).reshape(-1)
        Q, R = jnp.linalg.qr(sum(PB) / nr[:, None])
        Abs_ = sum(pb) / nr

        def beta_of_k(k):
            kk = jnp.asarray(k).reshape(-1)[0]
            y = rhs_f / (kk * nr) - Abs_
            eta = jax.scipy.linalg.solve_triangular(R, Q.T @ y)
            return beta_star + W @ eta

    else:
        # Orthotropic: each block carries its own coefficient, so the projected
        # matrix genuinely depends on k and the QR is per-particle. Still far
        # cheaper than lstsq (measured 2.0 vs 4.6 ms/particle).
        def beta_of_k(k):
            kv = jnp.asarray(k).reshape(-1)
            M = sum(kv[i] * PB[i] for i in range(len(PB)))
            y = rhs_f - sum(kv[i] * pb[i] for i in range(len(pb)))
            nr = jnp.linalg.norm(M, axis=1, keepdims=True)
            nr = jnp.where(nr < 1e-300, 1.0, nr).reshape(-1)
            Q, R = jnp.linalg.qr(M / nr[:, None])
            eta = jax.scipy.linalg.solve_triangular(R, Q.T @ (y / nr))
            return beta_star + W @ eta

    return beta_of_k, rank


def make_beta_of_k_varying(
    basis,
    x_f,
    rhs_f,
    constraints_of_k: Callable,
    *,
    rcond: float = config.RCOND,
    ls_method: str = "svd",
    k_rank=None,
    ns_method: str = "svd",
) -> Callable:
    """k -> beta when the CONSTRAINTS depend on k (Case 2's Robin and flux rows).

    Nothing can be hoisted: H_c(k) moves with k, so its null space does too,
    and each call rebuilds the rows (cheap, from precomputed face blocks inside
    ``constraints_of_k``) and runs the full projection of ``solver.py`` -- one
    SVD and one least squares per particle.

    Measured cost under vmap: batched SVD runs as a serial loop on both CPU and
    GPU (261 ms per 1100 x 847 matrix at batch 1, 8 and 32 on the 2080 Ti), so
    an ensemble costs J single solves. The Laplacian blocks are still
    k-independent and are built once.

    ``ls_method`` picks the reduced least-squares solver (``solver.LS_METHODS``)
    and ``ns_method`` the null-space factorisation (``solver.NS_METHODS``).
    "qr", "cholqr3" and "qrcp" slice W by a rank fixed at setup, so they need
    ``k_rank``: the k (as passed to ``constraints_of_k``) at which rank(H_c) is
    measured once. The rank must not move with k; a particle where it does
    comes back as NaN, which the samplers already treat as a failed solve.
    """
    from .solver import LS_METHODS, NS_METHODS, _project_and_solve, constraint_rank

    if ls_method not in LS_METHODS:
        raise ValueError(f"unknown ls_method {ls_method!r}; choose from {LS_METHODS}")
    if ns_method not in NS_METHODS:
        raise ValueError(f"unknown ns_method {ns_method!r}; choose from {NS_METHODS}")
    x_f = jnp.asarray(x_f)
    rhs_f = jnp.asarray(rhs_f).reshape(-1)
    blocks = basis.laplacian_blocks(x_f)

    null_dim = None
    if ls_method != "svd" or ns_method != "svd":
        if k_rank is None:
            raise ValueError(f"ls_method={ls_method!r}, ns_method={ns_method!r} need "
                             "k_rank to fix rank(H_c)")
        c = constraints_of_k(jnp.asarray(k_rank, dtype=float).reshape(-1))
        null_dim = basis.n_star - int(constraint_rank(_row_normalise(c.H, c.K)[0], rcond))
        if null_dim <= 0:
            raise ValueError(f"constraints exhaust the basis: null space {null_dim} at k_rank")

    def beta_of_k(k):
        kv = jnp.asarray(k).reshape(-1)
        H_f, K_f = _row_normalise(sum(kv[i] * blocks[i] for i in range(len(blocks))), rhs_f)
        c = constraints_of_k(kv)
        H_c, K_c = _row_normalise(c.H, c.K)
        return _project_and_solve(H_f, K_f, H_c, K_c, rcond, ls_method, null_dim,
                                  ns_method)[0]

    return beta_of_k


def make_forward_fn(beta_of_k, obs: Observations, *, log_space: bool = False,
                    t_offset: float = 0.0, t_scale: float = 1.0) -> Callable:
    """Build the Gamma-whitened data residual map, p -> (T_pred - T_obs)/sigma.

    T_pred = t_offset + t_scale * (Phi beta): the defaults are for a solver
    whose field is T itself; a scaled field theta = (T - T_inf)/DT passes
    t_offset = T_inf, t_scale = DT, so observations and sigma stay in kelvin.

    THE WHITENING IS THE POINT: dividing by sigma is exactly the Gamma-weighted
    inner product, since sum((r/sigma)^2) = r^T Gamma^-1 r for Gamma = sigma^2 I.
    Carrying Gamma implicitly this way makes a whole class of bug
    unrepresentable -- the unwhitened form would need Gamma^-1 = sigma^-2 I
    inside the sampler, and dropping that factor shrinks the EKS drift, so the
    sampler appears to converge smoothly while erroring nowhere.

    NO PRIOR BLOCK. EKS gets its prior from the C Sigma^-1 damping term;
    appending a prior residual here as well would double-count it.

    The returned function is pure: no printing, no diagnostics, no
    block_until_ready. It is traced once by jax.jit(jax.vmap(...)) inside
    run_eks, where anything else would be paid for on every particle of every
    iteration.

    LOG SPACE. With ``log_space=True`` the argument is theta = log k and the
    exponential is taken inside, so every consumer -- EKS, NUTS, and later a
    CES emulator -- works in one set of coordinates without transporting
    covariances between them. Two reasons this is the better default for
    conductivities:

      * POSITIVITY IS STRUCTURAL. k = exp(theta) cannot be non-positive, so no
        particle or trajectory can reach the region where the PDE operator
        degenerates. Sampling k directly, EKS survives excursions there only
        via its non-finite resampling path, and NUTS does not survive them at
        all -- measured at acceptance 0.000 with 7966 divergences in 8000
        draws before this flag existed.

      * THE COORDINATES MATCH. An EKS ensemble covariance in k cannot be used
        as a NUTS mass matrix in theta without a delta-method transform, and
        the two samplers would otherwise be given different priors and so be
        answering different questions -- which makes any comparison between
        them meaningless.
    """
    Phi, T_obs, sigma = obs.Phi, obs.T, obs.sigma

    if log_space:

        def forward_fn(theta):
            return (t_offset + t_scale * (Phi @ beta_of_k(jnp.exp(theta))) - T_obs) / sigma

    else:

        def forward_fn(k):
            return (t_offset + t_scale * (Phi @ beta_of_k(k)) - T_obs) / sigma

    return forward_fn
