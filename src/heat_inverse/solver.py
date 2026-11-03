"""Null-space-projected PIELM solver for linear steady heat conduction.

Solves, for a FIXED conductivity k, the constrained least-squares problem

    minimise  || H_f beta - K_f ||     subject to  H_c beta = K_c  exactly

where H_f are the PDE-residual rows (sum_i k_i Phi_ii) and H_c the boundary
rows. This is the forward map that the inverse layer calls once per trial k:
with k fixed, equation (1) of the MIPINN paper is linear in beta, so the solve
is direct and needs no iteration.

WHY NULL-SPACE PROJECTION rather than stacking the boundary rows into the
least-squares system with a weight: the constraints are then satisfied to
machine precision by construction rather than traded off against the PDE
residual. That turns the constraint residual into an independent correctness
signal -- if it is not ~1e-14, something is wrong with the boundary assembly or
the derivative code, and no amount of weight tuning will hide it.

TWO DEPARTURES from the reference implementation in
``pde_dl/random_feature_functions/train.py``, both deliberate:

1. ONE SVD, NO GRAM-SCHMIDT. The reference orthogonalises the constraint rows
   with a Python-loop modified Gram-Schmidt (train.py:61-78), one iteration per
   row. That is workable in eager PyTorch but would be catastrophic here: Case 2
   has 9000 boundary points, and a 9000-iteration Python loop cannot be traced.
   A single SVD of the row-normalised H_c achieves the same orthogonalisation in
   one LAPACK call -- the MGS loop was approximating the SVD's row space anyway.

2. RELATIVE RANK THRESHOLD. The reference uses an absolute 1e-16. See
   ``config.RCOND`` for why that does not survive a change of scale.

THE REDUCED LEAST SQUARES, min || H_f W eta - (K_f - H_f beta_star) ||, has
three solvers (``ls_method``):

    "svd"      jnp.linalg.lstsq, the default and the reference. SVD-based, so
               it tolerates the zero columns of the masked W below.
    "qr"       Householder QR plus one triangular solve.
    "cholqr3"  shifted CholeskyQR3 (``shifted_cholqr3``): QR built from Gram
               matrices, Cholesky factors and triangular solves only, which all
               batch on a GPU. Error grows like kappa, as for Householder, not
               like kappa^2 as for the normal equations -- but it breaks down
               (NaN) once kappa(H_f W) passes ~1e11 at Case 2's sizes; see
               ``shifted_cholqr3``.

The last two need a full-column-rank matrix, so they cannot take the masked W
(``_project_and_solve``); they slice W to its N* - rank genuine null directions
instead, which under jit needs that count as a Python int (``null_dim``).

THE NULL SPACE of H_c has two rank-revealing factorisations (``ns_method``,
``_null_space``): "svd", the default and the reference, and "qrcp", QR with
column pivoting of H_c^T. H_c is rank-deficient by construction, so plain QR
cannot be used here. "qrcp" also slices W, so it too needs ``null_dim``.
"""

from __future__ import annotations

import time
from functools import partial
from typing import NamedTuple

from . import config  # noqa: F401  (enables x64 before any array is made)

import jax
import jax.numpy as jnp
from jax.lax import linalg as lax_linalg
from jax.scipy.linalg import solve_triangular

LS_METHODS = ("svd", "qr", "cholqr3")
NS_METHODS = ("svd", "qrcp")


def svd(H, full_matrices: bool):
    """SVD pinned to LAPACK/cuSOLVER gesvd (bidiagonal QR iteration).

    Still a full SVD -- "QR" names the iteration used on the bidiagonal, not a
    QR factorisation. Pinned because JAX's defaults are not robust for these
    matrices: on CPU, divide-and-conquer (gesdd) returned all-NaN singular
    values for Case 2's H_c at degree (8, 6, 8), where 245 of 567 are below
    1e-13 sigma_max; on GPU the default switches to Jacobi for small matrices,
    up to 8x slower there and two digits less accurate. gesvd matched numpy's
    rank at every size tested and was never slower.
    """
    return lax_linalg.svd(
        H, full_matrices=full_matrices, algorithm=lax_linalg.SvdAlgorithm.QR
    )


class Constraints(NamedTuple):
    """Assembled constraint rows: H_c beta = K_c, enforced exactly.

    Dirichlet and Neumann rows live in the same block -- once assembled, the
    projection does not care which is which. ``labels`` is carried only so the
    diagnostics can report residuals per condition type, which is what localises
    a fault to the value code or the derivative code.
    """

    H: jnp.ndarray  # (n_c, N*)
    K: jnp.ndarray  # (n_c,)
    labels: tuple[str, ...] = ()


class Solution(NamedTuple):
    """Result of one forward solve."""

    beta: jnp.ndarray  # (N*,) coefficient vector
    rank: int  # rank of H_c -> constraints actually enforced
    n_star: int  # basis size; must exceed rank
    cond_pde: float  # condition number of the projected PDE matrix
    cond_constr: float  # condition number of H_c
    pde_residual: jnp.ndarray  # (n_f,) H_f beta - K_f, row-normalised
    constr_residual: jnp.ndarray  # (n_c,) H_c beta - K_c
    compile_time: float  # seconds, first (tracing + compiling) call
    solve_time: float  # seconds, steady-state call, device-synchronised


class OverConstrainedError(RuntimeError):
    """Raised when the constraints leave the basis no freedom to fit the PDE."""


def _row_normalise(H, K):
    """Scale each row to unit L2 norm, applying the same factor to the RHS.

    Row scaling leaves the solution set untouched but removes the scale
    disparity between second-derivative rows (O(1/L^2)) and boundary-value rows
    (O(1)), which would otherwise let the largest rows dominate the least
    squares. Follows train.py:490-498.
    """
    norms = jnp.linalg.norm(H, axis=1, keepdims=True)
    norms = jnp.where(norms < 1e-300, 1.0, norms)
    return H / norms, K / norms.reshape(-1)


def _cholqr(A):
    """One CholeskyQR pass: R from the Cholesky factor of A^T A, Q = A R^-1."""
    R = jnp.linalg.cholesky(A.T @ A).T
    Q = solve_triangular(R, A.T, trans="T", lower=False).T
    return Q, R


def shifted_cholqr3(A):
    """Thin QR of a full-column-rank A (m >= n) by shifted CholeskyQR3.

    Fukaya, Kannan, Nakatsukasa, Yamamoto and Yanagisawa, SIAM J. Sci. Comput.
    42 (2020) A477. Plain CholeskyQR forms A^T A, so its Cholesky breaks down
    once kappa(A) passes ~1e8. The first pass here factors A^T A + s I instead,
    which always succeeds but leaves Q1 with kappa(Q1) ~ sqrt(s)/||A||_2 kappa(A).
    Two plain passes (CholeskyQR2) then restore orthogonality to working
    precision, and R = R3 R2 R1 -- PROVIDED kappa(Q1) <~ 1e8.

    THE LIMIT DEPENDS ON THE SIZE, because the shift does: s = 11 (mn + n(n+1))
    u ||A||^2. Measured on Case 2's projected PDE matrix at rho_f = 2:

        (16,10,16)  6358 x 2025   kappa 2.7e10   kappa(Q1) 3.0e7   ok, ||Q'Q - I|| 1e-15
        (18,12,18)  9386 x 3179   kappa 2.4e12   kappa(Q1) 4.3e9   breaks in pass 2

    so the usable range is kappa(A) <~ 1e11 here, not the ~1e15 quoted for
    small matrices. A breakdown gives NaN, never a wrong answer. ||A||_F^2
    stands in for ||A||_2^2 (exact would need an SVD); it overestimates by
    ~30-40x here, which costs a factor ~6 in kappa(Q1) and does not change
    where it breaks.
    """
    m, n = A.shape
    eps = jnp.finfo(A.dtype).eps
    s = 11.0 * (m * n + n * (n + 1)) * eps * jnp.sum(A * A)
    R1 = jnp.linalg.cholesky(A.T @ A + s * jnp.eye(n, dtype=A.dtype)).T
    Q = solve_triangular(R1, A.T, trans="T", lower=False).T
    Q, R2 = _cholqr(Q)
    Q, R3 = _cholqr(Q)
    return Q, R3 @ R2 @ R1


def _reduced_lstsq(A, b, method: str, rcond):
    """argmin ||A x - b||, returning (x, singular values of A).

    The singular values are for the condition number only. For "qr" and
    "cholqr3" they come from the small n x n factor R, which has the same
    singular values as A; callers that drop them let jit drop that SVD too.
    That SVD is pinned to gesvd like ``svd``: JAX's GPU default (Jacobi) took
    244 ms on Case 2's 405 x 405 R, against 24 ms, and 16 ms for the whole
    CholeskyQR3 it was reporting on.
    """
    if method == "svd":
        x, _, _, sv = jnp.linalg.lstsq(A, b, rcond=rcond)
        return x, sv
    if method == "qr":
        Q, R = jnp.linalg.qr(A)
    elif method == "cholqr3":
        Q, R = shifted_cholqr3(A)
    else:
        raise ValueError(f"unknown ls_method {method!r}; choose from {LS_METHODS}")
    x = solve_triangular(R, Q.T @ b, lower=False)
    return x, lax_linalg.svd(R, full_matrices=False, compute_uv=False,
                             algorithm=lax_linalg.SvdAlgorithm.QR)


@jax.jit
def constraint_rank(H_c, rcond):
    """rank(H_c) under the same relative threshold as ``_project_and_solve``."""
    S = lax_linalg.svd(H_c, full_matrices=False, compute_uv=False,
                       algorithm=lax_linalg.SvdAlgorithm.QR)
    return jnp.sum(S > rcond * S[0])


def _null_space(H_c, K_c, rcond, ns_method: str = "svd", null_dim: int | None = None):
    """(beta_star, W, rank, cond_constr) of the boundary rows H_c beta = K_c.

    beta_star is the minimum-norm least-squares solution and W an orthonormal
    basis of null(H_c). Both come from the SAME factorisation with the SAME
    rank cut, so W is exactly orthogonal to the row space of H_c and a
    correction W eta cannot contaminate the constraints.

      "svd"   H_c = U S V^T, rank = #{S > rcond S_0}, W = trailing columns of V.
              With null_dim None, W is the whole of V with its first `rank`
              columns zeroed, so the shapes stay static under jit.
      "qrcp"  H_c^T[:, p] = Q R with column pivoting, so |R_jj| is
              non-increasing and rank = #{|R_jj| > rcond |R_00|}. The first r
              columns Q_1 of Q span the row space of H_c and the rest, W, its
              null space. beta_star = Q_1 y with y = argmin ||R_1^T y - K_c[p]||,
              R_1 = R[:r], since H_c[p] = R_1^T Q_1^T: the row space holds the
              minimum-norm solution. r = N* - null_dim is fixed before tracing.
              cond_constr is the estimate |R_00| / |R_(r-1)(r-1)|.
    """
    n_c, n_star = H_c.shape
    if ns_method == "qrcp":
        if null_dim is None:
            raise ValueError('ns_method="qrcp" needs null_dim = N* - rank(H_c)')
        r = n_star - null_dim
        Q, R, p = lax_linalg.qr(H_c.T, pivoting=True, full_matrices=True)
        d = jnp.abs(jnp.diagonal(R))
        rank = jnp.sum(d > rcond * d[0])
        Qy, Ry = jnp.linalg.qr(R[:r].T)
        beta_star = Q[:, :r] @ solve_triangular(Ry, Qy.T @ K_c[p], lower=False)
        return beta_star, Q[:, r:], rank, d[0] / d[r - 1]
    if ns_method != "svd":
        raise ValueError(f"unknown ns_method {ns_method!r}; choose from {NS_METHODS}")

    # Full V is needed for the null space; full U never is. When n_c >= N* the
    # thin SVD already returns a square V, and skips an n_c x n_c U that for
    # Case 2's boundary counts runs to hundreds of MB.
    U, S, Vt = svd(H_c, full_matrices=n_c < n_star)

    # Relative rank threshold -- see config.RCOND.
    keep = S > rcond * S[0]
    rank = jnp.sum(keep)

    # Pseudo-inverse with the discarded directions zeroed rather than sliced,
    # so the shapes stay static and the whole function remains jittable.
    S_inv = jnp.where(keep, 1.0 / jnp.where(keep, S, 1.0), 0.0)
    k = min(n_c, n_star)
    beta_star = Vt[:k].T @ (S_inv * (U[:, :k].T @ K_c))

    if null_dim is None:
        # Zero columns contribute nothing to H_f @ W and are annihilated by
        # the SVD least squares, so carrying them is harmless.
        W = jnp.where((jnp.arange(n_star) >= rank)[None, :], Vt.T, 0.0)  # (N*, N*)
    else:
        W = Vt[n_star - null_dim:].T  # (N*, null_dim)
    cond_constr = S[0] / jnp.min(jnp.where(keep, S, jnp.inf))
    return beta_star, W, rank, cond_constr


@partial(jax.jit, static_argnames=("ls_method", "null_dim", "ns_method"))
def _project_and_solve(H_f, K_f, H_c, K_c, rcond, ls_method: str = "svd",
                       null_dim: int | None = None, ns_method: str = "svd"):
    """The pure-JAX core: null-space projection then a projected least squares.

    Split out and jitted as one unit so the whole solve compiles once and the
    timing below measures compute rather than dispatch.

    Steps:
        beta_star, W = ``_null_space(H_c, K_c)``   by ``ns_method``
        eta          = argmin ||H_f W eta - (K_f - H_f beta_star)||   by ``ls_method``
        beta         = beta_star + W eta

    "qr" and "cholqr3" need a full-column-rank H_f W, and "qrcp" a static
    rank; both need ``null_dim`` = N* - rank(H_c), fixed before tracing, and W
    is then sliced to its null_dim genuine columns. If the rank found here
    differs from N* - null_dim, that slice is wrong, and beta is returned as
    NaN rather than as a silently wrong solution.
    """
    if (ls_method != "svd" or ns_method != "svd") and null_dim is None:
        raise ValueError(f"ls_method={ls_method!r}, ns_method={ns_method!r} need "
                         "null_dim = N* - rank(H_c)")
    n_star = H_c.shape[1]
    beta_star, W, rank, cond_constr = _null_space(H_c, K_c, rcond, ns_method, null_dim)

    H_proj = H_f @ W  # (n_f, N*) or (n_f, null_dim)
    K_proj = K_f - H_f @ beta_star  # (n_f,)

    eta, sv_proj = _reduced_lstsq(H_proj, K_proj, ls_method, rcond)
    beta = beta_star + W @ eta
    if null_dim is not None:
        beta = jnp.where(rank == n_star - null_dim, beta, jnp.nan)

    # Condition number, reported rather than acted on. sv_proj carries the
    # zero columns of the masked W, so the ratio is taken over the retained
    # part only.
    sv_nz = jnp.where(sv_proj > rcond * sv_proj[0], sv_proj, jnp.inf)
    cond_pde = sv_proj[0] / jnp.min(sv_nz)

    return beta, rank, cond_pde, cond_constr


def solve_forward(
    basis,
    x_f,
    rhs_f,
    constraints: Constraints,
    k,
    *,
    rcond: float = config.RCOND,
    time_it: bool = True,
    normalise_constraints: bool = True,
    ls_method: str = "svd",
    ns_method: str = "svd",
) -> Solution:
    """One forward solve at fixed conductivity ``k``.

    Args:
        basis:       a ``TensorBasis``.
        x_f:         (n_f, d) interior collocation points.
        rhs_f:       (n_f,) the source term q''' at those points.
        constraints: assembled boundary rows (see ``build_constraints``).
        k:           scalar (isotropic) or length-d sequence (orthotropic) of
                     conductivities. The PDE rows are sum_i k_i Phi_ii.
        rcond:       relative rank threshold for both SVD truncations.
        time_it:     if True, run once to force compilation, then time a second
                     device-synchronised call. If False, solve once and report
                     the elapsed time as the compile time.
        normalise_constraints: scale each boundary row to unit norm (the
                     default). Pass False when the caller has already weighted
                     the rows -- e.g. Case 3's flux-unit rows times sqrt(area),
                     whose weights decide how an unsatisfiable boundary
                     condition (the heater edge) shares out its residual, and
                     which per-row normalisation would overwrite.
        ls_method:   solver for the reduced least squares, one of LS_METHODS
                     (module docstring). "qr" and "cholqr3" first find
                     rank(H_c) in a separate, untimed SVD, to fix null_dim.
        ns_method:   factorisation for the null space of H_c, one of
                     NS_METHODS; "qrcp" fixes null_dim the same way.

    Returns:
        A ``Solution``.

    Raises:
        OverConstrainedError: if rank(H_c) >= N*, leaving no null space. The
            solve would then return the constraint-only solution beta_star and
            ignore the PDE entirely -- a silent wrong answer, so it is an error.
    """
    x_f = jnp.asarray(x_f)
    rhs_f = jnp.asarray(rhs_f).reshape(-1)

    k_vec = jnp.broadcast_to(jnp.asarray(k, dtype=x_f.dtype), (basis.dim,))

    # PDE rows: sum_i k_i Phi_ii. The blocks are k-independent, which is what
    # lets an inference loop hoist them out and only re-weight per trial k.
    blocks = basis.laplacian_blocks(x_f)
    H_f = sum(k_vec[i] * blocks[i] for i in range(basis.dim))

    H_f, K_f = _row_normalise(H_f, rhs_f)
    if normalise_constraints:
        H_c, K_c = _row_normalise(constraints.H, constraints.K)
    else:
        H_c, K_c = constraints.H, constraints.K

    n_star = basis.n_star

    if ls_method not in LS_METHODS:
        raise ValueError(f"unknown ls_method {ls_method!r}; choose from {LS_METHODS}")
    if ns_method not in NS_METHODS:
        raise ValueError(f"unknown ns_method {ns_method!r}; choose from {NS_METHODS}")
    null_dim = None
    if ls_method != "svd" or ns_method != "svd":
        null_dim = n_star - int(constraint_rank(H_c, rcond))
        if null_dim <= 0:
            raise OverConstrainedError(
                f"constraints exhaust the basis: rank(H_c) = {n_star - null_dim} >= "
                f"N* = {n_star}. Raise the basis degree or reduce the boundary points."
            )

    def _call():
        out = _project_and_solve(H_f, K_f, H_c, K_c, rcond, ls_method, null_dim, ns_method)
        return jax.block_until_ready(out)

    t0 = time.perf_counter()
    beta, rank, cond_pde, cond_constr = _call()
    compile_time = time.perf_counter() - t0

    if time_it:
        t0 = time.perf_counter()
        beta, rank, cond_pde, cond_constr = _call()
        solve_time = time.perf_counter() - t0
    else:
        solve_time = compile_time

    rank = int(rank)
    if rank >= n_star:
        raise OverConstrainedError(
            f"constraints exhaust the basis: rank(H_c) = {rank} >= N* = {n_star}. "
            f"The null space is empty, so the PDE cannot be fitted at all. "
            f"Raise the basis degree or reduce the number of boundary points."
        )

    return Solution(
        beta=beta,
        rank=rank,
        n_star=n_star,
        cond_pde=float(cond_pde),
        cond_constr=float(cond_constr),
        pde_residual=H_f @ beta - K_f,
        constr_residual=constraints.H @ beta - constraints.K,
        compile_time=compile_time,
        solve_time=solve_time,
    )
