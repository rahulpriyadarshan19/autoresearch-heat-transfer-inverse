"""Tensor-product Bernstein basis in JAX, with first and second derivatives.

The 1D kernels are lifted from
``~/Documents/ces_pielm/src/ces_pielm/solvers/bernstein.py`` (itself a port of
the PyTorch ``BernsteinELM`` in ``~/Documents/pde_dl``). The N-D tensor product
on top of them is new, and is what this project needs: Case 1.1 is 2D and
Case 2 is 3D, whereas both predecessors are 1D-only (the PyTorch class raises
outright on ``input_size != 1``).

Degree-n basis on [0, 1]:

    B_{n,nu}(s) = C(n,nu) s^nu (1-s)^(n-nu),   nu = 0..n

EVALUATION: split ratio recurrence, O(n) per point for all n+1 values
--------------------------------------------------------------------
    forward  (s <= 0.5):  B_{n,0} = (1-s)^n
                          B_{n,nu+1} = ((n-nu)/(nu+1)) * (s/(1-s)) * B_{n,nu}
    backward (s >  0.5):  B_{n,n} = s^n
                          B_{n,nu-1} = (nu/(n-nu+1)) * ((1-s)/s) * B_{n,nu}

Splitting at s = 0.5 keeps the ratio multiplier in [0, 1], avoiding the overflow
a naive C(n,nu) s^nu (1-s)^(n-nu) hits at high degree -- the binomial
coefficient is never formed. Endpoints are handled exactly, sidestepping 0/0 in
the ratio.

A NOTE ON CONDITIONING, because it shapes how this basis should be used.
Bernstein's celebrated stability is a property of EVALUATING a polynomial from
known coefficients (partition of unity, non-negativity, de Casteljau). It does
NOT transfer to SOLVING for those coefficients, which is what a PIELM
collocation solve does. Measurements recorded in the permafrost project for the
1D fitting problem: kappa ~ 4.6e8 at degree 30 and ~4.5e14 at degree 50, with
0.968 correlation between adjacent columns. Adjacent Bernstein functions overlap
heavily, so the design matrix is near-collinear -- a defect column equilibration
cannot repair. A tensor product multiplies the column count per axis and
compounds this. Hence: keep per-axis degrees modest, and read the condition
number the diagnostics report prints.

INDEX CONVENTION, used consistently here and in every consumer:
column j of a feature matrix corresponds to the multi-index (nu_0, ..., nu_{d-1})
in C order (last axis varies fastest), i.e. the flattening produced by
``jnp.einsum`` followed by ``reshape(N, -1)``. So for degrees (n_x, n_y),
column j holds nu_x = j // (n_y + 1), nu_y = j % (n_y + 1).
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

from . import config  # noqa: F401  (enables x64 before any array is made)

import jax.numpy as jnp


# --------------------------------------------------------------------------
# 1D kernels
# --------------------------------------------------------------------------


def bernstein_values(s, n: int):
    """All n+1 Bernstein basis values at each point of ``s``.

    Args:
        s: (N,) points in [0, 1].
        n: polynomial degree (static).

    Returns:
        (N, n+1) array of B_{n,nu}(s).
    """
    s = jnp.asarray(s)
    if n < 0:
        return jnp.zeros((s.shape[0], 0), dtype=s.dtype)
    if n == 0:
        return jnp.ones((s.shape[0], 1), dtype=s.dtype)

    one_m_s = 1.0 - s

    # Guard the ratios before use. Each branch is only *selected* where its
    # denominator is safe, but JAX evaluates both, so unguarded divisions would
    # still produce inf/nan and poison the gradients through jnp.where.
    safe_s = jnp.where(s > 0.0, s, 1.0)
    safe_oms = jnp.where(one_m_s > 0.0, one_m_s, 1.0)

    # --- forward recurrence, accurate for s <= 0.5 ---------------------------
    ratio_f = s / safe_oms
    cols_f = [one_m_s**n]
    for nu in range(n):
        cols_f.append(((n - nu) / (nu + 1.0)) * ratio_f * cols_f[nu])
    B_forward = jnp.stack(cols_f, axis=1)

    # --- backward recurrence, accurate for s > 0.5 ---------------------------
    ratio_b = one_m_s / safe_s
    cols_b = [None] * (n + 1)
    cols_b[n] = s**n
    for nu in range(n, 0, -1):
        cols_b[nu - 1] = (nu / (n - nu + 1.0)) * ratio_b * cols_b[nu]
    B_backward = jnp.stack(cols_b, axis=1)

    B = jnp.where((s <= 0.5)[:, None], B_forward, B_backward)

    # Exact endpoints: at s=0 only B_{n,0}=1; at s=1 only B_{n,n}=1.
    nu_idx = jnp.arange(n + 1)
    basis_zero = (nu_idx == 0).astype(B.dtype)[None, :]
    basis_one = (nu_idx == n).astype(B.dtype)[None, :]
    B = jnp.where((s == 0.0)[:, None], basis_zero, B)
    B = jnp.where((s == 1.0)[:, None], basis_one, B)
    return B


def bernstein_d1(s, n: int, length: float = 1.0):
    """First derivative d/dx of the degree-n basis, where x = x_lo + length*s.

        dB_{n,nu}/ds = n (B_{n-1,nu-1} - B_{n-1,nu})

    Chain rule contributes 1/length.
    """
    s = jnp.asarray(s)
    if n == 0:
        return jnp.zeros((s.shape[0], 1), dtype=s.dtype)
    Bm1 = bernstein_values(s, n - 1)  # (N, n)
    zc = jnp.zeros((s.shape[0], 1), dtype=s.dtype)
    B_nu = jnp.concatenate([Bm1, zc], axis=1)  # B_{n-1,nu}
    B_nu_m1 = jnp.concatenate([zc, Bm1], axis=1)  # B_{n-1,nu-1}
    return (n / length) * (B_nu_m1 - B_nu)


def bernstein_d2(s, n: int, length: float = 1.0):
    """Second derivative d2/dx2 of the degree-n basis.

        d2B_{n,nu}/ds2 = n(n-1) (B_{n-2,nu-2} - 2 B_{n-2,nu-1} + B_{n-2,nu})
    """
    s = jnp.asarray(s)
    N = s.shape[0]
    if n < 2:
        return jnp.zeros((N, n + 1), dtype=s.dtype)
    Bm2 = bernstein_values(s, n - 2)  # (N, n-1)
    zc = jnp.zeros((N, 1), dtype=s.dtype)
    zc2 = jnp.zeros((N, 2), dtype=s.dtype)
    B_nu = jnp.concatenate([Bm2, zc2], axis=1)
    B_nu_m1 = jnp.concatenate([zc, Bm2, zc], axis=1)
    B_nu_m2 = jnp.concatenate([zc2, Bm2], axis=1)
    return (n * (n - 1) / (length**2)) * (B_nu_m2 - 2.0 * B_nu_m1 + B_nu)


# Dispatch table: derivative order -> 1D kernel. Keyed this way so the tensor
# product below needs one code path rather than one per derivative order, and
# so a second basis family (Chebyshev) can be added as another table without
# touching the tensor-product logic or the solver.
_KERNELS_BERNSTEIN = {
    0: lambda s, n, L: bernstein_values(s, n),
    1: bernstein_d1,
    2: bernstein_d2,
}

_FAMILIES = {"bernstein": _KERNELS_BERNSTEIN}


# --------------------------------------------------------------------------
# N-D tensor product
# --------------------------------------------------------------------------


class TensorBasis(NamedTuple):
    """Tensor-product polynomial basis over an axis-aligned box in d dimensions.

    The basis functions are products of 1D functions, one per axis:

        Phi_{nu_0..nu_{d-1}}(x) = prod_i B_{n_i, nu_i}(s_i),   s_i = (x_i - lo_i)/L_i

    giving N* = prod_i (n_i + 1) columns. Degrees are per-axis by design: Case 2
    is a 250 x 200 x 50 mm slab whose through-plane direction warrants a
    different resolution from the in-plane ones, and making this a tuple now
    costs nothing while retrofitting it later would touch every consumer.

    Attributes:
        degrees: (n_0, ..., n_{d-1}) per-axis polynomial degree.
        bounds:  ((lo_0, hi_0), ...) axis-aligned domain box.
        family:  basis family name; only "bernstein" is wired up today.
    """

    degrees: tuple[int, ...]
    bounds: tuple[tuple[float, float], ...]
    family: str = "bernstein"

    @property
    def dim(self) -> int:
        return len(self.degrees)

    @property
    def n_star(self) -> int:
        """Number of basis functions (columns of every feature matrix)."""
        out = 1
        for n in self.degrees:
            out *= n + 1
        return out

    @property
    def lengths(self) -> tuple[float, ...]:
        return tuple(float(hi - lo) for lo, hi in self.bounds)

    def _to_unit(self, X):
        """Affine map of each axis onto [0, 1]. X is (N, d) -> (N, d)."""
        lo = jnp.asarray([b[0] for b in self.bounds])
        L = jnp.asarray(self.lengths)
        return (jnp.asarray(X) - lo) / L

    def _tensor(self, X, orders: Sequence[int]):
        """Tensor-product feature matrix with a derivative order per axis.

        ``orders[i]`` is the number of derivatives taken along axis i, so
        orders = (0, 0) gives Phi, (2, 0) gives Phi_xx, and (1, 0) gives Phi_x.
        Mixed derivatives fall out of the same path, though the second-order
        heat equation here never needs them.

        The product is formed by successive outer products and flattened in C
        order (last axis fastest), which is the index convention documented in
        the module docstring.
        """
        kernels = _FAMILIES[self.family]
        S = self._to_unit(X)
        N = S.shape[0]

        out = jnp.ones((N, 1), dtype=S.dtype)
        for i, (n_i, order_i, L_i) in enumerate(
            zip(self.degrees, orders, self.lengths)
        ):
            if order_i not in kernels:
                raise ValueError(
                    f"derivative order {order_i} not supported "
                    f"(have {sorted(kernels)})"
                )
            B_i = kernels[order_i](S[:, i], n_i, L_i)  # (N, n_i+1)
            # (N, m) x (N, n_i+1) -> (N, m*(n_i+1)), last axis varying fastest.
            out = (out[:, :, None] * B_i[:, None, :]).reshape(N, -1)
        return out

    # -- public API ---------------------------------------------------------

    def features(self, X):
        """Phi(X), shape (N, N*)."""
        return self._tensor(X, (0,) * self.dim)

    def features_d1(self, X, axis: int):
        """d Phi / d x_axis, shape (N, N*)."""
        orders = [0] * self.dim
        orders[axis] = 1
        return self._tensor(X, orders)

    def features_d2(self, X, axis: int):
        """d2 Phi / d x_axis2, shape (N, N*)."""
        orders = [0] * self.dim
        orders[axis] = 2
        return self._tensor(X, orders)

    def laplacian_blocks(self, X):
        """[Phi_xx, Phi_yy, ...] -- one second-derivative block per axis.

        Returned separately rather than summed because the inverse problem
        weights them by different unknowns: the orthotropic PDE is
        k_xx Phi_xx + k_yy Phi_yy + k_zz Phi_zz, so the blocks are the
        k-independent part that can be computed once and reused across every
        trial k in an inference loop.
        """
        return [self.features_d2(X, i) for i in range(self.dim)]

    def evaluate(self, beta, X):
        """Evaluate the expansion Phi(X) @ beta at points X. Returns (N,)."""
        return self.features(X) @ jnp.asarray(beta).reshape(-1)
