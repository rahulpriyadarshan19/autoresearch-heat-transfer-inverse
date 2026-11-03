"""Frozen Case 2 design problem: constants, geometry, prior banks, D-criterion and spacing penalty."""

from __future__ import annotations

from heat_inverse import config  # noqa: F401  (enables x64 first)

import hashlib
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from heat_inverse.basis import TensorBasis
from heat_inverse.cases import case2

DEGREES_IN = (10, 6)
DEGREES = (*DEGREES_IN, DEGREES_IN[0])
RHO_F, RHO_B = 3.0, 2.0
SEED_F, SEED_B = 0, 1
LS_METHOD, NS_METHOD = "cholqr3", "qrcp"
LOG_PRIOR_MU = np.log(np.array([44.5, 11.0]))
PRIOR_SD = np.array([0.7, 0.7])
SIGMA = 1.0  # K
STEP = 1e-4  # central-difference step in log k

S_TRAIN, S_VAL = 512, 128
SEED_TRAIN, SEED_VAL = 10, 11
CHUNK = 128  # solves per GPU batch
M_LO = np.full(3, 1e-3)  # m, margin from the low edge along x, y, z
M_HI = np.full(3, 1e-3)  # m, margin from the high edge along x, y, z
D_MIN = 1e-2  # m
MU = 10.0  # spacing-penalty weight

FACES = ("y0", "x0", "x1", "z0", "z1")  # allowed faces, in config-string order; y1 carries the heater
FACE = {f.name: f for f in case2.FACES}
IN_PLANE = np.array([[1, 2], [0, 2], [0, 1]])  # in-plane axes of a face with normal axis 0, 1, 2
L3 = np.array([case2.LX, case2.LY, case2.LZ])
A0 = np.diag(1.0 / PRIOR_SD**2)  # NumPy: a jnp array here would start JAX on the GPU at import
BASIS = TensorBasis(degrees=DEGREES, bounds=case2.BOUNDS)

BANK_FILE = config.OUTPUTS / "case2_ar_cache" / "prior_banks.npz"
# byte-identical to the key of heat_transfer_inverse/scripts/run_dopt_case2.py, so its cached banks load here;
# a change in any of these settings invalidates the banks
BANK_KEY = repr((S_TRAIN, S_VAL, SEED_TRAIN, SEED_VAL, STEP, DEGREES, RHO_F, RHO_B, SEED_F, SEED_B, LS_METHOD,
                 NS_METHOD, LOG_PRIOR_MU.tolist(), PRIOR_SD.tolist()))


def make_problem():
    """case2.make_inverse_problem with the forward model of the cached banks."""
    prob = case2.make_inverse_problem(degrees=DEGREES_IN, rho_f=RHO_F, rho_b=RHO_B, sigma_abs=SIGMA,
                                      seed_train=SEED_F, seed_bc=SEED_B, ls_method=LS_METHOD,
                                      ns_method=NS_METHOD)
    assert prob.basis == BASIS
    return prob


def prior_bank(S, seed):
    """(S, 2) samples of theta from the prior."""
    return LOG_PRIOR_MU + PRIOR_SD * np.random.default_rng(seed).standard_normal((S, 2))


def B_of(beta_batch, log_k):
    """(S, 2) theta -> (S, N*, 2) dbeta/dtheta by central differences, 4 solves per sample."""
    E = STEP * np.eye(2)
    k = np.exp(np.stack([log_k + E[0], log_k - E[0], log_k + E[1], log_k - E[1]], axis=1)).reshape(-1, 2)
    beta = np.concatenate([beta_batch(jnp.asarray(k[i:i + CHUNK])) for i in range(0, len(k), CHUNK)])
    beta = beta.reshape(len(log_k), 2, 2, -1)  # sample, parameter, +/-, N*
    return np.moveaxis((beta[:, :, 0] - beta[:, :, 1]) / (2 * STEP), 1, 2)


def beta_batch_fn(prob=None):
    """Jitted, vmapped (n, 2) k -> (n, N*) beta."""
    return jax.jit(jax.vmap((prob or make_problem()).beta_of_k))


def compute_banks(path=BANK_FILE):
    """Compute both banks, save them with BANK_KEY to path -> (B_train, B_val, seconds)."""
    beta_batch = beta_batch_fn()
    t0 = time.perf_counter()
    B_train = B_of(beta_batch, prior_bank(S_TRAIN, SEED_TRAIN))
    B_val = B_of(beta_batch, prior_bank(S_VAL, SEED_VAL))
    seconds = time.perf_counter() - t0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, key=BANK_KEY, B_train=B_train, B_val=B_val)
    return B_train, B_val, seconds


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_banks(path=BANK_FILE):
    """(B_train, B_val) as device arrays; refuses a file whose key is not BANK_KEY."""
    z = np.load(path)
    if str(z["key"]) != BANK_KEY:
        raise ValueError(f"bank key of {path} is not BANK_KEY")
    return jnp.asarray(z["B_train"]), jnp.asarray(z["B_val"])


def g_bank(X, B):
    """(N, 3) points, (S, N*, 2) bank -> (S, N, 2) sensitivities in K."""
    return case2.DT * jnp.einsum("nk,skj->snj", BASIS.features(X), B)


def information(X, B):
    """(S, 2, 2) posterior precision A_s = G_s^T G_s / sigma^2 + Sigma_0^-1."""
    G = g_bank(X, B)
    return jnp.einsum("sni,snj->sij", G, G) / SIGMA**2 + A0


def phi(X, B):
    """-(1/S) sum_s log det(F_s + Sigma_0^-1), minimised."""
    A = information(X, B)
    return -jnp.mean(jnp.log(A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] ** 2))


def spacing(X):
    """sum_{i<j} max(0, 1 - d_ij^2 / D_MIN^2)^2."""
    # squared distances: the gradient of sqrt is NaN at d = 0
    d2 = jnp.sum((X[:, None] - X[None]) ** 2, axis=-1)
    i, j = jnp.triu_indices(len(X), 1)
    return jnp.sum(jnp.maximum(0.0, 1.0 - d2[i, j] / D_MIN**2) ** 2)
