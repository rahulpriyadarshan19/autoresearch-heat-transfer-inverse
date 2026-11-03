"""Project-wide configuration: precision and per-run output paths.

IMPORTANT: this module enables JAX float64 and must be imported *before* any JAX
array is created. Import it first in every module that touches JAX.

float64 is non-negotiable in this project. The PIELM collocation matrices built
from a tensor-product Bernstein basis are badly conditioned -- the 1D
measurements recorded in ``ces_pielm/src/ces_pielm/solvers/bernstein.py`` put
kappa at 4.6e8 by degree 30 and 4.5e14 by degree 50, and a tensor product
compounds that per axis. In float32 the null-space projection would still return
a plausible-looking coefficient vector, with no error raised and no diagnostic
out of range; the solution would simply be wrong. Enabling x64 is what makes the
residual diagnostics in ``diagnostics.py`` meaningful as a correctness signal.
"""

from __future__ import annotations

import jax

# Must run before ANY JAX array is created -- x64 is a global, import-time
# setting that is silently ignored once the first array exists.
jax.config.update("jax_enable_x64", True)

from pathlib import Path  # noqa: E402  (import after the x64 flag, deliberately)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = PROJECT_ROOT / "outputs"


def run_dir(prefix: str) -> Path:
    """Directory for one run's artefacts, created on demand.

    Every run (forward solve, degree sweep, and later EKS) writes its figures,
    arrays and logs into ``outputs/<prefix>/`` with that same prefix in each
    filename. The convention is borrowed from the permafrost project, where it
    was adopted after two runs differing only in a flag value produced figures
    whose names could not be told apart. Bundling artefacts per run makes a
    figure self-describing: the settings that produced it sit beside it.
    """
    d = OUTPUTS / prefix
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# Numerical defaults
# --------------------------------------------------------------------------
# Relative rank threshold for the truncated SVD in the null-space projection:
# a singular value counts as nonzero when sigma > RCOND * sigma_max.
#
# Relative, not absolute. The reference implementation
# (pde_dl/random_feature_functions/train.py) uses an absolute 1e-16, which is
# only meaningful once the problem happens to be scaled so that sigma_max ~ 1.
# With a tensor-product Bernstein basis the singular values of H_constr span
# many orders of magnitude and their absolute scale moves with the degree and
# the domain size, so an absolute cutoff either keeps numerical noise (inflating
# the rank, over-constraining the solve) or discards genuine constraint
# directions (silently dropping a boundary condition).
RCOND = 1e-13

# Condition numbers above this are flagged in the diagnostics report. Not an
# error: a well-converged solve at high degree can sit here legitimately. It
# marks the point where float64 starts losing digits the residuals cannot show.
COND_WARN = 1e12
