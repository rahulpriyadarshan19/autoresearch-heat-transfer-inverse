"""NP-PIELM inverse heat conduction.

Null-space-projected physics-informed extreme learning machines for the inverse
heat-conduction cases of the MIPINN paper (``project_docs/mipinn.pdf``).

Import ``heat_inverse.config`` before touching JAX anywhere: it enables float64,
which is an import-time global and cannot be set once an array exists.
"""

from . import config  # noqa: F401  (enables x64 before any array is made)

__all__ = ["config"]
