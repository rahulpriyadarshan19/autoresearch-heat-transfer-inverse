"""sensor_layout: integer sensor counts per allowed face, its config string, enumeration and start rule."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import qmc

from heat_inverse.cases import case2

from . import problem as P

MIRROR_PAIRS = ((1, 2), (3, 4))  # (x0, x1), (z0, z1) in P.FACES order


def parse(config: str, n: int | None = None) -> tuple[int, ...]:
    """'8-3-3-3-3' -> (8, 3, 3, 3, 3); ValueError with the reason if malformed or Σ != n."""
    parts = str(config).strip().split("-")
    if len(parts) != len(P.FACES) or not all(p.isdigit() for p in parts):
        raise ValueError(f"not {len(P.FACES)} non-negative integers joined by '-'")
    counts = tuple(int(p) for p in parts)
    if n is not None and sum(counts) != n:
        raise ValueError(f"sum is {sum(counts)}, not N = {n}")
    return counts


def fmt(counts) -> str:
    return "-".join(str(int(c)) for c in counts)


def mirror(counts) -> tuple[int, ...]:
    c = list(counts)
    for a, b in MIRROR_PAIRS:
        c[a], c[b] = c[b], c[a]
    return tuple(c)


def canonical(counts) -> tuple[int, ...]:
    """Mirror representative with n_x0 >= n_x1 and n_z0 >= n_z1."""
    c = list(counts)
    for a, b in MIRROR_PAIRS:
        if c[a] < c[b]:
            c[a], c[b] = c[b], c[a]
    return tuple(c)


def enumerate_layouts(n: int, mirror_reduce: bool = True) -> list[tuple[int, ...]]:
    """All sensor_layouts with Σ = n, in lexicographic order; canonical ones only if mirror_reduce."""
    out = []
    for cuts in itertools.combinations(range(n + len(P.FACES) - 1), len(P.FACES) - 1):
        bounds = (-1, *cuts, n + len(P.FACES) - 1)
        counts = tuple(bounds[i + 1] - bounds[i] - 1 for i in range(len(P.FACES)))
        if not mirror_reduce or counts == canonical(counts):
            out.append(counts)
    return sorted(out)


def halton_unit(n: int) -> np.ndarray:
    """First n points of the unscrambled 2-D Halton sequence (bases 2, 3) without the origin, (n, 2)."""
    if n == 0:
        return np.zeros((0, 2))
    return qmc.Halton(d=2, scramble=False).random(n + 1)[1:]


def start_design(counts) -> tuple[np.ndarray, np.ndarray]:
    """Halton start positions inside the margins -> (X (N, 3) in m, faces (N,))."""
    X, faces = [], []
    for name, n_f in zip(P.FACES, counts):
        if n_f == 0:
            continue
        f = P.FACE[name]
        ip = P.IN_PLANE[f.axis]
        Xf = np.zeros((n_f, 3))
        Xf[:, f.axis] = case2.BOUNDS[f.axis][f.side]
        lo, span = P.M_LO[ip], P.L3[ip] - P.M_LO[ip] - P.M_HI[ip]
        Xf[:, ip] = lo + halton_unit(n_f) * span
        X.append(Xf)
        faces += [name] * n_f
    return np.concatenate(X), np.array(faces)
