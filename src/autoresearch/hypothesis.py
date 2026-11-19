"""Hypothesis of the learnings run: region and context (linear constraints on the face counts), its set S(h), the tree
of hypotheses, τ_h and the verdict."""

from __future__ import annotations

import functools
import re
from typing import NamedTuple

import numpy as np

from . import layout, noise
from . import problem as P

OPS = ("<=", ">=", "=")  # "<=" and ">=" first: "=" is a substring of both
STEPS = ("root", "child", "sibling", "back")
PREDICTIONS = ("lower", "higher")
VERDICTS = ("supported", "refuted", "inconclusive")
# column orders of the 4 mirror images: identity, x0 <-> x1, z0 <-> z1, both
MIRROR_PERMS = ((0, 1, 2, 3, 4), (0, 2, 1, 3, 4), (0, 1, 2, 4, 3), (0, 2, 1, 4, 3))
_TERM = re.compile(rf"([+-]?)(?:(\d+)\*?)?({'|'.join(P.FACES)})")
_CONST = re.compile(r"([+-]?)(\d+)")


class Condition(NamedTuple):
    coef: tuple[int, ...]  # one integer per face, P.FACES order
    op: str
    rhs: int


def _linear(expr: str):
    """'2*y0 - x1 + 3' -> (integer coefficient per face, integer constant)."""
    s = re.sub(r"\s+", "", expr)
    if not s:
        raise ValueError("one side of a condition is empty")
    terms = re.findall(r"[+-]?[^+-]+", s)
    if "".join(terms) != s:
        raise ValueError(f"cannot read {expr.strip()!r}")
    coef, const = np.zeros(len(P.FACES), dtype=int), 0
    for t in terms:
        if m := _TERM.fullmatch(t):
            coef[P.FACES.index(m[3])] += (-1 if m[1] == "-" else 1) * int(m[2] or 1)
        elif m := _CONST.fullmatch(t):
            const += (-1 if m[1] == "-" else 1) * int(m[2])
        else:
            raise ValueError(f"cannot read the term {t!r}: use integers and the faces {', '.join(P.FACES)}")
    return coef, const


def parse_condition(text: str) -> Condition:
    """'x0+x1>=12' -> Condition; all terms moved to the left side, constants to the right side."""
    op = next((o for o in OPS if o in text), None)
    if op is None:
        raise ValueError(f"{text.strip()!r} has no '=', '<=' or '>='")
    lhs, rhs = text.split(op, 1)
    if any(ch in lhs + rhs for ch in "<>="):
        raise ValueError(f"{text.strip()!r} has more than one comparison")
    (cl, kl), (cr, kr) = _linear(lhs), _linear(rhs)
    coef = cl - cr
    if not coef.any():
        raise ValueError(f"{text.strip()!r} has no face count")
    return Condition(tuple(int(c) for c in coef), op, int(kr - kl))


def parse(text: str) -> tuple[Condition, ...]:
    """Conditions joined by ';' -> constraint; ValueError with the reason if one does not parse."""
    parts = [p for p in str(text).split(";") if p.strip()]
    if not parts:
        raise ValueError("no condition")
    return tuple(parse_condition(p) for p in parts)


def _fmt_condition(c: Condition) -> str:
    s = ""
    for k, f in zip(c.coef, P.FACES):
        if k:
            s += ("-" if k < 0 else "+" if s else "") + ("" if abs(k) == 1 else f"{abs(k)}*") + f
    return f"{s}{c.op}{c.rhs}"


def fmt(constraint) -> str:
    """Normal form of a constraint, for the logs."""
    return "; ".join(_fmt_condition(c) for c in constraint)


def holds(constraint, counts) -> np.ndarray:
    """(M,) mask: rows of counts (M, 5), or one sensor_layout, that satisfy every condition."""
    A = np.atleast_2d(np.asarray(counts, dtype=int))
    ok = np.ones(len(A), dtype=bool)
    for c in constraint:
        v = A @ np.asarray(c.coef)
        ok &= (v == c.rhs) if c.op == "=" else (v <= c.rhs) if c.op == "<=" else (v >= c.rhs)
    return ok


def holds_mirror(constraint, counts) -> np.ndarray:
    """(M,) mask: the sensor_layout or one of its mirror images satisfies every condition."""
    A = np.atleast_2d(np.asarray(counts, dtype=int))
    return np.logical_or.reduce([holds(constraint, A[:, list(p)]) for p in MIRROR_PERMS])


@functools.lru_cache(maxsize=4)
def _canonical_layouts(n: int) -> np.ndarray:
    return np.array(layout.enumerate_layouts(n, mirror_reduce=True), dtype=int)


def region_mask(constraint, n: int) -> np.ndarray:
    """Mask over the canonical sensor_layouts with Σ = n: the layout or a mirror image satisfies the constraint."""
    return holds_mirror(constraint, _canonical_layouts(n))


def members(constraint, n: int) -> list[tuple[int, ...]]:
    """S(h): canonical sensor_layouts with Σ = n of which the layout or a mirror image satisfies the constraint."""
    A = _canonical_layouts(n)
    return [tuple(int(v) for v in a) for a in A[region_mask(constraint, n)]]


def tau_h(tau: float, sigma: float, n_h: int, n_ref: int) -> float:
    """τ + max(0, c(n_h) − c(n_ref)) σ_R: the best of n_h scores with equal true Φ is low by c(n_h) σ_R on average,
    and the reference, a best of n_ref, by c(n_ref) σ_R."""
    return float(tau + max(0.0, noise.expected_max(n_h) - noise.expected_max(n_ref)) * sigma)


def verdict(delta, prediction: str, threshold: float) -> str:
    """supported: Δ has the predicted sign and |Δ| > threshold; refuted: opposite sign and |Δ| > threshold."""
    if delta is None or not np.isfinite(delta) or abs(delta) <= threshold:
        return "inconclusive"
    return "supported" if (delta < 0) == (prediction == "lower") else "refuted"


def path_to(hyps: dict, start: str, ancestor: str) -> list[str]:
    """Hypothesis ids from start up to, not including, ancestor along parent_id; empty if ancestor is not above
    start."""
    path, h = [], start
    while h and h in hyps and h != ancestor:
        path.append(h)
        h = hyps[h]["parent_id"]
    return path if h == ancestor else []


def selection_size(hyps: dict, trial_id: str) -> int:
    """n_ref: smallest compared set in which the trial was the best member; 1 if it was never the best member."""
    sizes = [h["n_cmp"] for h in hyps.values() if h["best_id"] == trial_id and h["n_cmp"]]
    return min(sizes) if sizes else 1
