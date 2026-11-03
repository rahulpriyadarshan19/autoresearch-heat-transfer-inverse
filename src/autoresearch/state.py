"""State: the part of the log that the harness sends to the planner in each call."""

from __future__ import annotations

from . import layout
from . import problem as P


def _batch(row_id: str) -> str:
    return row_id.split("-")[0]


def _line(r: dict, notes: dict) -> str:
    note = f" [{notes[r['id']]}]" if r["id"] in notes else ""
    if r["score"] is None:
        metrics = "no result"
    else:
        metrics = f"score {r['score']:.5f}  s1 {r['s1']:.4g}  s2 {r['s2']:.4g}  rho {r['rho']:+.3f}"
    return f"{r['id']}  {r['config']}  {metrics}  {r['status']}{note}  | {r['description']}"


def scored(rows, notes):
    """Rows with a valid score: not crashed and not infeasible."""
    return [r for r in rows if r["score"] is not None and notes.get(r["id"]) != "infeasible"]


def face_statistics(rows, notes) -> list[str]:
    """For each face: the best score among the tried sensor_layouts with each value of n_f."""
    out = []
    for k, name in enumerate(P.FACES):
        best = {}
        for r in scored(rows, notes):
            n_f = layout.parse(r["config"])[k]
            best[n_f] = min(best.get(n_f, r["score"]), r["score"])
        out.append(f"{name}: " + "  ".join(f"{n}:{best[n]:.4f}" for n in sorted(best)) if best else f"{name}: -")
    return out


def build(rows, notes, *, n, b, m, remaining_s, untried, marks=()) -> str:
    """State text: facts, best row, best M rows, last two batches, face statistics, marks of the last reply."""
    good = sorted(scored(rows, notes), key=lambda r: r["score"])
    batches = sorted({_batch(r["id"]) for r in rows})[-2:]
    best_ids = {r["id"] for r in good[:m]}
    recent = [r for r in rows if _batch(r["id"]) in batches and r["id"] not in best_ids]
    lines = [
        f"N = {n} sensors. Propose B = {b} new sensor_layouts.",
        f"Trials so far: {len(rows)}. Untried sensor_layouts: {untried}. Remaining time: {remaining_s / 3600:.2f} h.",
        "",
        "Best so far: " + (_line(good[0], notes) if good else "none"),
        "",
        f"Best {min(m, len(good))} trials (id, config y0-x0-x1-z0-z1, score, s1, s2, rho, status | description):",
        *(_line(r, notes) for r in good[:m]),
        "",
        f"Other trials of the last {len(batches)} batches:",
        *(_line(r, notes) for r in recent),
        "",
        "Best score for each n_f on each face (n_f:score):",
        *face_statistics(rows, notes),
    ]
    if marks:
        lines += ["", "Your previous reply:", *marks]
    return "\n".join(lines)
